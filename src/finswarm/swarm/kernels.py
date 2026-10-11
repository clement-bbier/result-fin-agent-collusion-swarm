"""Swarm kernels: alias sampling, Poisson arrivals, lazy state refresh, logit routing, fills.

All kernels are pure functions of arrays. Randomness enters only through a *uniform tape*:
a float64 array of U[0, 1) draws taken from the ``swarm`` stream and consumed sequentially.
Because consumption is sequential and the tape is the concatenation of stream blocks, the
result of ``advance(a); advance(b)`` equals ``advance(a + b)`` bit for bit, whatever the
refill timing. Consumption per tick is fixed given the state:

* ``n_pieces = max(1, ceil(Lambda / POIS_PIECE))`` uniforms for the arrival count
  (Poisson by inversion, split into pieces to avoid ``exp(-Lambda)`` underflow);
* 2 uniforms for the global sentiment factor (Box-Muller, cosine branch);
* 4 uniforms per arrival: alias draw, routing draw, one Box-Muller pair giving the sentiment
  innovation ``z1`` and the private signal noise ``z2``.

Interventions (shocks, quote changes) therefore never desynchronise the swarm's random
numbers: paired branches share common random numbers by construction.

The public limit-order fringe (mechanism B, :func:`dealer_fringe_ticks`) consumes exactly the
same uniforms: an arriving ZI buyer whose routing uniform ``ur`` is below the fringe share
becomes a fringe seller and reuses ``ur / share`` for its cost draw; the other ZI buyers route
with ``(ur - share) / (1 - share)``. Turning the fringe on or off keeps the arrivals,
identities and signal noise of a paired branch identical.
"""

from __future__ import annotations

import math

import numpy as np

from finswarm.swarm._jit import jit

POIS_PIECE = 30.0
U_PER_ARRIVAL = 4
U_PER_TICK_EXTRA = 2
#: seller id of the anonymous public limit-order fringe (trades, flow rows)
FRINGE_ID = -2
#: slack when snapping a price onto the grid (in grid units)
GRID_EPS = 1e-9

# SwarmType values that buy from dealers in the one-sided dealer market (A).
_ZI_BUYER = 0
_INFORMED = 2
_NOISE = 3
_TREND = 4


# --------------------------------------------------------------------------- alias (Vose)
@jit
def build_alias(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Vose alias table for non-negative weights ``w``; returns ``(prob, alias, total)``."""
    n = w.shape[0]
    prob = np.zeros(n, dtype=np.float64)
    alias = np.empty(n, dtype=np.int64)
    for i in range(n):
        alias[i] = i
    total = 0.0
    for i in range(n):
        total += w[i]
    if n == 0 or total <= 0.0:
        return prob, alias, 0.0
    scaled = np.empty(n, dtype=np.float64)
    small = np.empty(n, dtype=np.int64)
    large = np.empty(n, dtype=np.int64)
    ns = 0
    nl = 0
    for i in range(n):
        scaled[i] = w[i] * n / total
        if scaled[i] < 1.0:
            small[ns] = i
            ns += 1
        else:
            large[nl] = i
            nl += 1
    while ns > 0 and nl > 0:
        ns -= 1
        s = small[ns]
        nl -= 1
        g = large[nl]
        prob[s] = scaled[s]
        alias[s] = g
        scaled[g] = (scaled[g] + scaled[s]) - 1.0
        if scaled[g] < 1.0:
            small[ns] = g
            ns += 1
        else:
            large[nl] = g
            nl += 1
    while nl > 0:
        nl -= 1
        prob[large[nl]] = 1.0
    while ns > 0:  # rounding leftovers (their scaled weight is ~1)
        ns -= 1
        i = small[ns]
        prob[i] = 1.0 if w[i] > 0.0 else 0.0
    return prob, alias, total


@jit
def alias_draw(prob: np.ndarray, alias: np.ndarray, u: float) -> int:
    """One alias draw from a single uniform (column from the integer part, coin from the rest)."""
    n = prob.shape[0]
    x = u * n
    i = int(x)
    if i >= n:
        i = n - 1
    if x - i < prob[i]:
        return i
    return alias[i]


@jit
def alias_sample_many(prob: np.ndarray, alias: np.ndarray, u: np.ndarray) -> np.ndarray:
    out = np.empty(u.shape[0], dtype=np.int64)
    for k in range(u.shape[0]):
        out[k] = alias_draw(prob, alias, u[k])
    return out


# ------------------------------------------------------------------------ scalar helpers
@jit
def poisson_inv(u: float, lam: float) -> int:
    """Poisson(lam) by inversion of one uniform; ``lam`` should stay <= ~30."""
    if lam <= 0.0:
        return 0
    p = math.exp(-lam)
    cdf = p
    k = 0
    kmax = int(10.0 * lam + 50.0)
    while u > cdf and k < kmax:
        k += 1
        p *= lam / k
        cdf += p
    return k


@jit
def n_poisson_pieces(lam_total: float) -> int:
    if lam_total <= POIS_PIECE:
        return 1
    return int(math.ceil(lam_total / POIS_PIECE))


@jit
def box_muller(u1: float, u2: float) -> tuple[float, float]:
    r = math.sqrt(-2.0 * math.log(1.0 - u1))
    th = 2.0 * math.pi * u2
    return r * math.cos(th), r * math.sin(th)


@jit
def ar1_gap_sd(rho: float, sigma: float, k: int) -> float:
    """Sd of the accumulated AR(1) innovation over ``k`` ticks."""
    if k <= 0:
        return 0.0
    if rho != 1.0:
        rk = rho**k
        return sigma * math.sqrt((1.0 - rk * rk) / (1.0 - rho * rho))
    return sigma * math.sqrt(float(k))


# ------------------------------------------------------------------- lazy / eager refresh
@jit
def refresh_agent(
    j: int,
    t: int,
    z: float,
    sentiment: np.ndarray,
    last_update: np.ndarray,
    belief_F: np.ndarray,
    shock_delay: np.ndarray,
    rho: float,
    sigma: float,
    F_now: float,
) -> None:
    """Bring agent ``j`` from ``last_update[j]`` to tick ``t`` in O(1).

    Sentiment deviation: ``x <- rho**k x + sd_k z`` (exact AR(1) transition over k ticks).
    Belief: once ``t >= shock_delay[j]`` the agent knows the current fundamental.
    """
    k = t - last_update[j]
    if k > 0:
        if rho != 1.0:
            rk = rho**k
            sd = sigma * math.sqrt((1.0 - rk * rk) / (1.0 - rho * rho))
        else:
            rk = 1.0
            sd = sigma * math.sqrt(float(k))
        sentiment[j] = rk * sentiment[j] + sd * z
        last_update[j] = t
    d = shock_delay[j]
    if d >= 0 and t >= d:
        belief_F[j] = F_now
        shock_delay[j] = -1


@jit
def refresh_lazy_many(
    ids: np.ndarray,
    t: int,
    z: np.ndarray,
    sentiment: np.ndarray,
    last_update: np.ndarray,
    belief_F: np.ndarray,
    shock_delay: np.ndarray,
    rho: float,
    sigma: float,
    F_now: float,
) -> None:
    for k in range(ids.shape[0]):
        refresh_agent(
            ids[k], t, z[k], sentiment, last_update, belief_F, shock_delay, rho, sigma, F_now
        )


@jit
def refresh_eager_step(
    t: int,
    z: np.ndarray,
    sentiment: np.ndarray,
    last_update: np.ndarray,
    belief_F: np.ndarray,
    shock_delay: np.ndarray,
    rho: float,
    sigma: float,
    F_now: float,
) -> None:
    """Reference: advance *every* agent by one tick to ``t`` (used to validate lazy refresh)."""
    for j in range(sentiment.shape[0]):
        sentiment[j] = rho * sentiment[j] + sigma * z[j]
        last_update[j] = t
        d = shock_delay[j]
        if d >= 0 and t >= d:
            belief_F[j] = F_now
            shock_delay[j] = -1


# ------------------------------------------------------------------------------- routing
@jit
def route_choice(
    v: float,
    ask_ticks: np.ndarray,
    ask_prices: np.ndarray,
    u0: float,
    tau: float,
    ur: float,
    util: np.ndarray,
    w: np.ndarray,
) -> int:
    """Logit routing: P(dealer i) ∝ exp((v - ask_i)/tau), outside ∝ exp(u0/tau).

    Returns the dealer index or -1 for the outside option. ``tau <= 0`` is exact price
    priority (ties, including with the outside option, split uniformly by ``ur``).
    """
    nd = ask_ticks.shape[0]
    best = u0
    for d in range(nd):
        if ask_ticks[d] >= 0:
            util[d] = v - ask_prices[d]
            if util[d] > best:
                best = util[d]
    if tau > 0.0:
        tot = math.exp((u0 - best) / tau)
        for d in range(nd):
            if ask_ticks[d] >= 0:
                w[d] = math.exp((util[d] - best) / tau)
            else:
                w[d] = 0.0
            tot += w[d]
        r = ur * tot
        acc = 0.0
        for d in range(nd):
            acc += w[d]
            if r < acc:
                return d
        return -1
    cnt = 1 if u0 >= best else 0
    for d in range(nd):
        if ask_ticks[d] >= 0 and util[d] >= best:
            cnt += 1
    k = int(ur * cnt)
    if k >= cnt:
        k = cnt - 1
    for d in range(nd):
        if ask_ticks[d] >= 0 and util[d] >= best:
            if k == 0:
                return d
            k -= 1
    return -1


@jit
def is_dealer_buyer(st: int) -> bool:
    return st == _ZI_BUYER or st == _INFORMED or st == _NOISE or st == _TREND


# ------------------------------------------------------------------- dealer-mode ticks
@jit
def dealer_ticks(
    t0: int,
    n_ticks: int,
    tape: np.ndarray,
    tpos: int,
    alias_prob: np.ndarray,
    alias_idx: np.ndarray,
    lam_total: float,
    stype: np.ndarray,
    value_base: np.ndarray,
    noise_sd: np.ndarray,
    trend_k: np.ndarray,
    sentiment: np.ndarray,
    belief_F: np.ndarray,
    last_update: np.ndarray,
    shock_delay: np.ndarray,
    inventory: np.ndarray,
    cash: np.ndarray,
    max_inv: np.ndarray,
    glob: np.ndarray,
    params: np.ndarray,
    ask_ticks: np.ndarray,
    ask_prices: np.ndarray,
    dealer_cash: np.ndarray,
    dealer_inv: np.ndarray,
    tr_tick: np.ndarray,
    tr_buyer: np.ndarray,
    tr_dealer: np.ndarray,
    tr_price: np.ndarray,
    tr_value: np.ndarray,
    tr_pos: int,
    flow_buy: np.ndarray,
    n_arr: np.ndarray,
    n_out: np.ndarray,
) -> tuple[int, int, int]:
    """Run up to ``n_ticks`` ticks of background flow against fixed dealer asks.

    ``glob`` = [G (global OU factor), H (smoothed factor), F (fundamental)], updated in place.
    ``params`` = [rho, sigma, w_s, alpha, tau, u0, trend_signal].
    Reservation value: ``v = alpha * (belief_F + eps + noise_sd*z2 + w_s*(H + x)) + trend_k*trend``.
    Stops early (without partial effects) when the tape or the trade buffer cannot hold the
    next tick; returns ``(ticks_done, tpos, tr_pos)``.
    """
    rho = params[0]
    sigma = params[1]
    w_s = params[2]
    alpha = params[3]
    tau = params[4]
    u0 = params[5]
    trend = params[6]
    nd = ask_ticks.shape[0]
    ntape = tape.shape[0]
    cap = tr_tick.shape[0]
    n_pieces = n_poisson_pieces(lam_total)
    piece = lam_total / n_pieces
    util = np.empty(nd, dtype=np.float64)
    w = np.empty(nd, dtype=np.float64)
    done = 0
    while done < n_ticks:
        t = t0 + done
        start = tpos
        if tpos + n_pieces + U_PER_TICK_EXTRA > ntape:
            break
        m = 0
        for _ in range(n_pieces):
            m += poisson_inv(tape[tpos], piece)
            tpos += 1
        if tpos + U_PER_TICK_EXTRA + U_PER_ARRIVAL * m > ntape or tr_pos + m > cap:
            tpos = start
            break
        zg, _unused = box_muller(tape[tpos], tape[tpos + 1])
        tpos += 2
        g_now = glob[0]
        h_now = glob[1]
        f_now = glob[2]
        nout = 0
        for _a in range(m):
            j = alias_draw(alias_prob, alias_idx, tape[tpos])
            ur = tape[tpos + 1]
            z1, z2 = box_muller(tape[tpos + 2], tape[tpos + 3])
            tpos += U_PER_ARRIVAL
            refresh_agent(
                j, t, z1, sentiment, last_update, belief_F, shock_delay, rho, sigma, f_now
            )
            if not is_dealer_buyer(stype[j]) or inventory[j] >= max_inv[j]:
                nout += 1
                continue
            v = (
                alpha
                * (belief_F[j] + value_base[j] + noise_sd[j] * z2 + w_s * (h_now + sentiment[j]))
                + trend_k[j] * trend
            )
            d = route_choice(v, ask_ticks, ask_prices, u0, tau, ur, util, w)
            if d < 0:
                nout += 1
                continue
            price = ask_prices[d]
            dealer_cash[d] += price
            dealer_inv[d] -= 1.0
            cash[j] -= price
            inventory[j] += 1.0
            tr_tick[tr_pos] = t
            tr_buyer[tr_pos] = j
            tr_dealer[tr_pos] = d
            tr_price[tr_pos] = ask_ticks[d]
            tr_value[tr_pos] = v
            tr_pos += 1
            flow_buy[done, d] += 1.0
        n_arr[done] = m
        n_out[done] = nout
        glob[1] = rho * h_now + (1.0 - rho) * g_now
        glob[0] = rho * g_now + sigma * zg
        done += 1
    return done, tpos, tr_pos


# ------------------------------------------------------------- public limit-order fringe
#
# Fringe depth lives on the dealer price grid, aggregated per level (no per-order objects):
#
# * ``depth[L]``     lots resting at level ``L``;
# * ``wheel[s, L]``  lots at level ``L`` posted at a tick ``t`` with ``t % H == s`` (expiry
#                    wheel, ``H`` = order lifetime in ticks); a lot posted at tick ``t`` can be
#                    hit during ticks ``t .. t+H-1`` and expires at the start of tick ``t+H``;
# * ``meta[s]``      ``[lots in slot s, lowest level, highest level]`` (bounds the expiry scan);
# * ``st``           ``[best level or -1, resting lots, posted, filled, expired]`` (the last
#                    three are cumulative: ``posted == filled + expired + resting`` always).
#
# A fill at level ``L`` consumes the oldest lot resting there (price-time priority).


@jit
def fringe_next_level(depth: np.ndarray, start: int) -> int:
    """Lowest level ``>= start`` with resting depth, or -1."""
    for lv in range(max(start, 0), depth.shape[0]):
        if depth[lv] > 0:
            return lv
    return -1


@jit
def fringe_expire(
    t: int, depth: np.ndarray, wheel: np.ndarray, meta: np.ndarray, st: np.ndarray
) -> int:
    """Remove the lots posted at tick ``t - H`` (wheel slot ``t % H``); returns their number."""
    s = t % wheel.shape[0]
    n = meta[s, 0]
    if n <= 0:
        return 0
    for lv in range(meta[s, 1], meta[s, 2] + 1):
        c = wheel[s, lv]
        if c > 0:
            depth[lv] -= c
            wheel[s, lv] = 0
    meta[s, 0] = 0
    st[1] -= n
    st[4] += n
    b = st[0]
    if st[1] <= 0:
        st[0] = -1
    elif b >= 0 and depth[b] <= 0:
        st[0] = fringe_next_level(depth, b)
    return n


@jit
def fringe_post(
    t: int, lv: int, depth: np.ndarray, wheel: np.ndarray, meta: np.ndarray, st: np.ndarray
) -> None:
    """Rest one lot at level ``lv``, posted at tick ``t``."""
    s = t % wheel.shape[0]
    if meta[s, 0] <= 0:
        meta[s, 1] = lv
        meta[s, 2] = lv
    else:
        if lv < meta[s, 1]:
            meta[s, 1] = lv
        if lv > meta[s, 2]:
            meta[s, 2] = lv
    meta[s, 0] += 1
    wheel[s, lv] += 1
    depth[lv] += 1
    st[1] += 1
    st[2] += 1
    if st[0] < 0 or lv < st[0]:
        st[0] = lv


@jit
def fringe_take(
    t: int, lv: int, depth: np.ndarray, wheel: np.ndarray, meta: np.ndarray, st: np.ndarray
) -> None:
    """Fill one lot at level ``lv`` against the oldest lot resting there (time priority)."""
    h = wheel.shape[0]
    for k in range(1, h + 1):
        s = (t + k) % h  # the oldest live posting tick is t-H+1, i.e. slot (t+1) % H
        if wheel[s, lv] > 0:
            wheel[s, lv] -= 1
            meta[s, 0] -= 1
            break
    depth[lv] -= 1
    st[1] -= 1
    st[3] += 1
    if depth[lv] <= 0 and st[0] == lv:
        st[0] = -1 if st[1] <= 0 else fringe_next_level(depth, lv)


@jit
def fringe_level(r: float, p_min: float, tick: float) -> int:
    """Grid level of a seller's reservation price, rounded UP (it never sells below it)."""
    lv = int(math.ceil((r - p_min) / tick - GRID_EPS))
    return lv if lv > 0 else 0


@jit
def route_choice_fringe(
    v: float,
    ask_ticks: np.ndarray,
    ask_prices: np.ndarray,
    f_price: float,
    u0: float,
    tau: float,
    ur: float,
    util: np.ndarray,
    w: np.ndarray,
) -> int:
    """Logit routing over the dealers, the best fringe ask and the outside option.

    P(dealer i) ∝ exp((v - ask_i)/tau), P(fringe) ∝ exp((v - f_price)/tau), outside ∝
    exp(u0/tau). Returns the dealer index, ``nd`` (the number of dealers) for the fringe, or
    -1 for the outside option. ``tau <= 0`` is exact price priority, ties split uniformly.
    """
    nd = ask_ticks.shape[0]
    best = u0
    for d in range(nd):
        if ask_ticks[d] >= 0:
            util[d] = v - ask_prices[d]
            if util[d] > best:
                best = util[d]
    uf = v - f_price
    if uf > best:
        best = uf
    if tau > 0.0:
        tot = math.exp((u0 - best) / tau)
        for d in range(nd):
            if ask_ticks[d] >= 0:
                w[d] = math.exp((util[d] - best) / tau)
            else:
                w[d] = 0.0
            tot += w[d]
        wf = math.exp((uf - best) / tau)
        tot += wf
        r = ur * tot
        acc = 0.0
        for d in range(nd):
            acc += w[d]
            if r < acc:
                return d
        if r < acc + wf:
            return nd
        return -1
    cnt = 1 if u0 >= best else 0
    for d in range(nd):
        if ask_ticks[d] >= 0 and util[d] >= best:
            cnt += 1
    if uf >= best:
        cnt += 1
    k = int(ur * cnt)
    if k >= cnt:
        k = cnt - 1
    for d in range(nd):
        if ask_ticks[d] >= 0 and util[d] >= best:
            if k == 0:
                return d
            k -= 1
    if uf >= best and k == 0:
        return nd
    return -1


@jit
def dealer_fringe_ticks(
    t0: int,
    n_ticks: int,
    tape: np.ndarray,
    tpos: int,
    alias_prob: np.ndarray,
    alias_idx: np.ndarray,
    lam_total: float,
    stype: np.ndarray,
    value_base: np.ndarray,
    noise_sd: np.ndarray,
    trend_k: np.ndarray,
    sentiment: np.ndarray,
    belief_F: np.ndarray,
    last_update: np.ndarray,
    shock_delay: np.ndarray,
    inventory: np.ndarray,
    cash: np.ndarray,
    max_inv: np.ndarray,
    glob: np.ndarray,
    params: np.ndarray,
    fparams: np.ndarray,
    ask_ticks: np.ndarray,
    ask_prices: np.ndarray,
    dealer_cash: np.ndarray,
    dealer_inv: np.ndarray,
    fr_depth: np.ndarray,
    fr_wheel: np.ndarray,
    fr_meta: np.ndarray,
    fr_st: np.ndarray,
    fr_acct: np.ndarray,
    tr_tick: np.ndarray,
    tr_buyer: np.ndarray,
    tr_dealer: np.ndarray,
    tr_price: np.ndarray,
    tr_value: np.ndarray,
    tr_pos: int,
    flow_buy: np.ndarray,
    fr_buy: np.ndarray,
    fr_post: np.ndarray,
    n_arr: np.ndarray,
    n_out: np.ndarray,
) -> tuple[int, int, int]:
    """:func:`dealer_ticks` plus the public limit-order fringe (mechanism B).

    ``fparams`` = [share, dealers' marginal cost (price units), p_min, tick size]. Per tick the
    lots posted ``H`` ticks ago expire, then arrivals are processed in order. An arriving ZI
    buyer with ``ur < share`` is a fringe seller: its reservation price is ``c + u (v - c)^+``
    (``c`` the dealers' marginal cost, ``v`` the reservation value it would have as a buyer,
    ``u = ur / share``) and it rests one lot there, rounded UP to the grid (dropped above the
    grid). Every other buyer routes by logit over the dealers, the best fringe level and the
    outside option; a fringe fill consumes the oldest lot at the best level and is recorded
    with dealer index ``FRINGE_ID``. ``fr_acct`` = [cash, inventory] of the anonymous fringe.
    Uniform consumption is exactly that of :func:`dealer_ticks`.
    """
    rho = params[0]
    sigma = params[1]
    w_s = params[2]
    alpha = params[3]
    tau = params[4]
    u0 = params[5]
    trend = params[6]
    share = fparams[0]
    mcost = fparams[1]
    p_min = fparams[2]
    tick = fparams[3]
    n_levels = fr_depth.shape[0]
    nd = ask_ticks.shape[0]
    ntape = tape.shape[0]
    cap = tr_tick.shape[0]
    n_pieces = n_poisson_pieces(lam_total)
    piece = lam_total / n_pieces
    util = np.empty(nd, dtype=np.float64)
    w = np.empty(nd, dtype=np.float64)
    done = 0
    while done < n_ticks:
        t = t0 + done
        start = tpos
        if tpos + n_pieces + U_PER_TICK_EXTRA > ntape:
            break
        m = 0
        for _ in range(n_pieces):
            m += poisson_inv(tape[tpos], piece)
            tpos += 1
        if tpos + U_PER_TICK_EXTRA + U_PER_ARRIVAL * m > ntape or tr_pos + m > cap:
            tpos = start
            break
        fringe_expire(t, fr_depth, fr_wheel, fr_meta, fr_st)
        zg, _unused = box_muller(tape[tpos], tape[tpos + 1])
        tpos += 2
        g_now = glob[0]
        h_now = glob[1]
        f_now = glob[2]
        nout = 0
        for _a in range(m):
            j = alias_draw(alias_prob, alias_idx, tape[tpos])
            ur = tape[tpos + 1]
            z1, z2 = box_muller(tape[tpos + 2], tape[tpos + 3])
            tpos += U_PER_ARRIVAL
            refresh_agent(
                j, t, z1, sentiment, last_update, belief_F, shock_delay, rho, sigma, f_now
            )
            st_j = stype[j]
            if not is_dealer_buyer(st_j):
                nout += 1
                continue
            v = (
                alpha
                * (belief_F[j] + value_base[j] + noise_sd[j] * z2 + w_s * (h_now + sentiment[j]))
                + trend_k[j] * trend
            )
            if st_j == _ZI_BUYER and share > 0.0:
                if ur < share:
                    gap = v - mcost
                    r = mcost + (ur / share) * gap if gap > 0.0 else mcost
                    lv = fringe_level(r, p_min, tick)
                    if lv < n_levels:
                        fringe_post(t, lv, fr_depth, fr_wheel, fr_meta, fr_st)
                        fr_post[done] += 1.0
                    else:
                        nout += 1
                    continue
                ur = (ur - share) / (1.0 - share)
            if inventory[j] >= max_inv[j]:
                nout += 1
                continue
            bl = fr_st[0]
            if bl >= 0:
                d = route_choice_fringe(
                    v, ask_ticks, ask_prices, p_min + bl * tick, u0, tau, ur, util, w
                )
            else:
                d = route_choice(v, ask_ticks, ask_prices, u0, tau, ur, util, w)
            if d < 0:
                nout += 1
                continue
            if d == nd:
                price = p_min + bl * tick
                fringe_take(t, bl, fr_depth, fr_wheel, fr_meta, fr_st)
                fr_acct[0] += price
                fr_acct[1] -= 1.0
                cash[j] -= price
                inventory[j] += 1.0
                tr_tick[tr_pos] = t
                tr_buyer[tr_pos] = j
                tr_dealer[tr_pos] = FRINGE_ID
                tr_price[tr_pos] = bl
                tr_value[tr_pos] = v
                tr_pos += 1
                fr_buy[done] += 1.0
                continue
            price = ask_prices[d]
            dealer_cash[d] += price
            dealer_inv[d] -= 1.0
            cash[j] -= price
            inventory[j] += 1.0
            tr_tick[tr_pos] = t
            tr_buyer[tr_pos] = j
            tr_dealer[tr_pos] = d
            tr_price[tr_pos] = ask_ticks[d]
            tr_value[tr_pos] = v
            tr_pos += 1
            flow_buy[done, d] += 1.0
        n_arr[done] = m
        n_out[done] = nout
        glob[1] = rho * h_now + (1.0 - rho) * g_now
        glob[0] = rho * g_now + sigma * zg
        done += 1
    return done, tpos, tr_pos


def uniforms_needed_upper(lam_total: float, n_ticks: int) -> int:
    """A generous tape size for ``n_ticks`` ticks (mean + 8 sd of the arrival count)."""
    pieces = max(1, math.ceil(lam_total / POIS_PIECE)) if lam_total > 0 else 1
    mean = lam_total * n_ticks
    m = mean + 8.0 * math.sqrt(max(mean, 1.0)) + 16
    return int(n_ticks * (pieces + U_PER_TICK_EXTRA) + U_PER_ARRIVAL * m)
