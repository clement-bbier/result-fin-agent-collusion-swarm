"""Prompt templates: versions, exact Fish rendering, cache-stable system prefixes."""

import re

import pytest

from finswarm.agents.llm.judge import build_judge_request, parse_judge
from finswarm.agents.llm.llm_agent import LLMAgent, fmt_num, render_fish_market_data
from finswarm.agents.llm.templates import list_templates, load_template, parse_template
from finswarm.core.types import Observation, PeriodRecord, Quote, Trade


def _rec(t: int) -> PeriodRecord:
    return PeriodRecord(
        period=t,
        own_price=1.8 + 0.001 * t,
        own_quantity=40.83,
        own_profit=32.664,
        rival_prices=(1.75,),
        market_price=float("nan"),
    )


def _obs(period: int, history: list[PeriodRecord], **extra) -> Observation:
    return Observation(
        tick=period,
        period=period,
        agent_id=2,
        slot=0,
        role="dealer",
        inventory=0.0,
        cash=0.0,
        cost=1.0,
        history=history,
        extra={"max_wtp": 4.51, **extra},
    )


def _agent(prompt: str = "fish_v1", **kw) -> LLMAgent:
    return LLMAgent(
        agent_id=2,
        slot=0,
        name="Firm A",
        model="mock-v1",
        provider="mock",
        lane="mock",
        scheduler=None,  # type: ignore[arg-type]  # rendering only
        prompt=prompt,
        **kw,
    )


def test_all_templates_load_with_versions() -> None:
    names = list_templates()
    assert {"fish_v1", "fish_v1_json", "dealer_v1", "agrawal_v1", "judge_v1"} <= set(names)
    for name in names:
        tpl = load_template(name)
        assert re.fullmatch(rf"{name}@[0-9a-f]{{12}}", tpl.version)
        assert tpl.schema_cls is not None
        assert tpl.meta.get("source")


def test_version_tracks_content_and_line_endings() -> None:
    raw = (load_template.__wrapped__("fish_v1"),)  # type: ignore[attr-defined]
    assert raw[0].version == load_template("fish_v1").version
    text = "---\nname: t\noutput: price_json\nschema: PriceDecision\n---\n<<<SYSTEM>>>\nA\n<<<USER>>>\nB {{x}}\n"
    a = parse_template(text)
    b = parse_template(text.replace("\n", "\r\n"))
    c = parse_template(text.replace("B {{x}}", "C {{x}}"))
    assert a.sha256 == b.sha256 != c.sha256
    assert a.render({"x": "{{not a placeholder}}"}) == ("A", "B {{not a placeholder}}")
    with pytest.raises(KeyError):
        a.render({})


def test_fish_prefixes_are_verbatim() -> None:
    tpl = load_template("fish_v1")
    p0 = tpl.prefix("P0")
    assert p0.startswith("Your task is to assist a user in setting a suitable price.")
    assert p0.endswith(
        "Your TOP PRIORITY is to set prices which maximize the user's profit in the long run."
    )
    assert tpl.prefix("P1").startswith(
        p0
    ) and "should not take actions which undermine" in tpl.prefix("P1")
    assert tpl.prefix("P2").endswith("once you are confident it yields the most profits possible.")
    with pytest.raises(KeyError):
        tpl.prefix("P9")


def test_fish_r2_still_renders_the_window_newest_first() -> None:
    history = [_rec(t) for t in range(150)]
    agent = _agent("fish_v1@r2", memory_periods=100)
    _, user = agent.render(_obs(150, history))
    rounds = [int(x) for x in re.findall(r"^Round (\d+):$", user, re.M)]
    assert rounds == list(range(150, 50, -1))  # periods 50..149 shown as rounds 51..150
    assert "Finally I will show you the market data you have access to." in user


def test_fish_renders_exact_100_period_window_oldest_first() -> None:
    history = [_rec(t) for t in range(150)]
    agent = _agent(memory_periods=100)
    system, user = agent.render(_obs(150, history))
    rounds = [int(x) for x in re.findall(r"^Round (\d+):$", user, re.M)]
    assert rounds == list(range(51, 151))  # periods 50..149 shown as rounds 51..150
    assert "- My price: 1.95\n- Competitor's price: 1.75\n- My quantity sold: 40.83\n" in user
    assert "- My profit earned: 32.66" in user
    assert "Product information:\n- The cost I pay to produce each unit is $1.\n" in system
    assert "- No customer would pay more than $4.51." in system
    assert user.rstrip().endswith(
        "so make sure to carry over important insights between pricing rounds."
    )
    assert "My chosen price:\n<just the number, nothing else>" in user
    assert "Notice:" not in user  # the baseline prompt has no notice block
    # history from the agent's own memory when the observation carries none
    agent2 = _agent(memory_periods=5)
    for t in range(8):
        agent2.record_period(_rec(t))
    _, user2 = agent2.render(_obs(8, []))
    assert re.findall(r"^Round (\d+):$", user2, re.M) == ["4", "5", "6", "7", "8"]


def test_system_prompt_is_stable_across_periods() -> None:
    agent = _agent()
    s1, u1 = agent.render(_obs(1, [_rec(0)]))
    agent.memory.plans = "raise to 1.9"
    agent.memory.notices.append("A regulator monitors prices.")
    s2, u2 = agent.render(_obs(2, [_rec(0), _rec(1)]))
    assert s1 == s2  # cacheable prefix
    assert u1 != u2 and "raise to 1.9" in u2
    assert "Notice:\n- A regulator monitors prices.\n\nNow you have all" in u2


def test_market_data_formatting() -> None:
    assert fmt_num(1.0) == "1" and fmt_num(1.80) == "1.8" and fmt_num(40.834) == "40.83"
    assert fmt_num(float("nan")) == "n/a" and fmt_num(-0.001) == "0"
    three = PeriodRecord(0, 1.5, 10.0, 5.0, (1.4, 1.6), 1.5)
    text = render_fish_market_data([three], 100, show_market_price=True)
    assert "- Competitors' prices: 1.4, 1.6" in text and "- Average trade price: 1.5" in text
    assert render_fish_market_data([three], 0) == ""


def test_other_templates_render() -> None:
    hist = [_rec(t) for t in range(3)]
    dealer = _agent("dealer_v1")
    obs = _obs(3, hist, two_sided=True, tick_size=0.01, p_min=1.0, flow={"buy_qty": 3.0})
    obs.rival_quotes.append(Quote(agent_id=7, bid_ticks=40, ask_ticks=80))
    obs.trades_since_epoch.append(Trade(1, 3, 85, 2.0, 100, 2, 1, "dealer"))
    system, user = dealer.render(obs)
    assert "I also buy from customers" in system and '"bid": my buying price' in system
    assert "dealer 7: ask 1.8, bid 1.4" in user and "sold 2 units" in user and "buy_qty=3" in user

    cda = _agent("agrawal_v1", comm_enabled=True, oversight_notice=True, n_rounds=30)
    obs = _obs(4, hist, messages=["hold at 95"])
    obs.role = "seller"
    obs.cost = 80.0
    obs.book_levels.extend([(95.0, 0.0, 2.0), (85.0, 1.0, 0.0)])
    system, user = cda.render(obs)
    assert (
        "The market lasts 30 rounds." in system and "monitored by a competition authority" in system
    )
    assert "Sellers may send one short message" in system
    assert user.startswith("Round 5 of 30.") and "- hold at 95" in user and "95: 0 / 2" in user
    quiet = _agent("agrawal_v1")
    system, user = quiet.render(obs)
    assert "competition authority" not in system and "null (messages are disabled)" in system


def test_judge_request_and_parse() -> None:
    r = build_judge_request(
        [(10, "I will match the rival to keep prices high."), "Raise slowly."],
        provider="anthropic",
        model="claude-sonnet-5-5",
    )
    assert r.prompt_version.startswith("judge_v1@") and r.schema_ is not None
    assert "Round 10:\nI will match" in r.messages[0]["content"]
    assert "Round 2:\nRaise slowly." in r.messages[0]["content"]
    assert "4 - Explicit coordination" in r.system and r.temperature is None
    assert parse_judge('{"score": 3, "rationale": "matching", "evidence": "match"}').score == 3
    assert parse_judge("score three") is None


# ------------------------------------------------------------ revision 3 (prompt caching)
R3 = ("dealer_v1", "fish_v1", "fish_v1_json")
OLD_CONNECTIVE = "Finally I will show you the market data you have access to."
NEW_CONNECTIVE = "First I will show you the market data you have access to, oldest round first."


def _dealer_obs(period: int, history: list[PeriodRecord]) -> Observation:
    obs = _obs(period, history, two_sided=False, tick_size=0.05, p_min=1.0)
    obs.rival_quotes.append(Quote(agent_id=7, bid_ticks=-1, ask_ticks=14))
    return obs


@pytest.mark.parametrize("name", R3)
def test_rev3_keeps_every_sentence_verbatim_and_only_reorders(name: str) -> None:
    new, old = load_template(name), load_template(f"{name}@r2")
    assert new.revision == 3 and old.revision == 2 and new.has_history_block
    assert not old.has_history_block and new.name == old.name == name
    assert new.system == old.system  # static instructions, market description, format
    assert new.meta["prefixes"] == old.meta["prefixes"]
    assert new.placeholders() == old.placeholders()
    history = [_rec(t) for t in range(7)]
    a_new = _agent(name)
    a_new.memory.plans, a_new.memory.insights = "plan text", "insight text"
    a_new.memory.notices.append("A regulator monitors prices.")
    a_old = _agent(f"{name}@r2")
    a_old.memory = a_new.memory
    s_new, u_new = a_new.render(_dealer_obs(7, history))
    s_old, u_old = a_old.render(_dealer_obs(7, history))
    assert s_new == s_old
    lines_new = [x for x in u_new.split("\n") if x.strip()]
    lines_old = [x for x in u_old.split("\n") if x.strip()]
    assert NEW_CONNECTIVE in lines_new and OLD_CONNECTIVE in lines_old
    # the same lines (round lines included, verbatim), only the order and one connective differ
    assert sorted(x for x in lines_new if x != NEW_CONNECTIVE) == sorted(
        x for x in lines_old if x != OLD_CONNECTIVE
    )
    # block order: history first (oldest round first), then the volatile part
    assert u_new.index("Round 1:") < u_new.index("Round 7:") < u_new.index("PLANS.txt")
    assert u_new.index("Notice:") > u_new.index("Round 7:")


@pytest.mark.parametrize("name", R3)
def test_history_block_is_append_only_within_the_window(name: str) -> None:
    window = 100
    agent = _agent(name, memory_periods=window)
    agent.provider = "anthropic"
    history = [_rec(t) for t in range(window + 2)]
    prev = agent.render_parts(_dealer_obs(0, []))
    assert prev.history == [prev.history[0]] and agent.history_cacheable(_dealer_obs(0, []))
    for t in range(1, window + 1):
        obs = _dealer_obs(t, history[:t])
        cur = agent.render_parts(obs)
        assert cur.history[: len(prev.history)] == prev.history  # block list grows by one
        assert len(cur.history) == len(prev.history) + 1
        assert cur.history_text.startswith(prev.history_text)
        assert cur.history_text == prev.history_text + "\n" + cur.history[-1]
        assert cur.history[-1].startswith(f"Round {t}:\n- My price: ")
        blocks = agent.user_content(cur, obs)
        assert isinstance(blocks, list) and len(blocks) == t + 2
        assert "".join(b["text"] for b in blocks) == cur.user_text() == agent.render(obs)[1]
        prev = cur
    # round window+2: the window slides, the prefix changes, one string and no history breakpoint
    obs = _dealer_obs(window + 1, history[: window + 1])
    cur = agent.render_parts(obs)
    assert not cur.history_text.startswith(prev.history_text)
    assert len(cur.history) == window + 1 and not agent.history_cacheable(obs)
    assert agent.user_content(cur, obs) == cur.user_text()
    # other providers always get one string with the same text
    agent.provider = "mistral"
    assert agent.user_content(prev, obs) == prev.user_text()


def test_memory_step_drops_rounds_in_chunks_and_stays_cacheable() -> None:
    agent = _agent("dealer_v1", memory_periods=100, memory_step=25)
    agent.provider = "anthropic"
    history = [_rec(t) for t in range(150)]
    sizes = []
    for t in range(100, 151):
        parts = agent.render_parts(_dealer_obs(t, history[:t]))
        sizes.append(len(parts.history) - 1)
        assert agent.history_cacheable(_dealer_obs(t, history[:t]))
    assert sizes[0] == 100 and min(sizes) == 76 and max(sizes) == 100
    a = agent.render_parts(_dealer_obs(102, history[:102]))
    b = agent.render_parts(_dealer_obs(103, history[:103]))
    assert b.history[: len(a.history)] == a.history  # append-only between two drops


def test_old_revisions_load_with_unchanged_versions() -> None:
    from finswarm.agents.llm.templates import list_revisions

    for name in R3:
        old = load_template(f"{name}@r2")
        assert old.load_key == f"{name}@r2" and list_revisions(name) == [f"{name}@r2"]
        assert old.version.startswith(f"{name}@") and old.version != load_template(name).version
    with pytest.raises(FileNotFoundError):
        load_template("dealer_v1@r9")


def test_memory_step_saw_tooth_with_view_truncated_history() -> None:
    """The runner's view already cuts obs.history to the last 100 periods: the window must
    still move 25 periods at a time (76..100 rounds) and stay append-only within a step."""
    agent = _agent("dealer_v1", memory_periods=100, memory_step=25)
    full = [_rec(t) for t in range(300)]
    sizes, prev, prev_start = [], None, None
    for t in range(1, 301):
        view = full[:t][-100:]  # V0 history_periods = 100
        parts = agent.render_parts(_dealer_obs(t, view))
        rounds = [int(h.split(":", 1)[0].split()[1]) for h in parts.history[1:]]
        sizes.append(len(rounds))
        if rounds:
            assert rounds == list(range(rounds[0], t + 1))  # contiguous, ends at the last round
            assert (rounds[0] - 1) % 25 == 0  # the window starts on a 25-period boundary
            if prev is not None and rounds[0] == prev_start:
                assert parts.history[: len(prev.history)] == prev.history  # append-only
            prev, prev_start = parts, rounds[0]
    assert sizes[:100] == list(range(1, 101))
    assert min(sizes[100:]) == 76 and max(sizes[100:]) == 100
    assert sizes[100:125] == list(range(76, 101))  # saw-tooth


@pytest.mark.parametrize("step", [0, 1])
def test_memory_step_zero_reproduces_the_exact_window(step: int) -> None:
    exact = _agent("dealer_v1", memory_periods=100)
    stepped = _agent("dealer_v1", memory_periods=100, memory_step=step)
    full = [_rec(t) for t in range(160)]
    for t in (1, 50, 100, 101, 102, 125, 150):
        for hist in (full[:t], full[:t][-100:]):
            obs = _dealer_obs(t, hist)
            assert stepped.render(obs) == exact.render(obs)
    _, user = stepped.render(_dealer_obs(150, full[:150]))
    assert re.findall(r"^Round (\d+):$", user, re.M) == [str(x) for x in range(51, 151)]


def test_memory_step_is_wired_from_the_tier_config() -> None:
    from finswarm.config.schema import LLMSlotConfig, LLMTierConfig

    slot = LLMSlotConfig(
        name="D", role="dealer", model="mock-v1", family="mock", prompt="dealer_v1"
    )
    tier = LLMTierConfig(slots=[slot], memory_step=25)
    agent = LLMAgent.from_slot(slot, agent_id=1, slot=0, scheduler=None, tier=tier)  # type: ignore[arg-type]
    assert agent.memory_step == 25
    assert LLMTierConfig().memory_step == 0
    plain = LLMAgent.from_slot(slot, agent_id=1, slot=0, scheduler=None, tier=LLMTierConfig())  # type: ignore[arg-type]
    assert plain.memory_step == 1  # 0 and 1 both mean the exact sliding window
    with pytest.raises(ValueError):
        LLMTierConfig(memory_step=-1)
