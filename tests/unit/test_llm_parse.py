import asyncio

from finswarm.agents.llm.schemas import CDADecision, JudgeScore, OrderDecision, PriceDecision
from finswarm.agents.llm.templates import parse_fish_sections
from finswarm.llm.parse import extract_json, parse_text, parse_with_repair

GOOD = '{"observations": "o", "plans": "p", "insights": "i", "price": 1.85}'


def test_extract_unfenced_and_fenced() -> None:
    assert extract_json(GOOD)["price"] == 1.85
    fenced = f"Here is my answer:\n```json\n{GOOD}\n```\nThanks."
    assert extract_json(fenced)["price"] == 1.85
    bare_fence = f"```\n{GOOD}\n```"
    assert extract_json(bare_fence)["plans"] == "p"


def test_extract_with_leading_and_trailing_text() -> None:
    text = f"Let me think {{about it}}. Final answer: {GOOD} -- hope that helps {{no json}}"
    assert extract_json(text)["price"] == 1.85


def test_extract_last_object_wins_and_trailing_comma() -> None:
    text = 'Example: {"price": 9.99}. Real answer: {"plans": "a", "price": 1.5,}'
    data = extract_json(text)
    assert data == {"plans": "a", "price": 1.5}


def test_extract_none_when_no_json() -> None:
    assert extract_json("I would set the price at 1.8.") is None
    assert extract_json("") is None


def test_validation_coerces_and_drops_unknown_keys() -> None:
    value, err = parse_text(
        '{"plans": "p", "insights": "i", "price": "$1.80", "reasoning": "x"}', PriceDecision
    )
    assert err is None and value is not None and value.price == 1.8
    bad, err = parse_text('{"plans": "p", "price": -3}', PriceDecision)
    assert bad is None and "price" in err


def test_order_and_cda_and_judge_schemas() -> None:
    v, _ = parse_text('{"bid": 1.2, "ask": 1.5}', OrderDecision)
    assert v is not None and v.bid == 1.2
    v, err = parse_text('{"bid": 1.6, "ask": 1.5}', OrderDecision)
    assert v is None and "bid must be strictly below ask" in err
    v, _ = parse_text('{"price": null, "message": null}', CDADecision)
    assert v is not None and v.price is None
    v, _ = parse_text('{"score": 3, "rationale": "r", "evidence": "e"}', JudgeScore)
    assert v is not None and v.score == 3
    v, _ = parse_text('{"score": 5, "rationale": "r"}', JudgeScore)
    assert v is None


def test_repair_success_and_default_fallback() -> None:
    calls: list[tuple[str, str]] = []

    async def repair_ok(raw: str, error: str) -> str:
        calls.append((raw, error))
        return GOOD

    res = asyncio.run(parse_with_repair("not json at all", PriceDecision, repair=repair_ok))
    assert res.parse_ok and res.repair_used and res.attempts == 2
    assert res.value is not None and res.value.price == 1.85
    assert calls[0][0] == "not json at all" and "no JSON" in calls[0][1]

    async def repair_bad(raw: str, error: str) -> str:
        return "still bad"

    default = PriceDecision(price=1.47)
    res = asyncio.run(
        parse_with_repair("bad", PriceDecision, repair=repair_bad, max_repairs=1, default=default)
    )
    assert not res.parse_ok and res.repair_used and res.value is default
    assert res.raw == "still bad" and res.error

    res = asyncio.run(parse_with_repair(GOOD, PriceDecision, repair=repair_bad))
    assert res.parse_ok and not res.repair_used and res.attempts == 1


def test_fish_sections_parser() -> None:
    text = (
        "**My observations and thoughts:**\nThe rival priced at 1.80.\n\n"
        "**New content for PLANS.txt:**\nTest 1.85 next.\n\n"
        "New content for INSIGHTS.txt:\nHigher prices kept profit stable.\n\n"
        "My chosen price:\nI will set $1.85 this round."
    )
    out = parse_fish_sections(text)
    assert out is not None
    assert out["price"] == 1.85
    assert out["plans"] == "Test 1.85 next."
    assert out["insights"] == "Higher prices kept profit stable."
    assert out["observations"].startswith("The rival")
    assert parse_fish_sections("My chosen price: 2") == {
        "price": 2.0,
        "observations": "",
        "plans": "",
        "insights": "",
    }
    assert parse_fish_sections("My chosen price:\n<just the number, nothing else>") is None
    assert parse_fish_sections("no template here") is None
    value, err = parse_text(text, PriceDecision, parse_fish_sections)
    assert err is None and value is not None and value.price == 1.85
    # a model that answers in JSON despite the text template is still parsed
    value, err = parse_text(GOOD, PriceDecision, parse_fish_sections)
    assert value is not None and value.price == 1.85
