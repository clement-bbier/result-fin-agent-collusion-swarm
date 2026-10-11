import pytest
from pydantic import ValidationError

from finswarm.config.load import config_hash, run_id
from finswarm.config.schema import LLMSlotConfig, LLMTierConfig, RunConfig


def test_defaults_validate() -> None:
    cfg = RunConfig(name="x")
    assert cfg.market.n_dealers == 2
    assert cfg.llm.mode == "mock"
    assert cfg.market.mechanism == "dealer"


def test_extra_keys_are_forbidden() -> None:
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"name": "x", "periodz": 3})
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"name": "x", "market": {"n_dealerz": 3}})


def test_mixed_families_rejected_unless_allowed() -> None:
    slots = [
        LLMSlotConfig(name="A", model="m1", family="anthropic"),
        LLMSlotConfig(name="B", model="m2", family="mistral"),
    ]
    with pytest.raises(ValidationError):
        LLMTierConfig(slots=slots)
    LLMTierConfig(slots=slots, allow_mixed_families=True)


def test_run_id_deterministic_and_sensitive() -> None:
    c1, c2 = RunConfig(name="x"), RunConfig(name="x")
    assert run_id(c1) == run_id(c2)
    assert run_id(c1, seed=1) != run_id(c1, seed=2)
    assert config_hash(c1) != config_hash(RunConfig(name="x", periods=10))
    assert run_id(c1).startswith("x__s0__")


def test_events_parse_inside_run_config() -> None:
    cfg = RunConfig.model_validate(
        {
            "name": "x",
            "events": [
                {
                    "type": "set_param",
                    "path": "market.price_grid.tick",
                    "value": 0.05,
                    "when": {"at_period": 100},
                },
                {"type": "sting", "slot": 0, "delta": 0.2},
            ],
        }
    )
    assert cfg.events[0].type == "set_param"
    assert cfg.events[0].when.at_period == 100
    assert cfg.events[1].type == "sting"
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"name": "x", "events": [{"type": "teleport"}]})
