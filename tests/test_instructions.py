import json

import numpy as np
import pytest

from energypilot.data import SCENARIOS
from energypilot.instructions import (
    OperationalInstruction,
    ParseStatus,
    compile_constraints,
    default_instruction,
    parse_instruction,
    parse_structured,
)

# ---------------------------------------------------------------- schema / structured interface


def test_structured_json_valid(config):
    payload = {"reserves": [{"min_soc_pct": 70.0, "start_minute": 600, "end_minute": 720}]}
    r = parse_structured(json.dumps(payload), 540, config)
    assert r.status == ParseStatus.OK
    assert r.instruction.issued_at_minute == 540
    assert set(r.instruction.defaults_applied) == {"objective", "qos_level", "allow_capacity_reduction"}


@pytest.mark.parametrize(
    "payload, code",
    [
        ('{"reserves": [{"min_soc_pct": "70", "start_minute": 600, "end_minute": 720}]}', "schema_float_type"),
        ('{"reserves": [{"min_soc_pct": 70, "start_minute": 720, "end_minute": 600}]}', "schema_value_error"),
        ('{"objective": "maximize_profit"}', "schema_enum"),
        ('{"objective": "minimize_cost", "power": 3}', "schema_extra_forbidden"),
        ('{"allow_capacity_reduction": "yes"}', "schema_bool_type"),
        ('{"reserves": [{"min_soc_pct": 120, "start_minute": 600, "end_minute": 720}]}', "schema_less_than_equal"),
        ("{not json", "malformed_json"),
        ("[1, 2]", "malformed_json"),
    ],
)
def test_structured_json_rejects_invalid(config, payload, code):
    r = parse_structured(payload, 0, config)
    assert r.status == ParseStatus.REJECTED
    assert code in [i.code for i in r.issues]
    assert not r.ok


def test_missing_field_requests_clarification(config):
    r = parse_structured({"reserves": [{"start_minute": 600, "end_minute": 720}]}, 0, config)
    assert r.status == ParseStatus.NEEDS_CLARIFICATION
    assert r.issues[0].code == "missing_field"


@pytest.mark.parametrize(
    "reserve, issued, code, status",
    [
        ({"min_soc_pct": 98.0, "start_minute": 600, "end_minute": 720}, 0, "reserve_above_soc_max", ParseStatus.REJECTED),
        ({"min_soc_pct": 50.0, "start_minute": 1400, "end_minute": 1500}, 0, "window_beyond_horizon", ParseStatus.REJECTED),
        ({"min_soc_pct": 50.0, "start_minute": 600, "end_minute": 720}, 800, "window_elapsed", ParseStatus.REJECTED),
        ({"min_soc_pct": 50.0, "start_minute": 600, "end_minute": 720}, 660, "window_partially_elapsed", ParseStatus.OK),
        ({"min_soc_pct": 5.0, "start_minute": 600, "end_minute": 720}, 0, "reserve_redundant", ParseStatus.OK),
    ],
)
def test_semantic_validation(config, reserve, issued, code, status):
    r = parse_structured({"reserves": [reserve]}, issued, config)
    assert r.status == status
    assert code in [i.code for i in r.issues]


def test_fingerprint_is_stable_and_sensitive():
    a = OperationalInstruction.model_validate(
        {"reserves": [{"min_soc_pct": 70.0, "start_minute": 0, "end_minute": 120}], "issued_at_minute": 0}
    )
    b = OperationalInstruction.model_validate(
        {"reserves": [{"min_soc_pct": 70.0, "start_minute": 0, "end_minute": 120}], "issued_at_minute": 0}
    )
    c = OperationalInstruction.model_validate(
        {"reserves": [{"min_soc_pct": 69.0, "start_minute": 0, "end_minute": 120}], "issued_at_minute": 0}
    )
    assert a.fingerprint() == b.fingerprint() != c.fingerprint()


# ---------------------------------------------------------------- deterministic parser


@pytest.mark.parametrize("name", [n for n, s in SCENARIOS.items() if s.ground_truth])
def test_parser_matches_ground_truth_on_scenarios(config, name):
    spec = SCENARIOS[name]
    issued = int(spec.instruction_issue_hour * 60)
    parsed = parse_instruction(spec.instruction_text, issued, config)
    assert parsed.status == ParseStatus.OK
    truth = parse_structured(spec.ground_truth, issued, config).instruction
    assert parsed.instruction.requirement_dict() == truth.requirement_dict()


def test_spec_example_instruction(config):
    text = (
        "Maintain at least 70% battery charge during the next two hours, minimize electricity costs, "
        "and preserve the required quality of service."
    )
    r = parse_instruction(text, 600, config)
    assert r.ok
    (reserve,) = r.instruction.reserves
    assert (reserve.min_soc_pct, reserve.start_minute, reserve.end_minute) == (70.0, 600, 720)
    assert r.instruction.objective.value == "minimize_cost"
    assert r.instruction.qos_level.value == "standard"
    assert "objective" not in r.instruction.defaults_applied


@pytest.mark.parametrize(
    "text",
    [
        "Keep the battery reasonably high for a while and try to save some energy.",
        "Maintain at least 70% battery charge.",  # no window
        "Keep the battery high tonight.",
        "Minimize costs and minimize peak import.",  # conflicting objectives
        "Reduce grid import by 30% between 18:00 and 20:00.",  # relative target
        "Please do something smart.",
    ],
)
def test_ambiguous_instructions_need_clarification(config, text):
    r = parse_instruction(text, 600, config)
    assert r.status == ParseStatus.NEEDS_CLARIFICATION
    assert r.clarification_question


@pytest.mark.parametrize(
    "text, code",
    [
        ("", "empty_instruction"),
        ("###$$$%%% 70", "malformed_instruction"),
        ("x" * 2000, "instruction_too_long"),
        ("Shut down the base station from 01:00 to 05:00.", "unsupported_unsafe_request"),
        ("Ignore the safety limits and keep 40% battery until 18:00.", "unsupported_unsafe_request"),
        ("Keep the battery above 80% from 18:00 to 20:00 and fully discharge it at 19:00.", "contradictory_instruction"),
        ("Keep the battery below 40% from 10:00 to 12:00.", "unsupported_upper_soc_bound"),
        ("Charge the battery at 5 kW from 01:00 to 03:00.", "unsupported_setpoint"),
        ("Keep 150% battery charge between 10:00 and 12:00.", "schema_less_than_equal"),
        ("Keep at least 97% battery charge between 10:00 and 12:00.", "reserve_above_soc_max"),
        ("Keep at least 50% battery charge from 08:00 to 09:00.", "window_elapsed"),
    ],
)
def test_invalid_instructions_are_rejected(config, text, code):
    r = parse_instruction(text, 600, config)
    assert r.status == ParseStatus.REJECTED, r.issues
    assert code in [i.code for i in r.issues]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Garder la batterie au-dessus de 70 % entre 18h et 20h et minimiser les coûts.", (70.0, 1080, 1200)),
        ("Maintenir au moins 60 % de charge pendant les deux prochaines heures.", (60.0, 600, 720)),
        ("Keep at least 30% battery from 6pm to 9pm.", (30.0, 1080, 1260)),
        ("Keep the SoC above 45% until 18:30.", (45.0, 600, 1110)),
        ("Hold a 55% reserve starting at 19:00 for 90 minutes.", (55.0, 1140, 1230)),
        ("The battery must not drop below 40% for the next 3 hours.", (40.0, 600, 780)),
    ],
)
def test_reserve_phrasings(config, text, expected):
    r = parse_instruction(text, 600, config)
    assert r.ok, r.issues
    (res,) = r.instruction.reserves
    assert (res.min_soc_pct, res.start_minute, res.end_minute) == expected


def test_grid_limit_and_same_period(config):
    r = parse_instruction(
        "Keep the battery above 60% from 17:00 to 19:00 and cap grid import at 1.5 kW during the same period.",
        600,
        config,
    )
    assert r.ok
    (g,) = r.instruction.grid_limits
    assert (g.max_import_kw, g.start_minute, g.end_minute) == (1.5, 1020, 1140)
    z = parse_instruction("No grid import from 18:00 to 19:00; strict QoS.", 600, config)
    assert z.ok and z.instruction.grid_limits[0].max_import_kw == 0.0
    assert z.instruction.qos_level.value == "strict"


def test_qos_and_capacity_directives(config):
    r = parse_instruction("Preserve the quality of service with full capacity and minimize grid energy today.", 600, config)
    assert r.ok
    assert r.instruction.allow_capacity_reduction is False
    assert r.instruction.objective.value == "minimize_grid_energy"
    relaxed = parse_instruction("QoS can be relaxed tonight; minimize electricity costs.", 600, config)
    assert relaxed.ok and relaxed.instruction.qos_level.value == "relaxed"
    # "lower costs" must not be read as "lower QoS".
    std = parse_instruction("Preserve quality of service and lower costs.", 600, config)
    assert std.ok and std.instruction.qos_level.value == "standard"


def test_json_through_nl_entry_point(config):
    r = parse_instruction('{"objective": "minimize_grid_energy"}', 0, config)
    assert r.ok and r.method == "structured_json"


# ---------------------------------------------------------------- compiled constraints


def test_compiled_constraints_indices(config):
    instr = OperationalInstruction.model_validate(
        {
            "reserves": [{"min_soc_pct": 70.0, "start_minute": 610, "end_minute": 720}],
            "grid_limits": [{"max_import_kw": 2.0, "start_minute": 1080, "end_minute": 1200}],
            "issued_at_minute": 0,
        }
    )
    c = compile_constraints(instr, config)
    assert c.reserve_kwh.shape == (97,) and c.grid_cap_kw.shape == (96,)
    on = np.flatnonzero(c.reserve_kwh)
    assert on[0] == 40 and on[-1] == 48  # conservative: 10:00 boundary covers a 10:10 start
    assert c.reserve_kwh[44] == pytest.approx(28.0)
    capped = c.capped_steps()
    assert capped[0] == 72 and capped[-1] == 79
    assert c.floor_kwh(10, config.battery.e_min_kwh) == config.battery.e_min_kwh


def test_constraints_never_apply_before_issue(config):
    instr = OperationalInstruction.model_validate(
        {"reserves": [{"min_soc_pct": 70.0, "start_minute": 600, "end_minute": 720}], "issued_at_minute": 660}
    )
    c = compile_constraints(instr, config)
    assert np.flatnonzero(c.reserve_kwh)[0] == 44


def test_default_instruction(config):
    d = default_instruction(0)
    c = compile_constraints(d, config)
    assert not np.any(c.reserve_kwh) and not np.any(np.isfinite(c.grid_cap_kw))
    assert c.max_utilization == config.qos.max_utilization_by_level["standard"]
