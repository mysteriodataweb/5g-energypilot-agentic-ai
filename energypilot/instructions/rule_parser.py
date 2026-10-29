"""Deterministic natural-language instruction parser (English and French subset).

This is the *non-LLM* interpretation baseline. It recognises a documented set of
phrasings with regular expressions and never guesses: a reserve without a level
or a window, vague wording, conflicting objectives or unsupported requests lead
to a clarification request or a rejection, never to an invented value.

Recognised directives
---------------------
* Minimum battery reserve: "<lower-bound word> N %" + a time window.
* Grid-import cap: "<cap word> grid import ... N kW" (+ window), or zero-import phrases.
* Objective: minimise cost / grid energy / peak import.
* QoS level: strict / standard / relaxed, utilisation cap "utilisation below N %",
  and "no capacity reduction / full capacity".
* Time windows: "between/from HH[:MM] and/to HH[:MM]", "next N hours/minutes",
  "for N hours", "until HH:MM", "starting at HH for N hours", "from HH onwards",
  "today / rest of the day", and "the same period" (refers to the previous window).

Known limitations are deliberate and documented in docs/instruction_interface.md.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from energypilot.config.settings import AppConfig
from energypilot.instructions.schema import Issue, Objective, ParseResult, QoSLevel
from energypilot.instructions.validation import (
    build_result,
    instruction_from_mapping,
    parse_structured,
)

METHOD = "deterministic_rules"

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a": 1, "an": 1,
    "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "sept": 7, "huit": 8,
    "neuf": 9, "dix": 10, "onze": 11, "douze": 12,
}  # fmt: skip
_NUM = r"(\d+(?:[.,]\d+)?|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")"
_DIGITS = r"(\d+(?:[.,]\d+)?)"
_CLK = r"(\d{1,2}(?::\d{2}|h\d{2}|h)?(?:\s*(?:am|pm))?|noon|midday|midnight|midi|minuit)"
_HOURS = r"(hours?|hrs?|h|heures?)"
_MINUTES = r"(minutes?|mins?)"
_UNIT = r"(hours?|hrs?|h|heures?|minutes?|mins?)"

# --- windows -------------------------------------------------------------------------
_W_BETWEEN = re.compile(
    rf"\b(?:between|from|de|entre)\s+{_CLK}\s*(?:and|to|until|till|-|à|a|et|jusqu'à|jusqu'a)\s*{_CLK}"
)
_W_START_FOR = re.compile(
    rf"\b(?:starting|beginning|from|à partir de|a partir de|dès)\s+(?:at\s+|de\s+)?{_CLK}\s+(?:for|pendant)\s+{_NUM}\s+{_UNIT}\b"
)
_W_NEXT_EN = re.compile(rf"\b(?:the\s+)?(?:next|coming|following)\s+(?:{_NUM}\s+)?(half[- ]hour|{_UNIT[1:-1]})\b")
_W_NEXT_FR = re.compile(rf"\b(?:les\s+|la\s+)?(?:{_NUM}\s+)?prochaines?\s+(heures?|minutes?|h|demi-heure)\b")
_W_FOR = re.compile(rf"\b(?:for|during|over|pendant|durant)\s+{_NUM}\s+{_UNIT}\b")
_W_HALF_HOUR = re.compile(r"\b(?:for|during|over|pendant)\s+(?:the\s+next\s+)?(?:half\s+an?\s+hour|une\s+demi-heure)\b")
_W_UNTIL = re.compile(rf"\b(?:until|till|up to|jusqu'à|jusqu'a)\s+{_CLK}")
_W_ONWARDS = re.compile(rf"\b(?:from|after|à partir de|a partir de|dès)\s+{_CLK}(?:\s+(?:onwards?|on))?\b")
_W_DAY = re.compile(
    r"\b(today|all day|the whole day|the entire day|(?:the\s+)?rest of the day|until the end of the day|"
    r"aujourd'hui|toute la journée|(?:le\s+)?reste de la journée|jusqu'à la fin de la journée)\b"
)
_W_SAME = re.compile(r"\b(same (?:period|time|window|interval|hours)|that period|this period|même période|même créneau)\b")
_VAGUE_TIME = re.compile(
    r"\b(for a while|for some time|a while|some time|later|soon|this evening|tonight|this afternoon|"
    r"this morning|un moment|un certain temps|plus tard|bientôt|ce soir|cet après-midi|ce matin)\b"
)

# --- values --------------------------------------------------------------------------
_PCT = re.compile(rf"{_DIGITS}\s*(?:%|percent\b|per cent\b|pour ?cent\b|pourcent\b)")
_KW = re.compile(rf"{_DIGITS}\s*(?:kw|kilowatts?)\b")
_BATTERY_CTX = re.compile(
    r"\b(battery|batteries|batterie|soc|state of charge|état de charge|charge|charged|reserve|réserve|"
    r"backup|autonomy|autonomie|storage|stockage)\b"
)
_GRID_CTX = re.compile(r"\b(grid|réseau|reseau|import|imports|soutirage|utility|mains|power draw)\b")
_UTIL_CTX = re.compile(r"\b(utili[sz]ation|load factor|taux d'utilisation)\b")
_LOWER_NEGATED = re.compile(
    r"(not|never|ne pas|jamais)\s+(?:\w+\s+){0,2}(below|under|less than|sous|en dessous)"
)
_LOWER_WORDS = re.compile(
    r"(at least|minimum|min\b|above|over|no less than|>=|keep|maintain|raise|increase|hold|retain|"
    r"reserve|au moins|au minimum|au-dessus|supérieure? à|plus de|garder|maintenir|conserver|relever|"
    r"augmenter|réserve)"
)
_UPPER_WORDS = re.compile(
    r"(at most|below|under|less than|no more than|maximum|max\b|<=|au plus|en dessous|inférieure? à|"
    r"moins de|discharge to|décharger à)"
)
_ZERO_IMPORT = re.compile(
    r"\b(no grid import|zero grid import|without (?:any )?grid(?: import| power)?|no (?:power|energy) from the grid|"
    r"island(?:ed)? mode|off[- ]grid|aucun soutirage|sans (?:le )?réseau|zéro import|pas d'import)\b"
)
_BATTERY_POWER_SETPOINT = re.compile(
    rf"\b(charge|discharge|charger|décharger)\b[^.;]{{0,40}}?\bat\s+{_DIGITS}\s*(?:kw|kilowatts?)\b"
)

# --- objective / QoS -----------------------------------------------------------------
_OBJ_COST = re.compile(
    r"\b(minimi[sz]e|minimi[sz]ing|reduce|reducing|lower|lowering|cut|cutting|optimi[sz]e|save on|keep down|"
    r"minimiser|réduire|reduire|baisser|diminuer)\s+(?:the\s+|les\s+|la\s+|le\s+|our\s+)?"
    r"(?:electricity\s+|energy\s+|power\s+|operating\s+|grid\s+|d'électricité\s+|énergétiques?\s+)?"
    r"(costs?|bills?|spend(?:ing)?|expenses?|coûts?|couts?|factures?|dépenses?)\b|"
    r"\b(save money|cheapest|lowest cost|least cost|moindre coût|économiser de l'argent)\b"
)
_OBJ_GRID = re.compile(
    r"\b(minimi[sz]e|reduce|lower|cut|minimiser|réduire)\s+(?:the\s+|total\s+|le\s+|la\s+)*"
    r"(grid (?:energy|imports?|consumption|usage|draw)|consommation réseau|soutirage)\b|"
    r"\bself[- ]consumption\b|\bautoconsommation\b|"
    r"\b(maximi[sz]e|use as much)\s+(?:the\s+)?(?:local\s+)?(?:solar|pv|photovoltaic)"
)
_OBJ_PEAK = re.compile(
    r"\b(peak[- ]shaving|shave (?:the )?peaks?|(?:minimi[sz]e|reduce|lower|limit) (?:the )?peak"
    r"(?: grid)?(?: import| demand| power)?|écrêt\w*|réduire (?:la )?pointe)\b"
)
_VAGUE_OBJECTIVE = re.compile(r"\b(save (?:some )?energy|be (?:more )?efficient|économiser (?:de l')?énergie)\b")
_QOS_TERM = r"(?:qos|quality of service|service quality|qualité de service|service level)"
# A level word must be attached to the QoS term ("strict QoS", "QoS can be relaxed", "qualité de service stricte").
_QOS_LEVEL = re.compile(
    rf"\b(strict|high|highest|premium|guaranteed|relaxed|reduced|degraded|lower|low|minimal|best[- ]effort|"
    rf"standard|normal|nominal|usual|required)\s+(?:level of\s+)?{_QOS_TERM}\b|"
    rf"\b{_QOS_TERM}\s+(?:is\s+|must be\s+|should be\s+|can be\s+|may be\s+|remains\s+|level\s+)?(?:\w+\s+)?"
    rf"(strict|high|relaxed|reduced|degraded|lowered|standard|normal|stricte|élevée|réduite|dégradée|normale|"
    rf"garantie|requise|minimale)\b"
)
_QOS_PRESERVE = re.compile(
    rf"\b(preserv\w*|maintain\w*|keep\w*|ensur\w*|guarantee\w*|respect\w*|préserver|maintenir|garantir|respecter)\s+"
    rf"(?:the\s+|la\s+)?(?:required\s+|current\s+|usual\s+|normal\s+|requise\s+)?{_QOS_TERM}\b"
)
_QOS_WORD_LEVEL = {
    **dict.fromkeys(("strict", "high", "highest", "premium", "guaranteed", "stricte", "élevée", "garantie"), QoSLevel.STRICT),
    **dict.fromkeys(
        ("relaxed", "reduced", "degraded", "lower", "low", "lowered", "minimal", "best effort", "best-effort",
         "réduite", "dégradée", "minimale"),
        QoSLevel.RELAXED,
    ),
    **dict.fromkeys(("standard", "normal", "nominal", "usual", "required", "normale", "requise"), QoSLevel.STANDARD),
}  # fmt: skip
_UTIL_LIMIT = re.compile(
    rf"\butili[sz]ation\s+(?:below|under|at most|<=|no more than|max(?:imum)?|sous|inférieure? à|en dessous de)\s+{_DIGITS}\s*%"
)
_NO_CAPACITY_REDUCTION = re.compile(
    r"\b(full capacity|all (?:carriers|cells|layers)(?: active| on)?|no (?:energy[- ]saving|sleep(?: mode)?s?|capacity reduction)|"
    r"disable (?:energy[- ]saving|sleep(?: mode)?s?)|without (?:energy[- ]saving|sleep(?: mode)?s?|capacity reduction)|"
    r"keep all (?:carriers|cells)|capacité maximale|sans mise en veille|pas de mise en veille)\b"
)

# --- unsafe / contradictory ----------------------------------------------------------
_UNSAFE = re.compile(
    r"\b(shut ?down|switch off|turn off|power off|power down|disable|deactivate|éteindre|couper|désactiver|arrêter)\s+"
    r"(?:the\s+|le\s+|la\s+|les\s+|l')?(?:whole\s+|entire\s+)?(base station|site|cell site|station|bts|gnodeb|gnb|"
    r"enodeb|radios?|antennas?|cells|network|antennes?)\b|"
    r"\b(bypass|ignore|override|skip|disable|deactivate|turn off|switch off|contourner|ignorer|désactiver)\s+(?:the\s+|la\s+|le\s+|les\s+)?(?:\w+\s+)?"
    r"(safety|validator|validation|limits?|constraints?|sécurité|contraintes?|limites?)\b"
)
_EXPORT = re.compile(r"\b(export|sell|inject|revendre|injecter)\b[^.;]{0,40}\b(grid|réseau)\b")
_DRAIN = re.compile(
    r"\b(fully discharge|discharge (?:the battery )?(?:fully|completely|to (?:empty|zero))|empty the battery|"
    r"drain the battery|use all (?:of )?the battery|vider la batterie|décharger complètement|décharge complète)\b"
)
_VAGUE_LEVEL = re.compile(
    r"\b(reasonably|fairly|quite|pretty|sufficiently|rather|enough|high|higher|full|topped up|healthy|"
    r"good level|assez|suffisamment|élevée?|haute?|pleine?|bon niveau)\b"
)


@dataclass
class _Window:
    start: int  # char span in the sentence
    end: int
    minutes: tuple[int, int] | None  # None = "same period" (resolved later)


def _num(token: str) -> float:
    token = token.strip()
    if token in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[token])
    return float(token.replace(",", "."))


def _clock(token: str, *, as_end: bool = False) -> int | None:
    t = token.strip()
    if t in ("noon", "midday", "midi"):
        return 720
    if t in ("midnight", "minuit"):
        return 1440 if as_end else 0
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2})|h(\d{2})|h)?\s*(am|pm)?", t)
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2) or m.group(3) or 0)
    if m.group(4):
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if m.group(4) == "pm" else 0)
    if minute >= 60 or hour > 24 or (hour == 24 and minute):
        return None
    return hour * 60 + minute


def _duration_minutes(value: str | None, unit: str) -> int:
    unit = unit.strip()
    if unit.startswith(("half", "demi")):
        return 30
    amount = _num(value) if value else 1.0
    return int(round(amount * (1 if unit.startswith("min") else 60)))


def _normalise(text: str) -> str:
    t = text.lower()
    for a, b in (("’", "'"), ("≥", ">="), ("≤", "<="), ("–", "-"), ("—", "-"), (" ", " "), (" ", " ")):
        t = t.replace(a, b)
    t = re.sub(r"\bmin\.\s", "min ", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[;!?\n]|\.(?=\s|$)", text) if s.strip()]


def _find_windows(sentence: str, issued: int, horizon: int) -> list[_Window]:
    """All window expressions in a sentence, without overlaps, in text order."""
    found: list[_Window] = []
    taken: list[tuple[int, int]] = []

    def free(span: tuple[int, int]) -> bool:
        return all(span[1] <= a or span[0] >= b for a, b in taken)

    def add(m: re.Match, minutes: tuple[int, int] | None) -> None:
        taken.append(m.span())
        found.append(_Window(m.start(), m.end(), minutes))

    for m in _W_BETWEEN.finditer(sentence):
        a, b = _clock(m.group(1)), _clock(m.group(2), as_end=True)
        if a is not None and b is not None and free(m.span()):
            add(m, (a, b if b > a else b + 1440))
    for m in _W_START_FOR.finditer(sentence):
        a = _clock(m.group(1))
        if a is not None and free(m.span()):
            add(m, (a, a + _duration_minutes(m.group(2), m.group(3))))
    for pattern in (_W_NEXT_EN, _W_NEXT_FR):
        for m in pattern.finditer(sentence):
            if free(m.span()):
                add(m, (issued, issued + _duration_minutes(m.group(1), m.group(2))))
    for m in _W_HALF_HOUR.finditer(sentence):
        if free(m.span()):
            add(m, (issued, issued + 30))
    for m in _W_FOR.finditer(sentence):
        if free(m.span()):
            add(m, (issued, issued + _duration_minutes(m.group(1), m.group(2))))
    for m in _W_UNTIL.finditer(sentence):
        b = _clock(m.group(1), as_end=True)
        if b is not None and free(m.span()):
            add(m, (issued, b))
    for m in _W_ONWARDS.finditer(sentence):
        a = _clock(m.group(1))
        if a is not None and free(m.span()):
            add(m, (a, horizon))
    for m in _W_DAY.finditer(sentence):
        if free(m.span()):
            add(m, (issued, horizon))
    for m in _W_SAME.finditer(sentence):
        if free(m.span()):
            add(m, None)
    return sorted(found, key=lambda w: w.start)


def _nearest_window(windows: list[_Window], start: int, end: int) -> _Window:
    def distance(w: _Window) -> float:
        return (w.start - end) if w.start >= end else (start - w.end) + 5  # slight preference for "after"

    return min(windows, key=distance)


def _bound_kind(sentence: str, pct_start: int) -> str:
    """'lower', 'upper' or 'none' depending on the closest comparator before the value."""
    before = sentence[max(0, pct_start - 60) : pct_start]
    neg = list(_LOWER_NEGATED.finditer(before))
    low = list(_LOWER_WORDS.finditer(before))
    up = list(_UPPER_WORDS.finditer(before))
    best = ("none", -1)
    for kind, matches in (("lower", neg), ("lower", low), ("upper", up)):
        if matches and matches[-1].end() > best[1]:
            best = (kind, matches[-1].end())
    if neg and best[0] == "upper" and neg[-1].end() >= best[1]:
        return "lower"
    return best[0]


def safety_screen(text: str) -> list[Issue]:
    """Deterministic screen applied to every instruction, whichever component interprets it."""
    t = _normalise(text)
    issues: list[Issue] = []
    if _UNSAFE.search(t):
        issues.append(Issue(code="unsupported_unsafe_request", severity="error",
                            message="the request would disable equipment or bypass safety checks; not permitted"))  # fmt: skip
    if _EXPORT.search(t):
        issues.append(Issue(code="unsupported_export", severity="error",
                            message="grid export is not supported by this site model"))  # fmt: skip
    if _BATTERY_POWER_SETPOINT.search(t):
        issues.append(Issue(code="unsupported_setpoint", severity="error",
                            message="direct battery power set-points are not accepted; express a reserve or a grid limit"))  # fmt: skip
    return issues


def parse_instruction(text: str, issued_at_minute: int, config: AppConfig) -> ParseResult:
    """Interpret a natural-language (or JSON) instruction without any LLM."""
    started = time.perf_counter()
    raw = text if isinstance(text, str) else repr(text)
    horizon = config.simulation.horizon_hours * 60

    def result(instr, issues):
        return build_result(instr, issues, method=METHOD, raw_input=raw, config=config, started=started)

    if not isinstance(text, str) or not text.strip():
        return result(None, [Issue(code="empty_instruction", severity="error", message="the instruction is empty")])
    if len(text) > config.agent.max_instruction_chars:
        return result(
            None,
            [Issue(code="instruction_too_long", severity="error",
                   message=f"instruction longer than {config.agent.max_instruction_chars} characters")],
        )  # fmt: skip
    if text.lstrip().startswith(("{", "[")):
        structured = parse_structured(text, issued_at_minute, config)
        return structured.model_copy(update={"method": "structured_json"})

    t = _normalise(text)
    letters = sum(c.isalpha() for c in t)
    if letters < 0.4 * len(t.replace(" ", "")) or not re.search(r"[a-zà-ÿ]{3,}", t):
        return result(
            None,
            [Issue(code="malformed_instruction", severity="error",
                   message="the instruction is not readable text; use the structured JSON interface")],
        )  # fmt: skip

    issues: list[Issue] = safety_screen(text)
    reserves: list[dict] = []
    grid_limits: list[dict] = []
    all_windows: list[tuple[int, int]] = []
    pending: list[tuple[str, float, int, list[_Window], int, int]] = []  # kind, value, sentence idx, windows, span
    sentence_windows: list[list[_Window]] = []
    last_concrete: tuple[int, int] | None = None
    vague_time = bool(_VAGUE_TIME.search(t))

    for si, sentence in enumerate(_sentences(t)):
        windows = _find_windows(sentence, issued_at_minute, horizon)
        for w in windows:  # resolve "same period" references in text order
            if w.minutes is None:
                w.minutes = last_concrete
            else:
                last_concrete = w.minutes
        windows = [w for w in windows if w.minutes is not None]
        sentence_windows.append(windows)
        all_windows.extend(w.minutes for w in windows)

        util = _UTIL_LIMIT.search(sentence)
        has_battery = bool(_BATTERY_CTX.search(sentence))
        has_grid = bool(_GRID_CTX.search(sentence))
        for m in _PCT.finditer(sentence):
            if util and util.start() <= m.start() <= util.end():
                continue
            value = _num(m.group(1))
            if has_battery:
                kind = _bound_kind(sentence, m.start())
                if kind == "upper":
                    issues.append(Issue(code="unsupported_upper_soc_bound", severity="error", field="reserves",
                                        message="only minimum battery reserves are supported, not upper SoC targets"))  # fmt: skip
                else:
                    pending.append(("reserve", value, si, windows, m.start(), m.end()))
            elif has_grid:
                issues.append(Issue(code="relative_grid_target", severity="clarification", field="grid_limits",
                                    message="please give the grid-import limit as an absolute power in kW"))  # fmt: skip
            else:
                issues.append(Issue(code="unrecognised_percentage", severity="clarification",
                                    message=f"the value {value:g}% could not be attributed to a supported directive"))  # fmt: skip
        if has_battery and not _PCT.search(sentence) and _VAGUE_LEVEL.search(sentence):
            issues.append(Issue(code="reserve_level_missing", severity="clarification", field="reserves",
                                message="which minimum battery charge (in %) should be maintained?"))  # fmt: skip
        for m in _KW.finditer(sentence):
            if _BATTERY_POWER_SETPOINT.search(sentence):
                continue
            if has_grid:
                before = sentence[max(0, m.start() - 40) : m.start()]
                if re.search(r"at least|minimum|au moins", before):
                    issues.append(Issue(code="unsupported_grid_minimum", severity="error", field="grid_limits",
                                        message="minimum grid-import requirements are not supported"))  # fmt: skip
                else:
                    pending.append(("grid", _num(m.group(1)), si, windows, m.start(), m.end()))
            else:
                issues.append(Issue(code="unrecognised_power_value", severity="clarification",
                                    message=f"the value {m.group(0)} could not be attributed to a supported directive"))  # fmt: skip
        z = _ZERO_IMPORT.search(sentence)
        if z and not _KW.search(sentence):
            pending.append(("grid", 0.0, si, windows, z.start(), z.end()))

    distinct_windows = sorted(set(all_windows))
    for kind, value, _si, windows, start, end in pending:
        if windows:
            window = _nearest_window(windows, start, end).minutes
        elif len(distinct_windows) == 1:
            window = distinct_windows[0]
        else:
            label = "reserve" if kind == "reserve" else "grid-import limit"
            msg = (
                f"over which time window should the {label} apply?"
                if not vague_time
                else f"the time window of the {label} is vague; please give explicit start and end times"
            )
            issues.append(Issue(code="window_missing", severity="clarification", field=kind, message=msg))
            continue
        start_min, end_min = window
        if kind == "reserve":
            reserves.append({"min_soc_pct": float(value), "start_minute": int(start_min), "end_minute": int(end_min)})
        else:
            grid_limits.append({"max_import_kw": float(value), "start_minute": int(start_min), "end_minute": int(end_min)})

    # Same window with different reserve levels is contradictory.
    seen: dict[tuple[int, int], float] = {}
    for r in reserves:
        key = (r["start_minute"], r["end_minute"])
        if key in seen and seen[key] != r["min_soc_pct"]:
            issues.append(Issue(code="conflicting_reserves", severity="clarification", field="reserves",
                                message="two different reserve levels are given for the same window"))  # fmt: skip
        seen[key] = r["min_soc_pct"]

    if _DRAIN.search(t):
        if any(p[0] == "reserve" for p in pending):
            issues.append(Issue(code="contradictory_instruction", severity="error",
                                message="the instruction asks both to keep a reserve and to fully discharge the battery"))  # fmt: skip
        else:
            issues.append(Issue(code="unsupported_deep_discharge", severity="error",
                                message="discharging below the battery's protection limits is not permitted"))  # fmt: skip

    mapping: dict = {"reserves": reserves, "grid_limits": grid_limits}
    objectives = set()
    if _OBJ_COST.search(t):
        objectives.add(Objective.MINIMIZE_COST)
    for m in _OBJ_GRID.finditer(t):
        if not re.match(r"\s*(?:to|at|below|under|à)\s+\d", t[m.end() : m.end() + 12]):
            objectives.add(Objective.MINIMIZE_GRID_ENERGY)
    if _OBJ_PEAK.search(t):
        objectives.add(Objective.MINIMIZE_PEAK_IMPORT)
    if len(objectives) > 1:
        issues.append(Issue(code="multiple_objectives", severity="clarification", field="objective",
                            message="several optimisation objectives were requested; which one has priority?"))  # fmt: skip
    elif objectives:
        mapping["objective"] = objectives.pop().value
    elif _VAGUE_OBJECTIVE.search(t):
        issues.append(Issue(code="vague_objective", severity="clarification", field="objective",
                            message="'saving energy' is ambiguous: minimise cost, grid energy, or peak import?"))  # fmt: skip

    levels = {_QOS_WORD_LEVEL[(m.group(1) or m.group(2)).strip()] for m in _QOS_LEVEL.finditer(t)}
    if _QOS_PRESERVE.search(t):
        levels.add(QoSLevel.STANDARD)
    if QoSLevel.STRICT in levels and QoSLevel.RELAXED in levels:
        issues.append(Issue(code="conflicting_qos", severity="clarification", field="qos_level",
                            message="both strict and relaxed quality of service were requested"))  # fmt: skip
    elif levels:
        # "preserve strict QoS" -> strict: an explicit level wins over a generic "preserve".
        for level in (QoSLevel.STRICT, QoSLevel.RELAXED, QoSLevel.STANDARD):
            if level in levels:
                mapping["qos_level"] = level.value
                break
    util = _UTIL_LIMIT.search(t)
    if util:
        mapping["max_utilization"] = _num(util.group(1)) / 100.0
    if _NO_CAPACITY_REDUCTION.search(t):
        mapping["allow_capacity_reduction"] = False

    recognised = reserves or grid_limits or ("objective" in mapping) or levels or util or (
        "allow_capacity_reduction" in mapping
    )
    if not recognised and not any(i.severity in ("error", "clarification") for i in issues):
        issues.append(Issue(code="no_directive", severity="clarification",
                            message="no supported directive (reserve, grid limit, objective, QoS) was recognised"))  # fmt: skip

    instruction, schema_issues = instruction_from_mapping(mapping, issued_at_minute)
    return result(instruction, issues + schema_issues)


def describe_supported_grammar() -> str:
    """Short human-readable summary, shown in the dashboard and in clarification messages."""
    return json.dumps(
        {
            "reserve": "keep at least 70% battery charge between 17:00 and 21:00",
            "grid_limit": "limit grid import to 2 kW from 18:00 to 20:00",
            "objective": "minimize electricity costs | minimize grid energy | minimize peak import",
            "qos": "strict / standard / relaxed quality of service; no capacity reduction",
            "relative_windows": "during the next two hours; for 90 minutes; until 18:00; today",
        },
        indent=2,
    )
