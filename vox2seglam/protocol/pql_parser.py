from __future__ import annotations

import re
from dataclasses import dataclass, field

import yaml

from .constants import CONSTRAINT_MAP, PLANE_CONSTRUCTORS

__all__ = [
    "PlaneSpec",
    "RuleSpec",
    "Clause",
    "ProtocolSpec",
    "parse_pql",
]

_ASSIGN_RE = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(.+)$")
_CALL_RE = re.compile(r"^([A-Za-z_]\w*)\s*\((.*)\)$")


@dataclass
class PlaneSpec:
    plane_id: str
    plane_type: int
    landmark_ids: list[int]


@dataclass
class Clause:
    plane_id: str
    sign: int
    reassign_to: str | None = None
    enabled: bool = True
    strict: bool = False  # half-space excludes its plane (terminal boundary)


@dataclass
class RuleSpec:
    region_name: str
    region_class: int
    clauses: list[Clause] = field(default_factory=list)
    enabled: bool = True
    reassign_to: str | None = None
    or_groups: list[list[Clause]] = field(default_factory=list)


@dataclass
class ProtocolSpec:
    planes: list[PlaneSpec]
    rules: list[RuleSpec]

    def plane_by_id(self, pid: str) -> PlaneSpec | None:
        for p in self.planes:
            if p.plane_id == pid:
                return p
        return None


# {short name: id} from a vocabulary file (protocol/data/*.yaml): `key` holds a list
# of {id, short, name} entries
def _load_yaml_map(path: str, key: str) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        entries = (yaml.safe_load(f) or {}).get(key) or []
    result: dict[str, int] = {}
    for e in entries:
        short, idx = str(e["short"]), int(e["id"])
        if short in result:
            raise ValueError(f"{path}: duplicate {key} entry {short!r}")
        result[short] = idx
    return result


# removes one pair of parentheses only when it encloses the whole expression, so
# "(a) and (b)" is left intact
def _strip_outer_parens(s: str) -> str:
    if not (s.startswith("(") and s.endswith(")")):
        return s
    depth = 0
    for i, ch in enumerate(s):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and i < len(s) - 1:
            return s
    return s[1:-1].strip()


def _parse_call(text: str) -> tuple[str, list[str]]:
    m = _CALL_RE.match(text.strip())
    if m is None:
        raise ValueError(f"Expected fn(args), got: {text!r}")
    fn = m.group(1).strip()
    args = [a.strip() for a in m.group(2).split(",") if a.strip()]
    return fn, args


def parse_pql(
    pql_path: str, lm_yaml: str | None = None, reg_yaml: str | None = None
) -> ProtocolSpec:
    from .constants import PROTOCOL_LANDMARKS, PROTOCOL_REGIONS

    lm_yaml = lm_yaml or PROTOCOL_LANDMARKS
    reg_yaml = reg_yaml or PROTOCOL_REGIONS
    lm_map = _load_yaml_map(lm_yaml, "landmarks")
    reg_map = _load_yaml_map(reg_yaml, "regions")

    named_planes: dict[str, PlaneSpec] = {}
    unique_implicit: dict[tuple[int, int], PlaneSpec] = {}
    all_planes: list[PlaneSpec] = []
    rules: list[RuleSpec] = []

    def _get_or_add_implicit(plane_type: int, lm_id: int) -> PlaneSpec:
        key = (plane_type, lm_id)
        if key in unique_implicit:
            return unique_implicit[key]
        short = {v: k for k, v in lm_map.items()}.get(lm_id, f"LM{lm_id}")
        suffix = {0: "axial", 1: "coronal", 2: "sagittal"}[plane_type]
        spec = PlaneSpec(
            plane_id=f"{short}_{suffix}",
            plane_type=plane_type,
            landmark_ids=[lm_id],
        )
        unique_implicit[key] = spec
        all_planes.append(spec)
        return spec

    def _parse_and_group(group_str: str, lineno: int) -> tuple[list[Clause], str | None]:
        s = _strip_outer_parens(group_str.strip())
        where = f"{pql_path}:{lineno}"

        clauses: list[Clause] = []
        rule_reassign: str | None = None

        for clause_str in [_strip_outer_parens(c.strip()) for c in s.split(" and ") if c.strip()]:
            try:
                fn_c, args_c = _parse_call(clause_str)
            except ValueError:
                raise SyntaxError(f"{where}: invalid clause {clause_str!r}") from None

            if fn_c == "reassign_to":
                if len(args_c) != 1:
                    raise ValueError(f"{where}: reassign_to takes 1 arg, got {len(args_c)}")
                target = args_c[0]
                if target not in reg_map:
                    raise KeyError(f"{pql_path}:{lineno}: unknown region {target!r}")
                rule_reassign = target
                continue

            if fn_c not in CONSTRAINT_MAP:
                raise SyntaxError(
                    f"{where}: unknown constraint {fn_c!r}; expected one "
                    f"of {sorted(CONSTRAINT_MAP)}"
                )
            if len(args_c) not in (1, 2):
                raise ValueError(f"{where}: {fn_c} takes 1 or 2 args, got {len(args_c)}")

            expected_type, sign = CONSTRAINT_MAP[fn_c]
            arg = args_c[0]

            clause_reassign_to = None
            if len(args_c) == 2:
                clause_reassign_to = args_c[1]
                if clause_reassign_to not in reg_map:
                    raise KeyError(f"{pql_path}:{lineno}: unknown region {clause_reassign_to!r}")

            if arg in named_planes:
                ref_plane = named_planes[arg]
                if ref_plane.plane_type != expected_type:
                    raise ValueError(
                        f"{pql_path}:{lineno}: {fn_c!r} expects "
                        f"plane type {expected_type}, got {ref_plane.plane_type}"
                    )
                clauses.append(
                    Clause(
                        plane_id=ref_plane.plane_id,
                        sign=sign,
                        reassign_to=clause_reassign_to,
                        strict=fn_c.startswith("strictly_"),
                    )
                )
            elif arg in lm_map:
                lm_id = lm_map[arg]
                ref_plane = _get_or_add_implicit(expected_type, lm_id)
                clauses.append(
                    Clause(
                        plane_id=ref_plane.plane_id,
                        sign=sign,
                        reassign_to=clause_reassign_to,
                        strict=fn_c.startswith("strictly_"),
                    )
                )
            else:
                raise KeyError(f"{pql_path}:{lineno}: unknown reference {arg!r}")

        return clauses, rule_reassign

    with open(pql_path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            m = _ASSIGN_RE.match(line)
            if not m:
                raise SyntaxError(f"{pql_path}:{lineno}: expected 'NAME = expr', got {line!r}")

            lhs = m.group(1).strip()
            rhs = m.group(2).strip()

            try:
                fn, args = _parse_call(rhs)
            except ValueError:
                fn, args = None, []

            if fn in PLANE_CONSTRUCTORS:
                # one landmark: plane normal to the scanner axis of its type; two: the
                # plane through both landmarks
                if len(args) not in (1, 2):
                    raise ValueError(
                        f"{pql_path}:{lineno}: {fn} takes 1 or 2 landmarks, got {len(args)}"
                    )
                unknown = [a for a in args if a not in lm_map]
                if unknown:
                    raise KeyError(f"{pql_path}:{lineno}: unknown landmark(s) {unknown}")
                if lhs in named_planes or lhs in reg_map:
                    raise ValueError(f"{pql_path}:{lineno}: {lhs!r} is already defined")
                lm_ids = [lm_map[a] for a in args]
                spec = PlaneSpec(
                    plane_id=lhs,
                    plane_type=PLANE_CONSTRUCTORS[fn],
                    landmark_ids=lm_ids,
                )
                named_planes[lhs] = spec
                all_planes.append(spec)
                continue

            if lhs not in reg_map:
                hint = (
                    f" ({fn!r} is not a plane constructor: {sorted(PLANE_CONSTRUCTORS)})"
                    if fn is not None and fn not in CONSTRAINT_MAP
                    else ""
                )
                raise KeyError(f"{pql_path}:{lineno}: unknown region {lhs!r}{hint}")
            if any(r.region_name == lhs for r in rules):
                raise ValueError(f"{pql_path}:{lineno}: a rule for {lhs!r} is already defined")
            region_class = reg_map[lhs]
            rule = RuleSpec(region_name=lhs, region_class=region_class)

            or_parts = [p.strip() for p in rhs.split(" or ") if p.strip()]
            primary_clauses, rt = _parse_and_group(or_parts[0], lineno)
            rule.clauses = primary_clauses
            if rt:
                rule.reassign_to = rt

            for part in or_parts[1:]:
                or_clauses, rt = _parse_and_group(part, lineno)
                rule.or_groups.append(or_clauses)
                if rt and rule.reassign_to is None:
                    rule.reassign_to = rt

            rules.append(rule)

    return ProtocolSpec(planes=all_planes, rules=rules)
