"""Protocol post-processing of a segmentation: "plane shift + exact protocol"."""

import json
from pathlib import Path

import numpy as np

from ..protocol import PROTOCOL_LANDMARKS, PROTOCOL_REGIONS, parse_pql
from ..protocol.geometry import allowed, clause_depths, clause_groups, protocol_planes, rule_depth

__all__ = ["PlaneShift", "DEFAULT_PQL", "SECTION_PLANES", "to_uint8"]

DEFAULT_PQL = str(Path(__file__).resolve().parents[1] / "protocol" / "hoa_protocol.pql")

#: the planes put on the section of their landmark in the papers: the coronal planes that end a
#: structure or a contact zone (LM3-LM8, LM10), chosen on the validation subjects
SECTION_PLANES = (
    "put_nacc_contact_ant_L",
    "put_nacc_contact_ant_R",
    "put_nacc_contact_post_L",
    "put_nacc_contact_post_R",
    "nacc_end_L",
    "nacc_end_R",
    "ac_coronal",
)


def to_uint8(probabilities: np.ndarray) -> np.ndarray:
    """(C, Z, Y, X) probabilities as 8-bit values; uint8 input is returned as is.

    Values outside [0, 1] are clipped, not wrapped (1.2 would become 50). asarray rather than
    astype avoids copying a float32 input, 1.2 GB at full resolution.
    """
    if probabilities.dtype == np.uint8:
        return probabilities
    values = np.asarray(probabilities, dtype=np.float32)
    return np.round(np.clip(values, 0.0, 1.0) * 255).astype(np.uint8)


def shift_grid(low: float, high: float, step: float) -> np.ndarray:
    """Candidate shifts from low to high every `step` voxels, always including 0."""
    if high <= low:
        return np.array([0.0])
    return np.unique(np.concatenate([np.arange(low, high + 1e-9, step), [0.0]]))


class PlaneShift:
    def __init__(
        self,
        shift_range: dict[str, tuple[float, float]],
        tolerance: dict[str, float] | None = None,
        step: float = 0.1,
        margin: float = 2.0,
        pql: str = DEFAULT_PQL,
        fitted_on: dict | None = None,
        section_planes: list[str] | None = None,
    ):
        self.shift_range = {k: (float(lo), float(hi)) for k, (lo, hi) in shift_range.items()}
        self.tolerance = {k: float(v) for k, v in (tolerance or {}).items()}
        self.step, self.margin, self.pql = float(step), float(margin), pql
        self.fitted_on = fitted_on or {}
        self.spec = parse_pql(pql, PROTOCOL_LANDMARKS, PROTOCOL_REGIONS)
        self.section_planes = list(section_planes or [])
        one = {p.plane_id for p in self.spec.planes if len(p.landmark_ids) == 1}
        bad = set(self.section_planes) - one
        if bad:
            raise ValueError(f"section_planes must be planes through one landmark: {sorted(bad)}")
        used = {c.plane_id for r in self.spec.rules for g in clause_groups(r) for c in g}
        self.planes = [p.plane_id for p in self.spec.planes if p.plane_id in used]
        # per plane: the rules that read it, and the rest (unchanged while it slides)
        self.rules_of = {}
        for pid in self.planes:
            uses = [
                any(c.plane_id == pid for g in clause_groups(r) for c in g) for r in self.spec.rules
            ]
            self.rules_of[pid] = (
                [r for r, u in zip(self.spec.rules, uses) if u],
                [r for r, u in zip(self.spec.rules, uses) if not u],
            )

    def __call__(
        self,
        probabilities: np.ndarray,
        landmarks: np.ndarray,
        segmentation: np.ndarray | None = None,
        planes: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Post-processed segmentation and the shift kept for each plane (voxels).

        probabilities: (C, Z, Y, X), float in [0, 1] or uint8; landmarks: (K, 3) ZYX voxels;
        segmentation: the network's labels, recomputed from the 8-bit probabilities if omitted.
        planes: the geometry to slide and enforce, instead of the planes `landmarks` would
        build. The caller owns the unit then -- these, `shift_range`, `step`, `margin` and
        `tolerance` must all be in the same one -- which is how the same enforcement runs on a
        scan's own grid instead of the reference one.
        """
        prob = to_uint8(probabilities)
        raw = prob.argmax(0) if segmentation is None else segmentation
        idx = tuple(np.argwhere(prob[1:].max(0) > 0).T)
        vox = np.stack(idx, axis=1).astype(np.float64)
        prob_c = prob[(slice(None), *idx)]
        placed, shifts = self._fit(prob_c, raw[idx], vox, landmarks, planes)
        out = raw.copy()
        out[idx] = self._constrained(prob_c.astype(np.float32), vox, placed, self.spec.rules)
        return out, shifts

    def violations(
        self,
        labels: np.ndarray,
        points: np.ndarray,
        planes,
        tolerance: dict[str, float] | None = None,
    ) -> dict[str, int]:
        """{region: how many of its voxels its own rule forbids}, at `points`.

        The check the rules themselves define, run on whatever grid the labels are on. Same
        `points`, `planes` and `tolerance` must share one unit.
        """
        tolerance = self.tolerance if tolerance is None else tolerance
        labels, points = np.asarray(labels), np.asarray(points, dtype=np.float64)
        out = {}
        for rule in self.spec.rules:
            sel = np.flatnonzero(labels == rule.region_class)
            out[rule.region_name] = (
                0
                if not len(sel)
                else int((rule_depth(points[sel], planes, rule, tolerance) > 0).sum())
            )
        return out

    def _constrained(self, prob: np.ndarray, vox: np.ndarray, planes, rules) -> np.ndarray:
        """Most probable class each voxel may take under `rules` (prob: (C, N), modified).

        Where no allowed class has non-zero probability (8-bit rounding zeroes the others of a
        confident network), the voxel takes the class named by `reassign_to` for the clause it
        violates, if allowed there, else background; an argmax over ties at 0 would be
        background by index alone. The cost of a shift is unchanged (both have p 0).
        """
        top = prob.argmax(0)
        for rule in rules:
            prob[rule.region_class] *= allowed(vox, planes, rule, self.tolerance)
        out = prob.argmax(0)
        empty = prob.max(0) == 0
        if empty.any():
            out[empty] = self._reassign(top[empty], vox[empty], planes, rules, self.tolerance)
        return out

    def _reassign(
        self, cls: np.ndarray, vox: np.ndarray, planes, rules, tolerance: dict[str, float]
    ) -> np.ndarray:
        """For voxels of class `cls` that their rule forbids: the target of the clause they
        violate (worst clause of the least-violated OR-group) if allowed there, else 0.

        `tolerance` is a parameter because a caller may work in another unit than the fit.
        """
        out = np.zeros(len(cls), dtype=cls.dtype)
        by_class = {r.region_class: r for r in rules}
        by_name = {r.region_name: r for r in self.spec.rules}
        for c in np.unique(cls):
            rule = by_class.get(int(c))
            if rule is None:
                continue
            sel = np.flatnonzero(cls == c)
            groups = clause_groups(rule)
            depths = clause_depths(vox[sel], planes, rule, tolerance)
            g = np.argmin(np.stack([d.max(0) for d in depths]), axis=0)
            for gi, group in enumerate(groups):
                k = np.flatnonzero(g == gi)
                worst = depths[gi][:, k].argmax(0)
                for ci, clause in enumerate(group):
                    target = by_name.get(clause.reassign_to) if clause.reassign_to else None
                    if target is None:
                        continue
                    i = sel[k[worst == ci]]
                    ok = allowed(vox[i], planes, target, tolerance)
                    out[i[ok]] = target.region_class
        return out

    def _fit(
        self,
        prob: np.ndarray,
        old: np.ndarray,
        vox: np.ndarray,
        landmarks,
        planes: dict | None = None,
    ):
        planes = dict(planes) if planes is not None else protocol_planes(self.spec, landmarks)
        shifts = {}

        # section planes are placed before the greedy loop: the planes after them in file
        # order are optimised against their final position
        for pid in self.planes:
            if pid not in self.section_planes:
                continue
            centroid, normal = planes[pid]
            axis = int(np.argmax(np.abs(normal)))
            c = centroid.copy()
            c[axis] = np.round(c[axis])
            shifts[pid] = float((c - centroid) @ normal)  # from the predicted plane
            planes[pid] = (c, normal)

        # sliding planes, each against the positions already settled
        for pid in self.planes:
            if pid in self.section_planes:
                continue
            low, high = self.shift_range.get(pid, (0.0, 0.0))
            centroid, normal = planes[pid]
            signed = (vox - centroid) @ normal
            near = (signed >= low - self.margin) & (signed <= high + self.margin)
            p = prob[:, near].astype(np.float32) / 255.0
            best = self._best_shift(
                pid, shift_grid(low, high, self.step), planes, p, old[near], vox[near]
            )
            planes[pid] = (centroid + best * normal, normal)
            shifts[pid] = best
        return planes, shifts

    def _best_shift(self, pid: str, grid: np.ndarray, planes, p, old, vox) -> float:
        """The shift with the smallest destroyed confidence; ties go to the smallest |shift|."""
        moving, fixed = self.rules_of[pid]
        base = p.copy()
        self._constrained(base, vox, planes, fixed)  # rules that do not see this plane
        cols = np.arange(len(old))
        centroid, normal = planes[pid]
        best_cost, best = np.inf, 0.0
        for shift in grid:
            trial = {**planes, pid: (centroid + shift * normal, normal)}
            new = self._constrained(base.copy(), vox, trial, moving)
            ch = new != old
            cost = float((p[old[ch], cols[ch]] - p[new[ch], cols[ch]]).sum())
            tie = abs(cost - best_cost) <= 1e-9 and abs(shift) < abs(best)
            if cost < best_cost - 1e-9 or tie:
                best_cost, best = cost, float(shift)
        return best

    def scaled(self, factor: float) -> "PlaneShift":
        """The same fit with every length multiplied by `factor`.

        `shift_range`, `step`, `margin` and `tolerance` are all reference voxels, because that
        is the grid the ranges were fitted on. Searching and enforcing on a scan's own grid
        means working in millimetres there, so the lengths have to come along: factor is the
        reference spacing, 0.7 mm. Nothing is refitted -- the same interval, read in the unit
        the geometry is expressed in.
        """
        factor = float(factor)
        return PlaneShift(
            shift_range={k: (lo * factor, hi * factor) for k, (lo, hi) in self.shift_range.items()},
            tolerance={k: v * factor for k, v in self.tolerance.items()},
            step=self.step * factor,
            margin=self.margin * factor,
            pql=self.pql,
            fitted_on={**self.fitted_on, "scaled_by": factor},
            section_planes=self.section_planes,
        )

    def strict(self) -> "PlaneShift":
        """The same fit with every per-plane tolerance removed.

        The fitted tolerances are granted to the planes the *ground truth itself* crosses in at
        least half the training subjects -- put_nacc_border_L/R in 77 and 78 of 80, midline_ac
        in 47 -- so that enforcing them would correct the reference rather than the prediction.
        Dropping them makes the output obey every rule with no slack, which is a different
        claim and a different number: the border between the putamen and the nucleus accumbens
        then follows the plane exactly, whatever the annotator did there.
        """
        clone = PlaneShift(
            shift_range=self.shift_range,
            tolerance={},
            step=self.step,
            margin=self.margin,
            pql=self.pql,
            fitted_on={**self.fitted_on, "tolerance_dropped": True},
            section_planes=self.section_planes,
        )
        return clone

    def to_dict(self) -> dict:
        return {
            "method": "plane shift + protocol",
            "pql": Path(self.pql).name,
            "step": self.step,
            "margin": self.margin,
            "shift_range": self.shift_range,
            "tolerance": self.tolerance,
            "section_planes": self.section_planes,
            "fitted_on": self.fitted_on,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "PlaneShift":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        pql = str(Path(DEFAULT_PQL).with_name(d["pql"]))
        return cls(
            d["shift_range"],
            d.get("tolerance"),
            d["step"],
            d["margin"],
            pql,
            d.get("fitted_on"),
            d.get("section_planes"),
        )
