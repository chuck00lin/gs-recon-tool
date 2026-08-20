"""How the sharpness filter divides a capture into groups.

Kept free of cv2 and Qt on purpose: the GUI, the CLI and the filter tool all
describe the *same* selection to the user, and the GUI has to do it live --
before a single frame exists on disk.

The user-facing number is the group size: how many consecutive frames compete
against each other, with the sharpest of each group surviving. Everything else
(group count, how many survive per group, the resulting spacing) follows from
it plus the totals, so it is derived here rather than configured.
"""

from __future__ import annotations

from typing import Optional

DEFAULT_GROUP_SIZE = 10
MIN_GROUP_SIZE = 2


def group_count_for(total: int, group_size: Optional[int]) -> int:
    """How many groups ``total`` frames split into at this group size."""
    if total <= 0:
        return 1
    size = max(MIN_GROUP_SIZE, int(group_size or DEFAULT_GROUP_SIZE))
    return max(1, min(total, round(total / size)))


def kept_per_group(total: int, target_count: int, group_size: Optional[int]) -> float:
    return target_count / group_count_for(total, group_size)


def _round(value: float) -> str:
    return f"{value:.0f}" if abs(value - round(value)) < 0.05 else f"{value:.1f}"


def describe_selection(
    total: Optional[int],
    target_count: Optional[int],
    group_size: Optional[int],
    *,
    ratio: Optional[float] = None,
) -> str:
    """One line explaining what the filter will actually do.

    ``total`` may be an estimate (the GUI predicts it from clip length) or None
    when nothing is known yet, in which case only the ratio is described.
    """
    size = max(MIN_GROUP_SIZE, int(group_size or DEFAULT_GROUP_SIZE))
    if not total or not target_count:
        if ratio:
            return (
                f"Groups of {size} frames — the sharpest ~{_round(size * ratio)} "
                f"of every {size} survive ({ratio:.0%} kept)."
            )
        return f"Groups of {size} consecutive frames; the sharpest of each survive."

    groups = group_count_for(total, group_size)
    per_group = target_count / groups
    spacing = total / target_count
    # One or two survivors per group is the sweet spot: enough choice for the
    # sharpness test to matter, not enough for the winners to clump together.
    caveat = ""
    if per_group < 1:
        caveat = (
            f"  ⚠ finer than the keep rate, so some groups keep nothing — "
            f"try {max(MIN_GROUP_SIZE, round(spacing)):,} for one per group."
        )
    elif per_group > 3:
        caveat = (
            f"  ⚠ big groups let the survivors clump — "
            f"try {max(MIN_GROUP_SIZE, round(2 * spacing)):,} for ~2 per group."
        )
    return (
        f"{total:,} frames → {groups:,} groups of ~{size} → keep the sharpest "
        f"~{_round(per_group)} per group ≈ {target_count:,} frames "
        f"(1 in {spacing:.1f}).{caveat}"
    )
