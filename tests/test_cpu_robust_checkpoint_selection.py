import pytest

from src.rl.checkpoint_selection import (
    CPU_MAXIMIN_SELECTION_RULE,
    cpu_validation_ranges,
    robust_checkpoint_score,
    should_replace_robust_checkpoint,
    validation_grid_digest,
)


def test_cpu_grid_contract_and_digest_are_deterministic():
    evaluation = {
        "checkpoint_selection": CPU_MAXIMIN_SELECTION_RULE,
        "cpu_capacity_ranges_cycles_per_s": [[1.0, 2.0], [2.0, 3.0]],
    }
    ranges = cpu_validation_ranges(evaluation)
    assert ranges == ((1.0, 2.0), (2.0, 3.0))
    assert validation_grid_digest(ranges) == validation_grid_digest(ranges)


def test_maximin_score_uses_cell_means_and_macro_tie_break():
    first = robust_checkpoint_score([1.0, 3.0, 8.0, 10.0], cell_count=2, seeds_per_cell=2)
    assert first.minimum_cell_social_welfare == 2.0
    assert first.macro_mean_social_welfare == 5.5
    assert first.cell_social_welfare_means == (2.0, 9.0)
    worse_minimum = robust_checkpoint_score(
        [1.0, 2.0, 20.0, 20.0], cell_count=2, seeds_per_cell=2
    )
    assert not should_replace_robust_checkpoint(first, worse_minimum)
    better_macro = robust_checkpoint_score(
        [1.0, 3.0, 10.0, 12.0], cell_count=2, seeds_per_cell=2
    )
    assert should_replace_robust_checkpoint(first, better_macro)
    assert not should_replace_robust_checkpoint(better_macro, better_macro)


def test_signed_welfare_and_invalid_shapes_are_handled():
    score = robust_checkpoint_score([-4.0, -2.0, -1.0, 1.0], cell_count=2, seeds_per_cell=2)
    assert score.minimum_cell_social_welfare == -3.0
    with pytest.raises(ValueError, match="declared grid"):
        robust_checkpoint_score([1.0], cell_count=2, seeds_per_cell=1)
    with pytest.raises(ValueError):
        cpu_validation_ranges({
            "checkpoint_selection": CPU_MAXIMIN_SELECTION_RULE,
        })
