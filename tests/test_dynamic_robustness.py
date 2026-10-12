from __future__ import annotations

from scripts.run_dynamic_robustness import build_cells


def test_dynamic_stress_design_covers_all_configured_dimensions():
    cells = build_cells()
    assert len(cells) == 15
    assert {cell.experiment for cell in cells} == {
        "occlusion", "path_loss", "rsu_bandwidth"
    }
    assert sum(cell.experiment == "occlusion" for cell in cells) == 2
    assert sum(cell.experiment == "path_loss" for cell in cells) == 7
    assert sum(cell.experiment == "rsu_bandwidth" for cell in cells) == 6


def test_stress_cells_are_shape_safe():
    unsafe = {"num_vehicles", "urgency_profiles"}
    assert all(not (unsafe & dict(cell.overrides).keys()) for cell in build_cells())
