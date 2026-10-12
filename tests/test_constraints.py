import unittest

from src.models.constraints import latency_feasible, total_latency


class TestConstraints(unittest.TestCase):
    def test_total_latency_is_additive(self) -> None:
        self.assertEqual(total_latency(1.0, 2.0, 3.0), 6.0)

    def test_exact_deadline_is_feasible(self) -> None:
        self.assertTrue(latency_feasible(1.0, 2.0, 3.0, 6.0))

    def test_over_deadline_is_infeasible(self) -> None:
        self.assertFalse(latency_feasible(1.0, 2.0, 3.1, 6.0))

    def test_negative_deadline_raises(self) -> None:
        with self.assertRaises(ValueError):
            latency_feasible(1.0, 2.0, 3.0, -1.0)


if __name__ == "__main__":
    unittest.main()
