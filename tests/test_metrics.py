import math

from metrics import nasa_score, rmse


def test_nasa_score_is_zero_for_perfect_predictions():
    assert nasa_score([10, 50], [10, 50]) == 0


def test_nasa_score_penalises_late_more_than_early():
    late = nasa_score([50], [70])
    early = nasa_score([50], [30])
    assert math.isclose(late, math.exp(2) - 1)
    assert math.isclose(early, math.exp(20 / 13) - 1)
    assert late > early


def test_rmse():
    assert rmse([0, 0], [3, 4]) == math.sqrt(12.5)
