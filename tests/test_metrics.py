import numpy as np
import pytest

from demandcast.evaluation.metrics import all_metrics, mape, rmse, wape


def test_perfect_predictions_are_zero_error():
    y_true = [10, 20, 30]
    assert mape(y_true, y_true) == 0
    assert rmse(y_true, y_true) == 0
    assert wape(y_true, y_true) == 0


def test_rmse_matches_hand_computed_value():
    y_true = [10, 20]
    y_pred = [12, 16]
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt((4 + 16) / 2))


def test_mape_ignores_zero_actuals():
    y_true = [0, 10, 20]
    y_pred = [5, 10, 22]
    # only the non-zero rows (10 vs 10, 20 vs 22) should count
    assert mape(y_true, y_pred) == pytest.approx((0 + 0.1) / 2 * 100)


def test_mape_raises_when_all_actuals_are_zero():
    with pytest.raises(ValueError):
        mape([0, 0], [1, 2])


def test_wape_weights_by_volume_not_by_row_count():
    y_true = [1, 100]
    y_pred = [2, 90]
    # errors: 1 and 10, total actual: 101 -> wape = 11/101 * 100
    assert wape(y_true, y_pred) == pytest.approx(11 / 101 * 100)


def test_all_metrics_returns_expected_keys():
    result = all_metrics([10, 20], [10, 20])
    assert set(result) == {"mape", "rmse", "wape"}
