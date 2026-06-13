import numpy as np
import pytest

from strategy import MarkovChainStrategy


@pytest.fixture
def strategy():
    # _build_transition_matrix and _get_dynamic_states don't touch the queue or data handler.
    return MarkovChainStrategy(data_handler=None, events_queue=None)


def test_transition_matrix_rows_sum_to_one(strategy):
    states = [0, 1, 2, 0, 1, 2, 1, 0]
    matrix = strategy._build_transition_matrix(states)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_transition_matrix_counts(strategy):
    # Sequence 0,1,2,0,1,2 -> transitions: 0->1 (x2), 1->2 (x2), 2->0 (x1).
    matrix = strategy._build_transition_matrix([0, 1, 2, 0, 1, 2])
    assert matrix[0][1] == pytest.approx(1.0)  # state 0 always went to 1
    assert matrix[1][2] == pytest.approx(1.0)  # state 1 always went to 2
    assert matrix[2][0] == pytest.approx(1.0)  # state 2 always went to 0


def test_transition_matrix_uniform_when_too_short(strategy):
    matrix = strategy._build_transition_matrix([1])
    assert np.allclose(matrix, 1 / 3)


def test_dynamic_states_labels_extremes(strategy):
    # A sharp positive return after calm ones should be flagged Bullish (2);
    # a sharp negative one Bearish (0).
    returns = [0.001, 0.001, 0.001, 0.001, 0.05, -0.05]
    states = strategy._get_dynamic_states(returns, z_window=3)
    assert states[-2] == 2
    assert states[-1] == 0
