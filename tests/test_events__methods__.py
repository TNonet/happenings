import datetime
import operator

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, assume
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from evendar.event import MONTHS

JAN_1_2020 = datetime.datetime.strptime("2020-01-01", "%Y-%M-%d")
JAN_1_2021 = datetime.datetime.strptime("2021-01-01", "%Y-%M-%d") - datetime.timedelta(days=1)
DATES_2020 = pd.date_range(start=JAN_1_2020, end=JAN_1_2021, freq="D")

JAN_EVENT = MONTHS['January']
JAN_EVENT_2020_VALUES = np.zeros_like(DATES_2020).astype(bool)
JAN_EVENT_2020_VALUES[0:31] = True

BINARY_COMPARISON_OPS = (operator.lt,
                         operator.le,
                         operator.eq,
                         operator.ne,
                         operator.ge,
                         operator.gt)

BINARY_BITWISE_OPS = (operator.and_,
                      operator.xor,
                      operator.or_)


BINARY_MATH_OPS_ZERO_ISSUES = (operator.mod,
                               operator.floordiv,
                               operator.truediv)


BINARY_NUM_OPS = (operator.add,
                  operator.floordiv,
                  operator.lshift,
                  operator.mod,
                  operator.mul,
                  operator.matmul,
                  operator.pow,
                  operator.rshift,
                  operator.sub,
                  operator.truediv)


def test_functional_event():
    np.testing.assert_array_equal(JAN_EVENT.sample_from(DATES_2020),
                                  JAN_EVENT_2020_VALUES)


@pytest.mark.parametrize('op', BINARY_COMPARISON_OPS)
@given(a=st.floats(allow_nan=False))
def test_cmp_scalar(a, op):

    np.testing.assert_array_equal(op(a, JAN_EVENT).sample_from(DATES_2020),
                                  op(a, JAN_EVENT_2020_VALUES))

    np.testing.assert_array_equal(op(JAN_EVENT, a).sample_from(DATES_2020),
                                  op(JAN_EVENT_2020_VALUES, a))


@pytest.mark.parametrize('op', BINARY_COMPARISON_OPS)
@given(a=npst.arrays(shape=len(JAN_EVENT_2020_VALUES), dtype=st.sampled_from([np.int, np.float])))
def test_cmp_array(a, op):

    # Not supported as the cmp operators are handled by numpy comparing each item individually.
    # np.testing.assert_array_equal(op(a, JAN_EVENT).sample_from(DATES_2020),
    #                               op(a, JAN_EVENT_2020_VALUES))

    np.testing.assert_array_equal(op(JAN_EVENT, a).sample_from(DATES_2020),
                                  op(JAN_EVENT_2020_VALUES, a))


@pytest.mark.parametrize('op', BINARY_COMPARISON_OPS)
def test_cmp_event(op):
    np.testing.assert_array_equal(op(JAN_EVENT, JAN_EVENT).sample_from(DATES_2020),
                                  op(JAN_EVENT_2020_VALUES, JAN_EVENT_2020_VALUES))


@pytest.mark.parametrize('op', BINARY_BITWISE_OPS)
@given(a=st.booleans())
def test_bitwise_scalar(a, op):

    np.testing.assert_array_equal(op(a, JAN_EVENT).sample_from(DATES_2020),
                                  op(a, JAN_EVENT_2020_VALUES))

    np.testing.assert_array_equal(op(JAN_EVENT, a).sample_from(DATES_2020),
                                  op(JAN_EVENT_2020_VALUES, a))


@pytest.mark.parametrize('op', BINARY_BITWISE_OPS)
@given(a=npst.arrays(shape=len(JAN_EVENT_2020_VALUES), dtype=np.bool))
def test_bitwise_array(a, op):

    # Not supported as the bitwise operators are handled by numpy comparing each item individually.
    # np.testing.assert_array_equal(op(a, JAN_EVENT).sample_from(DATES_2020),
    #                               op(a, JAN_EVENT_2020_VALUES))

    np.testing.assert_array_equal(op(JAN_EVENT, a).sample_from(DATES_2020),
                                  op(JAN_EVENT_2020_VALUES, a))


@pytest.mark.parametrize('op', BINARY_BITWISE_OPS)
@given(a=npst.arrays(shape=len(JAN_EVENT_2020_VALUES), dtype=np.bool))
def test_bitwise_event(a, op):
    np.testing.assert_array_equal(op(JAN_EVENT, JAN_EVENT).sample_from(DATES_2020),
                                  op(JAN_EVENT_2020_VALUES, JAN_EVENT_2020_VALUES))


@pytest.mark.parametrize('op', BINARY_MATH_OPS_ZERO_ISSUES)
@given(a=st.floats(allow_nan=False, allow_infinity=False))
def test_div_scalar(a, op):
    np.testing.assert_array_equal(op(a, 1+JAN_EVENT).sample_from(DATES_2020),
                                  op(a, 1+JAN_EVENT_2020_VALUES))

    assume(a != 0)
    np.testing.assert_array_equal(op(JAN_EVENT, a).sample_from(DATES_2020),
                                  op(JAN_EVENT_2020_VALUES, a))


@pytest.mark.parametrize('op', BINARY_MATH_OPS_ZERO_ISSUES)
@given(a=npst.arrays(shape=len(JAN_EVENT_2020_VALUES),
                     dtype=np.float,
                     elements=st.floats(allow_infinity=False, allow_nan=False)))
def test_div_array(a, op):
    # Not supported as the cmp operators are handled by numpy comparing each item individually.
    # np.testing.assert_array_equal(op(a, JAN_EVENT).sample_from(DATES_2020),
    #                               op(a, JAN_EVENT_2020_VALUES))

    a[a == 0] = 1

    np.testing.assert_array_equal(op(JAN_EVENT, a).sample_from(DATES_2020),
                                  op(JAN_EVENT_2020_VALUES, a))


@pytest.mark.parametrize('op', BINARY_MATH_OPS_ZERO_ISSUES)
def test_div_event(op):
    np.testing.assert_array_equal(op(JAN_EVENT, JAN_EVENT+1).sample_from(DATES_2020),
                                  op(JAN_EVENT_2020_VALUES, JAN_EVENT_2020_VALUES+1))
