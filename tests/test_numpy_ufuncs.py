import numpy as np
import pytest
from hypothesis import given

from happenings import MONTH, DAY_OF_MONTH
from tests.utils.utils import date_range_indexes

NUMPY_UFUNCS = [
    np.add,
    np.subtract,
    np.multiply,
    np.matmul,
    np.divide,
    np.logaddexp,
    np.logaddexp2,
    np.true_divide,
    np.floor_divide,
    np.negative,
    np.positive,
    np.power,
    np.float_power,
    np.remainder,
    np.mod,
    np.fmod,
    np.divmod,
    np.absolute,
    np.fabs,
    np.rint,
    np.sign,
    np.heaviside,
    np.conj,
    np.conjugate,
    np.exp,
    np.exp2,
    np.log,
    np.log2,
    np.log10,
    np.expm1,
    np.log1p,
    np.sqrt,
    np.square,
    np.cbrt,
    np.reciprocal,
    np.gcd,
    np.lcm,
    np.sin,
    np.cos,
    np.tan,
    np.arcsin,
    np.arccos,
    np.arctan,
    np.arctan2,
    np.hypot,
    np.sinh,
    np.cosh,
    np.tanh,
    np.arcsinh,
    np.arccosh,
    np.arctanh,
    np.degrees,
    np.radians,
    np.deg2rad,
    np.rad2deg,
    np.bitwise_and,
    np.bitwise_or,
    np.bitwise_xor,
    np.invert,
    np.left_shift,
    np.right_shift,
    np.greater,
    np.greater_equal,
    np.less,
    np.less_equal,
    np.not_equal,
    np.equal,
    np.logical_and,
    np.logical_or,
    np.logical_xor,
    np.logical_not,
    np.maximum,
    np.minimum,
    np.fmax,
    np.fmin,
    np.isfinite,
    np.isinf,
    np.isnan,
    np.isnat,
    np.fabs,
    np.signbit,
    np.copysign,
    np.nextafter,
    np.spacing,
    np.modf,
    np.ldexp,
    np.frexp,
    np.fmod,
    np.floor,
    np.ceil,
    np.trunc,
]


@pytest.mark.parametrize("event", [MONTH, DAY_OF_MONTH])
@pytest.mark.parametrize("ufunc", NUMPY_UFUNCS)
@given(t=date_range_indexes())
def test_unary_ufunc(event, ufunc, t):
    if ufunc.nin != 1:
        pytest.skip("Not a valid ufunc for test.")

    if ufunc == np.isnat and np.issubdtype(MONTH.sample_from(t).dtype, np.number):
        with pytest.raises(TypeError):
            _ = ufunc(event.sample_from(t).to_numpy())
        with pytest.raises(TypeError):
            _ = ufunc(event).sample_from(t).to_numpy()
        return

    if ufunc in [np.modf, np.frexp]:
        with pytest.raises(NotImplementedError):
            _ = ufunc(event)
        return

    np.testing.assert_array_equal(ufunc(event.sample_from(t).to_numpy()),
                                  ufunc(event).sample_from(t).to_numpy())


@pytest.mark.parametrize("event", [MONTH, DAY_OF_MONTH])
@pytest.mark.parametrize("ufunc", NUMPY_UFUNCS)
@given(t=date_range_indexes())
def test_binary_ufunc(event, ufunc, t):
    if ufunc.nin != 2:
        pytest.skip("Not a valid ufunc for test.")

    if ufunc == np.divmod:
        with pytest.raises(NotImplementedError):
            _ = ufunc(event, event).sample_from(t)
        return

    np.testing.assert_array_equal(ufunc(event.sample_from(t).to_numpy(), event.sample_from(t).to_numpy()),
                                  ufunc(event, event).sample_from(t).to_numpy())