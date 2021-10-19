import datetime
import operator
from abc import ABC, abstractmethod, ABCMeta, abstractproperty
from dataclasses import dataclass, field, asdict
from functools import wraps
from math import ceil, floor
from numbers import Number
from typing import Callable, Type, TypeVar, Sequence, Any, Union, Iterable, Optional, ClassVar, Tuple

import numpy as np
import pandas as pd
from pandas import DatetimeIndex
from scipy import signal

from happenings.utils import (
    is_increasing_time,
    extract,
    support_sections,
    increase_date_freq,
)

ERT = TypeVar("ERT")

DATETIME_LIKE = Union[datetime.datetime, pd.Timestamp, np.datetime64]
TIMEDELTA_LIKE = Union[datetime.timedelta, pd.Timedelta, np.timedelta64]


@dataclass
class NameFactoryBase(ABC):
    """
    Abstract Base Class (ABC) for NameFactory that will be used to name events that are derived/returned.

    This class can be subclassed and `event_name` and `event_name_from_elements` must be implemented.

    All Event method and helper/friend functions should accept a NameFactorf
    """

    num: ClassVar[int] = 0
    delim: ClassVar[str] = "_"

    @classmethod
    @abstractmethod
    def event_name(cls, *args, **kwargs) -> str:
        """ Function called when the creation of a new event requires a name.

        Returns
        -------
        event_name : str
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def event_name_from_elements(cls, *args, **kwargs) -> str:
        """ Function called when the derivation of a new event from other elements requires a name.

        Returns
        -------
        event_name : str
        """
        raise NotImplementedError()


NameFactory = TypeVar("NameFactory", bound=NameFactoryBase)
DE = TypeVar("DE", bound="DerivedEvent")


@dataclass
class DefaultNameFactory(NameFactoryBase):
    """
    DefaultNameFactory that is used as a default when no other NameFactory is provided
    """

    @classmethod
    def event_name(cls, base_name: str = "Event") -> str:
        name = cls.delim.join([base_name, str(cls.num)])
        cls.num += 1
        return name

    @classmethod
    def event_name_from_elements(cls, function: Callable[[Any], ERT], base_elements: Sequence["Event"]) -> str:
        base_elements_names = []
        for element in base_elements:
            try:
                base_elements_names.append(element.name)
            except AttributeError:
                base_elements_names.append(str(element))

        args = ", ".join(base_elements_names)
        return cls.event_name(f"{function.__name__}({args})")


def return_as_series(sample_from: Callable):
    @wraps(sample_from)
    def _as_series(self: Event, dates: DatetimeIndex) -> pd.Series:
        return pd.Series(sample_from(self, dates), dtype=self.dtype)

    return _as_series


def validates_dates(sample_from: Callable):
    @wraps(sample_from)
    def _validate_dates(self, dates: DatetimeIndex, *args, **kwargs) -> Sequence:
        self._validate_dates(dates)
        return sample_from(self, dates, *args, **kwargs)

    return _validate_dates


def flip_dates_when_reversed(sample_from: Callable):
    @wraps(sample_from)
    def _sample_from_reverse(self: Event, dates: DatetimeIndex) -> Sequence:
        self._validate_dates(dates)

        # The following if statement is strange: It invokes from self in one case but the other is the wrapped
        # function. I don't think this is the best way to handel this...

        if is_increasing_time(dates.freq):
            return self._sample_from_reverse(dates)
        else:
            return sample_from(self, dates)

    return _sample_from_reverse


class TransparentNumpySubclass(ABCMeta):
    """
    Metaclass used to make any derived class (class that has TransparentNumpySubclass as a metaclass) will be a type of
    a Numpy ndarray, and thus have access to the __array_*__ subclassing features.
    """

    def __instancecheck__(self, instance):
        return super().__instancecheck__(instance) or isinstance(instance, np.ndarray)


class NumpyDunders(ABC):
    """
    Abstract Base Class for Numpy ndarray __array_*__ (double under) methods.
    """

    # @abstractmethod
    # @property
    # def __array_priority__(self):
    #     raise NotImplementedError
    #
    # @abstractmethod
    # def __array__(self, dtype):
    #     raise NotImplementedError()
    #
    # @abstractmethod
    # def __array_finalize__(self, obj):
    #     raise NotImplementedError()
    #
    # @abstractmethod
    # def __array_wrap__(self, out_arr, context=None):
    #     raise NotImplementedError()
    #
    # @abstractmethod
    # def __array_prepare__(self, array, context=None):
    #     raise NotImplementedError()
    #
    # @abstractmethod
    # def __array_function__(self, func, types, args, kwargs):
    #     raise NotImplementedError()

    @staticmethod
    def not_implemented(func: Optional[Callable] = None, not_implemented_ufuncs: Sequence[np.ufunc] = ()):
        def _not_implemented(_func: Callable):
            @wraps(_func)
            def _raised_not_implemented(self, ufunc, method: str, *inputs: Tuple[Any], **kwargs):
                if ufunc in not_implemented_ufuncs:
                    raise NotImplementedError(f"{ufunc} is not supported")
                return _func(self, ufunc, method, *inputs, **kwargs)

            return _raised_not_implemented

        if func:
            return _not_implemented(func)
        return _not_implemented

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        raise NotImplementedError()

    @abstractmethod
    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs: Tuple[Any], **kwargs):
        """ A function that execututes when a numpy ufuncs if called on a subclass of NumpyDunder and thus
        is used to override the default ndarray.__array_ufunc__ method.

        Parameters
        ----------
        ufunc: the ufunc object that was called.
        method:  string indicating how the Ufunc was called, either:
            ``"__call__"`` to indicate it was called directly, or one of its
            :ref:`methods<ufuncs.methods>`: ``"reduce"``, ``"accumulate"``,
            ``"reduceat"``, ``"outer"``, or ``"at"``.
        inputs: a tuple of the input arguments to the ``ufunc``
        kwargs: contains any optional or keyword arguments passed to the
          function. This includes any ``out`` arguments, which are always
          contained in a tuple.

        Returns
        -------


        Notes
        -----
        This method is executed instead of the ufunc and should return either the result of the operation,
         or NotImplemented if the operation requested is not implemented.

        """
        raise NotImplementedError()


@dataclass(frozen=True, eq=False, order=False)
class Event(NumpyDunders, ABC, metaclass=TransparentNumpySubclass):
    """
    Abstract Base Class for Events.

    Implements all __*__ methods (including __array_*__ from NumpyDunders) that are relevant for Events
    """

    name: str
    dtype: ERT
    name_factory: Type[NameFactory]

    def renamed(self, name: str) -> "Event":
        data = asdict(self)
        data["name"] = name
        return type(self)(**data)

    @abstractmethod
    def sample_from(self, dates: DatetimeIndex) -> pd.Series:
        raise NotImplementedError()

    def _sample_from_reverse(self, dates: DatetimeIndex) -> pd.Series:
        return self.sample_from(dates[::-1])[::-1]

    def _validate_dates(self, dates: DatetimeIndex):
        if not isinstance(dates, DatetimeIndex):
            raise TypeError(f"expected dates to be a single dimension DatetimeIndex, but got {dates}")
        if dates.freq is None:
            raise ValueError("expected dates to have a set freq, but got None")

    def convolve(
        self,
        window: np.ndarray,
        sub_sampling: int = 2,
        mode="same",
        method="auto",
        dtype: Type = np.float64,
        name: Optional[str] = None,
    ) -> "ConvolvedEvent":
        """See convolve"""
        return convolve(
            event=self,
            window=window,
            sub_sampling=sub_sampling,
            mode=mode,
            method=method,
            dtype=dtype,
            name=name,
            name_factory=self.name_factory,
        )

    def buffer(self, by: TIMEDELTA_LIKE, dtype: Type = np.float64, name: Optional[str] = None) -> "BufferedEvent":
        """See buffer"""
        return buffer(event=self, by=by, name=name, dtype=dtype, name_factory=self.name_factory)

    def scale(self, by: float, dtype: Type = np.float64, name: Optional[str] = None) -> "ScaledEvent":
        """See scale"""
        return scale(event=self, by=by, name=name, dtype=dtype, name_factory=self.name_factory)

    def astype(self, dtype, order=None, casting=None, subok=None, copy=None) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=lambda x: x.astype(dtype), base_elements=[self], name_factory=self.name_factory,
        )

    @NumpyDunders.not_implemented(not_implemented_ufuncs=[np.modf, np.frexp, np.divmod])
    # This could be replaced with ufunc.nout > 1, but this is a more generic approach that allows for future changes
    def __array_ufunc__(self, ufunc: Callable, method: str, *inputs: "Event", **kwargs) -> "DerivedEvent":
        """ Method called whenever an numpy ufunc is called with an Event as one of the inputs

        Parameters
        ----------
        See NumpyDunders for Parameters

        Returns
        -------
        event : The derivived event that has the respective unfunc applied on it.

        Notes
        -----
        No kwargs are supported at the moment
        Certain numpy ufuncs will not work as intended or not work at all.

        """
        if kwargs:
            raise NotImplementedError(f"Events do not currently support kwargs.")
        return DerivedEvent.from_base_elements(derivation=ufunc, base_elements=inputs, name_factory=self.name_factory,)

    def __mul__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.mul, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __rmul__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.mul, base_elements=[other, self], name_factory=self.name_factory,
        )

    def __add__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.add, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __radd__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.add, base_elements=[other, self], name_factory=self.name_factory,
        )

    def __sub__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.sub, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __rsub__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.sub, base_elements=[other, self], name_factory=self.name_factory,
        )

    def __lt__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.lt, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __le__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.le, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __eq__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.eq, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __ne__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.ne, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __ge__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.ge, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __gt__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.gt, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __neg__(self) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.neg, base_elements=[self], name_factory=self.name_factory,
        )

    def __abs__(self) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.abs, base_elements=[self], name_factory=self.name_factory,
        )

    def __and__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.and_, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __rand__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.and_, base_elements=[other, self], name_factory=self.name_factory,
        )

    def __floordiv__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.floordiv, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __rfloordiv__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.floordiv, base_elements=[other, self], name_factory=self.name_factory,
        )

    def __invert__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.inv, base_elements=[other], name_factory=self.name_factory,
        )

    def __lshift__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.lshift, base_elements=[other], name_factory=self.name_factory,
        )

    def __mod__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.mod, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __rmod__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.mod, base_elements=[other, self], name_factory=self.name_factory,
        )

    def __matmul__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.matmul, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __rmatmul__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.matmul, base_elements=[other, self], name_factory=self.name_factory,
        )

    def __or__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.or_, base_elements=[other, self], name_factory=self.name_factory,
        )

    def __ror__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.or_, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __pos__(self) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.pos, base_elements=[self], name_factory=self.name_factory,
        )

    def __pow__(self, power, modulo=None) -> "DerivedEvent":
        if modulo is not None:
            raise NotImplementedError(f"modulo argument is not supported for {self.__class__.__name__}")
        return DerivedEvent.from_base_elements(
            derivation=operator.pow, base_elements=[self, power], name_factory=self.name_factory,
        )

    def __rshift__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.rshift, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __truediv__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.truediv, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __rtruediv__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.truediv, base_elements=[other, self], name_factory=self.name_factory,
        )

    def __xor__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.xor, base_elements=[self, other], name_factory=self.name_factory,
        )

    def __rxor__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(
            derivation=operator.xor, base_elements=[other, self], name_factory=self.name_factory,
        )


def function_event(
    function: Callable[[DATETIME_LIKE], ERT],
    vectorized: bool = False,
    name: Optional[str] = None,
    dtype: Type = np.float64,
    name_factory: Type[NameFactory] = DefaultNameFactory,
) -> "FunctionalEvent":
    if name is None:
        name = name_factory.event_name(base_name="FunctionalEvent")

    return FunctionalEvent(name=name, function=function, vectorized=vectorized, dtype=dtype, name_factory=name_factory,)


@dataclass(frozen=True, eq=False, order=False)
class FunctionalEvent(Event):
    function: Callable[[DATETIME_LIKE], ERT] = field(hash=True, repr=False)
    vectorized: bool

    @return_as_series
    @validates_dates
    @flip_dates_when_reversed
    def sample_from(self, dates: DatetimeIndex) -> pd.Series:
        if self.vectorized:
            return self.function(dates)
        else:
            return dates.map(self.function)


@dataclass(frozen=True, eq=False, order=False)
class DerivedEvent(Event):
    derivation: Callable[[Any], ERT] = field(hash=False, repr=False)
    base_elements: Sequence[Union[Event, float, int]] = field(hash=False, repr=False)

    @classmethod
    def from_base_elements(
        cls: Type[DE],
        derivation: Callable[[Any], ERT],
        base_elements: Sequence[Event],
        name: Optional[str] = None,
        dtype: Type = np.float64,
        name_factory: Type[NameFactory] = DefaultNameFactory,
    ) -> DE:
        if name is None:
            name = name_factory.event_name_from_elements(derivation, base_elements)
        return cls(
            name=name, derivation=derivation, base_elements=base_elements, dtype=dtype, name_factory=name_factory,
        )

    @return_as_series
    @validates_dates
    @flip_dates_when_reversed
    def sample_from(self, dates: DatetimeIndex) -> pd.Series:
        # Recursive method. Could refactor to stack/queue with memorization, but don't see the need.

        samples = []
        for element in self.base_elements:
            if issubclass(type(element), Event):
                samples.append(element.sample_from(dates))
            else:
                samples.append(element)
        return self.derivation(*samples)


CE = TypeVar("CE", bound="ConvolvedEvent")


def convolve(
    event: Event,
    window: Union[Event, np.ndarray, Iterable, int, float],
    sub_sampling: int = 2,
    mode: str = "same",
    method: str = "auto",
    name: Optional[str] = None,
    dtype: Type = np.float64,
    name_factory: Type[NameFactory] = DefaultNameFactory,
) -> "ConvolvedEvent":
    if name is None:
        name = name_factory.event_name(f"Convolved {event.name}")
    return ConvolvedEvent(
        name=name,
        event=event,
        window=window,
        sub_sampling=sub_sampling,
        mode=mode,
        method=method,
        dtype=dtype,
        name_factory=name_factory,
    )


@dataclass(frozen=True, eq=False, order=False)
class ConvolvedEvent(Event):
    event: Event = field(hash=False, repr=False)
    window: Union[Event, np.ndarray, Iterable, int, float] = field(hash=False, repr=False)
    sub_sampling: int = 2
    mode: str = "same"
    method: str = "auto"

    @classmethod
    def from_base_elements(
        cls: Type[CE],
        event: Event,
        window: Union[Event, np.ndarray, Iterable, int, float],
        sub_sampling: int = 2,
        mode: str = "same",
        method: str = "auto",
        name: Optional[str] = None,
        dtype: Type = np.float64,
        name_factory: Type[NameFactory] = DefaultNameFactory,
    ) -> CE:
        if name is None:
            name = name_factory.event_name(f"Convolved {event.name}")

        return cls(
            name=name,
            event=event,
            window=window,
            sub_sampling=sub_sampling,
            mode=mode,
            method=method,
            dtype=dtype,
            name_factory=name_factory,
        )

    @return_as_series
    @validates_dates
    @flip_dates_when_reversed
    def sample_from(self, dates: DatetimeIndex) -> pd.Series:
        infilled_dates = increase_date_freq(dates, self.sub_sampling)

        if issubclass(type(self.window), Event):
            win = self.window.sample_from(infilled_dates)
        else:
            win = self.window

        infilled_convolve = signal.convolve(
            in1=self.event.sample_from(infilled_dates), in2=win, mode=self.mode, method=self.method,
        )

        return extract(infilled_convolve, extract_num=self.sub_sampling) #/ sum(win)


def to_periods(
    dates: Union[DatetimeIndex, np.ndarray],
    period: Union[pd.Timedelta, datetime.timedelta, np.datetime64],
    base_date: pd.Timestamp = None,
) -> pd.Series:
    if base_date is None:
        base_date = dates[0]
    return pd.Series((dates - base_date) / period)


def interp(
    x_dates: np.ndarray,
    y_values: np.ndarray,
    left: Optional[Number] = None,
    right: Optional[Number] = None,
    period: Optional[TIMEDELTA_LIKE] = None,
    to_float_datetime: TIMEDELTA_LIKE = np.timedelta64(1, "ns"),
    name: Optional[str] = None,
    dtype: Type = np.float64,
    name_factory: Type[NameFactory] = DefaultNameFactory,
) -> "LinearInterpolationEvent":

    if name is None:
        name = name_factory.event_name(base_name="InterpolatedEvent")
    return LinearInterpolationEvent(
        x_dates=x_dates,
        y_values=y_values,
        left=left,
        right=right,
        period=period,
        to_float_datetime=to_float_datetime,
        name=name,
        dtype=dtype,
        name_factory=name_factory,
    )


@dataclass(frozen=True, eq=False, order=False)
class LinearInterpolationEvent(Event):
    x_dates: np.ndarray
    y_values: np.ndarray
    left: Optional[Number] = None
    right: Optional[Number] = None
    period: Optional[TIMEDELTA_LIKE] = None
    to_float_datetime: TIMEDELTA_LIKE = np.timedelta64(1, "ns")

    @return_as_series
    @validates_dates
    @flip_dates_when_reversed
    def sample_from(self, dates: DatetimeIndex) -> pd.Series:
        dates_from_origin = to_periods(dates=dates, period=self.to_float_datetime, base_date=self.x_dates[0])
        x_dates_from_origin = to_periods(dates=self.x_dates, period=self.to_float_datetime, base_date=self.x_dates[0])

        if self.period is not None:
            period_as_ns = self.period / self.to_float_datetime
        else:
            period_as_ns = None

        return np.interp(
            dates_from_origin, x_dates_from_origin, self.y_values, left=self.left, right=self.right, period=period_as_ns,
        )


@dataclass(frozen=True, eq=False, order=False)
class FourierFunctionEvent(FunctionalEvent):
    seasonal_period: TIMEDELTA_LIKE
    order: int

    @return_as_series
    @validates_dates
    @flip_dates_when_reversed
    def sample_from(self, dates: DatetimeIndex) -> pd.Series:
        dates_as_periods = to_periods(dates=dates, period=self.seasonal_period)
        return self.function(2 * np.pi * self.order * dates_as_periods)


def buffer(
    event: Event,
    by: TIMEDELTA_LIKE,
    name: Optional[str] = None,
    dtype: Type = np.float64,
    name_factory: Type[NameFactory] = DefaultNameFactory,
) -> "BufferedEvent":
    if name is None:
        name = name_factory.event_name(f"Buffered {event.name} by {str(by)}")
    return BufferedEvent(name=name, event=event, by=by, dtype=dtype, name_factory=name_factory)


@dataclass(frozen=True, eq=False, order=False)
class BufferedEvent(Event):
    # TODO: define buffer function and refactor as a DerivedEvent. How to handle `sample`?
    event: Event
    by: TIMEDELTA_LIKE

    def __post_init__(self):
        if not isinstance(self.event, Event):
            raise TypeError(f"expected `event` to be an instance of {Event}, but got {type(self.event)}")
        if not isinstance(self.by, TIMEDELTA_LIKE.__args__):
            raise TypeError(f"expected `by` to be of type {TIMEDELTA_LIKE}")
        if not pd.Timedelta(self.by) >= pd.Timedelta(datetime.timedelta(0)):
            raise ValueError(f"expected `by` to be a positive width instance of {TIMEDELTA_LIKE}")

    @return_as_series
    @validates_dates
    @flip_dates_when_reversed
    def sample_from(self, dates: pd.DatetimeIndex) -> pd.Series:
        values = self.event.sample_from(dates)

        buffered_values = values.copy()
        nonzero_sections = support_sections(values)

        for (sect_start_index, sect_stop_end) in nonzero_sections:
            buffer_width = pd.Timedelta(self.by) // dates.freq
            buffered_values[max(0, sect_start_index - buffer_width) : sect_start_index] = values[sect_start_index]
            buffered_values[sect_stop_end : min(len(dates), sect_stop_end + buffer_width)] = values[sect_stop_end - 1]

        return buffered_values


def scale(
    event: Event,
    by: float,
    name: Optional[str] = None,
    dtype: Type = np.float64,
    name_factory: Type[NameFactory] = DefaultNameFactory,
) -> "ScaledEvent":
    if name is None:
        name = name_factory.event_name(f"Scaled {event.name} by {str(by)}")
    return ScaledEvent(name=name, event=event, by=by, dtype=dtype, name_factory=name_factory)


@dataclass(frozen=True, eq=False, order=False)
class ScaledEvent(Event):
    # TODO: define scale function and refactor as a DerivedEvent. How to handle `sample`?
    event: Event
    by: float

    def __post_init__(self):
        if self.by <= 0:
            raise ValueError(f"espected `by` to be a positive float, but got {self.by}")
        if not isinstance(self.event, Event):
            raise ValueError(f"expected event to be an instance of {Event}, but got {type(self.event)}")

    @return_as_series
    @validates_dates
    @flip_dates_when_reversed
    def sample_from(self, dates: pd.DatetimeIndex):
        values = self.event.sample_from(dates)
        scaled_values = np.zeros_like(values)

        nonzero_sections = support_sections(values)

        for (section_start_index, section_end_index) in nonzero_sections:
            num_section_periods = section_end_index - section_start_index  # Number of non-zero items found in section
            scaled_section_periods = round(self.by * num_section_periods)  # Number of non-zero items in scaled section
            scaled_section_date_freq = (dates[section_end_index] - dates[section_start_index]) / scaled_section_periods
            section_center = num_section_periods // 2 + section_start_index

            scaled_section_start = section_center - floor(scaled_section_periods / 2)
            scaled_section_end = section_center + ceil(scaled_section_periods / 2)

            start_offset = max(0, scaled_section_start)
            end_offset = min(len(dates), scaled_section_end)
            valid_interp_slice = slice(
                start_offset - scaled_section_start, scaled_section_periods - scaled_section_end + end_offset,
            )

            interp_dates = pd.date_range(
                start=dates[section_start_index], end=dates[section_end_index], freq=scaled_section_date_freq,
            )

            scaled_section_values = self.event.sample_from(interp_dates)
            scaled_values[start_offset:end_offset] += scaled_section_values[valid_interp_slice]

        return scaled_values
