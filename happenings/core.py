import datetime
import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from functools import wraps
from math import ceil, floor
from numbers import Number
from typing import Callable, Type, TypeVar, Sequence, Any, Union, Iterable, Optional, ClassVar

import numpy as np
import pandas as pd
from pandas import DatetimeIndex
from scipy import signal

from evendar.utils import is_increasing_time, extract, support_sections, increase_date_freq

ERT = TypeVar("ERT")

DATETIME_LIKE = Union[datetime.datetime, pd.Timestamp, np.datetime64]
TIMEDELTA_LIKE = Union[datetime.timedelta, pd.Timedelta, np.timedelta64]


@dataclass
class NameFactoryBase(ABC):
    num: ClassVar[int] = 0
    delim: ClassVar[str] = "_"

    @classmethod
    @abstractmethod
    def event_name(cls, *args, **kwargs) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def event_name_from_elements(cls, *args, **kwargs) -> str:
        raise NotImplementedError


TNameFactor = TypeVar("TNameFactor", bound=NameFactoryBase)


@dataclass
class DefaultNameFactory(NameFactoryBase):

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


def validates_date_freq(sample_from: Callable):
    @wraps(sample_from)
    def _validate_date_freq(self: Event, dates: DatetimeIndex) -> Sequence:
        self._validate_dates(dates)
        return sample_from(self, dates)

    return _validate_date_freq


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


@dataclass(frozen=True, eq=False, order=False)
class Event(ABC):
    name: str
    dtype: ERT
    name_factory: Type[TNameFactor]

    def renamed(self, name: str) -> "Event":
        data = asdict(self)
        data['name'] = name
        return type(self)(**data)

    def sample_from(self, dates: DatetimeIndex) -> pd.Series:
        raise NotImplementedError

    def _sample_from_reverse(self, dates: DatetimeIndex) -> pd.Series:
        return self.sample_from(dates[::-1])[::-1]

    def _validate_dates(self, dates: DatetimeIndex):
        if dates.freq is None:
            raise ValueError("expected dates to have a set freq, but got None")

    def convolve(self, window: np.ndarray, sub_sampling: int = 2, mode='same', method='auto',
                 dtype: Type = np.float64, name: Optional[str] = None) -> "ConvolvedEvent":
        """See convolve"""
        return convolve(event=self, window=window, sub_sampling=sub_sampling, mode=mode, method=method, dtype=dtype,
                        name=name, name_factory=self.name_factory)

    def buffer(self, by: TIMEDELTA_LIKE, dtype: Type = np.float64, name: Optional[str] = None) -> "BufferedEvent":
        """See buffer"""
        return buffer(event=self, by=by, name=name, dtype=dtype, name_factory=self.name_factory)

    def scale(self, by: float, dtype: Type = np.float64, name: Optional[str] = None) -> "ScaledEvent":
        """See scale"""
        return scale(event=self, by=by, name=name, dtype=dtype, name_factory=self.name_factory)

    def __mul__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.mul, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __rmul__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.mul, base_elements=[other, self],
                                               name_factory=self.name_factory)

    def __add__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.add, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __radd__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.add, base_elements=[other, self],
                                               name_factory=self.name_factory)

    def __sub__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.sub, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __rsub__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.sub, base_elements=[other, self],
                                               name_factory=self.name_factory)

    def __lt__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.lt, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __le__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.le, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __eq__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.eq, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __ne__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.ne, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __ge__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.ge, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __gt__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.gt, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __neg__(self) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.neg, base_elements=[self],
                                               name_factory=self.name_factory)

    def __abs__(self) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.abs, base_elements=[self],
                                               name_factory=self.name_factory)

    def __and__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.and_, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __rand__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.and_, base_elements=[other, self],
                                               name_factory=self.name_factory)

    def __floordiv__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.floordiv, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __rfloordiv__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.floordiv, base_elements=[other, self],
                                               name_factory=self.name_factory)

    def __invert__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.inv, base_elements=[other],
                                               name_factory=self.name_factory)

    def __lshift__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.lshift, base_elements=[other],
                                               name_factory=self.name_factory)

    def __mod__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.mod, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __rmod__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.mod, base_elements=[other, self],
                                               name_factory=self.name_factory)

    def __matmul__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.matmul, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __rmatmul__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.matmul, base_elements=[other, self],
                                               name_factory=self.name_factory)

    def __or__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.or_, base_elements=[other, self],
                                               name_factory=self.name_factory)

    def __ror__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.or_, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __pos__(self) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.pos, base_elements=[self],
                                               name_factory=self.name_factory)

    def __pow__(self, power, modulo=None) -> "DerivedEvent":
        if modulo is not None:
            raise NotImplementedError(f"modulo argument is not supported for {self.__class__.__name__}")
        return DerivedEvent.from_base_elements(derivation=operator.pow, base_elements=[self, power],
                                               name_factory=self.name_factory)

    def __rshift__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.rshift, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __truediv__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.truediv, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __rtruediv__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.truediv, base_elements=[other, self],
                                               name_factory=self.name_factory)

    def __xor__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.xor, base_elements=[self, other],
                                               name_factory=self.name_factory)

    def __rxor__(self, other) -> "DerivedEvent":
        return DerivedEvent.from_base_elements(derivation=operator.xor, base_elements=[other, self],
                                               name_factory=self.name_factory)


def function_event(function: Callable[[DATETIME_LIKE], ERT], vectorized: bool = False,
                   name: Optional[str] = None, dtype: Type = np.float64,
                   name_factory: Type[TNameFactor] = DefaultNameFactory) -> "FunctionalEvent":
    if name is None:
        name = name_factory.event_name(base_name="FunctionalEvent")

    return FunctionalEvent(name=name, function=function, vectorized=vectorized, dtype=dtype, name_factory=name_factory)


@dataclass(frozen=True, eq=False, order=False)
class FunctionalEvent(Event):
    function: Callable[[DATETIME_LIKE], ERT] = field(hash=True, repr=False)
    vectorized: bool

    @return_as_series
    @validates_date_freq
    @flip_dates_when_reversed
    def sample_from(self, dates: DatetimeIndex) -> pd.Series:
        if self.vectorized:
            return self.function(dates)
        else:
            return dates.map(self.function)


DE = TypeVar("DE", bound="DerivedEvent")


@dataclass(frozen=True, eq=False, order=False)
class DerivedEvent(Event):
    derivation: Callable[[Any], ERT] = field(hash=False, repr=False)
    base_elements: Sequence[Union[Event, float, int]] = field(hash=False, repr=False)

    @classmethod
    def from_base_elements(cls: Type[DE], derivation: Callable[[Any], ERT], base_elements: Sequence[Event],
                           name: Optional[str] = None, dtype: Type = np.float64,
                           name_factory: Type[TNameFactor] = DefaultNameFactory) -> DE:
        if name is None:
            name = name_factory.event_name_from_elements(derivation, base_elements)
        return cls(name=name,
                   derivation=derivation,
                   base_elements=base_elements,
                   dtype=dtype,
                   name_factory=name_factory)

    @return_as_series
    @validates_date_freq
    @flip_dates_when_reversed
    def sample_from(self, dates: DatetimeIndex) -> pd.Series:
        # Recursive method. Could refactor to stack/queue with memorization, but don't see the need.

        samples = []
        for element in self.base_elements:
            if isinstance(element, Event):
                samples.append(element.sample_from(dates))
            else:
                samples.append(element)
        return self.derivation(*samples)


CE = TypeVar("CE", bound="ConvolvedEvent")


def convolve(event: Event, window: Union[Event, np.ndarray, Iterable, int, float], sub_sampling: int = 2,
             mode: str = 'same', method: str = "auto", name: Optional[str] = None,
             dtype: Type = np.float64, name_factory: Type[TNameFactor] = DefaultNameFactory) -> "ConvolvedEvent":
    if name is None:
        name = name_factory.event_name(f"Convolved {event.name}")
    return ConvolvedEvent(name=name, event=event, window=window, sub_sampling=sub_sampling, mode=mode, method=method,
                          dtype=dtype, name_factory=name_factory)


@dataclass(frozen=True, eq=False, order=False)
class ConvolvedEvent(Event):
    event: Event = field(hash=False, repr=False)
    window: Union[Event, np.ndarray, Iterable, int, float] = field(hash=False, repr=False)
    sub_sampling: int = 2
    mode: str = 'same'
    method: str = "auto"

    @classmethod
    def from_base_elements(cls: Type[CE], event: Event, window: Union[Event, np.ndarray, Iterable, int, float],
                           sub_sampling: int = 2, mode: str = 'same', method: str = "auto",
                           name: Optional[str] = None, dtype: Type = np.float64,
                           name_factory: Type[TNameFactor] = DefaultNameFactory) -> CE:
        if name is None:
            name = name_factory.event_name(f"Convolved {event.name}")

        return cls(name=name,
                   event=event,
                   window=window,
                   sub_sampling=sub_sampling,
                   mode=mode,
                   method=method,
                   dtype=dtype,
                   name_factory=name_factory)

    @return_as_series
    @validates_date_freq
    @flip_dates_when_reversed
    def sample_from(self, dates: DatetimeIndex) -> pd.Series:
        if dates.ndim != 1:
            raise NotImplementedError(f"Only 1D dates are allowed in {ConvolvedEvent} samples.")

        infilled_dates = increase_date_freq(dates, self.sub_sampling)

        if isinstance(self.window, Event):
            win = self.window.sample_from(infilled_dates)
        else:
            win = self.window

        infilled_convolve = signal.convolve(in1=self.event.sample_from(infilled_dates), in2=win,
                                            mode=self.mode, method=self.method)

        return extract(infilled_convolve, extract_num=self.sub_sampling) / sum(win)


def to_periods(dates: Union[DatetimeIndex, np.ndarray], period: Union[pd.Timedelta, datetime.timedelta, np.datetime64],
               base_date: pd.Timestamp = None) -> pd.Series:
    if base_date is None:
        base_date = dates[0]
    return pd.Series((dates - base_date) / period)


@dataclass(frozen=True, eq=False, order=False)
class LinearInterpolationEvent(Event):
    x_dates: np.ndarray
    y_values: np.ndarray
    left: Optional[Number] = None
    right: Optional[Number] = None
    period: Optional[TIMEDELTA_LIKE] = None
    to_float_datetime: TIMEDELTA_LIKE = np.timedelta64(1, 'ns')

    @return_as_series
    @validates_date_freq
    @flip_dates_when_reversed
    def sample_from(self, dates: DatetimeIndex) -> pd.Series:
        dates_from_origin = to_periods(dates=dates, period=self.to_float_datetime, base_date=self.x_dates[0])
        x_dates_from_origin = to_periods(dates=self.x_dates, period=self.to_float_datetime, base_date=self.x_dates[0])

        if self.period is not None:
            period_as_ns = self.period / self.to_float_datetime
        else:
            period_as_ns = None

        return np.interp(dates_from_origin, x_dates_from_origin, self.y_values,
                         left=self.left, right=self.right, period=period_as_ns)


@dataclass(frozen=True, eq=False, order=False)
class FourierFunctionEvent(FunctionalEvent):
    seasonal_period: TIMEDELTA_LIKE
    order: int

    @return_as_series
    @validates_date_freq
    @flip_dates_when_reversed
    def sample_from(self, dates: DatetimeIndex) -> pd.Series:
        dates_as_periods = to_periods(dates=dates, period=self.seasonal_period)
        return self.function(2 * np.pi * self.order * dates_as_periods)


def buffer(event: Event, by: TIMEDELTA_LIKE, name: Optional[str] = None, dtype: Type = np.float64,
           name_factory: Type[TNameFactor] = DefaultNameFactory) -> "BufferedEvent":
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
    @validates_date_freq
    @flip_dates_when_reversed
    def sample_from(self, dates: pd.DatetimeIndex) -> pd.Series:
        values = self.event.sample_from(dates)

        buffered_values = values.copy()
        nonzero_sections = support_sections(values)

        for (sect_start_index, sect_stop_end) in nonzero_sections:
            buffer_width = pd.Timedelta(self.by) // dates.freq
            buffered_values[max(0, sect_start_index - buffer_width):sect_start_index] = values[sect_start_index]
            buffered_values[sect_stop_end:min(len(dates), sect_stop_end + buffer_width)] = values[sect_stop_end - 1]

        return buffered_values


def scale(event: Event, by: float, name: Optional[str] = None, dtype: Type = np.float64,
          name_factory: Type[TNameFactor] = DefaultNameFactory) -> "ScaledEvent":
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
    @validates_date_freq
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
            valid_interp_slice = slice(start_offset - scaled_section_start,
                                       scaled_section_periods - scaled_section_end + end_offset)

            interp_dates = pd.date_range(start=dates[section_start_index],
                                         end=dates[section_end_index],
                                         freq=scaled_section_date_freq)

            scaled_section_values = self.event.sample_from(interp_dates)
            scaled_values[start_offset:end_offset] += scaled_section_values[valid_interp_slice]

        return scaled_values
