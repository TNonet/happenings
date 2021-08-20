import calendar
from dataclasses import dataclass
from enum import IntEnum, EnumMeta
from typing import Optional, Union, Sequence

import pandas as pd

from evendar.core import Event


class ContainmentMixin(EnumMeta):
    """
    https://stackoverflow.com/questions/43634618/how-do-i-test-if-int-value-exists-in-python-enum-without-using-try-catch
    """

    def __contains__(cls, item):
        return item in cls.members

    @property
    def members(cls):
        return cls.__members__

    def __getitem__(self, item):
        return self.members[item]


class StandardWeekStartDates(IntEnum, metaclass=ContainmentMixin):
    """
    See Also
    --------
    `calendar` for conversion
    """
    Monday = calendar.MONDAY
    Tuesday = calendar.TUESDAY
    Wednesday = calendar.WEDNESDAY
    Thursday = calendar.THURSDAY
    Friday = calendar.FRIDAY
    Saturday = calendar.SATURDAY
    Sunday = calendar.SUNDAY


def week_of_year(dates: pd.DatetimeIndex, week_start_day: str = "Monday",
                 week_start_date_enum: IntEnum = StandardWeekStartDates,
                 leading_year_zeroes: int = 4, delim: str = "-") -> pd.Series:
    if week_start_day not in week_start_date_enum:
        raise ValueError(f"expected one of {week_start_date_enum.members}, but got {week_start_day}")

    date_offset = week_start_date_enum[week_start_day] - calendar.firstweekday()

    shifted_dates = (dates - pd.Timedelta(days=date_offset)).isocalendar().reset_index(drop=True)

    str_shifted_dates = (shifted_dates.year.astype(str).str.zfill(leading_year_zeroes)
                         + delim + shifted_dates.week.astype(str).str.zfill(2))

    return pd.Series(data=str_shifted_dates.to_numpy(),
                     index=dates,
                     dtype=str)


@dataclass
class EventCalendar:
    events: Sequence[Event]

    def sample_from(self,
                    dates: pd.DatetimeIndex,
                    groupy_by: Optional[Union[str, Event]] = None) -> pd.DataFrame:
        if isinstance(groupy_by, str):
            if groupy_by not in [e.name for e in self.events]:
                raise ValueError(f"expected `aggregate_by` to be a name of an event in self.events")

        df = pd.DataFrame(columns=[e.name for e in self.events],
                          index=dates)

        for e in self.events:
            df[e.name] = e.sample_from(dates).values

        if groupy_by is None:
            pass
        elif isinstance(groupy_by, str):
            df.groupby(by=groupy_by).agg('sum')
        else:
            df[groupy_by.name] = groupy_by.sample_from(dates)
            df.groupby(by=groupy_by.name).agg('sum')

        return df
