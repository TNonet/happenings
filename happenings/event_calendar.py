import calendar
from dataclasses import dataclass
from enum import IntEnum, EnumMeta
from typing import Optional, Union, Sequence, Dict

import pandas as pd

from happenings.core import Event


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


def week_of_year(
    dates: pd.DatetimeIndex,
    week_start_day: str = "Monday",
    week_start_date_enum: IntEnum = StandardWeekStartDates,
    leading_year_zeroes: int = 4,
    delim: str = "-",
) -> pd.Series:
    if week_start_day not in week_start_date_enum:
        raise ValueError(f"expected one of {week_start_date_enum.members}, but got {week_start_day}")

    date_offset = week_start_date_enum[week_start_day] - calendar.firstweekday()

    shifted_dates = (dates - pd.Timedelta(days=date_offset)).isocalendar().reset_index(drop=True)

    str_shifted_dates = (
        shifted_dates.year.astype(str).str.zfill(leading_year_zeroes)
        + delim
        + shifted_dates.week.astype(str).str.zfill(2)
    )

    return pd.Series(data=str_shifted_dates.to_numpy(), index=dates, dtype=str)


@dataclass
class EventCalendar:
    events: Sequence[Event]

    def sample_from(
        self,
        dates: pd.DatetimeIndex,
        group_by: Optional[Union[str, Event, Sequence[str], Sequence[Event]]] = None,
        agg: Union[str, Dict[str, Union[pd.NamedAgg, str]]] = "sum",
        index_agg: Optional[Union[str, Dict[str, Union[pd.NamedAgg, str]]]] = None,
    ) -> pd.DataFrame:

        if group_by is not None:
            if isinstance(group_by, (str, Event)):
                group_by = [group_by]

            for g in group_by:
                if isinstance(g, str):
                    if g not in [e.name for e in self.events]:
                        raise ValueError(f"expected `aggregate_by` to be a name of an event in self.events")

        df = pd.DataFrame({e.name: e.sample_from(dates).values for e in self.events}, index=dates)

        if group_by is None:
            return df

        if index_agg is not None:
            index_name = dates.name if dates.name is not None else "index"

            if isinstance(agg, str):
                agg = {c: agg for c in df.columns}
                [agg.pop(g) for g in group_by]
            else:
                agg = agg

            agg[index_name] = index_agg

            df.reset_index(inplace=True)

        else:
            agg = agg

        group_by_cols = []
        for g in group_by:
            if isinstance(g, str):
                group_by_cols.append("__group_by_" + g + "__")
            else:
                df["__group_by_" + g.name + "__"] = g.sample_from(dates)

        return df.groupby(by=group_by).agg(agg)
