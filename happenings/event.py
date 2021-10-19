from functools import partial

import numpy as np
from dateutil.easter import easter, EASTER_WESTERN  # ,EASTER_ORTHODOX, EASTER_JULIAN

from happenings.event_calendar import week_of_year
from happenings.core import function_event

YEAR = function_event(lambda x: x.year, vectorized=True, name="Year", dtype=np.int16)

MONTH = function_event(lambda x: x.month, vectorized=True, name="Month", dtype=np.int8)

MONTHS = {}
for index, event_name in enumerate(
    [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
):

    def _lambda(i):
        return lambda x: x.month == i + 1

    MONTHS[event_name] = function_event(_lambda(index), vectorized=True, name=event_name, dtype=np.bool_)

DAY_OF_MONTH = function_event(lambda x: x.day, vectorized=True, name="Day of month", dtype=np.int8)
DAY_OF_YEAR = function_event(lambda x: x.dayofyear, vectorized=True, name="Day of year", dtype=np.int8)

DAYS_OF_WEEK = {}
WEEK_OF_YEAR = {}

for day_of_week in [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]:

    def _lambda(i):
        return lambda x: x.day_name() == i

    DAYS_OF_WEEK[day_of_week] = function_event(_lambda(day_of_week), vectorized=True, name=day_of_week, dtype=np.bool_)

    def _lambda(i):
        return i

    WEEK_OF_YEAR[day_of_week] = function_event(
        partial(week_of_year, week_start_day=day_of_week), vectorized=True, name=f"W-{day_of_week}", dtype=str,
    )


# Source: https://en.wikipedia.org/wiki/Federal_holidays_in_the_United_States
US_FEDERAL_HOLIDAYS = {}

US_FEDERAL_HOLIDAYS["New Year's Day"] = function_event(
    lambda x: x.month == 1 and x.day == 1, vectorized=True, name="New Year's Day", dtype=np.bool_,
)


US_FEDERAL_HOLIDAYS["Birthday of Martin Luther King, Jr."] = function_event(
    lambda x: (x.month == 1) & (15 < x.day) & (x.day < 21) & (x.day_name() == "Monday"),
    vectorized=True,
    name="Birthday of Martin Luther King, Jr.",
    dtype=np.bool_,
)

US_FEDERAL_HOLIDAYS["Washington's Birthday"] = function_event(
    US_FEDERAL_HOLIDAYS["Birthday of Martin Luther King, Jr."].function,
    vectorized=True,
    name="Washington's Birthday",
    dtype=np.bool_,
)

US_FEDERAL_HOLIDAYS["Memorial Day"] = function_event(
    lambda x: (x.month == 5) & (24 < x.day) & (x.day_name() == "Monday"),
    vectorized=True,
    name="Memorial Day",
    dtype=np.bool_,
)

US_FEDERAL_HOLIDAYS["Juneteenth National Independence Day"] = function_event(
    lambda x: (x.month == 6) & (x.day == 19),
    vectorized=True,
    name="Juneteenth National Independence Day",
    dtype=np.bool_,
)

US_FEDERAL_HOLIDAYS["Independence Day"] = function_event(
    lambda x: (x.month == 7) & (x.day == 4), vectorized=True, name="Independence Day", dtype=np.bool_,
)

US_FEDERAL_HOLIDAYS["Labor Day"] = function_event(
    lambda x: (x.month == 9) & (x.day < 8) & (x.day_name() == "Monday"),
    vectorized=True,
    name="Labor Day",
    dtype=np.bool_,
)

US_FEDERAL_HOLIDAYS["Columbus Day"] = function_event(
    lambda x: (x.month == 10) & (x.day > 7) & (x.day < 15) & (x.day_name() == "Monday"),
    vectorized=True,
    name="Columbus Day",
    dtype=np.bool_,
)

US_FEDERAL_HOLIDAYS["Veterans Day"] = function_event(
    lambda x: (x.month == 11) & (x.day == 11), vectorized=True, name="Veterans Day", dtype=np.bool_,
)

US_FEDERAL_HOLIDAYS["Thanksgiving Day"] = function_event(
    lambda x: (x.month == 11) & (x.day > 21) & (x.day < 29) & x.day_name() == "Thursday",
    vectorized=True,
    name="Thanksgiving Day",
    dtype=np.bool_,
)

US_FEDERAL_HOLIDAYS["Christmas Day"] = function_event(
    lambda x: (x.month == 12) & (x.day == 25), vectorized=True, name="Christmas Day", dtype=np.bool_,
)

US_OTHER_HOLIDAYS = {}

US_OTHER_HOLIDAYS["Mother's Day"] = function_event(
    lambda x: (x.month == 5) & (x.day > 7) & (x.day < 15) & (x.day_name() == "Sunday"),
    vectorized=True,
    name="Mother's Day",
    dtype=np.bool_,
)

EASTER_DATES_BY_YEAR = {i: easter(i, method=EASTER_WESTERN) for i in range(1583, 4099)}

US_OTHER_HOLIDAYS["Easter"] = function_event(
    lambda x: (EASTER_DATES_BY_YEAR[x.year].month == x.month) & (EASTER_DATES_BY_YEAR[x.year].day == x.day),
    vectorized=False,
    name="Easter",
    dtype=np.bool_,
)

US_OTHER_HOLIDAYS["Father's Day"] = function_event(
    lambda x: (x.month == 6) & (x.day > 14) & (x.day < 22) & (x.day_name() == "Sunday"),
    vectorized=False,
    name="Father's Day",
    dtype=np.bool_,
)

US_OTHER_HOLIDAYS["Halloween"] = function_event(
    lambda x: (x.month == 10) & (x.day == 31), vectorized=False, name="Halloween", dtype=np.bool_,
)

US_OTHER_HOLIDAYS["Valentine's Day"] = function_event(
    lambda x: (x.month == 2) & (x.day == 14), vectorized=False, name="Valentine's Day", dtype=np.bool_,
)

US_OTHER_HOLIDAYS["Saint Patrick's Day"] = function_event(
    lambda x: (x.month == 3) & (x.day == 17), vectorized=False, name="Valentine's Day", dtype=np.bool_,
)
