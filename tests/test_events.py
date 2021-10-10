import datetime

import numpy as np
import pandas as pd

from happenings.core import function_event
from happenings.event import MONTHS

JAN_1_2019 = datetime.datetime.strptime("2019-01-01", "%Y-%M-%d")
JAN_1_2020 = datetime.datetime.strptime("2020-01-01", "%Y-%M-%d")
JAN_1_2021 = datetime.datetime.strptime("2021-01-01", "%Y-%M-%d") - datetime.timedelta(days=1)

DATES_2020 = pd.date_range(start=JAN_1_2020, end=JAN_1_2021, freq="D")
DATES_2019 = pd.date_range(start=JAN_1_2019, end=JAN_1_2020, freq="D")

JAN_EVENT = MONTHS["January"]
FEB_EVENT = MONTHS["February"]
JAN_OR_FEB_EVENT = function_event(name="Jan_or_Feb",
                                  function=lambda x: x.month in [1, 2],
                                  vectorized=False)


def test_month_events():
    assert sum(JAN_EVENT.sample_from(dates=DATES_2020)) == 31
    assert sum(FEB_EVENT.sample_from(dates=DATES_2020)) == 29  # Leap Year
    assert sum(FEB_EVENT.sample_from(dates=DATES_2019)) == 28


def test_derived_events():
    assert sum((JAN_EVENT * FEB_EVENT).sample_from(DATES_2019)) == 0
    assert sum((JAN_EVENT + FEB_EVENT).sample_from(DATES_2020)) == 31 + 29


def test_union_events():
    np.testing.assert_array_equal((JAN_EVENT + FEB_EVENT).sample_from(DATES_2020),
                                  JAN_OR_FEB_EVENT.sample_from(dates=DATES_2020))
    np.testing.assert_array_equal((JAN_EVENT | FEB_EVENT).sample_from(DATES_2020),
                                  JAN_OR_FEB_EVENT.sample_from(dates=DATES_2020))
