import datetime
from typing import Optional

import pandas as pd
from hypothesis import assume
from hypothesis.strategies import composite, dates, times, integers, timedeltas

MIN_PANDAS_DATE = pd.Timestamp.min.to_pydatetime().date()
MAX_PANDAS_DATE = pd.Timestamp.max.to_pydatetime().date()


@composite
def date_range_indexes(draw, start: Optional[str] = None, end: Optional[str] = None, periods: Optional[str] = None,
                       freq: Optional[str] = None,
                       max_periods: int = 1000, max_freq_magnitude=datetime.timedelta(days=10)):

    if start is None:
        start_date = draw(dates(MIN_PANDAS_DATE + datetime.timedelta(days=1),
                                MAX_PANDAS_DATE - datetime.timedelta(days=1)))
        start_time = draw(times())
        start = datetime.datetime.combine(start_date, start_time)

    if end is not None:
        raise NotImplementedError("Currently, only start, periods, and freq are supported.")

    if periods is None:
        periods = draw(integers(min_value=1, max_value=max_periods))

    if freq is None:
        freq = draw(timedeltas(min_value=-max_freq_magnitude, max_value=max_freq_magnitude))

    assume(freq)
    try:
        return pd.date_range(start=start, periods=periods, freq=freq)
    except pd.errors.OutOfBoundsDatetime:
        assume(False)