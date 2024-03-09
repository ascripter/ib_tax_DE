import datetime
import locale
import re
from typing import Union
from decimal import Decimal
from pathlib import Path

import pandas as pd

import setup

EXCHANGE_TABLES = {}
TABLE_COLUMNS = (
    lambda x: ["Open", "High", "Low", "Close"]
    if x == 0
    else (["High", "OC_max"] if x > 0 else ["Low", "OC_min"])
)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def parse_date(date: str):
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date) is not None:
        return datetime.datetime.strptime(date, "%Y-%m-%d")
    elif re.match(r"(\d{1,2})([A-Za-z]{3})(\d{2})", date) is not None:
        locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
        return datetime.datetime.strptime(date, "%d%b%y")
    raise ValueError(f"Couldn't parse date: {date}")


def tax_round(x: float, direction: str = "down") -> int:
    """Tax-conform rounding down to whole values"""
    assert direction in ("up", "down")
    if (direction == "down" and x > 0) or (direction == "up" and x < 0) or x == 0:
        return int(x)
    elif direction == "down" and x < 0:
        return int(x - 1)
    elif direction == "up" and x > 0:
        return int(x + 1)


def converter_decimal(x):
    try:
        return Decimal(x)
    except Exception:
        return None


class _ExchangeRate(metaclass=Singleton):
    PATH: str = "data_forex"

    def __init__(self):
        self.rates = {}
        self.read_yahoo_finance_csv()

    def read_yahoo_finance_csv(self):
        for path in Path(self.PATH).glob("*.csv"):
            df = pd.read_csv(path, parse_dates=["Date"])
            self.rates[path.stem] = df

    def exchange_rate_df(
        self, currency: str, base_currency: str = setup.BASE_CURRENCY
    ) -> pd.DataFrame:
        if currency not in setup.CURRENCIES:
            raise ValueError(f"Currency not supported: {currency}")
        return self.rates[f"{base_currency.upper()}{currency.upper()}"]

    def at(
        self,
        date: datetime.datetime,
        currency: str,
        sign: float = 1,  # default column
        base_currency: str = setup.BASE_CURRENCY,
    ) -> float:
        if currency == base_currency:
            return 1.0
        df = self.exchange_rate_df(currency, base_currency)
        df["OC_max"] = df.loc[:, ["Open", "Close"]].max(axis=1)
        df["OC_min"] = df.loc[:, ["Open", "Close"]].min(axis=1)
        ix = (
            (df["Date"] <= date)
            & pd.notna(df["Open"])
            & pd.notna(df["High"])
            & pd.notna(df["Low"])
            & pd.notna(df["Close"])
        )
        try:
            row = df.loc[ix, :].iloc[-1, :]
            rate = row[TABLE_COLUMNS(sign)].mean()
        except IndexError:
            raise IndexError(f"No data for {date} in {base_currency.upper()}{currency.upper()}")
        return rate

    def convert(
        self,
        amount: Union[float, int],
        date: datetime.datetime,
        currency: str,
        sign: float = 1,  # default column
        base_currency: str = setup.BASE_CURRENCY,
    ) -> float:
        """Convert amount from foreign currency to base currency at given date."""
        if currency == base_currency:
            return amount
        return amount / self.at(date, currency, sign, base_currency)

    def convert_back(
        self,
        amount: Union[float, int],
        date: datetime.datetime,
        currency: str,
        sign: float = 1,  # default column
        base_currency: str = setup.BASE_CURRENCY,
    ) -> float:
        """Convert amount from base currency to foreign currency at given date."""
        if currency == base_currency:
            return amount
        return amount * self.at(date, currency, sign, base_currency)


ExchangeRate = _ExchangeRate()
