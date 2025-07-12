import datetime
import locale
import re
from typing import Union
from decimal import Decimal
from pathlib import Path

import pandas as pd

import setup

EXCHANGE_TABLES = {}
TABLE_COLUMNS = lambda x: (
    ["Open", "High", "Low", "Close"]
    if x == 0
    else (["High", "OC_max"] if x > 0 else ["Low", "OC_min"])
)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def wrap_text(s: str, n_chars: int) -> str:
    """Wrap string after at least n_chars, but not in the middle of a word"""
    if not s or n_chars <= 0:
        return s

    # Split on whitespace and hyphens
    tokens = re.split(r"\s|(-)", s)

    # Filter out empty strings that might result from splitting
    tokens = [token for token in tokens if token]

    if not tokens:
        return s

    result = []
    current_line = tokens[0]

    for prev_token, token in zip(tokens[:-1], tokens[1:]):
        # if token is a hyphen, always add it
        if token == "-":
            current_line += "-"
            continue

        # Check if adding the next token would exceed n_chars
        potential_line = current_line + ("" if prev_token == "-" else " ") + token

        if len(potential_line) > n_chars:
            # Only wrap if current line is at least n_chars long
            if len(current_line) >= n_chars:
                result.append(current_line)
                current_line = token
            else:
                # Current line is shorter than n_chars, so add the token anyway
                current_line = potential_line
        else:
            current_line = potential_line

    # Add the last line
    result.append(current_line.strip())

    return "\n".join(result)


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
        self._read_yahoo_finance_csv()

    def _read_yahoo_finance_csv(self):
        for path in Path(self.PATH).glob("*.csv"):
            df = pd.read_csv(path, parse_dates=["Date"])
            df["OC_max"] = df.loc[:, ["Open", "Close"]].max(axis=1)
            df["OC_min"] = df.loc[:, ["Open", "Close"]].min(axis=1)
            self.rates[path.stem] = df

    def exchange_rate_df(
        self, currency: str, base_currency: str = setup.BASE_CURRENCY
    ) -> pd.DataFrame:
        key = f"{base_currency.upper()}{currency.upper()}"
        if key not in self.rates.keys():
            raise KeyError(f"Exchange rate not found: {key}. Download it from Yahoo Finance.")
        return self.rates[key]

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
        ix = (
            (df["Date"] <= date)
            & pd.notna(df["Open"])
            & pd.notna(df["High"])
            & pd.notna(df["Low"])
            & pd.notna(df["Close"])
        )
        if ix.size == 0 or df["Date"].max() < date:
            raise IndexError(
                f"No data for {date} in {base_currency.upper()}{currency.upper()}. "
                "Download current data from Yahoo Finance."
            )
        row = df.loc[ix, :].iloc[-1, :]
        rate = row[TABLE_COLUMNS(sign)].mean()
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
