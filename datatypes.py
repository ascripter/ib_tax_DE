from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Union

import pandas as pd

import setup
from utils import ExchangeRate, Singleton


TAX_POT_NAMES = [
    "DIVIDEND",
    "STOCK_GAIN",
    "OTHER_GAIN",
    "OTHER_GAIN_LIMITED",  # Stillhalterprämien und Termingeschäfte
    "STOCK_LOSS",
    "OTHER_LOSS",
    "OTHER_LOSS_LIMITED",  # Termingeschäfte; bis 20k € gem. §20 Abs. 6 S. 5
    "WORTHLESS_LIMITED",  # Wirtschaftsgüter; bis 20k € gem. §20 Abs. 6 S. 6
    # "WITHHOLDING_TAX",
]


# class TaxPot(Enum):
#     STOCK_GAIN = 1
#     STOCK_LOSS = 11
#     OTHER_GAIN = 2
#     OTHER_LOSS = 12
#     OTHER_LOSS_LIMITED = 22
#     WORTHLESS_LIMITED = 23
#     WITHHOLDING_TAX = 3


@dataclass
# class _TaxPots(metaclass=Singleton):
class TaxPots:
    DIVIDEND: float = 0.0
    STOCK_GAIN: float = 0.0
    OTHER_GAIN: float = 0.0
    OTHER_GAIN_LIMITED: float = 0.0  # Stillhalterprämien und Termingeschäfte
    STOCK_LOSS: float = 0.0
    OTHER_LOSS: float = 0.0
    OTHER_LOSS_LIMITED: float = 0.0  # Termingeschäfte; bis 20k € gem. §20 Abs. 6 S. 5
    WORTHLESS_LIMITED: float = 0.0  # Wirtschaftsgüter; bis 20k € gem. §20 Abs. 6 S. 6
    # WITHHOLDING_TAX: float = 0.0  # not implemented

    def __str__(self):
        names = self.__dict__.keys()
        lst = [f"{name:<18s} : {getattr(self, name):>11.2f}" for name in names]
        return "<TaxPots\n  " + "\n  ".join(lst) + ">"

    def __repr__(self):
        return self.__str__()

    def add_trade(self, trade: Trade):
        """Distribute outcome of a trade"""
        net_gain = trade.net_gain()
        if trade.worthless:
            self.WORTHLESS_LIMITED += net_gain
        elif trade.type == ItemType.DIVIDEND_PAYOUT and trade.stock.country != "DE":
            # German dividends are already taxed
            self.DIVIDEND += net_gain
        elif trade.type == ItemType.DIVIDEND_PAYOUT and trade.stock.country == "DE":
            print(f"Excluding dividend payout for German stock: {trade}")
        elif trade.type in ItemType.TAXPOT_LIMITED:
            self.OTHER_GAIN_LIMITED += max(net_gain, 0)
            self.OTHER_LOSS_LIMITED += min(net_gain, 0)
        elif trade.type in ItemType.TAXPOT_OTHER:
            self.OTHER_GAIN += max(net_gain, 0)
            self.OTHER_LOSS += min(net_gain, 0)
        elif trade.type in ItemType.TAXPOT_STOCK:
            self.STOCK_GAIN += max(net_gain, 0)
            self.STOCK_LOSS += min(net_gain, 0)
        else:
            raise ValueError(f"No TaxPot for {trade.type}: {trade}")

    def add_trades(self, trades: TradeList):
        for i, trade in enumerate(trades):
            self.add_trade(trade)

    def KAP_line18(self) -> float:
        """Inländische Kapitalerträge (ohne Betrag lt. Zeile 26)
        (IBKR is a non-german broker, so "Inländische Kapitalerträge" is zero)
        """
        return 0

    def KAP_line19(self) -> float:
        """Ausländische Kapitalerträge (ohne Betrag lt. Zeile 50)"""
        s1 = sum(map(lambda x: getattr(self, x), self.__dict__.keys()))
        s2 = (
            self.DIVIDEND
            + self.STOCK_GAIN
            + self.OTHER_GAIN
            + self.OTHER_GAIN_LIMITED
            + self.STOCK_LOSS  # losses have negative sign, so add them
            + self.OTHER_LOSS
            + self.OTHER_LOSS_LIMITED
            + self.WORTHLESS_LIMITED
        )
        assert s1 == s2
        return round(s1, 2)

    def KAP_line20(self) -> float:
        """In den Zeilen 18 und 19 enthaltene Gewinne aus Aktienveräußerungen
        i. S. d. § 20 Abs. 2 Satz 1 Nr. 1 EStG
        (Anteile einer Körperschaft)
        """
        return round(self.STOCK_GAIN, 2)

    def KAP_line21(self) -> float:
        """In den Zeilen 18 und 19 enthaltene Einkünfte aus Stillhalterprämien
        und Gewinne aus Termingeschäften
        (entspricht vermutlich §20 Abs 1. Nr. 11 (Stillhalterprämien) und
        §20 Abs. 2 Nr. 3 (steuerliche Termingeschäfte))
        Unklar: Zertifikate, Optionsscheine etc.?
        """
        return round(self.OTHER_GAIN + self.OTHER_GAIN_LIMITED, 2)

    def KAP_line22(self) -> float:
        """In den Zeilen 18 und 19 enthaltene Verluste ohne Verluste
        aus der Veräußerung von Aktien
        """
        return -round(self.OTHER_LOSS + self.OTHER_LOSS_LIMITED + self.WORTHLESS_LIMITED, 2)

    def KAP_line23(self) -> float:
        """In den Zeilen 18 und 19 enthaltene Verluste aus der Veräußerung
        von Aktien i. S. d. § 20 Abs. 2 Satz 1 Nr. 1 EStG
        """
        return -round(self.STOCK_LOSS, 2)

    def KAP_line24(self) -> float:
        """Verluste aus Termingeschäften gem. §20 Abs. 2 Nr. 3
        (Verluste, für die Verlustanrechnungsbeschränkung nach §20 Abs. 6 S. 5 gilt,
        nur verrechenbar mit §20 Abs 1. Nr. 11 (Stillhalterprämien) und
        §20 Abs. 2 Nr. 3 (Gewinne aus steuerlichen Termingeschäften))"""
        return -round(self.OTHER_LOSS_LIMITED, 2)

    def KAP_line25(self) -> float:
        """Verluste aus der ganzen oder teilweisen Uneinbringlichkeit einer
        Kapitalforderung, Ausbuchung, Übertragung wertlos gewordener Wirtschaftsgüter
        i. S. d. § 20 Abs. 1 EStG oder aus einem sonstigen Ausfall
        von Wirtschaftsgütern i. S. d. § 20 Abs. 1 EStG
        (Verluste, für die Verlustanrechnungsbeschränkung nach §20 Abs. 6 S. 6 gilt)
        """
        return -round(self.WORTHLESS_LIMITED, 2)

    def KAP_lines(self) -> dict[int, float]:
        result = {}
        for i in range(18, 26):
            result[i] = getattr(self, f"KAP_line{i}")()
        return result


# TaxPots = _TaxPots()  # instantiate Singleton class


class ItemType(Enum):
    UNDEFINED = -1
    STOCK_LONG = 1
    STOCK_SHORT = 51
    OPTION_LONG_CLOSED = 2
    OPTION_LONG_CALL_EXERCISED = 3  # or assigned
    OPTION_LONG_PUT_EXERCISED = 4  # or assigned
    OPTION_LONG_CALL_EXERCISED_CASH_SETTLED = 13  # or assigned
    OPTION_LONG_PUT_EXERCISED_CASH_SETTLED = 14  # or assigned
    OPTION_LONG_EXPIRED = 5
    OPTION_SHORT_CLOSED = 52
    OPTION_SHORT_CALL_EXERCISED = 53  # or assigned
    OPTION_SHORT_PUT_EXERCISED = 54  # or assigned
    OPTION_SHORT_CALL_EXERCISED_CASH_SETTLED = 63  # or assigned
    OPTION_SHORT_PUT_EXERCISED_CASH_SETTLED = 64  # or assigned
    OPTION_SHORT_EXPIRED = 55
    CFD_LONG = 6
    CFD_SHORT = 56
    WARRANT_LONG = 7
    STRUCTURED_LONG = 8
    ETF_LONG = 9
    DIVIDEND_PAYOUT = 101

    # following types not implemented (not traded)
    @classmethod
    @property
    def TAXPOT_STOCK(cls) -> list[ItemType]:
        """All ItemTypes that put losses into OTHER_LOSS"""
        return [cls.STOCK_LONG, cls.STOCK_SHORT]

    @classmethod
    @property
    def TAXPOT_OTHER(cls) -> list[ItemType]:
        """All ItemTypes that put losses into OTHER_LOSS"""
        return [
            cls.OPTION_SHORT_CLOSED,
            cls.OPTION_SHORT_CALL_EXERCISED,
            cls.OPTION_SHORT_PUT_EXERCISED,
            cls.OPTION_SHORT_EXPIRED,
            cls.WARRANT_LONG,
            cls.STRUCTURED_LONG,
            cls.ETF_LONG,
        ]
        # excluded = [cls.STOCK_LONG, cls.STOCK_SHORT, cls.DIVIDEND_PAYOUT] + cls.TAXPOT_LIMITED
        # return [_ for _ in cls if _ not in excluded]

    @classmethod
    @property
    def TAXPOT_LIMITED(cls) -> list[ItemType]:
        """All ItemTypes that put losses into OTHER_LOSS_LIMITED"""
        result = cls.OPTION_CASH_SETTLED
        result.extend(
            [cls.OPTION_LONG_CLOSED, cls.OPTION_LONG_EXPIRED, cls.CFD_LONG, cls.CFD_SHORT]
        )
        return result

    @classmethod
    @property
    def TRADE(cls) -> list[ItemType]:
        return [_ for _ in cls if _.value <= 100]

    @classmethod
    @property
    def OTHER(cls) -> list[ItemType]:
        return [_ for _ in cls if _.value > 100]

    @classmethod
    @property
    def LONG(cls) -> list[ItemType]:
        return [_ for _ in cls if _.value <= 50]

    @classmethod
    @property
    def SHORT(cls) -> list[ItemType]:
        return [_ for _ in cls if 50 <= _.value <= 100]

    @classmethod
    @property
    def OPTION(cls) -> list[ItemType]:
        return [_ for _ in cls if _.name.startswith("OPTION_")]

    @classmethod
    @property
    def OPTION_LONG(cls) -> list[ItemType]:
        return [_ for _ in cls if _.name.startswith("OPTION_LONG")]

    @classmethod
    @property
    def OPTION_SHORT(cls) -> list[ItemType]:
        return [_ for _ in cls if _.name.startswith("OPTION_SHORT")]

    @classmethod
    @property
    def OPTION_EXERCISED(cls) -> list[ItemType]:
        return [_ for _ in cls if "OPTION" in _.name and "EXERCISED" in _.name]

    @classmethod
    @property
    def OPTION_CASH_SETTLED(cls) -> list[ItemType]:
        return [_ for _ in cls if "OPTION" in _.name and "CASH_SETTLED" in _.name]

    def is_long(self):
        return self in self.LONG

    def is_short(self):
        return self in self.SHORT


@dataclass
class Stock:
    name: str = None
    country: str = None
    isin: str = None
    symbol: str = None

    def __str__(self):
        country = f" ({self.country})" if self.country is not None else ""
        if self.symbol is not None:
            return f"{self.symbol}{country}"
        if self.isin is not None:
            return self.isin
        return f"{self.name}{country}"

    def __repr__(self):
        return f"<{self.__str__()}>"


@dataclass
class _Item:
    """Single item / line in the csv file's Trade section"""

    stock: Stock
    timestamp: datetime.datetime
    price: float
    size: float
    type: ItemType = ItemType.UNDEFINED
    commission: float = 0  # negative
    currency: str = "EUR"
    codes: list[str] = field(default_factory=list)  # IBKR Codes
    symbol_option: str = None
    option_type: str = None
    option_date: datetime.date = None
    option_strike: float = None
    interest: float = 0  # Margin interest paid (negative; in base currency)
    worthless: bool = False  # whether this is a worthless item (separate tax pot)

    def __str__(self):
        code = self.get_code()
        proc = self.proceeds(False)
        a, b = self._str_base()
        try:
            gain = f" => {self.net_gain()} {setup.BASE_CURRENCY}"
        except NotImplementedError:
            gain = ""
        return f"{proc:>9.2f} {self.currency} {a}  {code} {b}{gain}"

    def __repr__(self):
        return f"<{self.__str__()}>"

    def get_code(self):
        if "C" in self.codes and "O" in self.codes:
            code = f"Closing/Open {self.__class__.__name__}"
        elif "C" in self.codes:
            code = f"Closing {self.__class__.__name__}"
        elif "O" in self.codes:
            code = f"Open {self.__class__.__name__}"
        else:
            code = self.__class__.__name__
        if "A" in self.codes:
            code = f"Assignment {code}"
        return code

    def _str_base(self):
        symbol = self.symbol_option if self.type in ItemType.OPTION else self.stock.symbol
        return (
            f"{symbol:5s} on {self.timestamp.strftime('%Y-%m-%d')}:",
            f"{self.type.name:<12s}  {self.size:>6.1f} x {self.price:>8.3f} {self.commission:+.2f}",
        )

    @classmethod
    def _item_type_from_row(cls, row):
        """Defer trade type from *Closing Trade* defined in row"""
        codes = row["Code"].split(";")
        if "C" not in codes:
            return ItemType.UNDEFINED
        ls = "LONG" if row["Quantity"] < 0 else "SHORT"
        if row["Asset Category"] == "Equity and Index Options":
            cp = "CALL" if row["Symbol"][-1] == "C" else "PUT"
            if "Ex" in codes or "A" in codes:
                return getattr(ItemType, f"OPTION_{ls}_{cp}_EXERCISED")
            closing = "EXPIRED" if "Ep" in codes else "CLOSED"
            return getattr(ItemType, f"OPTION_{ls}_{closing}")
        elif row["Asset Category"] == "Stocks":
            return getattr(ItemType, f"STOCK_{ls}")
        elif row["Asset Category"] == "Structured Products":
            return getattr(ItemType, f"STRUCTURED_{ls}")
        elif row["Asset Category"] == "Warrants":
            return getattr(ItemType, f"WARRANT_{ls}")
        raise ValueError("Unexpected row: ", row)
        # todo: CFD, ETF, cash settled Option

    @classmethod
    def from_row(cls, row):
        codes = row["Code"].split(";")
        tp = cls._item_type_from_row(row)
        commission = row["Comm/Fee"] if pd.notna(row["Comm/Fee"]) else 0
        worthless = {
            "Stocks": False,
            "Equity and Index Options": False,
            "Warrants": ("Ex" in codes or "Ep" in codes) and row["T. Price"] == 0,
            "Structured Products": ("Ex" in codes or "Ep" in codes) and row["T. Price"] == 0,
        }[row["Asset Category"]]
        return cls(
            stock=Stock(symbol=row["Symbol"]),
            timestamp=row["Date/Time"],
            price=row["T. Price"],
            size=row["Quantity"],
            type=tp,
            commission=commission,
            currency=row["Currency"],
            codes=codes,
            symbol_option=row["symbol_option"],
            option_date=row["option_date"],
            option_strike=row["option_strike"],
            option_type=row["option_type"],
            interest=0,
            worthless=worthless,
        )

    def proceeds(self, base_currency: bool = True, size: float = None) -> float:
        """Change on account balance = Anschaffungskosten oder Veräußerungsertrag.
        If size is None, this Item's proceeds will be returned, otherwise size can
        be overridden i.e. by the parent Trade of a ClosedLot.
        """
        lotsize = 100.0 if self.type in ItemType.OPTION else 1.0
        size = -self.size if size is None else size

        amount = size * self.price * lotsize + self.commission
        if base_currency:
            amount_base = ExchangeRate.convert(amount, self.timestamp, self.currency, amount)
            return amount_base + self.interest

        inter_cur = ExchangeRate.convert_back(self.interest, self.timestamp, self.currency, amount)
        return amount + inter_cur

    def net_gain(self):
        raise NotImplementedError


@dataclass
class ClosedLot(_Item):
    """Single item / line in the csv file's Trade section
    DataDiscriminator = ClosedLot"""

    trade: Trade = None

    @property
    def type(self):
        return ItemType.UNDEFINED if self.trade is None else self.trade.type

    @type.setter
    def type(self, value):
        pass


@dataclass
class Order(_Item):
    """Single item / line in the csv file's Trade section
    DataDiscriminator = Order"""

    trades: list[Trade] = field(default_factory=list)

    def validate(self):
        trades_size = 0
        for trade in self.trades:
            trade.validate()
            trades_size += trade.size
        if trades_size != self.size:
            raise ValueError(f"Size mismatch: {trades_size} != {self.size} (Trades vs. Order)")


@dataclass
class Trade(_Item):
    """Single item / line in the csv file's Trade section
    DataDiscriminator = Trade,
    but with associated closedLots"""

    order: Order = None
    closed_lots: list[ClosedLot] = field(default_factory=list)

    def validate(self):
        if self.type.value > 100:
            if self.closed_lots:
                raise TypeError(
                    f"ItemType {self.type} should not have closed_lots, Trade {str(self)}"
                )
            return
        closed_lots_size = 0
        for lot in self.closed_lots:
            closed_lots_size += lot.size
        if closed_lots_size != -self.size and not "O" in self.codes:
            raise ValueError(
                f"Size mismatch: {closed_lots_size} != {-self.size} (closedLots vs. Trade) in {str(self)}"
            )
        elif closed_lots_size != -self.size and "C" in self.codes:
            print(
                f"imbalanced closing trade (only {closed_lots_size} from {-self.size} closed: {str(self)})"
            )

    def trade_weight(self) -> float:
        """A 'weight' of the trade measured in base_currency trade size
        times number of days the trade was open. Used for accruing interest
        to that trade.
        """
        td = list(map(lambda x: (self.timestamp - x.timestamp), self.closed_lots))
        td = [_.total_seconds() / 24 / 3600 for _ in td]
        cap_base = list(map(lambda x: x.proceeds(), self.closed_lots))
        return sum(map(lambda x, y: x * y, td, cap_base))

    def size_closed(self) -> float:
        """Size of the Trade that was closed"""
        if self.type not in ItemType.TRADE:
            # special item types that aren't actual trades
            return self.size
        return sum(map(lambda x: x.size, self.closed_lots))

    def proceeds_closed(self, base_currency: bool = True) -> float:
        """Only the fraction of lots that were closed by this trade are tax relevant"""
        return self.proceeds(base_currency, self.size_closed())

    def proceeds_open(self, base_currency: bool = True) -> float:
        """Sum of all associated open position that were closed"""
        return sum(map(lambda x: x.proceeds(base_currency), self.closed_lots))

    def net_gain(self, base_currency: bool = True) -> float:
        return self.proceeds_closed(base_currency) + self.proceeds_open(base_currency)


class TradeList(list):
    def __str__(self):
        return "\n".join([str(t) for t in self])

    def __repr__(self):
        return f"<TradeList: {len(self)} trades>"

    def sort_by_date_asc(self):
        self.sort(key=lambda x: x.timestamp)

    def trade_weight_total(self):
        """Total base_currency amount times days of exposed capital"""
        return sum(map(lambda x: x.trade_weight(), self))

    def for_order(self, order: Order):
        return [t for t in self if t.order == order]

    def by_type(self, item_type: Union[ItemType, list[ItemType]]):
        if isinstance(item_type, list):
            return [t for t in self if t.type in item_type]
        else:
            return [t for t in self if t.type == item_type]

    def by_symbol(self, symbol: str, timestamp: datetime.datetime = None):
        timestamp = datetime.datetime(1970, 1, 1) if timestamp is None else timestamp
        result = []
        for i, trade in enumerate(self):
            if trade.stock.symbol == symbol and trade.timestamp >= timestamp:
                result.append(trade)
        return result

    def dividends_by_country(self):
        result = {}
        for item in self.by_type(ItemType.DIVIDEND_PAYOUT):
            if item.stock.country not in result.keys():
                result[item.stock.country] = 0
            result[item.stock.country] += ExchangeRate.convert(
                item.close.price,
                item.close.timestamp,
                item.close.currency,
                item.close.price,
            )
        return result

    def distribute_interest(self, interest: float):
        """Get accrued interest for *all* trades as argument
        and distribute them weighed by allocated capital to each trade
        """
        weight_total = self.trade_weight_total()
        for trade in self:
            if trade.type.value > 100:
                continue
            wt = trade.trade_weight() / weight_total
            trade.interest = wt * interest

    def to_df(self):
        rows = []
        for trade in self:
            row = {
                "symbol": trade.stock.symbol,
                "country": trade.stock.country,
                "option": re.sub(r"^\w+\b\s+", "", trade.symbol_option)
                if trade.symbol_option is not None
                else "",
                "code": trade.get_code(),
                "trade_type": trade.type.name,
                "currency": trade.currency,
                "timestamp": trade.timestamp,
                "exchange_rate": ExchangeRate.at(
                    trade.timestamp, trade.currency, trade.proceeds_closed()
                ),
                "size": trade.size_closed(),
                "price": trade.price,
                "commission": trade.commission,
                f"interest_{setup.BASE_CURRENCY}": trade.interest,
                f"proceeds_{setup.BASE_CURRENCY}": trade.proceeds_closed(),
                f"proceeds_{setup.BASE_CURRENCY}_open": trade.proceeds_open(),
                f"net_gain_{setup.BASE_CURRENCY}": trade.net_gain(),
            }
            for i, cl in enumerate(trade.closed_lots):
                row[f"timestamp_open{i:02d}"] = cl.timestamp
                row[f"exchange_rate_open{i:02d}"] = ExchangeRate.at(
                    cl.timestamp, cl.currency, cl.proceeds()
                )
                row[f"size_open{i:02d}"] = cl.size
                row[f"price_open{i:02d}"] = cl.price
                row[f"commission_open{i:02d}"] = cl.commission
                row[f"proceeds_{setup.BASE_CURRENCY}_open{i:02d}"] = cl.proceeds()

            rows.append(row)
        df = pd.DataFrame.from_records(rows)
        return df

    def line19(self):
        """Get Result German Tax report line 19"""
        pass
