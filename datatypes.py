from __future__ import annotations

import datetime
from dataclasses import dataclass, field, fields
from enum import Enum
import re
from typing import Union, Optional
from pathlib import Path
import pandas as pd

import setup
from utils import ExchangeRate, tax_round, wrap_text


class TaxRule(Enum):
    """Marker for applied tax legislation"""

    DE = 0  # current law (of 2021 onwards)


class ItemType(Enum):
    UNDEFINED = -1
    STOCK_LONG = 1
    OPTION_LONG_CLOSED = 2
    OPTION_LONG_EXERCISED = 3  # or assigned
    OPTION_LONG_EXERCISED_CASH_SETTLED = 4  # or assigned
    OPTION_LONG_EXPIRED = 5
    CFD_LONG = 6
    WARRANT_LONG = 7
    STRUCTURED_LONG = 8
    ETF_LONG = 9
    STOCK_SHORT = 51
    OPTION_SHORT_CLOSED = 52
    OPTION_SHORT_EXERCISED = 53  # or assigned
    OPTION_SHORT_EXERCISED_CASH_SETTLED = 54  # or assigned
    OPTION_SHORT_EXPIRED = 55
    CFD_SHORT = 56
    OPTION_LONG_OPEN = 101
    OPTION_SHORT_OPEN = 102
    DIVIDEND_PAYOUT = 103
    WITHHOLDING_TAX = 104

    # @classmethod
    # @property
    # def TAXPOT_STOCK(cls) -> list[ItemType]:
    #     """All ItemTypes that put losses into OTHER_LOSS"""
    #     return [cls.STOCK_LONG, cls.STOCK_SHORT]
    #
    # @classmethod
    # @property
    # def TAXPOT_OTHER(cls) -> list[ItemType]:
    #     """All ItemTypes that put losses into OTHER_LOSS"""
    #     return [
    #         cls.OPTION_SHORT_CLOSED,
    #         cls.OPTION_SHORT_EXERCISED,
    #         # cls.OPTION_SHORT_PUT_EXERCISED,
    #         cls.OPTION_SHORT_EXPIRED,
    #         cls.WARRANT_LONG,
    #         cls.STRUCTURED_LONG,
    #         cls.ETF_LONG,
    #     ]
    #     # excluded = [cls.STOCK_LONG, cls.STOCK_SHORT, cls.DIVIDEND_PAYOUT] + cls.TAXPOT_LIMITED
    #     # return [_ for _ in cls if _ not in excluded]
    #
    # @classmethod
    # @property
    # def TAXPOT_LIMITED(cls) -> list[ItemType]:
    #     """All ItemTypes that put losses into OTHER_LOSS_LIMITED"""
    #     result = cls.OPTION_CASH_SETTLED
    #     result.extend(
    #         [cls.OPTION_LONG_CLOSED, cls.OPTION_LONG_EXPIRED, cls.CFD_LONG, cls.CFD_SHORT]
    #     )
    #     return result

    def is_round_trade(self) -> bool:
        return self.value <= 100

    def is_open_trade(self) -> bool:
        return self.name.endswith("_OPEN")

    def is_long(self):
        return self.value < 50

    def is_short(self):
        return self.value >= 51 and self.value <= 100

    def is_other(self) -> bool:
        return self.value > 100

    def is_option(self) -> bool:
        return self.name.startswith("OPTION_")

    def is_option_long(self) -> bool:
        return self.name.startswith("OPTION_LONG")

    def is_option_short(self) -> bool:
        return self.name.startswith("OPTION_SHORT")

    def is_option_exercised(self) -> bool:
        return self.name.endswith("_EXERCISED") and "OPTION" in self.name

    def is_option_cash_settled(self) -> bool:
        return "CASH_SETTLED" in self.name and "OPTION" in self.name

    @classmethod
    def from_row(cls, row):
        """Defer trade type from row in csv file"""
        codes = row["Code"].split(";")
        if row["DataDiscriminator"] == "ClosedLot":
            return cls.UNDEFINED
        if "C" in codes and "O" in codes:
            # in this case use only the closing part
            codes.remove("O")

        ls = "LONG"
        if ("C" in codes and row["Quantity"] > 0) or ("O" in codes and row["Quantity"] < 0):
            ls = "SHORT"

        if row["Asset Category"] == "Equity and Index Options":
            # cp = "CALL" if row["Symbol"][-1] == "C" else "PUT"
            if "O" in codes:
                return getattr(cls, f"OPTION_{ls}_OPEN")
            elif "Ex" in codes or "A" in codes:
                return getattr(cls, f"OPTION_{ls}_EXERCISED")
            elif "Ep" in codes:
                return getattr(cls, f"OPTION_{ls}_EXPIRED")
            elif "C" in codes:
                return getattr(cls, f"OPTION_{ls}_CLOSED")
            raise ValueError("Unexpected row: ", row)
        elif row["Asset Category"] == "Stocks":
            return getattr(cls, f"STOCK_{ls}")
        elif row["Asset Category"] == "Structured Products":
            return getattr(cls, f"STRUCTURED_{ls}")
        elif row["Asset Category"] == "Warrants":
            return getattr(cls, f"WARRANT_{ls}")
        raise ValueError("Unexpected row: ", row)
        # todo: CFD, ETF, cash settled Option


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
    is_worthless: bool = False  # whether this is a worthless item (tax pot §20 Abs 6 S 6)

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
        symbol = self.symbol_option if self.type.is_option() else self.stock.symbol
        symbol += f"~{self.stock.country}" if self.stock.country is not None else ""
        return (
            f"{symbol:8s} on {self.timestamp.strftime('%Y-%m-%d')}:",
            f"{self.type.name:<12s}  {self.size:>6.1f} x {self.price:>8.3f} {self.commission:+.2f}",
        )

    @classmethod
    def from_row(cls, row):
        """Generate _Item from a row in the csv file"""
        codes = row["Code"].split(";")
        tp = ItemType.from_row(row)
        commission = row["Comm/Fee"] if pd.notna(row["Comm/Fee"]) else 0
        is_worthless = {
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
            interest=0,  # in base currency (that's avl. in the report)
            is_worthless=is_worthless,
        )

    def proceeds(self, base_currency: bool = True, size: float = None) -> float:
        """Change on account balance = Anschaffungskosten oder Veräußerungsertrag.
        If size is None, this Item's proceeds will be returned, otherwise size can
        be overridden i.e. by the parent Trade of a ClosedLot.
        """
        lotsize = 100.0 if self.type.is_option() else 1.0
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
        if self.type.is_open_trade():
            return -self.size
        elif not self.type.is_round_trade():
            # special item types that aren't actual trades
            return self.size
        # part of the trade that was closed (if not fully)
        return sum(map(lambda x: x.size, self.closed_lots))

    def proceeds_closed(self, base_currency: bool = True) -> float:
        """Only the fraction of lots that were closed by this trade are tax relevant"""
        if self.type.is_open_trade():
            return 0
        return self.proceeds(base_currency, self.size_closed())

    def proceeds_open(self, base_currency: bool = True) -> float:
        """Sum of all associated open position that were closed"""
        if self.type.is_open_trade():
            return self.proceeds(base_currency, -self.size)
        elif self.type.is_option_short():
            # premium was already handled by opening trade
            return 0
        return sum(map(lambda x: x.proceeds(base_currency), self.closed_lots))

    def net_gain(self, base_currency: bool = True) -> float:
        return self.proceeds_closed(base_currency) + self.proceeds_open(base_currency)


class TradeList(list):
    def __init__(self, tax_rule: Optional[TaxRule] = None):
        self.tax_rule = tax_rule
        super().__init__()

    def __str__(self):
        return "\n".join([str(t) for t in self])

    def __repr__(self):
        return f"<TradeList: {len(self)} trades>"

    def __add__(self, other):
        if isinstance(other, TradeList):
            result = TradeList(self.tax_rule)
            result.extend(self)
            result.extend(other)
            return result
        return super().__add__(other)

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
        if symbol == "":
            return []
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

    def to_df(self, excel_filename: str | Path = None, open_lots: bool = True):
        rows = []
        for trade in self:
            row = {
                "symbol": wrap_text(trade.stock.symbol, 20),
                "country": trade.stock.country,
                "option": (
                    re.sub(r"^\w+\b\s+", "", trade.symbol_option)
                    if trade.symbol_option is not None
                    else ""
                ),
                "code": wrap_text(trade.get_code(), 15),
                "trade_type": trade.type.name,
                "currency": trade.currency,
                "timestamp": trade.timestamp,
                "exchange_rate": round(
                    ExchangeRate.at(trade.timestamp, trade.currency, trade.proceeds_closed()), 5
                ),
                "size": trade.size_closed(),
                "price": trade.price,
                "commission": round(trade.commission, 2),
                f"interest_{setup.BASE_CURRENCY}": round(trade.interest, 2),
                f"proceeds_{setup.BASE_CURRENCY}": round(trade.proceeds_closed(), 2),
                f"proceeds_{setup.BASE_CURRENCY}_open": round(trade.proceeds_open(), 2),
                f"net_gain_{setup.BASE_CURRENCY}": round(trade.net_gain(), 2),
                "comment": "",
            }
            if self.tax_rule is not None:
                if not TradeFilter(self.tax_rule).include(trade, True):
                    row["comment"] = "Not tax relevant"

            if not open_lots:
                continue
            for i, cl in enumerate(trade.closed_lots):
                row[f"timestamp_open{i:02d}"] = cl.timestamp
                row[f"exchange_rate_open{i:02d}"] = round(
                    ExchangeRate.at(cl.timestamp, cl.currency, cl.proceeds()), 5
                )
                row[f"size_open{i:02d}"] = cl.size
                row[f"price_open{i:02d}"] = round(cl.price, 2)
                row[f"commission_open{i:02d}"] = round(cl.commission, 2)
                row[f"proceeds_{setup.BASE_CURRENCY}_open{i:02d}"] = round(cl.proceeds(), 2)

            rows.append(row)
        df = pd.DataFrame.from_records(rows)
        if excel_filename:
            df.to_excel(excel_filename, index=False)
        return df


@dataclass
# class _TaxPots(metaclass=Singleton):
class TaxCalc:
    _trade_filter: TradeFilter = field(init=False)
    tax_rule: TaxRule
    DIVIDEND_DE: float = 0.0  # Dividend (and Lieu of Dividend) for German stocks
    DIVIDEND_OTHER: float = 0.0  # Dividend (and Lieu of Dividend) for non-German stocks
    WRITER_PREMIUM: float = 0.0  # Stillhalterprämien
    STOCK_GAIN: float = 0.0
    OTHER_GAIN_LIMITED: float = 0.0  # Termingeschäfte i. S. d. dt. Steuerrechts
    OTHER_GAIN: float = 0.0  # all other instruments
    STOCK_LOSS: float = 0.0
    OTHER_LOSS_LIMITED: float = 0.0  # Termingeschäfte; bis 20k € gem. §20 Abs. 6 S. 5
    OTHER_LOSS: float = 0.0  # all other instruments
    WORTHLESS_LIMITED: float = 0.0  # Wirtschaftsgüter; bis 20k € gem. §20 Abs. 6 S. 6
    WITHHOLDING_TAX_DE: float = 0.0  # W. from dividend payments of German stocks
    WITHHOLDING_TAX_OTHER: float = 0.0  # W. from dividend payments of non-German stocks
    WITHHOLDING_TAX_OTHER_trades: TradeList = field(default_factory=TradeList)
    DIVIDEND_DE_trades: TradeList = field(default_factory=TradeList)
    DIVIDEND_OTHER_trades: TradeList = field(default_factory=TradeList)
    WRITER_PREMIUM_trades: TradeList = field(default_factory=TradeList)
    STOCK_GAIN_trades: TradeList = field(default_factory=TradeList)
    OTHER_GAIN_LIMITED_trades: TradeList = field(default_factory=TradeList)
    OTHER_GAIN_trades: TradeList = field(default_factory=TradeList)
    STOCK_LOSS_trades: TradeList = field(default_factory=TradeList)
    OTHER_LOSS_LIMITED_trades: TradeList = field(default_factory=TradeList)
    OTHER_LOSS_trades: TradeList = field(default_factory=TradeList)
    WORTHLESS_LIMITED_trades: TradeList = field(default_factory=TradeList)
    WITHHOLDING_TAX_DE_trades: TradeList = field(default_factory=TradeList)

    def __post_init__(self):
        self._trade_filter = TradeFilter(self.tax_rule)

    def __str__(self):
        def amount_str(v: Union[float, int]):
            return f"{v:>8d}.--" if isinstance(v, int) else f"{v:>11.2f}"

        names = self._float_field_names()
        b = setup.BASE_CURRENCY
        pots = [f"{name:<21s} : {getattr(self, name):>11.2f} {b}" for name in names]
        kap = [
            f"KAP line {str(i):<12s} : {amount_str(v)} {b}  ({self.KAP_lines_annotations[i]})"
            for i, v in self.KAP_lines.items()
        ]
        return "<TaxCalc\n  " + "\n  ".join(pots + ["---"] + kap) + ">"

    def __repr__(self):
        return self.__str__()

    def _float_field_names(self):
        return [f.name for f in fields(self) if f.type == "float"]

    def add_trade(self, trade: Trade):
        self._trade_filter.add_trade(trade, self)

    def add_trades(self, trades: TradeList):
        for i, trade in enumerate(trades):
            self.add_trade(trade)

    def KAP_line18(self) -> int:
        """Inländische Kapitalerträge (ohne Betrag lt. Zeile 26)
        (IBKR is a non-german broker, so "Inländische Kapitalerträge" is zero
        except for dividends from German stocks)
        """
        s = self.DIVIDEND_DE + self.WITHHOLDING_TAX_DE
        return tax_round(s, "down")

    def KAP_line18_trades(self) -> list[Trade]:
        return self.DIVIDEND_DE_trades + self.WITHHOLDING_TAX_DE_trades

    def KAP_line19(self) -> int:
        """Ausländische Kapitalerträge (ohne Betrag lt. Zeile 50)"""
        s1 = sum(map(lambda x: getattr(self, x), self._float_field_names())) - (
            self.DIVIDEND_DE + self.WITHHOLDING_TAX_DE
        )
        s2 = (
            self.DIVIDEND_OTHER
            + self.WRITER_PREMIUM
            + self.STOCK_GAIN
            + self.OTHER_GAIN
            + self.OTHER_GAIN_LIMITED
            + self.STOCK_LOSS  # losses have negative sign, so add them
            + self.OTHER_LOSS
            + self.OTHER_LOSS_LIMITED
            + self.WORTHLESS_LIMITED
            + self.WITHHOLDING_TAX_OTHER
        )
        assert s1 == s2
        return tax_round(s1, "down")

    def KAP_line19_trades(self) -> list[Trade]:
        return (
            self.DIVIDEND_OTHER_trades
            + self.WRITER_PREMIUM_trades
            + self.STOCK_GAIN_trades
            + self.OTHER_GAIN_trades
            + self.OTHER_GAIN_LIMITED_trades
            + self.STOCK_LOSS_trades
            + self.OTHER_LOSS_trades
            + self.OTHER_LOSS_LIMITED_trades
            + self.WORTHLESS_LIMITED_trades
            + self.WITHHOLDING_TAX_OTHER_trades
        )

    def KAP_line20(self) -> int:
        """In den Zeilen 18 und 19 enthaltene Gewinne aus Aktienveräußerungen
        i. S. d. § 20 Abs. 2 Satz 1 Nr. 1 EStG
        (Anteile einer Körperschaft)
        """
        return tax_round(self.STOCK_GAIN, "down")

    def KAP_line20_trades(self) -> list[Trade]:
        return self.STOCK_GAIN_trades

    def KAP_line21(self) -> int:
        """In den Zeilen 18 und 19 enthaltene Einkünfte aus Stillhalterprämien
        und Gewinne aus Termingeschäften
        (entspricht vermutlich §20 Abs 1. Nr. 11 (Stillhalterprämien) und
        §20 Abs. 2 Nr. 3 (steuerliche Termingeschäfte))
        Unklar: Zertifikate, Optionsscheine etc.? -> vermutlich nicht
        """
        return tax_round(self.WRITER_PREMIUM + self.OTHER_GAIN_LIMITED, "down")

    def KAP_line21_trades(self) -> list[Trade]:
        return self.WRITER_PREMIUM_trades + self.OTHER_GAIN_LIMITED_trades

    def KAP_line22(self) -> int:
        """In den Zeilen 18 und 19 enthaltene Verluste ohne Verluste
        aus der Veräußerung von Aktien
        """
        v = -self.OTHER_LOSS - self.OTHER_LOSS_LIMITED - self.WORTHLESS_LIMITED
        return tax_round(v, "up")

    def KAP_line22_trades(self) -> list[Trade]:
        return (
            self.OTHER_LOSS_trades + self.OTHER_LOSS_LIMITED_trades + self.WORTHLESS_LIMITED_trades
        )

    def KAP_line23(self) -> int:
        """In den Zeilen 18 und 19 enthaltene Verluste aus der Veräußerung
        von Aktien i. S. d. § 20 Abs. 2 Satz 1 Nr. 1 EStG
        """
        return tax_round(-self.STOCK_LOSS, "up")

    def KAP_line23_trades(self) -> list[Trade]:
        return self.STOCK_LOSS_trades

    def KAP_line24(self) -> int:
        """Verluste aus Termingeschäften gem. §20 Abs. 2 Nr. 3
        (Verluste, für die Verlustanrechnungsbeschränkung nach §20 Abs. 6 S. 5 gilt,
        nur verrechenbar mit §20 Abs 1. Nr. 11 (Stillhalterprämien) und
        §20 Abs. 2 Nr. 3 (Gewinne aus steuerlichen Termingeschäften))"""
        return tax_round(-self.OTHER_LOSS_LIMITED, "up")

    def KAP_line24_trades(self) -> list[Trade]:
        return self.OTHER_LOSS_LIMITED_trades

    def KAP_line25(self) -> int:
        """Verluste aus der ganzen oder teilweisen Uneinbringlichkeit einer
        Kapitalforderung, Ausbuchung, Übertragung wertlos gewordener Wirtschaftsgüter
        i. S. d. § 20 Abs. 1 EStG oder aus einem sonstigen Ausfall
        von Wirtschaftsgütern i. S. d. § 20 Abs. 1 EStG
        (Verluste, für die Verlustanrechnungsbeschränkung nach §20 Abs. 6 S. 6 gilt)
        """
        return tax_round(-self.WORTHLESS_LIMITED, "up")

    def KAP_line25_trades(self) -> list[Trade]:
        return self.WORTHLESS_LIMITED_trades

    def KAP_line41(self):
        """Noch anzurechnende ausländische Steuern"""
        return round(-self.WITHHOLDING_TAX_OTHER, 2)

    def KAP_line41_trades(self) -> list[Trade]:
        return self.WITHHOLDING_TAX_OTHER_trades

    def KAP_line43(self):
        """Einbehaltene inländische Steuerabzüge: Anzurechnende Kapitalertragssteuer"""
        # total fraction is 1.055 since aggregate value *includes* 5.5% Soli
        # return tax_round(-self.WITHHOLDING_TAX_DE * (1 / 1.055), "up")
        return round(-self.WITHHOLDING_TAX_DE * (1 / 1.055), 2)

    def KAP_line43_trades(self) -> list[Trade]:
        return self.WITHHOLDING_TAX_DE_trades

    def KAP_line44(self):
        """Einbehaltene inländische Steuerabzüge: Anzurechnender Solidaritätszuschlag"""
        # return tax_round(-self.WITHHOLDING_TAX_DE * (0.055 / 1.055), "up")
        return round(-self.WITHHOLDING_TAX_DE * (0.055 / 1.055), 2)

    def KAP_line44_trades(self) -> list[Trade]:
        return self.WITHHOLDING_TAX_DE_trades

    @property
    def KAP_lines(self) -> dict[int, float]:
        result = {}
        for i in list(range(18, 26)) + [41, 43, 44]:
            result[i] = getattr(self, f"KAP_line{i}")()
        return result

    @property
    def KAP_lines_trades(self) -> dict[int, TradeList]:
        result = {}
        for i in list(range(18, 26)) + [41, 43, 44]:
            result[i] = getattr(self, f"KAP_line{i}_trades")()
        return result

    @property
    def KAP_lines_annotations(self) -> dict[int, str]:
        return {
            18: "Inländische Kapitalerträge",
            19: "Ausländische Kapitalerträge",
            20: "In 18 und 19 enth. Gewinne aus Aktienveräußerungen",
            21: "In 18 und 19 enth. Stillhalterprämien und Gewinne aus Termingeschäften",
            22: "In 18 und 19 enth. Verluste (ohne Verluste Aktien)",
            23: "In 18 und 19 enth. Verluste aus Aktien",
            24: "Verluste aus Termingeschäften mit Verlustanrechnungsbeschränkung",
            25: "Verluste aus Uneinbringlichkeit von Kapitalforderungen",
            41: "Noch anzurechnende ausländische Steuern",
            43: "Einbehaltene inländische Steuerabzüge: Anzurechnende Kapitalertragssteuer",
            44: "Einbehaltene inländische Steuerabzüge: Anzurechnender Solidaritätszuschlag",
        }


class TradeFilter:
    """Manager to implement different tax legislation (TaxRule)."""

    def __init__(self, tax_rule: TaxRule):
        self.tax_rule = tax_rule

    def include(self, trade: Trade, silent: bool = False) -> bool:
        if self.tax_rule == TaxRule.DE:
            return True
            # if (
            #     trade.type in (ItemType.DIVIDEND_PAYOUT, ItemType.WITHHOLDING_TAX)
            #     and trade.stock.country == "DE"
            # ):
            #     if not silent:
            #         print(f"{self.tax_rule}: Already taxed => exclude from TaxReport: {trade}")
            #     return False
            # return True
        else:
            raise NotImplementedError()

    def _add_trade_DE(self, trade: Trade, tax_calc: TaxCalc):
        if self.include(trade) is False:
            return
        if trade.is_worthless:
            tax_calc.WORTHLESS_LIMITED += trade.net_gain()
            tax_calc.WORTHLESS_LIMITED_trades.append(trade)
        elif trade.type == ItemType.DIVIDEND_PAYOUT and trade.stock.country == "DE":
            tax_calc.DIVIDEND_DE += trade.net_gain()
            tax_calc.DIVIDEND_DE_trades.append(trade)
        elif trade.type == ItemType.DIVIDEND_PAYOUT and trade.stock.country != "DE":
            tax_calc.DIVIDEND_OTHER += trade.net_gain()
            tax_calc.DIVIDEND_OTHER_trades.append(trade)
        elif trade.type == ItemType.WITHHOLDING_TAX and trade.stock.country == "DE":
            tax_calc.WITHHOLDING_TAX_DE += trade.net_gain()
            tax_calc.WITHHOLDING_TAX_DE_trades.append(trade)
        elif trade.type == ItemType.WITHHOLDING_TAX and trade.stock.country != "DE":
            tax_calc.WITHHOLDING_TAX_OTHER += trade.net_gain()
            tax_calc.WITHHOLDING_TAX_OTHER_trades.append(trade)
        elif trade.type == ItemType.OPTION_SHORT_OPEN:
            tax_calc.WRITER_PREMIUM += trade.net_gain()
            tax_calc.WRITER_PREMIUM_trades.append(trade)
        elif trade.type == ItemType.OPTION_SHORT_CLOSED:
            # only closing half trade, since open was already added to writer premium
            tax_calc.OTHER_LOSS += trade.proceeds_closed()
            tax_calc.OTHER_LOSS_trades.append(trade)
        elif trade.type == ItemType.OPTION_SHORT_EXERCISED_CASH_SETTLED:
            # only closing half trade, since open was already added to writer premium
            tax_calc.OTHER_LOSS_LIMITED += trade.proceeds_closed()
            tax_calc.OTHER_LOSS_LIMITED_trades.append(trade)
            raise NotImplementedError("Option short exercised cash settled not tested")
        elif trade.type in (
            ItemType.OPTION_LONG_EXERCISED_CASH_SETTLED,
            ItemType.OPTION_LONG_CLOSED,
            ItemType.OPTION_LONG_EXPIRED,
            ItemType.CFD_LONG,
            ItemType.CFD_SHORT,
        ):
            # all round trades classified as "Termingeschäfte i. S. d. dt. Steuerrechts"
            tax_calc.OTHER_GAIN_LIMITED += max(trade.net_gain(), 0)
            tax_calc.OTHER_LOSS_LIMITED += min(trade.net_gain(), 0)
            if trade.net_gain() > 0:
                tax_calc.OTHER_GAIN_LIMITED_trades.append(trade)
            elif trade.net_gain() < 0:
                tax_calc.OTHER_LOSS_LIMITED_trades.append(trade)
        elif trade.type == ItemType.OPTION_LONG_EXERCISED:
            raise NotImplementedError("Option long exercised not tested")
            # todo: find corresponding stock trade and add option premium to cost
        elif trade.type in (
            ItemType.WARRANT_LONG,
            ItemType.STRUCTURED_LONG,
            ItemType.ETF_LONG,
        ):
            tax_calc.OTHER_GAIN += max(trade.net_gain(), 0)
            tax_calc.OTHER_LOSS += min(trade.net_gain(), 0)
            if trade.net_gain() > 0:
                tax_calc.OTHER_GAIN_trades.append(trade)
            elif trade.net_gain() < 0:
                tax_calc.OTHER_LOSS_trades.append(trade)
        elif trade.type in (ItemType.STOCK_LONG, ItemType.STOCK_SHORT):
            tax_calc.STOCK_GAIN += max(trade.net_gain(), 0)
            tax_calc.STOCK_LOSS += min(trade.net_gain(), 0)
            if trade.net_gain() > 0:
                tax_calc.STOCK_GAIN_trades.append(trade)
            elif trade.net_gain() < 0:
                tax_calc.STOCK_LOSS_trades.append(trade)
        elif trade.type in (
            ItemType.OPTION_LONG_OPEN,
            ItemType.OPTION_SHORT_EXERCISED,
            ItemType.OPTION_SHORT_EXPIRED,
        ):
            pass  # writer premium already considered; exercise via stock trade
        else:
            raise NotImplementedError(f"No TaxCalc for {trade.type}: {trade}")

    def add_trade(self, trade: Trade, tax_calc: TaxCalc):
        """Distribute outcome of a trade to the buckets of a TaxCalc object."""
        return getattr(self, f"_add_trade_{self.tax_rule.name}")(trade, tax_calc)
