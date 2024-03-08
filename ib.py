# Notes: Tested for Stocks (long / short)
# short options with either closing trade, expiry or assignment / exercise
# no long options, no cash settled options (not traded; no data)


import re
from decimal import Decimal

import pandas as pd

from bs4 import BeautifulSoup
from io import StringIO
import sys

import setup
from datatypes import ItemType, Stock, ClosedLot, Trade, Order, TradeList, TaxPots
from utils import parse_date


class TaxReportIBKR:
    def __init__(self, tax_year: int, account: str = setup.ACCOUNT):
        self.html_file = setup.HTML_REPORT[tax_year]
        self.csv_file = setup.CSV_REPORT[tax_year]
        self.account = account
        self.tax_year = tax_year
        self.dividends = TradeList()
        self.trades = TradeList()
        self._trades_assigned_open = TradeList()
        self.tax_pots = TaxPots()
        self.kap_lines = None
        self._read_html()
        self._read_csv()

    def _read_html(self):
        with open(self.html_file, "r") as f:
            content = f.read()
        self.soup = BeautifulSoup(content, "html.parser")

    def _read_csv(self):
        # introduce fake columns in top row
        with open(self.csv_file, "r") as f:
            content = f.read()
        content = "," * 20 + "\n" + content
        self.csv = pd.read_csv(
            StringIO(content),
            dtype="object",
            header=None,
            usecols=[_ for _ in range(17)],
        )

    def _csv_to_df_base(self, section: str) -> pd.DataFrame:
        df0 = self.csv.loc[self.csv[0] == section]
        header = df0.loc[df0[1] == "Header", :].values[0]
        df = df0.loc[df0[1] == "Data", 2:].copy()
        df.columns = header[2:]
        df.dropna(axis=1, how="all", inplace=True)
        return df

    # def get_dividends_df(self):
    #     df = self._csv_to_df_base("Dividends")
    #     df["Date"] = df["Date"].astype("datetime64[s]")
    #     df["Currency"] = df["Currency"].apply(lambda x: x.upper())
    #     to_stock = lambda x:
    #     df["Stock"] =
    #     df["ISIN"] = df["ISIN"].apply(lambda x: x.upper()
    def get_dividends(self):
        table = (
            self.soup.find("div", {"id": f"secCombDiv_{self.account}Heading"})
            .find_next_sibling("div")
            .find("table")
        )
        currency = None
        for row in table.find_all("tr"):
            td = row.find_all("td")
            if len(td) == 1 and td[0].text in setup.CURRENCIES:
                # currency section
                currency = td[0].text
            elif len(td) == 3:
                match = re.search(r"^([A-Z1-9.]+)\(((\w{2})\w{10})\)", td[1].text)
                if match:
                    symbol = match.group(1)
                    isin = match.group(2)
                    country = match.group(3)
                else:
                    raise ValueError(f"Couldn't parse ISIN: {td[1].text}")
                stock = Stock(name=td[1].text, symbol=symbol, country=country, isin=isin)
                item = Trade(
                    stock=stock,
                    size=1,
                    type=ItemType.DIVIDEND_PAYOUT,
                    timestamp=parse_date(td[0].text),
                    price=float(td[2].text),
                    currency=currency,
                )
                self.dividends.append(item)
        return self.dividends

    def get_broker_interest_paid_df(self):
        df = self._csv_to_df_base("Broker Interest Paid")
        df["Date"] = df["Date"].astype("datetime64[s]")
        df["Amount"] = df["Amount"].str.replace(",", "").astype(float)
        return df

    def get_broker_interest_paid(self):
        df = self.get_broker_interest_paid_df()
        return df.iloc[-1, :]["Amount"]  # total amount in base currency

    def get_trades_df(self):
        df = self._csv_to_df_base("Trades")
        df["Date/Time"] = df["Date/Time"].astype("datetime64[s]")
        df["Quantity"] = df["Quantity"].str.replace(",", "").astype(float)
        df["T. Price"] = df["T. Price"].str.replace(",", "").astype(float)
        df["Comm/Fee"] = df["Comm/Fee"].astype(float)
        # df["is_processed"] = False  # marker for processed entries
        df["symbol_option"] = None
        df["option_date"] = None
        df["option_strike"] = None
        df["option_type"] = None
        for ix, row in df.iterrows():
            # read option symbol and distribute it to several columns
            match = re.search(
                r"(\w+)\s?(\d{1,2}[A-Za-z]{3}\d{2})\s?(\d+\.\d+)\s?(\w)",
                row["Symbol"],
            )
            if match is not None:
                df.loc[ix, "symbol_option"] = df.loc[ix, "Symbol"]
                df.loc[ix, "Symbol"] = match.group(1)
                df.loc[ix, "option_date"] = parse_date(match.group(2))
                df.loc[ix, "option_strike"] = Decimal(match.group(3))
                df.loc[ix, "option_type"] = match.group(4)
        df["option_date"] = df["option_date"].astype("datetime64[s]")
        df["option_strike"] = df["option_strike"].astype(float)
        df["option_type"] = df["option_type"].astype(str)
        return df

    def _get_trades_base(self, df, categories: list[str]):
        current_order = None
        current_trade = None
        for ix, row in df.loc[df["Asset Category"].isin(categories)].iterrows():
            codes = row["Code"].split(";")
            if row["DataDiscriminator"] == "Order" and "C" in codes:
                current_order = Order.from_row(row)
            elif row["DataDiscriminator"] == "Trade" and "C" in codes:
                current_trade = Trade.from_row(row)
                current_trade.order = current_order
                current_order.trades.append(current_trade)
                self.trades.append(current_trade)
            elif row["DataDiscriminator"] == "Trade":
                # non-closing trade is ignored
                current_trade = None
            elif row["DataDiscriminator"] == "ClosedLot" and current_trade is not None:
                closed_lot = ClosedLot.from_row(row)
                closed_lot.trade = current_trade
                current_trade.closed_lots.append(closed_lot)

    def get_trades(self):
        df = self.get_trades_df()
        self.trades = TradeList()
        self._trades_assigned_open = TradeList()
        categories = ["Warrants", "Structured Products", "Equity and Index Options"]
        self._get_trades_base(df, categories)

        # now link all assigned / exercised stocks to option trades
        # because source table doesn't provide correct closedLot trade prices
        # ("Stillhalterprämie" is added as virtual gain.
        # Keep assigned opened trades in lookup queue
        for ix, row in df.loc[df["Asset Category"] == "Stocks"].iterrows():
            codes = row["Code"].split(";")
            if row["DataDiscriminator"] == "Order" and "A" in codes and "O" in codes:
                current_order = Order.from_row(row)
            elif row["DataDiscriminator"] == "Trade" and "A" in codes and "O" in codes:
                trade = Trade.from_row(row)
                trade.order = current_order
                # if there is an equivalent trade (same strike, date), add size
                found = False
                for t in self._trades_assigned_open:
                    if (
                        t.stock.symbol == trade.stock.symbol
                        and t.timestamp == trade.timestamp
                        and t.price == trade.price
                    ):
                        t.size += trade.size
                        found = True
                if not found:
                    self._trades_assigned_open.append(trade)
        self._trades_assigned_open.sort_by_date_asc()

        # now do all the stocks. stocks with assigned open trades get modified
        current_order = None
        current_trade = None
        for ix, row in df.loc[df["Asset Category"] == "Stocks"].iterrows():
            codes = row["Code"].split(";")
            if row["DataDiscriminator"] == "Order" and "C" in codes:
                current_order = Order.from_row(row)
            elif row["DataDiscriminator"] == "Trade" and "C" in codes:
                current_trade = Trade.from_row(row)
                current_trade.order = current_order
                current_order.trades.append(current_trade)
                self.trades.append(current_trade)
            elif row["DataDiscriminator"] == "Trade":
                # non-closing trade is ignored
                current_trade = None
            elif row["DataDiscriminator"] == "ClosedLot" and current_trade is not None:
                cl = ClosedLot.from_row(row)
                cl.trade = current_trade
                # check if the closed lot fits symbol and timestamp of an assigned open lot
                # if so, adjust price of closed lot

                opens = self._trades_assigned_open
                opens = opens[::-1] if "LI" in codes else opens
                price_size = []
                cl_size = cl.size
                for t in opens:
                    if t.stock.symbol == cl.stock.symbol and t.timestamp == cl.timestamp:
                        if t.size >= cl_size:
                            price_size.append((t.price, cl_size))
                            t.size -= cl_size
                            cl_size = 0
                            break
                        else:
                            price_size.append((t.price, t.size))
                            cl_size -= t.size
                            t.size = 0
                if price_size:
                    cl.price = sum(map(lambda x: x[0] * x[1], price_size)) / cl.size

                current_trade.closed_lots.append(cl)

        self.trades.distribute_interest(self.get_broker_interest_paid())
        return self.trades

    def process(self):
        trades = self.get_trades()
        div = self.get_dividends()
        self.tax_pots.add_trades(trades)
        self.tax_pots.add_trades(div)
        print(self.tax_pots)
        self.kap_lines = self.tax_pots.KAP_lines()
        print("KAP lines: ", self.kap_lines)


if __name__ == "__main__":
    r1 = TaxReportIBKR(2021)
    # r2 = TaxReportIBKR(2022)
    r1.process()
    # r2.process()
