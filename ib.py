# Notes: Tested for Stocks (long / short)
# short options with either closing trade, expiry or assignment / exercise
# no long options, no cash settled options (not traded; no data)


import datetime
import re
from decimal import Decimal
from pathlib import Path
from typing import Optional
import pandas as pd

import bs4
from bs4 import BeautifulSoup
from io import StringIO
import sys

import setup
from datatypes import TaxRule, ItemType, Stock, ClosedLot, Trade, Order, TradeList, TaxCalc
from utils import parse_date, ExchangeRate


class TaxReportIBKR:
    def __init__(self, tax_year: int, tax_rule: TaxRule, account: str = setup.ACCOUNT):
        self.tax_rule: TaxRule = tax_rule
        self.html_file: str = setup.HTML_REPORT[tax_year]
        self.csv_file: str = setup.CSV_REPORT[tax_year]
        self.account: str = account
        self.tax_year: int = tax_year
        self.dividends: Optional[TradeList] = None
        self.trades: Optional[TradeList] = None
        self._trades_assigned_open: Optional[TradeList] = None
        self.tax_calc: TaxCalc = TaxCalc(tax_rule)
        self._read_html()
        self._read_csv()

    def _read_html(self):
        if not Path(self.html_file).exists():
            print("HTML file doesn't exist, skip dividends and withholding tax: ", self.html_file)
            self.soup = None
            return
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

    def _csv_to_df_base(self, section: str) -> Optional[pd.DataFrame]:
        df0 = self.csv.loc[self.csv[0] == section]
        if df0.shape[0] == 0:
            print("Section not found: ", section)
            return None
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
    def _parse_html_table(
        self, div_id: str, data_len: int, itemtype: ItemType, tradelist: TradeList
    ):
        """Parse table under section `div_id` in HTML file.
        Data rows have `data_len` columns of type `itemtype`.
        Each result is appended to `tradelist`.
        Used to fetch dividends and withholding tax sections.
        """
        if self.soup is None:
            return
        table = self.soup.find("div", {"id": div_id}).find_next_sibling("div").find("table")
        currency = None
        for row in table.find_all("tr"):
            td = row.find_all("td")
            if len(td) == 1 and "header-currency" in td[0]["class"]:
                # currency section
                currency = td[0].text
            elif len(td) == data_len:
                match = re.search(r"^([A-Z1-9.]+)\(((\w{2})\w{10})\)", td[1].text)
                if match:
                    symbol = match.group(1)
                    isin = match.group(2)
                    country = match.group(3)
                    stock = Stock(name=td[1].text, symbol=symbol, country=country, isin=isin)
                else:
                    stock = Stock(name=td[1].text)
                    print(
                        f'{div_id}: Couldn\'t parse text in HTML file "{td[1].text}": "{td[2].text} {currency}". '
                        "No info added to stock object."
                    )
                item = Trade(
                    stock=stock,
                    size=1,
                    type=itemtype,
                    timestamp=parse_date(td[0].text),
                    price=float(td[2].text.replace(",", "")),
                    currency=currency,
                )
                tradelist.append(item)

    def get_dividends(self) -> TradeList:
        self.dividends = TradeList(self.tax_rule)

        div_id = f"secCombDiv_{self.account}Heading"
        itemtype = ItemType.DIVIDEND_PAYOUT
        self._parse_html_table(div_id, 3, itemtype, self.dividends)

        div_id = f"secWithholdingTax_{self.account}Heading"
        itemtype = ItemType.WITHHOLDING_TAX
        self._parse_html_table(div_id, 4, itemtype, self.dividends)

        return self.dividends

    def get_transaction_fees_df(self) -> pd.DataFrame:
        df = self._csv_to_df_base("Transaction Fees")
        if df is None:
            return pd.DataFrame()
        df["Date/Time"] = df["Date/Time"].astype("datetime64[s]")
        df["Amount"] = df["Amount"].str.replace(",", "").astype(float)
        return df.loc[df["Asset Category"] != "Total", :].copy()

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

    def _get_trades_base(
        self, df, categories: list[str], apply_trades_assigned_open: bool = False
    ):
        current_order = None
        current_trade = None
        for ix, row in df.loc[df["Asset Category"].isin(categories)].iterrows():
            codes = row["Code"].split(";")
            is_option_open = (row["Asset Category"] == "Equity and Index Options") and "O" in codes
            if row["DataDiscriminator"] == "Order" and ("C" in codes or is_option_open):
                current_order = Order.from_row(row)
            elif row["DataDiscriminator"] == "Trade" and ("C" in codes or is_option_open):
                current_trade = Trade.from_row(row)
                current_trade.order = current_order
                current_order.trades.append(current_trade)
                self.trades.append(current_trade)
            elif row["DataDiscriminator"] == "Trade":
                # other trades are ignored
                current_trade = None
            elif row["DataDiscriminator"] == "ClosedLot" and current_trade is not None:
                cl = ClosedLot.from_row(row)
                cl.trade = current_trade
                current_trade.closed_lots.append(cl)
                if not apply_trades_assigned_open:
                    continue

                opens = self._trades_assigned_open
                opens = opens[::-1] if "LI" in codes else opens
                price_size = []
                cl_size = cl.size
                for t in opens:
                    if t.stock.symbol == cl.stock.symbol and t.timestamp == cl.timestamp:
                        if t.size >= cl_size:
                            price_size.append((t.price, cl_size))
                            t.size -= cl_size
                            break
                        else:
                            price_size.append((t.price, t.size))
                            cl_size -= t.size
                            t.size = 0
                if price_size:
                    cl.price = sum(map(lambda x: x[0] * x[1], price_size)) / cl.size

    def get_trades(self):
        df = self.get_trades_df()

        self.trades = TradeList(self.tax_rule)
        self._trades_assigned_open = TradeList(self.tax_rule)
        categories = ["Warrants", "Structured Products", "Equity and Index Options"]
        self._get_trades_base(df, categories)

        # now link all assigned / exercised stocks to option trades
        # because source table doesn't provide correct closedLot trade prices
        # ("Stillhalterprämie" is added there as virtual gain which violates tax law)
        # Keep assigned opened trades in lookup queue
        current_order = None
        current_trade = None
        for ix, row in df.loc[df["Asset Category"] == "Stocks"].iterrows():
            codes = row["Code"].split(";")
            if row["DataDiscriminator"] == "Order" and "A" in codes and "O" in codes:
                current_order = Order.from_row(row)
            elif row["DataDiscriminator"] == "Trade" and "A" in codes and "O" in codes:
                current_trade = Trade.from_row(row)
                current_trade.order = current_order
                # if there is an equivalent trade (same strike, date), add size
                already_found = False
                for t in self._trades_assigned_open:
                    if (
                        t.stock.symbol == current_trade.stock.symbol
                        and t.timestamp == current_trade.timestamp
                        and t.price == current_trade.price
                    ):
                        t.size += current_trade.size
                        already_found = True
                if not already_found:
                    self._trades_assigned_open.append(current_trade)
        self._trades_assigned_open.sort_by_date_asc()

        # now do all the stocks. stocks with assigned open trades get modified
        self._get_trades_base(df, ["Stocks"], apply_trades_assigned_open=True)

        # if transaction fees were paid, assign them as additional commission
        trans_fees = self.get_transaction_fees_df()
        for ix, row in trans_fees.iterrows():
            dt = row["Date/Time"].date()
            if row["Symbol"] == "":
                print(f"WARNING. Transaction fee has no symbol and is not considered: {row}")
                continue
            found_trade = False
            for t in self.trades.by_symbol(row["Symbol"]):
                for cl in t.closed_lots:
                    if cl.timestamp.date() != dt:
                        continue
                    cl.commission += ExchangeRate.convert(
                        row["Amount"], dt, row["Currency"], row["Amount"]
                    )
                    found_trade = True
            if not found_trade:
                # fallback: try closing trade also
                for t in self.trades.by_symbol(row["Symbol"]):
                    if t.timestamp.date() != row["Date/Time"].date():
                        continue
                    t.commission += ExchangeRate.convert(
                        row["Amount"], dt, row["Currency"], row["Amount"]
                    )
                    found_trade = True
                if not found_trade:
                    print(f"WARNING. Trade to transaction fee could not be found: {row}")

        # finally distribute interest across all trades
        self.trades.distribute_interest(self.get_broker_interest_paid())
        return self.trades

    def process(self):
        trades = self.get_trades()
        div = self.get_dividends()
        self.tax_calc.add_trades(trades)
        self.tax_calc.add_trades(div)
        print(self.tax_calc)


if __name__ == "__main__":
    r = TaxReportIBKR(2022)
    r.process()
    r.trades.to_df("trades2022.xlsx")
