from ib import TaxReportIBKR
from datatypes import TaxRule

if __name__ == "__main__":
    r = TaxReportIBKR(2023, TaxRule.DE)
    r.process(save=False)
    # r.dividends.to_df("dividends2023.xlsx")
    # r.trades.to_df("trades2023.xlsx")
