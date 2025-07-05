# ib_tax_DE
Berechnung KAP-Zeilen der deutschen Steuererklärung aus Activity Statement von Interactive Brokers.

## How to Use
- Das eigene Activity Statements bei IB für das komplette Steuerjahr in CSV- und HTML-Format herunterladen.
- Zur Währungsumrechnung werden EOD-Kurse von Yahoo Finance verwendet. Diese müssen manuell heruntergeladen und in `data_forex/` abgelegt werden (s.a. `data_forex/_note.txt`).
- Angabe des Pfads zu CSV- und HTML-Statement in `setup.py` für das betreffende Steuerjahr
- Anpassung von `main.py` (Instanzierung von `TaxReportIBKR` mit aktuellem Steuerjahr und optional erstellen von *dividends* und *trades* Tabellen)
- Ausführung von `main.py`

## Disclaimer
Keine Garantie für Korrektheit der Berechnungen. Getestet nur für die von mir gehandelten Instrumente (Einzelaktien, Standardized Options (CBOE), deutsche Hebelzertifikate).
Ich habe das geltende Steuerrecht nach bestem Wissen und Gewissen implementiert. Letztes Update 08.2024.

## License
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en)
- Free to share, use and modify
- No commercial distribution
