# ib_tax_DE
Berechnung KAP-Zeilen der deutschen Steuererklärung aus Activity Statement von Interactive Brokers.

## How to Use
- Das eigene Activity Statements bei IB für das komplette Steuerjahr in CSV- und HTML-Format herunterladen.
- Zur Währungsumrechnung werden EOD-Kurse von Yahoo Finance verwendet. Diese müssen manuell heruntergeladen und in `data_forex/` abgelegt werden (s.a. `data_forex/_note.txt`).
- Angabe des Pfads zu CSV- und HTML-Statement in `setup.py` für das betreffende Steuerjahr
- Anpassung von `main.py` (Instanzierung von `TaxReportIBKR` mit aktuellem Steuerjahr und optional erstellen von *dividends* und *trades* Tabellen)
- Ausführung von `main.py`

## Implementation Details
Kurzbeschreibung der Funktionsweise: Jede Zeile des Activity Statements wird zunächst in ein `_Item` Objekt umgewandelt, das eine einzelne Transkation widerspiegelt. Dies enthält u.a. Informationen
- über das Instrument (Stock / Option / Warrant)
- Die Art des Transkation (`ItemType`: Open / Close // Bei Options zus. Exercised / Exercised_cash_settled / Expired)
- Timestamp
- Stückzahl und Betrag
- Basiswährung der Transaktion
- Flag, ob das Instrument wertlos verfallen ist (Warrants und Structured Products / Zertifikate)

Mehrere einzelne Transaktionen werden zu einer `Trade`-Instanz zusammengefasst, indem Closed Transactions den entsprechenden Opens zugeordnet werden. Daraus kann berechnet werden:
- Gewinn/Verlust in EUR

Eine `TaxCalc`-Instanz akkumuliert die einzelnen Trades und berechnet in Verbindung mit einer `TradeFilter`-Instanz, welche die eigentliche Implementierung der Regeln des deutschen Steuerrechts. kumulierte Werte aus:
- deutscher Dividende
- ausländischer Dividende
- Stillhalterprämien
- Aktiengewinn
- Gewinn von Termingeschäften i. S. d. dt. Steuerrechts
- Gewinn aller übrigen Finanzinstrumente
- Aktienverlust
- Verlust durch Termingeschäften i. S. d. dt. Steuerrechts
- Verlust aller übrigen Finanzinstrumente
- Wertloser Verfall (von Zertifikaten / Warrants / Structured Products)
- Einbehaltene Quellensteuer deutscher Aktien
- Einbehaltene Quellensteuer ausländischer Aktien

Im letzten Schritt entscheidet die `TradeFilter`-Klasse, welcher akkumulierte Topf welcher KAP-Zeile zugeordnet wird. Dies repräsentiert die eigentliche Implementierung der Regeln des deutschen Steuerrechts. Gleichzeitig werden die zugehörigen Trades zu jedem Topf / jeder KAP-Zeile geloggt und ausgegeben.

## Disclaimer
Keine Garantie für Korrektheit der Berechnungen. Getestet nur für die von mir gehandelten Instrumente (Einzelaktien, Standardized Options (CBOE), deutsche Hebelzertifikate).
Ich habe das geltende Steuerrecht nach bestem Wissen und Gewissen implementiert. Letztes Update 08.2024.

## License
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en)
- Free to share, use and modify
- No commercial distribution
