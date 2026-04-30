# Laufbericht `run_1` – 11.11.2025

## Kurzfassung
- Aufruf: `python ollama_dataset_builder.py --out data --counts ac_host:48,12,12;angry_customer:48,12,12;scorer:40,12,12`
- Dauer: ~17 Minuten (15:06 – 15:23)
- Ergebnis: 208 `chosen`-Samples, 208 Preference-Paare (`prefs.jsonl`)
- Near-Dedupe musste vereinzelt nachgenerieren (insb. `scorer`, `ac_host`), blieb aber im einstelligen Bereich.
- `angry_customer` löste weiterhin Validierungsfehler aus, lieferte nach Fallback jedoch genügend valide Antworten.

## Laufprotokoll und Auffälligkeiten
| Zeit | Persona | Ereignis / Logauszug |
|------|---------|-----------------------|
| 15:06 | ac_host | Start – Ziel: 72 `chosen` |
| 15:08 | ac_host | Near-Dedupe entfernte 2 → Nachgenerierung |
| 15:08 | ac_host | Weitere 1 entfernt → Nachgenerierung |
| 15:11 | angry_customer | Start – Ziel: 72 `chosen` |
| 15:12 | angry_customer | **Warnung:** „nicht 1–2 Sätze, keine Forderung/Frist“ (1x) |
| 15:14 | angry_customer | Near-Dedupe entfernte 3 → Nachgenerierung |
| 15:16 | scorer | Start – Ziel: 64 `chosen` |
| 15:19 | scorer | Near-Dedupe entfernte 14 → Nachgenerierung |
| 15:20 | scorer | Weitere 4 entfernt → Nachgenerierung |
| 15:23 | alle | Export abgeschlossen, Splits + `prefs.jsonl` geschrieben |

Die neue Prompt-Augmentierung funktioniert: Near-Dedupe-Werte sind deutlich kleiner als bei früheren Läufen, dennoch tauchen vereinzelt ähnliche Varianten auf (v. a. bei `ac_host`).

## Dateninspektion

### `ac_host`
- 48/12/12 Splits, insgesamt 72 Ausgaben.
- Tonalität konsistent freundlich und formell; alle Antworten enden mit genau einer Frage.
- Wiederholungen noch erkennbar (häufige Formulierungen „Könnten Sie uns kurz erläutern…“), aber deutlich mehr Variation als zuvor.
- Ein Beispiel (`train` Zeile 9) enthält einen doppelten Zeilenumbruch mit zweiter Frage („Darf ich Ihnen auch ein paar Kekse anbieten?“) – dies verstößt gegen Regel „GENAU eine Folgefrage“. RFO: Prüfer-Validator hat das offenbar akzeptiert; Nachschärfen der Validator-Logik empfehlenswert.

### `angry_customer`
- 48/12/12 Splits.
- Forderungen/Fristen klar vorhanden, Ton sachlich, keine Beleidigungen.
- Einige Outputs nennen Platzhalter („Herr/Frau Meier“) – stilistisch akzeptabel, aber ggf. personalisieren oder neutral formulieren.
- Validierungs-Warnungen traten nur in frühen Versuchen auf; Fallback ohne Negativ-Appendix griff korrekt.

### `scorer`
- 40/12/12 Splits.
- JSON konsistent formatiert, Werte in 0.5-Schritten zwischen 0–5.
- Kommentare variieren, referenzieren die Eingangsantwort sinnvoll.
- Near-Dedupe spielte hier am stärksten – viele ähnliche Bewertungen beim gleichen Input; dennoch ausreichend Varianz nach Nachgeneration.

### Präferenzdaten
- `prefs.jsonl` enthält 208 Paare (für jedes `chosen` ein `rejected`).
- Negativvarianten greifen die Defekt-Labels auf (z. B. „not_json“, „no_actionable“), keine Offensichtlichen Formatfehler festgestellt (Stichprobenprüfung).

## Bewertung & Empfehlungen
1. **Qualität:** Datensätze wirken insgesamt valide, Anforderungen pro Persona werden überwiegend erfüllt. Einzelfälle (AC-Host mit zwei Fragen) sollten noch durch strengere Validatoren abgefangen werden.
2. **Varianz:** Der Prompt-Cache erhöht die Diversität. Für `ac_host` könnten zusätzliche User-Seeds helfen, um wiederkehrende Fragen („Welche Erwartungen…“) weiter zu reduzieren.
3. **Validatoren:** Regex-Liste für `angry_customer` ggf. um weitere Forderungssignale erweitern; außerdem spezifische Prüfung auf mehr als eine Frage bei `ac_host` verschärfen.
4. **Performance:** Laufdauer akzeptabel; Near-Dupe-Schleifen blieben innerhalb weniger Iterationen.

**Fazit:** `run_1` liefert eine solide Basis mit 208 hochwertigen Beispielen. Für kommende Läufe werden Verbesserungen bei den AC-Host-Validatoren und zusätzliche Seeds empfohlen, um die Restduplikate vollständig auszuräumen.*** End Patch*** End Patch

