# 🧠 Synthetic Data Generation for LLM Fine-Tuning (LoRA)

Dieses README beschreibt ausschließlich das Skript `ollama_dataset_builder.py`.  
Der Builder erzeugt vollständig über **Ollama**:

- validierte `chosen`-Beispiele für drei deutschsprachige Personas,
- dazu passende `rejected`-Antworten mit gezielten Defekten,
- train/dev/test-Splits sowie ein gemeinsames Preference-File.
- einen Kurzzeit-Promptspeicher pro Persona, damit neue Antworten sich klar von bereits generierten Outputs unterscheiden.

- 🗣️ **ac_host** – professioneller Assessment-Center-Host  
- 😠 **angry_customer** – verärgerter, aber realistisch reagierender Kunde  
- 🧾 **scorer** – objektiver Bewerter, der strukturierte JSON-Ausgaben liefert

Die generierten Datensätze werden in **Alpaca-Format (JSONL)** ausgegeben und enthalten strukturierte Metadaten, Qualitätsfilterung, semantische Deduplizierung und optionale Präferenzdaten (für DPO/ORPO).

---

## ⚙️ Anforderungen

### 1. Abhängigkeiten
```bash
pip install ollama
```

### 2. Ollama-Modelle

Standardmäßig werden folgende Modelle genutzt (konfigurierbar via CLI):

| Rolle | Modell (Default) | Hinweis |
|-------|------------------|---------|
| **Generator (Teacher)** | `gemma3:27b` | verwendet für alle validierten und negativen Samples |
| **Judge (optional)** | `gemma3:27b` | aktuell im Code nicht aktiv genutzt (s. Schwachstellen) |
| **Embeddings** | `jina/jina-embeddings-v2-base-de` | Near-Dupe-Erkennung via `/api/embeddings` |

Modelle abrufen:
```bash
ollama pull gemma3:27b
ollama pull jina/jina-embeddings-v2-base-de
```

---

## 🚀 Nutzung – `ollama_dataset_builder.py`

```bash
python ollama_dataset_builder.py --out out_builder --counts ac_host:24,6,6
```

### Wichtige Parameter

| Parameter | Beschreibung | Standard |
|-----------|--------------|----------|
| `--out` | Ausgabepfad für Splits und Preference-Paare | `out` |
| `--teacher` | Generierungsmodell (Ollama) | `gemma3:27b` |
| `--judge` | Judge-Modell (derzeit ungenutzt) | `gemma3:27b` |
| `--emb` | Embedding-Modell für Near-Dupe-Filter | `jina/jina-embeddings-v2-base-de` |
| `--seed` | Zufallsseed | `17` |
| `--retry` | Retries pro API-Call mit Backoff | `4` |
| `--tmin` / `--tmax` | Temperaturbereich | `0.4` / `0.9` |
| `--near` | Cosine-Schwelle für semantische Deduplizierung | `0.94` |
| `--counts` | Splitgrößen je Persona `persona:train,dev,test;…` | `ac_host:24,6,6;angry_customer:24,6,6;scorer:20,6,6` |

### Pipeline-Überblick

```text
1. Persona auswählen und User-Template ziehen
2. Zuletzt generierte Outputs der Persona in den Prompt einbetten („Bitte nicht wiederholen“)
3. Validierte CHOSEN-Antwort via Teacher-Model generieren
4. Strenge Persona-Validatoren prüfen Länge, Stil, JSON-Format etc.
5. Deduplizierung (Hash + optional Cosine über Embeddings), ggf. Nachgenerierung bis genug Vielfalt vorliegt
6. Für jedes CHOSEN ein fehlerhaftes REJECTED erzeugen (Defekttyp zufällig, ebenfalls mit Anti-Duplikat-Hinweis)
7. Split in train/dev/test gemäß Counts
8. Export: Persona-Splits + prefs.jsonl mit Preference-Paaren
9. Abschlussbericht im Log
```

### Wiederholungen vermeiden
- Pro Persona wird ein Sliding Window (max. ~60 Outputs) geführt.
- Vor jeder Generierung werden 1–3 bisherige Antworten als Negativbeispiele in den User-Prompt eingefügt.
- Die Modelle werden explizit angewiesen, eine klar unterscheidbare Variante zu liefern.
- Persona-spezifische Hinweise (z. B. „1–2 Sätze, klare Forderung“) halten die strengen Persona-Regeln präsent; wenn alle Versuche scheitern, wird als Fallback ohne Zusatz generiert.
- So sinkt die Zahl der Near-Duplicates deutlich, ohne das Kontextfenster zu überfüllen.

## 💾 Beispielausgabe (ac_host, train-Split)

```json
{
  "instruction": "Du bist ein engagierter Assessment-Center-Host. ...",
  "input": "Ich bin sehr nervös, ist das normal?",
  "output": "Sie haben sich gut vorbereitet, Herr Müller. Wir starten gleich mit einer kurzen Vorstellungsrunde. Was hilft Ihnen in solchen Situationen, wieder ruhig zu werden?",
  "meta": {
    "persona": "ac_host",
    "chosen_meta": {
      "attempt": 1,
      "temperature": 0.4,
      "errors": []
    }
  }
}
```

---

## 🧠 Speicheroptimierung & Performance

Bei 24 GB VRAM kann Ollama beim Wechsel zwischen mehreren Modellen Modelle „entladen“ und neu laden.

### Lösungen:
- `setx OLLAMA_KEEP_ALIVE "30m"` → Modelle bleiben im Speicher  
- `setx OLLAMA_MAX_LOADED_MODELS "3"`  
- `ollama pull gemma3:27b:q4_K_S` → ~17–18 GB VRAM  
- Near-Dupe-Schwelle (`--near`) bei Ressourcenmangel leicht senken oder Embedding-Schritt deaktivieren  
- Anzahl pro Persona (`--counts`) gezielt reduzieren, wenn schnelle Prototypen genügen  

---

## ⚠️ Bekannte Schwachstellen – `ollama_dataset_builder.py`

- **Judge-Modell ungenutzt:** Obwohl ein `judge_model` konfigurierbar ist, wird es im Code nicht aufgerufen. Alle Validierungen erfolgen lokal per Python-Validatoren; externe Qualitätssicherung entfällt.
- **Negativ-Beispiele ohne Verifikation:** Nach dem Erzeugen eines `rejected`-Samples wird nicht geprüft, ob der gewünschte Defekttyp tatsächlich erfüllt ist. Fehlklassifizierte Negatives können in den Daten landen.
- **Eingabediversität weiterhin begrenzt:** Der Anti-Duplikat-Prompt reduziert Wiederholungen, ersetzt aber nicht eine breitere Palette an User-Templates oder dynamisch generierte Seeds.
- **Fehlerfall bei strenger Deduplizierung:** Entfernt die Embedding-basierte Dedupe viele Ausgaben, füllt der Code zwar nach, die Laufzeit steigt aber stark an; ein Abbruch erfolgt erst, wenn `generate_valid` wiederholt scheitert.
- **Keine Nutzung des `instruction`-Feldes im Prompt:** Das Instruktions-Template wird lediglich in den Export übernommen, aber nicht an das Ollama-Modell gesendet. Die Generierung basiert allein auf System- und User-Prompt.

## 🧾 Lizenz und Hinweis
Dieses Skript erzeugt **synthetische Texte** für Forschungs- und Trainingszwecke.  
Achte bei der Weiterverwendung auf:
- keine personenbezogenen Daten in Seeds,
- keine vertraulichen Inhalte,
- Einhaltung der Modell-Lizenzen (Gemma 3, Mistral, Jina).

---

**Autor:** KI-Workshop-Team  
**Version:** 1.3 (November 2025)  
**Kontakt:** synthetic-data@ai-assess.de
