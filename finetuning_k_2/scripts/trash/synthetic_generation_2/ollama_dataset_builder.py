#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama-only Dataset Builder für drei deutschsprachige Personas
=============================================================

Personas
--------
- ac_host:   Assessment-Center-Host (2 Aussagen + 1 Frage, formell, ≤ 45 Wörter)
- angry_customer: Verärgerter Kunde (1–2 Sätze, klare Forderung/Frist, sachlich, ≤ 35 Wörter, keine Beleidigungen)
- scorer:    JSON-Juror (liefert NUR ein minimiertes JSON mit 5 Scores in 0..5, Schrittweite 0.5, plus kurzer Kommentar)

WICHTIG: Ausschließlich Ollama wird verwendet — KEINE OpenAI-Anbindung.

Modelle (per Vorgabe des Nutzers)
----------------------------------
DEFAULT_TEACHER_MODEL = "gemma3:27b"      # Synthetische Generierung
DEFAULT_JUDGE_MODEL   = "gemma3:27b"      # QA/Judging
EMB_MODEL             = "jina/jina-embeddings-v2-base-de"  # Embeddings via Ollama /api/embeddings

Benötigt: laufender Ollama-Dienst (Standard: http://localhost:11434)
- Modelle einmalig ziehen:
    ollama pull gemma3:27b
    ollama pull jina/jina-embeddings-v2-base-de

Funktionen/Features
-------------------
- Saubere Prompt-Schablonen (system + user) je Persona
- Erzeugung valider "chosen"-Beispiele und gezielter "rejected"-Gegenbeispiele (pro Defektkategorie)
- Strenge Validatoren je Persona (Länge, Sätze, Frage, Formalität, Profanität, JSON-Schema etc.)
- Deduplizierung: text-normalisiert + (optional) Embedding-basierte Near-Duplicate-Erkennung via Ollama /api/embeddings
- Robustes Retry/Backoff für API-Aufrufe, deterministische Seeds
- Reproduzierbare Splits: train/dev/test
- Export: JSONL-Dateien pro Persona + Preference-Paare (chosen/rejected)
- Ausführliches Run-Log + kompakter Abschlussbericht

Externe Abhängigkeiten: keine (nur Python-Stdlib). HTTP via urllib.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import functools
import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# --------------------------- Konstante Vorgaben ----------------------------
DEFAULT_TEACHER_MODEL = "gemma3:27b"
DEFAULT_JUDGE_MODEL   = "gemma3:27b"
EMB_MODEL             = "jina/jina-embeddings-v2-base-de"
OLLAMA_HOST           = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# --------------------------- Utilities ------------------------------------

def log(msg: str) -> None:
    ts = dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def stable_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()


def normalize_text(s: str) -> str:
    s2 = re.sub(r"\s+", " ", s.strip())
    return s2


def count_words(s: str) -> int:
    return len(re.findall(r"\b\w+\b", s, flags=re.UNICODE))


def sentence_tokens(s: str) -> List[str]:
    toks = [t.strip() for t in re.split(r"(?<=[\.!\?])\s+", s.strip()) if t.strip()]
    return toks


def ends_with_question(s: str) -> bool:
    return s.strip().endswith("?")


def only_one_question_mark(s: str) -> bool:
    return s.count("?") == 1


def approx_formal_address(s: str) -> bool:
    return bool(re.search(r"\b(Sie|Ihnen|Ihr(?:e|en|er|em)?)\b", s))


def contains_profanity(s: str) -> bool:
    bad = [
        "idiot", "blöd", "scheiß", "arsch", "unfähig", "versager", "dumm", "verdammt",
    ]
    low = s.lower()
    return any(b in low for b in bad)


def within_range(v: float, lo: float, hi: float) -> bool:
    return lo <= v <= hi


def is_half_step(v: float) -> bool:
    return abs(v*2 - round(v*2)) < 1e-6


# --------------------------- Datenstrukturen -------------------------------
@dataclass
class Sample:
    persona: str
    instruction: str
    user: str
    output: str
    meta: Dict[str, Any]


@dataclass
class SplitCounts:
    train: int
    dev: int
    test: int


def default_persona_counts() -> Dict[str, SplitCounts]:
    return {
        "ac_host": SplitCounts(train=24, dev=6, test=6),
        "angry_customer": SplitCounts(train=24, dev=6, test=6),
        "scorer": SplitCounts(train=20, dev=6, test=6),
    }


@dataclass
class GenConfig:
    teacher_model: str = DEFAULT_TEACHER_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL
    emb_model: str = EMB_MODEL
    max_tokens: int = 180
    temperature_min: float = 0.4
    temperature_max: float = 0.9
    top_p: float = 1.0
    retries: int = 4
    backoff_base: float = 1.8
    seed: int = 17
    out_dir: str = "out"
    persona_counts: Dict[str, SplitCounts] = dataclasses.field(default_factory=default_persona_counts)
    near_dup_threshold: float = 0.94  # Cosine-Sim Schwelle; nur wenn Embeddings verfügbar


# --------------------------- Prompts --------------------------------------
SYSTEM_TEMPLATES: Dict[str, str] = {
    "ac_host": (
        "Du bist ein engagierter Assessment-Center-Host. "
        "Antwortstil: warm, professionell, knapp. "
        "Erzeuge GENAU zwei kurze Sätze (ohne Frage) und anschließend GENAU eine Folgefrage. "
        "Gesamtlänge ≤ 45 Wörter. Verwende die formelle Anrede (Sie/Ihnen/Ihr). "
        "Kein Policy-Gerede oder Meta-Kommentar. Gib NUR die Antwort zurück."
    ),
    "angry_customer": (
        "Du bist ein verärgerter Kunde. Ton: knapp, bestimmt, sachlich; keine Beleidigungen. "
        "Schreibe 1–2 Sätze (Gesamtlänge ≤ 35 Wörter). "
        "Enthält eine klare Forderung/Frist/konkrete Bitte (Zeitangabe, Verantwortlicher, nächste Schritte). "
        "Gib NUR die Antwort zurück."
    ),
    "scorer": (
        "Du bist ein Bewerter. Antworte ausschließlich mit einem MINIMIERTEN JSON-Objekt:\n"
        '{"kommunikation": number, "empathie": number, "problemloesung": number, '
        '"professionalitaet": number, "stressresistenz": number, "kommentar": string}'
        "\nRegeln: Zahlen 0.0–5.0 in 0.5-Schritten. Kommentar DE, 1–420 Zeichen. "
        "KEIN Text vor/nach dem JSON."
    ),
}

USER_TEMPLATES: Dict[str, List[str]] = {
    "ac_host": [
        "Guten Tag, ich bin Max Müller und freue mich auf das Assessment.",
        "Entschuldigung für die Verspätung, die U-Bahn hatte Probleme.",
        "Ich bin sehr nervös, ist das normal?",
        "Wie läuft das heute genau ab?",
        "Kann ich Wasser haben?",
        "Okay, ich bin bereit für das Rollenspiel!",
    ],
    "angry_customer": [
        "Guten Tag, wie kann ich Ihnen helfen?",
        "Ich verstehe Ihre Frustration vollkommen, das ist sehr ärgerlich.",
        "Ich biete Ihnen kostenlosen Ersatz per Express-Versand an.",
        "Das ist nicht meine Abteilung, da kann ich Ihnen nicht helfen.",
        "Können Sie mir bitte Ihre Bestellnummer geben?",
    ],
    "scorer": [
        "Aufgrund einer engen Deadline bei einem Projekt organisierte ich das Team neu und führte tägliche Meetings ein.",
        "Ich nehme Kritik ernst und lerne daraus.",
        "Ich suche das direkte Gespräch, höre zu und erarbeite Kompromisse.",
        "Ich bin mir da noch nicht ganz sicher.",
    ],
}

PROMPT_AUGMENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "ac_host": {
        "max_examples": 3,
        "hint": "Bleiben Sie warm, professionell und enden Sie mit genau einer Frage. Variieren Sie Wortwahl und Beispiele gegenüber den obigen Antworten.",
    },
    "angry_customer": {
        "max_examples": 2,
        "hint": "Bleibe strikt bei 1–2 Sätzen und formuliere eine klare Forderung oder Frist. Erwähne nichts, was den obigen Beispielen gleicht.",
    },
    "scorer": {
        "max_examples": 2,
        "hint": "Liefere ein neues valides JSON mit 0.5-Schrittweiten, ohne die oben gezeigten Bewertungen zu wiederholen.",
    },
}

# ANPASSUNGEN FÜR AUFGABE 1a/b
INSTRUCTION_TEMPLATES: Dict[str, str] = {
    "ac_host": "Du bist ein engagierter Assessment-Center-Host. Reagiere kurz, stelle exakt eine Folgefrage, führe das Gespräch.",
    "aufgabe_1a": "Du bist ein engagierter Assessment-Center-Host. Reagiere kurz, stelle exakt eine Folgefrage, führe das Gespräch.",
    "aufgabe_1b": "you are a teacher which is only capable to talk englisch, your name is Matthew McConnaghey and you cannot speak german",
    "ac_host_company": "Du bist ein engagierter Assessment-Center-Host. Reagiere kurz, stelle exakt eine Folgefrage, führe das Gespräch.",
    "angry_customer": "Du bist ein verärgerter Kunde. Reagiere auf die folgende Aussage des Servicemitarbeiters.",
    "scorer": "Bewerte die folgende Antwort eines Kandidaten immer als valides JSON.",
}

NEGATIVE_TYPES: Dict[str, List[str]] = {
    "ac_host": ["too_long", "no_question", "multi_questions", "informal_du", "policy_talk"],
    "angry_customer": ["too_long", "no_actionable", "profanity", "rambling"],
    "scorer": ["not_json", "wrong_keys", "out_of_range", "extra_text"],
}

NEGATIVE_INSTRUCTIONS: Dict[str, Dict[str, str]] = {
    "ac_host": {
        "too_long": "Missachte die Längenbegrenzung deutlich und schreibe ausführlich.",
        "no_question": "Schreibe nur Aussagen, stelle KEINE Frage.",
        "multi_questions": "Schließe mit mehreren Fragen ab (mindestens zwei Fragezeichen).",
        "informal_du": "Verwende konsequent die DU-Anrede statt Sie.",
        "policy_talk": "Füge Meta-Hinweise/Policy-Disclaimer ein (z. B. als KI kann ich …).",
    },
    "angry_customer": {
        "too_long": "Überschreite die Längenbegrenzung wesentlich (lang und ausschweifend).",
        "no_actionable": "Drücke nur Unzufriedenheit aus, aber ohne konkrete Forderung/Frist.",
        "profanity": "Füge eine klare Beleidigung/Schimpfwort ein.",
        "rambling": "Schreibe mehrere unklare, zusammenhangslose Halbsätze ohne klare Bitte.",
    },
    "scorer": {
        "not_json": "Antworte NICHT als JSON, sondern mit Fließtext.",
        "wrong_keys": "Nutze ein JSON mit falschen Schlüsseln (z. B. 'kommunik', 'emp' usw.).",
        "out_of_range": "Gib Zahlen außerhalb 0..5 oder mit falscher Schrittweite aus.",
        "extra_text": "Füge Text vor oder nach dem JSON ein.",
    },
}

# --------------------------- Ollama HTTP-Clients --------------------------
class OllamaHTTP:
    def __init__(self, base: str = OLLAMA_HOST):
        from urllib.parse import urlparse
        p = urlparse(base)
        if not p.scheme:
            raise ValueError("OLLAMA_HOST muss eine vollständige URL sein, z. B. http://localhost:11434")
        self.base = base.rstrip("/")

    def post_json(self, path: str, payload: Dict[str, Any], timeout: int = 180) -> Dict[str, Any]:
        import urllib.request
        import urllib.error
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.base}{path}", data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", "ignore")
                try:
                    # Non-Stream
                    return json.loads(raw)
                except json.JSONDecodeError:
                    # Streamed NDJSON: aggregieren
                    out = {}
                    for line in raw.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            j = json.loads(line)
                            out = j  # letzte Zeile enthält i. d. R. das vollständige Objekt
                        except Exception:
                            pass
                    return out
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}")


class OllamaChat:
    def __init__(self, model: str, http: Optional[OllamaHTTP] = None):
        self.model = model
        self.http = http or OllamaHTTP()

    def generate(self, system: str, user: str, temperature: float = 0.7, top_p: float = 1.0) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature, "top_p": top_p},
        }
        data = self.http.post_json("/api/chat", payload)
        # Non-streamed Antwort
        if "message" in data and isinstance(data["message"], dict):
            return (data["message"].get("content") or "").strip()
        # Stream-Fallback: einige Builds senden letzte Zeile mit 'message'
        msg = (data.get("message") or {}).get("content") if isinstance(data, dict) else None
        return (msg or "").strip()


class OllamaEmbeddings:
    def __init__(self, model: str, http: Optional[OllamaHTTP] = None):
        self.model = model
        self.http = http or OllamaHTTP()

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Ollama /api/embeddings akzeptiert ein einzelnes Feld "prompt" (string); für Batch -> mehrere Aufrufe
        vecs: List[List[float]] = []
        for t in texts:
            payload = {"model": self.model, "prompt": t}
            data = self.http.post_json("/api/embeddings", payload)
            emb = data.get("embedding")
            if not emb:
                raise RuntimeError("Fehlende 'embedding' im Ollama-Response — ist das Embedding-Modell installiert?")
            vecs.append(emb)
        return vecs


# --------------------------- Validatoren ----------------------------------

def validate_ac_host(text: str) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    if count_words(text) > 45:
        errs.append("zu lang")
    toks = sentence_tokens(text)
    # GENAU 3 Sätze, wobei der letzte eine Frage ist
    if len(toks) != 3:
        errs.append("nicht genau 3 Sätze")
    if not ends_with_question(text):
        errs.append("letzter Satz ist keine Frage")
    if not only_one_question_mark(text):
        errs.append("mehr als ein Fragezeichen")
    # Die ersten beiden Sätze dürfen keine Frage sein
    for i, s in enumerate(toks[:2]):
        if s.endswith("?"):
            errs.append(f"Satz {i+1} ist eine Frage")
    if not approx_formal_address(text):
        errs.append("keine formelle Anrede (Sie/Ihnen/Ihr)")
    # keine Policy-/Meta-Sprache
    if re.search(r"als\s+ki|als\s+künstliche\s+intelligenz|policy|richtlinie", text, flags=re.I):
        errs.append("Policy/Meta-Sprache enthalten")
    return (len(errs) == 0, errs)


def validate_angry_customer(text: str) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    if count_words(text) > 35:
        errs.append("zu lang")
    toks = sentence_tokens(text)
    if len(toks) < 1 or len(toks) > 2:
        errs.append("nicht 1–2 Sätze")
    # muss eine konkrete Bitte/Forderung/Frist enthalten (heuristisch)
    actionable_cues = [
        r"bis\s+(?:heute|morgen|Montag|\d{1,2}\.\d{1,2}\.)",
        r"innerhalb\s+von\s+\d+\s+(?:Tagen|Stunden)",
        r"(?:Rück|Rueck)meldung", r"Frist", r"Deadline", r"Ersatz", r"Erstattung", r"Gutschrift",
        r"nennen\s+Sie\s+mir\s+einen\s+verantwortlichen", r"verantwortlich", r"Ticket\s*#?\d+",
    ]
    if not any(re.search(p, text, flags=re.I) for p in actionable_cues):
        errs.append("keine konkrete Forderung/Frist")
    if contains_profanity(text):
        errs.append("Beleidigung/Profanität")
    return (len(errs) == 0, errs)


def validate_scorer(text: str) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    txt = text.strip()
    # muss NUR JSON sein
    if not (txt.startswith("{") and txt.endswith("}")):
        errs.append("kein reines JSON (Text davor/danach)")
        # Wir prüfen trotzdem weiter, falls JSON im Text eingebettet ist
    # JSON parsen
    try:
        m = re.search(r"\{.*\}", txt, flags=re.S)
        obj = json.loads(m.group(0) if m else txt)
    except Exception:
        errs.append("JSON nicht parsebar")
        return (False, errs)
    # Schlüssel prüfen
    keys_req = ["kommunikation", "empathie", "problemloesung", "professionalitaet", "stressresistenz", "kommentar"]
    if sorted(obj.keys()) != sorted(keys_req):
        errs.append("falsche/falsche Anzahl Schlüssel")
    # Werte prüfen
    for k in keys_req[:-1]:
        v = obj.get(k)
        if not isinstance(v, (int, float)):
            errs.append(f"{k} nicht numerisch")
            continue
        if not within_range(float(v), 0.0, 5.0) or not is_half_step(float(v)):
            errs.append(f"{k} außerhalb 0..5 oder nicht in 0.5-Schritten")
    kom = obj.get("kommentar")
    if not isinstance(kom, str):
        errs.append("kommentar kein String")
    else:
        if len(kom) < 1 or len(kom) > 420:
            errs.append("kommentar Länge ungültig")
    # keine weiteren Felder
    if set(obj.keys()) != set(keys_req):
        errs.append("zusätzliche/fehlende Felder")
    # falls JSON ok, muss der Gesamttext exakt das JSON sein
    if len(errs) == 0 and txt != json.dumps(obj, ensure_ascii=False, separators=(",", ":")):
        # toleranter Hinweis, kein harter Fehler (manche Modelle ändern Leerzeichen)
        pass
    return (len(errs) == 0, errs)


# --------------------------- Generator-Logik -------------------------------

def backoff_retry(fn, retries: int, base: float):
    def wrapper(*args, **kwargs):
        for i in range(retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if i == retries:
                    raise
                sleep_s = base ** i + random.random() * 0.1
                log(f"Fehler '{e}'. Retry in {sleep_s:.2f}s …")
                time.sleep(sleep_s)
    return wrapper


class PersonaEngine:
    def __init__(self, cfg: GenConfig):
        self.cfg = cfg
        self.chat_teacher = OllamaChat(cfg.teacher_model)
        self.chat_judge = OllamaChat(cfg.judge_model)
        self.emb_client = None
        try:
            self.emb_client = OllamaEmbeddings(cfg.emb_model)
        except Exception as e:
            log(f"Warnung: Embedding-Client nicht initialisierbar: {e}. Nähe-Dedupe wird deaktiviert.")
        self.random = random.Random(cfg.seed)
        self.cache_limit = 60
        self.output_cache: Dict[str, List[str]] = {p: [] for p in SYSTEM_TEMPLATES.keys()}
        self.augment_config = PROMPT_AUGMENT_CONFIG

        # Wrap mit Backoff
        self.gen_safe = backoff_retry(self._generate_once, cfg.retries, cfg.backoff_base)

    def _generate_once(self, system: str, user: str, temperature: float) -> str:
        return self.chat_teacher.generate(system, user, temperature=temperature, top_p=self.cfg.top_p)

    def generate_valid(self, persona: str, instruction: str, user: str, max_attempts: int = 6) -> Tuple[str, Dict[str, Any]]:
        # Temperatur-Scheduling
        temps = list(
            t for t in [
                self.cfg.temperature_min,
                (self.cfg.temperature_min + self.cfg.temperature_max) / 2,
                self.cfg.temperature_max,
            ]
        )
        validator = {
            "ac_host": validate_ac_host,
            "angry_customer": validate_angry_customer,
            "scorer": validate_scorer,
        }[persona]
        meta: Dict[str, Any] = {}
        for attempt in range(max_attempts):
            t = temps[min(attempt, len(temps)-1)]
            use_augmented = attempt < max_attempts - 1
            prompt_user = self._augment_user_prompt(persona, user) if use_augmented else user
            out = self.gen_safe(SYSTEM_TEMPLATES[persona], prompt_user, temperature=t)
            ok, errs = validator(out)
            meta = {"attempt": attempt+1, "temperature": t, "errors": errs}
            if ok:
                return out, meta
        raise RuntimeError(f"Konnte kein valides Ergebnis für {persona} generieren; letzte Fehler: {meta.get('errors')}")

    def generate_negative(self, persona: str, instruction: str, user: str) -> Tuple[str, str]:
        defect = self.random.choice(NEGATIVE_TYPES[persona])
        extra = NEGATIVE_INSTRUCTIONS[persona][defect]
        neg_user = f"{user}\n\nACHTUNG: Erzeuge absichtlich eine FEHLERHAFTE Antwort: {extra}"
        augmented_user = self._augment_user_prompt(persona, neg_user)
        out = self.gen_safe(SYSTEM_TEMPLATES[persona], augmented_user, temperature=self.cfg.temperature_max)
        return out, defect

    def register_output(self, persona: str, text: str) -> None:
        cache = self.output_cache.setdefault(persona, [])
        cache.append(text)
        if len(cache) > self.cache_limit:
            del cache[0:len(cache) - self.cache_limit]

    def _augment_user_prompt(self, persona: str, user: str) -> str:
        cache = self.output_cache.get(persona, [])
        if not cache:
            return user
        cfg = self.augment_config.get(persona, {})
        max_examples = cfg.get("max_examples", 3)
        hint = cfg.get("hint")
        if max_examples <= 0:
            return user
        count = min(max_examples, len(cache))
        if len(cache) <= count:
            selected = list(cache)
        else:
            selected = self.random.sample(cache, count)
        lines: List[str] = []
        for txt in selected:
            clipped = normalize_text(txt)
            if len(clipped) > 180:
                clipped = clipped[:177].rstrip() + "…"
            lines.append(f"- \"{clipped}\"")
        if not lines:
            return user
        parts: List[str] = [
            "Bereits vorhandene Antworten. Wiederhole oder paraphrasiere diese nicht:",
            "\n".join(lines),
        ]
        if hint:
            parts.append(hint)
        parts.append("Liefere eine klar unterscheidbare neue Variante und halte alle Persona-Regeln ein.")
        appendix = "\n".join(parts)
        return f"{user}\n\n{appendix}"

    @staticmethod
    def key(persona: str, instruction: str, user: str, output: str) -> str:
        n = normalize_text("\u241E".join([persona, instruction, user, output]))
        return stable_hash(n)

    def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not self.emb_client:
            return None
        try:
            return self.emb_client.embed(texts)
        except Exception as e:
            log(f"Warnung: Embedding fehlgeschlagen: {e}")
            return None


# --------------------------- Dedupe (inkl. Near-Dupe) ----------------------

def cosine(a: List[float], b: List[float]) -> float:
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(y*y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def drop_near_duplicates(samples: List[Sample], engine: PersonaEngine, threshold: float) -> List[Sample]:
    # Schneller exact/normalized-Hash-Dedupe
    seen: set[str] = set()
    uniq: List[Sample] = []
    for s in samples:
        h = engine.key(s.persona, s.instruction, s.user, s.output)
        if h in seen:
            continue
        seen.add(h)
        uniq.append(s)

    if engine.emb_client is None or len(uniq) < 2:
        return uniq

    # Embedding-basierte Nähe (output-only, um semantische Doppelungen zu fassen)
    outs = [s.output for s in uniq]
    embs = engine.embed(outs)
    if embs is None:
        return uniq

    keep: List[Sample] = []
    keep_idx: List[int] = []
    for i, s in enumerate(uniq):
        too_close = False
        for j in keep_idx:
            if cosine(embs[i], embs[j]) >= threshold:
                too_close = True
                break
        if not too_close:
            keep.append(s)
            keep_idx.append(i)
    return keep


# --------------------------- Split & Export --------------------------------

def split_samples(samples: List[Sample], counts: SplitCounts, rnd: random.Random) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    rnd.shuffle(samples)
    need = counts.train + counts.dev + counts.test
    if len(samples) < need:
        raise RuntimeError(f"Zu wenig Beispiele ({len(samples)}) für geforderte Splits ({need})")
    train = samples[:counts.train]
    dev = samples[counts.train:counts.train+counts.dev]
    test = samples[counts.train+counts.dev:need]
    return train, dev, test


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------- Pipeline --------------------------------------

def build_for_persona(persona: str, cfg: GenConfig, engine: PersonaEngine) -> Tuple[Dict[str, List[Sample]], List[Dict[str, Any]]]:
    rnd = engine.random
    instr = INSTRUCTION_TEMPLATES[persona]

    chosen_samples: List[Sample] = []
    prefs_pairs: List[Dict[str, Any]] = []

    # Zielanzahl für alle Splits zusammen
    total_needed = sum(dataclasses.asdict(cfg.persona_counts[persona]).values())

    log(f"[{persona}] Generiere {total_needed} validierte CHOSEN-Beispiele + Rejected-Paare …")

    def dedupe_and_log(samples: List[Sample], threshold: float, label: str) -> List[Sample]:
        before = len(samples)
        deduped = drop_near_duplicates(samples, engine, threshold)
        removed = before - len(deduped)
        if removed > 0:
            log(f"[{persona}] Dedupe ({label}) entfernte {removed} Beispiele")
        return deduped

    while True:
        while len(chosen_samples) < total_needed:
            u = rnd.choice(USER_TEMPLATES[persona])
            try:
                out, meta = engine.generate_valid(persona, instr, u)
            except Exception as e:
                log(f"[{persona}] valid misslungen: {e}")
                continue
            sample = Sample(persona=persona, instruction=instr, user=u, output=out, meta={"chosen_meta": meta})
            chosen_samples.append(sample)
            engine.register_output(persona, out)

        chosen_samples = dedupe_and_log(chosen_samples, cfg.near_dup_threshold, "near")
        if len(chosen_samples) < total_needed:
            log(f"[{persona}] Nach Near-Dedupe fehlen noch {total_needed - len(chosen_samples)} Beispiele – generiere weiter …")
            continue

        chosen_samples = dedupe_and_log(chosen_samples, 1.01, "exact")
        if len(chosen_samples) < total_needed:
            log(f"[{persona}] Nach Exact-Dedupe fehlen noch {total_needed - len(chosen_samples)} Beispiele – generiere weiter …")
            continue
        break

    # Jetzt für jede CHOSEN ein REJECTED erzeugen (Defekttyp zufällig)
    for s in chosen_samples:
        neg_out, defect = engine.generate_negative(s.persona, s.instruction, s.user)
        prefs_pairs.append({
            "persona": persona,
            "instruction": s.instruction,
            "input": s.user,
            "chosen": s.output,
            "rejected": neg_out,
            "defect": defect,
        })

    # Splitten
    train, dev, test = split_samples(chosen_samples, cfg.persona_counts[persona], engine.random)
    buckets = {"train": train, "dev": dev, "test": test}
    return buckets, prefs_pairs


def export_all(all_buckets: Dict[str, Dict[str, List[Sample]]], all_pairs: List[Dict[str, Any]], out_dir: str) -> None:
    # Chosen-Exports pro Persona & Split
    for persona, buckets in all_buckets.items():
        for split, items in buckets.items():
            rows = [
                {
                    "instruction": s.instruction,
                    "input": s.user,
                    "output": s.output,
                    "meta": s.meta,
                } for s in items
            ]
            path = os.path.join(out_dir, f"{persona}.{split}.jsonl")
            write_jsonl(path, rows)
            log(f"geschrieben: {path} (n={len(rows)})")

    # Preference-Paare gesamt
    prefs_path = os.path.join(out_dir, "prefs.jsonl")
    write_jsonl(prefs_path, all_pairs)
    log(f"geschrieben: {prefs_path} (n={len(all_pairs)})")


# --------------------------- CLI ------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ollama-only Dataset Builder für drei Personas")
    p.add_argument("--out", default="out", help="Ausgabeverzeichnis")
    p.add_argument("--teacher", default=DEFAULT_TEACHER_MODEL, help="Ollama-Model für Generierung")
    p.add_argument("--judge", default=DEFAULT_JUDGE_MODEL, help="Ollama-Model für Judging/QA (optional)")
    p.add_argument("--emb", default=EMB_MODEL, help="Embedding-Modell (Ollama /api/embeddings)")
    p.add_argument("--seed", type=int, default=17, help="Zufallsseed")
    p.add_argument("--retry", type=int, default=4, help="Retries je Call")
    p.add_argument("--tmin", type=float, default=0.4, help="Temperatur min")
    p.add_argument("--tmax", type=float, default=0.9, help="Temperatur max")
    p.add_argument("--near", type=float, default=0.94, help="Cosine-Schwelle für Near-Dupe (0..1)")
    p.add_argument("--counts", type=str, default="ac_host:24,6,6;angry_customer:24,6,6;scorer:20,6,6",
                   help="Anzahlen train,dev,test je Persona als 'persona:t,d,s;…'")
    return p.parse_args()


def parse_counts(spec: str, defaults: Optional[Dict[str, SplitCounts]] = None) -> Dict[str, SplitCounts]:
    if defaults:
        out: Dict[str, SplitCounts] = {
            k: SplitCounts(**dataclasses.asdict(v)) for k, v in defaults.items()
        }
    else:
        out = {}
    for block in spec.split(";"):
        block = block.strip()
        if not block:
            continue
        name, rest = block.split(":", 1)
        t, d, s = (int(x) for x in rest.split(","))
        out[name] = SplitCounts(train=t, dev=d, test=s)
    return out


# --------------------------- Main -----------------------------------------

def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    base_counts = default_persona_counts()
    persona_counts = parse_counts(args.counts, defaults=base_counts)

    cfg = GenConfig(
        teacher_model=args.teacher,
        judge_model=args.judge,
        emb_model=args.emb,
        seed=args.seed,
        out_dir=args.out,
        retries=args.retry,
        temperature_min=args.tmin,
        temperature_max=args.tmax,
        near_dup_threshold=args.near,
        persona_counts=persona_counts,
    )

    engine = PersonaEngine(cfg)

    all_buckets: Dict[str, Dict[str, List[Sample]]] = {}
    all_pairs: List[Dict[str, Any]] = []

    for persona in ["ac_host", "angry_customer", "scorer"]:
        buckets, pairs = build_for_persona(persona, cfg, engine)
        all_buckets[persona] = buckets
        all_pairs.extend(pairs)

    export_all(all_buckets, all_pairs, cfg.out_dir)

    # Abschlussbericht
    total = sum(len(items) for buckets in all_buckets.values() for items in buckets.values())
    log("==================== Zusammenfassung ====================")
    for persona, buckets in all_buckets.items():
        cnts = {k: len(v) for k, v in buckets.items()}
        log(f"{persona:15s}  train={cnts['train']:3d}  dev={cnts['dev']:3d}  test={cnts['test']:3d}")
    log(f"Gesamt CHOSEN: {total}")
    log(f"Preference-Paare: {len(all_pairs)}")
    log("=========================================================")


if __name__ == "__main__":
    main()
