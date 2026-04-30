# -*- coding: utf-8 -*-
"""
Interaktives SFT-Training:
- Modell aus `models/base` auswählen
- Persona-Datensatz aus `synthetic_generation_2/data` wählen
- Supervised Fine-Tuning (SFT) mit TRL durchführen (standardmäßig LoRA-frei)
- Ergebnisse & Logfiles unter `models/persona/<persona>/<timestamp>` speichern
"""
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from packaging.version import parse as V
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# =============================================================================
# Pfad-Setup
# =============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_BASE_DIR = PROJECT_ROOT / "models" / "base"
PERSONA_DIR = PROJECT_ROOT / "models" / "persona"
DATA_DIR = PROJECT_ROOT / "data"


os.chdir(PROJECT_ROOT)

# =============================================================================
# Utility-Funktionen
# =============================================================================
def ensure_dirs() -> None:
    MODELS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    PERSONA_DIR.mkdir(parents=True, exist_ok=True)


def transformers_version_check(min_ver: str = "4.56.0") -> None:
    import transformers as tf

    if V(tf.__version__) < V(min_ver):
        print(
            f"⚠️  Deine transformers-Version ist {tf.__version__}. "
            f"Für moderne Architekturen bitte mindestens {min_ver} installieren:\n"
            "    pip install -U 'transformers>={min_ver}' accelerate torch\n"
        )


def is_trainable_model(model_dir: Path) -> bool:
    if not (model_dir / "config.json").exists():
        return False
    name = model_dir.name.lower()
    if "gguf" in name or "mlx" in name:
        return False
    return True


def list_available_models() -> List[Path]:
    if not MODELS_BASE_DIR.exists():
        return []
    return sorted([d for d in MODELS_BASE_DIR.iterdir() if d.is_dir() and is_trainable_model(d)])


def scan_personas() -> Dict[str, Dict[str, Path]]:
    personas: Dict[str, Dict[str, Path]] = {}
    for file in DATA_DIR.glob("*.jsonl"):
        parts = file.name.split(".")
        if len(parts) < 2:
            continue
        persona = parts[0]
        split = parts[1]
        personas.setdefault(persona, {})[split] = file
    return personas


def select_from_list(options: List[Path], title: str) -> Path:
    if not options:
        print(f"❌ Keine Optionen für '{title}' gefunden.")
        sys.exit(1)

    print("\n" + "=" * 72)
    print(f"📋 {title}")
    print("=" * 72)
    for idx, opt in enumerate(options, 1):
        print(f"[{idx:2d}] {opt.name}")
    print("=" * 72)

    while True:
        try:
            choice = int(input(f"👉 Auswahl (1-{len(options)}): ").strip())
            if 1 <= choice <= len(options):
                selected = options[choice - 1]
                print(f"✅ Ausgewählt: {selected.name}\n")
                return selected
            print(f"❌ Bitte eine Zahl zwischen 1 und {len(options)} eingeben.")
        except ValueError:
            print("❌ Ungültige Eingabe – bitte Zahl eingeben.")
        except KeyboardInterrupt:
            print("\nAbbruch durch Benutzer.")
            sys.exit(0)


def select_persona(persona_map: Dict[str, Dict[str, Path]]) -> Tuple[str, Dict[str, Path]]:
    if not persona_map:
        print(f"❌ Keine Persona-Datensätze in {DATA_DIR} gefunden.")
        sys.exit(1)

    personas = sorted(persona_map.keys())
    print("\n" + "=" * 72)
    print("🧠 Verfügbare Personas")
    print("=" * 72)
    for idx, name in enumerate(personas, 1):
        splits = ", ".join(sorted(persona_map[name].keys()))
        print(f"[{idx:2d}] {name}  ({splits})")
    print("=" * 72)

    while True:
        try:
            choice = int(input(f"👉 Persona wählen (1-{len(personas)}): ").strip())
            if 1 <= choice <= len(personas):
                persona = personas[choice - 1]
                print(f"✅ Persona gewählt: {persona}\n")
                return persona, persona_map[persona]
            print(f"❌ Bitte eine Zahl zwischen 1 und {len(personas)} eingeben.")
        except ValueError:
            print("❌ Ungültige Eingabe – bitte Zahl eingeben.")
        except KeyboardInterrupt:
            print("\nAbbruch durch Benutzer.")
            sys.exit(0)

def select_epochs() -> int:
    print("\n" + "=" * 72)
    print("⏱️  Training-Epochen")
    print("=" * 72)
    print("Empfehlung: 3-5 Epochen für kleine Datensätze (<100 Samples)")
    print("=" * 72)
    
    while True:
        try:
            epochs = int(input("👉 Anzahl Epochen (1-100): ").strip())
            if 1 <= epochs <= 100:
                print(f"✅ Training mit {epochs} Epochen\n")
                return epochs
            print("❌ Bitte eine Zahl zwischen 1 und 100 eingeben.")
        except ValueError:
            print("❌ Ungültige Eingabe – bitte Zahl eingeben.")
        except KeyboardInterrupt:
            print("\nAbbruch durch Benutzer.")
            sys.exit(0)


def build_prompt(example: Dict[str, str], tokenizer, model_type: str) -> str:
    instruction = example.get("instruction", "")
    user_input = example.get("input", "")
    output = example.get("output", "")

    # Llama 3.1 & Qwen nutzen Chat Template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        # Check ob Multi-Turn (enthält "\nKevin:" oder "\nUser:")
        if "\nKevin:" in user_input or "\nUser:" in user_input:
            # Parse Multi-Turn aus input
            messages = [{"role": "system", "content": instruction}]
            
            # Split by turns
            lines = user_input.split("\n")
            for line in lines:
                if line.startswith("Kevin:"):
                    messages.append({"role": "assistant", "content": line[6:].strip()})
                elif line.startswith("User:"):
                    messages.append({"role": "user", "content": line[5:].strip()})
                elif line.strip() and not any(line.startswith(prefix) for prefix in ["Kevin:", "User:"]):
                    # Erste Zeile ohne Prefix = User
                    messages.append({"role": "user", "content": line.strip()})
            
            # Output als Assistant
            messages.append({"role": "assistant", "content": output})
        else:
            # Single-Turn
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": output},
            ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return prompt
    
    # Fallback: Mistral Format (bleibt wie es ist)
    return f"<s>[INST] {instruction}\n{user_input} [/INST]\n{output}</s>"


def select_quantization(has_cuda: bool) -> bool:
    print("\n" + "=" * 72)
    print("💾 4-bit Quantisierung (QLoRA)")
    print("=" * 72)
    print("Spart VRAM, indem das Modell in 4-bit geladen wird.")
    print("Erfordert 'bitsandbytes' und eine CUDA-GPU.")

    if not has_cuda:
        print("⚠️  Keine CUDA-GPU gefunden. Quantisierung ist nicht möglich.")
        print("=" * 72)
        print("❌ Quantisierung deaktiviert.\n")
        return False
    
    print("=" * 72)

    while True:
        answer = input("👉 4-bit Quantisierung aktivieren? [j/N]: ").strip().lower()

        if not answer: # User drückt Enter (Default ist 'N')
            answer = 'n'
        
        if answer in ['n', 'nein', 'no']:
            print("❌ Quantisierung deaktiviert\n")
            return False
        if answer in ['j', 'ja', 'y', 'yes']:
            print("✅ 4-bit Quantisierung aktiviert\n")
            return True
        print(f"❌ Bitte 'j' (Ja) oder 'n' (Nein) eingeben.")


def select_debug_mode() -> bool:
    print("\n" + "=" * 72)
    print("🐛 Debug-Modus")
    print("=" * 72)
    print("Zeigt den ersten formatierten Trainings-Prompt zur Überprüfung an.")
    print("=" * 72)
    
    while True:
        answer = input("👉 Debug-Modus aktivieren? [j/N]: ").strip().lower()
        if not answer or answer in ['n', 'nein', 'no']:
            print("❌ Debug-Modus deaktiviert\n")
            return False
        if answer in ['j', 'ja', 'y', 'yes']:
            print("✅ Debug-Modus aktiviert\n")
            return True
        print("❌ Bitte 'j' oder 'n' eingeben.")


def create_output_dir(persona: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PERSONA_DIR / persona / f"sft_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("train_sft")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logfile = logging.FileHandler(output_dir / "training.log", encoding="utf-8")
    logfile.setLevel(logging.INFO)
    logfile.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(logfile)
    return logger


# =============================================================================
# Hauptlogik
# =============================================================================
def main() -> None:
    ensure_dirs()
    transformers_version_check()

    print("🚀 SFT-Training (Supervised Fine-Tuning)")
    print("=" * 72)
    print(f"📂 Projekt-Root: {PROJECT_ROOT}")
    print(f"📂 Modelle (Basis): {MODELS_BASE_DIR}")
    print(f"📂 Personas: {DATA_DIR}\n")

    available_models = list_available_models()
    if not available_models:
        print(f"❌ Keine trainierbaren Modelle in {MODELS_BASE_DIR} gefunden.")
        print("   → Bitte zuerst ein Modell mit `scripts/download_model.py` herunterladen.")
        sys.exit(1)

    model_path = select_from_list(available_models, "Trainingsmodell wählen")
    persona_name, persona_files = select_persona(scan_personas())
    num_epochs = select_epochs()  
    debug_mode = select_debug_mode()


    train_file = persona_files.get("train")
    if not train_file:
        print(f"❌ Kein train-Split für Persona '{persona_name}' gefunden.")
        sys.exit(1)

    eval_file = persona_files.get("dev") or persona_files.get("validation")

    output_dir = create_output_dir(persona_name)
    logger = setup_logging(output_dir)

    logger.info("Start des Trainings")
    logger.info(f"Modell: {model_path.name}")
    logger.info(f"Persona: {persona_name}")
    logger.info(f"Output-Verzeichnis: {output_dir}")

    logger.info("Lade Modell-Konfiguration …")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    logger.info("Konfiguration geladen.")

    has_cuda = torch.cuda.is_available()
    logger.info(f"CUDA verfügbar: {has_cuda}")
    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        bf16_ok = torch.cuda.get_device_capability(0)[0] >= 8
    else:
        bf16_ok = False

    torch_dtype = torch.bfloat16 if bf16_ok else (torch.float16 if has_cuda else torch.float32)
    logger.info(f"Verwendeter Torch-Datentyp: {torch_dtype}")

    use_quantization = select_quantization(has_cuda)

    logger.info("Lade Modell-Gewichte …")

# Check Modell-Größe für Auto-Quantisierung
    try:
        if use_quantization and has_cuda:
            logger.info(f"Aktiviere 4-bit Quantisierung (QLoRA) auf User-Anfrage...")
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=True,
            )
            logger.info("✅ 4-bit Quantisierung aktiviert.")
        else:
            # Normales Laden
            if use_quantization and not has_cuda:
                logger.warning("Quantisierung gewünscht, aber keine CUDA-GPU gefunden. Lade normal.")
            else:
                logger.info("Lade Modell normal (ohne Quantisierung).")
            
            device_map = {"": 0} if has_cuda else "cpu"
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
                local_files_only=True,
            )
            logger.info("Modell geladen.")
    
    except Exception as e:
        logger.error(f"Kritisches Problem beim Laden des Modells: {e}")
        logger.error("Stelle sicher, dass alle Dependencies (transformers, accelerate, bitsandbytes) aktuell sind.")
        sys.exit(1)

    model.config.use_cache = False
    logger.info("Modell-Konfiguration abgeschlossen.")

    logger.info("Initialisiere LoRA-Adapter …")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info("LoRA-Adapter aktiviert.")

    logger.info("Lade Tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    logger.info("Tokenizer geladen.")

    data_files = {"train": str(train_file)}
    if eval_file:
        data_files["validation"] = str(eval_file)
    logger.info(f"Lade Datensatz ({len(data_files)} Split(s)) …")
    dataset_dict = load_dataset("json", data_files=data_files)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict.get("validation")
    logger.info("Datensatz erfolgreich geladen.")
    logger.info(f"Train-Examples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval-Examples:  {len(eval_dataset)}")

    model_type = getattr(config, "model_type", "")
    has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template
    logger.info(f"Prompt-Format: {'Chat Template ({model_type})' if has_chat_template else 'Mistral [INST] Format'}")

    def formatting_func(example):
        return build_prompt(example, tokenizer, model_type)
    
    if debug_mode and len(train_dataset) > 0:
        sample_prompt = formatting_func(train_dataset[0])
        logger.info("=" * 72)
        logger.info("🐛 DEBUG: Beispiel-Trainings-Prompt (erstes Sample)")
        logger.info("=" * 72)
        logger.info(sample_prompt)
        logger.info("=" * 72)

    if has_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.03,
        optim="adamw_torch",
        fp16=(torch_dtype == torch.float16),
        bf16=(torch_dtype == torch.bfloat16),
        max_grad_norm=0.3,
        report_to="none",
        gradient_checkpointing=True,
    )

    logger.info("Initialisiere SFTTrainer …")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        processing_class=tokenizer,
        args=training_args,
    )
    logger.info("Trainer initialisiert, starte Training …")

    train_result = trainer.train()
    logger.info("Training abgeschlossen.")

    eval_metrics: Optional[Dict[str, float]] = None
    if eval_dataset:
        logger.info("Starte Evaluation auf Validation-Split …")
        eval_metrics = trainer.evaluate()
        logger.info(f"Evaluationsergebnisse: {eval_metrics}")

    logger.info("Speichere LoRA-Adapter & Tokenizer …")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Speichere Trainingszusammenfassung …")
    eval_metrics_json = None
    if eval_metrics:
        eval_metrics_json = {
            k: (float(v) if isinstance(v, (int, float)) else v) for k, v in eval_metrics.items()
        }
    summary = {
        "model": model_path.name,
        "persona": persona_name,
        "timestamp": datetime.now().isoformat(),
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset) if eval_dataset else 0,
        "train_runtime": getattr(train_result, "metrics", {}).get("train_runtime", None),
        "train_samples_per_second": getattr(train_result, "metrics", {}).get("train_samples_per_second", None),
        "eval_metrics": eval_metrics_json,
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("✅ Training erfolgreich abgeschlossen.")
    logger.info(f"Adapter & Artefakte: {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining abgebrochen.")

