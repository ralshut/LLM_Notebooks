# -*- coding: utf-8 -*-
import os
import sys
import json
import warnings
from pathlib import Path

import torch
import transformers
from packaging.version import parse as V

# --- Shim: Top-Level PreTrainedModel fehlt in neueren transformers ---
try:
    _ = transformers.PreTrainedModel  # existiert in manchen Versionen noch
except Exception:
    from transformers.modeling_utils import PreTrainedModel as _PTM
    transformers.PreTrainedModel = _PTM
# --------------------------------------------------------------------

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ================================
# Pfad-Setup (Skript liegt in scripts/)
# ================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.chdir(PROJECT_ROOT)

print(f"📂 Arbeitsverzeichnis: {PROJECT_ROOT}\n")

# ================================
# Helfer
# ================================
def _transformers_version_check(min_ver="4.56.0"):
    import transformers as tf
    if V(tf.__version__) < V(min_ver):
        print(
            f"⚠️  Deine transformers-Version ist {tf.__version__} – "
            f"für gpt_oss bitte mind. {min_ver} installieren:\n"
            f"    pip install -U 'transformers>={min_ver}' kernels accelerate torch\n"
        )

def is_valid_training_model(model_dir: Path):
    """
    Prüft, ob ein lokales Modell grundsätzlich trainierbar ist:
    - keine GGUF/MLX Ordner
    - hat config.json
    - gpt_oss ist ERLAUBT (mit aktuellem transformers)
    """
    name = model_dir.name.lower()

    if "gguf" in name:
        return False, "GGUF-Format (nur Inferenz)"
    if "mlx" in name:
        return False, "MLX-Format (nur Apple-Inferenz)"

    cfg_file = model_dir / "config.json"
    if not cfg_file.exists():
        return False, "Keine config.json gefunden"

    try:
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "")

        # WICHTIG: gpt_oss NICHT mehr aussortieren – wird via transformers (>=4.56) + trust_remote_code geladen
        # Andere exotische Typen ggf. hier prüfen/blacklisten.
        return True, None
    except Exception as e:
        return False, f"Fehler beim Lesen der Config: {e}"

def list_models():
    """Listet alle verfügbaren Modelle in models/base auf."""
    base_dir = PROJECT_ROOT / "models" / "base"
    if not base_dir.exists():
        print(f"❌ Verzeichnis {base_dir} existiert nicht!")
        print("💡 Tipp: 'python scripts/download_model.py <repo-id>' verwenden.")
        sys.exit(1)

    all_models = [d for d in base_dir.iterdir() if d.is_dir()]
    models = [m for m in all_models if is_valid_training_model(m)[0]]

    if not models:
        print(f"❌ Keine trainierbaren Modelle in {base_dir} gefunden!")
        if all_models:
            print("\n⚠️  Gefundene, aber nicht-trainierbare Modelle:")
            for m in all_models:
                ok, reason = is_valid_training_model(m)
                if not ok:
                    print(f"    - {m.name} ({reason})")
        sys.exit(1)

    return sorted(models)

def list_datasets():
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.exists():
        print(f"❌ Verzeichnis {data_dir} existiert nicht!")
        sys.exit(1)

    ds = sorted(data_dir.glob("*.json"))
    if not ds:
        print(f"❌ Keine JSON-Datensätze in {data_dir} gefunden!")
        sys.exit(1)
    return ds

def select_from_list(items, item_type="Option"):
    print(f"\n{'='*60}")
    print(f"📋 Verfügbare {item_type}:")
    print(f"{'='*60}")
    for idx, item in enumerate(items, 1):
        print(f"  [{idx}] {item.name}")
    print(f"{'='*60}")

    while True:
        try:
            choice = input(f"\n👉 Wähle {item_type} (1-{len(items)}): ").strip()
            i = int(choice) - 1
            if 0 <= i < len(items):
                sel = items[i]
                print(f"✅ Ausgewählt: {sel.name}\n")
                return sel
            print(f"❌ Ungültige Auswahl! Bitte 1–{len(items)}.")
        except ValueError:
            print("❌ Ungültige Eingabe! Bitte Zahl eingeben.")
        except KeyboardInterrupt:
            print("\n\n❌ Abgebrochen!")
            sys.exit(0)

# ================================
# Interaktive Auswahl
# ================================
print("🚀 LoRA Training - Interaktive Konfiguration")
print("="*60)

available_models = list_models()
selected_model = select_from_list(available_models, "Modell")
MODEL_PATH = str(selected_model.absolute())

available_datasets = list_datasets()
selected_dataset = select_from_list(available_datasets, "Datensatz")
DATA_PATH = str(selected_dataset.absolute())

dataset_name = selected_dataset.stem
PERSONA = dataset_name[:-9] if dataset_name.endswith("_training") else dataset_name
OUTPUT_DIR = str(PROJECT_ROOT / "models" / "persona" / PERSONA)

print(f"\n{'='*60}")
print(f"📊 Training-Konfiguration:")
print(f"{'='*60}")
print(f"  🤖 Modell:    {selected_model.name}")
print(f"  📁 Datensatz: {selected_dataset.name}")
print(f"  💾 Output:    {OUTPUT_DIR}")
print(f"{'='*60}\n")

try:
    confirm = input("▶️  Training starten? (j/n): ").strip().lower()
    if confirm not in ("j", "ja", "y", "yes"):
        print("❌ Training abgebrochen!")
        sys.exit(0)
except KeyboardInterrupt:
    print("\n\n❌ Abgebrochen!")
    sys.exit(0)

print(f"\n🎯 Starte Training für {PERSONA}...\n")

# ================================
# 0) Env-Checks
# ================================
_transformers_version_check()

# ================================
# 1) Modell + Tokenizer laden
# ================================
print("📂 Loading model...")

# Config zuerst mit Remote-Code laden (ältere Transformers-Versionen stolpern sonst über model_type)
try:
    print("   → Lade Config...")
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=False)
    print(f"   → Config geladen: model_type={getattr(config, 'model_type', 'unknown')}")
except Exception as e:
    print(f"⚠️  AutoConfig konnte nicht geladen werden ({e}). "
          f"Bitte transformers aktualisieren oder Internetzugang prüfen.")
    raise

# GPU/Precision-Logik
has_cuda = torch.cuda.is_available()
print(f"   → CUDA verfügbar: {has_cuda}")

if has_cuda:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   → GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    bf16_ok = torch.cuda.get_device_capability(0)[0] >= 8  # Ampere+ kann BF16
    print(f"   → BF16 unterstützt: {bf16_ok}")
else:
    bf16_ok = False
    print("   ⚠️  Kein CUDA! Training auf CPU wird sehr langsam sein.")

use_bf16 = bf16_ok
use_fp16 = (has_cuda and not use_bf16)
print(f"   → Verwendete Precision: {'BF16' if use_bf16 else ('FP16' if use_fp16 else 'FP32')}")

# Laden mit besserem device_map für LoRA-Training
# "auto" kann auf CPU/Disk offloaden, was beim PEFT-Setup Probleme macht
print("   → Lade Modell-Weights...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config,
    device_map={"": 0} if has_cuda else "cpu",  # Forciere GPU 0 oder CPU, kein Offloading
    trust_remote_code=True,
    torch_dtype=(torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)),
    local_files_only=False,
)
print("   → Modell-Weights geladen")

print("   → Lade Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=False, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print("✅ Model & Tokenizer loaded")

# ================================
# 2) LoRA-Config
# ================================
print("\n🔧 Erstelle LoRA Adapter...")
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # ggf. up/down/gate_proj ergänzen
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

print("   → Wende LoRA auf Modell an...")
model = get_peft_model(model, lora_config)
print("✅ LoRA-Adapter erstellt")
print("\n🎯 Trainable Parameters:")
model.print_trainable_parameters()

# ================================
# 3) Dataset laden
# ================================
print(f"\n📊 Loading dataset: {DATA_PATH}")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
print(f"   → Dataset size: {len(dataset)} examples")
print(f"   → Beispiel-Keys: {list(dataset[0].keys())}")

# ================================
# 3.1) Prompt-Formatierung
# ================================
IS_GPT_OSS = (getattr(config, "model_type", "") == "gpt_oss")
print(f"\n🎨 Prompt-Format: {'GPT-OSS (Chat-Template)' if IS_GPT_OSS else 'Mistral (Instruct)'}")

def format_prompt(example):
    instr = example.get("instruction", "")
    inp = example.get("input", "")
    out = example.get("output", "")

    if IS_GPT_OSS:
        # Chat-Template für GPT-OSS verwenden
        user_txt = instr if not inp else f"{instr}\n{inp}"
        messages = [
            {"role": "user", "content": user_txt},
            {"role": "assistant", "content": out},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return text  # TRL ≥0.24 erwartet STRING statt Liste
    else:
        # Mistral/klassisch
        text = f"<s>[INST] {instr}\n{inp} [/INST]\n{out}</s>"
        return text  # TRL ≥0.24 erwartet STRING statt Liste

# ================================
# 4) Training-Args
# ================================
print("\n⚙️  Erstelle Training-Konfiguration...")

# Training-Precision:
args_fp16 = use_fp16
args_bf16 = use_bf16
print(f"   → Training Precision: {'BF16' if args_bf16 else ('FP16' if args_fp16 else 'FP32')}")

# Speicher sparen:
model.config.use_cache = False  # für Grad-Checkpointing
gradient_ckpt = True
print(f"   → Gradient Checkpointing: {gradient_ckpt}")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="constant",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch",
    fp16=args_fp16,
    bf16=args_bf16,
    max_grad_norm=0.3,
    report_to="none",
    gradient_checkpointing=gradient_ckpt,
    max_length=512,
)

# kleiner Stabilitäts-Tweak
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

print("\n🏗️  Erstelle SFTTrainer...")
print("   ⚠️  Dies kann einen Moment dauern (PEFT-Setup)...")

# WICHTIG: peft_config=None, da wir bereits get_peft_model() aufgerufen haben
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=None,
    formatting_func=format_prompt,
    processing_class=tokenizer,
    args=training_args,
)

print("✅ Trainer erstellt\n")
print("🏋️  Starting training...")
print("="*60)
trainer.train()

# ================================
# 5) Speichern
# ================================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Training complete! Adapter saved to: {OUTPUT_DIR}")
