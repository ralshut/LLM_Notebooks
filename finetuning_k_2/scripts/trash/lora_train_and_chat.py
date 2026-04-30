# -*- coding: utf-8 -*-
"""
End-to-end Skript:
1. Persona-Namen angeben
2. Trainings-Datensatz aus ./dataset auswählen
3. Basis-Modell aus ./models/base wählen
4. LoRA-SFT durchführen (nur Trainingssplit)
5. Ergebnis in ./models/persona/<persona>/sft_<timestamp> speichern
6. Optional direkt mit dem frisch trainierten Adapter chatten
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
)
from trl import SFTConfig, SFTTrainer

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_BASE_DIR = PROJECT_ROOT / "models" / "base"
PERSONA_DIR = PROJECT_ROOT / "models" / "persona"
DATASET_DIR = PROJECT_ROOT / "dataset"
SYSTEMPROMPT_DIR = PROJECT_ROOT / "systemprompts"

os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from synthetic_generation_2.ollama_dataset_builder import INSTRUCTION_TEMPLATES
except Exception:
    INSTRUCTION_TEMPLATES = {}


def list_dirs(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted([d for d in path.iterdir() if d.is_dir()])


def list_files(path: Path, suffix: str) -> List[Path]:
    if not path.exists():
        return []
    return sorted([f for f in path.iterdir() if f.is_file() and f.suffix == suffix])


def select_option(options: List[Path], title: str) -> Path:
    if not options:
        print(f"❌ Keine Optionen für '{title}' gefunden.")
        sys.exit(1)

    print("\n" + "=" * 72)
    print(title)
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


def yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[J/n]" if default else "[j/N]"
    while True:
        try:
            choice = input(f"{prompt} {suffix}: ").strip().lower()
        except KeyboardInterrupt:
            print("\nAbbruch durch Benutzer.")
            sys.exit(0)
        if not choice:
            return default
        if choice in {"j", "ja", "y", "yes"}:
            return True
        if choice in {"n", "nein", "no"}:
            return False
        print("❌ Bitte mit 'j' oder 'n' antworten.")


def ensure_dirs() -> None:
    MODELS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    PERSONA_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEMPROMPT_DIR.mkdir(parents=True, exist_ok=True)


def build_manual_prompt(conversation: List[dict], system_prompt: Optional[str]) -> str:
    segments: List[str] = []
    pending_system: Optional[str] = system_prompt

    for message in conversation:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            if pending_system:
                combined_content = f"{pending_system}\n{content}"
                segments.append(f"<s>[INST] {combined_content} [/INST]\n")
                pending_system = None
            else:
                segments.append(f"<s>[INST] {content} [/INST]\n")
        elif role == "assistant":
            if not segments:
                continue
            segments[-1] = f"{segments[-1]}{content}</s>"
        elif role == "system":
            pending_system = content

    if segments and not segments[-1].endswith("</s>"):
        segments[-1] = f"{segments[-1]}</s>"

    return "".join(segments)


def extract_text(obj) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("content", "text", "generated_text", "message"):
            if key in obj:
                text = extract_text(obj[key])
                if text:
                    return text
        return None
    if isinstance(obj, list):
        for item in reversed(obj):
            text = extract_text(item)
            if text:
                return text
        return None
    return None

    MODELS_BASE_DIR.mkdir(parents=True, exist_ok=True)
    PERSONA_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEMPROMPT_DIR.mkdir(parents=True, exist_ok=True)


def format_example(
    example: dict,
    is_gpt_oss: bool,
    tokenizer: AutoTokenizer,
    training_mode: str,
) -> str:
    if training_mode == "multi_turn":
        messages = example.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("Datensatz-Eintrag enthält keine 'messages' für Multi-Turn-Training.")
        if is_gpt_oss:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return build_manual_prompt(
            [msg for msg in messages if msg.get("role") != "system"],
            next((msg.get("content") for msg in messages if msg.get("role") == "system"), None),
        )

    instruction = example.get("instruction", "")
    user_input = example.get("input", "")
    output = example.get("output", "")

    if is_gpt_oss:
        merged_prompt = instruction if not user_input else f"{instruction}\n{user_input}"
        messages = [
            {"role": "user", "content": merged_prompt},
            {"role": "assistant", "content": output},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return f"<s>[INST] {instruction}\n{user_input} [/INST]\n{output}</s>"


def prompt_lora_params() -> dict:
    print("\n2️⃣ Wahl der LoRA-Hyperparameter")
    print("────────────────────────────────────")
    print("🔹 Rank r (Typisch 4–16) – höher = mehr Kapazität, aber mehr Overfitting.")
    print("🔹 Alpha (Typisch 4–32) – Gewicht der LoRA-Änderungen.")
    print("🔹 Dropout (Typisch 0–0.1) – reduziert Overfitting bei kleinen Datensätzen.")

    def prompt_numeric(prompt: str, default: float, cast_type, min_value=None, max_value=None):
        while True:
            try:
                raw = input(f"{prompt} [Default: {default}]: ").strip()
                if not raw:
                    value = default
                else:
                    value = cast_type(raw)
                if min_value is not None and value < min_value:
                    print(f"❌ Wert muss mindestens {min_value} sein.")
                    continue
                if max_value is not None and value > max_value:
                    print(f"❌ Wert darf höchstens {max_value} sein.")
                    continue
                return value
            except ValueError:
                print("❌ Ungültige Eingabe – bitte Zahl eingeben.")
            except KeyboardInterrupt:
                print("\nAbbruch durch Benutzer.")
                sys.exit(0)

    r = prompt_numeric("Rank r", default=8, cast_type=int, min_value=1)
    alpha = prompt_numeric("Alpha", default=16, cast_type=int, min_value=1)
    dropout = prompt_numeric("Dropout", default=0.05, cast_type=float, min_value=0.0, max_value=0.5)

    print(f"✅ LoRA-Parameter: r={r}, alpha={alpha}, dropout={dropout}\n")
    return {"r": r, "alpha": alpha, "dropout": dropout}


def train_lora(
    persona_name: str,
    dataset_path: Path,
    base_model_path: Path,
    training_mode: str,
    lora_params: dict,
    system_prompt_name: Optional[str] = None,
) -> Path:
    ensure_dirs()

    print("\n📥 Lade Basis-Modell & Tokenizer …")
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    has_cuda = torch.cuda.is_available()
    torch_dtype = (
        torch.bfloat16
        if has_cuda and torch.cuda.get_device_capability(0)[0] >= 8
        else (torch.float16 if has_cuda else torch.float32)
    )
    device_map = {"": 0} if has_cuda else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch_dtype,
    )
    model.config.use_cache = False

    print("🔧 Initialisiere LoRA …")
    lora_config = LoraConfig(
        r=lora_params["r"],
        lora_alpha=lora_params["alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=lora_params["dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"\n📊 Lade Datensatz: {dataset_path.name}")
    dataset = load_dataset("json", data_files={"train": str(dataset_path)}, split="train")
    if len(dataset) == 0:
        print("❌ Datensatz enthält keine Beispiele.")
        sys.exit(1)
    print(f"   → {len(dataset)} Trainingsbeispiele")

    sample = dataset[0]
    if training_mode == "multi_turn":
        if "messages" not in sample:
            print("❌ Der ausgewählte Datensatz enthält keine 'messages'-Einträge. Bitte Multi-Turn-Datensatz auswählen.")
            sys.exit(1)
    else:
        if "instruction" not in sample or "output" not in sample:
            print("❌ Der ausgewählte Datensatz enthält nicht die erwarteten 'instruction'/'output'-Felder.")
            sys.exit(1)

    is_gpt_oss = getattr(config, "model_type", "") == "gpt_oss"

    training_args = SFTConfig(
        output_dir=str(PERSONA_DIR / persona_name),
        num_train_epochs=3,
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

    print("\n🏗️ Starte Training …")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=lambda ex: format_example(ex, is_gpt_oss, tokenizer, training_mode),
        args=training_args,
        peft_config=None,
        processing_class=tokenizer,
    )

    train_result = trainer.train()
    print("✅ Training abgeschlossen.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PERSONA_DIR / persona_name / f"sft_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"💾 Speichere LoRA-Adapter nach {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    summary = {
        "persona": persona_name,
        "base_model": base_model_path.name,
        "dataset": dataset_path.name,
        "timestamp": timestamp,
        "system_prompt": system_prompt_name,
        "training_mode": training_mode,
        "lora_rank": lora_params["r"],
        "lora_alpha": lora_params["alpha"],
        "lora_dropout": lora_params["dropout"],
        "train_runtime": getattr(train_result, "metrics", {}).get("train_runtime"),
        "train_samples_per_second": getattr(train_result, "metrics", {}).get("train_samples_per_second"),
        "num_examples": len(dataset),
    }
    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return output_dir


def chat_with_adapter(
    persona_name: str,
    base_model_path: Path,
    adapter_path: Path,
    system_prompt: Optional[str],
    temperature: float = 0.7,
    top_p: float = 0.9,
    enable_history: bool = True,
) -> None:
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, local_files_only=True)
    tokenizer.chat_template = None

    chat_template_file = adapter_path / "chat_template.jinja"
    if chat_template_file.exists():
        try:
            tokenizer.chat_template = chat_template_file.read_text(encoding="utf-8")
        except Exception:
            pass

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    has_cuda = torch.cuda.is_available()
    torch_dtype = (
        torch.bfloat16
        if has_cuda and torch.cuda.get_device_capability(0)[0] >= 8
        else (torch.float16 if has_cuda else torch.float32)
    )

    print("\n🔄 Lade Basis-Modell & Adapter für den Chat …")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch_dtype,
        device_map="auto" if has_cuda else "cpu",
    )
    model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch_dtype, local_files_only=True)
    model = model.merge_and_unload()
    model.eval()

    pipeline = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
    )

    conversation: List[dict] = []
    if enable_history and system_prompt:
        conversation.append({"role": "system", "content": system_prompt})

    print("\n" + "=" * 72)
    print(f"💬 Chat gestartet – Persona: {persona_name}")
    print("Tippe `exit`, `quit` oder `q`, um zu beenden.")
    print("=" * 72)

    try:
        while True:
            user_input = input("\n🧑 Du: ").strip()
            if user_input.lower() in {"exit", "quit", "q"}:
                print("👋 Chat beendet.")
                break

            if enable_history:
                conversation.append({"role": "user", "content": user_input})
                conv_for_prompt = conversation
            else:
                conv_for_prompt = []
                if system_prompt:
                    conv_for_prompt.append({"role": "system", "content": system_prompt})
                conv_for_prompt.append({"role": "user", "content": user_input})

            use_template = bool(getattr(tokenizer, "chat_template", None))
            if use_template:
                prompt = tokenizer.apply_chat_template(conv_for_prompt, tokenize=False, add_generation_prompt=True)
            else:
                conv_no_system = [turn for turn in conv_for_prompt if turn["role"] != "system"]
                prompt = build_manual_prompt(conv_no_system, system_prompt)

            outputs = pipeline(
                prompt,
                max_new_tokens=256,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
            )
            generated = outputs[0]["generated_text"]

            response: Optional[str]
            if isinstance(generated, str) and generated.startswith(prompt):
                response = generated[len(prompt) :].strip()
            else:
                response = extract_text(generated)
                if response is not None:
                    response = response.strip()

            if not response:
                print("[DEBUG] Konnte keine Antwort extrahieren. Rohdaten:", outputs[0])
                response = "(keine Ausgabe)"

            print(f"🤖 {persona_name}: {response}")
            if enable_history:
                conversation.append({"role": "assistant", "content": response})
    except KeyboardInterrupt:
        print("\n👋 Chat unterbrochen.")


def handle_training() -> None:
    persona_name = input("🔤 Wie soll die Persona heißen? ").strip()
    if not persona_name:
        print("❌ Persona-Name darf nicht leer sein.")
        return

    dataset_files = list_files(DATASET_DIR, ".jsonl")
    dataset_path = select_option(dataset_files, "Trainingsdatensatz aus 'dataset/' wählen")

    training_mode = prompt_training_mode()

    lora_params = prompt_lora_params()

    base_models = list_dirs(MODELS_BASE_DIR)
    base_model_path = select_option(base_models, "Basis-Modell aus 'models/base/' wählen")

    # Runs auswählen
    while True:
        try:
            runs = int(input("🔁 Wie viele Trainingsdurchläufe sollen hintereinander ausgeführt werden? ").strip())
            if runs < 1:
                print("❌ Bitte eine Zahl ≥ 1 eingeben.")
                continue
            break
        except ValueError:
            print("❌ Ungültige Eingabe – bitte Zahl eingeben.")
        except KeyboardInterrupt:
            print("\nAbbruch durch Benutzer.")
            return

    for idx in range(1, runs + 1):
        print(f"\n=== Trainingslauf {idx}/{runs} ===")
        adapter_dir = train_lora(persona_name, dataset_path, base_model_path, training_mode, lora_params)
        print("✨ Fertig. Adapter liegt unter:", adapter_dir)

        # Speicher aufräumen, damit der nächste Durchlauf nicht den GPU-Speicher sprengt
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if idx == runs and yes_no("Möchtest du jetzt direkt mit dem neuen Adapter chatten?", default=True):
            base_models = list_dirs(MODELS_BASE_DIR)
            base_model_path = select_option(base_models, "Basis-Modell aus 'models/base/' wählen")
            system_prompt_content: Optional[str] = None
            system_prompt_files = list_files(SYSTEMPROMPT_DIR, ".txt")
            if system_prompt_files:
                system_prompt_options = ["Kein Systemprompt verwenden"]
                system_prompt_mapping: dict[int, Optional[Path]] = {1: None}
                for idx_opt, file in enumerate(system_prompt_files, start=2):
                    system_prompt_options.append(file.name)
                    system_prompt_mapping[idx_opt] = file

                print("\n" + "=" * 72)
                print("Systemprompt zum Chatten auswählen")
                print("=" * 72)
                for idx_opt, label in enumerate(system_prompt_options, start=1):
                    print(f"[{idx_opt:2d}] {label}")
                print("=" * 72)

                while True:
                    try:
                        choice = int(input(f"👉 Auswahl (1-{len(system_prompt_options)}): ").strip())
                        if 1 <= choice <= len(system_prompt_options):
                            break
                        print(f"❌ Bitte eine Zahl zwischen 1 und {len(system_prompt_options)} eingeben.")
                    except ValueError:
                        print("❌ Ungültige Eingabe – bitte Zahl eingeben.")
                    except KeyboardInterrupt:
                        print("\nAbbruch durch Benutzer.")
                        return

                selected_prompt_path = system_prompt_mapping.get(choice)
                if selected_prompt_path is not None:
                    system_prompt_content = selected_prompt_path.read_text(encoding="utf-8").strip()
            if system_prompt_content is None and persona_name in INSTRUCTION_TEMPLATES:
                system_prompt_content = INSTRUCTION_TEMPLATES[persona_name]

            temperature = prompt_temperature()
            top_p = prompt_top_p()
            enable_history = yes_no("Soll der Chat die bisherigen Nachrichten berücksichtigen?", default=True)
            chat_with_adapter(
                persona_name,
                base_model_path,
                adapter_dir,
                system_prompt_content,
                temperature=temperature,
                top_p=top_p,
                enable_history=enable_history,
            )


def handle_chat_only() -> None:
    personas = list_dirs(PERSONA_DIR)
    if not personas:
        print("ℹ️  Noch keine LoRA-Personas vorhanden. Bitte zuerst trainieren.")
        return

    persona_dir = select_option(personas, "Persona wählen")
    runs = list_dirs(persona_dir)
    if not runs:
        print("ℹ️  In diesem Persona-Ordner liegen keine Adapter. Bitte training ausführen.")
        return

    adapter_path = select_option(runs, "Adapter auswählen")

    base_models = list_dirs(MODELS_BASE_DIR)
    base_model_path = select_option(base_models, "Basis-Modell aus 'models/base/' wählen")

    system_prompt_file = adapter_path / "training_summary.json"
    system_prompt_content: Optional[str] = None
    if yes_no("Möchtest du einen Systemprompt auswählen?", default=True):
        system_prompt_files = list_files(SYSTEMPROMPT_DIR, ".txt")
        system_prompt_options = ["Kein Systemprompt verwenden"]
        system_prompt_mapping: dict[int, Optional[Path]] = {1: None}
        if system_prompt_files:
            for idx, file in enumerate(system_prompt_files, start=2):
                system_prompt_options.append(file.name)
                system_prompt_mapping[idx] = file

        print("\n" + "=" * 72)
        print("Systemprompt zum Chatten auswählen")
        print("=" * 72)
        for idx, label in enumerate(system_prompt_options, start=1):
            print(f"[{idx:2d}] {label}")
        print("=" * 72)

        while True:
            try:
                choice = int(input(f"👉 Auswahl (1-{len(system_prompt_options)}): ").strip())
                if 1 <= choice <= len(system_prompt_options):
                    break
                print(f"❌ Bitte eine Zahl zwischen 1 und {len(system_prompt_options)} eingeben.")
            except ValueError:
                print("❌ Ungültige Eingabe – bitte Zahl eingeben.")
            except KeyboardInterrupt:
                print("\nAbbruch durch Benutzer.")
                return

        selected_prompt_path = system_prompt_mapping.get(choice)
        if selected_prompt_path is not None:
            system_prompt_content = selected_prompt_path.read_text(encoding="utf-8").strip()
    elif system_prompt_file.exists():
        try:
            summary = json.loads(system_prompt_file.read_text(encoding="utf-8"))
            name = summary.get("system_prompt")
            if name:
                system_prompt_content = INSTRUCTION_TEMPLATES.get(summary["persona"], None)
                if system_prompt_content is None and not name.startswith("builtin:"):
                    path = SYSTEMPROMPT_DIR / name
                    if path.exists():
                        system_prompt_content = path.read_text(encoding="utf-8").strip()
        except Exception:
            pass

    temperature = prompt_temperature()
    top_p = prompt_top_p()
    enable_history = yes_no("Soll der Chat die bisherigen Nachrichten berücksichtigen?", default=True)
    chat_with_adapter(
        persona_dir.name,
        base_model_path,
        adapter_path,
        system_prompt_content,
        temperature=temperature,
        top_p=top_p,
        enable_history=enable_history,
    )


def prompt_temperature() -> float:
    while True:
        try:
            print("🔥 Temperatur steuert die Kreativität: niedrige Werte → deterministischer, hohe Werte → vielfältiger.")
            value = input("Temperatur (0.01 – 2.0, Enter für 0.7): ").strip()
            if not value:
                return 0.7
            temp = float(value)
            if 0.01 <= temp <= 2.0:
                return temp
            print("❌ Bitte Wert zwischen 0.01 und 2.0 eingeben.")
        except ValueError:
            print("❌ Ungültige Zahl. Bitte erneut versuchen.")
        except KeyboardInterrupt:
            print("\nAbbruch durch Benutzer.")
            sys.exit(0)


def prompt_top_p() -> float:
    while True:
        try:
            print("🎯 Top-p begrenzt die Auswahl auf die wahrscheinlichsten Tokens (Summe der Wahrscheinlichkeiten).")
            value = input("Top-p (0.0 – 1.0, Enter für 0.9): ").strip()
            if not value:
                return 0.9
            top_p = float(value)
            if 0.0 < top_p <= 1.0:
                return top_p
            print("❌ Bitte Wert zwischen 0.0 (exklusiv) und 1.0 eingeben.")
        except ValueError:
            print("❌ Ungültige Zahl. Bitte erneut versuchen.")
        except KeyboardInterrupt:
            print("\nAbbruch durch Benutzer.")
            sys.exit(0)


def prompt_training_mode() -> str:
    options = [
        "Single-Turn (instruction/input/output)",
        "Multi-Turn (messages-Konversation)",
    ]
    print("\n" + "=" * 72)
    print("Trainingsmodus wählen")
    print("=" * 72)
    for idx, label in enumerate(options, start=1):
        print(f"[{idx:2d}] {label}")
    print("=" * 72)

    while True:
        try:
            choice = int(input(f"👉 Auswahl (1-{len(options)}): ").strip())
            if choice == 1:
                return "single_turn"
            if choice == 2:
                return "multi_turn"
            print(f"❌ Bitte eine Zahl zwischen 1 und {len(options)} eingeben.")
        except ValueError:
            print("❌ Ungültige Eingabe – bitte Zahl eingeben.")
        except KeyboardInterrupt:
            print("\nAbbruch durch Benutzer.")
            sys.exit(0)

def main() -> None:
    ensure_dirs()

    while True:
        print("\n" + "=" * 72)
        print("LoRA Training & Chat – Hauptmenü")
        print("=" * 72)
        print("[1] Training durchführen")
        print("[2] Mit bestehender Persona chatten")
        print("[3] Beenden")
        print("=" * 72)

        try:
            choice = input("👉 Auswahl: ").strip()
        except KeyboardInterrupt:
            print("\nProgramm beendet.")
            sys.exit(0)

        if choice == "1":
            handle_training()
        elif choice == "2":
            handle_chat_only()
        elif choice == "3":
            print("👋 Auf Wiedersehen!")
            break
        else:
            print("❌ Ungültige Auswahl. Bitte 1, 2 oder 3 eingeben.")


if __name__ == "__main__":
    main()

