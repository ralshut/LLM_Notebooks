# -*- coding: utf-8 -*-
"""
Einfacher Terminal-Chat mit einem LoRA-adaptierten Modell:
- Basis-Modell aus `models/base` auswählen
- Persona/Adapter aus `models/persona` wählen
- LoRA-Adapter per Peft laden und interaktiven Chat starten
- Mit/ohne Chat-History
- Debug-Modus um Prompts zu sehen
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_BASE_DIR = PROJECT_ROOT / "models" / "base"
PERSONA_DIR = PROJECT_ROOT / "models" / "persona"

os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INSTRUCTION_TEMPLATES= {
    "ac_host": "Du bist ein engagierter Assessment-Center-Host. Reagiere kurz, stelle exakt eine Folgefrage, führe das Gespräch.",
    "aufgabe_1a": "Du bist ein engagierter Assessment-Center-Host. Reagiere kurz, stelle exakt eine Folgefrage, führe das Gespräch.",
    "aufgabe_1b": "you are a teacher which is only capable to talk english, your name is Matthew McConnaghey and you cannot speak german",
    "ac_host_company": "Du bist ein engagierter Assessment-Center-Host. Reagiere kurz, stelle exakt eine Folgefrage, führe das Gespräch.",
    "angry_customer": "Du bist ein verärgerter Kunde. Reagiere auf die folgende Aussage des Servicemitarbeiters.",
    "scorer": "Bewerte die folgende Antwort eines Kandidaten immer als valides JSON.",
}


def list_dirs(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted([d for d in path.iterdir() if d.is_dir()])


def select_string(options: List[str], title: str) -> str:
    if not options:
        raise ValueError(f"Keine Optionen für '{title}' verfügbar.")

    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for idx, opt in enumerate(options, 1):
        print(f"[{idx:2d}] {opt}")
    print("=" * 72)

    while True:
        try:
            choice = int(input(f"👉 Auswahl (1-{len(options)}): ").strip())
            if 1 <= choice <= len(options):
                selected = options[choice - 1]
                print(f"✅ Ausgewählt: {selected}\n")
                return selected
            print(f"❌ Bitte eine Zahl zwischen 1 und {len(options)} eingeben.")
        except ValueError:
            print("❌ Ungültige Eingabe – bitte Zahl eingeben.")
        except KeyboardInterrupt:
            print("\nAbbruch durch Benutzer.")
            sys.exit(0)


def select_yes_no(question: str, default: bool = True) -> bool:
    """Ja/Nein Frage"""
    default_text = "J/n" if default else "j/N"
    while True:
        answer = input(f"{question} [{default_text}]: ").strip().lower()
        if not answer:
            return default
        if answer in ['j', 'ja', 'y', 'yes']:
            return True
        if answer in ['n', 'nein', 'no']:
            return False
        print("❌ Bitte 'j' oder 'n' eingeben.")


def select_option(options: List[Path], title: str) -> Path:
    if not options:
        print(f"❌ Keine Optionen für '{title}' gefunden.")
        sys.exit(1)

    print("\n" + "=" * 72)
    print(f"{title}")
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


def list_persona_adapters(persona: str) -> List[Path]:
    target = PERSONA_DIR / persona
    if not target.exists():
        return []
    return sorted([d for d in target.iterdir() if d.is_dir()])


def build_manual_prompt_mistral(
    conversation: List[dict],
    system_prompt: Optional[str],
    use_history: bool,
) -> str:
    """
    Baut Mistral-Format mit oder ohne History
    """
    if not use_history:
        # Nur letzte User-Message
        last_user = None
        for turn in reversed(conversation):
            if turn["role"] == "user":
                last_user = turn["content"]
                break
        
        if system_prompt:
            return f"<s>[INST] {system_prompt}\n{last_user} [/INST]"
        return f"<s>[INST] {last_user} [/INST]"
    
    # Mit History: alle Turns einbauen
    prompt_parts: List[str] = []
    conv_wo_system = [turn for turn in conversation if turn["role"] != "system"]
    
    i = 0
    first_turn = True
    while i < len(conv_wo_system):
        if conv_wo_system[i]["role"] == "user":
            user_text = conv_wo_system[i]["content"]
            
            # Erste User-Message: System Prompt mit einbauen
            if first_turn and system_prompt:
                prompt_parts.append(f"<s>[INST] {system_prompt}\n{user_text} [/INST]")
                first_turn = False
            else:
                prompt_parts.append(f"<s>[INST] {user_text} [/INST]")
            
            # Checke ob Assistant-Antwort folgt
            if i + 1 < len(conv_wo_system) and conv_wo_system[i + 1]["role"] == "assistant":
                prompt_parts.append(f" {conv_wo_system[i + 1]['content']}</s>")
                i += 2
            else:
                # Letzte User-Message ohne Antwort → offen lassen
                i += 1
        else:
            i += 1
    
    return "\n".join(prompt_parts)


def main() -> None:
    print("💬 LoRA Chat")
    print("=" * 72)
    print(f"📂 Basis-Modelle: {MODELS_BASE_DIR}")
    print(f"📂 Persona-Adapter: {PERSONA_DIR}\n")

    # Modell auswählen
    base_models = list_dirs(MODELS_BASE_DIR)
    if not base_models:
        print(f"❌ Keine Basis-Modelle in {MODELS_BASE_DIR} gefunden.")
        sys.exit(1)
    base_model_path = select_option(base_models, "Basis-Modell wählen")
    
    # Persona auswählen
    available_personas = sorted(INSTRUCTION_TEMPLATES.keys())
    if not available_personas:
        print("❌ Keine Personas in den Instruktionsvorlagen gefunden.")
        sys.exit(1)
    persona_name = select_string(available_personas, "Persona wählen")
    persona_instruction = INSTRUCTION_TEMPLATES.get(persona_name)

    # LoRA Adapter auswählen (optional)
    adapter_path: Optional[Path] = None
    adapters_for_persona = list_persona_adapters(persona_name)
    if adapters_for_persona:
        mode_choice = select_string(
            [
                "Nur Systemprompt (kein LoRA)",
                "LoRA-Adapter verwenden",
            ],
            f"Modus für Persona '{persona_name}' wählen",
        )
        if mode_choice.startswith("LoRA"):
            adapter_path = select_option(adapters_for_persona, f"LoRA-Adapter für {persona_name} wählen")
    else:
        print(f"ℹ️  Keine LoRA-Adapter für Persona '{persona_name}' gefunden – verwende nur Systemprompt.")

    # Chat-Einstellungen
    use_history = select_yes_no("💭 Chat-History verwenden?", default=True)
    debug_mode = select_yes_no("🐛 Debug-Modus (Prompts anzeigen)?", default=False)

    print("\n📥 Lade Basis-Modell & Tokenizer …")
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, local_files_only=True)
    
    if adapter_path and adapter_path.joinpath("chat_template.jinja").exists():
        try:
            tokenizer.chat_template = adapter_path.joinpath("chat_template.jinja").read_text(encoding="utf-8")
        except Exception:
            pass
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    has_cuda = torch.cuda.is_available()
    torch_dtype = torch.bfloat16 if has_cuda and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16 if has_cuda else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch_dtype,
        device_map=0 if has_cuda else "cpu",
    )

    if adapter_path:
        print(f"🔌 Lade LoRA-Adapter ({persona_name}) …")
        model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch_dtype, local_files_only=True)
        model = model.merge_and_unload()
        model.eval()
    else:
        print("ℹ️  Kein LoRA-Adapter geladen – verwende Basis-Modell mit Persona-Systemprompt.")
        model.eval()

    pipeline = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
    )

    print("=" * 72)
    print(f"💬 Chat gestartet – Persona: {persona_name}")
    print(f"📜 Chat-History: {'✅ Aktiviert' if use_history else '❌ Deaktiviert'}")
    print(f"🐛 Debug-Modus: {'✅ Aktiviert' if debug_mode else '❌ Deaktiviert'}")
    if persona_instruction:
        print(f"Systemprompt: {persona_instruction}")
    print("Tippe `exit`, `quit` oder `q`, um zu beenden.")
    print("=" * 72)

    conversation: List[dict] = []
    if persona_instruction:
        conversation.append({"role": "system", "content": persona_instruction})

    try:
        while True:
            user_input = input("\n🧑 Du: ").strip()
            if user_input.lower() in {"exit", "quit", "q"}:
                print("👋 Chat beendet.")
                break

            conversation.append({"role": "user", "content": user_input})

            # Prompt bauen
            use_template = bool(getattr(tokenizer, "chat_template", None))
            
            if use_template:
                # Llama/Qwen: Chat Template nutzen
                if use_history:
                    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                else:
                    # Nur letzte Message
                    conv_single = [{"role": "system", "content": persona_instruction}] if persona_instruction else []
                    conv_single.append(conversation[-1])  # Letzte User-Message
                    prompt = tokenizer.apply_chat_template(conv_single, tokenize=False, add_generation_prompt=True)
            else:
                # Mistral: Manuelles Format
                prompt = build_manual_prompt_mistral(conversation, persona_instruction, use_history)

            if debug_mode:
                print("\n" + "="*72)
                print("🐛 DEBUG: Generierter Prompt:")
                print("="*72)
                print(prompt)
                print("="*72 + "\n")

            outputs = pipeline(
                prompt,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )
            
            generated = outputs[0]["generated_text"]
            response = generated[len(prompt):].strip() if generated.startswith(prompt) else generated.strip()
            
            if not response:
                response = "(keine Ausgabe)"

            print(f"🤖 {persona_name}: {response}")

            conversation.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        print("\n👋 Chat unterbrochen.")


if __name__ == "__main__":
    main()