import os
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, hf_hub_download
import argparse

# ============================================
# Config
# ============================================
# Projekt-Root ermitteln (ein Verzeichnis über scripts/)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_MODEL_DIR = PROJECT_ROOT / "models" / "base"

# Erstelle models/base/ Verzeichnis falls nicht vorhanden
BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"📂 Modelle werden gespeichert in: {BASE_MODEL_DIR.absolute()}")

# ============================================
# Argument Parser
# ============================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face Modelle ins lokale base/ Verzeichnis"
    )
    parser.add_argument(
        "model_name",
        type=str,
        help="Name des Hugging Face Modells (z.B. 'mistralai/Mistral-7B-Instruct-v0.2')"
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Nur Modell herunterladen, keinen Tokenizer"
    )
    parser.add_argument(
        "--tokenizer-only",
        action="store_true",
        help="Nur Tokenizer herunterladen, kein Modell"
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16", "auto"],
        help="Torch Datentyp für das Modell (default: float16)"
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Raw Download (snapshot_download) für Custom-Architekturen wie gpt-oss"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Spezifische Datei herunterladen (z.B. 'gpt-oss-20b-Q8_0.gguf' für GGUF-Quantisierungen)"
    )
    parser.add_argument(
        "--allow-patterns",
        type=str,
        default=None,
        help="Dateimuster für Download (z.B. '*.gguf' oder '*.safetensors')"
    )
    
    return parser.parse_args()

# ============================================
# Download Funktionen
# ============================================
def download_single_file(model_name: str, filename: str, local_path: Path):
    """
    Lädt eine einzelne Datei herunter (z.B. GGUF-Quantisierung).
    """
    print(f"📥 Lade einzelne Datei herunter: {filename}")
    
    try:
        # Erstelle Verzeichnis
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Lade Datei herunter
        downloaded_path = hf_hub_download(
            repo_id=model_name,
            filename=filename,
            local_dir=local_path,
            local_dir_use_symlinks=False,  # Windows-kompatibel
            resume_download=True
        )
        
        print(f"✅ Datei erfolgreich heruntergeladen!")
        print(f"📂 Gespeichert als: {downloaded_path}")
        return downloaded_path
        
    except Exception as e:
        print(f"\n❌ Fehler beim Download: {str(e)}")
        raise

def download_model_raw(model_name: str, local_path: Path, allow_patterns: str = None):
    """
    Raw Download für Custom-Architekturen (z.B. gpt-oss).
    Lädt alle Dateien oder gefilterte Dateien ohne Transformers-Verarbeitung herunter.
    """
    print("📥 Verwende Raw Download (snapshot_download)...")
    if allow_patterns:
        print(f"🔍 Filter: {allow_patterns}")
    else:
        print("⚠️  Für Custom-Architekturen wie gpt-oss")
    
    try:
        kwargs = {
            "repo_id": model_name,
            "local_dir": local_path,
            "local_dir_use_symlinks": False,  # Windows-kompatibel
            "resume_download": True
        }
        
        if allow_patterns:
            kwargs["allow_patterns"] = allow_patterns
        
        snapshot_download(**kwargs)
        
        print(f"✅ Modell erfolgreich heruntergeladen!")
        return str(local_path.absolute())
        
    except Exception as e:
        print(f"\n❌ Fehler beim Raw Download: {str(e)}")
        raise

def download_model(model_name: str, no_tokenizer: bool = False, tokenizer_only: bool = False, 
                   torch_dtype: str = "float16", raw: bool = False, file: str = None, 
                   allow_patterns: str = None):
    """
    Lädt ein Hugging Face Modell herunter und speichert es im models/base/ Verzeichnis.
    
    Args:
        model_name: Name des Modells auf Hugging Face (z.B. "mistralai/Mistral-7B-Instruct-v0.2")
        no_tokenizer: Wenn True, wird kein Tokenizer heruntergeladen
        tokenizer_only: Wenn True, wird nur der Tokenizer heruntergeladen
        torch_dtype: Datentyp für das Modell
        raw: Wenn True, wird snapshot_download verwendet (für Custom-Architekturen)
        file: Spezifische Datei zum Herunterladen (z.B. GGUF)
        allow_patterns: Dateimuster für gefilterten Download
    """
    # Erstelle sauberen Ordnernamen aus model_name (ersetze / durch _)
    safe_model_name = model_name.replace("/", "_")
    local_path = BASE_MODEL_DIR / safe_model_name
    
    print(f"\n{'='*60}")
    print(f"🎯 Modell: {model_name}")
    print(f"💾 Lokaler Pfad: {local_path}")
    if file:
        print(f"📄 Datei: {file}")
    if allow_patterns:
        print(f"🔍 Muster: {allow_patterns}")
    print(f"{'='*60}\n")
    
    # Torch dtype mapping
    dtype_map = {
        "float16": "torch.float16",
        "float32": "torch.float32", 
        "bfloat16": "torch.bfloat16",
        "auto": "auto"
    }
    
    try:
        # Einzelne Datei herunterladen (z.B. GGUF)
        if file:
            downloaded_path = download_single_file(model_name, file, local_path)
            
            print(f"\n{'='*60}")
            print(f"✨ Download abgeschlossen!")
            print(f"📂 Gespeichert in: {local_path.absolute()}")
            print(f"📄 Datei: {os.path.basename(downloaded_path)}")
            print(f"{'='*60}\n")
            
            # Unterschiedliche Hinweise je nach Dateityp
            if file.endswith('.gguf'):
                print("💡 GGUF-Datei für llama.cpp, Ollama oder LM Studio:")
                print(f'   # Mit Ollama:')
                print(f'   ollama create mein-modell -f Modelfile')
                print(f'   # Mit llama.cpp:')
                print(f'   ./main -m "{downloaded_path}" -p "Dein Prompt"')
            else:
                print("💡 Datei erfolgreich heruntergeladen!")
            print()
            
            return downloaded_path
        
        # Raw Download für Custom-Architekturen oder mit Dateimuster
        if raw or allow_patterns:
            download_model_raw(model_name, local_path, allow_patterns)
            
            print(f"\n{'='*60}")
            print(f"✨ Download abgeschlossen!")
            print(f"📂 Gespeichert in: {local_path.absolute()}")
            print(f"{'='*60}\n")
            
            print("💡 So verwendest du das Modell in deinem Code:")
            print(f'   from transformers import AutoModelForCausalLM, AutoTokenizer')
            print(f'   model = AutoModelForCausalLM.from_pretrained("{local_path.absolute()}", trust_remote_code=True)')
            print(f'   tokenizer = AutoTokenizer.from_pretrained("{local_path.absolute()}", trust_remote_code=True)')
            print()
            
            return str(local_path.absolute())
        
        # Standard Download mit Transformers
        # Modell herunterladen
        if not tokenizer_only:
            print("📥 Lade Modell herunter...")
            import torch
            torch_dtype_value = eval(dtype_map[torch_dtype]) if torch_dtype != "auto" else "auto"
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype_value,
                device_map="cpu",  # Auf CPU laden zum Speichern
                trust_remote_code=True
            )
            
            print(f"💾 Speichere Modell nach {local_path}...")
            model.save_pretrained(local_path)
            print("✅ Modell erfolgreich gespeichert!")
            
            # Speicher freigeben
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Tokenizer herunterladen
        if not no_tokenizer:
            print("\n📥 Lade Tokenizer herunter...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            print(f"💾 Speichere Tokenizer nach {local_path}...")
            tokenizer.save_pretrained(local_path)
            print("✅ Tokenizer erfolgreich gespeichert!")
        
        print(f"\n{'='*60}")
        print(f"✨ Download abgeschlossen!")
        print(f"📂 Gespeichert in: {local_path.absolute()}")
        print(f"{'='*60}\n")
        
        # Nutzungshinweis
        print("💡 So verwendest du das Modell in deinem Code:")
        print(f'   model = AutoModelForCausalLM.from_pretrained("{local_path.absolute()}", trust_remote_code=True)')
        print(f'   tokenizer = AutoTokenizer.from_pretrained("{local_path.absolute()}", trust_remote_code=True)')
        print()
        
        return str(local_path.absolute())
        
    except Exception as e:
        error_msg = str(e).lower()
        print(f"\n❌ Fehler beim Download: {str(e)}")
        
        # Spezifische Fehlermeldungen
        if "does not recognize this architecture" in error_msg or "model type" in error_msg:
            print("\n💡 Tipp: Dieses Modell hat eine Custom-Architektur.")
            print("   Versuche es mit der --raw Option:")
            print(f"   python download_model.py {model_name} --raw")
        
        print("\nMögliche Ursachen:")
        print("  - Modell hat Custom-Architektur (versuche --raw)")
        print("  - Modellname falsch geschrieben")
        print("  - Modell erfordert Authentifizierung (huggingface-cli login)")
        print("  - Nicht genug Speicherplatz")
        print("  - Keine Internetverbindung")
        sys.exit(1)

# ============================================
# Main
# ============================================
if __name__ == "__main__":
    args = parse_args()
    
    if args.tokenizer_only and args.no_tokenizer:
        print("❌ Fehler: --tokenizer-only und --no-tokenizer können nicht gleichzeitig verwendet werden!")
        sys.exit(1)
    
    if args.file and (args.tokenizer_only or args.no_tokenizer):
        print("⚠️  Warnung: --file überschreibt --tokenizer-only und --no-tokenizer")
    
    download_model(
        model_name=args.model_name,
        no_tokenizer=args.no_tokenizer,
        tokenizer_only=args.tokenizer_only,
        torch_dtype=args.torch_dtype,
        raw=args.raw,
        file=args.file,
        allow_patterns=args.allow_patterns
    )

