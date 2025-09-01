# main.py
import sys
import os
from pathlib import Path

def setup_python_path():
    """Ensure we can import modules from this folder (and its parent)."""
    here = Path(__file__).parent.resolve()
    parent = here.parent
    for p in (here, parent):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

def _import_qa_main():
    """
    Try several import patterns so this launcher works
    whether Model.py is at project root OR in an AI/ package.
    """
    # 1) Current uploaded setup: Model.py next to main.py
    try:
        from Model import main as qa_main  # type: ignore
        return qa_main
    except ImportError:
        pass

    # 2) Package-style: AI/Model.py with __init__.py present
    try:
        from AI.Model import main as qa_main  # type: ignore
        return qa_main
    except ImportError:
        pass

    # 3) Last resort: importlib with explicit paths
    import importlib.util
    here=Path(__file__).parent.resolve()
    candidates = [
        here / "Model.py",
        here / "AI" / "Model.py",
    ]
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("dynamic_Model", str(path))
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "main"):
                    return getattr(mod, "main")

    # 4) Re-exported main from AI/__init__.py
    try:
        from AI import main as qa_main  # type: ignore
        return qa_main
    except Exception:
        pass

    raise ImportError(
        "Could not locate a `main()` in Model.py. Checked sibling, AI.Model, direct file paths, and AI.__init__."
    )

def main():
    """Main entry point."""
    try:
        setup_python_path()
        qa_main = _import_qa_main()
        return qa_main()  # hand off to Model.main() which parses args itself
    except ImportError as e:
        print("❌ Import Error:")
        print(f"   {e}")
        print("\n🔧 Troubleshooting:")
        print("   1) If Model.py is next to main.py, the name must be 'Model.py'")
        print("   2) If using a package layout, ensure AI/__init__.py exists and AI/Model.py is present")
        print(f"   3) CWD: {os.getcwd()}")
        print(f"   4) Script folder: {Path(__file__).parent}")
        return 1
    except FileNotFoundError as e:
        print("❌ File Not Found:")
        print(f"   {e}")
        print("\n💡 Tips:")
        print("   • Provide your dataset: --data path/to/training_ready.jsonl")
        print("   • Or run --validate to check file paths.")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("💡 Run with --help via Model.py to see available flags.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
