"""Test runner that executes all test functions across test files without requiring pytest."""

import inspect
from pathlib import Path
import sys
import traceback

# Ensure src is on python path
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tests.test_preprocessing as t_prep
import tests.test_model as t_model
import tests.test_dataset_and_split as t_data
import tests.test_retrieval as t_ret


def run_tests():
    modules = [t_prep, t_model, t_data, t_ret]
    passed = 0
    failed = 0
    total = 0

    print("=" * 60)
    print("RUNNING UNIT TEST SUITE")
    print("=" * 60)

    for mod in modules:
        mod_name = mod.__name__
        print(f"\nTesting module: {mod_name}")
        for attr_name in dir(mod):
            if attr_name.startswith("test_"):
                fn = getattr(mod, attr_name)
                if callable(fn):
                    total += 1
                    try:
                        # Inspect parameters
                        sig = inspect.signature(fn)
                        if "tmp_path" in sig.parameters:
                            import tempfile
                            with tempfile.TemporaryDirectory() as tmp_dir:
                                fn(tmp_path=Path(tmp_dir))
                        else:
                            fn()
                        print(f"  [PASS] {attr_name}")
                        passed += 1
                    except Exception as e:
                        print(f"  [FAIL] {attr_name}")
                        traceback.print_exc()
                        failed += 1

    print("\n" + "=" * 60)
    print(f"TEST SUMMARY: {passed}/{total} Passed, {failed} Failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
