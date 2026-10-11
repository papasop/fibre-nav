"""Read-only historical V4 failure check; no GPU or model execution."""
import importlib.util
import json
from pathlib import Path

root = Path(__file__).resolve().parent
path = root.parent / 'gpt2_b5_xor_continuation_a100_v5/verify_archive.py'
spec = importlib.util.spec_from_file_location('b5_xor_archive_verifier', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if __name__ == '__main__':
    print(json.dumps(module.verify(root / 'gpt2_b5_xor_truth_table_a100_results.zip', 'v4'), indent=2))
