"""Original revision-metadata entry, with scheduling that keeps finished prefixes."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import continue_author_clone_worker as worker

root = Path(__file__).resolve().parents[1]
output = Path(sys.argv[sys.argv.index('--output') + 1])
amendment = json.loads((output / 'continuation-amendment.json').read_text())
for name, digest in amendment['source_hashes'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
sys.modules['run_author_clone_worker'] = worker
runpy.run_path(str(root / 'scripts/author_clone_worker_entry.py'), run_name='__main__')
