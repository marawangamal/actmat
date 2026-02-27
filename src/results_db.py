"""Append-only results database backed by a JSON lines file."""

import json
import os
from datetime import datetime


def args_to_dict(args):
    """Convert an argparse Namespace to a JSON-serializable dict."""
    return {
        k: v if isinstance(v, (bool, int, float, str, list, type(None))) else str(v)
        for k, v in vars(args).items()
    }


def record_exists(db_path, record):
    """Return True if a record matching all fields of `record` (except timestamp) exists."""
    if not os.path.exists(db_path):
        return False
    with open(db_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            existing = json.loads(line)
            if all(existing.get(k) == v for k, v in record.items() if k != "timestamp"):
                return True
    return False


def append_result(db_path, record):
    """Append one result record to a JSON lines file.

    Each line is a self-contained JSON object.  Load the whole file with:
        import pandas as pd
        df = pd.read_json(db_path, lines=True)
    """
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(), **record}
    with open(db_path, "a") as f:
        f.write(json.dumps(record) + "\n")
