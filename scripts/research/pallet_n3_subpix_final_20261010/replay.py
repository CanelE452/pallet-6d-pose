"""Replay the fixed evaluator into a fresh output namespace without replacing evidence."""
import argparse
from pathlib import Path

from . import common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Fresh output directory containing a PASS INPUT_AUDIT.json from preflight.py')
    args = parser.parse_args()
    output = args.output.resolve()
    assert not (output / 'PREDICTIONS.jsonl.gz').exists(), 'Existing evidence must be preserved'
    assert not (output / 'INPUT_AND_METHOD_LOCK.json').exists(), 'Existing execution lock must be preserved'
    assert (output / 'INPUT_AUDIT.json').is_file(), 'Run preflight.py for this namespace first'
    assert common.read(output / 'INPUT_AUDIT.json')['status'] == 'PASS'
    common.DOC = output
    from .evaluate import main as evaluate
    evaluate()


if __name__ == '__main__':
    main()
