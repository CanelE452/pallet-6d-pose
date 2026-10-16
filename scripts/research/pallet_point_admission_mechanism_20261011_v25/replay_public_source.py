"""Adapt the published previous audit layout to the original executed reader."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkout',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();checkout=a.checkout.resolve();out=a.output.resolve()
    prior=checkout/'_docs/experiments/pallet_source_corner_admission_20261011_v24'
    assert (prior/'ROWS.jsonl.gz').is_file() and (prior/'decoder/DECODED_FRAMES.jsonl.gz').is_file()
    assert not out.exists(),'Choose a new output directory; preserve previous results.'
    with tempfile.TemporaryDirectory(prefix='pallet-v25-published-reader-') as temp:
        alias=Path(temp);(alias/'output').symlink_to(prior,target_is_directory=True)
        (alias/'decoder').symlink_to(prior/'decoder',target_is_directory=True)
        subprocess.run([sys.executable,'-I','-S','-B',str(Path(__file__).with_name('audit_saved_mechanism.py')),
            '--source',str(checkout),'--prior',str(alias),'--output',str(out)],check=True)


if __name__=='__main__':
    main()
