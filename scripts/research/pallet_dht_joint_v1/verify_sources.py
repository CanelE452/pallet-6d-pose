"""Revalidate every actual synthetic image/label against the frozen manifest."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from scripts.research.pallet_dht_joint_v1.train import SOURCE_MANIFEST, SOURCE_SHA256, R0_PATH, R0_SHA256, sha


def main():
    out=Path('data/pallet/results/pallet_dht_joint_v1/SOURCE_REVALIDATION.json')
    assert sha(SOURCE_MANIFEST)==SOURCE_SHA256
    assert sha(R0_PATH)==R0_SHA256
    data=json.loads(SOURCE_MANIFEST.read_text())
    started=time.time()
    def check(row):
        for field in ['image','label']:
            assert sha(row[field])==row[field+'_sha256'], row[field]
        return row['index']
    count=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(check,data['records']):
            count+=1
            if count%10000==0: print(f'verified {count}/60000 synthetic image+label pairs',flush=True)
    assert count==60000
    result=dict(complete=True,PASS=True,source_manifest_sha256=SOURCE_SHA256,
                baseline_sha256=R0_SHA256,checked_images=count,checked_labels=count,
                train=55980,validation=4020,filter_applied=False,
                elapsed_seconds=time.time()-started,
                source_code_sha256=sha(Path(__file__)))
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__': main()
