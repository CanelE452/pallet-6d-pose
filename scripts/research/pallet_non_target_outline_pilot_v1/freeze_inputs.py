"""Seal fixed highest-quality masks after an explicit diagnostic RGB review.

No prompt, mask alternative, pixel, or threshold is changed by this step.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def run(supplier_folder,review_path,output):
    started=time.monotonic();folder,review_path,out=map(Path,(supplier_folder,review_path,output))
    review=read(review_path);source=folder/'NON_TARGET_INPUT_MANIFEST.jsonl'
    if sha(source)!=review['supplier_manifest_sha256']:raise ValueError('Supplier changed after visual review')
    rows=[json.loads(l)for l in source.read_text().splitlines()]
    if len(rows)!=6 or [r['image_id']for r in rows]!=[r['id']for r in review['cases']]:raise ValueError('Six review entries required')
    out.mkdir(parents=True,exist_ok=False);result=[]
    for row,decision in zip(rows,review['cases']):
        approved=decision['admissible_for_censoring']
        if type(approved)is not bool:raise ValueError('Explicit admissibility required')
        record=dict(id=row['image_id'],status='ASSISTED_REVIEWED'if approved else row['status'],
            admissible_for_censoring=approved,image_sha256=row['rgb_file_sha256'],image_hw=row['image_hw'],
            mask_path=None,mask_sha256=None,pixel_bytes_sha256=None,source_supplier_mode='ASSISTED',
            reviewer_kind='model',reviewed_as_physical_GT=False,physical_depth_order_asserted=False)
        if approved:
            if sha(row['all_masks_path'])!=row['all_masks_file_sha256']:raise ValueError('Saved alternatives changed')
            with np.load(row['all_masks_path'],allow_pickle=False)as data:
                selected=int(np.argmax(data['quality']));mask=data['masks'][selected].astype(np.uint8)
                if selected!=int(data['selected_index'])or selected!=row['selected_index']:raise ValueError('Maximum-quality first-tie rule changed')
            digest=hashlib.sha256(mask.tobytes()).hexdigest()
            if digest!=row['selected_pixel_sha256']:raise ValueError('Mask pixels changed')
            path=out/(row['image_id'].replace(':','__')+'.npy')
            with path.open('xb')as stream:np.save(stream,mask,allow_pickle=False)
            record.update(mask_path=str(path),mask_sha256=sha(path),pixel_bytes_sha256=digest,selected_index=selected)
        result.append(record)
    path=out/'INPUTS_FROZEN.jsonl'
    with path.open('x')as stream:
        for row in result:stream.write(json.dumps(row,separators=(',',':'))+'\n')
    receipt=dict(schema='FIXED_INPUT_BEFORE_NEW_SELECTION_V1',frozen_epoch=time.time(),
        source_manifest_sha256=sha(source),visual_review_sha256=sha(review_path),input_manifest_sha256=sha(path),
        rows=6,admitted=sum(r['admissible_for_censoring']for r in result),
        new_prompts=0,new_mask_pixels=0,new_NN=0,threshold_or_alternative_changed=False,
        elapsed_seconds=time.monotonic()-started)
    (out/'INPUT_SEAL.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('supplier-folder','review','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();print(run(a.supplier_folder,a.review,a.output))
