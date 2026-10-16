"""Check compact source geometry against exact prior Git-blob input bytes."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
STAGE=Path('/tmp/pallet-observation-publication-stage-20261011')
PARENT='942d162d0657eeb7c18d53946e9b1971220ce490'


def save(p,r):
    with p.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')


def main():
    save(ROOT/'PUBLIC_READER_INITIAL_LAYOUT_FAILURE.json',dict(passed=False,actual_exit_code=1,
        phase='before geometry projection, at upstream input hash access',
        reason='Sparse STAGE did not materialize READY_SOURCE_TARGET_ROWS.jsonl.gz; actual FileNotFoundError.',
        stderr='summarized from actual tool result, not a synthetic traceback',
        original_source_audit_passed=True,source_data_or_numerics_changed=False,scalar_projection_calls=0))
    fixture=ROOT/'git_blob_reader_fixture';assert not fixture.exists();fixture.mkdir()
    protocol=json.loads((ROOT/'output/PROTOCOL.json').read_text());manifest=[]
    rels={'ready':'_docs/experiments/pallet_kp_supervision_repair_20261010_v1/READY_SOURCE_TARGET_ROWS.jsonl.gz',
        'ceiling':'_docs/experiments/pallet_kp_difficulty_20261010_v1/SOURCE_CEILING_ROWS.jsonl.gz',
        'saved_corners':'_docs/experiments/pallet_point_admission_mechanism_20261011_v25/source/CORNER_ROWS.jsonl.gz'}
    for key,rel in rels.items():
        b=subprocess.check_output(['git','-C',str(STAGE),'show',PARENT+':'+rel]);expected=protocol['inputs'][key]
        digest=hashlib.sha256(b).hexdigest();assert digest==expected['sha256'] and len(b)==expected['bytes']
        p=fixture/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        manifest.append(dict(path=rel,bytes=len(b),sha256=digest,source_commit=PARENT))
    save(ROOT/'PUBLIC_FIXTURE_PROTOCOL.json',dict(parent=PARENT,files=manifest,
        fixture='explicit prior Git-blob input subset; not fresh clone',original_labels_copied=False,
        code_sha256=hashlib.sha256((ROOT/'check_public.py').read_bytes()).hexdigest()))
    r=json.loads(subprocess.check_output(['/home/minjae/anaconda3/bin/python3','-I','-S','-B',
        str(ROOT/'check_public.py'),str(ROOT/'output'),'--checkout',str(fixture)]))
    assert r['passed'];save(ROOT/'PRIVATE_PUBLIC_CHECKS.json',r)
    print(json.dumps(r))


if __name__=='__main__':main()
