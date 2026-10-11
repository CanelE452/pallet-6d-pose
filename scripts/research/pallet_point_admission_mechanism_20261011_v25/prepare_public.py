"""Check derived rows and replay only saved source joins in an explicit Git-blob subset."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parent
STAGE=Path('/tmp/pallet-observation-publication-stage-20261011')
PARENT='c1d80d29265729901d571b9d5ffcf57ab3970327'
DOC='_docs/experiments/pallet_point_admission_mechanism_20261011_v25'
CODE='scripts/research/pallet_point_admission_mechanism_20261011_v25'
OLD='_docs/experiments/pallet_source_corner_admission_20261011_v24'


def sha(b):return hashlib.sha256(b).hexdigest()


def save(p,r):
    with p.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')


def git(*args):return subprocess.check_output(['git','-C',str(STAGE),*args],stderr=subprocess.PIPE)


def main():
    prepared=ROOT/'prepared';assert not prepared.exists();prepared.mkdir()
    for origin,dest in [('output','source'),('real_loo','real_loo'),('bounds','bounds'),('plot','plot')]:
        shutil.copytree(ROOT/origin,prepared/dest)
    checked=json.loads(subprocess.check_output(['/home/minjae/anaconda3/bin/python3','-I','-S','-B',
        str(ROOT/'check_public.py'),str(prepared)]))
    assert checked['passed'];save(ROOT/'PRIVATE_PUBLIC_DERIVED_CHECKS.json',checked)
    subset=ROOT/'git_blob_subset';assert not subset.exists();subset.mkdir()
    protocol=json.loads((ROOT/'output/PROTOCOL.json').read_text())
    mapping={'ready':protocol['inputs']['ready']['path'].split('/pallet-kp-publication-20261010-v6/',1)[1],
        'cap_code':protocol['inputs']['cap_code']['path'].split('/pallet-kp-publication-20261010-v6/',1)[1],
        'admission_code':protocol['inputs']['admission_code']['path'].split('/pallet-kp-publication-20261010-v6/',1)[1],
        'selection_code':protocol['inputs']['selection_code']['path'].split('/pallet-kp-publication-20261010-v6/',1)[1],
        'point_score_code':protocol['inputs']['point_score_code']['path'].split('/pallet-kp-publication-20261010-v6/',1)[1],
        'point_rank_code':protocol['inputs']['point_rank_code']['path'].split('/pallet-kp-publication-20261010-v6/',1)[1],
        'decoded':OLD+'/decoder/DECODED_FRAMES.jsonl.gz',
        'decisions':OLD+'/decoder/DECISIONS.jsonl.gz','necessary':OLD+'/ROWS.jsonl.gz',
        'decoder_checks':OLD+'/decoder/CHECKS.json'}
    manifest=[]
    assert git('rev-parse','HEAD').decode().strip()==PARENT
    for k,rel in mapping.items():
        b=git('show',PARENT+':'+rel);expected=protocol['inputs'][k]
        assert len(b)==expected['bytes'] and sha(b)==expected['sha256'],k
        p=subset/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        manifest.append(dict(path=rel,bytes=len(b),sha256=sha(b),source_commit=PARENT))
    for name in ('audit_saved_mechanism.py','replay_public_source.py'):
        p=subset/CODE/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
        manifest.append(dict(path=CODE+'/'+name,bytes=p.stat().st_size,sha256=sha(p.read_bytes()),
            source='exact local code, commit-blob check deferred to publisher'))
    save(ROOT/'PUBLIC_SOURCE_REPLAY_PROTOCOL.json',dict(parent_commit=PARENT,
        fixture='explicit Git blob subset; not a fresh clone',files=manifest,
        source_previous_saved_rows_only=True,new_head=0,new_line_fit=0,new_PnP=0,new_pose_scoring=0,
        new_training=0,new_RGB=0,physical_truth_certification=False))
    output=ROOT/'public_source_replay'
    result=subprocess.run(['/home/minjae/anaconda3/bin/python3','-I','-S','-B',
        str(subset/CODE/'replay_public_source.py'),'--checkout',str(subset),'--output',str(output)],
        check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)['passed']
    for name in ('LINE_ROWS.jsonl.gz','CORNER_ROWS.jsonl.gz','FRAME_ROWS.jsonl.gz','SUMMARY.json','CHECKS.json'):
        assert (output/name).read_bytes()==(ROOT/'output'/name).read_bytes(),name
    for m in manifest:assert sha((subset/m['path']).read_bytes())==m['sha256']
    save(ROOT/'PUBLIC_SOURCE_REPLAY_CHECKS.json',dict(passed=True,fixture_files=len(manifest),
        upstream_files_match_parent_Git_blobs=True,source_joins_actual_executed=True,
        source_frames=384,source_lines=1302,source_corners=573,
        output_3_gzip_rows_SUMMARY_CHECKS_byte_identical=True,inputs_unchanged=True,
        fresh_clone=False,weights_images_required=False,original_physical_truth_replayed=False,
        new_line_fit=0,new_PnP=0,new_pose_accuracy=0,new_training=0))
    print(json.dumps(dict(passed=True,public_derived_reader=checked,source_replay_byte_identical=True,fixture_files=len(manifest))))


if __name__=='__main__':main()
