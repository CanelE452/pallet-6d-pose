"""Freeze a declared copied-input fixture and execute the public saved-join adapter."""
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
V21='_docs/experiments/pallet_whole_u_closure_20261011_v21'
V23='_docs/experiments/pallet_consensus_cause_audit_20261011_v23'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def save(p,r):
    with p.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')


def git(*args):return subprocess.check_output(['git','-C',str(STAGE),*args],stderr=subprocess.PIPE)


def main():
    fixture=ROOT/'real_replay_fixture';assert not fixture.exists();fixture.mkdir()
    protocol=json.loads((ROOT/'real_loo/PROTOCOL.json').read_text())
    original={Path(k).name:(Path(k),v) for k,v in protocol['inputs'].items()}
    manifest=[]
    g,expected=original['GEOMETRY.jsonl.gz']
    archive=json.loads(git('show',PARENT+':'+V21+'/ARCHIVE_MANIFEST.json'))
    entry=next(v['original'] for v in archive['files'] if v['original']['path']==V21+'/geometry/GEOMETRY.jsonl.gz')
    assert entry['bytes']==expected['bytes'] and entry['sha256']==expected['sha256']
    p=fixture/entry['path'];p.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(g,p);assert sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes']
    manifest.append(dict(path=entry['path'],bytes=entry['bytes'],sha256=entry['sha256'],
        provenance='copied frozen original matching previously published archive manifest; not restored anew'))
    for rel in (V21+'/evaluation/POSTHOC_ROWS.jsonl.gz',V23+'/SELECTED_BOUNDARY_ROWS.jsonl'):
        b=git('show',PARENT+':'+rel);expected=original[Path(rel).name][1]
        assert len(b)==expected['bytes'] and hashlib.sha256(b).hexdigest()==expected['sha256']
        p=fixture/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        manifest.append(dict(path=rel,bytes=len(b),sha256=expected['sha256'],provenance='exact parent Git blob'))
    shutil.copytree(ROOT/'real_loo',fixture/DOC/'real_loo')
    for name in ('audit_saved_real_loo.py','replay_public_real.py'):
        p=fixture/CODE/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)
        manifest.append(dict(path=CODE+'/'+name,bytes=p.stat().st_size,sha256=sha(p),provenance='exact code bytes; Git check at publication'))
    save(ROOT/'PUBLIC_REAL_FIXTURE_PROTOCOL.json',dict(fixture='explicit copied byte fixture; not fresh clone',
        files=manifest,parent_commit=PARENT,no_new_archive_restore=True,
        wrapper_code_sha256=sha(ROOT/'replay_public_real.py'),new_PnP=0,new_training=0,new_pose_accuracy=0))
    subprocess.run(['/home/minjae/anaconda3/bin/python3','-I','-S','-B',
        str(fixture/CODE/'replay_public_real.py'),'--checkout',str(fixture),'--output',str(ROOT/'public_real_replay')],check=True)
    check=json.loads((ROOT/'public_real_replay/PUBLIC_REAL_REPLAY_CHECKS.json').read_text());assert check['passed']
    for m in manifest:assert sha(fixture/m['path'])==m['sha256']
    save(ROOT/'PUBLIC_REAL_REPLAY_CHECKS.json',dict(**check,
        fixture='copied frozen geometry plus explicit Git blobs; not a fresh clone',
        geometry_matches_previously_published_archive_manifest=True,
        current_archive_restoration_executed=False))
    print(json.dumps(dict(passed=True,all245_join_replayed=True,all133_byte_exact=True,new_PnP=0)))


if __name__=='__main__':main()
