"""Prepare and open the isolated30-frame annotation editor; never train."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/annotate'))
from annotate_review import load_review_contexts, OUTPUT_REL
from object_geometry_registry import load_object_geometry_registry, DEFAULT_REGISTRY_PATH

MANIFEST=ROOT/'outputs/pallet_type_selftrain_v1/large_corner_recovery_v1/real_support_review_v2/CANDIDATES.json'
OUT=ROOT/OUTPUT_REL


def prepare():
    sessions,contexts=load_review_contexts(MANIFEST,argparse.Namespace(),
        load_object_geometry_registry(DEFAULT_REGISTRY_PATH),ROOT)
    rows=json.loads(MANIFEST.read_text());mapping=[]
    for row in rows:
        source=str((ROOT/row['image']['path']).resolve())
        ctx=next(c for c in contexts.values() if source in c['frame_paths'])
        mapping.append(dict(id=row['id'],image=row['image'],recording_id=row['recording_id'],
            role=row['proposed_role'],split=ctx['args'].default_split,
            annotation=str((Path(ctx['out_dir'])/(Path(source).stem+'.json')).relative_to(ROOT))))
    OUT.mkdir(parents=True,exist_ok=True)
    dest=OUT/'WORKSPACE.json'
    payload=dict(source_manifest=str(MANIFEST.relative_to(ROOT)),rows=mapping,
        warning='Review workspace, not a promoted training/evaluation dataset. Only manually confirmed visible corners may be supervision.')
    if dest.exists():assert json.loads(dest.read_text())==payload
    else:
        with dest.open('x') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
    return mapping


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--status',action='store_true');args=p.parse_args()
    rows=prepare()
    if args.status:
        counts={role:dict(candidates=0,saved=0,manual_visible_corners=0) for role in ['proposed_support','proposed_validation']}
        for row in rows:
            c=counts[row['role']];c['candidates']+=1;path=ROOT/row['annotation']
            if path.exists():
                doc=json.loads(path.read_text());obj=doc['objects'][0];c['saved']+=1
                c['manual_visible_corners']+=sum(x.get('source')=='manual_click' and x.get('visibility')==2
                    for x in obj.get('keypoint_annotations',[])[:8])
        print(json.dumps(counts,ensure_ascii=False,indent=2));return
    if args.prepare_only:print('Prepared',len(rows),'frames:',OUT);return
    pidfile=OUT/'EDITOR_PID.json'
    if pidfile.exists():
        pid=json.loads(pidfile.read_text())['pid'];cmd=Path(f'/proc/{pid}/cmdline')
        if cmd.exists() and str(MANIFEST).encode() in cmd.read_bytes():
            print('Already running:',pid);return
    argv=[sys.executable,'-u',str(ROOT/'scripts/annotate/annotate.py'),
        '--review-manifest',str(MANIFEST),'--stride','1','--population-role','DEV']
    env=os.environ.copy();env.setdefault('DISPLAY',':0');env.setdefault('MPLCONFIGDIR','/tmp/pallet-annotation-review-mpl')
    with (OUT/'editor.log').open('a') as log:
        child=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(2)
    if child.poll() is not None:raise RuntimeError(f'Editor exited {child.returncode}; see {OUT/"editor.log"}')
    pidfile.write_text(json.dumps(dict(pid=child.pid,command=argv),ensure_ascii=False,indent=2))
    print('EDITOR_RUNNING',child.pid,'LOG',OUT/'editor.log',flush=True)


if __name__=='__main__':main()
