"""Package exact V253 saved Source diagnostic directories and selected FIT pixels."""
from pathlib import Path
import argparse,hashlib,json,os,re
R=Path(__file__).absolute().parent
H=R/'V253_ROOT_SAVED_FAILURE_DIAGNOSIS'
P=R/'V253_PUBLICATION_PACKAGE'
TREE={
    'V253_ROOT_SAVED_FAILURE_DIAGNOSIS':{'saved_failure_diagnosis_v1':{},'source_fit_crop_cards_v1':{'raw96':{},'cards':{}}},
    'V253_DEPLOYMENT_SEMANTICS_A':{},
    'V253_SAVED_FAILURE_DIAGNOSIS_B':{},
    'V253_PREVIOUS_PROGRESS_AND_ALIGNMENT_QA_C':{},
    'V253_DIAGNOSTIC_CODE_QA_C':{},
    'V253_ACTUAL_DIAGNOSIS_QA_C':{'actual_review_v1':{},'pixel_review_v1':{}},
    'V253_PUBLIC_REPLAY_ROOT':{},
}

def safe(p):
    p=Path(p)
    if not p.is_absolute() or '..' in p.parts or any(t in str(p).lower() for t in ('v112','cache','.pyc')):raise ValueError(str(p))
    for a in reversed([p,*p.parents]):
        if a.is_symlink():raise ValueError('Artifact symlink')
    return p
def read(p):return safe(p).read_bytes()
def sha(b):return hashlib.sha256(b).hexdigest()

def verify(pkg):
    pkg=safe(pkg);m=json.loads(read(pkg/'MANIFEST.json'));assert m['completed'] and m['PASS']
    for row in m['files']:
        rel=Path(row['relative_path']);assert not rel.is_absolute() and '..' not in rel.parts
        b=read(pkg/rel);assert sha(b)==row['sha256'] and len(b)==row['size_bytes']
    checked=0
    names=['README_KO.md','ROOT_READOUT_KO.html','artifacts/V253_ROOT_SAVED_FAILURE_DIAGNOSIS/source_fit_crop_cards_v1/PORTABLE_GALLERY_KO.html']
    for name in names:
        p=pkg/name;t=read(p).decode()
        links=re.findall(r'(?:href|src)=[\"\']([^\"\']+)',t) if p.suffix=='.html' else re.findall(r'\]\(([^)]+)\)',t)
        for link in links:
            if link.startswith(('https:','http:','#','mailto:')):continue
            assert safe(p.parent/link.split('#')[0]).is_file(),link;checked+=1
    result=dict(completed=True,PASS=True,leaves=len(m['files']),bytes=sum(r['size_bytes'] for r in m['files']),entrypoint_links_checked=checked,original_absolute_fallback=False)
    print(json.dumps(result));return result

def build():
    assert json.loads(read(R/'V253_ACTUAL_DIAGNOSIS_QA_C/REPORT_REVIEW.json'))['PASS']
    assert not safe(P).exists();P.mkdir();rows=[];seen=set()
    def take(src,rel):
        assert rel not in seen;seen.add(rel);b=read(src);p=safe(P/rel);p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(b)
        rows.append(dict(relative_path=rel,sha256=sha(b),size_bytes=len(b),original_path=str(src)))
    def direct(d,prefix,children):
        for name in sorted(os.listdir(safe(d))):
            src=safe(d/name)
            if src.is_dir():
                assert name in children,name;direct(src,prefix+'/'+name,children[name])
            elif src.is_file():
                assert src.suffix in ('.py','.md','.json','.jsonl','.csv','.npz','.npy','.png','.html'),name
                take(src,prefix+'/'+name)
            else:raise ValueError('Nonregular artifact')
    for name,ch in TREE.items():direct(R/name,'artifacts/'+name,ch)
    for name in ('NATIVE_ADDRESS_FAILURE_AND_SUPPORT.png','TARGET_SELECTED_COSINE_ALL_VIEWS.png','TARGET_MARGIN_ALL_VIEWS.png','ROOT_READOUT_KO.html'):take(H/name,name)
    take(H/'PUBLIC_README_KO.md','README_KO.md')
    for name in ('build_v253_readout.py','package_v253_results.py','stage_v253_main.py'):take(R/name,'tools/'+name)
    manifest=dict(schema='V253_SAVED_FAILURE_SOURCE_FIT_CROP_AUDIT_PUBLICATION_V1',completed=True,PASS=True,files=rows,phase='ACTUAL_SAVED_SCORE_DIAGNOSIS_AND_FIXED_SOURCE_FIT_CROP_AUDIT',new_training=0,new_model_forward_or_CP=0,new_reference_bank_fit=0,new_threshold_or_view_choice=False,selected_FIT_RGB_rows=66,numeric_DEV_pixel_rows=0,full_Source_RGB_containers_opaque_hash_includes_DEV_bytes=True,source_teacher_derived_labels_and_conditions_read=True,original_raw_GT_or_RGB_frame_archive_read=0,current_or_held_payload_or_their_historical_aggregate_read=0,PnP=0,model_adopted=False,goal_status='active_unsolved',absolute_paths='Inert provenance; portable replay binds exact six public repository-relative Source score/reference inputs. Original full RGB/teacher generation is not replayed; selected FIT pixels are included.')
    with safe(P/'MANIFEST.json').open('x') as f:json.dump(manifest,f,ensure_ascii=False,indent=2);f.write('\n')
    verify(P)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--build',action='store_true');g.add_argument('--verify',type=Path);a=p.parse_args()
    if a.build:build()
    else:verify(a.verify)
