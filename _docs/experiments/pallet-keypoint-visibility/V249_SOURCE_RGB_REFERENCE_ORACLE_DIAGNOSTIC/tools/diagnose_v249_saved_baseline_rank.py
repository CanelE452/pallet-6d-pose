"""Posthoc SOURCE-only baseline rank appendix. No threshold search or model I/O."""
from pathlib import Path
import csv, io, json, hashlib
R=Path(__file__).absolute().parent
H=R/'V249_ROOT_TARGET_RGB_DIAGNOSTIC'

def read(p):
    p=Path(p)
    if any(x in str(p).lower() for x in ('v112','__pycache__','.pyc')) or any(a.is_symlink() for a in [p,*p.parents]):raise ValueError(str(p))
    return p.read_bytes()

def belongs(d,s):
    return s=='all' or (d is None and s=='unknown') or (d is not None and ((s=='<=1px' and d<=1) or (s=='>1_to_3px' and 1<d<=3) or (s=='<=3px' and d<=3) or (s=='>3px' and d>3)))

def run():
    scope=json.loads(read(H/'POSTHOC_BASELINE_RANK_APPENDIX_SCOPE_BEFORE_SCORE_READ.json'))
    assert scope['posthoc'] and scope['source_score_read_authorized']
    meta={r['row_id']:r for r in csv.DictReader(io.StringIO(read(scope['query_error_input']['path']).decode())) if r['view']=='full_CLS'}
    assert len(meta)==397
    output=[]
    for arm,b in scope['SOURCE_baseline_inputs'].items():
        raw=read(b['path']);assert hashlib.sha256(raw).hexdigest()==b['sha256'] and len(raw)==b['size_bytes']
        rows=[r for r in csv.DictReader(io.StringIO(raw.decode())) if r['family']=='1' and r['role']=='DEV']
        assert len(rows)==397 and {r['row_id'] for r in rows}==set(meta)
        for r in rows:
            m=meta[r['row_id']];assert r['target']==m['target'] and r['sample_index']==m['sample_index'] and r['available']=='True'
        for s in ('all','<=1px','>1_to_3px','<=3px','>3px','unknown'):
            for obj in [None]+sorted({int(m['object_id']) for m in meta.values()}):
                group=[r for r in rows if belongs(float(meta[r['row_id']]['q_r_distance_px']) if meta[r['row_id']]['q_r_distance_px'] else None,s) and (obj is None or int(meta[r['row_id']]['object_id'])==obj)]
                d=[float(r['selected_family_logit']) for r in group if r['target']=='0'];e=[float(r['selected_family_logit']) for r in group if r['target']=='1']
                auc=None if not d or not e else sum((x>y)+.5*(x==y) for x in e for y in d)/(len(d)*len(e))
                cm=[[sum(int(r['target'])==y and int(float(r['selected_family_logit'])>=0)==p for r in group) for p in (0,1)] for y in (0,1)]
                output.append({'arm':arm,'q_r_stratum':s,'object_id':obj,'D':len(d),'E':len(e),'raw_E_logit_AUROC':auc,'CM_at_original_zero_logit':cm})
    result={'completed':True,'PASS':True,'posthoc_SOURCE_only':True,'score_direction_fixed':'E positive raw selected_family_logit','threshold_changed':False,'model_choice':False,'current_admission':False,'tables':output}
    p=H/'POSTHOC_BASELINE_RANK_RESULTS.json'
    with p.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps([t for t in output if t['object_id'] is None and t['q_r_stratum'] in ('all','<=3px')]))

if __name__=='__main__':run()
