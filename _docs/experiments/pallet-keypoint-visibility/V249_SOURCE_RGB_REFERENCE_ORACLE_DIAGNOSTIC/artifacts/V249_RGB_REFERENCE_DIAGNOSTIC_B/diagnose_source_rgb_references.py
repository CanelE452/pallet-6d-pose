"""Fixed SOURCE teacher-identity-oracle diagnostic; no RGB/model/fit reader.

run_diagnostic(rows, features, samples) is independent of local paths and I/O.
The entry CLI admits exactly the three frozen SOURCE files after Root approval.
"""
import argparse, csv, hashlib, io, json, math, os, re, traceback
from pathlib import Path
from collections import Counter
import numpy as np

R = Path(__file__).absolute().parent.parent
H = R/'V249_RGB_REFERENCE_DIAGNOSTIC_B'
ROOT = R/'V249_ROOT_TARGET_RGB_DIAGNOSTIC'
PROTOCOL = {'path': str(ROOT/'PROTOCOL_BEFORE_SOURCE_REFERENCE_PAYLOADS.json'), 'sha256': 'c00bba0de52c0213ec07b6470bc968f8e900b4250311c015a584c9a25de1b5f9', 'size_bytes': 5318}
COLS = ('row_id','sample_index','role','scene','frame_id','object_id','aux_slot','associated_r_state','q_x','q_y','r_x_GT_diagnostic_only','r_y_GT_diagnostic_only','q_reference_distance_px','L')
VIEWS = {'full_CLS': (0,384), 'fixed_centre_Q': (384,768), 'centre_CLS': (768,1152)}
STRATA = ('all','<=1px','>1_to_3px','<=3px','>3px','unknown')
SAMPLE_KEYS = ('target','family','fit','available','row_id')


def guard(value):
    text = str(value)
    if any(t in text.casefold() for t in ('v112','cache','.pyc','current','held','private','holdout','oldval','v177','v178','v182')):
        raise ValueError('Forbidden lexical path before FS')
    p = Path(text)
    if not p.is_absolute() or '..' in p.parts: raise ValueError('Absolute traversal-free path required')
    for a in reversed((p,*p.parents)):
        if a.is_symlink(): raise ValueError('Symlink artifact forbidden')
    return p


def identity(b):
    if not {'path','sha256'} <= set(b) <= {'path','sha256','size_bytes'} or re.fullmatch('[0-9a-f]{64}',str(b['sha256'])) is None:
        raise ValueError('Exact binding required')
    if 'size_bytes' in b and (type(b['size_bytes']) is not int or b['size_bytes'] < 0): raise ValueError('Integer size required')
    return str(guard_lexical(b['path'])),b['sha256']


def guard_lexical(value):
    text = str(value)
    if any(t in text.casefold() for t in ('v112','cache','.pyc','current','held','private','holdout','oldval','v177','v178','v182')):
        raise ValueError('Forbidden lexical path before FS')
    p=Path(text)
    if not p.is_absolute() or '..' in p.parts: raise ValueError('Absolute traversal-free path required')
    return p


def binding(path):
    p=guard(path);h=hashlib.sha256();n=0
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk);n+=len(chunk)
    return {'path':str(p),'sha256':h.hexdigest(),'size_bytes':n}


def checked(b):
    identity(b)
    actual=binding(b['path'])
    if actual['sha256']!=b['sha256'] or ('size_bytes' in b and actual['size_bytes']!=b['size_bytes']): raise ValueError('SHA/size changed')
    return guard(b['path'])


def jread(b):
    return json.loads(checked(b).read_bytes())


def jwrite(path,obj):
    with guard(path).open('x') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return binding(path)


def finite_float(value):
    if value is None or str(value).strip() in ('','UNKNOWN','null','None','NaN','nan'): return None
    result=float(value)
    return result if math.isfinite(result) else None


def summarize(values):
    a=np.asarray([v for v in values if v is not None],np.float64)
    if not len(a): return {'n':0,'mean':None,'quantiles':None}
    if not np.isfinite(a).all(): raise ValueError('Finite scores only')
    return {'n':len(a),'mean':float(a.mean()),'quantiles':dict(zip(('min','p05','p25','median','p75','p95','max'),map(float,np.quantile(a,[0,.05,.25,.5,.75,.95,1]))))}


def auc_E(labels,scores):
    pairs=[(int(y),float(s)) for y,s in zip(labels,scores) if s is not None]
    pos=np.asarray([s for y,s in pairs if y==1],np.float64);neg=np.asarray([s for y,s in pairs if y==0],np.float64)
    if not len(pos) or not len(neg): return {'auc':None,'D_supported':len(neg),'E_supported':len(pos),'reason':'BOTH_CLASSES_REQUIRED'}
    value=sum(float((p>neg).sum())+.5*float((p==neg).sum()) for p in pos)/(len(pos)*len(neg))
    return {'auc':value,'D_supported':len(neg),'E_supported':len(pos),'reason':None,'ties':'half'}


def belongs(row,stratum):
    d=row['q_r_distance_px']
    return (stratum=='all' or (stratum=='unknown' and d is None) or
            (d is not None and ((stratum=='<=1px' and d<=1) or (stratum=='>1_to_3px' and 1<d<=3) or
                                (stratum=='<=3px' and d<=3) or (stratum=='>3px' and d>3))))


def run_diagnostic(metadata_rows,features,samples):
    """Return artifacts using only passed frozen SOURCE arrays/selected metadata.

    Reference mean uses raw stored descriptors, ascending sample index, FP64
    addition and final unit-L2. Query unit-L2 is FP64. No r/q correction occurs.
    """
    if set(samples)!=set(SAMPLE_KEYS): raise ValueError('Exactly five nonpixel sample members')
    n=len(samples['target'])
    if features.dtype!=np.float32 or features.shape!=(n,1186): raise ValueError('Frozen FP32[N,1186] features')
    for k in SAMPLE_KEYS:
        if samples[k].shape!=(n,): raise ValueError('Original sample vector shape')
    if samples['target'].dtype!=np.int8 or samples['family'].dtype!=np.int8 or samples['fit'].dtype!=bool or samples['available'].dtype!=bool or samples['row_id'].dtype.kind!='U': raise ValueError('Original sample dtypes')
    if len(set(samples['row_id'].tolist()))!=n: raise ValueError('Unique original sample row IDs')
    rows=[];seen=set();indices=set()
    for raw in metadata_rows:
        if set(raw)!=set(COLS): raise ValueError('Only the fourteen declared metadata fields')
        si=int(raw['sample_index']);rid=raw['row_id'];state=raw['associated_r_state'];role=raw['role']
        if si in indices or rid in seen or not 0<=si<n or samples['family'][si]!=1 or str(samples['row_id'][si])!=rid: raise ValueError('Exact row_id AND sample_index YCB join')
        if state not in ('D','E') or role not in ('FIT','DEV') or int(samples['target'][si])!=int(state=='E') or bool(samples['fit'][si])!=(role=='FIT'): raise ValueError('Original target/role unchanged')
        seen.add(rid);indices.add(si)
        q=[finite_float(raw[k]) for k in ('q_x','q_y')];r=[finite_float(raw[k]) for k in ('r_x_GT_diagnostic_only','r_y_GT_diagnostic_only')]
        distance=None if any(v is None for v in q+r) else math.hypot(q[0]-r[0],q[1]-r[1])
        saved=finite_float(raw['q_reference_distance_px'])
        if distance is not None and (saved is None or not np.isclose(distance,saved,rtol=1e-6,atol=1e-5)): raise ValueError('Stored q-r distance mismatch')
        rows.append({'row_id':rid,'sample_index':si,'role':role,'scene':raw['scene'],'frame_id':raw['frame_id'],
                     'object_id':int(raw['object_id']),'logical_aux_slot':int(raw['aux_slot']),'associated_r_state':state,
                     'target':int(state=='E'),'available':bool(samples['available'][si]),'q_xy':q,'r_xy_GT_diagnostic_only':r,
                     'q_r_distance_px':distance,'L':finite_float(raw['L'])})
    if indices!=set(map(int,np.flatnonzero(samples['family']==1))): raise ValueError('All original known YCB rows retained')
    rows.sort(key=lambda r:r['sample_index'])
    fit=[r for r in rows if r['role']=='FIT'];dev=[r for r in rows if r['role']=='DEV']
    objects=sorted({r['object_id'] for r in fit});wrong={obj:(objects[(i+1)%len(objects)] if len(objects)>1 else None) for i,obj in enumerate(objects)}
    prototypes={};support=[];output=[]
    for view,(lo,hi) in VIEWS.items():
        # Only YCB rows are numerically selected; no SYN image descriptor replay.
        for obj in objects:
            all_D=[r for r in fit if r['object_id']==obj and r['associated_r_state']=='D']
            candidates=[r for r in all_D if r['available']]
            accepted=[];total=np.zeros(384,np.float64)
            for row in candidates:
                v=np.asarray(features[row['sample_index'],lo:hi],np.float64)
                if np.isfinite(v).all() and np.linalg.norm(v)>0:
                    total=total+v;accepted.append(row)
            p=None;reason='NO_FINITE_NONZERO_FIT_D_REFERENCE'
            if accepted:
                mean=total/len(accepted);norm=float(np.linalg.norm(mean))
                if math.isfinite(norm) and norm>0: p=mean/norm;reason=None
                else: reason='ZERO_OR_NONFINITE_PROTOTYPE_MEAN'
            prototypes[(view,obj)]=p
            support.append({'view':view,'object_id':obj,'fixed_other_object_id':wrong[obj],
                            'FIT_D_total_rows':len(all_D),'FIT_D_original_unavailable':len(all_D)-len(candidates),'FIT_D_candidates_available':len(candidates),
                            'FIT_D_descriptor_nonfinite_or_zero':len(candidates)-len(accepted),
                            'FIT_D_rows_used':len(accepted),'distinct_scenes':len({r['scene'] for r in accepted}),
                            'distinct_scene_frames':len({(r['scene'],r['frame_id']) for r in accepted}),
                            'logical_aux_slot_counts':dict(Counter(str(r['logical_aux_slot']) for r in accepted)),
                            'prototype_supported':p is not None,'unsupported_reason':reason,
                            'prototype_FP64_sha256':None if p is None else hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()})
        for row in dev:
            obj=row['object_id'];other=wrong.get(obj);v=np.asarray(features[row['sample_index'],lo:hi],np.float64)
            norm=float(np.linalg.norm(v)) if np.isfinite(v).all() else float('nan')
            supported=row['available'] and math.isfinite(norm) and norm>0
            p=prototypes.get((view,obj));op=prototypes.get((view,other));true=othercos=margin=None;reasons=[]
            if not row['available']: reasons.append('ORIGINAL_INPUT_UNAVAILABLE')
            elif not supported: reasons.append('QUERY_DESCRIPTOR_NONFINITE_OR_ZERO')
            if p is None: reasons.append('NOMINAL_OBJECT_PROTOTYPE_UNSUPPORTED')
            if other is None: reasons.append('NO_FIXED_DIFFERENT_FIT_OBJECT')
            elif op is None: reasons.append('FIXED_OTHER_PROTOTYPE_UNSUPPORTED')
            if supported:
                query=v/norm
                if p is not None: true=float(np.dot(query,p))
                if op is not None: othercos=float(np.dot(query,op))
                if true is not None and othercos is not None: margin=true-othercos
            output.append(dict(row,view=view,fixed_other_object_id=other,query_descriptor_supported=supported,
                               nominal_prototype_supported=p is not None,other_prototype_supported=op is not None,
                               cosine_nominal_target=true,cosine_fixed_other_target=othercos,nominal_minus_other_margin=margin,
                               support_reasons=reasons,reference_selection_is_teacher_oracle=True,native_mapping='UNKNOWN'))
    tables=[]
    for view in VIEWS:
        selected=[r for r in output if r['view']==view]
        for stratum in STRATA:
            for obj in [None]+sorted({r['object_id'] for r in dev}):
                group=[r for r in selected if belongs(r,stratum) and (obj is None or r['object_id']==obj)]
                d=[r for r in group if r['target']==0];e=[r for r in group if r['target']==1]
                distributions={score:{'D':summarize(r[score] for r in d),'E':summarize(r[score] for r in e)} for score in ('cosine_nominal_target','cosine_fixed_other_target','nominal_minus_other_margin')}
                margins=[r['nominal_minus_other_margin'] for r in d if r['nominal_minus_other_margin'] is not None]
                tables.append({'view':view,'q_r_stratum':stratum,'object_id':obj,'rows':len(group),'D':len(d),'E':len(e),
                               'query_supported':sum(r['query_descriptor_supported'] for r in group),
                               'distributions':distributions,
                               'E_AUROC_negative_nominal_cosine':auc_E([r['target'] for r in group],[None if r['cosine_nominal_target'] is None else -r['cosine_nominal_target'] for r in group]),
                               'E_AUROC_negative_margin':auc_E([r['target'] for r in group],[None if r['nominal_minus_other_margin'] is None else -r['nominal_minus_other_margin'] for r in group]),
                               'D_nominal_gt_other':{'supported':len(margins),'count':sum(v>0 for v in margins),'ties':sum(v==0 for v in margins),'fraction':None if not margins else sum(v>0 for v in margins)/len(margins)}})
    counts={'known_YCB':len(rows),'FIT':len(fit),'DEV':len(dev),'FIT_D':sum(r['target']==0 for r in fit),'FIT_E':sum(r['target']==1 for r in fit),
            'DEV_D':sum(r['target']==0 for r in dev),'DEV_E':sum(r['target']==1 for r in dev),'descriptor_views':3,'DEV_view_rows':len(output),
            'unknown_q_r_rows_DEV':sum(r['q_r_distance_px'] is None for r in dev)}
    summary={'completed':True,'PASS':True,'counts':counts,'nominal_FIT_objects':objects,'fixed_wrong_object_cyclic_next':wrong,
             'pooled_all':{t['view']:t for t in tables if t['q_r_stratum']=='all' and t['object_id'] is None},
             'no_best_view_or_threshold_selected':True,'reference_identity_is_oracle':True,'current_admission':False,
             'limits':['All source roles/labels/q/L and frozen features unchanged; no q/r correction or owner filter.',
                       'Associated-r D reference is an equally weighted q crop feature, not a target-only clean visible cutout.',
                       'Teacher object ID chooses an oracle bank, not a deployable input or reference retriever.',
                       'SOURCE DEV is exposed/reused; no native physical gold, causal mechanism or independent generalization certification.',
                       'AUC direction fixed to E=-nominalcos and E=-margin. Wrong cosine is reported as a distribution only.',
                       'No optimizer, encoder/head forward, RGB/weights/current/held input, PnP or label/role modification.']}
    return {'DEV_scores':output,'prototype_support':support,'prototypes':prototypes,'objects':objects,'tables':tables,'summary':summary}


def write_results(output_directory,result):
    """Write pure result artifacts to a fresh caller-owned directory."""
    out=guard(output_directory)
    if out.exists(): raise ValueError('Fresh immutable output directory required')
    out.mkdir(parents=True);outputs={}
    for name,rows in [('DEV397_DESCRIPTOR_SCORES.csv',result['DEV_scores']),('PROTOTYPE_SUPPORT.csv',result['prototype_support'])]:
        if not rows: raise ValueError('Declared evaluation/ref schema must not disappear')
        with guard(out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
            for row in rows: w.writerow({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
        outputs[name]=binding(out/name)
    outputs['DESCRIPTIVE_TABLES.json']=jwrite(out/'DESCRIPTIVE_TABLES.json',{'completed':True,'PASS':True,'tables':result['tables']})
    outputs['SUMMARY.json']=jwrite(out/'SUMMARY.json',result['summary'])
    objects=result['objects'];p=np.zeros((3,len(objects),384),np.float64);ok=np.zeros((3,len(objects)),bool)
    for vi,view in enumerate(VIEWS):
        for oi,obj in enumerate(objects):
            value=result['prototypes'][(view,obj)]
            if value is not None: p[vi,oi]=value;ok[vi,oi]=True
    np.savez(guard(out/'PROTOTYPES.npz'),prototypes=p,supported=ok,object_id=np.asarray(objects,np.int64),view=np.asarray(list(VIEWS)))
    outputs['PROTOTYPES.npz']=binding(out/'PROTOTYPES.npz')
    text=['# V249 SOURCE oracle RGB-reference diagnostic','','Teacher-selected true-object bank is an oracle diagnostic, not deployable input. No training, threshold/best-view selection or original-query correction.','', '|View|D / E|AUC E=-targetcos|AUC E=-margin|D target>other|','|---|---:|---:|---:|---:|']
    for view,t in result['summary']['pooled_all'].items():
        text.append(f"|{view}|{t['D']} / {t['E']}|{t['E_AUROC_negative_nominal_cosine']['auc']}|{t['E_AUROC_negative_margin']['auc']}|{t['D_nominal_gt_other']['fraction']}|")
    text+=['','All3 views, every predeclared q-r stratum, per-object groups, unsupported/empty strata and full distributions are in [DESCRIPTIVE_TABLES.json](DESCRIPTIVE_TABLES.json). No view is selected.','']+result['summary']['limits']
    with guard(out/'PORTABLE_REPORT_KO.md').open('x') as f:f.write('\n'.join(text)+'\n')
    outputs['PORTABLE_REPORT_KO.md']=binding(out/'PORTABLE_REPORT_KO.md')
    return outputs


def run(entry_path):
    e=json.loads(guard(entry_path).read_bytes())
    if e.get('approved_SOURCE_oracle_reference_diagnostic') is not True or e.get('automatic_retry') is not False:
        raise ValueError('Root exact true entry required before any SOURCE payload')
    if identity(e['code'])!=identity(binding(__file__)): raise ValueError('Exact code pin')
    if identity(e['root_protocol'])!=identity(PROTOCOL): raise ValueError('Fixed prospective protocol')
    protocol=jread(e['root_protocol']);scope=jread(e['root_scope'])
    contract=dict(e);contract.pop('root_scope')
    if scope.get('approved') is not True or scope.get('entry_contract')!=contract: raise ValueError('Exact Root scope contract required')
    expected={'original_features':protocol['features'],'samples':protocol['sample_metadata'],'YCB_metadata':protocol['SOURCE_metadata']}
    if set(e['inputs'])!=set(expected) or any(identity(e['inputs'][k])!=identity(v) for k,v in expected.items()): raise ValueError('Only exact original SOURCE3inputs')
    for b in e['inputs'].values(): identity(b)
    if list(protocol['metadata_columns_used_only'])!=list(COLS) or {k:tuple(v) for k,v in protocol['descriptor_views_fixed'].items()}!=VIEWS: raise ValueError('Fixed columns/views changed')
    paths={k:checked(b) for k,b in e['inputs'].items()}
    with paths['YCB_metadata'].open(newline='') as f:
        reader=csv.DictReader(f)
        if not set(COLS)<=set(reader.fieldnames): raise ValueError('Writer metadata schema mismatch')
        rows=[{k:raw[k] for k in COLS} for raw in reader]
    with np.load(paths['samples'],allow_pickle=False) as z: samples={k:z[k] for k in SAMPLE_KEYS}
    features=np.load(paths['original_features'],mmap_mode='r',allow_pickle=False)
    result=run_diagnostic(rows,features,samples)
    counts=result['summary']['counts'];expected_counts={'known_YCB':1521,'FIT':1124,'DEV':397,'FIT_D':841,'FIT_E':283,'DEV_D':340,'DEV_E':57,'descriptor_views':3,'DEV_view_rows':1191}
    if len(samples['target'])!=7617 or any(counts[k]!=v for k,v in expected_counts.items()): raise ValueError('Original SOURCE census unchanged')
    if e['output_directory']!=str(ROOT/'oracle_rgb_reference_diagnostic_v1'): raise ValueError('Fixed fresh Root output')
    outputs=write_results(e['output_directory'],result)
    complete={'completed':True,'PASS':True,'code':e['code'],'root_protocol':e['root_protocol'],'root_scope':e['root_scope'],'entry':binding(entry_path),
              'inputs':e['inputs'],'outputs':outputs,'counts':counts,'five_nonpixel_sample_members_only':list(SAMPLE_KEYS),
              'old_SOURCE_model_score_columns_numerically_interpreted_or_used':False,'CSV_transport':'Whole SOURCE CSV tokenization, retain only fourteen declared columns; old score strings not numerically interpreted or output.',
              'original_features_numerically_selected':'YCB known FIT-D references/DEV only; opaque whole-file SHA includes original SOURCE packet, no raw pixels.',
              'new_training_or_model_forward':0,'RGB_weights_GT_current_or_held_payloads':0,'current_admission':False,'automatic_retry':False}
    jwrite(Path(e['output_directory'])/'COMPLETE.json',complete)
    print(json.dumps({'completed':True,'PASS':True,'complete':binding(Path(e['output_directory'])/'COMPLETE.json')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
