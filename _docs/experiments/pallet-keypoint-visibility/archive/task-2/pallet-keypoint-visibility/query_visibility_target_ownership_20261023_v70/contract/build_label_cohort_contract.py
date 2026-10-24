"""V70 identity/GT-only audit; no RGB, features, weights or model imports."""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT.parent
V66 = BASE / 'query_visibility_controlled_real_20261022_v66'
SOURCES = ['SYNTH', 'ORIGINAL_YCB', 'CONTROLLED_YCB', 'PALLOX']
PAIRS = np.array([[0,0],[1,0],[1,1],[2,0],[2,1],[2,2]])
GT_KEYS = ['sample_id','source','group_id','image_sha256','global_id',
           'global_image_sha256','global_box','target_index',
           'dense_allowed','dense_valid','dense_exact_state']
LABEL_KEYS = ['sample_id','global_id','target_index','native_corner','query',
              'side','global_box','image_sha256','dense_allowed','dense_valid',
              'dense_exact_state']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def binding(path):
    p=Path(path)
    return dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size)

def save(path, obj):
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def writecsv(path, rows):
    path=Path(path)
    if path.exists(): raise FileExistsError(path)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def exact_state(allowed, valid):
    a=np.asarray(allowed,dtype=np.int64)
    v=np.asarray(valid,bool)
    if a.shape!=v.shape or np.any((a<1)|(a>63)): raise ValueError('invalid allowed/valid')
    if np.any((~v)&(a!=63)): raise ValueError('invalid native cells must remain all63')
    out=np.full(a.shape,-1,np.int8)
    for state in range(3):
        bits=sum(1<<i for i in range(6) if PAIRS[i,0]==state)
        out[v&((a&~bits)==0)]=state
    return out

def eligible(allowed, valid):
    st=exact_state(allowed,valid)
    return np.isin(st,[0,1])

def lookup(ids):
    vals=[str(x) for x in ids]
    if len(vals)!=len(set(vals)):raise ValueError('duplicate sample ID')
    return {x:i for i,x in enumerate(vals)}

def check_pair_identity(orig, ctrl, oq, cq, ob, cb, oside, cside, split, tsplit, dsplit):
    if split!=tsplit or split!=dsplit:raise ValueError('target/donor split crossing')
    if not np.array_equal(oq,cq):raise ValueError('query mismatch')
    if not np.array_equal(ob,cb):raise ValueError('box mismatch')
    if oside!=cside:raise ValueError('side mismatch')
    if orig==0 and ctrl not in (0,1):raise ValueError('original V no longer V/E')
    if orig!=0 and ctrl!=orig:raise ValueError('non-V changed to unsupported new state')

def mock():
    masks=np.arange(1,64,dtype=np.uint8)
    states=exact_state(masks,np.ones(63,bool))
    brute=[]
    for mask in masks:
        opts={int(PAIRS[i,0]) for i in range(6) if int(mask)&(1<<i)}
        brute.append(next(iter(opts)) if len(opts)==1 else -1)
    assert np.array_equal(states,brute)
    assert masks[states==0].tolist()==[1]
    assert masks[states==1].tolist()==[2,4,6]
    assert masks[eligible(masks,np.ones(63,bool))].tolist()==[1,2,4,6]
    assert exact_state(np.array([63]),np.array([False])).tolist()==[-1]
    rejected=0
    for kw in [dict(cq=np.array([2.,3.])),dict(cb=np.array([0.,0.,4.,5.])),
               dict(cside=9.),dict(dsplit='VAL'),dict(tsplit='VAL'),dict(ctrl=2)]:
        args=dict(orig=0,ctrl=1,oq=np.array([1.,3.]),cq=np.array([1.,3.]),
                  ob=np.array([0.,0.,5.,5.]),cb=np.array([0.,0.,5.,5.]),
                  oside=10.,cside=10.,split='TRAIN',tsplit='TRAIN',dsplit='TRAIN')
        args.update(kw)
        try: check_pair_identity(**args)
        except ValueError:rejected+=1
    assert rejected==6
    ids=['b','a','c'];table=['C','B','A'];ti=np.array([1,2,0]);order=[2,0,1]
    assert [table[ti[lookup(ids)[ids[i]]]] for i in order]==['C','B','A']
    # A source with V only demonstrates why VE-only precision is a weaker gate.
    truth=np.array([0,0,2]);pred=np.array([0,0,0])
    assert ((truth==0)&(pred==0)).sum()/(pred==0).sum()==2/3
    return dict(completed=True,status='PASS',all63_mask_projection=True,
                V_allowed=[1],E_allowed=[2,4,6],O_U_excluded=True,
                pair_guard_rejections=rejected,permuted_query_target_table=True,
                VEO_vs_VE_precision_counterexample=True,actual_source_reads=0,
                RGB_feature_weight_current_reads=0,forward=0,fit=0)

def load_npz(path,keys):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k] for k in keys}

def build(lockpath):
    lock=json.loads(Path(lockpath).read_text())
    if lock['builder_sha256']!=sha(__file__):raise ValueError('builder changed after lock')
    if lock['training_authorized'] or lock['current_access_authorized']:
        raise ValueError('contract audit scope cannot authorize fit/current')
    for b in lock['metadata_bindings']:
        if sha(b['path'])!=b['sha256']:raise ValueError('metadata SHA mismatch')
    joined_seal=json.loads((V66/'joined_source_cache/SEAL.json').read_text())
    sealed={Path(x['path']).name:x for x in joined_seal['outputs']}
    targets=[json.loads(x) for x in (V66/'provenance/TARGETS.jsonl').read_text().splitlines()]
    target={x['id']:x for x in targets}
    comps=[json.loads(x) for x in (V66/'controlled/COMPOSITES.jsonl').read_text().splitlines()]
    deny=json.loads((BASE/'query_visibility_public_real_20261022_v63/contract/EVALUATION_SHA256_DENYLIST.json').read_text())
    deny=set(deny['image_sha256'])
    allrows=[];targetrows=[];pairrows=[];counts=[];data={};opened=[]
    for split in ['TRAIN','VAL']:
        path=V66/f'joined_source_cache/CONTROLLED_PLUS_PALLOX_{split}.npz'
        seal=sealed[path.name]
        if path.stat().st_size!=seal['bytes']:raise ValueError('sealed joined cache size differs')
        d=load_npz(path,GT_KEYS);opened.append(dict(path=str(path),arrays=GT_KEYS,
            prior_immutable_SHA=seal['sha256'],sealed_bytes=seal['bytes'],raw_feature_payload_hash_repeated=False))
        data[split]=d
        a=d['dense_allowed'][:,4,4];v=d['dense_valid'][:,4,4]
        state=exact_state(a,v)
        if not np.array_equal(state,d['dense_exact_state'][:,4,4]):raise ValueError('stored exact state differs')
        if not np.array_equal(exact_state(d['dense_allowed'],d['dense_valid']),d['dense_exact_state']):raise ValueError('dense state differs')
        si=lookup(d['sample_id']);d['_lookup']=si;d['_state']=state
        if np.any(d['global_id'][d['target_index']]==''):raise ValueError('missing target')
        if set(map(str,d['image_sha256']))&deny:raise ValueError('evaluation image collision')
        if not np.array_equal(d['global_image_sha256'][d['target_index']],d['image_sha256']):raise ValueError('RGB-target mapping differs')
        for source in SOURCES:
            m=d['source']==source;k=m&np.isin(state,[0,1]);ti=d['target_index']
            counts.append(dict(split=split,source=source,all_queries=int(m.sum()),native_valid=int((m&v).sum()),
                V=int((m&(state==0)).sum()),E=int((m&(state==1)).sum()),O=int((m&(state==2)).sum()),
                UNKNOWN_native_valid=int((m&v&(state<0)).sum()),invalid=int((m&~v).sum()),
                eligible_VE=int(k.sum()),all_targets=len(set(ti[m].tolist())),
                eligible_VE_targets=len(set(ti[k].tolist())),
                conditional_NLL_denominator=int(k.sum())))
        for i,sample in enumerate(d['sample_id']):
            t=int(d['target_index'][i]);allrows.append(dict(split=split,joined_index=i,
                source=str(d['source'][i]),sample_id=str(sample),global_id=str(d['global_id'][t]),
                target_index=t,group_id=str(d['group_id'][i]),image_sha256=str(d['image_sha256'][i]),
                center_allowed=int(a[i]),native_valid=bool(v[i]),exact_state=int(state[i]),
                eligible_VE=bool(state[i] in (0,1)),binary_V_target=int(state[i]==0) if state[i] in (0,1) else -1))
        for t,gid in enumerate(d['global_id']):
            m=d['target_index']==t
            targetrows.append(dict(split=split,global_id=str(gid),target_index=t,
                source=str(d['source'][m][0]),queries=int(m.sum()),V=int((m&(state==0)).sum()),
                E=int((m&(state==1)).sum()),O=int((m&(state==2)).sum()),
                UNKNOWN_or_invalid=int((m&(state<0)).sum()),image_sha256=str(d['global_image_sha256'][t])))
        labels={}
        for source in ['ORIGINAL_YCB','CONTROLLED']:
            lp=V66/f'label_cache_v2/{source}_{split}.npz'
            labels[source]=load_npz(lp,LABEL_KEYS)
            labels[source]['_lookup']=lookup(labels[source]['sample_id'])
            opened.append(dict(path=str(lp),arrays=LABEL_KEYS,sha256=sha(lp)))
        o,c=labels['ORIGINAL_YCB'],labels['CONTROLLED']
        for rec in comps:
            if rec['split']!=('VALIDATION' if split=='VAL' else split):continue
            ot=rec['target_original'];don=rec['occluder_original'];tr=target[ot['id']]
            if ot['image']['sha256']!=tr['image']['sha256']:raise ValueError('original image lineage differs')
            tsplit='VAL' if tr['split']=='VALIDATION' else tr['split']
            dsplit='VAL' if don['split']=='VALIDATION' else don['split']
            for corner in range(8):
                osample=rec['original_sample_ids'][corner];csample=rec['id']+':'+str(corner)
                oi=o['_lookup'][osample];ci=c['_lookup'][csample]
                oj=si[osample];cj=si[csample]
                os=int(d['_state'][oj]);cs=int(d['_state'][cj])
                ob=o['global_box'][o['target_index'][oi]];cb=c['global_box'][c['target_index'][ci]]
                check_pair_identity(os,cs,o['query'][oi],c['query'][ci],ob,cb,o['side'][oi],c['side'][ci],split,tsplit,dsplit)
                if not np.array_equal(o['query'][oi],rec['query'][corner]) or not np.array_equal(ob,rec['box']):raise ValueError('manifest coordinate mismatch')
                if not np.array_equal(o['query'][oi],tr['query'][corner]) or not np.array_equal(ob,tr['box']):raise ValueError('original TARGETS coordinate mismatch')
                for table,i,j in [(o,oi,oj),(c,ci,cj)]:
                    if table['image_sha256'][i]!=d['image_sha256'][j] or int(table['dense_allowed'][i,4,4])!=int(a[j]):raise ValueError('joined-label identity/GT differs')
                    if str(table['global_id'][table['target_index'][i]])!=str(d['global_id'][d['target_index'][j]]):raise ValueError('joined-label target differs')
                    if not np.array_equal(table['global_box'][table['target_index'][i]],d['global_box'][d['target_index'][j]]):raise ValueError('joined-label box differs')
                    if int(table['native_corner'][i])!=corner:raise ValueError('native corner differs')
                if c['image_sha256'][ci]!=rec['image']['sha256']:raise ValueError('composite image SHA differs')
                pairtype='V_TO_E' if os==0 and cs==1 else ('V_TO_V' if os==0 and cs==0 else 'NON_V_UNCHANGED')
                pairrows.append(dict(split=split,pair_type=pairtype,original_joined_index=oj,
                    controlled_joined_index=cj,original_index=oj,controlled_index=cj,
                    original_sample_id=osample,controlled_sample_id=csample,
                    original_global_id=ot['id'],controlled_global_id=rec['id'],native_corner=corner,
                    original_exact_state=os,controlled_exact_state=cs,
                    original_allowed=int(a[oj]),controlled_allowed=int(a[cj]),
                    query_x=float(o['query'][oi,0]),query_y=float(o['query'][oi,1]),
                    side=float(o['side'][oi]),box_xyxy_json=json.dumps(ob.tolist(),separators=(',',':')),
                    original_image_sha256=ot['image']['sha256'],controlled_image_sha256=rec['image']['sha256'],
                    donor_id=don['id'],donor_image_sha256=don['image']['sha256'],
                    target_scene=int(ot['scene_id']),donor_scene=int(don['scene_id']),
                    GT_ONLY_MODEL_INPUT_FORBIDDEN=True))
    # Same target, original frame and donor frame have exactly one split, globally.
    identity_splits={}
    for rec in comps:
        sp='VAL' if rec['split']=='VALIDATION' else rec['split']
        for node in [rec['target_original'],rec['occluder_original']]:
            for identity in [('YCB_TARGET',node['id']),('YCB_RGB_SHA',node['image']['sha256']),('YCB_SCENE',node['scene_id'])]:
                identity_splits.setdefault(identity,set()).add(sp)
    if any(len(s)!=1 for s in identity_splits.values()):raise ValueError('target/image/donor/scene split leakage')
    if set(map(str,data['TRAIN']['image_sha256']))&set(map(str,data['VAL']['image_sha256'])):raise ValueError('TRAIN/VAL RGB collision')
    split_summary={}
    for source in SOURCES:
        a,b=[d for d in data.values()]
        ga=set(map(str,a['group_id'][a['source']==source]));gb=set(map(str,b['group_id'][b['source']==source]))
        if ga&gb:raise ValueError('cache group split overlap')
        split_summary[source]=dict(TRAIN_groups=len(ga),VAL_groups=len(gb),shared_groups=0)
    pal_sessions={sp:sorted({str(g).split(':')[1] for g in d['group_id'][d['source']=='PALLOX']}) for sp,d in data.items()}
    if set(pal_sessions['TRAIN'])&set(pal_sessions['VAL']):raise ValueError('PALLOX session crossing')
    paircounts={sp:dict(Counter(r['pair_type'] for r in pairrows if r['split']==sp)) for sp in ['TRAIN','VAL']}
    writecsv(HERE/'ALL_QUERY_COHORT_INDEX.csv',allrows)
    writecsv(HERE/'TARGET_COHORT_COUNTS.csv',targetrows)
    writecsv(HERE/'SOURCE_COHORT_COUNTS.csv',counts)
    writecsv(HERE/'GT_PAIR_INDEX.csv',pairrows)
    writecsv(HERE/'TRAIN_CHANGED_PAIR_INDEX.csv',[r for r in pairrows if r['split']=='TRAIN' and r['pair_type']=='V_TO_E'])
    writecsv(HERE/'TRAIN_PRESERVED_V_PAIR_INDEX.csv',[r for r in pairrows if r['split']=='TRAIN' and r['pair_type']=='V_TO_V'])
    writecsv(HERE/'VAL_PAIR_INDEX.csv',[r for r in pairrows if r['split']=='VAL'])
    result=dict(schema='V70_PRETRAINING_LABEL_COHORT_AUDIT_V1',completed=True,status='PASS',
        source_counts=counts,paired_original_controlled=paircounts,total_pair_rows=len(pairrows),
        all_query_rows=len(allrows),all_target_rows=len(targetrows),
        donor_target_RGB_scene_split_crossings=0,cache_group_audit=split_summary,
        PALLOX_session_split=pal_sessions,source_RGB_SHA_train_val_overlap=0,
        evaluation_denylist_collisions=0,arrays_opened=opened,
        input_lock=binding(lockpath),builder=binding(__file__),
        conditional_E_is_not_global_E=True,semantic_reference_labels_admitted=0,
        RGB_masks_features_weights_or_current_reads=0,forward=0,fit=0,
        prior_artifacts_modified=False)
    save(HERE/'LABEL_COHORT_AUDIT.json',result)
    files=['ALL_QUERY_COHORT_INDEX.csv','TARGET_COHORT_COUNTS.csv','SOURCE_COHORT_COUNTS.csv',
           'GT_PAIR_INDEX.csv','TRAIN_CHANGED_PAIR_INDEX.csv','TRAIN_PRESERVED_V_PAIR_INDEX.csv',
           'VAL_PAIR_INDEX.csv','LABEL_COHORT_AUDIT.json','LABEL_COHORT_PROTOCOL.json',
           'GENERATED_LABEL_COHORT_TESTS_V3.json']
    save(HERE/'LABEL_COHORT_SEAL.json',dict(schema='V70_LABEL_COHORT_SEAL_BEFORE_FIT',completed=True,
        artifacts=[binding(HERE/f) for f in files],training_authorized=False,
        current_access_authorized=False,GT_pair_index_not_model_input=True))
    print(json.dumps(dict(status='PASS',counts=counts,pairs=paircounts,seal=binding(HERE/'LABEL_COHORT_SEAL.json')),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mock',action='store_true');p.add_argument('--lock')
    args=p.parse_args()
    if args.mock:
        r=mock();r['builder']=binding(__file__);save(HERE/'GENERATED_LABEL_COHORT_TESTS_V3.json',r);print(json.dumps(r,indent=2))
    elif args.lock:build(args.lock)
    else:p.error('--mock or --lock required')
