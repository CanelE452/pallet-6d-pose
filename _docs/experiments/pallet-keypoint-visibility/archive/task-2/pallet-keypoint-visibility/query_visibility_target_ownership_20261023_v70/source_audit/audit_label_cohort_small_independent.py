"""Small independent63-mask and2 fixed pair GT/metadata replay; no features."""
from pathlib import Path
import ast,collections,csv,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'contract';V66=ROOT.parent/'query_visibility_controlled_real_20261022_v66'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(p)}
def scalar_state(mask,valid):
    if not valid:return -1
    exact=(0,1,1,2,2,2)
    options={exact[b] for b in range(6) if mask&(1<<b)}
    return next(iter(options)) if len(options)==1 else -1
def main():
    output=ROOT/'source_audit/INDEPENDENT_SMALL_LABEL_COHORT_AUDIT.json';assert not output.exists()
    protocol=json.loads((C/'LABEL_COHORT_PROTOCOL.json').read_text())
    published=json.loads((C/'LABEL_COHORT_AUDIT.json').read_text())
    tests=json.loads((C/'GENERATED_LABEL_COHORT_TESTS_V3.json').read_text())
    tree=ast.parse((C/'build_label_cohort_contract.py').read_text())
    wanted=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='exact_state']
    env={'np':np,'PAIRS':np.array([[0,0],[1,0],[1,1],[2,0],[2,1],[2,2]])}
    exec(compile(ast.Module(body=wanted,type_ignores=[]),'<isolated pure function>','exec'),env)
    masks=np.arange(1,64,dtype=np.uint8);actual=env['exact_state'](masks,np.ones(63,bool))
    expected=np.array([scalar_state(int(m),True) for m in masks],dtype=np.int8)
    assert np.array_equal(actual,expected)
    known={str(k):masks[expected==k].tolist() for k in [-1,0,1,2]}
    assert known['0']==[1] and known['1']==[2,4,6] and known['2']==[8,16,24,32,40,48,56]
    assert env['exact_state'](np.array([63]),np.array([False])).tolist()==[-1]
    illegal_rejected=0
    for mask,valid in [(1,False),(0,True),(64,True)]:
        try:env['exact_state'](np.array([mask]),np.array([valid]))
        except ValueError:illegal_rejected+=1
    assert illegal_rejected==3
    with (C/'GT_PAIR_INDEX.csv').open() as f:rows=list(csv.DictReader(f))
    counts={sp:dict(collections.Counter(r['pair_type'] for r in rows if r['split']==sp)) for sp in ['TRAIN','VAL']}
    assert len(rows)==15992==published['total_pair_rows']
    assert counts==published['paired_original_controlled']
    assert all(r['GT_ONLY_MODEL_INPUT_FORBIDDEN']=='True' for r in rows)
    assert all((int(r['original_exact_state']),int(r['controlled_exact_state']))==(0,1) for r in rows if r['pair_type']=='V_TO_E')
    assert all((int(r['original_exact_state']),int(r['controlled_exact_state']))==(0,0) for r in rows if r['pair_type']=='V_TO_V')
    assert all(int(r['original_exact_state'])==int(r['controlled_exact_state'])!=0 for r in rows if r['pair_type']=='NON_V_UNCHANGED')
    assert len({r['controlled_sample_id'] for r in rows})==len(rows)
    chosen=[next(r for r in rows if r['split']==sp and r['pair_type']=='V_TO_E') for sp in ['TRAIN','VAL']]
    composites={r['id']:r for r in map(json.loads,(V66/'controlled/COMPOSITES.jsonl').read_text().splitlines())}
    targets={r['id']:r for r in map(json.loads,(V66/'provenance/TARGETS.jsonl').read_text().splitlines())}
    opened=[];pairresults=[]
    for r in chosen:
        split=r['split'];cid=r['controlled_global_id'];rec=composites[cid];tr=targets[r['original_global_id']];don=rec['occluder_original'];k=int(r['native_corner'])
        normalize=lambda s:'VAL' if s=='VALIDATION' else s
        assert {normalize(rec['split']),normalize(tr['split']),normalize(don['split'])}=={split}
        assert rec['original_sample_ids'][k]==r['original_sample_id'] and r['controlled_sample_id']==cid+':'+str(k)
        arrays={}
        keys=['sample_id','global_id','target_index','query','side','global_box','image_sha256','dense_allowed','dense_valid','dense_exact_state','native_corner']
        for source in ['ORIGINAL_YCB','CONTROLLED']:
            path=V66/f'label_cache_v2/{source}_{split}.npz'
            with np.load(path,allow_pickle=False) as z:arrays[source]={key:z[key] for key in keys}
            opened.append({'path':str(path),'decoded_arrays':keys,'sha256':sha(path)})
        o,c=arrays['ORIGINAL_YCB'],arrays['CONTROLLED']
        oi=np.flatnonzero(o['sample_id']==r['original_sample_id']);ci=np.flatnonzero(c['sample_id']==r['controlled_sample_id']);assert len(oi)==len(ci)==1
        oi=int(oi[0]);ci=int(ci[0]);ob=o['global_box'][o['target_index'][oi]];cb=c['global_box'][c['target_index'][ci]]
        assert np.array_equal(o['query'][oi],c['query'][ci]) and np.array_equal(ob,cb)
        assert np.array_equal(ob,np.array(json.loads(r['box_xyxy_json']))) and np.array_equal(ob,np.array(rec['box']))
        assert np.array_equal(o['query'][oi],np.array([float(r['query_x']),float(r['query_y'])]))
        assert np.array_equal(o['query'][oi],np.array(tr['query'][k])) and np.array_equal(o['query'][oi],np.array(rec['query'][k]))
        assert o['side'][oi]==c['side'][ci]==float(r['side'])
        jp=V66/f'joined_source_cache/CONTROLLED_PLUS_PALLOX_{split}.npz'
        jkeys=['sample_id','source','dense_allowed','dense_valid','dense_exact_state','image_sha256']
        with np.load(jp,allow_pickle=False) as z:joined={key:z[key] for key in jkeys}
        opened.append({'path':str(jp),'decoded_arrays':jkeys,'feature_payload_not_read_or_rehashed':True})
        for table,ix,col,prefix in [(o,oi,'original_joined_index','original'),(c,ci,'controlled_joined_index','controlled')]:
            j=int(r[col]);assert joined['sample_id'][j]==r[prefix+'_sample_id']
            assert table['image_sha256'][ix]==joined['image_sha256'][j]==r[prefix+'_image_sha256']
            a=int(table['dense_allowed'][ix,4,4]);v=bool(table['dense_valid'][ix,4,4]);st=scalar_state(a,v)
            assert a==int(r[prefix+'_allowed'])==int(joined['dense_allowed'][j,4,4])
            assert st==int(r[prefix+'_exact_state'])==int(table['dense_exact_state'][ix,4,4])==int(joined['dense_exact_state'][j,4,4])
            assert int(table['native_corner'][ix])==k
        assert tr['image']['sha256']==r['original_image_sha256']
        assert rec['image']['sha256']==r['controlled_image_sha256']
        assert don['image']['sha256']==r['donor_image_sha256'] and don['id']==r['donor_id']
        pairresults.append({'split':split,'original_sample_id':r['original_sample_id'],'controlled_sample_id':r['controlled_sample_id'],'known_V_allowed':int(r['original_allowed']),'known_E_allowed':int(r['controlled_allowed']),'same_query_side_box':True,'target_donor_split_parity':True,'label_joined_CSV_parity':True})
    result={'schema':'INDEPENDENT_SMALL_V70_LABEL_COHORT_AUDIT_V1','status':'PASS','scope':'Pure63mask projection + all15992 CSV state/count invariants + fixed first TRAIN/VAL changed pair GT/geometry/lineage replay. Not a feature or training implementation audit.',
        'pure_mask_logic':{'all63_match':True,'known_exact_masks_by_state':known,'invalid63_is_unknown':True,'illegal_inputs_rejected':illegal_rejected},
        'CSV_pair_rows':len(rows),'CSV_pair_counts':counts,'fixed_pair_replay':pairresults,
        'metric_scope_review':'Protocol preserves ALL native-valid known V/E/O for fixed quality precision/recall. VE-only selector is separately named; original YCB has no known E and cannot use VE-only precision as the quality gate. O/unknown are not relabeled E.',
        'findings':[],'major_blocker_found':False,'training_or_current_authorized_by_this_audit':False,
        'opened_arrays':opened,'bindings':[bind(C/n) for n in ['LABEL_COHORT_PROTOCOL.json','build_label_cohort_contract.py','GENERATED_LABEL_COHORT_TESTS_V3.json','LABEL_COHORT_AUDIT.json','GT_PAIR_INDEX.csv','LABEL_COHORT_SEAL.json']],
        'RGB_mask_depth_feature_weight_current_arrays_read':0,'forward_fit_PnP':0,'prior_artifacts_modified':False,'audit_code':bind(Path(__file__))}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'path':str(output),'sha256':sha(output),'status':'PASS','pairs':pairresults},ensure_ascii=False))
if __name__=='__main__':main()
