"""Bounded saved root-record consistency, not new visual or semantic labeling."""
from pathlib import Path
from collections import Counter
import ast,csv,hashlib,io,json

V=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_public_train_part_query_support_contract_20261025_v216')
O=V/'final_scientific_report_review_c'
def read(p):
    p=Path(p);assert 'v112' not in str(p).casefold() and '..' not in p.parts
    assert V in p.parents
    for q in [p,*p.parents]:assert not q.is_symlink()
    assert p.suffix in ('.json','.csv','.py','.md','.html')
    return p.read_bytes()
def bind(p):b=read(p);return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))
def bound(b):
    raw=read(b['path']);assert hashlib.sha256(raw).hexdigest()==b['sha256']
    if 'size_bytes' in b:assert len(raw)==b['size_bytes']
    return json.loads(raw)
def bid(b):return b['path'],b['sha256']
def main():
    code=V/'ROOT_RECORD_ACTUAL_ALL12_SEMANTIC_QA.py';rootpath=V/'ROOT_ACTUAL_ALL12_VISUAL_AND_SEMANTIC_QA.json';csvpath=V/'ROOT_ALL96_SEMANTIC_REVIEW.csv'
    tree=ast.parse(read(code));withheld=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='WITHHELD' for x in n.targets));chosen=ast.literal_eval(withheld.value)
    assert set(chosen)=={(4,3),(6,7)}
    root=json.loads(read(rootpath));assert root['completed'] is True and root['PASS'] is True
    joined=bound(root['sealed_raw96'])['rows96'];complete=bound(root['sealed_raw_join']);qa=bound(root['independent_saved_numeric_validity_QA'])
    assert complete['completed'] and complete['PASS'] and qa['completed'] and qa['PASS']
    rows=root['rows96'];actual=list(csv.DictReader(io.StringIO(read(csvpath).decode())))
    assert len(joined)==len(rows)==len(actual)==96
    for i,(j,r,c) in enumerate(zip(joined,rows,actual)):
        m=j['input'];raw=j['raw_helper_candidate'];pos,n=divmod(i,8)
        assert r['row_id']==m['row_id'] and r['frame_position']==pos==m['position'] and r['native_slot']==n==m['native_slot']
        assert r['frame_key']==m['frame_key'] and r['internal_source_role']==m['internal_source_role'] and r['original_role']==m['role']=='TRAIN'
        assert r['original_inputU']==(not m['original_query_available']) and r['raw_native_weak_candidate']==raw['native_weak_visibility_agreement_candidate'] and r['raw_spatial_witness_candidate']==raw['spatial_witness_agreement_candidate'] and r['raw_query_support_agreement']==raw['query_support_agreement']
        if r['original_inputU']:status='ORIGINAL_INPUT_U_RETAINED'
        elif (pos,n) in chosen:
            assert raw['native_weak_visibility_agreement_candidate']=='D';status='RAW_D_WITHHELD_SAME_FEATURE_UNRESOLVED'
            assert r['reason_ko']==chosen[pos,n]
        elif raw['native_weak_visibility_agreement_candidate']=='D':status='SAME_PALLET_PART_VISIBLE_WEAK'
        elif raw['spatial_witness_agreement_candidate']:status='LOCAL_FEATURE_ONLY_NATIVE_UNRESOLVED'
        else:status='NO_NATIVE_CANDIDATE_RETAIN_UNKNOWN'
        assert r['root_semantic_review_status']==status and r['root_weak_part_state']==('D' if status=='SAME_PALLET_PART_VISIBLE_WEAK' else 'UNKNOWN')
        assert r['native_coordinate_correctness_GT']==r['physical_native_point_visibility_gold']=='UNKNOWN' and r['primary_DE_training_admission'] is False and r['auxiliary_training_admission'] is False
        assert c=={k:str(v) for k,v in r.items()}
    counts={}
    for role in ('ALL','SOURCE_FIT','SOURCE_DEV'):
        rr=[r for r in rows if role=='ALL' or r['internal_source_role']==role]
        counts[role]=dict(N=len(rr),original_inputU=sum(r['original_inputU'] for r in rr),root_weak_part_state=dict(Counter(r['root_weak_part_state'] for r in rr)),review_status=dict(Counter(r['root_semantic_review_status'] for r in rr)))
    assert counts==root['counts'] and counts['ALL']['root_weak_part_state']=={'D':9,'UNKNOWN':87}
    assert counts['SOURCE_FIT']['root_weak_part_state'].get('D',0)==7 and counts['SOURCE_DEV']['root_weak_part_state'].get('D',0)==2
    native={str(n):dict(N=sum(r['native_slot']==n for r in rows),raw_D=sum(r['native_slot']==n and r['raw_native_weak_candidate']=='D' for r in rows),root_weak_D=sum(r['native_slot']==n and r['root_weak_part_state']=='D' for r in rows),original_inputU=sum(r['native_slot']==n and r['original_inputU'] for r in rows)) for n in range(8)}
    assert sum(x['raw_D'] for x in native.values())==11 and sum(x['root_weak_D'] for x in native.values())==9
    views=root['all12_actual_viewed'];assert [r['frame_position'] for r in views]==list(range(12))
    for f,c in zip(views,complete['new_overlay_PNGs']):
        assert f['actually_viewed'] is True and 'view_image' in f['viewer'] and f['frame_key']==c['frame_key'] and bid(f['actual_overlay_view'])==bid(c['binding'])
    extra=root['additional_two_actual_clean96_views'];assert {(x['frame_position'],x['native_slot']) for x in extra}==set(chosen)
    for x in extra:assert x['actually_viewed'] is True and bid(x['binding'])==bid(joined[x['frame_position']*8+x['native_slot']]['display_bindings']['clean96'])
    result=dict(completed=True,PASS=True,provisional_root_record_only=True,root_recording_code=bind(code),root_visual_semantic_record=bind(rootpath),root_semantic_CSV=bind(csvpath),raw_join=root['sealed_raw_join'],independent_saved_QA=root['independent_saved_numeric_validity_QA'],counts=counts,native8=native,withheld_rows=[r for r in rows if r['root_semantic_review_status']=='RAW_D_WITHHELD_SAME_FEATURE_UNRESOLVED'],root_declared_actual_views=dict(overlays12=True,additional_original_clean96=2),C_new_image_views=0,overlay_PNG_body_SHA_not_rehashed_in_this_check=True,unchanged_raw_D11_vs_root_weak_D9_distinct=True,no_gold_or_training_admission=True,final_report_READY_still_required=True)
    p=O/'ROOT_RECORD_CONSISTENCY.json'
    with p.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,receipt=bind(p),counts=counts,native8=native),ensure_ascii=False))
if __name__=='__main__':main()
