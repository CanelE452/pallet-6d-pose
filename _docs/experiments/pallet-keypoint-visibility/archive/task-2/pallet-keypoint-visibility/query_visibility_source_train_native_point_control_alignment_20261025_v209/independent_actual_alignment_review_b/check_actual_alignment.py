"""Independent saved TRAIN alpha/point QA; decode exactly foreign_alpha only."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,io,json,math,os,stat,traceback
import numpy as np
H=Path(__file__).parent;V=H.parent;P=V.parent
S=V/'ROOT_ACTUAL_SOURCE_TRAIN82_ALPHA_ALIGNMENT_SCOPE.json'
C=V/'source_train_native_point_control_alignment_v1/COMPLETE.json'
S_SHA='96276f1d86c94a7b8a347c418c13ab8ef027a79c0d99e64f46fe3f941753dc5d'
C_SHA='fe44f7650a570f4a364d231ecab7453aaf83a176c99239651970f2d406b1756b'

def raw(path,asset=None):
    p=Path(path)
    assert p.is_absolute()and P in p.parents and '..'not in p.parts and 'v112'not in str(p).casefold()
    assert p.name!='SCENES_GT_ONLY.json','No original GT access'
    if asset is None:assert p.suffix in {'.json','.jsonl','.py'}
    else:assert p.suffix=='.npz'and asset['path']==str(p)and p.name=='CONTROL_TRIPLET_INPUTS.npz'
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    assert stat.S_ISREG(os.fstat(n).st_mode)
    with os.fdopen(n,'rb')as f:return f.read()
def binding(path):return dict(path=str(path),sha256=hashlib.sha256(raw(path)).hexdigest())
def get(b):
    z=raw(b['path']);assert hashlib.sha256(z).hexdigest()==b['sha256'];return json.loads(z)
def lines(b):
    z=raw(b['path']);assert hashlib.sha256(z).hexdigest()==b['sha256'];return [json.loads(s)for s in z.splitlines()if s.strip()]
def put(name,obj):
    with(H/name).open('x',encoding='utf-8')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def independent_relation(alpha,ref):
    if ref is None or len(ref)!=2 or not all(math.isfinite(t)for t in ref):
        return dict(status='UNKNOWN',positive_taps=[],reference_in_canvas_extent=None,reference_in_integer_center_extent=None)
    x,y=ref;h,w=alpha.shape;taps=[]
    # Independent linear basis on neighboring integer centers; no projection/ray.
    for j in range(math.floor(y),math.floor(y)+2):
        for i in range(math.floor(x),math.floor(x)+2):
            weight=max(0.,1-abs(x-i))*max(0.,1-abs(y-j))
            if weight>0:
                inside=0<=i<w and 0<=j<h
                taps.append(dict(xy=[i,j],weight=weight,inside=inside,opaque=bool(alpha[j,i])if inside else None))
    flags=[t['opaque']for t in taps]
    status='UNKNOWN'if any(f is None for f in flags)else'COVERED'if all(flags)else'CLEAR'if not any(flags)else'EDGE_UNKNOWN'
    return dict(status=status,positive_taps=taps,reference_in_canvas_extent=(-.5<=x<w-.5 and -.5<=y<h-.5),reference_in_integer_center_extent=(0<=x<=w-1 and 0<=y<=h-1))

def run():
    assert binding(S)['sha256']==S_SHA and binding(C)['sha256']==C_SHA
    scope=get(binding(S));complete=get(binding(C));assert scope['approved_SOURCE_TRAIN_only']is True and complete['completed']and complete['PASS']
    entry=get(complete['entry']);assert entry['approved_SOURCE_TRAIN_point_alpha_alignment']is True and entry['source_TRAIN_only']is True
    assert entry['root_actual_scope']==binding(S)and entry['decode_only_foreign_alpha']is True and entry['new_labels_or_admission']is False
    assert complete['whitelist']==scope['exact_whitelist']==entry['whitelist']and complete['associations']==scope['associations']==entry['associations']
    assert scope['decoded_members_only']==['foreign_alpha']and scope['new_native_labels_or_training_admission']is False
    wl=get(scope['exact_whitelist']);before=lines(scope['associations'])
    saved=lines(complete['outputs']['ALL1968_TRAIN_NATIVE_POINT_ALPHA_ALIGNMENT.jsonl']);summary=get(complete['outputs']['SUMMARY.json'])
    assert len(wl['packets'])==82 and len(before)==len(saved)==1968
    assert [r['control_join_key']for r in saved]==[r['control_join_key']for r in before]
    assert len({tuple(r['control_join_key'])for r in saved})==1968 and len({r['row_id']for r in saved})==1968
    saved_by={r['row_id']:r for r in saved};by_attempt=defaultdict(list)
    for row in before:
        assert row['role']=='TRAIN'and row['control_join_key']==['SYNTH_FINITE_END',row['attempt_id'],row['variant'],row['native_slot']]
        assert all(saved_by[row['row_id']][k]==value for k,value in row.items()),'Original q/L/native/U/source fields drift'
        assert row['new_native_point_state']is None and row['new_visibility_target_created']is False and row['training_admission']is False
        by_attempt[row['attempt_id']].append(row)
    assert len(by_attempt)==82
    declarations={p['attempt_id']:p for p in wl['packets']};assert set(declarations)==set(by_attempt)
    counts={v:dict(all_native=Counter(),requested_native=Counter(),other_native=Counter(),by_original_point0_state={s:Counter()for s in ('D','S','UNKNOWN')})for v in ('BEFORE','COVER','SHAM')}
    triplets=Counter();requested_sources=Counter();physical_cross=Counter();masses=[];max_weight_error=0.;opened=[];inputU=0
    for packet in wl['packets']:
        name=packet['attempt_id'];b=packet['binding'];rr=by_attempt[name]
        assert len(rr)==24 and packet['role']=='TRAIN'and packet['allowed_npz_members_to_decode']==['foreign_alpha']
        assert {(r['variant'],r['native_slot'])for r in rr}=={(v,k)for v in ('BEFORE','COVER','SHAM')for k in range(8)}
        assert all(r['source_packet']==b and r['raw_hw']==packet['raw_hw']for r in rr)
        expected=str(P/'query_visibility_sampled_domain_direct_end_20261024_v189/synthetic_sampled_end_controls_v1'/name/'CONTROL_TRIPLET_INPUTS.npz')
        assert b['path']==expected,'Exact original frozen packet whitelist before access'
        payload=raw(b['path'],asset=b);assert hashlib.sha256(payload).hexdigest()==b['sha256'],'Opaque full container identity before member decode'
        with np.load(io.BytesIO(payload),allow_pickle=False)as archive:
            alpha=archive['foreign_alpha'] # Sole decoded member; no iteration/access of other members.
        assert alpha.dtype==np.bool_ and list(alpha.shape)==packet['expected_shape']==[3,*packet['raw_hw']]
        assert not alpha[0].any()
        opened.append(b);masses.append(dict(attempt_id=name,BEFORE=int(alpha[0].sum()),COVER=int(alpha[1].sum()),SHAM=int(alpha[2].sum())))
        requested=[];requested_source=[]
        for row in rr:
            v=row['variant'];vindex=('BEFORE','COVER','SHAM').index(v)
            result=independent_relation(alpha[vindex],row['saved_anchor_projected_reference_xy']);stored=saved_by[row['row_id']]
            relation=stored['foreground_alpha_relation']
            for key in ('status','reference_in_canvas_extent','reference_in_integer_center_extent'):assert relation[key]==result[key],(row['row_id'],key)
            assert len(relation['positive_taps'])==len(result['positive_taps'])
            for actual,want in zip(relation['positive_taps'],result['positive_taps']):
                for key in ('xy','inside','opaque'):assert actual[key]==want[key]
                difference=abs(actual['weight']-want['weight']);max_weight_error=max(max_weight_error,difference);assert difference<1e-13
            source=row['saved_original_point0_combined_state'];status=result['status']
            assert stored['original_input_U_preserved']==(not row['original_query_available'])
            inputU+=int(not row['original_query_available']);counts[v]['all_native'][status]+=1;counts[v]['by_original_point0_state'][source][status]+=1
            physical_cross[(source,status)]+=1
            if row['native_slot']==row['requested_native_slot']:
                counts[v]['requested_native'][status]+=1;requested.append((vindex,status));requested_source.append(source)
            else:counts[v]['other_native'][status]+=1
        assert len(requested)==3 and len(set(requested_source))==1
        requested_sources[requested_source[0]]+=1;triplets[tuple(s for i,s in sorted(requested))]+=1
    assert len(opened)==len({b['path']for b in opened})==82
    recomputed={}
    for v,c in counts.items():
        recomputed[v]=dict(rows=sum(c['all_native'].values()),all_native=dict(c['all_native']),requested_native=dict(c['requested_native']),by_original_point0_state={s:dict(x)for s,x in c['by_original_point0_state'].items()})
    assert recomputed==summary['variants']==complete['variants']
    assert inputU==summary['original_inputU_rows']==complete['original_inputU_rows']
    assert summary['new_native_point_states_created']==complete['new_native_point_states_created']==0
    assert summary['new_labels_or_admission']is False and complete['new_labels_or_admission']is False
    decoded=complete['decoded_packet_members'];assert len(decoded)==82
    assert [d['binding']for d in decoded]==opened
    assert all(d['decoded_members']==['foreign_alpha']and d['shape']==declarations[wl['packets'][i]['attempt_id']]['expected_shape']for i,d in enumerate(decoded))
    result=dict(completed=True,PASS=True,code=binding(Path(__file__)),root_actual_scope=binding(S),root_actual_entry=complete['entry'],complete=binding(C),
        whitelist=scope['exact_whitelist'],associations=scope['associations'],saved_rows=complete['outputs']['ALL1968_TRAIN_NATIVE_POINT_ALPHA_ALIGNMENT.jsonl'],
        unique_container_SHA_verified=82,only_decoded_member='foreign_alpha',all1968_original_fields_controlkey_order_and_positive_taps_exact=True,
        original_points=len({(r['source_base_index'],r['native_slot'])for r in before}),parents=len({r['source_base_index']for r in before}),physical_rows=len(before),inputU=inputU,
        variants=recomputed,other_native_relations={v:dict(c['other_native'])for v,c in counts.items()},
        requested_triplet_patterns=[dict(BEFORE=p[0],COVER=p[1],SHAM=p[2],N=n)for p,n in sorted(triplets.items())],requested_original_source_point_states=dict(requested_sources),
        source_point_vs_alpha_relation_physical_rows=[dict(saved_BEFORE_point0=s,alpha_relation=a,N=n)for(s,a),n in sorted(physical_cross.items())],
        saved_alpha_positive_pixel_counts=masses,maximum_independent_positive_tap_weight_difference=max_weight_error,
        producer_or_loader_imports=0,original_GT_RGB_owner_depth_endmask_raw96_valid_SAM_or_other_member_decode=0,models_fit_projection_ray_PnP_or_protected_access=0,
        new_labels_or_training_admission=False,
        authority='Saved procedural original point D/S and changed-control raster opacity remain separate. CLEAR/COVERED do not themselves define new native visibility labels, physical gold, observed RGB evidence or continuous-ray results.')
    put('REVIEW.json',result)
    with(H/'KO.md').open('x',encoding='utf-8')as f:
        f.write('V209 실제 saved-alpha 정합 독립 검산 PASS. 승인된 TRAIN82 container SHA를 먼저 확인하고 foreign_alpha 한 member만 해독했다. 1968행의 원 q/L/native/U·control-key·순서와 저장 reference의 모든 양수 bilinear tap 좌표·가중치·opacity가 일치했다.\n\n')
        f.write('원 BEFORE point D/S와 raster CLEAR/COVERED/EDGE_UNKNOWN/UNKNOWN은 별도 축이다. 새 native E·target·학습 admission을 만들지 않았고, 원 GT·RGB·owner·depth·raw96·SAM·모델·투영·ray 접근은 0이다.\n')
    print(json.dumps(dict(completed=True,PASS=True,review=binding(H/'REVIEW.json'),packets=82,rows=1968,variants=recomputed,requested_patterns=result['requested_triplet_patterns'],source_cross=result['source_point_vs_alpha_relation_physical_rows'],max_weight_error=max_weight_error)))

if __name__=='__main__':
    try:run()
    except Exception as error:
        put('FAILURE.json',dict(completed=False,PASS=False,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),automatic_blind_retry=False))
        raise
