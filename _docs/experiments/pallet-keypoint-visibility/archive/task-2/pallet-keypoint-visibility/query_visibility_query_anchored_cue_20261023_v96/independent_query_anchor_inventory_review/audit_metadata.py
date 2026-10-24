"""Independent source metadata replay. Does not import the inventory builder."""
from pathlib import Path
import json, hashlib, math, itertools
from collections import Counter, defaultdict

HERE=Path(__file__).resolve().parent
V=HERE.parent
P=V.parent
def bind(p):
    p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def write(p,obj):
    with Path(p).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')
def load(b):
    assert bind(b['path'])['sha256']==b['sha256'],b
    return [json.loads(s) for s in Path(b['path']).read_text().splitlines() if s]
def j(p):return json.loads(Path(p).read_text())
def stencil(q,L):
    if q is None or L is None or not all(math.isfinite(z) for z in [*q,L]) or L<=0:return set()
    # Independent separable support: every x/y sample combination is present.
    axes=[]
    for c in q:
        support=set()
        for a in (c-L/192.,c+L/192.):
            k=math.floor(a);support.add(k)
            if a!=k:support.add(k+1)
        axes.append(support)
    return set(itertools.product(*axes))
def hits(box,foot,hw,closed):
    if box is None or len(box)!=4 or not all(math.isfinite(x) for x in box):return None
    x0,y0,x1,y1=box;h,w=hw
    if x1<x0 or y1<y0 or (not closed and (x0==x1 or y0==y1)):return None
    if closed:
        if min(x0,y0)<0 or x1>w-1 or y1>h-1:return None
        xmax,ymax=math.floor(x1+.5),math.floor(y1+.5)
    else:
        if x0<-.5 or y0<-.5 or x1>w-.5 or y1>h-.5:return None
        xmax,ymax=math.ceil(x1+.5)-1,math.ceil(y1+.5)-1
    xmin,ymin=math.floor(x0+.5),math.floor(y0+.5)
    return {(x,y) for x,y in foot if xmin<=x<=xmax and ymin<=y<=ymax}
def coverage(q,L,b):
    if q is None:return 'NO_QUERY'
    if b is None:return 'NULL_REFERENCE_U'
    a,c=q[0]-L/2,q[0]+L/2;b0,d=q[1]-L/2,q[1]+L/2
    x0,y0,x1,y1=b
    if a<=x0 and b0<=y0 and x1<=c and y1<=d:return 'FULL'
    return 'PARTIAL' if min(x1,c)>max(x0,a) and min(y1,d)>max(y0,b0) else 'MISS'

def main():
    assert not (HERE/'REVIEW.json').exists(),'No overwrite'
    proto=j(V/'INVENTORY_PROTOCOL_BEFORE_QUERY_ANCHOR_LABELS.json')
    write(HERE/'SCOPE_BEFORE_REPLAY.json',dict(completed=True,mode='METADATA_ONLY_INDEPENDENT_REPLAY',
        protocol=bind(V/'INVENTORY_PROTOCOL_BEFORE_QUERY_ANCHOR_LABELS.json'),code=bind(__file__),
        publisher_builder=bind(V/'build_metadata_inventory.py'),published_ledger=bind(V/'ALL960_QUERY_ANCHORED_AUX_INVENTORY_GT_ONLY.jsonl'),
        inputs=proto['inputs'],exposure='Inventory protocol/code/summary and sample metadata were read before this scope. Root also sent aggregate C1 diagnostic counts; no score bodies were read or used. No blind claim.',
        independent_arithmetic='Separable FP64 axis support Cartesian product; own inclusive integer intervals; original604-view and4832 cross-product joins; no import of publisher builder.',
        comparison='All960 output identity, source/rawfold, oldlabel, footprint, group evidence and decision fields, no threshold changes.',
        next_public_QA_selection='Union of all new E, all old U to new known, and minimum original-row D per source/rawfold. Fixed from metadata, dedup original row order, before any crop-array reads.',
        forbidden=['RGB pixels before metadata replay','P3 arrays','models/weights/forward/fit','current/private data','PnP','V94/V95 mutation']))
    d={k:load(b) for k,b in proto['inputs'].items()}
    out=[json.loads(x) for x in (V/'ALL960_QUERY_ANCHORED_AUX_INVENTORY_GT_ONLY.jsonl').read_text().splitlines()]
    assert list(map(len,[d['rows960'],d['original_labels960'],d['loco_root160'],d['pallox800'],d['all_cue_cross4832'],d['original_cue_views604'],out]))==[960,960,160,800,4832,604,960]
    teachers={r['cue_view_index']:r for r in d['original_cue_views604']};assert len(teachers)==604
    assert set(teachers)==set(range(604))
    frame_teachers=defaultdict(list)
    for t in teachers.values():
        frame_teachers[t['frame_input_id']].append(t['cue_view_index'])
        assert t['native8_semantic_id'] is None and not t['human_gold'] and not t['GT_forward_allowed']
    cross=defaultdict(list)
    for c in d['all_cue_cross4832']:
        t=teachers[c['cue_view_index']]
        for a,b in [('cue_id','cue_id'),('pair_id','pair_id'),('view','view'),('split','split'),('frame_input_id','frame_input_id'),('teacher_label','teacher_label'),('cue_reference_xy_unchanged','reference_xy_unchanged'),('cue_uncertainty_bbox_unchanged','uncertainty_bbox_xyxy_half_open_unchanged'),('certified_duplicate_group','duplicate_group_link_only'),('duplicate_group_size','duplicate_group_size'),('duplicate_weight','duplicate_weight')]:assert c[a]==t[b],(a,c['query_id'])
        assert c['reference_null']==(t['reference_xy_unchanged'] is None)
        assert c['coverage']==coverage(c['original_q'],c['original_L'],t['uncertainty_bbox_xyxy_half_open_unchanged'])
        cross[c['query_id']].append(c)
    assert len(cross)==800
    assert len({(r['input_id'],r['native_slot_logical_only']) for r in d['rows960']})==960
    residuals=[];unknown_touch=[];partial_touch=[]
    for i,(r,g,s) in enumerate(zip(d['rows960'],d['original_labels960'],out)):
        assert r['slot_index']==g['base_row_index']==s['row_index']==i
        assert [r['input_id'],r['native_slot_logical_only']]==g['query_key']
        for a,b in [('input_id','input_id'),('split','split'),('image_sha256','image_sha256')]:assert r[a]==g[b]
        assert r['query_xy']==g['original_q'] and r['L']==g['original_L']
        expected_identity=dict(input_id=r['input_id'],native_slot_logical_only=r['native_slot_logical_only'],source=g['source'],raw_split=r['split'],original_q=r['query_xy'],original_L=r['L'],image_sha256=r['image_sha256'],raw_hw=r['raw_hw'],original_input_valid=r['original_input_valid'],original_C1_V95_label_D0_E1_Uminus1=g['original_natural_label_GT_only'])
        for k,v in expected_identity.items():assert s[k]==v,(i,k)
        assert s['native_expected_part_state']=='UNKNOWN' and not any(s[k] for k in ['human_gold','training_admission','GT_fields_may_forward'])
        q,L,hw=r['query_xy'],r['L'],r['raw_hw'];foot=stencil(q,L);reasons=[]
        if not foot:reasons.append('MISSING_OR_NONFINITE_QUERY_U')
        if foot and not r['original_input_valid']:reasons.append('ORIGINAL_QUERY_INPUT_UNAVAILABLE_U')
        if any(not (0<=x<hw[1] and 0<=y<hw[0]) for x,y in foot):reasons.append('CENTRAL_NATIVE_BILINEAR_STENCIL_OOB_U')
        groups={};unlocated=set();bad=set();candidates=[];touch=[];views=[];matched=None;lab=-1
        if i<160:
            a=d['loco_root160'][i];assert a['query_key']==g['query_key'] and a['original_q']==q and a['original_L']==L and a['split']==r['split']
            assert g['original_natural_label_GT_only']==(0 if a['before_aux_state']=='D' else -1)
            if a['before_aux_state']=='D' and foot:
                masks=[hits(a[k],foot,hw,False) for k in ['support_bbox_xyxy','uncertainty_bbox_xyxy']]
                if any(m is None for m in masks):reasons.append('LOCO_UNUSABLE_CERTIFIED_GEOMETRY_U')
                elif set.union(*masks)==foot:candidates=['LOCO_SAME_QUERY_ROOT_SINGLE_CUE'];matched=candidates[0]
                else:reasons.append('CENTRAL_FOOTPRINT_NOT_FULLY_IN_ROOT_CUE_U')
            else:reasons.append('LOCO_SAME_QUERY_ROOT_NOT_KNOWN_D_U')
            if not reasons and matched is not None:lab=0
        else:
            a=d['pallox800'][i-160];assert a['row_index']==i-160
            for k,v in [('frame_input_id',r['input_id']),('native_slot',r['native_slot_logical_only']),('original_q',q),('original_L',L),('split',r['split']),('image_sha256',r['image_sha256']),('raw_hw',hw),('scalar_target_D0_E1_Uminus1',g['original_natural_label_GT_only'])]:assert a[k]==v,(i,k)
            xs=cross[a['query_id']];views=[x['cue_view_index'] for x in xs]
            assert views==a['source_GT_cue_view_indices']==frame_teachers[a['frame_input_id']]
            assert len(views)==len(set(views))
            retained={x['cue_view_index'] for x in a['retained_FULL_GT_cues']}
            assert retained=={x['cue_view_index'] for x in xs if x['coverage']=='FULL' and x['teacher_label'] in [0,1]}
            for x in xs:
                t=teachers[x['cue_view_index']]
                assert t['image_sha256']==r['image_sha256'] and t['raw_hw']==hw
                assert x['original_q']==q and x['original_L']==L and x['native_slot']==r['native_slot_logical_only']
                key=t['duplicate_group_link_only']
                v=groups.setdefault(key,dict(views=[],original_coverage=[],states=set(),unknown=False,touch=set(),full=set()))
                v['views'].append(t['cue_view_index']);v['original_coverage'].append(x['coverage'])
                if t['teacher_label'] in [0,1]:v['states'].add(t['teacher_label'])
                else:v['unknown']=True
                if t['reference_xy_unchanged'] is None or t['uncertainty_bbox_xyxy_half_open_unchanged'] is None:unlocated.add(key);continue
                hit=hits(t['uncertainty_bbox_xyxy_half_open_unchanged'],foot,hw,True)
                if hit is None:bad.add(key);continue
                v['touch'].update(hit)
                if hit and t['teacher_label']==-1:unknown_touch.append((i,t['cue_view_index']))
                if hit and x['coverage']=='PARTIAL':partial_touch.append((i,t['cue_view_index']))
                if t['cue_view_index'] in retained:v['full'].update(hit)
            candidates=[key for key,v in groups.items() if foot and v['full']==foot]
            touch=[key for key,v in groups.items() if v['touch']]
            if len(candidates)==0:reasons.append('NO_KNOWN_FULL_GROUP_CONTAINS_ENTIRE_CENTRAL_STENCIL_U')
            elif len(candidates)>1:reasons.append('MULTIPLE_DISTINCT_FULL_CANDIDATE_GROUPS_U')
            else:matched=candidates[0]
            if unlocated:reasons.append('UNLOCATED_POTENTIAL_CUE_COMPETITION_U')
            if bad:reasons.append('UNUSABLE_CUE_GEOMETRY_POTENTIAL_COMPETITION_U')
            if matched is not None:
                if set(touch)!={matched}:reasons.append('OTHER_DISTINCT_LOCATED_GROUP_TOUCHES_CENTRAL_STENCIL_U')
                v=groups[matched]
                if v['unknown'] or len(v['states'])!=1:reasons.append('MATCHED_GROUP_UNKNOWN_OR_CONTRADICTORY_STATE_U')
                if not reasons:lab=list(v['states'])[0]
        if lab!=-1:reasons=['QUALIFIED_QUERY_ANCHORED_D_AUX' if lab==0 else 'QUALIFIED_QUERY_ANCHORED_E_AUX']
        expected=dict(central4_positive_bilinear_native_cells=[list(p) for p in sorted(foot)],candidate_certified_groups=candidates,matched_certified_group_GT_only=matched if lab!=-1 else None,all_touching_groups=touch,unlocated_potential_competing_groups=sorted(unlocated),unusable_geometry_groups=sorted(bad),source_cue_view_indices=views,query_anchored_AUX_label_D0_E1_Uminus1=lab,primary_reason=reasons[0],all_reasons=reasons,
            group_evidence_GT_only={k:dict(views=v['views'],original_coverage=v['original_coverage'],known_states=sorted(v['states']),has_unknown=v['unknown'],central_cells_touched=len(v['touch']),central_cells_in_FULL_known_union=len(v['full'])) for k,v in groups.items()})
        for k,v in expected.items():assert s[k]==v,(i,k,s[k],v)
        residuals.append(dict(row_index=i,label=lab,stencil_count=len(foot),candidate_groups=len(candidates),touching_groups=len(touch),unlocated_groups=len(unlocated)))
    labels=Counter(x['query_anchored_AUX_label_D0_E1_Uminus1'] for x in out)
    assert labels=={0:61,1:4,-1:895}
    summary=j(V/'QUERY_ANCHORED_INVENTORY_SUMMARY.json')
    bygroup={s+'|'+sp:{str(k):sum(x['source']==s and x['raw_split']==sp and x['query_anchored_AUX_label_D0_E1_Uminus1']==k for x in out) for k in [0,1,-1]} for s,sp in sorted({(x['source'],x['raw_split']) for x in out})}
    transitions={str(a)+'->'+str(b):sum(x['original_C1_V95_label_D0_E1_Uminus1']==a and x['query_anchored_AUX_label_D0_E1_Uminus1']==b for x in out) for a in [0,1,-1] for b in [0,1,-1]}
    assert summary['labels']=={str(k):v for k,v in labels.items()} and summary['by_source_raw_split']==bygroup and summary['old_to_new_label_counts']==transitions
    assert summary['primary_reason_counts']==dict(Counter(x['primary_reason'] for x in out))
    assert summary['all_reason_counts']==dict(Counter(r for x in out for r in x['all_reasons']))
    assert {a for a,b in unknown_touch}=={546,547,554} and not partial_touch
    # Independent exact endpoint/zero-weight fixtures, not a relaxed tolerance.
    assert stencil([0.,0.],192.)=={(-1,-1),(-1,1),(1,-1),(1,1)}
    assert stencil([0.,0.],96.)==set(itertools.product([-1,0,1],repeat=2))
    assert hits([1.5,1.5,2.5,2.5],{(2,2),(3,3),(2,3)},[8,8],True)=={(2,2),(3,3),(2,3)}
    assert hits([1.5,1.5,2.5,2.5],{(2,2),(3,3)},[8,8],False)=={(2,2)}
    with (HERE/'ALL960_REPLAY_COUNTS.jsonl').open('x') as f:
        for x in residuals:f.write(json.dumps(x)+'\n')
    report=dict(completed=True,PASS=True,discrepancies=0,rows960=960,source_cue_views604=604,query_cue_join4832=4832,
        protocol=bind(V/'INVENTORY_PROTOCOL_BEFORE_QUERY_ANCHOR_LABELS.json'),publisher_code=bind(V/'build_metadata_inventory.py'),ledger=bind(V/'ALL960_QUERY_ANCHORED_AUX_INVENTORY_GT_ONLY.jsonl'),final_inventory_seal=bind(V/'FINAL_METADATA_INVENTORY_SEAL.json'),independent_code=bind(__file__),scope=bind(HERE/'SCOPE_BEFORE_REPLAY.json'),
        labels_D0_E1_Uminus1={str(k):v for k,v in labels.items()},by_source_raw_split=bygroup,original_label_transitions=transitions,
        joins_and_invariants='All source604 and query4832 Cartesian same-frame joins, original q/L/image/order/raw split/candidate lineage and labels preserved. All960 footprint/group/reason/output fields independently exact.',
        boundary_policy='FP64 positive supports; exact integer endpoint conventions; no epsilon, tolerances or coordinate adjustments. Zero-weight neighbors excluded. Fixed analytic fixtures PASS.',
        unknown_touch_rows=sorted({a for a,b in unknown_touch}),partial_touch_rows=[],
        scientific_limits=['Uncertainty ranges are conditional AI localization/association evidence, not target-owned material masks.','Pallox is one tall physical bin; LOCO supports flat pallets but has only six retained root D candidates.','Unique certified spatial group is an auxiliary cue binding, not native expected-corner identity.','No annotation/model accuracy is established by this metadata parity.','All native expected-part states remain UNKNOWN; human gold and training admission remain false.'],
        pixels_features_weights_models_fit_current_PnP=0)
    write(HERE/'REVIEW.json',report)
    write(HERE/'SEAL.json',dict(completed=True,PASS=True,outputs=[bind(p) for p in sorted(HERE.iterdir()) if p.is_file()]))
    print(json.dumps(dict(PASS=True,labels=labels,by_source_raw_split=bygroup,report=bind(HERE/'REVIEW.json'))))
if __name__=='__main__':main()
