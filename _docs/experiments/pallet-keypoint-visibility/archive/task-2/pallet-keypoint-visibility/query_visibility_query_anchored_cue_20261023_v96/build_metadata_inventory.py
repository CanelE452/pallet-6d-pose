"""One metadata-only inventory. No image/feature/model/score reader."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,math
import numpy as np

HERE=Path(__file__).resolve().parent
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def load(b):
    assert bind(b['path'])['sha256']==b['sha256']
    return [json.loads(x)for x in Path(b['path']).read_text().splitlines()if x]
def cells(q,L):
    out=set()
    for sy in [-1,1]:
        for sx in [-1,1]:
            x=q[0]+sx*L/192;y=q[1]+sy*L/192;i=math.floor(x);j=math.floor(y);dx=x-i;dy=y-j
            for ox,oy,w in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
                if w>0:out.add((i+ox,j+oy))
    return sorted(out)
def contains(box,xy,hw,closed):
    if box is None or len(box)!=4 or not np.isfinite(box).all():return None
    x0,y0,x1,y1=box;h,w=hw
    if x0>x1 or y0>y1 or (not closed and(x0==x1 or y0==y1)):return None
    if closed:
        if x0<0 or y0<0 or x1>w-1 or y1>h-1:return None
        lo=[math.floor(x0+.5),math.floor(y0+.5)];hi=[math.floor(x1+.5)+1,math.floor(y1+.5)+1]
    else:
        if x0<-.5 or y0<-.5 or x1>w-.5 or y1>h-.5:return None
        lo=[math.floor(x0+.5),math.floor(y0+.5)];hi=[math.ceil(x1+.5),math.ceil(y1+.5)]
    return {p for p in xy if lo[0]<=p[0]<hi[0]and lo[1]<=p[1]<hi[1]}

def main():
    proto=json.loads((HERE/'INVENTORY_PROTOCOL_BEFORE_QUERY_ANCHOR_LABELS.json').read_text());d={k:load(b)for k,b in proto['inputs'].items()}
    assert len(d['rows960'])==len(d['original_labels960'])==960 and len(d['loco_root160'])==160 and len(d['pallox800'])==800 and len(d['all_cue_cross4832'])==4832
    teachers={x['cue_view_index']:x for x in d['original_cue_views604']};cross=defaultdict(list)
    for x in d['all_cue_cross4832']:
        t=teachers[x['cue_view_index']]
        assert x['cue_id']==t['cue_id']and x['teacher_label']==t['teacher_label']
        assert x['cue_reference_xy_unchanged']==t['reference_xy_unchanged'] and x['cue_uncertainty_bbox_unchanged']==t['uncertainty_bbox_xyxy_half_open_unchanged']
        assert x['certified_duplicate_group']==t['duplicate_group_link_only'];cross[x['query_id']].append(x)
    out=[]
    for i,(r,g)in enumerate(zip(d['rows960'],d['original_labels960'])):
        assert r['slot_index']==g['base_row_index']==i and r['query_xy']==g['original_q']and r['L']==g['original_L']
        q=r['query_xy'];L=r['L'];hw=r['raw_hw'];flags=[];label=-1;matched=None;groups={};unlocated=[];unusable=[]
        footprint=cells(q,L)if q is not None and L is not None and np.isfinite(q).all()and np.isfinite(L)and L>0 else []
        if not footprint:flags.append('MISSING_OR_NONFINITE_QUERY_U')
        if footprint and not r['original_input_valid']:flags.append('ORIGINAL_QUERY_INPUT_UNAVAILABLE_U')
        if any(x<0 or y<0 or x>=hw[1]or y>=hw[0]for x,y in footprint):flags.append('CENTRAL_NATIVE_BILINEAR_STENCIL_OOB_U')
        candidate_groups=[];touching=[];views=[]
        if i<160:
            root=d['loco_root160'][i];assert root['query_key']==[r['input_id'],r['native_slot_logical_only']]
            if root['before_aux_state']=='D'and footprint:
                memberships=[contains(root[k],footprint,hw,False)for k in ['support_bbox_xyxy','uncertainty_bbox_xyxy']]
                if any(m is None for m in memberships):flags.append('LOCO_UNUSABLE_CERTIFIED_GEOMETRY_U')
                else:
                    support=set().union(*memberships)
                    if len(support)==len(footprint):candidate_groups=['LOCO_SAME_QUERY_ROOT_SINGLE_CUE'];matched=candidate_groups[0]
                    else:flags.append('CENTRAL_FOOTPRINT_NOT_FULLY_IN_ROOT_CUE_U')
            else:flags.append('LOCO_SAME_QUERY_ROOT_NOT_KNOWN_D_U')
            if not flags and matched is not None:label=0
        else:
            old=d['pallox800'][i-160];assert old['original_q']==q and old['original_L']==L and old['scalar_target_D0_E1_Uminus1']==g['original_natural_label_GT_only']
            xs=cross[old['query_id']];assert [x['cue_view_index']for x in xs]==old['source_GT_cue_view_indices']
            views=[x['cue_view_index']for x in xs]
            retained={x['cue_view_index']for x in old['retained_FULL_GT_cues']}
            assert retained=={x['cue_view_index']for x in xs if x['coverage']=='FULL'and x['teacher_label']in[0,1]}
            for x in xs:
                key=x['certified_duplicate_group'];grp=groups.setdefault(key,dict(full_known=set(),touch=set(),labels=set(),unknown=False,views=[],coverage=[]))
                grp['views'].append(x['cue_view_index']);grp['coverage'].append(x['coverage'])
                if x['teacher_label']in[0,1]:grp['labels'].add(x['teacher_label'])
                else:grp['unknown']=True
                box=x['cue_uncertainty_bbox_unchanged']
                if x['reference_null']or box is None:unlocated.append(key);continue
                hit=contains(box,footprint,hw,True)
                if hit is None:unusable.append(key);continue
                grp['touch']|=hit
                if x['cue_view_index']in retained:grp['full_known']|=hit
            unlocated=sorted(set(unlocated));unusable=sorted(set(unusable))
            touching=[k for k,v in groups.items()if v['touch']]
            candidate_groups=[k for k,v in groups.items()if footprint and len(v['full_known'])==len(footprint)]
            if not candidate_groups:flags.append('NO_KNOWN_FULL_GROUP_CONTAINS_ENTIRE_CENTRAL_STENCIL_U')
            elif len(candidate_groups)>1:flags.append('MULTIPLE_DISTINCT_FULL_CANDIDATE_GROUPS_U')
            else:matched=candidate_groups[0]
            if unlocated:flags.append('UNLOCATED_POTENTIAL_CUE_COMPETITION_U')
            if unusable:flags.append('UNUSABLE_CUE_GEOMETRY_POTENTIAL_COMPETITION_U')
            if matched is not None:
                if set(touching)!={matched}:flags.append('OTHER_DISTINCT_LOCATED_GROUP_TOUCHES_CENTRAL_STENCIL_U')
                grp=groups[matched]
                if grp['unknown']or len(grp['labels'])!=1:flags.append('MATCHED_GROUP_UNKNOWN_OR_CONTRADICTORY_STATE_U')
                if not flags:label=next(iter(grp['labels']))
        if label>=0:flags=['QUALIFIED_QUERY_ANCHORED_D_AUX'if label==0 else'QUALIFIED_QUERY_ANCHORED_E_AUX']
        out.append(dict(row_index=i,input_id=r['input_id'],native_slot_logical_only=r['native_slot_logical_only'],source=g['source'],raw_split=r['split'],original_q=q,original_L=L,
            image_sha256=r['image_sha256'],raw_hw=hw,original_input_valid=r['original_input_valid'],original_C1_V95_label_D0_E1_Uminus1=g['original_natural_label_GT_only'],
            query_anchored_AUX_label_D0_E1_Uminus1=label,primary_reason=flags[0],all_reasons=flags,central4_positive_bilinear_native_cells=[list(p)for p in footprint],
            candidate_certified_groups=candidate_groups,matched_certified_group_GT_only=matched if label>=0 else None,all_touching_groups=touching,
            unlocated_potential_competing_groups=unlocated,unusable_geometry_groups=unusable,source_cue_view_indices=views,
            group_evidence_GT_only={k:dict(views=v['views'],original_coverage=v['coverage'],known_states=sorted(v['labels']),has_unknown=v['unknown'],central_cells_touched=len(v['touch']),central_cells_in_FULL_known_union=len(v['full_known']))for k,v in groups.items()},
            native_expected_part_state='UNKNOWN',human_gold=False,training_admission=False,GT_fields_may_forward=False))
    assert len(out)==960 and Counter(r['original_C1_V95_label_D0_E1_Uminus1']for r in out)=={0:143,1:20,-1:797}
    ledger=HERE/'ALL960_QUERY_ANCHORED_AUX_INVENTORY_GT_ONLY.jsonl'
    with ledger.open('x')as f:
        for r in out:f.write(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n')
    summary=dict(completed=True,PASS=True,inventory_only=True,rows=960,protocol=bind(HERE/'INVENTORY_PROTOCOL_BEFORE_QUERY_ANCHOR_LABELS.json'),code=bind(__file__),ledger=bind(ledger),
        labels={str(k):sum(r['query_anchored_AUX_label_D0_E1_Uminus1']==k for r in out)for k in [0,1,-1]},
        by_source_raw_split={s+'|'+sp:{str(k):sum(r['source']==s and r['raw_split']==sp and r['query_anchored_AUX_label_D0_E1_Uminus1']==k for r in out)for k in [0,1,-1]}for s,sp in sorted({(r['source'],r['raw_split'])for r in out})},
        primary_reason_counts=dict(Counter(r['primary_reason']for r in out)),all_reason_counts=dict(Counter(reason for r in out for reason in r['all_reasons'])),
        old_to_new_label_counts={str(a)+'->'+str(b):sum(r['original_C1_V95_label_D0_E1_Uminus1']==a and r['query_anchored_AUX_label_D0_E1_Uminus1']==b for r in out)for a in[0,1,-1]for b in[0,1,-1]},
        q_L_fold_order_original_labels_unchanged=True,no_query_borrowing_LOCO=True,partial_unknown_unlocated_competition_preserved=True,
        native_expected_part_state='UNKNOWN',human_gold=False,training_admission=False,source_pixels_features_weights_models_fit_current_PnP=0)
    with (HERE/'QUERY_ANCHORED_INVENTORY_SUMMARY.json').open('x')as f:json.dump(summary,f,indent=2);f.write('\n')
    with (HERE/'INVENTORY_SEAL.json').open('x')as f:json.dump(dict(completed=True,PASS=True,summary=bind(HERE/'QUERY_ANCHORED_INVENTORY_SUMMARY.json'),ledger=bind(ledger),code=bind(__file__),protocol=bind(HERE/'INVENTORY_PROTOCOL_BEFORE_QUERY_ANCHOR_LABELS.json'),training_admission=False),f,indent=2);f.write('\n')
    print(json.dumps(summary))

if __name__=='__main__':main()
