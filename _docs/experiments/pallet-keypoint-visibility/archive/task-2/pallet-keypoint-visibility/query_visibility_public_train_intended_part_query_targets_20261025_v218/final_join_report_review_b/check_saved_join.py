#!/usr/bin/env python3
"""Independent saved-dictionary join/RLE/condition QA, with no image loader."""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct

H=Path(__file__).absolute().parent
V=H.parent
D='D_VISIBLE_EXPECTED_PART_WEAK'
E='E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK'
STATES=[D,E,'SELF_OR_OWN_STRUCTURE_UNOBSERVABLE','WRONG_PART_OR_GEOMETRY_UNRESOLVED','UNKNOWN','OUT_OF_FRAME']


def rb(path):
    p=Path(path)
    assert p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).casefold()
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    assert stat.S_ISREG(os.fstat(n).st_mode)
    with os.fdopen(n,'rb') as f:return f.read()


def sha(b):return hashlib.sha256(b).hexdigest()
def bind(p):return {'path':str(p),'sha256':sha(rb(p))}
def identity(b):return b['path'],b['sha256']


def checked(b):
    assert {'path','sha256'}<=set(b)<= {'path','sha256','size_bytes'}
    raw=rb(b['path']);assert sha(raw)==b['sha256']
    if 'size_bytes' in b:assert type(b['size_bytes']) is int and len(raw)==b['size_bytes']
    return raw


def put(p,value):
    assert H in p.parents
    with os.fdopen(os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644),'w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def rows(b):return [json.loads(line) for line in checked(b).decode().splitlines()]


def bounded_box(box):
    return isinstance(box,list) and len(box)==4 and all(type(x) is int for x in box) and 0<=box[0]<=box[2]<96 and 0<=box[1]<=box[3]<96


def valid_cells(runs):
    cells=set();previous=0
    for start,length in runs:
        assert type(start) is int and type(length) is int and length>0 and previous<=start<start+length<=9216
        cells.update(range(start,start+length));previous=start+length
    return cells


def nominal_expected(a,c):
    agree=a['state']==c['state'];state=a['state'] if agree else 'UNKNOWN'
    boxes={'A':a['optional_roi96'],'C':c['optionalroi96']}
    reasons=[] if agree else ['OBSERVER_STATE_DISAGREEMENT']
    weak=state in (D,E)
    if weak:
        for who,o in [('A',a),('C',c)]:
            if not o.get('part_identity_evidence') or not bounded_box(boxes[who]):
                reasons.append(who+'_PART_EVIDENCE_OR_ROI_MISSING')
            if state==E and (not o.get('foreign_evidence') or not o.get('occluder')):
                reasons.append(who+'_FOREIGN_EVIDENCE_MISSING')
        weak=not reasons
    return state,agree,weak,reasons,boxes


def transfer_expected(nominal,row):
    state,agree,weak,reasons,boxes=nominal
    dx,dy=(round(96*float(x)) for x in row['normalized_offset'])
    assert [dx,dy] in [[0,0],[24,0],[0,24]]
    cells=valid_cells(row['valid96_RLE'])
    tx,ty=47.5-dx,47.5-dy
    anchor_cells={y*96+x for y in (math.floor(ty),math.ceil(ty)) for x in (math.floor(tx),math.ceil(tx))}
    check={'query_available':row['generated_query_available'],'nominal_target_in_frame':row['nominal_target_in_frame'],
           'nominal_target_in_crop96':0<=tx<=95 and 0<=ty<=95,
           'nominal_target_strict_valid':anchor_cells<=cells,'evidence_boxes':{}}
    why=list(reasons)
    if weak:
        for who,box in boxes.items():
            moved=[box[0]-dx,box[1]-dy,box[2]-dx,box[3]-dy]
            inside=bounded_box(moved)
            selected={y*96+x for y in range(moved[1],moved[3]+1) for x in range(moved[0],moved[2]+1)} if inside else set()
            check['evidence_boxes'][who]={'original_roi96':box,'transformed_roi96':moved,'fully_inside':inside,
                                         'all_selected_pixels_strict_valid':inside and selected<=cells}
        if not all(check[k] for k in ('query_available','nominal_target_in_frame','nominal_target_in_crop96','nominal_target_strict_valid')):
            why.append('QUERY_OR_TARGET_SUPPORT_UNAVAILABLE')
        if not all(x['fully_inside'] and x['all_selected_pixels_strict_valid'] for x in check['evidence_boxes'].values()):
            why.append('EVIDENCE_BOX_LOST_OR_INVALID_AFTER_FIXED_SHIFT')
        weak=not why
    target=state if state not in (D,E) or weak else 'UNKNOWN'
    return {'target_state':target,'support_checks':check,'failure_reasons':why,
            'weak_loss_candidate':weak,'weak_loss_fit_candidate':weak and row['source_role']=='SOURCE_FIT',
            'weak_target_E_binary':int(target==E) if weak else None}


def condition_vector(row,group):
    k=row['native_slot'];one=[float(j==k) for j in range(8)];mask=[0.]*8;delta=[0.]*16
    if row['generated_query_available']:
        for j in range(8):
            if j!=k and group[j]['generated_query_available']:
                mask[j]=1.
                for axis in range(2):
                    value=(float(group[j]['generated_query_xy'][axis])-float(row['generated_query_xy'][axis]))/float(row['L'])
                    delta[2*j+axis]=struct.unpack('<f',struct.pack('<f',min(8.,max(-8.,value))))[0]
    return one+mask+delta


def selftest():
    a={'state':D,'part_identity_evidence':'invented part','foreign_evidence':'','occluder':'','optional_roi96':[30,30,40,40]}
    c={'state':D,'part_identity_evidence':'invented part','foreign_evidence':'','occluder':'','optionalroi96':[32,30,41,40]}
    n=nominal_expected(a,c);assert n[0]==D and n[2]
    cc=dict(c,state=E);assert nominal_expected(a,cc)[0]=='UNKNOWN'
    assert not nominal_expected(dict(a,part_identity_evidence=''),c)[2]
    row={'normalized_offset':[.25,0.],'valid96_RLE':[[0,9216]],'generated_query_available':True,'nominal_target_in_frame':True,'source_role':'SOURCE_DEV'}
    out=transfer_expected(n,row);assert out['target_state']==D and out['weak_loss_candidate'] and not out['weak_loss_fit_candidate']
    assert out['support_checks']['evidence_boxes']['A']['transformed_roi96']==[6,30,16,40]
    assert out['support_checks']['evidence_boxes']['A']['original_roi96']==a['optional_roi96']
    nn=nominal_expected(dict(a,optional_roi96=[0,30,20,40]),c)
    assert transfer_expected(nn,row)['target_state']=='UNKNOWN'
    hole=23*96+23;cells=sorted(set(range(9216))-{hole})
    row2=dict(row,normalized_offset=[.25,.0],valid96_RLE=[[0,hole],[hole+1,9216-hole-1]])
    # Remove the actual nominal target tap (x23,y47), independently of boxes.
    hole=47*96+23;row2['valid96_RLE']=[[0,hole],[hole+1,9216-hole-1]]
    assert not transfer_expected(n,row2)['support_checks']['nominal_target_strict_valid']
    rr={'native_slot':2,'generated_query_available':False,'L':2.,'generated_query_xy':[0.,0.]}
    assert condition_vector(rr,{j:dict(rr) for j in range(8)})==[0.,0.,1.,0.,0.,0.,0.,0.]+[0.]*24
    return {'completed':True,'PASS':True,'pure_groups':5,'checks':['same/different six-state/evidence rules','fixed ROI −24 translation without clipping','DEV fit false with evidence candidate retained','strict target-support hole rejects transfer','unavailable generated q condition is native-only'],'actual_observation_or_image_reads':0}


def audit(entry_path):
    e=json.loads(rb(entry_path));assert e['approved_saved_join_QA'] is True and e['code']==bind(Path(__file__))
    complete=json.loads(checked(e['join_COMPLETE']));assert complete['completed'] and complete['PASS']
    assert identity(complete['policy'])==identity(e['policy'])
    checked(e['policy'])
    # Both completed seal hashes close before loading either observation body.
    seals={who:json.loads(checked(b)) for who,b in complete['original_observer_completes'].items()}
    assert all(s['completed'] and s['PASS'] for s in seals.values())
    readers={}
    for who,s in seals.items():
        qa=json.loads(checked(s['schema_QA']));ind=json.loads(checked(s['independence']));view=json.loads(checked(s['view_receipt']))
        assert qa['completed'] and qa['PASS'] and ind['completed'] and ind['PASS'] and view['completed'] and view['PASS']
        assert identity(ind['own_observations_binding'])==identity(s['observations'])
        for key in (['other_reader_new_observation_files_read','other_reader_new_values_received','root_new_observation_values_read'] if who=='a' else ['A_new_observation_body_reads','root_new_observation_body_reads']):assert ind[key]==0
        data=rows(s['observations']) if who=='a' else json.loads(checked(s['observations']))['rows120']
        assert len(data)==120
        byid={r['row_id']:r for r in data};assert len(byid)==120
        viewed={identity(v['binding']) for v in view['views'] if v['actually_viewed'] is True}
        assert len(viewed)==15 and len(view['views'])==15
        assert all(identity(r['actualview_image_binding']) in viewed for r in data)
        readers[who]=byid
    source=rows(complete['original360']);assert len(source)==360
    generated={r['row_id']:r for r in source};assert len(generated)==360
    nominal=[r for r in source if r['normalized_offset']==[0.,0.]];assert len(nominal)==120
    assert set(readers['a'])==set(readers['c'])=={r['row_id'] for r in nominal}
    originals={r['original_input']['row_id']:r['original_input'] for r in source}
    assert len(originals)==96 and sum(not r['original_query_available'] for r in originals.values())==11
    actual120=rows(complete['outputs']['ALL120_JOINED_NOMINAL_OBSERVATIONS.jsonl'])
    actual360=rows(complete['outputs']['ALL360_CHECKED_WEAK_TARGET_CANDIDATES.jsonl'])
    expected_nominal={};matrix=[[0]*6 for _ in range(6)]
    for m,j in zip(nominal,actual120):
        uid=m['row_id'];a,c=readers['a'][uid],readers['c'][uid]
        assert j['nominal_row_id']==uid and j['A']==a and j['C']==c
        for key in ['family','frame_position','candidate_gt_list_index','native_slot','canonical_index','anatomical_family','component_id','generated_query_xy','L','generated_query_available','nominal_target_in_frame']:
            assert a[key]==c[key]==m[key],(uid,key)
        assert a['source_role']==c['SOURCE_role']==m['source_role']==j['source_role']
        assert a['original_inputU_unchanged']==(not m['original_input']['original_query_available'])
        assert c['original_input']==m['original_input']==j['original_input']
        assert a['state'] in STATES and c['state'] in STATES
        assert (a['state']=='OUT_OF_FRAME')==(c['state']=='OUT_OF_FRAME')==(not m['nominal_target_in_frame'])
        n=nominal_expected(a,c);expected_nominal[uid]=n
        for key,value in zip(['consensus_state','consensus_agreement','nominal_weak_eligible','failure_reasons','evidence_roi96_by_observer'],n):assert j[key]==value,(uid,key)
        assert j['training_admission'] is False
        matrix[STATES.index(a['state'])][STATES.index(c['state'])]+=1
    groups=collections.defaultdict(dict)
    for r in source:groups[r['group_id']][r['native_slot']]=r
    assert len(groups)==45 and all(set(g)==set(range(8)) for g in groups.values())
    assert len(actual120)==120 and len(actual360)==360
    for src,t in zip(source,actual360):
        uid=src['row_id'];base=uid.replace('_S1_','_S0_').replace('_S2_','_S0_')
        assert t['row_id']==uid and t['nominal_row_id']==base and t['generated_input']==src
        assert t['source_role']==src['source_role'] and t['nominal_consensus_state']==expected_nominal[base][0]
        expect=transfer_expected(expected_nominal[base],src)
        assert all(t[k]==v for k,v in expect.items()),uid
        assert t['deployable_condition32']==condition_vector(src,groups[src['group_id']]),uid
        assert t['training_admission'] is False and t['primary_physical_native_gold'] is False
        assert src['source_role']!='SOURCE_DEV' or t['weak_loss_fit_candidate'] is False
    saved_summary=json.loads(checked(complete['outputs']['SUMMARY.json']))
    expected_counts={'A_state_counts':dict(collections.Counter(r['state'] for r in readers['a'].values())),
        'C_state_counts':dict(collections.Counter(r['state'] for r in readers['c'].values())),
        'A_rows_C_columns_confusion':matrix,'agreements':sum(r['consensus_agreement'] for r in actual120),
        'disagreements':sum(not r['consensus_agreement'] for r in actual120),
        'in_frame_nominal_rows':sum(r['nominal_target_in_frame'] for r in nominal),
        'in_frame_agreements':sum(r['consensus_agreement'] for r in actual120 if r['consensus_state']!='OUT_OF_FRAME'),
        'consensus_state_counts':dict(collections.Counter(r['consensus_state'] for r in actual120)),
        'nominal_weak_eligible':sum(r['nominal_weak_eligible'] for r in actual120),
        'generated_state_counts':dict(collections.Counter(r['target_state'] for r in actual360)),
        'support_checked_weak_candidates':sum(r['weak_loss_candidate'] for r in actual360),
        'FIT_weak_loss_candidates':sum(r['weak_loss_fit_candidate'] for r in actual360),
        'DEV_weak_evaluation_candidates':sum(r['weak_loss_candidate'] and r['source_role']=='SOURCE_DEV' for r in actual360),
        'by_source_role':{role:dict(collections.Counter(r['target_state'] for r in actual360 if r['source_role']==role)) for role in ['SOURCE_FIT','SOURCE_DEV']},
        'weak_candidates_by_state':{state:dict(collections.Counter(r['target_state'] for r in actual360 if r['weak_loss_candidate'] and r['generated_input']['normalized_offset']==offset)) for state,offset in [('S0',[0.,0.]),('S1',[.25,0.]),('S2',[0.,.25])]},
        'consensus_D_E_native_counts':dict(collections.Counter(str(r['native_slot']) for r in actual120 if r['nominal_weak_eligible']))}
    assert all(saved_summary[k]==v for k,v in expected_counts.items())
    census=list(csv.DictReader(io.StringIO(checked(complete['outputs']['ALL360_TARGET_CENSUS.csv']).decode())))
    assert len(census)==360
    for c,r in zip(census,actual360):
        expected={'row_id':r['row_id'],'nominal_row_id':r['nominal_row_id'],'source_role':r['source_role'],
            'native_slot':str(r['generated_input']['native_slot']),'anatomical_family':r['generated_input']['anatomical_family'],
            'nominal_state':r['nominal_consensus_state'],'target_state':r['target_state'],
            'weak_loss_candidate':str(r['weak_loss_candidate']),'weak_loss_fit_candidate':str(r['weak_loss_fit_candidate']),
            'failure_reasons':'|'.join(r['failure_reasons'])}
        assert c==expected
    for s in ['training_admission','fully_solved']:assert saved_summary[s] is False
    assert saved_summary['primary_native_gold']==0 and saved_summary['training_model_protected_eval_or_PnP']==0
    out=H/'actual_join_v1';os.mkdir(out)
    result={'completed':True,'PASS':True,'code':bind(Path(__file__)),'protocol':e['protocol'],'entry':bind(entry_path),
        'join_COMPLETE':e['join_COMPLETE'],'original_observer_completes':complete['original_observer_completes'],
        'summary':expected_counts,'joined_nominal_rows':120,'jitter_rows':360,'condition32_rows_recomputed':360,
        'condition32_finite_scalars':11520,'original_unique96_U11_preserved':True,
        'role_counts':dict(collections.Counter(r['source_role'] for r in source)),
        'weak_support_by_role':{role:dict(collections.Counter(r['target_state'] for r in actual360 if r['weak_loss_candidate'] and r['source_role']==role)) for role in ['SOURCE_FIT','SOURCE_DEV']},
        'pixel_image_GT_array_model_protected_reads':0,'producer_helpers_imported':False,'training_admission':False,
        'authority_limit':'Assistant SOURCE intendedpart weak consensus, not primary native physical gold, human/blind or deployment accuracy. New family q itselfGT-derived; no additional GT pose/ref/ROI/mask features.',
        'report_comparison':'Pending exact root report build completion; numeric QA not rerun for text-only final step.'}
    put(out/'REVIEW.json',result)
    print(json.dumps({'REVIEW':bind(out/'REVIEW.json'),'counts':expected_counts,'weak_support_by_role':result['weak_support_by_role']}))


if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--selftest',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.selftest:
        result=selftest();put(H/'INVENTED_JOIN_CHECKS.json',result);print(json.dumps(result))
    else:audit(a.entry)
