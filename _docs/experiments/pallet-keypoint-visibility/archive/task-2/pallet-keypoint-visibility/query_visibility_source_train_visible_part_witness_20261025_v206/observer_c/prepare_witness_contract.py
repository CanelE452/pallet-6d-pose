"""Metadata-only C witness preparation; rectangle checks do not establish semantics."""
from pathlib import Path
import csv, hashlib, json, os, stat

H = Path(__file__).absolute().parent
V = H.parent
SCOPE = {'path':str(V/'ROOT_PROSPECTIVE_SOURCE_TRAIN_VISIBLE_PART_WITNESS_SCOPE.json'),'sha256':'2d90d7c29994cb02bb0c24fd2556e97493f0dfee2b7b84ffadefeac24cc6c4b4'}
SELECTION = {'path':str(V/'SOURCE8_WITNESS_CASES_BEFORE_NEW_ANNOTATION.json'),'sha256':'7600afe3cd0039ec2055f9cd094b99b6b4a73344c28d41baf0c457a53843a297'}

def read_bound(b):
    p=Path(b['path'])
    assert p.is_absolute() and V in p.parents and '..'not in p.parts
    assert all('v112'not in s.casefold()for s in p.parts) and p.suffix=='.json'
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for s in p.parts[1:-1]:
            n=os.open(s,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    with os.fdopen(n,'rb')as f:
        assert stat.S_ISREG(os.fstat(f.fileno()).st_mode)
        data=f.read()
    assert hashlib.sha256(data).hexdigest()==b['sha256']
    return json.loads(data)

def validate_witness(row, final=False):
    state=row['visible_intended_part_identified_YES_UNKNOWN']
    assert state in ({'YES','UNKNOWN'} if final else {'YES','UNKNOWN','UNREVIEWED'})
    box=row['witness_ROI_raw96_xyxy_or_null']
    if state!='YES':
        assert box is None
    else:
        assert isinstance(box,list) and len(box)==4
        assert all(type(v)is int and 0<=v<=95 for v in box)
        x0,y0,x1,y1=box;assert x0<=x1 and y0<=y1
        assert row['visible_feature_description'].strip() and row['positive_local_visual_evidence'].strip()
    if final:
        assert row['actually_viewed_image_bindings']
        if state=='UNKNOWN':assert row['ambiguity_note'].strip()

def put(p,x):
    with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def main():
    scope=read_bound(SCOPE);selection=read_bound(SELECTION)
    assert scope['fixed_case_selection']==SELECTION
    cases=selection['source_cases8'];assert len(cases)==8
    assert len({c['original_slot']['frame_key']for c in cases})==3
    rows=[]
    for c in cases:
        s=c['original_slot'];assert s['original_query_available']is True
        assert c['prior_relation']in {'ON','NEAR'}
        rows.append({
            'row_id':s['row_id'],'observer':'C','frame_key':s['frame_key'],'original_frame_index':s['original_frame_index'],
            'native_slot':s['native_slot'],'original_query_xy':s['query_xy'],'original_L':s['L'],'original_query_available':s['original_query_available'],
            'visible_intended_part_identified_YES_UNKNOWN':'UNREVIEWED','witness_ROI_raw96_xyxy_or_null':None,
            'visible_feature_description':'','positive_local_visual_evidence':'','ambiguity_note':'','actually_viewed_image_bindings':[],
            'coordinate_convention':'RAW96_INTEGER_INCLUSIVE_0_TO_95',
            'prior_relation_unchanged':c['prior_relation'],'prior_location_flags_unchanged':c['prior_location_flags'],
            'corrected_query_or_localization_output':False,'physical_native_gold':False,'training_admission':False})
    assert len({r['row_id']for r in rows})==8
    for r in rows:validate_witness(r)
    toy=dict(rows[0],visible_intended_part_identified_YES_UNKNOWN='YES',witness_ROI_raw96_xyxy_or_null=[0,0,95,95],visible_feature_description='invented corner',positive_local_visual_evidence='invented visible edge',actually_viewed_image_bindings=[{'invented':True}])
    validate_witness(toy,True)
    checks=['inclusive boundary [0,0,95,95] accepted']
    unknown=dict(toy,visible_intended_part_identified_YES_UNKNOWN='UNKNOWN',witness_ROI_raw96_xyxy_or_null=None,ambiguity_note='invented ambiguity')
    validate_witness(unknown,True);checks.append('UNKNOWN/null accepted')
    for name,change in [('negative',[-1,0,10,10]),('beyond95',[0,0,96,95]),('reversed',[20,0,10,10]),('fractional',[0,0,4.5,6]),('bool',[False,0,4,6]),('YES null',None)]:
        try:validate_witness(dict(toy,witness_ROI_raw96_xyxy_or_null=change),True)
        except AssertionError:checks.append(name+' rejected')
        else:raise AssertionError('Invalid invented ROI accepted')
    try:validate_witness(dict(unknown,witness_ROI_raw96_xyxy_or_null=[1,1,3,3]),True)
    except AssertionError:checks.append('UNKNOWN with ROI rejected')
    else:raise AssertionError('UNKNOWN ROI accepted')
    put(H/'EMPTY8_WITNESS_TEMPLATE.json',{'completed':False,'prepared_only':True,'observer':'C','scope':SCOPE,'selection':SELECTION,'rows':rows})
    flat=[{k:json.dumps(v,ensure_ascii=False)if isinstance(v,(list,dict))else v for k,v in r.items()}for r in rows]
    with (H/'EMPTY8_WITNESS_TEMPLATE.csv').open('x',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    put(H/'PROTOCOL_BEFORE_C_WITNESS_PIXEL_OBSERVATION.json',{
        'completed':True,'prepared_only':True,'observer':'C','scope':SCOPE,'selection':SELECTION,'code':bind(Path(__file__)),
        'fixed_cases8_frames3':True,'actual_annotation_states':['YES','UNKNOWN'],'preparation_state':'UNREVIEWED',
        'coordinate_definition':scope['coordinate_definition'],
        'semantic_evidence':'Observe a visible local target corner/end/block feature supporting D within the ORIGINAL CLEAN96, using bound full q8/frame context. Nearby wood alone is insufficient; UNKNOWN/null remains allowed.',
        'ROI_scope':'Axis-aligned inclusive rectangle in raw96 pixel indices. It is weak visible-feature evidence for possible future controlled masks; not corrected q, a keypoint, complete part segmentation, physical native/canonical gold or a localization output.',
        'q_and_location_axis':'Original q/L/native and q display at47.5 are unchanged. Seven prior locationflag disagreements are retained transparently, not used as a perfect-q prerequisite or ROI admissibility rule.',
        'independence':'No new observer A ROI values/files may be read until own annotations seal. Value-free field schema alone may be coordinated. V205 prior sealed SOURCE context is acknowledged; no blind human study claim.',
        'actual_gate':'Zero image views now. Actual selected PNG toolviews and ROI annotation require a separate root actual entry; no source array/raw RGB reopen, GT, old model scores, eval, new crop, mesh/PnP/initializer or fit.',
        'unknown_rule':scope['unknown_rule'],'training_admission':False,'V205_strict_join_changed':False,
        'metadata_bindings_followed':['scope','selection'],'parent_or_pixel_bindings_followed':0,'new_A_ROI_reads':0,'new_image_views':0})
    put(H/'INVENTED_ROI_VALIDATOR_CHECKS.json',{'completed':True,'PASS':True,'invented_only':True,'checks':checks,'original_rows8_UNREVIEWED_null':True,'actual_image_or_ROI_reads':0})
    put(H/'FINAL_READY.json',{'completed':True,'PASS':True,'prepared_only':True,'code':bind(Path(__file__)),'protocol':bind(H/'PROTOCOL_BEFORE_C_WITNESS_PIXEL_OBSERVATION.json'),
        'template':bind(H/'EMPTY8_WITNESS_TEMPLATE.json'),'CSV':bind(H/'EMPTY8_WITNESS_TEMPLATE.csv'),'checks':bind(H/'INVENTED_ROI_VALIDATOR_CHECKS.json'),
        'rows8':True,'frames3':True,'actual_image_views':0,'actual_ROIs':0,'new_A_ROI_reads':0,'training_admission':False})
    print(json.dumps(bind(H/'FINAL_READY.json')))

if __name__=='__main__':main()
