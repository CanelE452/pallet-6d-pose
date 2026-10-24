"""Small invented-only checks of the actual cache functions; source/body I/O=0."""
import copy
import numpy as np
import build_expanded_direct_control_cache as m
c=m.c

def expect_fail(fn):
    try:fn()
    except (AssertionError,KeyError,ValueError):return
    raise AssertionError('Expected fail-closed rejection')

def fixture():
    def binding(name):return dict(path=str(m.H/(name+'.toy')),sha256='0'*64)
    frames=[]
    for i in range(21):
        frames.append(dict(frame_index=100+i,split='SOURCE_DEVELOPMENT',raw_hw=[16,18],global_index=[800+8*i+k for k in range(8)],query_id=['real%d:%d'%(i,k) for k in range(8)],query_xy=[[float(k),float(k+1)] for k in range(8)],L=[12.0]*8,native_continuous_valid=[True]*7+[False],original_query_input_metadata=[dict(native_continuous_valid=k<7,nearest_valid=k<7,original_note='kept') for k in range(8)]))
    syn=[]
    for i in range(1024):
        syn.append(dict(attempt_id='S%04d'%i,base_index=i//8,native_slot=i%8,split='TRAIN',status='UNKNOWN_RETAINED',direct_end_states=[-1,-1,-1],outputs=None,query_xy=[float(i%8),5.0],L=12.0,original_global_index=i,original_occurrence_id='synthetic%d'%i,original_input_flags=dict(query_available=True,constructed=True)))
    syn[0].update(status='SUPPORTED',direct_end_states=[0,1,0],outputs=binding('synthetic_triplet'))
    attempts=[];real=[]
    for i in range(66):
        attempts.append(dict(pair_id='R%02d'%i,frame_index=frames[i%21]['frame_index'],case_id='Q%02d'%(i//3),donor_id='YCB:%02d'%i,native_slot=0))
        real.append(dict(pair_id='R%02d'%i,case_id='Q%02d'%(i//3),donor_id='YCB:%02d'%i,original_input=dict(native_slot=0),status='UNKNOWN_RETAINED',PASS=False,variants={},direct_end_weak_states=dict(BEFORE='UNKNOWN',COVER='UNKNOWN',SHAM='UNKNOWN')))
    real[0].update(PASS=True,status='SUPPORTED',variants={v:binding('real_'+v) for v in ('BEFORE','COVER','ADJACENT_SHAM')},direct_end_weak_states=dict(BEFORE='VISIBLE_WEAK',COVER='EXTERNALLY_BLOCKED_WEAK',SHAM='VISIBLE_WEAK'),closed_end_contract=dict(status='CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED'),closed14_actual_proof={k:True for k in m.PROOF14})
    real[1].update(status='GEOMETRY_FAILURE_RETAINED',variants=dict(BEFORE=binding('failure_before')))
    records={};sources=[]
    for domain,rows,key in [('SYNTH_FINITE_END',syn,'attempts'),('REAL_SEMANTIC_END_WEAK',real,'manifest')]:
        src=dict(domain=domain,complete=binding(domain+'_done'),independent_review=binding(domain+'_review'),census=binding(domain+'_rows'),expected_attempts=len(rows))
        sources.append(src);records[src['complete']['path']]=dict(completed=True,PASS=True,**{key:src['census']});records[src['independent_review']['path']]=dict(completed=True,PASS=True);records[src['census']['path']]=rows
    framebinding=binding('frames');records[framebinding['path']]=dict(frames=frames,attempts=attempts)
    groups=[]
    for domain,rows in [('SYNTH_FINITE_END',syn),('REAL_SEMANTIC_END_WEAK',real)]:
        for r in rows:
            aid=r.get('attempt_id',r.get('pair_id'));supported=r.get('direct_end_states')==[0,1,0] if domain=='SYNTH_FINITE_END' else r['PASS']
            for vi,var in enumerate(m.VARIANTS):
                packet=r['outputs'] if domain=='SYNTH_FINITE_END' else r['variants'].get('ADJACENT_SHAM' if var=='SHAM' else var)
                if packet is None:continue
                gid=('SYNTH__' if domain=='SYNTH_FINITE_END' else 'REAL__')+aid+'__'+var;mm=[]
                for k in range(8):
                    if domain=='SYNTH_FINITE_END':
                        origin=syn[8*r['base_index']+k];d={key:origin[key] for key in ('original_global_index','original_occurrence_id','query_xy','L','original_input_flags')};available=origin['original_input_flags']['query_available']
                    else:
                        frame=frames[int(aid[1:])%21];d=dict(original_global_index=frame['global_index'][k],original_occurrence_id=frame['query_id'][k],query_xy=frame['query_xy'][k],L=frame['L'][k],original_input_flags=frame['original_query_input_metadata'][k]);available=frame['native_continuous_valid'][k]
                    mm.append(dict(d,row_id=gid+'/native%d'%k,native_slot=k,original_query_available=available))
                groups.append(dict(domain=domain,attempt_id=aid,group_id=gid,variant=var,source_packet=packet,layout='SYNTH_TRIPLET' if domain=='SYNTH_FINITE_END' else 'REAL_SINGLE',variant_index=vi,selected_native_slot=0,source_status=r['status'],admitted=supported,admitted_target=(1 if var=='COVER' else 0) if supported else -1,authority='CONSTRUCTIVE_SOURCE_FIXED_NEAR_ANCHOR_END' if domain=='SYNTH_FINITE_END' else 'SOURCE_RGB_SEMANTIC_END_WEAK',metadata8=mm,raw_hw=[16,18]))
    admission=dict(completed=True,PASS=True,source_only=True,training_admission_direct_end=True,sources=sources,groups=groups,original_real_frame_plan=framebinding)
    return admission,records

def population(admission,records):
    old=(c.read_bound,c.lines)
    try:
        c.read_bound=lambda b:copy.deepcopy(records[b['path']]);c.lines=lambda b:copy.deepcopy(records[b['path']])
        return m.source_population(copy.deepcopy(admission))
    finally:c.read_bound,c.lines=old

def checks():
    results=[]
    def record(name,fn):fn();results.append(dict(name=name,PASS=True))
    a,r=fixture()
    def full_population():
        groups,census,_=population(a,r)
        assert len(census)==3270 and len(groups)==7 and all(len(g['metadata8'])==8 for g in groups)
        assert sum(g['admitted'] for g in groups)==6
        assert all(g['source_proof_binding'] for g in groups)
        assert {g['source_frame_index'] for g in groups if g['domain']=='REAL_SEMANTIC_END_WEAK'}=={100,101}
        assert all(g['variant']!='ADJACENT_SHAM' and (g['source_variant']=='ADJACENT_SHAM')==(g['domain']=='REAL_SEMANTIC_END_WEAK' and g['variant']=='SHAM') for g in groups)
        failure=[g for g in groups if g['attempt_id']=='R01'];assert len(failure)==1 and not failure[0]['admitted'] and failure[0]['admitted_target']==-1
    record('full1024_plus66_census_all8_and_before_only_failure_retained',full_population)
    record('old_plus_new_real_duplicate_domain_rejected',lambda:expect_fail(lambda:population(dict(a,sources=a['sources']+[a['sources'][1]]),r)))
    def corrupted_origin():
        bad=copy.deepcopy(a);bad['groups'][0]['metadata8'][2]['original_input_flags']['constructed']=False;expect_fail(lambda:population(bad,r))
    record('all8_original_flags_preserved',corrupted_origin)
    def unavailable_known():
        bad=copy.deepcopy(a)
        for g in bad['groups']:
            if g['attempt_id']=='R00':g['metadata8'][0]['original_query_available']=False
        expect_fail(lambda:population(bad,r))
    record('inputU_cannot_become_known',unavailable_known)
    def triplet():
        bad=copy.deepcopy(a);g=next(g for g in bad['groups'] if g['attempt_id']=='R00' and g['variant']=='SHAM');g.update(admitted=False,admitted_target=-1);expect_fail(lambda:population(bad,r))
    record('no_partial_verified_triplet_admission',triplet)
    def proof14():
        rec=copy.deepcopy(r);rr=next(v for v in rec.values() if isinstance(v,list) and len(v)==66);del rr[0]['closed14_actual_proof'][m.PROOF14[-1]];expect_fail(lambda:population(a,rec))
    record('every_frozen14_proof_required',proof14)
    def reusable():
        box=np.array([1.,2.,9.,10.]);key=m.raw_key([16,18],'RGB_A',box)
        assert key!=m.raw_key([16,18],'RGB_B',box) and key!=m.raw_key([18,16],'RGB_A',box) and key!=m.raw_key([16,18],'RGB_A',box+1e-10)
    record('old_raw4_reuse_requires_exact_fullRGB_shape_F64box',reusable)
    def old_packet():
        g=a['groups'][3];mm=g['metadata8'];box=np.array([0.,1.,7.,8.]);z=np.zeros((16,18),np.float32)
        packet=dict(image_logits=z,lowres_logits=np.zeros((256,256),np.float32),predicted_quality=np.array([.5],np.float32),mask_logits_gt0=z>0,box_xyxy=box,original_query_xy=np.asarray([v['query_xy'] for v in mm],np.float64),original_L=np.asarray([v['L'] for v in mm],np.float64),native_slot=np.arange(8,dtype=np.int8),original_global_index=np.asarray([v['original_global_index'] for v in mm],np.int64),original_occurrence_id=np.asarray([v['original_occurrence_id'] for v in mm]),original_query_available=np.asarray([v['original_query_available'] for v in mm],bool),prediction_available=np.asarray([v['original_query_available'] for v in mm],bool))
        raw=m.validate_old_raw4(packet,g,box);assert set(raw)==set(m.RAW4) and all(raw[k].tobytes()==packet[k].tobytes() for k in raw)
        raw['image_logits'][0,0]=3;assert packet['image_logits'][0,0]==0
        bad=dict(packet,mask_logits_gt0=np.ones_like(z,bool));expect_fail(lambda:m.validate_old_raw4(bad,g,box))
        bad=dict(packet,original_query_xy=packet['original_query_xy'].astype(np.float32));expect_fail(lambda:m.validate_old_raw4(bad,g,box))
    record('exact12_oldpacket_metadata_strictzero_and_raw4_copy_only',old_packet)
    def load_original():
        g=a['groups'][0];mm=g['metadata8'];arr=dict(full_RGB=np.zeros((3,16,18,3),np.uint8),rgb96=np.zeros((3,8,96,96,3),np.uint8),valid96=np.zeros((3,8,96,96),bool),query_xy=np.asarray([v['query_xy'] for v in mm],np.float64),L=np.asarray([v['L'] for v in mm],np.float64),native_slot=np.arange(8,dtype=np.int8),original_global_index=np.asarray([v['original_global_index'] for v in mm],np.int64),original_occurrence_id=np.asarray([v['original_occurrence_id'] for v in mm]),original_query_available=np.asarray([v['original_query_available'] for v in mm],bool),**{k:np.full((3,8),-1,np.int8) for k in ('new_native_visibility_target','new_quality_target','new_owner_target')})
        rgb,raw,valid,meta=m.load_pixels(g,lambda b,ks:{k:arr[k] for k in ks});assert rgb.dtype==np.uint8 and raw.shape==(8,96,96,3) and np.array_equal(meta['query_xy'],arr['query_xy'])
        arr['new_owner_target'][0,0]=0;expect_fail(lambda:m.load_pixels(g,lambda b,ks:{k:arr[k] for k in ks}))
    record('original_raw96_all8_META_and_oldtargetsUNKNOWN_only',load_original)
    def target_free():
        n=8;x={k:np.zeros((n,96,96,3) if k=='rgb96' else (n,96,96),np.uint8 if k=='rgb96' else bool) for k in ('rgb96','valid96','sam_mask96','sam_available96')};x.update(row_id=np.asarray(['toy%d'%i for i in range(n)]),original_global_index=np.zeros(n,np.int64),native_slot=np.arange(8,dtype=np.int8),query_xy=np.zeros((n,2),np.float64),L=np.ones(n,np.float64),original_query_available=np.ones(n,bool));c.validate_input10(x,n);expect_fail(lambda:c.validate_input10(dict(x,direct_end_external_target=np.zeros(n,np.int8)),n))
    record('exact_targetfree_INPUTS10_repeated_sourceIDs_allowed',target_free)
    record('forbidden_path_rejected_before_access',lambda:expect_fail(lambda:c.safe(m.P/'forbidden_v112'/'never_open')))
    def false_entry():
        old=c.read
        try:c.read=lambda p:dict(approved_direct_control_cache=False,source_only=True);expect_fail(lambda:m.build(m.H/'NOT_ACTUAL.toy'))
        finally:c.read=old
    record('false_entry_cannot_open_source_or_model',false_entry)
    return results

if __name__=='__main__':
    result=checks();c.write(m.H/'GENERATED_INVENTED_EXPANDED_CACHE_CHECKS.json',dict(completed=True,PASS=True,checks=result,count=len(result),actual_source_body_header_cache_weight_model_or_bulk_census_reads=0))
    print('INVENTED_EXPANDED_CACHE_CHECKS_PASS',len(result),flush=True)
