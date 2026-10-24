"""Invented metadata/logits only; no current artifact/model/GT body is opened."""
import argparse
import copy
from pathlib import Path
from collections import Counter
import numpy as np
import current_adapter as c
import evaluate_current_saved as ev


def csv_style(rows):return [{k:(''if v is None else str(v))for k,v in r.items()}for r in rows]


def fixture(core):
    prior=[];truth=[]
    for fi in range(223):
        for k in range(8):
            i=fi*8+k;q=[16.25+k,24.25+k];valid=i<1739
            prior.append(dict(id='INVENTED_%03d'%fi,native_corner=k,image_sha256='f'*64,
                query_x=q[0],query_y=q[1],pixel_x=16+k,pixel_y=24+k,roi_side=64.,valid=valid,
                saved_roi_pixel_valid=valid,geometry_self_hidden=i%11==0,prior_reason='GENERATED_PRIOR_U',
                status='OK'if valid else'UNKNOWN',reason=''if valid else'GENERATED_PRIOR_U'))
            group='WOOD_PRIMARY105'if fi<105 else'SQUARE118'
            state='DIRECT_VISIBLE'if k<4 else'EXTERNAL_OCCLUDED'if(k<6 or fi>=105)else'SELF_OCCLUDED'
            truth.append(dict(id=prior[-1]['id'],native_corner=str(k),image_sha256='f'*64,
                query_x=str(q[0]),query_y=str(q[1]),group=group,human_state=state,
                recording='INVENTED',present_at_fixed_0_5=str(k%2==0),
                legacy_stored_state='U'if k==7 else'S'if k==6 else'B'if k%2==0 else'O',
                paired_same_query_RGB_D_vs_E='True',paired_same_query_RGB_D_vs_ES='True',
                same_slot_pipeline_D_vs_E='True',same_slot_pipeline_D_vs_ES='True',
                initial_reference_error_px=('2.'if i%3==0 else'4.'if i%3==1 else''),
                human_reference_inside_nominal_crop='True'))
    # Missing human state remains in full row inventory, excluded only by frozen task definitions.
    truth[7]['human_state']=''
    manual=[]
    for group,count in [('WOOD_PRIMARY105',153),('SQUARE118',223)]:
        selected=[r for r in truth if r['group']==group and r['human_state']=='DIRECT_VISIBLE'][:count]
        manual.extend(dict(r,annotation_coordinate_source='manual_click',initial_reference_error_px='2.',query_valid='True')for r in selected)
    p=np.array([r['valid']for r in prior]);context=np.ones(1784,bool);context[100]=False;central=np.ones(1784,bool)
    a=p&context&central;z=np.where(np.arange(1784)%2,-1.,1.).astype(np.float32)
    z[:4]=np.array([-np.nextafter(np.float32(0),np.float32(1)),0.,-0.,np.nextafter(np.float32(0),np.float32(1))],np.float32)
    z[~a]=0.;pv=np.full(1784,.5,np.float32)
    records=core.decode_rows(prior,z,pv,a)
    for r in records:r.update(context_available=True,local_central4_available=True,context_Q_inside=False,
        context_box_mode=c.BOX_MODE,context_bbox_origin='INITIAL_Q8_DERIVED_NOT_SELECTED_DETECTION')
    records[100]['context_available']=False
    old=core.decode_rows(prior,-z,pv,p)
    for r in old:r['raw_physical_external_logit']=r.pop('raw_query_aux_hidden_logit')
    raw=dict(query_logit=z,valid=a,query_available=a,prior_valid=p,context_available=context,
        local_central4_available=central,context_Q_inside=np.zeros(1784,bool),
        original_query_xy=np.array([[r['query_x'],r['query_y']]for r in prior],np.float64),
        original_side_L=np.full(1784,64.,np.float64),id=np.array([r['id']for r in prior]),
        native_corner=np.array([r['native_corner']for r in prior],np.int8))
    return prior,truth,manual,csv_style(records),csv_style(old),raw


def run():
    core=c.pure_core();prior,truth,manual,records,old,raw=fixture(core)
    review=ev.validate_raw(core,records,raw)
    assert review['total_U']==46 and review['extra_U']==1
    assert records[0]['authoritative_decision']=='D'and records[0]['p_visible']=='0.5'
    assert all(records[i]['authoritative_decision']=='E'for i in(1,2,3))
    assert records[100]['authoritative_decision']=='UNKNOWN'and records[100]['geometry_self_hidden_if_valid']==''
    base=c.metric_base();tables=core.calculate_all(base,truth,prior,records,old,manual)
    assert tables['availability']['prior_input_U']==45 and tables['availability']['new_total_U']==46
    assert {r['N']for r in tables['manual_near_D']}=={153,223}
    assert not any(r['group']=='SQUARE118'and r['task']=='D_vs_S'for r in tables['metrics'])
    assert all(not r['learned_self_prediction']and not r['external_vs_self_identified']for r in tables['human_self_binary_response'])
    matrices=ev.matrices(tables['metrics']);new=core.score_index(records,'raw_query_aux_hidden_logit')
    for row in matrices:
        rr=[r for r in truth if r['group']==row['group']and r[row['cohort']+'_D_vs_E']=='True'and r['human_state']in(base.D,base.E)]
        assert row['human_D_N']+row['human_external_E_N']==len(rr)
        if row['model']in(c.NEW,c.NEW_GEO):
            counts=Counter((r['human_state'],core.response(new[core.key(r)],row['model']==c.NEW_GEO))for r in rr)
            for state,prefix in [(base.D,'D'),(base.E,'E')]:
                assert row[prefix+'_pred_D']==counts[state,'VISIBLE']
                assert row[prefix+'_pred_hidden']==counts[state,'ABSENT_NONVISIBLE']
                assert row[prefix+'_pred_U']==counts[state,'OTHER_UNRESOLVED']
    bins=ev.error_bins(core,base,truth,manual,records,old)
    for cohort in ('paired_same_query_RGB','same_slot_pipeline'):
        for group in base.GROUPS:
            for model in c.MODELS:
                b=[r for r in bins if r['cohort']==cohort and r['group']==group and r['model']==model]
                m=next(r for r in matrices if r['cohort']==cohort and r['group']==group and r['model']==model)
                assert len(b)==3 and sum(r['N']for r in b)==m['human_D_N']+m['human_external_E_N']
                assert all(r['N']>0 for r in b)
    historical=csv_style([r for r in tables['metrics']if r['model']in('v61','legacy',c.OLD,c.OLD_GEO)])
    checked=core.historical_replay(tables['metrics'],historical,{'v61','legacy',c.OLD,c.OLD_GEO})
    rejects=0
    for field in ('query_x','roi_side','id'):
        bad=copy.deepcopy(records);bad[0][field]='CHANGED'if field=='id'else'999.'
        try:ev.validate_raw(core,bad,raw)
        except(ValueError,AssertionError):rejects+=1
        else:raise AssertionError('Changed metadata accepted: '+field)
    bad=copy.deepcopy(records);bad[0]['authoritative_decision']='E'
    try:core.score_index(bad,'raw_query_aux_hidden_logit')
    except(ValueError,AssertionError):rejects+=1
    else:raise AssertionError('Wrong sign export accepted')
    bad=copy.deepcopy(records);bad[100]['geometry_self_hidden_if_valid']='True'
    try:core.score_index(bad,'raw_query_aux_hidden_logit')
    except(ValueError,AssertionError):rejects+=1
    else:raise AssertionError('Invalid geometry override accepted')
    bad=dict(raw);bad['valid']=raw['valid'].astype(np.int8)
    try:ev.validate_raw(core,records,bad)
    except(ValueError,AssertionError):rejects+=1
    else:raise AssertionError('Wrong dtype accepted')
    assert core.response({'valid':False,'geo':True,'z':-1.},True)=='OTHER_UNRESOLVED'
    assert core.response({'valid':True,'geo':True,'z':-1.},False)=='VISIBLE'
    assert core.response({'valid':True,'geo':True,'z':-1.},True)=='ABSENT_NONVISIBLE'
    return dict(completed=True,PASS=True,code=c.bind(ev.__file__),common_code=c.bind(c.__file__),test_code=c.bind(__file__),
        pinned_evaluator=c.bind(c.EVALUATOR),pinned_decoder=c.bind(c.DECODER),pinned_metric=c.bind(c.METRIC),
        generated_rows=1784,prior_U=45,extra_context_U=1,raw_sign_subnormal_signed_zero_and_half_display=True,
        raw_geometry_routes_and_invalid_priority=True,original_q_L_ID_order_rejections=rejects,
        exact_2x3_and_denominator_checks=len(matrices),historical_rows_replayed=checked,
        coordinate_bins_partition_exact=True,manual_near_D_denominators=[153,223],
        missing_human_state_preserved=True,learned_S_claim=False,
        actual_current_GT_RGB_scores_weights_reads=0,actual_model_forward_fit_PnP=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=run();c.write_new(args.output,result);print(result)
