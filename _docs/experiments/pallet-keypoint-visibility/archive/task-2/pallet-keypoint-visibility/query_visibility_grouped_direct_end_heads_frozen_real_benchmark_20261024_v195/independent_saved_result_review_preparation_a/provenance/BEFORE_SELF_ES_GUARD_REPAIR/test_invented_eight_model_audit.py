"""Invented metadata/scalars/packets only; no real source or result reads."""
from pathlib import Path
import json,tempfile
import numpy as np
import audit_saved_eight_model_results as a

def reject(fn):
    try:fn()
    except AssertionError:return
    raise AssertionError('Invalid invented fixture accepted')

def save(path,payload):
    with path.open('xb')as f:np.savez_compressed(f,**payload)
    return a.bind(path)

def checks():
    result=[]
    def test(name,fn):fn();result.append(dict(name=name,PASS=True))
    test('closed_V191_scalar_denominator_U_matrix7_fixtures',lambda:a.need(a.selftest()['PASS'],'Inherited scalar fixture'))
    def task():
        rows=[dict(id='I',native_corner=i,human_state=t,m_original_response=p)for i,(t,p)in enumerate([(a.D,'VISIBLE'),(a.S,'ABSENT_NONVISIBLE'),(a.E,'OTHER_UNRESOLVED')])]
        truth={a.key(r):dict(present_at_fixed_0_5='True'if i==0 else'False',legacy_stored_state='B'if i==0 else'S'if i==1 else'U')for i,r in enumerate(rows)}
        s=a.calculated(rows[:2],'legacy','D_vs_S',truth);e=a.calculated([rows[0],rows[2]],'legacy','D_vs_E',truth)
        assert s['TP']==s['TN']==1 and s['out_of_task_S_rate']==0 and e['U_occluded']==1 and e['unresolved_rate']==.5
        assert a.calculated([], 'm','D_vs_E',truth)['visible_precision']is None
    test('alltasks_legacyS_taskscope_and_zero_denominator',task)
    with tempfile.TemporaryDirectory(prefix='invented_only_',dir=a.H)as folder:
        folder=Path(folder);domains={};ledgers={};outputs={arm:{}for arm in a.NEW_ARMS};packets={}
        for domain,n,start,U in[('CURRENT223_1784',1784,0,45),('PUBLIC_HELD45_360',360,584,108)]:
            av=np.arange(n)<n-U;q=np.column_stack((np.arange(n,dtype=np.float64)+np.pi,np.full(n,17.25)));L=np.full(n,384.,np.float64)
            x=dict(row_id=np.asarray(['%s:%d'%(domain,i)for i in range(n)]),original_global_index=np.arange(start,start+n,dtype=np.int64),native_slot=np.arange(n,dtype=np.int64).astype(np.int8)%8,query_xy=q,L=L,original_query_available=av)
            mb=folder/(domain+'_ROWS.jsonl');metadata=[dict(row_id=str(x['row_id'][i]),original_query_available=bool(av[i]),native_slot=int(x['native_slot'][i]),query_xy=q[i].tolist(),L=float(L[i]))for i in range(n)];mb.write_text(''.join(json.dumps(r)+'\n'for r in metadata));domains[domain]=dict(inputs=save(folder/(domain+'_META.npz'),x),metadata=a.bind(mb))
            raw_s=np.zeros((n,2),np.float32);raw_o=np.zeros((n,3),np.float32);raw_d=np.where(np.arange(n)%2==0,0.,-np.finfo(np.float32).tiny).astype(np.float32);raw_s[~av]=np.nan;raw_o[~av]=np.nan;raw_d[~av]=np.nan
            sp=np.where(av[:,None],np.zeros((n,2),np.int8),np.full((n,2),-1,np.int8));op=np.where(av,0,-1).astype(np.int8);dp=np.where(av,(raw_d>=0).astype(np.int8),-1).astype(np.int8);proxy=np.full(n,-1,np.int8)
            z=dict(x,raw_support_logits=raw_s,raw_owner_logits=raw_o,raw_direct_end_external_logit=raw_d,support_prediction=sp,owner_prediction=op,direct_end_prediction_D0_E1_Uminus1=dp,regional_support_proxy_D0_E1_Uminus1=proxy,support_order=np.array(['visible_end_support','foreign_front_support']),owner_order=np.array(['TARGET','FOREIGN','BACKGROUND']),direct_end_order=np.array(['VISIBLE_END0','EXTERNAL_BLOCKED_END1','UNKNOWNminus1']));packets[domain]=z
            rows=[]
            for i in range(n):
                r=dict(global_index=i,native_corner=int(x['native_slot'][i]),query_x=float(q[i,0]),query_y=float(q[i,1]),roi_side=float(L[i]),query_available=bool(av[i]))if domain=='CURRENT223_1784'else dict(original_global_index=start+i,query_id=str(x['row_id'][i]))
                for tag in('C56','C57'):
                    r.update({tag+'_support_prediction':sp[i].tolist(),tag+'_owner_prediction':int(op[i]),tag+'_primary_prediction':int(dp[i]),tag+'_regional_support_proxy_D0_E1_Uminus1':int(proxy[i])})
                    if domain=='CURRENT223_1784':r.update({tag+'_raw_support_logits':[0.,0.]if av[i]else[None,None],tag+'_raw_owner_logits':[0.,0.,0.]if av[i]else[None]*3,tag+'_raw_direct_end_external_logit':float(raw_d[i])if av[i]else None})
                rows.append(r)
            ledgers[domain]=rows
            for arm in a.NEW_ARMS:outputs[arm][domain]=save(folder/(arm+'_'+domain+'.npz'),z)
        code=dict(path='AUTHORED_CODE',sha256='0'*64);proto=dict(path='AUTHORED_PROTO',sha256='1'*64);inp=dict(path='AUTHORED_INPUT_COMPLETE',sha256='2'*64);raw=dict(path='AUTHORED_ORIGINAL_RAW_COMPLETE',sha256='3'*64);source_path='AUTHORED_SOURCE_FINAL_COMPLETE';seal=dict(completed=True,PASS=True,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,annotations_reference_GT_oldreal_scores_or_fullRGB_read=False,all_batch_repeat2_state_and_input_bytes_exact=True,code=code,protocol=proto,selected_model=None,adopted=False,original_rows2144=2144,model_domain_rows4288=4288,input_complete=inp,original_raw_complete=raw,source_fit_complete=dict(path=source_path,sha256='4'*64),arms={arm:dict(updates=8000,outputs=outputs[arm])for arm in a.NEW_ARMS});p=dict(predictor_code=code,predictor_protocol=proto,input_domains=domains,input_complete=inp,original_raw_complete=raw,source_fit_complete_path=source_path)
        def all_packets():assert a.prediction_audit(seal,p,ledgers['CURRENT223_1784'],ledgers['PUBLIC_HELD45_360'])['new_packet_rows']==4288
        test('authored4288_packet_scalar_U_F64_META_originaljoins',all_packets)
        def drift():
            old=outputs[a.NEW_ARMS[0]]['CURRENT223_1784'];bad=dict(packets['CURRENT223_1784'],query_xy=packets['CURRENT223_1784']['query_xy'].astype(np.float32));outputs[a.NEW_ARMS[0]]['CURRENT223_1784']=save(folder/'BAD_F32.npz',bad);reject(lambda:a.prediction_audit(seal,p,ledgers['CURRENT223_1784'],ledgers['PUBLIC_HELD45_360']));outputs[a.NEW_ARMS[0]]['CURRENT223_1784']=old
        test('F32_query_alias_drift_rejected_before_numericalclaim',drift)
        def gate():
            false=folder/'FALSE_ENTRY.json';false.write_text(json.dumps(dict(approved_actual_independent_saved_result_audit=False,all_two_real_predictions_frozen=False,saved_evaluation_completed=False,automatic_retry=False)));reject(lambda:a.run(false))
        test('false_rootentry_cannot_read_any_actual_outputs_or_labels',gate)
    return dict(completed=True,PASS=True,invented_only=True,checks=result,count=len(result),production_real_source_input_prediction_label_result_array_header_weight_or_image_reads=0,model_instantiation_forward_fit_calls=0)

if __name__=='__main__':
    r=checks();a.write(a.H/'GENERATED_INVENTED_EIGHT_MODEL_AUDIT_CHECKS_FINAL.json',r);print('INVENTED_EIGHT_MODEL_AUDIT_PASS',r['count'],flush=True)
