"""Root-completion-only saved audit; no torch/model/producer imports or target reads."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
V181=P/'query_visibility_local_support_raw_head_20261024_v181'
META=['row_id','original_global_index','native_slot','query_xy','L','original_query_available']
ARMS=['C50_RAW96_DUAL_SUPPORT_OWNER_AUX','C51_RAW96_PLUS_FROZEN_SAM_DUAL_SUPPORT_OWNER_AUX']
def need(x,msg):
    if not x:raise AssertionError(msg)
def sha(p):
    p=Path(p).resolve();need(p.is_relative_to(P),'source project')
    need(not any(s in str(p).lower() for s in ['_v112','_v169','_v177','_v178','current223','synth_val','validation2048']),'TRAIN scope')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def binding(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def check(b):need(sha(b['path'])==b['sha256'],'SHA '+b['path']);return Path(b['path'])
def packet(b):
    with np.load(check(b),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def same(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return a.shape==b.shape and (np.array_equal(a,b) if a.dtype.kind==b.dtype.kind=='U' else a.dtype==b.dtype and a.tobytes()==b.tobytes())
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def sample(logits,q,L,on):
    m=np.zeros((96,96),bool);a=m.copy()
    if not on:return m,a
    need(q is not None and L is not None,'available qL')
    h,w=logits.shape;offset=((np.arange(96,dtype=np.float64)+.5)/96-.5)*L
    x,y=np.meshgrid(q[0]+offset,q[1]+offset);i=np.floor(x).astype(int);j=np.floor(y).astype(int);fx=x-i;fy=y-j;z=np.zeros((96,96),np.float64)
    for dx,dy in [(0,0),(1,0),(0,1),(1,1)]:
        px=i+dx;py=j+dy;inside=(px>=0)&(px<w)&(py>=0)&(py<h)
        wt=(fx if dx else 1-fx)*(fy if dy else 1-fy)
        z[inside]+=logits[py[inside],px[inside]].astype(np.float64)*wt[inside]
    a=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5);return (z>0)&a,a
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--complete',type=Path,required=True);a=ap.parse_args();start=time.monotonic()
    c=read(a.complete);need(c['completed'] and c['PASS'] and c['source_only'] and c['predictions_sealed_before_target_read'] and c['new_target_files_read']==0,'actual complete barrier')
    for key in ['scope','entry','code','protocol','head','common_input_complete','SAM_complete','frozen_V181_fit','all2_predictions_seal','input_lineage','SAM96_input_only','COVER_input_parity','COVER_score_delta']:check(c[key])
    proto=read(c['protocol']['path']);common=read(c['common_input_complete']['path']);sam=read(c['SAM_complete']['path']);fit=read(c['frozen_V181_fit']['path']);seal=read(c['all2_predictions_seal']['path'])
    need(seal['PASS'] and seal['all2_final8000_frozen'] and seal['new_target_files_read']==0 and seal['no_target_or_label_or_GT_model_inputs'],'sealed noGT receipt')
    x=packet(common['inputs']);ss=packet(c['SAM96_input_only']);rr=rows(check(common['metadata']));lineage=rows(c['input_lineage']['path']);facts=rows(Path(a.complete).parent/'ALL4096_SAM_INPUT_FACTS.jsonl')
    need(set(x)==set(META)|{'rgb96','valid96'} and set(ss)==set(META)|{'sam_mask96','sam_available96'},'input whitelist noGT')
    need(len(rr)==len(lineage)==len(facts)==4096,'all4096')
    for key in META:need(same(ss[key],x[key]),'SAMcache identity '+key)
    need(ss['sam_mask96'].dtype==ss['sam_available96'].dtype==bool and ss['sam_mask96'].shape==(4096,96,96),'SAM96schema')
    for i,r in enumerate(rr):
        need(r['diagnostic_row_index']==r['original_global_index']==i and r['split']=='TRAIN','new diagnostic IDs/split')
        need(all(lineage[i][k]==r[k] for k in lineage[i]),'dualorigin lineage')
        need(x['row_id'][i]==r['row_id'] and x['native_slot'][i]==r['native_slot'] and x['original_global_index'][i]==i,'rowmetadata')
        need(np.array_equal(x['query_xy'][i],np.asarray([np.nan,np.nan] if r['query_xy'] is None else r['query_xy'],np.float64),equal_nan=True),'qL metadata')
        need(np.array_equal(np.asarray(x['L'][i]),np.asarray(np.nan if r['L'] is None else r['L'],np.float64),equal_nan=True),'L metadata')
        need(x['original_query_available'][i]==r['anchor_original_flags']['query_available'],'anchor originalU')
    need(np.count_nonzero(~x['original_query_available'])==96,'U96')
    bygroup={g['group_id']:g for g in sam['groups']};need(len(bygroup)==512,'all512SAMgroups');source_bindings={};samrows=0
    for gi in range(512):
        inds=list(range(gi*8,gi*8+8));r0=rr[inds[0]];g=bygroup[r0['group_id']];z=None
        need(g['common_query_xy8']==[rr[i]['query_xy'] for i in inds] and g['common_L8']==[rr[i]['L'] for i in inds],'SAMq8/L8')
        if g['outputs'] is not None:
            source_bindings[g['outputs']['path']]=g['outputs'];z=packet(g['outputs']);need(z['image_logits'].dtype==np.float32 and np.isfinite(z['image_logits']).all(),'savedSAMfull logits')
            for key,ik in [('original_occurrence_id','row_id'),('original_global_index','original_global_index'),('native_slot','native_slot'),('original_query_xy','query_xy'),('original_L','L'),('original_query_available','original_query_available')]:need(same(z[key],x[ik][inds]),'fresh SAMdiagnosticmetadata '+key)
            need(same(z['prediction_available'],x['original_query_available'][inds]),'SAMpredav originals')
        else:need(not g['prompt_available'] or not g['runtime_available'],'nullSAM explicit')
        for k,i in enumerate(inds):
            on=z is not None and bool(z['prediction_available'][k]);m,av=sample(None if z is None else z['image_logits'],rr[i]['query_xy'],rr[i]['L'],on)
            need(same(m,ss['sam_mask96'][i]) and same(av,ss['sam_available96'][i]),'independently reconstructed SAM96 '+str(i));samrows+=1
            need(facts[i]['row_id']==rr[i]['row_id'] and facts[i]['sam_mask96_sha256']==ah(m) and facts[i]['sam_available96_sha256']==ah(av),'SAMfacts')
    oldmeta=read(check(proto['original_source_cache_metadata']));old=packet(oldmeta['domains']['SYNTH_TRAIN4096']['inputs']);covers=[]
    for i,r in enumerate(rr):
        if r['variant']!='COVER':continue
        j=r['anchor_original_global_index'];need(r['source_image_original_global_index']==j,'COVERsameoriginalimage')
        for key in ['rgb96','valid96','native_slot','query_xy','L','original_query_available']:need(same(x[key][i],old[key][j]),'COVERrawinput '+key)
        for key in ['sam_mask96','sam_available96']:need(same(ss[key][i],old[key][j]),'COVERsaminput '+key)
        covers.append((i,j))
    need(len(covers)==1024,'allCOVER1024')
    del old
    delta_saved=read(c['COVER_score_delta']['path']);deltas={};decode_rows=0
    for arm in ARMS:
        receipt=c['arms'][arm];ref=fit['arms'][arm];need(receipt==seal['arms'][arm],'arm frozen seal')
        need(ref['updates']==8000 and receipt['source_checkpoint']==ref['checkpoint']==proto['frozen_models'][arm]['checkpoint'],'original final8000 checkpoint binding')
        need(receipt['final_state_sha']==ref['final_state_sha'] and receipt['state_after_inference_unchanged'] is True,'state receipt')
        need(receipt['first_batch_repeat']['repeat2_exact'] is True and receipt['first_batch_repeat']['rows']==np.flatnonzero(x['original_query_available'])[:96].tolist(),'fixed firstbatch repeat receipt')
        z=packet(receipt['outputs']);need(set(z)==set(META)|{'raw_support_logits','raw_owner_logits','support_prediction','owner_prediction','regional_support_proxy_D0_E1_Uminus1','support_order','owner_order'},'exact13savedkeys')
        for key in META:need(same(z[key],x[key]),'savedoutput metadata '+key)
        need(z['support_order'].tolist()==['visible_end_support','foreign_front_support'] and z['owner_order'].tolist()==['TARGET','FOREIGN','BACKGROUND'],'task order')
        sp=np.full((4096,2),-1,np.int8);op=np.full(4096,-1,np.int8);proxy=np.full(4096,-1,np.int8)
        need(z['raw_support_logits'].dtype==z['raw_owner_logits'].dtype==np.float32,'FP32 outputs')
        for i in range(4096):
            if not x['original_query_available'][i]:need(np.isnan(z['raw_support_logits'][i]).all() and np.isnan(z['raw_owner_logits'][i]).all(),'UrawNaN');continue
            need(np.isfinite(z['raw_support_logits'][i]).all() and np.isfinite(z['raw_owner_logits'][i]).all(),'availablefinite')
            for k in range(2):sp[i,k]=int(float(z['raw_support_logits'][i,k])>0)
            best=0
            for k in [1,2]:
                if float(z['raw_owner_logits'][i,k])>float(z['raw_owner_logits'][i,best]):best=k
            op[i]=best;proxy[i]=1 if sp[i,1] else 0 if sp[i,0] else -1
        for k,w in [('support_prediction',sp),('owner_prediction',op),('regional_support_proxy_D0_E1_Uminus1',proxy)]:need(same(z[k],w),'independent scalar decoding '+k)
        decode_rows+=4096
        oldscore=packet(proto['frozen_models'][arm]['old_COVER_reference_scores']);ii=np.asarray([i for i,j in covers]);jj=np.asarray([j for i,j in covers]);av=x['original_query_available'][ii];rd={}
        for key in ['raw_support_logits','raw_owner_logits']:
            aa=z[key][ii][av];bb=oldscore[key][jj][av];d=aa.astype(np.float64)-bb.astype(np.float64)
            rd[key]=dict(exact_array_bytes_equal=aa.tobytes()==bb.tobytes(),max_abs_delta=float(np.abs(d).max()),mean_abs_delta=float(np.abs(d).mean()),nonzero_elements=int(np.count_nonzero(d)))
            need(rd[key]==delta_saved['comparison'][arm][key],'descriptive oldscore delta')
        for key in ['support_prediction','owner_prediction','regional_support_proxy_D0_E1_Uminus1']:
            rd[key+'_changed_elements']=int(np.count_nonzero(z[key][ii]!=oldscore[key][jj]));need(rd[key+'_changed_elements']==delta_saved['comparison'][arm][key+'_changed_elements'],'delta decoding census')
        deltas[arm]=rd
    for b in source_bindings.values():check(b)
    for key in ['inputs','metadata']:check(common[key])
    result=dict(completed=True,PASS=True,code=binding(__file__),actual_complete=binding(a.complete),coverage=dict(input_identity_qL_flags=4096,input_U96=96,SAM96_reconstructed_rows=samrows,SAM_full_packets=len(source_bindings),original_COVER_input_byteparity=1024,decoded_output_rows=decode_rows,frozen_heads=2),COVER_saved_score_delta_measurement_only=deltas,checkpoint_body_or_hash_reads=0,model_forward_or_fit=0,targets_or_source_GT_reads=0,private_held_VAL_or_V112_reads=0,model_receipt_scope='Checkpoint paths/SHA and final-state/repeat claims bound to original frozen fit and actual runner seal; checkpoint tensor state and repeatforward not reexecuted by this auditor.',no_GT_model_input_evidence='Exact8 input whitelist, SAM96 exact8 whitelist, sourcecode review and zero-target-read runner receipts; not an OS-level access trace.',no_output_based_selection=True,selected_model=None,adopted=False,fully_solved=False,seconds=time.monotonic()-start)
    with (H/'REVIEW.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(PASS=True,review=binding(H/'REVIEW.json'))),flush=True)
if __name__=='__main__':main()
