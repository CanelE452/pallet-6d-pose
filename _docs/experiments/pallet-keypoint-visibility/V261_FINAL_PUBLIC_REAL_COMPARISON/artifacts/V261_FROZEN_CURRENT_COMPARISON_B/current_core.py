"""Pure canonical local inference and fixed saved-ledger evaluation; no I/O."""
from collections import Counter
import math
import numpy as np
import torch

D='DIRECT_VISIBLE';E='EXTERNAL_OCCLUDED';S='SELF_OCCLUDED'
GROUPS=('WOOD_PRIMARY105','SQUARE118')

def require(ok,message):
    if not bool(ok):raise ValueError(message)

def canonical_frame_features(frame,rgb,encoder,op,device,repeat=False):
    """No GT, global bbox, context encoder or other-query image descriptors."""
    require(rgb.dtype==np.uint8 and list(rgb.shape)==frame['raw_hw']+[3],'Original decoded RGB HWC identity')
    feature=np.zeros((8,1186),np.float32);supported=np.zeros(8,bool);reasons=[None]*8
    pending=[];raws=[];valids=[];hash_inputs=[]
    import hashlib
    for k,q in enumerate(frame['queries']):
        require(q['native_slot']==k,'Original native order')
        if not q['original_query_available']:reasons[k]='ORIGINAL_INPUT_UNAVAILABLE';continue
        xy=np.asarray(q['query_xy'],np.float64);L=q['L']
        if xy.shape!=(2,)or not np.isfinite(xy).all()or L is None or not np.isfinite(L)or L<=0:
            reasons[k]='NONFINITE_OR_NONPOSITIVE_ORIGINAL_Q_L';continue
        raw,valid=op.crop(rgb,xy,L,96)
        hash_inputs.append({'native_slot':k,'RGB96_CHW_sha256':hashlib.sha256(raw.tobytes()).hexdigest(),'valid96_bool_sha256':hashlib.sha256(valid.tobytes()).hexdigest()})
        if not valid.any():reasons[k]='ALL_INVALID_ORIGINAL_LOCAL96';continue
        if not valid[:,47:49,47:49].all():reasons[k]='ORIGINAL_QUERY_CENTRAL4_UNSUPPORTED';continue
        pending.append(k);raws.append(raw);valids.append(valid)
    repeated=False
    if pending:
        raw=torch.from_numpy(np.stack(raws));valid=torch.from_numpy(np.stack(valids))
        with torch.inference_mode():
            pixel,fractions,c4=op.node_pixel_batch(encoder,raw,valid,op.preprocess_views,op.crop_module,device)
            if repeat:
                again,af,ac=op.node_pixel_batch(encoder,raw,valid,op.preprocess_views,op.crop_module,device)
                require(pixel.tobytes()==again.tobytes() and fractions.tobytes()==af.tobytes() and c4.tobytes()==ac.tobytes(),'First supported canonical encoder batch repeat byteexact');repeated=True
        require(c4.all() and pixel.dtype==np.float32 and np.isfinite(pixel).all(),'Supported canonical pixel1154')
        for j,k in enumerate(pending):
            condition=op.condition32(frame,frame['queries'][k],np)
            require(condition.dtype==np.float32 and condition.shape==(32,) and condition[k]==1 and condition[:8].sum()==1,'Original native condition32 only')
            feature[k]=np.concatenate((pixel[j],condition));supported[k]=True
    return feature,supported,reasons,{'frame_index':frame['frame_index'],'local_supported':int(supported.sum()),'local_crop_facts':hash_inputs,'first_batch_repeat_byteexact':repeated,'global_bbox_or_context_encoder_used':False,'teacher_input_used':False}

def prediction_rows(frame,features,supported,reasons,heads):
    """Frozen76098 heads: MAIN output0 only, rawlogit>=0 E; input U preserved."""
    result={name:[]for name in heads};take=np.flatnonzero(supported)
    for name,head in heads.items():
        logits=None
        if len(take):
            with torch.inference_mode():logits=head(torch.from_numpy(features[take])).cpu().numpy()
            require(logits.shape==(len(take),2)and np.isfinite(logits).all(),'Two finite frozen logits')
        values={int(k):logits[j]for j,k in enumerate(take)}
        for k,q in enumerate(frame['queries']):
            value=values.get(k);main=None if value is None else float(value[0])
            result[name].append({'domain':'CURRENT223_1784','id':q['id'],'group_id':q['group_id'],'original_global_index':q['original_global_index'],'global_index':q['original_global_index'],'row_id':q['row_id'],'frame_index':frame['frame_index'],'native_slot':k,'query_xy':q['query_xy'],'L':q['L'],'original_query_available':q['original_query_available'],'input_supported':bool(supported[k]),'logit':main,'logit_MAIN0':main,'logit_AUX1_diagnostic_only':None if value is None else float(value[1]),'prediction':'U'if value is None else('E'if main>=0 else'D'),'U_reason':reasons[k],'threshold_logit':0,'selected_output':0})
    return result

def same_number(a,b):
    try:x=float(a)
    except(TypeError,ValueError):x=float('nan')
    try:y=float(b)
    except(TypeError,ValueError):y=float('nan')
    return (math.isnan(x)and math.isnan(y))or x==y

def evaluate_saved_predictions(ledger,predictions,metric_ops,paired_self_flags=None):
    """Caller freezes both1784 new predictions before obtaining this ledger."""
    require(len(ledger)==1784 and all(len(p)==1784 for p in predictions.values()),'All1784 rows, no pair search')
    if paired_self_flags is not None:
        require(len(paired_self_flags)==1784 and all(type(v)is bool for v in paired_self_flags),'Inherited ordered pairedSELF flags')
        ledger=[dict(r,paired_same_query_RGB_D_vs_ES=str(v))for r,v in zip(ledger,paired_self_flags)]
        require(sum(v and r['human_state']==S for r,v in zip(ledger,paired_self_flags))==21,'Original pairedSELF21; no relabeling')
    models=('v61','legacy',*predictions)
    lookup={}
    for i,r in enumerate(ledger):
        require(int(r['global_index'])==i and int(r['frame_index'])==i//8 and int(r['native_corner'])==i%8 and r['row_id']==r['id']+':native'+str(i%8),'Frozen ledger native/global row identity')
        require(r['v61_D_E_U']in('D','E','U')and r['legacy_D_E_U']in('D','E','U'),'Original stored baseline responses')
        for baseline in('v61','legacy'):
            require(r[baseline+'_D_E_U']=={'VISIBLE':'D','ABSENT_NONVISIBLE':'E'}.get(r[baseline+'_original_response'],'U'),'Frozen baseline response/state namespace unchanged')
        for name,rows in predictions.items():
            p=rows[i]
            require(p['id']==r['id']and p['row_id']==r['row_id']and p['original_global_index']==i and p['frame_index']==i//8 and p['native_slot']==i%8,'Exact current ledger join without matching/permutation')
            require(same_number(p['query_xy'][0],r['query_x'])and same_number(p['query_xy'][1],r['query_y'])and same_number(p['L'],r['roi_side']),'Original q/L preserved')
            require(p['original_query_available']==(r['original_query_valid']=='True'),'Original availability preserved')
            require(p['prediction']in('D','E','U'),'D/E/U namespace')
            require((p['prediction']=='U'and p['logit']is None and bool(p['U_reason']))or(p['prediction']!='U'and p['original_query_available']and math.isfinite(p['logit'])and p['prediction']==('E'if p['logit']>=0 else'D')),'FixedMAIN0 sign or explicit inputU')
            if not p['original_query_available']:require(p['prediction']=='U','Original inputU retained')
            lookup[(name,i)]=p['prediction']
    def yes(r,key):return r[key]=='True'
    primary=[r for r in ledger if yes(r,'paired_same_query_RGB_D_vs_E')and r['human_state']in(D,E)]
    require((len(primary),sum(r['human_state']==D for r in primary),sum(r['human_state']==E for r in primary))==(1237,1147,90),'Frozen primary1237 D1147E90, not coordinate-filtered')
    require(sum(yes(r,'same_slot_pipeline_D_vs_E')and r['human_state']in(D,E)for r in ledger)==1248,'Distinct same-slot1248')
    require(sum(r['human_state']==S for r in ledger)==136,'Original humanSELF136')
    require(sum(r['original_query_valid']!='True'for r in ledger)==45,'Original inputU45')
    # Existing ledger preserves full old response namespace, including legacy out-of-task S.
    def call(r,model,task,probabilities=None,threshold=.5,rules=None):
        if model in predictions:return {'D':'VISIBLE','E':'ABSENT_NONVISIBLE','U':'OTHER_UNRESOLVED'}[lookup[(model,int(r['global_index']))]]
        response=r[model+'_original_response']
        if task!='D_vs_E'and response=='OUT_OF_TASK_GEOMETRY_S':return'ABSENT_NONVISIBLE'
        return response
    metric_ops.namespace['call']=call
    def calc(rr,model,task='D_vs_E'):return metric_ops.namespace['calculate'](rr,model,task,{},.5)
    tables={'PRIMARY':[],'D_VS_EXTERNAL_E_2X3':[],'COORDINATE_ERROR_BINS':[],'HUMAN_SELF_BINARY_RESPONSE':[],'PRIMARY_NATIVE8':[],'ALL1784_MODEL_INVENTORY':[],'PAIRED_RECOVERY_AND_DAMAGE':[],'SAME_SLOT':[]}
    for group in('ALL',*GROUPS):
        pp=[r for r in primary if group=='ALL'or r['group']==group]
        for model in models:
            row=dict(group=group,model=model,**calc(pp,model));tables['PRIMARY'].append(row)
            tables['D_VS_EXTERNAL_E_2X3'].append({'group':group,'model':model,'N':row['N'],'D_D':row['TP'],'D_E':row['FN_decided'],'D_U':sum(row[k]for k in('U_visible','out_of_task_S_visible','other_unresolved_visible')),'E_D':row['FP'],'E_E':row['TN'],'E_U':sum(row[k]for k in('U_occluded','out_of_task_S_occluded','other_unresolved_occluded'))})
            for bucket in('LE3','GT3','MISSING_REFERENCE_ERROR'):
                rr=[r for r in pp if metric_ops.error_bucket(r)==bucket];tables['COORDINATE_ERROR_BINS'].append(dict(group=group,error_bin=bucket,model=model,**calc(rr,model)))
            for native in range(8):tables['PRIMARY_NATIVE8'].append(dict(group=group,native_corner=native,model=model,**calc([r for r in pp if int(r['native_corner'])==native],model)))
            rr=[r for r in ledger if(group=='ALL'or r['group']==group)and yes(r,'same_slot_pipeline_D_vs_E')and r['human_state']in(D,E)]
            tables['SAME_SLOT'].append(dict(group=group,model=model,**calc(rr,model)))
            for cohort in('ALL_REGISTERED','PAIRED_SAME_QUERY'):
                rr=[r for r in ledger if r['human_state']==S and(group=='ALL'or r['group']==group)and(cohort=='ALL_REGISTERED'or r.get('paired_same_query_RGB_D_vs_ES')=='True')]
                # Older ledger does not contain this flag: report UNKNOWN instead of inferring it.
                if cohort=='PAIRED_SAME_QUERY'and not all('paired_same_query_RGB_D_vs_ES'in r for r in ledger):
                    tables['HUMAN_SELF_BINARY_RESPONSE'].append({'group':group,'model':model,'cohort':cohort,'N':None,'scope':'UNKNOWN: flag absent in saved ledger; no new GT follow'});continue
                cc=Counter(call(r,model,'D_vs_ES')for r in rr)
                tables['HUMAN_SELF_BINARY_RESPONSE'].append(dict(group=group,model=model,cohort=cohort,N=len(rr),**dict(cc),scope='Binary response on humanSELF, not learned SELF or causal label'))
            allrows=[r for r in ledger if group=='ALL'or r['group']==group]
            cc=Counter((lookup[(model,int(r['global_index']))]if model in predictions else r[model+'_D_E_U'])for r in allrows)
            tables['ALL1784_MODEL_INVENTORY'].append(dict(group=group,model=model,N=len(allrows),D=cc['D'],E=cc['E'],U=cc['U']))
        for baseline,new in[('v61',a)for a in predictions]+[(list(predictions)[0],list(predictions)[1])]:
            recovered=damaged=0
            for r in pp:
                truth='VISIBLE'if r['human_state']==D else'ABSENT_NONVISIBLE';a=call(r,baseline,'D_vs_E');b=call(r,new,'D_vs_E')
                recovered+=a!=truth and b==truth;damaged+=a==truth and b!=truth
            tables['PAIRED_RECOVERY_AND_DAMAGE'].append(dict(group=group,baseline=baseline,new=new,N=len(pp),recovered=recovered,damaged=damaged,net=recovered-damaged))
    joined=[]
    for i,r in enumerate(ledger):
        row=dict(r)
        for model in predictions:
            p=predictions[model][i];row[model+'_D_E_U']=p['prediction'];row[model+'_MAIN0_logit']=p['logit'];row[model+'_U_reason']=p['U_reason']
        joined.append(row)
    return {'completed':True,'PASS':True,'primary_N':1237,'primary_D':1147,'primary_E':90,'all_slots':1784,'original_input_U':45,'models':list(models),'tables':tables,'ledger':joined,'paired_SELF_flag_present':all('paired_same_query_RGB_D_vs_ES'in r for r in ledger),'v61_threshold':'Stored fixed presence0.5 unchanged; D is visible, newMAIN0>=0 is E','predictions_or_checkpoint_selected_by_result':False,'fresh_blind_validation':False,'legacy_CAL_and_repeated_exposure_limitations_preserved':True,'new_GT_matching_query_or_native_permutation':0,'PnP_changes':0}
