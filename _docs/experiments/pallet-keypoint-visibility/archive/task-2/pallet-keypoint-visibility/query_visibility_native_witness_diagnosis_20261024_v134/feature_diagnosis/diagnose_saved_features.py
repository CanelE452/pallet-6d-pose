"""Saved native feature retrieval/alignment/variant diagnostics; no fitting.

No Torch, model, checkpoint, image/crop I/O or trainer imports. Every retrieval
metric, match tier and subgroup is pinned before feature/label arrays are read.
"""
import os
for key in('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,collections,hashlib,json,time
import numpy as np

H=Path(__file__).resolve().parent
PROTOCOL=H/'PROTOCOL_BEFORE_SAVED_FEATURE_READS.json'
PROTOCOL_SHA='22da00f73bf2e6dc473d9b791d04d7efb704c3b34aaa70832af53cf590de4e27'
OUT=H/'saved_feature_diagnosis_v1'
BLOCKS=('T','Q','F','C','P')
TIERS=('ALL_TRAIN','CHANNEL','CHANNEL_MATERIAL','CHANNEL_MATERIAL_GEOMETRY','CHANNEL_MATERIAL_FOREIGN','CHANNEL_MATERIAL_GEOMETRY_FOREIGN')
MODELS=('C16','C24','C25')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path'];return Path(b['path'])
def packet(p):
    with np.load(p,allow_pickle=False)as f:return{k:f[k].copy()for k in f.files}
def rows(p):return[json.loads(x)for x in Path(p).read_text().splitlines()if x]
def write(p,d):
    with Path(p).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def stats(values):
    a=np.asarray([x for x in values if x is not None and np.isfinite(x)],np.float64)
    return dict(N=len(a),mean=float(a.mean())if len(a)else None,quantiles_0_25_50_75_90_95_100=np.quantile(a,[0,.25,.5,.75,.9,.95,1]).tolist()if len(a)else None)
def corr(a,b):
    pairs=[(x,y)for x,y in zip(a,b)if x is not None and y is not None and np.isfinite(x)and np.isfinite(y)]
    if len(pairs)<3:return dict(N=len(pairs),pearson=None)
    z=np.asarray(pairs,np.float64)
    return dict(N=len(pairs),pearson=float(np.corrcoef(z.T)[0,1])if z[:,0].std()>0 and z[:,1].std()>0 else None)

def load_data(p):
    b=p['inputs'];fs=read(check(b['feature_seal']));ds=read(check(b['GT_seal']));control=read(check(b['control_complete']))
    assert fs['completed']and fs['PASS']and fs['GT_read']is False
    assert ds['completed']and ds['PASS']and ds['RGB_read']is False
    assert control['completed']and control['PASS']and control['application_solved']is False
    f=packet(check(b['features']));models={m:packet(check(b[m+'_scores']))for m in MODELS}
    assert f['features'].shape==(6144,1928)and f['features'].dtype==np.float32 and np.isfinite(f['features']).all()
    assert f['feature_constructed'].sum()==6088 and f['query_available'].sum()==6056
    gt=rows(check(b['GT_ledger']));assert len(gt)==12288
    gt=gt[::2];assert all(r['convention']=='CF_PRIMARY_DECLARED'for r in gt)
    state=np.array([r['teacher_state']for r in gt]);target=np.where(state=='D',0,np.where(state=='E',1,-1)).astype(np.int8)
    meta=dict(state=state,target=target,split=np.array([r['split']for r in gt]),boundary=np.array([r['correspondence_boundary']for r in gt]),
        bands=np.array([r['position_band']for r in gt]),variant=np.array([r['variant']for r in gt]),global_index=np.arange(6144),
        base=np.arange(6144)//32,scene=np.arange(6144)//8,slot=np.arange(6144)%8,rows=gt)
    for factor in('material','geometry_family','foreign_family','aspect'):meta[factor]=np.array([r['factors'][factor]for r in gt])
    configs=read(check(b['base_configs']))['rows'];assert len(configs)==192
    selected=np.array([c['selected_CF_index']for c in configs],np.int8)
    meta['selected']=meta['slot']==selected[meta['base']]
    for i,r in enumerate(gt):
        assert r['scene_index']==i//8 and r['base_index']==i//32 and r['native_slot']==i%8
        assert r['query_id']==str(f['occurrence_id'][i])
        q=np.asarray(r['query_xy']if r['query_xy']is not None else[np.nan,np.nan])
        assert np.array_equal(q,f['query_xy'][i],equal_nan=True)
    for name,a in models.items():
        assert np.array_equal(a['global_index'],np.arange(6144))and np.array_equal(a['query_available'],f['query_available'])
        for key in('occurrence_id','query_xy','L'):
            assert np.array_equal(a[key],f[key],equal_nan=True)if a[key].dtype.kind in'fc'else np.array_equal(a[key],f[key]),(name,key)
        assert np.array_equal(a['prediction'],np.where(a['query_available'],(a['raw_head_logit_before_U']>=0).astype(np.int8),-1))
    proposals=rows(check(b['initializer_proposals']));assert len(proposals)==768
    context=[]
    for i,r in enumerate(proposals):
        assert r['scene_index']==i
        if r['selected_index']is None:context.append(None);continue
        q8=np.asarray(r['candidates'][r['selected_index']]['keypoints_xy'][:8],np.float64)
        finite=np.isfinite(q8).all(1)&~(q8==-1).all(1)
        context.append(np.r_[q8[finite].min(0),q8[finite].max(0)].tolist()if finite.any()else None)
    ref=np.array([r['reference_xy']for r in gt],np.float64);delta=ref-f['query_xy'];error=np.linalg.norm(delta,axis=1)
    risk=dict(reference_in_full_crop=np.all(np.abs(delta)<=f['L'][:,None]/2,axis=1)&f['feature_constructed'],
              reference_in_central32=np.all(np.abs(delta)<=f['L'][:,None]/6,axis=1)&f['feature_constructed'],
              reference_in_context=np.array([bool(c is not None and np.all(ref[i]>=np.asarray(c)[:2])and np.all(ref[i]<=np.asarray(c)[2:]))for i,c in enumerate([context[i//8]for i in range(6144)])]),
              error=error,context=context)
    for i,r in enumerate(gt):assert (r['error_px']is None and np.isnan(error[i]))or error[i]==r['error_px']
    return f,models,meta,risk

def collision_check(features,meta,available):
    out={};known=np.isin(meta['target'],[0,1])&available
    for fold in('TRAIN','VALIDATION','COMBINED'):
        keep=known if fold=='COMBINED'else known&(meta['split']==fold);groups={}
        for i in np.flatnonzero(keep):groups.setdefault(hashlib.sha256(features[i].tobytes()).hexdigest(),[]).append(int(i))
        mixed=[ids for ids in groups.values()if len(set(meta['state'][ids]))>1]
        out[fold]=dict(known_available=int(keep.sum()),unique_exact1928_F32=len(groups),opposite_DE_groups=len(mixed),
            minimum_deterministic_errors_due_to_collisions=sum(min(np.count_nonzero(meta['target'][ids]==0),np.count_nonzero(meta['target'][ids]==1))for ids in mixed),
            examples=mixed[:12],all_examples_listed=len(mixed)<=12)
    return out

def retrieval(features,meta,available):
    raw=features[:,:1920].astype(np.float64).reshape(6144,5,384)
    norms=np.linalg.norm(raw,axis=2);blocks=raw/np.maximum(norms[:,:,None],1e-12)
    refs=np.flatnonzero((meta['split']=='TRAIN')&available&(meta['target']>=0));assert np.bincount(meta['target'][refs]).tolist()==[1869,131]
    queries=np.flatnonzero(available&(meta['target']>=0));out={}
    for lo in range(0,len(queries),64):
        qq=queries[lo:lo+64]
        cosine=np.stack([blocks[qq,b]@blocks[refs,b].T for b in range(5)],axis=1);averages=cosine.mean(1)
        for rowno,q in enumerate(qq):
            baseok=meta['base'][refs]!=meta['base'][q]
            channel=meta['slot'][refs]==meta['slot'][q];material=meta['material'][refs]==meta['material'][q]
            geometry=meta['geometry_family'][refs]==meta['geometry_family'][q];foreign=meta['foreign_family'][refs]==meta['foreign_family'][q]
            masklist=(baseok,baseok&channel,baseok&channel&material,baseok&channel&material&geometry,
                      baseok&channel&material&foreign,baseok&channel&material&geometry&foreign)
            per={}
            for tier,keep in zip(TIERS,masklist):
                item={};bestglobal={};bestclass={};classmax={};blockmax={}
                for t,name in((0,'D'),(1,'E')):
                    ii=np.flatnonzero(keep&(meta['target'][refs]==t));item[name+'_reference_N']=len(ii)
                    if not len(ii):item[name+'_nearest']=None;item[name+'_top5_mean']=None;blockmax[name]=None;continue
                    vals=averages[rowno,ii];winner=ii[int(np.argmax(vals))];gi=int(refs[winner]);mx=float(vals.max());k=min(5,len(vals))
                    item[name+'_nearest']=dict(global_index=gi,base_index=int(meta['base'][gi]),native_slot=int(meta['slot'][gi]),mean5_cosine=mx)
                    item[name+'_top5_mean']=float(np.partition(vals,len(vals)-k)[-k:].mean())
                    bestglobal[name]=gi;bestclass[name]=t;classmax[name]=mx;blockmax[name]=cosine[rowno,:,ii].max(0).tolist()
                item['both_reference_classes']=len(classmax)==2
                if len(classmax)==2:
                    item['E_minus_D_maxcos']=classmax['E']-classmax['D']
                    item['E_minus_D_top5']=item['E_top5_mean']-item['D_top5_mean']
                    item['block_E_minus_D_maxcos']=(np.array(blockmax['E'])-np.array(blockmax['D'])).tolist()
                    winner=min(('D','E'),key=lambda n:(-classmax[n],bestglobal[n]));item['nearest_reference_class']=winner
                else:item.update(E_minus_D_maxcos=None,E_minus_D_top5=None,block_E_minus_D_maxcos=None,nearest_reference_class='UNSUPPORTED')
                per[tier]=item
            out[int(q)]=per
    norm_summary={b:stats(norms[available,i])for i,b in enumerate(BLOCKS)}
    return out,blocks,norm_summary

def cohorts(meta,models):
    folds=meta['split'];va=folds=='VALIDATION';nb=~meta['boundary'];small=meta['bands']=='LE3_PX';wood=meta['foreign_family']=='SAME_TARGET_MATERIAL'
    out={'TRAIN':folds=='TRAIN','VALIDATION':va,'VALIDATION_NONBOUNDARY':va&nb,'VALIDATION_BOUNDARY':va&~nb,
        'VALIDATION_LE3_NONBOUNDARY':va&small&nb,'VALIDATION_SAME_MATERIAL':va&wood,'VALIDATION_SAME_MATERIAL_NONBOUNDARY':va&wood&nb,
        'VALIDATION_SAME_MATERIAL_LE3_NONBOUNDARY':va&wood&small&nb,'VALIDATION_RENDER_SELECTED_NATIVE':va&meta['selected'],
        'VALIDATION_OTHER_NATIVE':va&~meta['selected']}
    for name in('variant','material','geometry_family','foreign_family','bands','slot'):
        for value in np.unique(meta[name][va]):out['VALIDATION::'+name+'::'+str(value)]=va&(meta[name]==value)
    for v in('BEFORE','COVER','NEAR_SHAM','FAR_SHAM'):
        out['VALIDATION::'+v+'::SELECTED']=va&(meta['variant']==v)&meta['selected']
        out['VALIDATION::'+v+'::OTHER']=va&(meta['variant']==v)&~meta['selected']
    for model in MODELS:
        for scope,mask in(('VALIDATION',va),('VAL_NONBOUNDARY',va&nb),('VAL_NONBOUNDARY_LE3',va&nb&small)):
            for state,t in(('D',0),('E',1)):
                for pred,label in((0,'D'),(1,'E'),(-1,'U')):out[scope+'::'+model+'::GT_'+state+'_PRED_'+label]=mask&(meta['target']==t)&(models[model]['prediction']==pred)
    return out

def cohort_summary(indices,nearest,features,models,meta,risk):
    ix=np.asarray(indices,np.int64);states=meta['state'][ix];known=ix[meta['target'][ix]>=0]
    result=dict(all_original_rows=len(ix),teacher_states={str(k):int(v)for k,v in collections.Counter(states).items()},
        bases=len(np.unique(meta['base'][ix])),model_availability=int(features['query_available'][ix].sum()),known_DE=len(known),
        known_input_U=int((~features['query_available'][known]).sum()),D_reference_alignment_risk={
            name:int(np.count_nonzero((meta['state'][ix]=='D')&~risk[name][ix]))for name in('reference_in_full_crop','reference_in_central32','reference_in_context')},
        error_px=stats(risk['error'][known]),models={})
    for model in MODELS:
        pred=models[model]['prediction'];cm=[[int(np.count_nonzero((meta['target'][ix]==t)&(pred[ix]==p)))for p in(0,1,-1)]for t in(0,1)]
        result['models'][model]=dict(CM_D_E_rows_D_E_U_cols=cm,raw_hidden_logit=stats(models[model]['raw_head_logit_before_U'][known]))
    result['retrieval']={}
    for tier in TIERS:
        available=[int(i)for i in known if int(i)in nearest];good=[i for i in available if nearest[i][tier]['both_reference_classes']]
        cm=np.zeros((2,3),np.int64)
        for i in known:
            n=nearest.get(int(i),{}).get(tier,{}).get('nearest_reference_class','UNSUPPORTED');cm[meta['target'][i],{'D':0,'E':1,'UNSUPPORTED':2}[n]]+=1
        entry=dict(CM_targetD_E_cols_nearestD_E_noComparison=cm.tolist(),known_original_denominator=len(known),
            queried=len(available),both_reference_classes=len(good),missing_match_class=len(available)-len(good),
            poolD_N=stats([nearest[i][tier]['D_reference_N']for i in available]),poolE_N=stats([nearest[i][tier]['E_reference_N']for i in available]))
        for state,t in(('D',0),('E',1)):
            selected=[i for i in good if meta['target'][i]==t]
            entry['GT_'+state]=dict(N=len(selected),E_minus_D_maxcos=stats([nearest[i][tier]['E_minus_D_maxcos']for i in selected]),
                E_minus_D_top5=stats([nearest[i][tier]['E_minus_D_top5']for i in selected]),
                block_E_minus_D={b:stats([nearest[i][tier]['block_E_minus_D_maxcos'][j]for i in selected])for j,b in enumerate(BLOCKS)})
        result['retrieval'][tier]=entry
    return result

def pair_diagnosis(features,blocks,models,meta,risk):
    pairs=[]
    for base in range(192):
        for slot in range(8):
            before=base*32+slot
            for variant in range(1,4):
                after=(base*4+variant)*8+slot;constructed=bool(features['feature_constructed'][before]and features['feature_constructed'][after])
                av=bool(features['query_available'][before]and features['query_available'][after])
                cos=np.sum(blocks[before]*blocks[after],axis=1)if constructed else None
                qdelta=float(np.linalg.norm(features['query_xy'][after]-features['query_xy'][before]))if constructed else None
                cc1=risk['context'][before//8];cc2=risk['context'][after//8]
                row=dict(base_index=base,native_slot=slot,before_index=before,after_index=after,split=str(meta['split'][before]),
                    variant=str(meta['variant'][after]),before_state=str(meta['state'][before]),after_state=str(meta['state'][after]),
                    boundary=bool(meta['boundary'][before]),renderer_selected_native=bool(meta['selected'][before]),
                    foreign_family=str(meta['foreign_family'][before]),material=str(meta['material'][before]),both_constructed=constructed,both_available=av,
                    q_movement_px=qdelta,q_exact_unchanged=bool(constructed and np.array_equal(features['query_xy'][before],features['query_xy'][after])),
                    L_ratio_after_before=float(features['L'][after]/features['L'][before])if constructed else None,
                    context_max_abs_movement=float(np.max(np.abs(np.asarray(cc1)-np.asarray(cc2))))if cc1 is not None and cc2 is not None else None,
                    block_cosine=cos.tolist()if cos is not None else None,mean5_cosine=float(cos.mean())if cos is not None else None,
                    before_error_px=meta['rows'][before]['error_px'],after_error_px=meta['rows'][after]['error_px'],scores={})
                for model in MODELS:
                    a=models[model];z0=a['raw_head_logit_before_U'][before];z1=a['raw_head_logit_before_U'][after]
                    row['scores'][model]=dict(before_raw=float(z0)if np.isfinite(z0)else None,after_raw=float(z1)if np.isfinite(z1)else None,
                        delta=float(z1-z0)if np.isfinite(z0)and np.isfinite(z1)else None,
                        before_pred=int(a['prediction'][before]),after_pred=int(a['prediction'][after]))
                pairs.append(row)
    assert len(pairs)==4608
    summary={}
    for fold in('ALL','TRAIN','VALIDATION','VAL_NONBOUNDARY'):
        foldrows=[r for r in pairs if fold=='ALL'or r['split']==fold or(fold=='VAL_NONBOUNDARY'and r['split']=='VALIDATION'and not r['boundary'])]
        for variant in('COVER','NEAR_SHAM','FAR_SHAM'):
            vr=[r for r in foldrows if r['variant']==variant]
            for transition in('ALL','D_TO_E','D_TO_D'):
                sr=vr if transition=='ALL'else[r for r in vr if r['before_state']=='D'and r['after_state']==transition[-1]]
                for family in('ALL','SAME_TARGET_MATERIAL'):
                    sub=sr if family=='ALL'else[r for r in sr if r['foreign_family']==family]
                    key='::'.join((fold,variant,transition,family))
                    summary[key]=dict(original_pairs=len(sub),both_available=sum(r['both_available']for r in sub),
                        both_constructed=sum(r['both_constructed']for r in sub),q_exact_unchanged=sum(r['q_exact_unchanged']for r in sub),
                        q_movement_px=stats([r['q_movement_px']for r in sub]),context_movement=stats([r['context_max_abs_movement']for r in sub]),
                        mean5_cosine=stats([r['mean5_cosine']for r in sub]),
                        block_cosine={b:stats([r['block_cosine'][i]if r['block_cosine']is not None else None for r in sub])for i,b in enumerate(BLOCKS)},
                        models={m:dict(score_delta=stats([r['scores'][m]['delta']for r in sub]),
                            hidden_score_increased=sum(r['scores'][m]['delta']is not None and r['scores'][m]['delta']>0 for r in sub),
                            before_after_both_correct=sum(r['both_available']and r['before_state']in'DE'and r['after_state']in'DE'and r['scores'][m]['before_pred']==(r['before_state']=='E')and r['scores'][m]['after_pred']==(r['after_state']=='E')for r in sub),
                            delta_vs_qmovement=corr([r['scores'][m]['delta']for r in sub],[r['q_movement_px']for r in sub]))for m in MODELS},
                        C24_C25_delta_correlation=corr([r['scores']['C24']['delta']for r in sub],[r['scores']['C25']['delta']for r in sub]))
    return pairs,summary

def run():
    assert sha(PROTOCOL)==PROTOCOL_SHA and not OUT.exists()
    p=read(PROTOCOL);assert p['source_only']and p['frozen_before_feature_label_or_score_array_payloads']
    started=time.monotonic();features,models,meta,risk=load_data(p)
    collisions=collision_check(features['features'],meta,features['query_available'])
    nearest,blocks,norms=retrieval(features['features'],meta,features['query_available'])
    groups=cohorts(meta,models);summaries={name:cohort_summary(np.flatnonzero(mask),nearest,features,models,meta,risk)for name,mask in groups.items()}
    pairs,pairsummary=pair_diagnosis(features,blocks,models,meta,risk)
    perbase=[]
    for base in range(128,192):
        mask=meta['base']==base;ix=np.flatnonzero(mask)
        perbase.append(dict(base_index=base,all32_original_slots=len(ix),teacher_states=dict(collections.Counter(meta['state'][ix])),
            available=int(features['query_available'][ix].sum()),known=int(np.isin(meta['target'][ix],[0,1]).sum()),
            renderer_selected_native_channel=int(meta['slot'][ix[meta['selected'][ix]]][0]),
            boundary=bool(meta['boundary'][ix][0]),models={m:
                [[int(np.count_nonzero((meta['target'][ix]==t)&(models[m]['prediction'][ix]==z)))for z in(0,1,-1)]for t in(0,1)]for m in MODELS}))
    assert len(perbase)==64 and sum(r['all32_original_slots']for r in perbase)==2048
    OUT.mkdir()
    with(OUT/'ALL6144_QUERY_DIAGNOSTICS.jsonl').open('x')as f:
        for i in range(6144):
            row=dict(global_index=i,base_index=int(meta['base'][i]),scene_index=int(meta['scene'][i]),native_slot=int(meta['slot'][i]),
                split=str(meta['split'][i]),variant=str(meta['variant'][i]),raw_state=str(meta['state'][i]),convention_boundary=bool(meta['boundary'][i]),
                renderer_selected_native=bool(meta['selected'][i]),position_band=str(meta['bands'][i]),error_px=meta['rows'][i]['error_px'],
                feature_constructed=bool(features['feature_constructed'][i]),query_available=bool(features['query_available'][i]),
                reference_in_full_crop=bool(risk['reference_in_full_crop'][i]),reference_in_central32=bool(risk['reference_in_central32'][i]),reference_in_context=bool(risk['reference_in_context'][i]),
                models={m:dict(pred=int(models[m]['prediction'][i]),raw_hidden_logit=float(models[m]['raw_head_logit_before_U'][i])if np.isfinite(models[m]['raw_head_logit_before_U'][i])else None)for m in MODELS},
                retrieval=nearest.get(i),no_retrieval_reason=None if i in nearest else'UNAVAILABLE_INPUT_OR_TEACHER_SU')
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    with(OUT/'ALL4608_ACTUALQ_VARIANT_PAIRS.jsonl').open('x')as f:
        for row in pairs:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    write(OUT/'ALL64_VAL_BASES_PRESERVED.json',dict(completed=True,rows=perbase,all64_kept=True))
    result=dict(completed=True,PASS=True,goal_status='active',fully_solved=False,protocol=bind(PROTOCOL),code=bind(__file__),
        original_slots=6144,TRAIN_bases=128,VAL_bases=64,all64_VAL_bases_kept=True,reference_TRAIN_D=1869,reference_TRAIN_E=131,
        feature_normalization='Fixed perblock L2 only, no fittedcalibration; first exactmax tieglobalindex',block_norms=norms,
        collision_check=collisions,cohorts=summaries,pair_summary=pairsummary,
        no_training_model_RGB_checkpoint_private_V112_PnP=0,seconds=time.monotonic()-started,
        limitations_KO=['가장 가까운 TRAIN 라벨의 검색은 진단이며 새 classifier 적합·문턱 선택·채택 결과가 아니다.',
            'D1869/E131 reference 수가 다르므로 최대 유사도와 최근접 클래스에는 표본수 편향이 있다. 작은 조건부 pool과 결측을 모두 보고한다.',
            'TRAIN 검색은 같은 base 전체를 제외한다. VAL64개 base와 S/U·입력무효 슬롯은 원분모에 남긴다.',
            '특징 충돌 0 또는 최근접 검색 실패는 비분리성·정보 소실의 증명이 아니다. 작은 거리 차이도 head가 활용할 수 있다.',
            'BEFORE/COVER/SHAM의 실제 초기 q·L·context는 달라질 수 있어 feature/score 차이를 가림의 단독 인과 효과로 해석하지 않는다.',
            'r가 crop 밖인 위치 위험을 표시해도 semantic 보임 라벨을 E로 바꾸지 않는다. RGB 지지 영역과 실제 가림체는 이번에 열람하지 않았다.'])
    write(OUT/'RESULTS.json',result)
    write(OUT/'DIAGNOSIS_SEAL.json',dict(completed=True,PASS=True,results=bind(OUT/'RESULTS.json'),
        outputs=[bind(x)for x in sorted(OUT.iterdir())if x.is_file()],protocol=bind(PROTOCOL),code=bind(__file__),RGB_read=False,model_fit=False,source_only=True))
    print(json.dumps(dict(completed=True,PASS=True,seal=bind(OUT/'DIAGNOSIS_SEAL.json'),seconds=result['seconds'],
        collisions=collisions,VAL_nonboundary=summaries['VALIDATION_NONBOUNDARY']['retrieval']['ALL_TRAIN'])),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.run:run()
    else:raise SystemExit('Use --run for pinned saved-feature-only diagnostics.')
