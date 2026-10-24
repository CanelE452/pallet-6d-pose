"""Saved SOURCE-only query-boundary diagnosis; no features/weights/inference.

Reads source GT masks/certificates, sealed label-only caches and saved source
outputs. Every current-evaluation path, input feature cache and model is unused.
"""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,time
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt,binary_erosion,maximum_filter

ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'source_audit';STEM='SOURCE_BOUNDARY_FAILURE_DIAGNOSIS'
MODELS=('V65_BASELINE','CONTROLLED_ONLY','CONTROLLED_PLUS_PALLOX')
SOURCES=('ORIGINAL_YCB','CONTROLLED_YCB')
FIELDS=('sample_id','source','global_id','image_sha256','visible_probability','joint_probability',
        'predicted_state','raw_SAM_visible_probability','center_native_valid','center_allowed','GT_state')
CLASSES=('V','E','O','U');PAIRS=np.array(((0,0),(1,0),(1,1),(2,0),(2,1),(2,2)),np.int8)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def bind(b):assert sha(b['path'])==b['sha256'],b['path']
def write(p,j):Path(p).write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')
def rows(p):return [json.loads(l)for l in Path(p).open()]
def ratio(n,d):return float(n)/int(d)if d else None
def csvwrite(path,records):
    if not records:return
    with Path(path).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
def npz(p,keys):
    with np.load(p,allow_pickle=False)as z:return {k:z[k]for k in keys}
def distance(mask):return distance_transform_edt(np.pad(mask,1,constant_values=False))[1:-1,1:-1]
def d_bin(d):return '(0,1]'if d<=1 else'(1,2]'if d<=2 else'(2,3]'if d<=3 else'>3'


def load_saved_outputs():
    outputs={};bindings=[];canonical=None
    for name in MODELS:
        folder=ROOT/'source_evaluation'/name;report=json.loads((folder/'SOURCE_METRICS.json').read_text())
        assert report['completed']and report['source_only']and not report['current223_RGB_GT_scores_read']
        registered={str(Path(b['path']).resolve()):b for b in report['outputs']}
        for filename in ['SOURCE_OUTPUTS.npz','PER_QUERY.csv']:
            p=folder/filename;bind(registered[str(p.resolve())]);bindings.append(ref(p))
        bindings.append(ref(folder/'SOURCE_METRICS.json'))
        d=npz(folder/'SOURCE_OUTPUTS.npz',FIELDS);assert len(d['sample_id'])==8590
        ids=[str(s)for s in d['sample_id']];assert len(set(ids))==len(ids)
        lookup={s:i for i,s in enumerate(ids)}
        if canonical is None:canonical=ids
        assert set(ids)==set(canonical)
        order=np.array([lookup[s]for s in canonical]);d={k:v[order]for k,v in d.items()}
        assert np.isfinite(d['visible_probability']).all()and np.isfinite(d['joint_probability']).all()
        p=d['joint_probability'];assert np.allclose(p.sum(1),1,atol=1e-12)
        assert np.array_equal(d['visible_probability'],np.stack([p[:,0],p[:,0]+p[:,1]+p[:,3]],1))
        expected=np.where((p.max(1)>=.5)[:,None],PAIRS[p.argmax(1)],3)
        assert np.array_equal(d['predicted_state'],expected)
        saved={r['sample_id']:r for r in csv.DictReader((folder/'PER_QUERY.csv').open())};assert set(saved)==set(canonical)
        for i,sid in enumerate(canonical):
            r=saved[sid];assert int(r['exact_GT'])==d['GT_state'][i,0]and int(r['radius2_GT'])==d['GT_state'][i,1]
            for k,scale in enumerate(['exact','radius2']):
                assert float(r[scale+'_pV'])==d['visible_probability'][i,k]
                assert float(r[scale+'_rawSAM_pV'])==d['raw_SAM_visible_probability'][i,k]
        if outputs:
            first=outputs[MODELS[0]]
            for key in ['sample_id','source','global_id','image_sha256','center_native_valid','center_allowed','GT_state','raw_SAM_visible_probability']:
                assert np.array_equal(first[key],d[key]),(name,key)
        outputs[name]=d
    return outputs,bindings


def visible_summary(records):
    out=[]
    for source in SOURCES:
        vv=[r for r in records if r['source']==source and r['native_valid']and r['exact_GT']==0]
        for label,predicate in [('ALL_KNOWN_V',lambda r:True)]+[(b,lambda r,b=b:r['visible_distance_bin']==b)for b in ['(0,1]','(1,2]','(2,3]','>3']]+[
          ('STRICT25_V_INTERIOR',lambda r:r['strict25_V']),('NONSTRICT25_V_BOUNDARY',lambda r:not r['strict25_V'])]:
            selected=[r for r in vv if predicate(r)]
            for model in MODELS+('RAW_SAM_STRICT_ZERO',):
                x=[r for r in selected if r['model']==(MODELS[0]if model=='RAW_SAM_STRICT_ZERO'else model)]
                if model=='RAW_SAM_STRICT_ZERO':tp=sum(r['rawSAM_exact_positive']for r in x);r2=sum(r['rawSAM_r2_positive']for r in x)
                else:tp=sum(r['head_exact_visible']for r in x);r2=sum(r['head_r2_visible']for r in x)
                n=len(x);counts=Counter((r['head_exact_visible'],r['rawSAM_exact_positive'])for r in x)
                out.append(dict(source=source,GT_scope='KNOWN_EXACT_VISIBLE_NATIVE_QUERY',stratum=label,model=model,N=n,targets=len({r['global_id']for r in x}),
                  TP_exact=tp,FN_exact=n-tp,recall_exact=ratio(tp,n),false_absence_rate_exact=ratio(n-tp,n),
                  radius2_TP_diagnostic=r2,radius2_FN_diagnostic=n-r2,radius2_recall_diagnostic=ratio(r2,n),
                  exact_FN_but_radius2_positive=sum((not r['head_exact_visible'])and r['head_r2_visible']for r in x)if model!='RAW_SAM_STRICT_ZERO'else sum((not r['rawSAM_exact_positive'])and r['rawSAM_r2_positive']for r in x),
                  head_V_and_rawSAM_V=counts[True,True]if model!='RAW_SAM_STRICT_ZERO'else'',
                  head_V_rawSAM_negative=counts[True,False]if model!='RAW_SAM_STRICT_ZERO'else'',
                  head_FN_rawSAM_positive=counts[False,True]if model!='RAW_SAM_STRICT_ZERO'else'',
                  head_FN_rawSAM_negative=counts[False,False]if model!='RAW_SAM_STRICT_ZERO'else''))
    return out


def confusion(records):
    out=[];e_recall=[]
    for source in SOURCES:
        for model in MODELS:
            allrows=[r for r in records if r['source']==source and r['model']==model and r['native_valid']]
            for scale in ['exact','radius2']:
                contexts=['ALL']+(['STRICT25_E_CORE','E_INTERSECTING_5X5_BAND','NO_E_IN_5X5']if source=='CONTROLLED_YCB'else[])
                for context in contexts:
                    rr=[r for r in allrows if context=='ALL'or r['E_context']==context]
                    known=[r for r in rr if r[scale+'_GT']>=0];table=np.zeros((3,4),np.int64)
                    for r in known:table[r[scale+'_GT'],r[scale+'_predicted']]+=1
                    row=dict(source=source,model=model,support=scale,stratum=context,total_native_valid=len(rr),known_GT=len(known),UNKNOWN_GT=len(rr)-len(known),
                      E_TP=int(table[1,1]),E_predicted_on_known_GT=int(table[:,1].sum()),E_GT=int(table[1].sum()),
                      E_precision_conditional_known=ratio(table[1,1],table[:,1].sum()),E_recall_all_known_E=ratio(table[1,1],table[1].sum()),
                      U_on_known_GT=int(table[:,3].sum()),unknown_predicted_E=sum(r[scale+'_GT']<0 and r[scale+'_predicted']==1 for r in rr),
                      precision_caveat='STRICT25_E_CORE is positive-only by construction; its precision is not a discriminative E precision. ALL precision uses allknownV/E/O.')
                    for g in range(3):
                        for pred in range(4):row['GT_'+CLASSES[g]+'_pred_'+CLASSES[pred]]=int(table[g,pred])
                    out.append(row)
            for stratum in ['ALL_KNOWN_EXACT_E','STRICT25_E_CORE','NONSTRICT25_E_BOUNDARY']:
                rr=[r for r in allrows if r['exact_GT']==1 and(stratum=='ALL_KNOWN_EXACT_E'or r['strict25_E']==(stratum=='STRICT25_E_CORE'))]
                n=len(rr);pred=Counter(r['exact_predicted']for r in rr)
                e_recall.append(dict(source=source,model=model,stratum=stratum,N=n,pred_V=pred[0],pred_E=pred[1],pred_O=pred[2],pred_U=pred[3],
                  E_recall_all_GT=ratio(pred[1],n),binary_absent_nonvisible=sum(not r['head_exact_visible']for r in rr),
                  rawSAM_absent_nonvisible=sum(not r['rawSAM_exact_positive']for r in rr),
                  binary_negative_is_not_E_class=True))
    return out,e_recall


def main():
    assert not(HERE/(STEM+'.json')).exists(),'Append-only source diagnosis already exists'
    cv2.setNumThreads(1);start=time.monotonic()
    recipe=dict(schema='V66_SAVED_SOURCE_BOUNDARY_DIAGNOSIS_RECIPE',source_only=True,models=list(MODELS),sources=list(SOURCES),
      visible_masks='ORIGINAL_YCB=original visible; CONTROLLED_YCB=original visible AND NOT opaque paste, fromsealed canonical evidence',
      native_position='Saved unchanged q8 nearest floor(q+.5), continuous androunded native bounds, no jitter/recentering',
      distance='SciPy exactEuclidean distance_transform_edt on visible mask paddedonefalsepixel; distance measured nativequerypixelcenter to nearest masknegative/outsideframe center',
      bins=['(0,1]','(1,2]','(2,3]','>3'],strict25='All actual25native5x5 pixels visible/newE with constant-false border; separate overlapping diagnostic from distancebins',
      primary='Known exactV only; headpV>=.5 TP/FN. SavedrawSAMpositive iff savedprobability>.5 (strictnative logit>0).',
      radius2='Separate learned neighborhood-support diagnostic; neverreplaces primary exactquery task',
      E='Source controlled newE only. Confident jointMAP>=.5 E classification,3x4 known V/E/O by V/E/O/U; UNKNOWN excludedfromFP/FN andreportedseparately. Core/boundary E recall; discriminative precision only ALLknown denominator.',
      interpretation='Boundary-knownV falseabsence remains an error under exactpixelGT. Stratification measures boundary sensitivity; interiorFNs are unsupported by boundary-only explanation. No private semanticcorner/pixel task equivalence is inferred.',
      forbidden=['current223RGB/GT/scores/inputs','sourceinputfeatures','weights/checkpoints','newinference','training','PnP','threshold/radius/modelselection'])
    recipepath=HERE/(STEM+'_RECIPE.json');assert not recipepath.exists();write(recipepath,recipe)
    outputs,bindings=load_saved_outputs();first=outputs[MODELS[0]];index={str(s):i for i,s in enumerate(first['sample_id'])}
    tpath=ROOT/'provenance/TARGETS.jsonl';cpath=ROOT/'controlled/COMPOSITES.jsonl'
    targets={r['id']:r for r in rows(tpath)if r['split']=='VALIDATION'}
    composites={r['id']:r for r in rows(cpath)if r['split']=='VALIDATION'};assert len(targets)==len(composites)==400
    bindings.extend([ref(tpath),ref(cpath),ref(ROOT/'controlled/SEAL.json'),ref(ROOT/'label_cache_v2/SEAL.json')])
    data=[];native_pixels=0;mask_bindings=[]
    for source,entries,labelname in [('ORIGINAL_YCB',targets,'ORIGINAL_YCB_VAL.npz'),('CONTROLLED_YCB',composites,'CONTROLLED_VAL.npz')]:
        labelpath=ROOT/'label_cache_v2'/labelname;bindings.append(ref(labelpath))
        labels=npz(labelpath,['sample_id','global_id','target_index','query','side','image_sha256','dense_allowed','dense_valid','dense_exact_state','dense_radius2_state'])
        lid={str(s):i for i,s in enumerate(labels['sample_id'])};assert len(lid)==3200
        for ident,record in entries.items():
            if source=='ORIGINAL_YCB':
                target=record;bind(target['visible']);mask_bindings.append(target['visible'])
                raw=cv2.imread(target['visible']['path'],cv2.IMREAD_UNCHANGED);assert set(np.unique(raw))<={0,255};visible=raw>0;external=np.zeros_like(visible)
                sampleids=target['original_sample_ids'];imagesha=target['image']['sha256']
            else:
                target=record['target_original'];bind(record['labels']);mask_bindings.append(record['labels'])
                cert=npz(record['labels']['path'],['original_visible','paste_mask','visible_positive','full_positive','visible_negative'])
                visible=cert['visible_positive'];external=cert['full_positive']&cert['visible_negative']
                assert np.array_equal(visible,cert['original_visible']&~cert['paste_mask'])
                assert np.array_equal(external,cert['original_visible']&cert['paste_mask'])
                sampleids=[ident+':'+str(k)for k in range(8)];imagesha=record['image']['sha256']
            assert visible.shape==external.shape==(480,640);native_pixels+=visible.size
            dist=distance(visible);edist=distance(external)if external.any()else np.zeros(visible.shape)
            vcore=binary_erosion(visible,np.ones((5,5),bool),border_value=0);ecore=binary_erosion(external,np.ones((5,5),bool),border_value=0)
            enear=maximum_filter(external.astype(np.uint8),size=5,mode='constant',cval=0)>0
            for k,sid in enumerate(sampleids):
                li=lid[sid];oi=index[sid];q=np.asarray(target['query'][k],float);px,py=np.floor(q+.5).astype(int)
                valid=bool(0<=q[0]<640 and 0<=q[1]<480 and 0<=px<640 and 0<=py<480)
                assert valid==bool(labels['dense_valid'][li,4,4])==bool(first['center_native_valid'][oi])
                assert np.array_equal(labels['query'][li],q)and labels['side'][li]==target['side']
                gt=[int(labels[key][li,4,4])for key in ['dense_exact_state','dense_radius2_state']]
                assert np.array_equal(gt,first['GT_state'][oi])and int(labels['dense_allowed'][li,4,4])==int(first['center_allowed'][oi])
                assert str(first['global_id'][oi])==ident and str(first['source'][oi])==source and str(first['image_sha256'][oi])==imagesha
                if valid:
                    if gt[0]==0:assert visible[py,px]and dist[py,px]>0
                    if gt[0]==1:assert source=='CONTROLLED_YCB'and external[py,px]
                    vd=float(dist[py,px]);ed=float(edist[py,px]);vc=bool(vcore[py,px]);ec=bool(ecore[py,px])
                    context='STRICT25_E_CORE'if ec else'E_INTERSECTING_5X5_BAND'if enear[py,px]else'NO_E_IN_5X5'
                else:vd=ed=0.;vc=ec=False;context='INVALID'
                original_sid=target['original_sample_ids'][k]
                base=dict(source=source,sample_id=sid,original_sample_id=original_sid,global_id=ident,image_sha256=imagesha,
                  native_corner=k,query_x=float(q[0]),query_y=float(q[1]),pixel_x=int(px),pixel_y=int(py),query_subpixel_dx=float(q[0]-px),query_subpixel_dy=float(q[1]-py),roi_side=float(target['side']),
                  native_valid=valid,exact_GT=gt[0],radius2_GT=gt[1],allowed=int(first['center_allowed'][oi]),
                  visible_inside_distance_px=vd,visible_distance_bin=d_bin(vd)if valid and gt[0]==0 else'NOT_KNOWN_VISIBLE',
                  strict25_V=vc,controlled_E_inside_distance_px=ed,strict25_E=ec,E_context=context)
                for model,d in outputs.items():
                    pv=d['visible_probability'][oi];sam=d['raw_SAM_visible_probability'][oi];pred=d['predicted_state'][oi]
                    data.append(dict(**base,model=model,exact_pV=float(pv[0]),radius2_pV=float(pv[1]),
                      head_exact_visible=bool(pv[0]>=.5),head_r2_visible=bool(pv[1]>=.5),
                      exact_predicted=int(pred[0]),radius2_predicted=int(pred[1]),rawSAM_exact_pV=float(sam[0]),rawSAM_r2_pV=float(sam[1]),
                      rawSAM_exact_positive=bool(sam[0]>.5),rawSAM_r2_positive=bool(sam[1]>.5)))
    assert len(data)==19200
    vv=visible_summary(data);cm,ee=confusion(data);matched=[]
    lookup={(r['model'],r['source'],r['original_sample_id']):r for r in data}
    for model in MODELS:
        for stratum in ['ALL_ORIGINAL_V_TO_CONTROLLED_E','STRICT25_CONTROLLED_E','NONSTRICT25_CONTROLLED_E']:
            pairs=[]
            for r in data:
                if r['model']!=model or r['source']!='CONTROLLED_YCB'or r['exact_GT']!=1 or not r['native_valid']:continue
                original=lookup[(model,'ORIGINAL_YCB',r['original_sample_id'])]
                assert original['exact_GT']==0 and original['native_valid']and original['query_x']==r['query_x']and original['query_y']==r['query_y']
                if stratum!='ALL_ORIGINAL_V_TO_CONTROLLED_E'and r['strict25_E']!=(stratum=='STRICT25_CONTROLLED_E'):continue
                pairs.append((original,r))
            n=len(pairs);matched.append(dict(model=model,stratum=stratum,N=n,both_binary_states_correct=sum(a['head_exact_visible']and not b['head_exact_visible']for a,b in pairs),
              original_V_binary_recall=ratio(sum(a['head_exact_visible']for a,b in pairs),n),controlled_E_binary_absence_recall=ratio(sum(not b['head_exact_visible']for a,b in pairs),n),
              both_confident_V_E_correct=sum(a['exact_predicted']==0 and b['exact_predicted']==1 for a,b in pairs),
              pV_correct_drop=sum(a['exact_pV']>b['exact_pV']for a,b in pairs),mean_pV_drop=float(np.mean([a['exact_pV']-b['exact_pV']for a,b in pairs]))if pairs else None,
              rawSAM_both_binary_states_correct=sum(a['rawSAM_exact_positive']and not b['rawSAM_exact_positive']for a,b in pairs)))
    csvwrite(HERE/(STEM+'_VISIBLE_DISTANCE.csv'),vv);csvwrite(HERE/(STEM+'_QUERY_ROWS.csv'),data)
    csvwrite(HERE/(STEM+'_SOURCE_STATE_CONFUSION.csv'),cm);csvwrite(HERE/(STEM+'_EXTERNAL_CORE_RECALL.csv'),ee);csvwrite(HERE/(STEM+'_MATCHED_INTERVENTION.csv'),matched)
    for b in bindings+mask_bindings:bind(b)
    modelscope=Counter((r['source'],r['exact_GT'])for r in data if r['model']==MODELS[0]and r['native_valid'])
    report=dict(schema='V66_SAVED_SOURCE_BOUNDARY_FAILURE_DIAGNOSIS',completed=True,recipe=ref(recipepath),code=ref(__file__),
      source_targets=800,query_ids=6400,model_query_rows=len(data),native_mask_pixels_processed=native_pixels,
      GT_counts=[dict(source=s,exact_GT=g,N=n)for(s,g),n in sorted(modelscope.items())],
      visible_distance=vv,source_state_confusion=cm,controlled_external_core_recall=ee,matched_intervention=matched,
      note='Known visible boundary FNs are still model errors for exactpixelpresence. Interior/strict25FNs demonstrate failures notexplained solely by nativeboundary sampling. No current semanticcorner ownership inference is made.',
      E_precision='ALLknownV/E/O denominator only is discriminative; strictcore is positive-only. UNKNOWN excludedfromFP/FN andreportedseparately.',
      rawSAM_rule='Saved rawSAMpV>.5 equals native logit>0; equality isnegative. It is a visible-only comparator, never an E/O mechanism classifier.',
      frozen_models_and_thresholds_unchanged=True,new_model_inference_training_PnP_GPU_calls=0,
      current_RGB_GT_scores_inputs_read=0,source_RGB_input_features_weights_read=0,
      source_bindings=bindings,source_mask_binding_count=len(mask_bindings),source_mask_bindings=mask_bindings,
      output_bindings=[ref(HERE/(STEM+suffix))for suffix in ['_VISIBLE_DISTANCE.csv','_QUERY_ROWS.csv','_SOURCE_STATE_CONFUSION.csv','_EXTERNAL_CORE_RECALL.csv','_MATCHED_INTERVENTION.csv']],
      seconds=time.monotonic()-start)
    write(HERE/(STEM+'.json'),report)
    print(json.dumps({k:report[k]for k in ['completed','source_targets','query_ids','model_query_rows','GT_counts','seconds']},indent=2))

if __name__=='__main__':main()
