"""Saved-source output-space loss decomposition, no model/features/current data."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];HERE=ROOT/'source_audit';STEM='SOURCE_LOSS_TRADEOFF_DIAGNOSIS'
MODELS=('V65_BASELINE','CONTROLLED_ONLY','CONTROLLED_PLUS_PALLOX')
SOURCES=('SYNTH','ORIGINAL_YCB','CONTROLLED_YCB','PALLOX')
JOINT=('VV','EV','EE','OV','OE','OO');PAIRS=np.array(((0,0),(1,0),(1,1),(2,0),(2,1),(2,2)),np.int8)
EXACT_NAMES={0:'KNOWN_VV',1:'KNOWN_EXACT_E',2:'KNOWN_EXACT_O',-1:'UNKNOWN_EXACT_PARTIAL'}
KEYS=('sample_id','source','image_sha256','center_native_valid','center_allowed','GT_state',
      'joint_logits','joint_probability','visible_probability','raw_SAM_visible_probability','predicted_state','partial_NLL')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def bind(b):assert sha(b['path'])==b['sha256'],b['path']
def write(p,j):Path(p).write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')
def csvwrite(path,rows):
    if not rows:return
    with Path(path).open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def ratio(a,b):return float(a)/int(b)if b else None
def stats(a):
    a=np.asarray(a,np.float64)
    if not a.size:return dict(N=0,mean=None,minimum=None,q10=None,q25=None,median=None,q75=None,q90=None,maximum=None)
    return dict(N=int(a.size),mean=float(a.mean()),minimum=float(a.min()),q10=float(np.quantile(a,.10)),q25=float(np.quantile(a,.25)),
                median=float(np.median(a)),q75=float(np.quantile(a,.75)),q90=float(np.quantile(a,.90)),maximum=float(a.max()))


def load():
    output={};bindings=[];canonical=None;errors={}
    for model in MODELS:
        folder=ROOT/'source_evaluation'/model;report=json.loads((folder/'SOURCE_METRICS.json').read_text())
        assert report['completed']and report['source_only']and not report['current223_RGB_GT_scores_read']
        registered={str(Path(b['path']).resolve()):b for b in report['outputs']}
        for name in ['SOURCE_OUTPUTS.npz','PER_QUERY.csv']:
            p=folder/name;bind(registered[str(p.resolve())]);bindings.append(ref(p))
        bindings.append(ref(folder/'SOURCE_METRICS.json'))
        with np.load(folder/'SOURCE_OUTPUTS.npz',allow_pickle=False)as z:d={k:z[k]for k in KEYS}
        ids=[str(s)for s in d['sample_id']];assert len(ids)==len(set(ids))==8590
        if canonical is None:canonical=ids
        assert set(ids)==set(canonical);lookup={s:i for i,s in enumerate(ids)};order=np.array([lookup[s]for s in canonical])
        d={k:v[order]for k,v in d.items()};p=d['joint_probability'];z=d['joint_logits'].astype(np.float64);a=d['center_allowed']
        permit=(a[:,None]&(1<<np.arange(6))[None])!=0
        assert np.isfinite(z).all()and np.isin(a,np.arange(1,64)).all()
        lse=np.logaddexp.reduce(z,axis=1);allowed_lse=np.logaddexp.reduce(np.where(permit,z,-np.inf),axis=1)
        recomputed=lse-allowed_lse;recomputed[a==63]=0.
        maxerr=float(np.max(np.abs(recomputed-d['partial_NLL'])));assert maxerr<=2e-12
        pcheck=np.exp(z-z.max(1,keepdims=True));pcheck/=pcheck.sum(1,keepdims=True)
        assert np.allclose(p,pcheck,atol=2e-15,rtol=2e-15)
        conditional=np.exp(np.where(permit,z-allowed_lse[:,None],-np.inf))
        gradient=p-conditional;gradient[a==63]=0.
        assert np.max(np.abs(gradient.sum(1)))<2e-14
        d['gradient']=gradient;d['cardinality']=permit.sum(1);d['informative']=d['center_native_valid']&(a!=63)
        assert (a[~d['center_native_valid']]==63).all()
        expected=np.where((p.max(1)>=.5)[:,None],PAIRS[p.argmax(1)],3)
        assert np.array_equal(expected,d['predicted_state'])
        assert np.array_equal(d['visible_probability'],np.stack((p[:,0],p[:,0]+p[:,1]+p[:,3]),1))
        exact_possible=np.stack([(a&bit)!=0 for bit in [1,6,56]],1)
        near_possible=np.stack([(a&bit)!=0 for bit in [11,20,32]],1)
        for k,possible in enumerate([exact_possible,near_possible]):
            state=np.where(possible.sum(1)==1,possible.argmax(1),-1);state[~d['center_native_valid']]=-1
            assert np.array_equal(state,d['GT_state'][:,k])
        csvrows={r['sample_id']:r for r in csv.DictReader((folder/'PER_QUERY.csv').open())};assert set(csvrows)==set(canonical)
        for i,sid in enumerate(canonical):
            r=csvrows[sid]
            assert int(r['allowed'])==int(a[i])and(r['native_valid']=='True')==bool(d['center_native_valid'][i])
            assert float(r['partial_NLL'])==d['partial_NLL'][i]
            assert float(r['exact_pV'])==p[i,0]
        allV=d['center_native_valid']&(d['GT_state'][:,0]==0)
        assert (a[allV]==1).all()and(d['GT_state'][allV,1]==0).all()
        assert np.allclose(d['partial_NLL'][allV],-np.log(p[allV,0]),atol=2e-12,rtol=2e-12)
        assert np.allclose(gradient[allV,0],p[allV,0]-1,atol=2e-15)
        means={s:float(d['partial_NLL'][d['informative']&(d['source']==s)].mean())for s in SOURCES}
        original=report['fixed_selection_statistic'];macro=float(np.mean(list(means.values())))
        assert abs(macro-original['common4domain_mean_informative_center_partial_NLL'])<2e-12
        if output:
            first=output[MODELS[0]]
            for key in ['sample_id','source','image_sha256','center_native_valid','center_allowed','GT_state','raw_SAM_visible_probability']:
                assert np.array_equal(d[key],first[key]),(model,key)
        output[model]=d;errors[model]=dict(recomputed_NLL_max_abs=maxerr,CPU_saved_macro=macro,source_means=means)
    return output,bindings,errors


def partitions(d):
    a=d['center_allowed'];card=d['cardinality'];gt=d['GT_state'][:,0]
    singleton=np.array([JOINT[int(np.log2(int(v)))]if int(c)==1 else'PARTIAL_CARD_'+str(int(c))for v,c in zip(a,card)])
    return {'EXACT_GT':np.array([EXACT_NAMES[int(v)]for v in gt]),
            'CARDINALITY':np.array(['CARD_'+str(int(v))for v in card]),
            'EXACT_X_CARDINALITY':np.array([EXACT_NAMES[int(g)]+'__CARD_'+str(int(c))for g,c in zip(gt,card)]),
            'SINGLETON_JOINT_OR_PARTIAL':singleton,
            'ALLOWED_BITMASK':np.array(['MASK_'+str(int(v))for v in a])}


def decomposition(outputs):
    rows=[];totals=[]
    for model,d in outputs.items():
        info=d['informative'];part=partitions(d)
        for source in SOURCES:
            si=(d['source']==source);ix=si&info;ns=int(ix.sum());assert ns>0
            totals.append(dict(model=model,source=source,total_rows=int(si.sum()),native_valid=int((si&d['center_native_valid']).sum()),
              informative_N=ns,uninformative63_native_valid=int((si&d['center_native_valid']&(d['center_allowed']==63)).sum()),
              source_mean_NLL=float(d['partial_NLL'][ix].mean()),macro_source_contribution=float(d['partial_NLL'][ix].sum()/(4*ns))))
            for pname,labels in part.items():
                checks=[];count=0
                for category in sorted(set(labels[si])):
                    ci=ix&(labels==category);n=int(ci.sum());loss=float(d['partial_NLL'][ci].sum())
                    row=dict(model=model,source=source,partition=pname,category=category,N=n,source_informative_denominator=ns,
                      source_count_fraction=n/ns,loss_sum=loss,conditional_mean_NLL=loss/n if n else None,
                      source_mean_contribution=loss/ns,macro_mean_contribution=loss/(4*ns))
                    for j,name in enumerate(JOINT):
                        row['output_logit_gradient_sum_'+name]=float(d['gradient'][ci,j].sum())
                        row['macro_output_logit_gradient_'+name]=float(d['gradient'][ci,j].sum()/(4*ns))
                    rows.append(row);checks.append(loss/ns);count+=n
                assert count==ns and abs(sum(checks)-float(d['partial_NLL'][ix].mean()))<2e-12
    return rows,totals


def contrasts(rows,outputs):
    index={(r['model'],r['source'],r['partition'],r['category']):r for r in rows};out=[];rollups=[]
    for before,after in [('V65_BASELINE','CONTROLLED_ONLY'),('V65_BASELINE','CONTROLLED_PLUS_PALLOX'),('CONTROLLED_ONLY','CONTROLLED_PLUS_PALLOX')]:
        for source in SOURCES:
            keys=[k for k in index if k[0]==before and k[1]==source]
            for _,_,partition,category in keys:
                a=index[before,source,partition,category];b=index[after,source,partition,category];assert a['N']==b['N']and a['source_informative_denominator']==b['source_informative_denominator']
                out.append(dict(before=before,after=after,source=source,partition=partition,category=category,N=a['N'],source_informative_denominator=a['source_informative_denominator'],
                  before_conditional_mean_NLL=a['conditional_mean_NLL'],after_conditional_mean_NLL=b['conditional_mean_NLL'],
                  delta_conditional_mean_NLL=(b['conditional_mean_NLL']-a['conditional_mean_NLL'])if a['N']else None,
                  delta_source_mean_contribution=b['source_mean_contribution']-a['source_mean_contribution'],
                  delta_macro_mean_contribution=b['macro_mean_contribution']-a['macro_mean_contribution']))
        exact=[r for r in out if r['before']==before and r['after']==after and r['partition']=='EXACT_GT']
        ycbv=sum(r['delta_macro_mean_contribution']for r in exact if r['source']in ['ORIGINAL_YCB','CONTROLLED_YCB']and r['category']=='KNOWN_VV')
        allv=sum(r['delta_macro_mean_contribution']for r in exact if r['category']=='KNOWN_VV')
        other=sum(r['delta_macro_mean_contribution']for r in exact if not(r['source']in ['ORIGINAL_YCB','CONTROLLED_YCB']and r['category']=='KNOWN_VV'))
        total=sum(r['delta_macro_mean_contribution']for r in exact)
        rollups.append(dict(before=before,after=after,delta_macro_NLL=total,YCB_knownV_delta_macro_contribution=ycbv,
          other_source_state_delta_macro_contribution=other,all4source_knownV_delta_macro_contribution=allv,
          nonV_delta_macro_contribution=total-allv,YCB_V_loss_increase_offset_by_other_terms=bool(ycbv>0 and other<0 and total<0),
          interpretation='Positive delta=worse NLL, negative delta=better NLL. Exact sum of original per-source informative means dividedby4.'))
    return out,rollups


def strict_visible(outputs,boundarypath):
    allrows=list(csv.DictReader(boundarypath.open()));gtrows=[r for r in allrows if r['model']==MODELS[0]]
    meta={r['sample_id']:r for r in gtrows};assert len(meta)==6400
    base=outputs[MODELS[0]];ids=[str(s)for s in base['sample_id']];index={s:i for i,s in enumerate(ids)}
    summaries=[];distributions=[];export=[];paired=[]
    for source in ['ORIGINAL_YCB','CONTROLLED_YCB']:
        cohort=np.array([index[s]for s,r in meta.items()if r['source']==source and r['native_valid']=='True'and r['exact_GT']=='0'and r['strict25_V']=='True'],int)
        assert (base['center_allowed'][cohort]==1).all()
        sampositive=base['raw_SAM_visible_probability'][cohort,0]>.5;chosen=cohort[sampositive]
        for model,d in outputs.items():
            p=d['joint_probability'];head=d['visible_probability'];sam=d['raw_SAM_visible_probability'];fn=head[:,0]<.5
            for scope,indices in [('STRICT25_KNOWN_V_ALL',cohort),('STRICT25_KNOWN_V_RAW_SAM_POSITIVE',chosen),('STRICT25_KNOWN_V_RAW_SAM_POSITIVE_HEAD_FN',chosen[fn[chosen]])]:
                n=len(indices);conf=Counter(JOINT[int(p[i].argmax())]if p[i].max()>=.5 else'U'for i in indices)
                summary=dict(source=source,model=model,scope=scope,N=n,TP=int((~fn[indices]).sum()),FN=int(fn[indices].sum()),
                  exact_visible_recall=ratio((~fn[indices]).sum(),n),radius2_visible_diagnostic=int((head[indices,1]>=.5).sum()),
                  canonical_mean_partialNLL=float(d['partial_NLL'][indices].mean())if n else None,
                  **{'confident_'+name:conf[name]for name in JOINT+('U',)})
                summaries.append(summary)
                values={**{'p'+name:p[indices,j]for j,name in enumerate(JOINT)},'pVisible_exact':head[indices,0],
                  'pVisible_radius2':head[indices,1],'rawSAM_exact_pV':sam[indices,0],'rawSAM_radius2_pV':sam[indices,1],
                  'partialNLL':d['partial_NLL'][indices],**{'output_logit_gradient_'+name:d['gradient'][indices,j]for j,name in enumerate(JOINT)}}
                for field,vector in values.items():distributions.append(dict(source=source,model=model,scope=scope,field=field,**stats(vector)))
            for i in chosen:
                row=dict(source=source,model=model,sample_id=ids[i],allowed=1,strict25_GT='VV',head_FN=bool(fn[i]),
                  rawSAM_exact_pV=float(sam[i,0]),rawSAM_radius2_pV=float(sam[i,1]),pVisible_exact=float(head[i,0]),pVisible_radius2=float(head[i,1]),
                  partialNLL=float(d['partial_NLL'][i]),joint_MAP=JOINT[int(p[i].argmax())],
                  confident_joint_or_U=JOINT[int(p[i].argmax())]if p[i].max()>=.5 else'U')
                row.update({'p'+name:float(p[i,j])for j,name in enumerate(JOINT)});export.append(row)
        for before,after in [('V65_BASELINE','CONTROLLED_ONLY'),('V65_BASELINE','CONTROLLED_PLUS_PALLOX'),('CONTROLLED_ONLY','CONTROLLED_PLUS_PALLOX')]:
            a=outputs[before];b=outputs[after];av=a['visible_probability'][chosen,0]>=.5;bv=b['visible_probability'][chosen,0]>=.5
            row=dict(source=source,before=before,after=after,scope='SAME_STRICT25_KNOWN_V_RAW_SAM_POSITIVE',N=len(chosen),
              both_V=int((av&bv).sum()),lost_V=int((av&~bv).sum()),gained_V=int((~av&bv).sum()),both_FN=int((~av&~bv).sum()),
              mean_delta_NLL=float(np.mean(b['partial_NLL'][chosen]-a['partial_NLL'][chosen])))
            for j,name in enumerate(JOINT):row['mean_delta_p'+name]=float(np.mean(b['joint_probability'][chosen,j]-a['joint_probability'][chosen,j]))
            paired.append(row)
    return summaries,distributions,export,paired


def main():
    reportpath=HERE/(STEM+'.json');assert not reportpath.exists(),'Append-only diagnosis'
    protocolpath=HERE/(STEM+'_PROTOCOL.json');assert not protocolpath.exists()
    protocol=dict(schema='V66_SAVED_SOURCE_LOSS_TRADEOFF_PROTOCOL',scope='Saved SOURCE outputs/PER_QUERY/partialGTbits/strictVdiagnosis only',
      models=list(MODELS),sources=list(SOURCES),selection_not_performed=True,
      loss='logsumexp(all6savedlogits)-logsumexp(allowed savedlogits); nativevalid AND allowed!=63 informative. Original CPU saved partial_NLL retained ascanonical and independently recomputed.',
      decomposition='For eachdisjointcategory/source: loss_sum/N_info_source is source contribution; divideby4for macro contribution. All category sums exactly reconstruct per-source means and common4domainmacro.',
      partitions=['ExactGT knownVV/knownE/knownO/unknownExact','Allowed cardinality','ExactGT cross cardinality','Singletonjoint or partialcardinality','Exact allowed bitmask'],
      visible_known='Verify every native-valid knownexactV has allowed1 (VV) andradius2V. Loss=-logpVV, output-logitgradient=p-onehotVV; no contradictory nearlabelatthese pixels.',
      strictV='Use frozen SOURCE_BOUNDARY_FAILURE_DIAGNOSIS_QUERY_ROWS.csv strict25flags, sourceGT+native5x5. Shared rawSAMpositive iff storedprob>.5 (original strict logit0sign). Head exactpV>=.5; radius2diagnostic only.',
      gradient='dL/dz_j=p_j - I_allowed(j)*exp(z_j)/sum_allowedexp(z). Sum accordingtofixedsourceinformative normalization. Output-space pressure only, neverparameter/branchcausal.',
      forbidden=['currentRGB/GT/scores/inputdata','featurecaches','weights','newmodelinference','optimizer','PnP','threshold/radius/modelreselection'],
      no_changes_to_frozen_V66_reports_or_seals=True)
    write(protocolpath,protocol);started=time.monotonic()
    outputs,bindings,errors=load();part,totals=decomposition(outputs);contrast,rollups=contrasts(part,outputs)
    boundary=HERE/'SOURCE_BOUNDARY_FAILURE_DIAGNOSIS_QUERY_ROWS.csv';bindings.append(ref(boundary))
    strict,distribution,query,paired=strict_visible(outputs,boundary)
    outputsets=[('_PARTITIONS.csv',part),('_SOURCE_TOTALS.csv',totals),('_MODEL_DELTA.csv',contrast),('_STRICT_V_SUMMARY.csv',strict),
      ('_STRICT_V_DISTRIBUTIONS.csv',distribution),('_STRICT_V_QUERY_ROWS.csv',query),('_STRICT_V_PAIRED_DELTAS.csv',paired)]
    for suffix,rows in outputsets:csvwrite(HERE/(STEM+suffix),rows)
    base=outputs[MODELS[0]];knownV=base['center_native_valid']&(base['GT_state'][:,0]==0)
    for b in bindings:bind(b)
    report=dict(schema='V66_SAVED_SOURCE_LOSS_TRADEOFF_DIAGNOSIS',completed=True,protocol=ref(protocolpath),code=ref(__file__),
      common_queries=8590,models=list(MODELS),source_counts=[dict(source=s,N=int((base['source']==s).sum()),informative=int(((base['source']==s)&base['informative']).sum()))for s in SOURCES],
      known_exactV_all_allowed1_verified=int(knownV.sum()),nativevalid_knownV_non_VV_masks=0,
      recomputation_audit=errors,source_totals=totals,macro_tradeoffs=rollups,
      exact_GT_partition=[r for r in part if r['partition']=='EXACT_GT'],strictV_summary=strict,strictV_paired_deltas=paired,
      interpretation=['This is the CPU saved-output criterion decomposition, not a reselection of the source-frozen GPU checkpoint/arm.',
       'KnownexactV has onlyVVcompatible: nearbytarget is alsoV, so no partial-label semantic contradiction on those rows.',
       'CommonmacroNLL is a weighted aggregate; class/source gains canoffset visible damage. Fixed0.5 recall is a separate statistic.',
       'Analytic logit derivatives do not identify featurebranch, parametergradient, or causaltrainingmechanism.',
       'StrictsourceVerrors do notresolve the distinct current semanticcorner/nativepixel task. Physicalmeshcorner support is not required.'],
      strictV_distributions_path=str(HERE/(STEM+'_STRICT_V_DISTRIBUTIONS.csv')),
      input_bindings=bindings,outputs=[ref(HERE/(STEM+suffix))for suffix,_ in outputsets],
      frozen_source_checkpoint_threshold_selection_preserved=True,current_data_reads=0,feature_or_weight_reads=0,
      new_model_inference_training_PnP_GPU_calls=0,seconds=time.monotonic()-started)
    write(reportpath,report)
    print(json.dumps({k:report[k]for k in ['completed','common_queries','source_counts','known_exactV_all_allowed1_verified','macro_tradeoffs','seconds']},indent=2))

if __name__=='__main__':main()
