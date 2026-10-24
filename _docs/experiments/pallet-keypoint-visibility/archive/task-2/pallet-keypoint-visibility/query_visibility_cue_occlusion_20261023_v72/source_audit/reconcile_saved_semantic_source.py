"""Independent metrics from sealed source GT and saved outputs; no model imports."""
import hashlib
import json
import math
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/CUE_ATTENTION_V3'


def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def bind(p):
    p=Path(p)
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def ratio(a,b):return float(a/b) if b else None


def stats(rows,z,indices):
    ix=[i for i in indices if rows[i]['training_state'] in ['D','E']]
    for i in ix:
        assert rows[i]['crop_availability']=='FULL' and rows[i]['query_valid'] and rows[i]['reference_xy'] is not None
    d=np.array([rows[i]['training_state']=='D' for i in ix],bool)
    zz=z[ix].astype(np.float64);pred=zz>=0
    tp,fn,fp,tn=int((d&pred).sum()),int((d&~pred).sum()),int((~d&pred).sum()),int((~d&~pred).sum())
    loss=np.logaddexp(0,zz)-d*zz
    return dict(N=len(ix),truth_D=int(d.sum()),truth_E=int((~d).sum()),
                confusion_GT_D_E_pred_D_E=[[tp,fn],[fp,tn]],TP_D=tp,FP_D=fp,FN_D=fn,TN_D=tn,
                D_precision=ratio(tp,tp+fp),D_recall=ratio(tp,tp+fn),
                E_precision_conditional_knownDE=ratio(tn,tn+fn),E_recall=ratio(tn,tn+fp),
                unweighted_NLL=float(loss.mean()) if len(loss) else None,
                balanced_NLL=float(.5*loss[d].mean()+.5*loss[~d].mean()) if d.any() and (~d).any() else None)


def distribution(values):
    a=np.array(values,np.float64)
    return dict(N=len(a),mean=float(a.mean()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                min=float(a.min()) if len(a) else None,max=float(a.max()) if len(a) else None)


def main():
    rows=lines(ROOT/'source_cache/GT_ONLY.jsonl')
    freeze=read(RUN/'MODEL_FREEZE.json');history=read(RUN/'HISTORY.json');protocol=read(ROOT/'FIT_PROTOCOL.json')
    complete=read(RUN/'FIT_COMPLETE.json')
    assert len(rows)==3905 and all(r['input_index']==i for i,r in enumerate(rows))
    assert len({r['row_id'] for r in rows})==3905
    assert freeze['completed'] and complete['completed'] and freeze['optimizer_updates']==protocol['updates']==1000
    assert protocol['validation_every']==50 and protocol['seed']==20261023
    assert [r['update'] for r in history]==list(range(50,1001,50))
    best=math.inf;beststep=None;transitions=[]
    for rec in history:
        v=rec['VAL_balanced_source_cue_NLL'];improved=v<best
        if improved:best=v;beststep=rec['update']
        assert rec['saved']==improved and rec['best_step']==beststep
        transitions.append(dict(update=rec['update'],criterion=v,strict_improvement=improved,earliest_running_best=beststep))
    assert beststep==freeze['best_update']==50 and best==freeze['best_VAL_balanced_source_cue_NLL']
    with np.load(RUN/'SOURCE_SCORES.npz',allow_pickle=False) as z:
        assert set(z.files)=={'logit','heatmap','attention'}
        z72=z['logit'].astype(np.float64);heat=z['heatmap'].astype(np.float64);att=z['attention'].astype(np.float64)
    with np.load(RUN/'V61_SOURCE_SCORES.npz',allow_pickle=False) as z:
        assert set(z.files)=={'logit'}
        z61=z['logit'].astype(np.float64)
    assert z72.shape==z61.shape==(3905,) and att.shape==heat.shape==(3905,9,9)
    assert all(np.isfinite(a).all() for a in [z72,z61,att,heat]) and (att>=0).all()
    assert max(abs(att.sum((1,2))-1))<2e-6
    nulls=[i for i,r in enumerate(rows) if r['reference_xy'] is None]
    assert len(nulls)==52 and all(rows[i]['training_state']=='U' for i in nulls)
    grouping=defaultdict(list)
    cues=defaultdict(list)
    for i,r in enumerate(rows):
        grouping[(r['source'],r['split'])].append(i)
        cues[(r['source'],r['split'],r['cue_refID'])].append(i)
    results={}
    for tag,z in [('V72_CUE_ATTENTION_V3',z72),('V61_PIXEL_HEAD_ON_SAME_INPUT',z61)]:
        source_results={str(k):stats(rows,z,ix) for k,ix in grouping.items()}
        cue_results={str(k):stats(rows,z,ix) for k,ix in cues.items()}
        vals=defaultdict(list)
        for (source,split,cue),ix in cues.items():
            if split!='VALIDATION':continue
            stat=stats(rows,z,ix)
            if stat['N']:
                assert stat['balanced_NLL'] is not None
                vals[source].append(stat['balanced_NLL'])
        selector=float(np.mean([np.mean(v) for v in vals.values()]))
        unknowns=defaultdict(list);offsets=defaultdict(list)
        for i,r in enumerate(rows):
            if r['training_state']=='U':unknowns[(r['source'],r['split'],r['ureason'],r['crop_availability'],r['variant'],r['reference_xy'] is None)].append(i)
            if r['reference_xy'] is None:continue
            a=max(abs((np.asarray(r['query_xy'])-np.asarray(r['reference_xy']))/r['L']))
            bucket='[0,.125]' if a<=.125 else '(.125,.25]' if a<=.25 else '(.25,.5]' if a<=.5 else '>.5'
            offsets[(r['source'],r['split'],bucket)].append(i)
        u=[]
        for key,ix in unknowns.items():
            u.append(dict(source=key[0],split=key[1],ureason_recorded=key[2],availability=key[3],variant=key[4],reference_is_null=key[5],N=len(ix),raw_D=int((z[ix]>=0).sum()),raw_E=int((z[ix]<0).sum()),semantic_accuracy=None))
        original={}
        for i,r in enumerate(rows):
            if r['variant'] in ['ORIGINAL','BASE_ORIGINAL']:
                key=(r['source'],r['split'],r['cue_refID'],tuple(r['query_xy']),r['L'])
                assert key not in original
                original[key]=i
        paired=defaultdict(lambda:dict(all_matched_rows=0,eligible_known_pair=0,original_D_pred_D=0,intervention_correct=0,both_correct=0,pred_transition_DD_DE_ED_EE=[0,0,0,0],not_known_DE_pair=0))
        for j,r in enumerate(rows):
            if r['variant'] not in ['FULL_COVER','BASE_EXISTING_NEUTRAL','NEUTRAL_D','QUERY_ONLY_CONTROL']:continue
            key=(r['source'],r['split'],r['cue_refID'],tuple(r['query_xy']),r['L'])
            i=original[key]
            kind='D_to_E' if r['variant']=='FULL_COVER' else 'D_to_D'
            rec=paired[(r['source'],r['split'],r['variant'],kind)]
            rec['all_matched_rows']+=1
            expected='E' if kind=='D_to_E' else 'D'
            if rows[i]['training_state']!='D' or r['training_state']!=expected:
                rec['not_known_DE_pair']+=1;continue
            rec['eligible_known_pair']+=1
            a,b=bool(z[i]>=0),bool(z[j]>=0)
            correct=(not b) if kind=='D_to_E' else b
            rec['original_D_pred_D']+=int(a);rec['intervention_correct']+=int(correct);rec['both_correct']+=int(a and correct)
            rec['pred_transition_DD_DE_ED_EE'][(0 if a else 2)+(0 if b else 1)]+=1
        for rec in paired.values():
            rec['both_correct_rate']=ratio(rec['both_correct'],rec['eligible_known_pair'])
            rec['intervention_correct_rate']=ratio(rec['intervention_correct'],rec['eligible_known_pair'])
        results[tag]=dict(by_source_split=source_results,by_source_split_cue=cue_results,
                          VAL_balanced_source_cue_NLL=selector,by_GT_offset={str(k):stats(rows,z,ix) for k,ix in offsets.items()},
                          unknown_output_distributions=u,paired={str(k):v for k,v in paired.items()})
    saved_nll=results['V72_CUE_ATTENTION_V3']['VAL_balanced_source_cue_NLL']
    delta=saved_nll-best
    assert abs(delta)<1e-4
    # Saved predicted-attention mass over uncertainty area, not fabricated visible-E localization.
    localization=defaultdict(lambda:dict(mass=[],peak_error_px=[],peak_error_over_L=[],peak_inside_uncertainty=[],CE=[]))
    for i,r in enumerate(rows):
        if r['training_state']!='D':continue
        assert r['reference_xy'] is not None
        q=np.asarray(r['query_xy'],np.float64);L=r['L'];bb=np.asarray(r['uncertainty_bbox_xyxy'],np.float64)
        ex=q[0]+(np.arange(10)/9-.5)*L;ey=q[1]+(np.arange(10)/9-.5)*L
        dx=np.maximum(0,np.minimum(ex[1:],bb[2])-np.maximum(ex[:-1],bb[0]))
        dy=np.maximum(0,np.minimum(ey[1:],bb[3])-np.maximum(ey[:-1],bb[1]))
        area=dy[:,None]*dx[None,:];target=area/area.sum();coverage=area/(L/9)**2
        mass=float((att[i]*coverage).sum());assert -1e-8<=mass<=1+1e-6
        py,px=np.unravel_index(np.argmax(att[i]),(9,9));peak=q+(np.array([px,py])-4)*L/9
        err=float(np.linalg.norm(peak-np.asarray(r['reference_xy'])))
        inside=bool(bb[0]<=peak[0]<bb[2] and bb[1]<=peak[1]<bb[3])
        ce=float(-(target*np.log(np.maximum(att[i],1e-300))).sum())
        for key in [(r['source'],r['split'],'ALL_D'),(r['source'],r['split'],r['cue_refID'])]:
            rec=localization[key];rec['mass'].append(mass);rec['peak_error_px'].append(err);rec['peak_error_over_L'].append(err/L);rec['peak_inside_uncertainty'].append(inside);rec['CE'].append(ce)
    loc={str(k):dict(D_count=len(v['mass']),uncertainty_area_attention_mass=distribution(v['mass']),
                    native9_peak_error_px=distribution(v['peak_error_px']),peak_error_over_L=distribution(v['peak_error_over_L']),
                    peak_inside_uncertainty_count=sum(v['peak_inside_uncertainty']),D_heatmap_area_target_CE=distribution(v['CE'])) for k,v in localization.items()}
    report=dict(schema='V72_INDEPENDENT_SAVED_SEMANTIC_SOURCE_RECONCILIATION_V1',completed=True,status='PASS',
      bindings=[bind(p) for p in [ROOT/'source_cache/GT_ONLY.jsonl',RUN/'SOURCE_SCORES.npz',RUN/'V61_SOURCE_SCORES.npz',RUN/'HISTORY.json',RUN/'MODEL_FREEZE.json',RUN/'FIT_COMPLETE.json',ROOT/'FIT_PROTOCOL.json',Path(__file__)]],
      row_count=3905,GT_counts=dict(Counter(r['training_state'] for r in rows)),null_reference_U_count=52,null_reference_offset_or_localization_fabricated=0,
      fixed_threshold_logit=0,tie_is_D=True,pixel_v61_interpretation='Same-input binary pixel-presence baseline scored against explicit conditional source cue D/E teachers, not a newly trained semantic head.',
      metrics=results,D_only_attention_localization=loc,
      selection_audit=dict(updates=1000,checks=20,validation_every=50,history_transitions=transitions,
                           earliest_strict_minimum=beststep,history_selected_NLL=best,saved_best_checkpoint_NLL=saved_nll,saved_minus_GPU_history=delta,
                           checkpoint_hash_from_freeze=freeze['checkpoint']['sha256'],checkpoint_weight_body_opened=False,
                           note='History used one40-row VAL batch; saved outputs use batches128. Tiny numerical delta is reported without replacing the original selector or choosing another checkpoint.'),
      paired_input_geometry_note='Pairs join immutable source/cue/split/q/L, with box preservation established by independent source-data audits; no feature/input NPZ arrays opened here.',
      attention_note='Mass integrates saved predicted attention times native9-cell fractional overlap with GT uncertainty. Peak is coarse native9; E/U have no visible-localization truth. Null references excluded.',
      U_reason_note='Recorded generic base-U reason is retained alongside availability and variant; no U is inferred as E and no null reference is replaced by q.',
      integrity_mismatches=0,new_forward_fit_features_RGB_current_PnP=0,model_or_evaluator_imports=0,
      source_findings='TRAIN known D/E perfect at selected checkpoint; reused three-cue YCB VAL has D0/30 and E10/10. This is source generalization failure, not goal completion.',
      original_artifacts_modified=False,reselection_or_refit=False)
    dest=ROOT/'source_audit/INDEPENDENT_SAVED_SOURCE_METRIC_RECONCILIATION.json';assert not dest.exists()
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=bind(dest),row_count=3905,GT_counts=report['GT_counts'],NLL_delta=delta,selected=beststep,
                         by_source=results['V72_CUE_ATTENTION_V3']['by_source_split'],paired=results['V72_CUE_ATTENTION_V3']['paired'],D_attention={k:v for k,v in loc.items() if 'ALL_D' in k}),indent=2))


if __name__=='__main__':main()
