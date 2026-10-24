"""Pure saved-score metrics for conditional source cue teachers, not pixel labels.

No file/model/image readers. The caller must verify seals and pass fixed row
order plus immutable GT-only metadata AFTER inference scores are sealed.
"""
import math
from collections import defaultdict

import numpy as np


def _safe(a,b):
    return float(a/b) if b else None


def _cohort(rows,logit,indices):
    ix = [i for i in indices if rows[i]['training_state'] in ['D','E']]
    for i in ix:
        if rows[i]['crop_availability'] != 'FULL' or not rows[i]['query_valid'] or not math.isfinite(float(logit[i])):
            raise ValueError('Every scored D/E teacher must be FULL, input valid and finite')
    d = np.array([rows[i]['training_state']=='D' for i in ix],bool)
    p = np.array([logit[i]>=0 for i in ix],bool)
    z = np.asarray([logit[i] for i in ix],np.float64)
    tp,fp,fn,tn = int((d&p).sum()),int((~d&p).sum()),int((d&~p).sum()),int((~d&~p).sum())
    return dict(N=len(ix),truth_D=int(d.sum()),truth_E=int((~d).sum()),
                confusion_GT_D_E_pred_D_E=[[tp,fn],[fp,tn]],TP_D=tp,FP_D=fp,FN_D=fn,TN_D=tn,
                D_precision=_safe(tp,tp+fp),D_recall=_safe(tp,tp+fn),
                E_precision_within_known_DE=_safe(tn,tn+fn),E_recall=_safe(tn,tn+fp),
                mean_known_DE_NLL=float(np.mean(np.logaddexp(0,z)-d*z)) if len(z) else None)


def evaluate_saved(rows,logit,pairs=()):
    """Canonical row fields: source,cue_refID,split,training_state,crop_availability,
    query_valid,ureason,query_xy,reference_xy,L. IDs/r are GT/eval metadata only.
    Pair fields original_index/intervention_index/kind; kind=D_to_E|D_to_D|D_to_U.
    """
    logit = np.asarray(logit,np.float64)
    if logit.shape != (len(rows),):
        raise ValueError('Saved logits must match immutable canonical row order')
    if any(r['training_state'] not in ['D','E','U'] for r in rows):
        raise ValueError('Use semantic source D/E/U teachers, never pixel V/E/O')
    allix = list(range(len(rows)))
    bysource,bycue,unknown = defaultdict(list),defaultdict(list),defaultdict(list)
    for i,r in enumerate(rows):
        bysource[r['source']].append(i)
        bycue[(r['source'],r['cue_refID'])].append(i)
        if r['training_state']=='U':
            unknown[(r['source'],r['ureason'],r['crop_availability'])].append(i)
    cue_results = {str(k):_cohort(rows,logit,ix) for k,ix in bycue.items()}
    source_cue_nll = defaultdict(list)
    for key,ix in bycue.items():
        valix=[i for i in ix if rows[i]['split']=='VALIDATION']
        nll=_cohort(rows,logit,valix)['mean_known_DE_NLL']
        if nll is not None:source_cue_nll[key[0]].append(nll)
    source_val={k:float(np.mean(v)) for k,v in source_cue_nll.items()}
    uresults=[]
    for (source,reason,availability),ix in unknown.items():
        usable=[i for i in ix if rows[i]['query_valid'] and math.isfinite(float(logit[i]))]
        uresults.append(dict(source=source,ureason=reason,crop_availability=availability,N=len(ix),
                            finite_input_valid_predictions=len(usable),
                            raw_D_at_fixed_zero=sum(logit[i]>=0 for i in usable),
                            raw_E_at_fixed_zero=sum(logit[i]<0 for i in usable),
                            unsupported_or_invalid=len(ix)-len(usable),
                            semantic_accuracy=None,reason='U is not a known negative or a trained rejection class'))
    offset_groups=defaultdict(list)
    for i,r in enumerate(rows):
        dx,dy=(np.asarray(r['query_xy'])-np.asarray(r['reference_xy']))/r['L']
        dist=max(abs(float(dx)),abs(float(dy)))
        group='[0,.125]' if dist<=.125 else '(.125,.25]' if dist<=.25 else '(.25,.5]' if dist<=.5 else '>.5'
        offset_groups[(r['source'],group)].append(i)
    paired=defaultdict(lambda:dict(N=0,both_correct=0,original_D_pred_D=0,intervention_correct=0))
    for pair in pairs:
        a,b,kind=pair['original_index'],pair['intervention_index'],pair['kind']
        ra,rb=rows[a],rows[b]
        if ra['source'] != rb['source'] or ra['cue_refID'] != rb['cue_refID'] or ra['split'] != rb['split'] or ra['query_xy'] != rb['query_xy'] or ra['L'] != rb['L']:
            raise ValueError('Pair must preserve source/cue/split/query/L identity')
        expected={'D_to_E':('D','E'),'D_to_D':('D','D'),'D_to_U':('D','U')}[kind]
        if (ra['training_state'],rb['training_state']) != expected:
            raise ValueError('Pair type differs from immutable teacher states')
        if not math.isfinite(float(logit[a])) or not math.isfinite(float(logit[b])):
            raise ValueError('Pair saved logits must be finite')
        rec=paired[(ra['source'],kind)]
        rec['N']+=1;rec['original_D_pred_D']+=int(logit[a]>=0)
        if kind!='D_to_U':
            correct=bool(logit[b]>=0) if kind=='D_to_D' else bool(logit[b]<0)
            rec['intervention_correct']+=int(correct)
            rec['both_correct']+=int(logit[a]>=0 and correct)
    return dict(overall=_cohort(rows,logit,allix),
                by_source={k:_cohort(rows,logit,ix) for k,ix in bysource.items()},
                by_fixed_cue=cue_results,
                VAL_selector_macro_source_cue_NLL=float(np.mean(list(source_val.values()))) if source_val else None,
                VAL_source_cue_NLL=source_val,
                unknown_output_distributions=uresults,
                by_GT_query_offset={str(k):_cohort(rows,logit,ix) for k,ix in offset_groups.items()},
                paired_transitions={str(k):v for k,v in paired.items()},
                threshold_logit=0.,tie_is_D=True,
                confidence_U_classifier_trained=False,native8_gold=False,
                current_authorization=False,full_goal_solved=False)
