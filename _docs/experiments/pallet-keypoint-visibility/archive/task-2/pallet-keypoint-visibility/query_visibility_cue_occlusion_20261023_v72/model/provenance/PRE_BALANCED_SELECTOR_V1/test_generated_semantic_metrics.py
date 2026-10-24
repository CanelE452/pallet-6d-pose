import hashlib
import json
from pathlib import Path

import numpy as np
from saved_semantic_metrics import evaluate_saved


def main():
    def row(state,source='YCB_REAL_CUE',cue='cue1',availability='FULL',reason='known_teacher',valid=True):
        return dict(source=source,cue_refID=cue,split='VALIDATION',training_state=state,
                    crop_availability=availability,query_valid=valid,ureason=reason,
                    query_xy=[10.,20.],reference_xy=[10.,20.],L=30.)
    rows=[row('D'),row('E'),row('D'),row('E'),row('U',availability='MISS',reason='cropMISS'),
          row('D','SYNTH','cue2'),row('E','SYNTH','cue2')]
    logits=np.array([0.,-.4,-.2,.3,5.,.6,-.7])
    pairs=[dict(original_index=0,intervention_index=1,kind='D_to_E'),
           dict(original_index=0,intervention_index=2,kind='D_to_D'),
           dict(original_index=0,intervention_index=4,kind='D_to_U')]
    r=evaluate_saved(rows,logits,pairs)
    a=r['overall']
    assert [a['TP_D'],a['FP_D'],a['FN_D'],a['TN_D']]==[2,1,1,2]
    assert a['D_precision']==a['D_recall']==a['E_recall']==2/3
    assert r['unknown_output_distributions'][0]['N']==1 and r['unknown_output_distributions'][0]['raw_D_at_fixed_zero']==1
    assert r['paired_transitions'][str(('YCB_REAL_CUE','D_to_E'))]['both_correct']==1
    assert r['paired_transitions'][str(('YCB_REAL_CUE','D_to_D'))]['both_correct']==0
    expected=np.mean([np.mean(np.logaddexp(0,logits[:4])-np.array([1,0,1,0])*logits[:4]),
                      np.mean(np.logaddexp(0,logits[5:])-np.array([1,0])*logits[5:])])
    assert abs(r['VAL_selector_macro_source_cue_NLL']-expected)<1e-12
    # Unknowns never enter NLL or confusion regardless of their raw predictions.
    logits[4]=-1000
    rr=evaluate_saved(rows,logits,pairs)
    assert rr['overall']==r['overall'] and rr['VAL_selector_macro_source_cue_NLL']==r['VAL_selector_macro_source_cue_NLL']
    bad=[row('E',availability='PARTIAL')]
    rejected=False
    try:evaluate_saved(bad,np.array([1.]))
    except ValueError:rejected=True
    assert rejected
    p=Path(__file__).resolve().parent
    b=lambda f:dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    out=p/'GENERATED_SEMANTIC_METRIC_TESTS.json'
    assert not out.exists()
    report=dict(completed=True,status='PASS',generated_only=True,code=b(p/'saved_semantic_metrics.py'),test_code=b(Path(__file__).resolve()),
                known_DE_confusion_and_precision_recall=True,U_no_negative_label_no_metric_denominator=True,
                logit_zero_tie_D=True,paired_bothcorrect_and_retention=True,
                source_cue_macro_NLL=True,nonFULL_label_rejected=True,
                source_scores_GT_RGB_model_current_reads=0,actual_forward_fit_PnP=0)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
