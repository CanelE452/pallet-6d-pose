"""Additive V75 saved-output audit; no Torch/model/RGB/feature import or fit."""
from pathlib import Path
import hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/SOURCE_PRETRAIN_V1'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(p.resolve()),sha256=sha(p))
def load(p):return json.loads(Path(p).read_text())
def check(b):assert sha(b['path'])==b['sha256'],b['path']

protocol=load(ROOT/'PRETRAIN_PROTOCOL.json');freeze=load(RUN/'MODEL_FREEZE.json');complete=load(RUN/'FIT_COMPLETE.json')
for b in protocol['bindings'].values():check(b)
for k in ('checkpoint','model_code','trainer_code','protocol','entry'):check(freeze[k])
for k in ('model_freeze','history','source_results','baseline0_scores','selected_scores'):check(complete[k])
assert freeze['optimizer_updates']==complete['optimizer_updates']==1000
history=load(RUN/'HISTORY.json');assert [x['update']for x in history]==list(range(50,1001,50))
earliest=min(history,key=lambda x:x['heldout9strata_macroNLL'])
assert earliest['update']==freeze['best_update']==750 and earliest['heldout9strata_macroNLL']==freeze['best_heldout9strata_macroNLL']
with np.load(protocol['bindings']['GT']['path'],allow_pickle=False) as z:
    y=z['proxy_state8'].reshape(-1);primary=z['primary_DES_loss_mask'].reshape(-1)
    part=np.repeat(z['partition'],8);source=np.repeat(z['source'],8)
    offset=z['query_to_reference_offset_over_L8'].reshape(-1,2)
    square=z['reference_in_local_square8'].reshape(-1)
sources=('G38','P0','TEX');results={};max_nll_diff=0.
for name,filename,reportname in [('update0','BASELINE_UPDATE0_SOURCE_SCORES.npz','BASELINE_UPDATE0_RESULTS.json'),('selected','SELECTED_SOURCE_SCORES.npz','SOURCE_RESULTS.json')]:
    with np.load(RUN/filename,allow_pickle=False) as z:
        logits=z['proxy_logits'].reshape(-1,3).astype(np.float64);nll=z['NLL_primary_FP32'].reshape(-1)
        attention=z['reference_attention_weights'].reshape(-1,9,9)
    assert logits.shape==(40960,3) and np.isfinite(logits).all() and np.isfinite(attention).all()
    report=load(RUN/reportname);metrics=report['metrics']if name=='update0' else report['selected_metrics']
    normalizer=np.logaddexp.reduce(logits,axis=1);p=np.exp(logits-normalizer[:,None]);pred=logits.argmax(1)
    max_nll_diff=max(max_nll_diff,float(np.max(np.abs((normalizer[primary]-logits[primary,y[primary]])-nll[primary]))))
    groups={}
    for partition,expected in [('train',29353),('heldout',7358)]:
        mask=primary&(part==partition);assert int(mask.sum())==expected
        cohorts={}
        for s in ('ALL',*sources):
            selected=mask if s=='ALL' else mask&(source==s)
            cm=np.zeros((3,3),np.int64);np.add.at(cm,(y[selected],pred[selected]),1)
            old=metrics[partition]['overall']if s=='ALL' else metrics[partition]['per_source'][s]
            assert cm.tolist()==old['confusion_rows_GT_cols_pred_DES']
            precision=[float(cm[c,c]/cm[:,c].sum())if cm[:,c].sum()else None for c in range(3)]
            recall=[float(cm[c,c]/cm[c].sum())if cm[c].sum()else None for c in range(3)]
            assert precision==old['class_precision'] and recall==old['class_recall']
            cohorts[s]=dict(N=int(selected.sum()),confusion_rowsGT_colsPred_DES=cm.tolist(),precision_DES=precision,recall_DES=recall)
        auxiliary=mask&(y==0)&square&np.isfinite(offset).all(1)
        assert int(auxiliary.sum())==metrics[partition]['auxiliary_D_primary_local_square']
        groups[partition]=dict(cohorts=cohorts,auxiliary_D_primary_local_square=int(auxiliary.sum()),primary_D=int(np.sum(mask&(y==0))),
            auxiliary_peak_error_native_px_median=metrics[partition]['auxiliary_peak_error_native_px']['0.5'],
            auxiliary_attention_mean_error_over_L_median=metrics[partition]['auxiliary_attention_mean_error_over_L']['0.5'])
    heldout9={f'{s}|{c}':float(nll[primary&(part=='heldout')&(source==s)&(y==c)].astype(np.float64).mean())for s in sources for c in range(3)}
    macro=float(np.mean(list(heldout9.values())))
    expectedmacro=report['heldout9strata_macroNLL']if name=='update0' else report['selection_heldout_macroNLL']
    # Selection used primary-heldout-only batches; the frozen full-slot saved
    # replay has different batch composition. Preserve both numerical values;
    # never reselect checkpoints from this post-fit replay.
    if name=='update0':assert macro==expectedmacro
    results[name]=dict(heldout9strata_macroNLL_saved_full_replay=macro,heldout9strata_macroNLL_frozen_selection=expectedmacro,
        saved_replay_minus_selection=macro-expectedmacro,cohorts=groups)
assert max_nll_diff<5e-6
out=ROOT/'SAVED_OUTPUT_AUDIT_AND_RECEIPT.json';assert not out.exists()
out.write_text(json.dumps(dict(completed=True,PASS=True,code=bind(Path(__file__)),protocol=bind(ROOT/'PRETRAIN_PROTOCOL.json'),entry=bind(ROOT/'TRAINER_ENTRYLOCK.json'),
    completion=bind(RUN/'FIT_COMPLETE.json'),model_freeze=bind(RUN/'MODEL_FREEZE.json'),checkpoint=freeze['checkpoint'],history=bind(RUN/'HISTORY.json'),
    baseline0_scores=bind(RUN/'BASELINE_UPDATE0_SOURCE_SCORES.npz'),selected_scores=bind(RUN/'SELECTED_SOURCE_SCORES.npz'),
    history_evaluations=20,selection_earliest_minimum_update=750,retained_frames=5120,retained_slots=40960,TRAIN_primary=29353,heldout_primary=7358,
    saved_FP32_NLL_vs_independent_double_logits_max_absdiff=max_nll_diff,results=results,
    important_limitations=[
        'Conditional renderer virtual-reference/ray/convex proxy supervision; human semantic visibility and native pixel-support GT are UNKNOWN.',
        'Historical upstream heldout is the checkpoint selector, not a new independent full-system or real-transfer test.',
        'Balanced source/class sampling raises E recall but E precision remains lower; full natural-prior confusion is reported.',
        'Peak reference error improves while attention-mean offset error worsens relative to nearly centered uniform baseline; do not report localization as uniformly improved.',
        'NLL_primary_FP32 is a saved numerical array for all retained slots with F/U labels clamped only for computation; only primary_DES_loss_mask rows are valid supervised NLL.',
        'Reference-location response is virtual reference proxy, not physical cue gold; no current/public-real adaptation claim.'
    ],current_or_public_real_reads_or_forward=0,new_backbone=0,PnP=0,new_training_after_locked1000=0),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(bind(out)))
