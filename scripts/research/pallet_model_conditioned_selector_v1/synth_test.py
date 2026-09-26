"""One post-lock synthetic cross-matrix; no training or real references."""
from . import common as C
READS=C.guard('inference')
import numpy as np
import torch
from scripts.research.pallet_selector_recovery_v1 import models as M

def checkpoints():
    lock=C.read(C.stage(2)/'SCORER_LOCK.json');b=C.read(C.DOC/'INPUT_BINDINGS.json')
    bindings={'OLD_GEO':b['old_scorer'],'S1SPEC_GEO':lock['checkpoints']['S1'],'HMANSPEC_GEO':lock['checkpoints']['H_MANUAL']}
    for v in bindings.values():C.verify(v)
    return {k:torch.load(C.ROOT/v['path'],map_location='cpu',weights_only=False) for k,v in bindings.items()}

def main():
    assert not (C.stage(2)/'SYNTH_TEST_ONCE_LOCK.json').exists(),'TEST already opened; no repeated scoring'
    lock=C.read(C.stage(2)/'SCORER_LOCK.json');C.verify(lock['features']);C.verify(lock['labels']);fl=C.read(C.stage(2)/'SYNTH_FEATURES_LOCK.json')
    for b in fl['files']:C.verify(b)
    C.save(C.stage(2)/'SYNTH_TEST_ONCE_LOCK.json',dict(created_at=C.now(),scorer_lock=C.bind(C.stage(2)/'SCORER_LOCK.json'),test_reads=1,new_fits_after_TEST_forbidden=True))
    z=dict(np.load(C.ROOT/fl['files'][0]['path']));l=dict(np.load(C.ROOT/lock['labels']['path']));assert np.array_equal(z['ids'],l['ids'])
    mask=l['split']=='TEST';ck=checkpoints();out={};allscore={}
    for model in C.MODELS:
        valid=z[model+'_valid'][mask];y=l['parity'][mask];current=z[model+'_current'][mask]
        out[model+'_D9']=dict(n=int(mask.sum()),correct=int((current==y).sum()),accuracy=float((current==y).mean()),valid_pairs=int(valid.sum()),brier=None,abs_score_margin_quantiles=None)
        for selector,checkpoint in ck.items():
            score=M.scores(checkpoint,z[model+'_geo'][mask]);idx=M.selection(score,C.HYP)
            np.testing.assert_array_equal(idx,1-M.selection(score[:,::-1],C.HYP[::-1]))
            chosen=np.where(valid,idx,current);prob=1/(1+np.exp(np.clip(score[:,1]-score[:,0],-80,80)))
            out[model+'_'+selector]=dict(n=int(mask.sum()),correct=int((chosen==y).sum()),accuracy=float((chosen==y).mean()),valid_pairs=int(valid.sum()),
                brier=float(np.mean((prob[valid]-y[valid])**2)) if valid.any() else None,brier_denominator=int(valid.sum()),fallback_count=int((~valid).sum()),
                abs_score_margin_quantiles=np.quantile(np.abs(score[valid,0]-score[valid,1]),[.1,.5,.9]).tolist())
            allscore[model+'_'+selector]=dict(scores=score,selected=chosen)
    # GT-free same-frame feature shift in fixed old TRAIN-pooled normalization units.
    both=z['S1_valid']&z['H_MANUAL_valid'];dif=(z['H_MANUAL_geo'][both]-z['S1_geo'][both])/ck['OLD_GEO']['std'];l2=np.linalg.norm(dif,axis=-1)
    names=C.read(C.OLD/'SELECTOR_FEATURE_CONTRACT.json')['features'];families={
        'reprojection':[i for i,n in enumerate(names) if n.startswith('reprojection')],
        'invariants':[i for i,n in enumerate(names) if n in ('cheirality_fraction','lr_violations','tb_violations','front_rear_violations','invariant_violations','upright_alignment','spread_ratio')],
        'R_t':[i for i,n in enumerate(names) if n.startswith(('R_cf','t_'))],
        'projected_geometry':[i for i,n in enumerate(names) if n.startswith(('cf_','front_area','rear_area','LR_edge','TB_edge','FR_edge','front_rear_depth'))],
        'residual':[i for i,n in enumerate(names) if n.startswith('residual')],
        'confidence':[i for i,n in enumerate(names) if n.startswith(('bbox_','kpt_conf'))]}
    assert sorted(i for indices in families.values() for i in indices)==list(range(94))
    shift=dict(normalization='old pooled synthetic TRAIN mean/std; fixed before this experiment',paired_frames=int(both.sum()),median_L2=float(np.median(l2)),P90_L2=float(np.quantile(l2,.9)),
        by_candidate={n:dict(median_L2=float(np.median(l2[:,i])),P90_L2=float(np.quantile(l2[:,i],.9))) for i,n in enumerate(C.HYP)},
        families={n:dict(features=[names[i] for i in ii],median_L2=float(np.median(np.linalg.norm(dif[:,:,ii],axis=-1))),P90_L2=float(np.quantile(np.linalg.norm(dif[:,:,ii],axis=-1),.9))) for n,ii in families.items()},GT_used=False)
    C.save(C.stage(2)/'FEATURE_SHIFT.json',shift);C.save(C.RAW/'SYNTH_TEST_SCORES.json',allscore)
    C.save(C.stage(2)/'SYNTH_COMPATIBILITY_MATRIX.json',dict(combinations=out,candidate_order_swap_invariance=True,TEST_once=True,previously_viewed_development_heldout=True,independent_final_test=False))
    render(out,shift)
    print('SYNTH_MATRIX',out,flush=True)

def render(out,shift):
    vals={m:C.read(C.stage(2)/('S1_SCORER_VAL.json' if m=='S1' else 'HMAN_SCORER_VAL.json')) for m in C.MODELS}
    C.figure(2,'01_synth_val.png','Own-model synthetic VAL accuracy (early stopping)',list(C.MODELS),{'Best VAL':[vals[m]['best_val_accuracy'] for m in C.MODELS]},'Accuracy')
    C.figure(2,'02_synth_test_matrix.png','Synthetic TEST compatibility (already-viewed development-heldout)',list(C.MODELS),{s:[out[m+'_'+s]['accuracy'] for m in C.MODELS] for s in C.SELECTORS},'Parity accuracy')
    C.figure(2,'03_model_feature_shift.png','Same-frame normalized feature change (GT-free)',list(shift['families']),{s:[v[s] for v in shift['families'].values()] for s in ('median_L2','P90_L2')},'L2 in old TRAIN std units')
    C.figure(2,'04_score_margin_shift.png','Synthetic TEST valid-pair score margin',list(C.MODELS),{s:[out[m+'_'+s]['abs_score_margin_quantiles'][1] for m in C.MODELS] for s in C.SELECTORS[1:]},'Median absolute score margin')
    text='# Stage 2 — model-conditioned GEO_LINEAR\n\n94개 특징, 공유 선형 scorer, 기존 optimizer/early-stop을 그대로 사용했다. 기존 old GEO는 S0/S1 pooled로 학습됐고 이번 두 selector는 각각 자기 모델의 합성 TRAIN만 사용했다.\n\nTRAIN4096 / VAL1024 / TEST1024의 frame ID·renderer group·RGB·K·치수·정확한 parity labels 모두 기존 split 그대로다. GT parity는 두 모델 prediction/features lock 후 연결했다. 키포인트 optimizer step=0, 신규 selector fit=2.\n\n'
    text+=C.table(['Model-selector','Correct / 1024','Accuracy','Valid pairs','Brier(valid pairs)'],[[k,v['correct'],v['accuracy'],v['valid_pairs'],v['brier']] for k,v in out.items()])
    text+='\nTEST는 이미 연구에 사용된 synthetic development-heldout이며 독립 증거가 아니다. TEST 후 재학습·조정하지 않는다. 점수 margin은 selector마다 척도가 달라 confidence의 직접 비교가 아니다. Feature shift는 고정 old TRAIN 표준편차로 정규화한 기술통계이며 특징 재선택에 쓰지 않았다.\n'
    for p in sorted((C.stage(2)/'figures').glob('*.png')):text+=f'\n![{p.stem}](figures/{p.name})\n'
    C.save(C.stage(2)/'STAGE2_REPORT_KO.md',text)

if __name__=='__main__':main()
