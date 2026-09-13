"""Close both topics independently; retain failed gates and original results."""
import json
from pathlib import Path
import numpy as np
import torch
import selective as G
import evaluate_reweight as V
from experiment import E,S,ROOT,RAW,DOC,read,write,sha,METHODS,SEEDS

def markdown(path,lines):
    value='\n'.join(lines)+'\n'
    if path.exists():assert path.read_text()==value
    else:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)

def selective_report():
    d=read(G.OUT/'RESULTS.json');lock=read(G.OUT/'THRESHOLD_LOCK.json')
    feature_audit=read(G.OUT/'FEATURE_AUDIT.json')
    assert sha(G.BASE/'FEATURES.json')==feature_audit['feature_sha256']
    features=read(G.BASE/'FEATURES.json');normalization=read(G.OUT/'STANDARDIZATION.json')
    train_x=np.array([features[k]['features'] for k in normalization['train_ids']],np.float32)
    np.testing.assert_array_equal(train_x.mean(0),normalization['mean'])
    np.testing.assert_array_equal(np.maximum(train_x.std(0),1e-6),normalization['scale'])
    test=read(G.BASE/'TEST_LABELS.json');scores=read(G.BASE/'TEST_SCORES.json')
    ids=[r['frame_id'] for r in read(DOC/'SPLIT.json')['evaluation']]
    y=np.array([test[k]['unsafe'] for k in ids]);valid=np.array([test[k]['valid'] for k in ids])
    for name,score in scores.items():
        calc=G.curve(score,y,valid);actual=d['results'][name]
        assert calc['aurc']==actual['aurc'] and calc['curve']==actual['curve']
        threshold=G.select_threshold(lock['calibration_scores'][name],lock['calibration_unsafe'],lock['calibration_valid'])
        assert threshold==lock['thresholds'][name]
        keep=np.zeros(len(ids),bool) if threshold['threshold'] is None else valid&(np.array(score)<=threshold['threshold'])
        assert int(keep.sum())==actual['accepted'] and int(y[keep].sum())==actual['unsafe_accepted']
    for name in ('logistic','mlp_seed1','mlp_seed2','mlp_seed3'):
        a=read(G.OUT/f'{name}_TRAIN.json');assert a['updates']==1000 and a['checkpoint_sha256']==sha(G.BASE/f'{name}.pt')
        ck=torch.load(G.BASE/f'{name}.pt',map_location='cpu')
        assert ck['steps']==1000 and all(torch.isfinite(v).all() for v in ck['state_dict'].values())
        model=G.gate_model('logistic' if name=='logistic' else 'mlp');model.load_state_dict(ck['state_dict']);model.eval()
        for selected_ids,expected in ((ids,scores[name]),(lock['calibration_ids'],lock['calibration_scores'][name])):
            x=(np.array([features[k]['features'] for k in selected_ids],np.float32)-ck['mean'])/ck['scale']
            with torch.no_grad():actual=model(torch.tensor(x)).flatten().numpy()
            np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-7)
    artifacts={str(p.relative_to(ROOT)):sha(p) for p in [G.BASE/'FEATURES.json',G.BASE/'TRAIN_CAL_LABELS.json',G.BASE/'TEST_LABELS.json',G.BASE/'TEST_SCORES.json',G.OUT/'THRESHOLD_LOCK.json']}
    write(G.OUT/'FINAL_AUDIT.json',dict(execution='COMPLETE',integrity='PASS',scientific_verdict=d['verdict'],
        gate_fits=4,gate_updates=4000,R0_updates=0,train=112,calibration=62,evaluation=145,
        prediction_only_features=True,calibration_threshold_recomputed=True,tie_invariant_AURC_recomputed=True,
        canonical_MAIN_pose_parity=True,artifacts=artifacts))
    lines=['# 포즈 사용·거부 판단 학습 — 완료','',f"판정: `{d['verdict']}`.",
        '고정 R0 포즈를 입력으로 하는 별도 분류기를 실제로 학습했다. 이전의 선 보정 선택기가 아니다.',
        '선형 모델 1개와 작은 MLP 3개를 각1000회, 총4000회 업데이트했다. R0 자체는 업데이트하지 않았다.',
        '112장 학습/62장 보정/145장 평가. 기존 GT-v2 기반 retrospective development이며 독립 안전 인증이 아니다.',
        '', '| 방법 | AURC↓ | 수용/145 | 수용률 | 잘못 수용한 비율 |','|---|---:|---:|---:|---:|']
    for name,r in d['results'].items():
        risk='정의되지 않음' if r['unsafe_risk'] is None else f"{r['unsafe_risk']:.4f}"
        lines.append(f"| {name} | {r['aurc']:.6f} | {r['accepted']} | {r['coverage']:.4f} | {risk} |")
    lines+=['', 'AURC는 신뢰 순서로 수용 범위를 늘릴 때의 위험을 요약한다. 동일 점수 tie는 무작위 순열의 기대 위험으로 처리한다. 모든 점수를 거부하는 운영 문턱과 무관하게 순위 품질은 전체 곡선으로 평가했다.',
        '정답상 부적합 조건: MAIN 위치 오차>10cm 또는 yaw>5° 또는 검출 IoU<0.5 또는 포즈 산출 실패. 이는 사전 고정한 탐색용 기준이지 실제 현장의 허용 오차나 인증 기준이 아니다.',
        '보정 데이터에서 잘못 수용하는 비율≤10%, 최소10장 수용을 만족하는 문턱만 허용했다. 만족하는 문턱이 없으면 전부 거부하며, 이를 위험0의 성공으로 처리하지 않는다.',
        f"평가145장 중 부적합은{d['unsafe_total']}장이다. 전체 수용의 부적합 비율은{d['always_accept_risk']:.4f}다.",
        '입력은 예측 신뢰도, 정규화된 박스/키포인트, 알려진 카메라/물체 규격과 예측 기반 PnP 정보뿐이다. GT 오차, 실제 축, session ID와 파일명은 입력하지 않았다.',
        '이 데이터와 고정 모델 설정에서 신경망의 우위를 확인하지 못했다. 임계값 완화, 특징 추가, 학습량 변경으로 재시도하지 않았다. 배경/팔레트 없음에서의 잘못 수용 위험은 이번 양성 전용 실험으로 검증되지 않았다.']
    markdown(G.OUT/'REPORT_KO.md',lines)

def reweight_report():
    V.audit();result={};rows={};artifacts={}
    for name in ['R0',*[f'{m}_seed{s}' for s in SEEDS for m in METHODS]]:
        d=V.W.BASE/'evaluation'/name;result[name]=read(d/'RESULT.json');rows[name]=read(d/'PER_FRAME.json')
        assert result[name]['actual_evaluation_positive']==145 and result[name]['negative_count']==2689
        keys={k for k,r in rows[name].items() if r['matched'] and r['errors_px']}
        assert E.geometry(rows[name],keys)==result[name]['geometry']
        r=result[name]['metrics'];assert abs(float(np.mean(list(r['box_ap_by_iou'].values())))-r['box_ap50_95'])<1e-12
        artifacts[name]={str(p.relative_to(ROOT)):sha(p) for p in [d/'RESULT.json',d/'PER_FRAME.json',d/'PREDICTIONS.json',d/'ACTUAL_POSE_BINDING.json']}
    paired=[]
    for seed in SEEDS:
        common=set.intersection(*[{k for k,r in rows[f'{m}_seed{seed}'].items() if r['matched'] and r['errors_px']} for m in METHODS])
        paired.append(dict(seed=seed,common_ids=sorted(common),geometry={m:E.geometry(rows[f'{m}_seed{seed}'],common) for m in METHODS}))
    means={}
    for m in METHODS:
        rr=[result[f'{m}_seed{s}'] for s in SEEDS]
        means[m]=dict(ap50_95=float(np.mean([r['metrics']['box_ap50_95'] for r in rr])),
            Det=float(np.mean([r['Det'] for r in rr])),
            geometry={k:float(np.mean([p['geometry'][m][k] for p in paired])) for k in ('median_px','p90_px','gross20')},
            pose={k:float(np.mean([r['pose']['ALL'][k] for r in rr])) for k in ('translation_median_cm','yaw_median_deg')},
            pose_coverage=float(np.mean([r['pose']['coverage'] for r in rr])))
    proposed=means['meta_weight'];gates={}
    for control in ('uniform','hard_loss'):
        c=means[control]
        gates[control]=dict(primary_better=proposed['geometry']['median_px']<c['geometry']['median_px'],
            two_of_three=sum(p['geometry']['meta_weight']['median_px']<p['geometry'][control]['median_px'] for p in paired)>=2,
            p90_nonworse=proposed['geometry']['p90_px']<=c['geometry']['p90_px'],gross20_nonworse=proposed['geometry']['gross20']<=c['geometry']['gross20'],
            AP_nonworse=proposed['ap50_95']>=c['ap50_95'],Det_nonworse=proposed['Det']>=c['Det'],
            pose_safe=all(proposed['pose'][k]<=1.1*c['pose'][k] for k in proposed['pose']) and proposed['pose_coverage']>=c['pose_coverage'])
    passed=all(v for block in gates.values() for v in block.values());verdict='SAMPLE_REWEIGHT_SIGNAL' if passed else 'SAMPLE_REWEIGHT_NO_SIGNAL'
    write(DOC/'reweight/RESULTS.json',dict(per_run=result,paired=paired,means=means,gates=gates,verdict=verdict))
    write(DOC/'reweight/EVALUATION_AUDIT.json',dict(status='PASS',artifacts=artifacts,independent_geometry_recomputation=True,
        pose_frame_id_binding=read(V.W.BASE/'evaluation_inputs/FRAME_ID_BINDING.json')))
    lines=['# 학습 샘플 중요도 학습 — 완료','',f'판정: `{verdict}`.','',
        '112장 학습/62장 메타/145장 평가. 합성1440장 replay. 균등/현재 손실 비례/메타 가중치의3방법×3seed×300회, 총2700회 실제 학생 업데이트를 수행했다.',
        '메타 가중치는 정해진 이미지 점수가 아니라 현재 학생의 메타 손실로부터 매 step 학습하는 샘플 가중치다. 다만 별도 weight MLP가 아니라 마지막 키포인트(x/y/visibility) 투영층7452개 파라미터 부분공간의 1차 meta-gradient다. 전체 파라미터 Ren 재현이나 신규성 증명으로 부르지 않는다.',
        'w_i = 8 max(0, grad_proxy(L_i) dot grad_proxy(L_meta)) / sum_j max(0, alignment_j). 모든 값이 비양수이면 real 가중치를0으로 두고 synthetic만 업데이트한다. 최종 weighted real loss는 모델 전체의 trainable 파라미터를 업데이트한다.',
        '균등과 현재 손실 비례 비교군도 동일한 메타 계산을 수행하되 가중치에 사용하지 않는다. 모든 arm의 synthetic/real/meta 실제 입력 hash, R0 초기값, 고정 BatchNorm 통계를 검증했다. Real/meta 혼합 증강은 끄고 표본별 identity를 유지했다.',
        '', '| 방법 | AP50-95 | 공통 kp median/P90 px | 위치 cm | yaw deg |','|---|---:|---|---:|---:|']
    for m,r in means.items():lines.append(f"| {m} | {r['ap50_95']:.6f} | {r['geometry']['median_px']:.4f}/{r['geometry']['p90_px']:.4f} | {r['pose']['translation_median_cm']:.4f} | {r['pose']['yaw_median_deg']:.4f} |")
    lines+=['','세 seed 통계의 평균이다. 키포인트는 seed별 세 방법 공통 검출 프레임을 쓴다.','']
    lines += [f'- {c}: '+', '.join(f'{k}={"PASS" if v else "FAIL"}' for k,v in d.items()) for c,d in gates.items()]
    base=result['R0']
    lines+=['', f"같은 평가 subset의 추가 학습 전 R0 참고값: AP50-95={base['metrics']['box_ap50_95']:.6f}, 개별 matched kp median/P90={base['geometry']['median_px']:.4f}/{base['geometry']['p90_px']:.4f}px, 위치={base['pose']['ALL']['translation_median_cm']:.4f}cm, yaw={base['pose']['ALL']['yaw_median_deg']:.4f}°. 키포인트의 R0 개별 matched 분모와 위 세 방법 공통 분모는 다르므로 직접 차감하지 않는다. 고정 gate는 두 추가학습 비교군 대비 신호 검사이며 R0 교체 승인이 아니다."]
    lines+=['', '기존 active-learning30-label 실험과 데이터/BN/real 손실 집계가 달라 그 결과와의 차이를 importance 단독 효과로 해석하지 않는다.',
        'Stock PoseLoss26의 visibility1은 감독을 유지한다. 개별 real 샘플 loss 합은 원래 batch 통합 정규화와 다를 수 있지만 세 arm 모두 동일하다. 모든 체크포인트는 마지막 step만 저장했으며 seed 선택이나 재학습은 하지 않았다.',
        '기존 optional Albumentations API 불일치로 해당 optional transform은 미적용이다. cuBLAS 엄밀 결정론 경고가 있었으며 bit-exact 학습을 주장하지 않는다. 실제 증강 입력 parity는 직접 검증했다.',
        'Repeated development이며 새 센서 GT나 독립 검증이 아니다. 실패하더라도 모든 sample reweighting이 불가능하다는 결론은 아니다.']
    markdown(DOC/'reweight/REPORT_KO.md',lines)
    return verdict

def main():
    selective_report();verdict=reweight_report()
    lock=read(DOC/'PROTOCOL_LOCK.json')
    for p,h in lock['source_bindings'].items():assert sha(ROOT/p)==h
    preserved=read(ROOT/'_docs/experiments/pallet_paper_contribution_screen_v1/PRESERVED_SOURCE_SHA.json')
    assert all(sha(ROOT/p)==h for p,h in preserved.items())
    selected=read(G.OUT/'RESULTS.json')['verdict']
    write(DOC/'FINAL_AUDIT.json',dict(execution='BOTH_REMAINING_TOPICS_COMPLETE',integrity='PASS',
        sample_importance=verdict,pose_acceptance=selected,student_fits=9,student_updates=2700,gate_fits=4,gate_updates=4000,
        total_new_optimizer_updates=6700,smoke_optimizer_updates=0,
        source_bindings_unchanged=True,older_preserved_sources=len(preserved),
        independent_confirmation=False,novelty_claim=False,R0_replaced=False,paper_final_modified=False,
        implementation_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}))
    markdown(DOC/'REPORT_KO.md',['# 세 주제 실행 상태','',
        '| 주제 | 실행 | 판정 |','|---|---|---|',
        '| 능동학습/데이터 선택 | 이전12학생 학습·평가 완료 | RETROSPECTIVE_AL_NO_SIGNAL |',
        f'| 학습 샘플 중요도 | 이번9학생 학습·평가 완료 | {verdict} |',
        f'| 포즈 사용·거부 판단 | 이번4 gate 학습·평가 완료 | {selected} |','',
        '이번에는 남은 두 주제 모두 실제 학습과 평가를 끝냈다. 실행 완료와 과학적 성공은 구분한다.',
        '세 주제 모두 기존 데이터를 활용한 bounded development 결과이며 논문 기여의 신규성/독립 일반화 입증은 아니다.',
        '', '[샘플 중요도 상세](reweight/REPORT_KO.md) · [포즈 판단 상세](selective/REPORT_KO.md) · [무결성 기록](FINAL_AUDIT.json)'])
    print('BOTH COMPLETE',verdict,selected,flush=True)

if __name__=='__main__':
    import sys
    if '--selective-only' in sys.argv:selective_report()
    else:main()
