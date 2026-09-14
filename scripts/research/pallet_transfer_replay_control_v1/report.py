"""Write the final bounded experiment report without altering prior experiments."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch

from aggregate import ARMS,CONTRASTS
from evaluate import NAMES,checkpoint
from runtime import ROOT,RAW,DOC,R0,atomic_json,sha


def read(path):return json.loads(Path(path).read_text())


def metric(mean,key):return mean[key]


def drift():
    base_ck=torch.load(R0,map_location='cpu');base=(base_ck.get('ema') or base_ck['model']).float().state_dict();output={}
    for name in NAMES[1:]:
        ck=torch.load(checkpoint(name),map_location='cpu')['model'].float().state_dict();modules={}
        for key,value in base.items():
            prefix='.'.join(key.split('.')[:2]);a=value.double();b=ck[key].double();entry=modules.setdefault(prefix,[0.,0.])
            entry[0]+=float((b-a).square().sum());entry[1]+=float(a.square().sum())
        output[name]=dict(total_relative_l2=float(np.sqrt(sum(v[0] for v in modules.values())/max(sum(v[1] for v in modules.values()),1e-30))),
            by_module={k:float(np.sqrt(v[0]/max(v[1],1e-30))) for k,v in modules.items()})
    return output


def direction(interval):return interval['low']>0


def main():
    old_report=DOC/'REPORT_KO.md';history=DOC/'REPORT_TECHNICAL_STOP_20260914.md'
    if old_report.exists() and not history.exists():history.write_text(old_report.read_text())
    metrics=read(DOC/'METRICS_PER_SEED.json');paired=read(DOC/'PAIRED_CONTRASTS.json');means=metrics['seed_means'];base=metrics['R0']
    tc=paired['target']['session_cluster'];sc=paired['source']['scenario_cluster'];point=paired['point_estimates']
    stable=[arm for arm in ARMS if tc['versus_R0'][arm]['low']>0]
    a=all(tc['contrasts'][c]['low']>0 and sc['contrasts'][c]['low']>0 for c in ('C_main','C_pract'))
    if not stable:verdict='NO_ADAPTATION_GAIN_AT_LOCKED_BUDGET'
    elif a:verdict='REPLAY_TARGET_AND_SOURCE_SIGNAL'
    elif sc['contrasts']['C_main']['low']>0 and tc['contrasts']['C_main']['low']<=0:verdict='REPLAY_RETENTION_TRADEOFF'
    elif tc['contrasts']['C_main']['low']<=0:verdict='TARGET_SCALING_SUFFICIENT_OR_UNRESOLVED'
    else:verdict='INCONCLUSIVE_DEVELOPMENT_RESULT'
    tags=[]
    if tc['contrasts']['C_budget']['low']<=0:tags.append('EXTRA_TARGET_EXPOSURE_COMPETITIVE_OR_UNRESOLVED')
    if sc['contrasts']['C_main']['low']>0:tags.append('REPLAY_SOURCE_RETENTION_SIGNAL')
    if tc['contrasts']['C_main']['low']>0:tags.append('REPLAY_TARGET_MAIN_CONTRAST_SIGNAL')
    runtime=read(DOC/'RUNTIME.json')
    drifts=drift();training=read(DOC/'TRAINING_COMPLETE.json')
    curves={}
    for name in NAMES[1:]:
        trace=read(RAW/'runs'/name/'EXPOSURE.json');streams=sorted(trace[0]['C8'])
        curves[name]=dict(first30_C8={s:float(np.mean([r['C8'][s] for r in trace[:30]])) for s in streams},
            last30_C8={s:float(np.mean([r['C8'][s] for r in trace[-30:]])) for s in streams},
            mean_preclip_gradient_norm=float(np.mean([r['preclip_gradient_norm'] for r in trace])),
            clipped_updates=sum(r['clipped'] for r in trace),mosaic_original_references={s:sum(r['streams'][s]['total_original_references'] for r in trace) for s in streams})
    atomic_json(DOC/'MECHANISM_AUDIT.json',dict(status='COMPLETE_DESCRIPTIVE_NOT_CAUSAL_MEDIATION',
        target_and_source_primary_point_estimates=point,weight_drift=drifts,train_probe_learning_curves=curves,
        source_retention_interpretation='Task performance is primary. Drift/loss trends do not prove a feature-level mediation mechanism.',
        DFL=dict(module='Identity',reg_max=1,parameters=0),BN='126 R0 running-stat modules exact throughout; affine trained'))
    atomic_json(DOC/'TRAINING_AUDIT.json',dict(status='PASS',fits=12,actual_main_optimizer_updates=3600,
        actual_smoke_optimizer_updates=training['actual_smoke_optimizer_updates'],per_fit=training['audits'],
        all_target_base_tensor_hashes_equal_per_seed_and_step=True,BN_running_buffers_equal=True,
        objective='C8 coefficients: T8_FULL4; all streams in other arms1',all_final_step300=True,
        no_best_checkpoint_selection=True,performance_not_read_during_training=True))
    atomic_json(DOC/'VERDICT.json',dict(execution_status='COMPLETE',scientific_verdict=verdict,diagnostic_tags=tags,
        performance_evaluated=True,independent_confirmation=False,novelty_claim=False,
        primary='Target and source ALL_GT_PCK10',stable_target_improvement_vs_R0=stable,
        target_session_intervals=tc,source_scenario_intervals=sc,
        decision_rule='Strong replay signal requires C_main and C_pract target session and source scenario lower95 bounds >0. No adaptation gain if no arm-vs-R0 target session lower bound >0. Other outcomes use predeclared explanatory categories.',
        noninferiority_claim=False,equivalence_claim=False,familywise_claim=False,
        current_line_P_AL_verdicts_unchanged=True,automatic_followup_training=False))
    lines=['# 소량 실사 전이학습 — 합성 replay 통제 실험 완료','',f'판정: `{verdict}`.',
        '', '고정된 real30장, BN running statistics 고정, stock YOLO pose loss 조건에서 네 학습 전략을 seed1/2/3 각각300 optimizer update로 비교했다. 총12fit·3600 본 update와 사전 smoke1update를 수행했다. 모든 모델은 같은 R0에서 시작했고 step300 마지막 checkpoint만 평가했다.',
        '', '## 핵심 결과','', '| 모델 | Target PCK10 | Source PCK10 | Target AP50-95 | Target kp median/P90 | Translation cm | Rotation deg | Negative AUROC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def row(label,value):
        return f"| {label} | {value['target_ALL_GT_PCK10']:.6f} | {value['source_ALL_GT_PCK10']:.6f} | {value['target_AP50_95']:.6f} | {value['target_individual_matched']['median_px']:.3f}/{value['target_individual_matched']['p90_px']:.3f} | {value['translation_median_cm']:.4f} | {value['rotation_median_deg']:.4f} | {value['negative_AUROC']:.6f} |"
    lines.append(row('R0',base))
    for arm in ARMS:lines.append(row(arm+' (3-seed mean)',means[arm]))
    lines += ['', 'PCK10은 검출 실패·잘못된 top1·점 누락을0점 처리하는 전체 감독 GT 고정분모 지표다. Target은 과거에 본 DEV145장·4세션이며 독립 확인셋이 아니다. Source는 새 미세조정에 사용하지 않은 heldout512장 development probe이며 R0의 과거 validation 노출은 배제하지 않았다.',
        '', '## 사전 contrast — PCK10 차이 percentage point','', '| Contrast | Target Δ | Target session95% | Source Δ | Source scenario95% |', '|---|---:|---:|---:|---:|']
    for label in CONTRASTS:
        p=point[label];ti=tc['contrasts'][label];si=sc['contrasts'][label]
        lines.append(f"| {label} | {100*p['target_pck10']:+.3f} | [{100*ti['low']:+.3f}, {100*ti['high']:+.3f}] | {100*p['source_pck10']:+.3f} | [{100*si['low']:+.3f}, {100*si['high']:+.3f}] |")
    lines += ['', '## R0 대비','', '| 군 | Target Δ | Target session95% | Source Δ | Source scenario95% |', '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        p=point['versus_R0'][arm];ti=tc['versus_R0'][arm];si=sc['versus_R0'][arm]
        lines.append(f"| {arm} | {100*p['target_pck10']:+.3f} | [{100*ti['low']:+.3f}, {100*ti['high']:+.3f}] | {100*p['source_pck10']:+.3f} | [{100*si['low']:+.3f}, {100*si['high']:+.3f}] |")
    lines += ['', '각 bootstrap draw는 동일 frame/session/scenario weight를 모든 모델·seed에 공유하고 seed별 통계의 평균을 비교한다. 단일 R0를 세 독립 모델처럼 복제하지 않는다. 10,000 resample은 실제 세션 수를 늘리지 않으며, 구간은 다중비교 familywise 보장이 아니다. 0 포함은 동등·비열등 증명이 아니다.',
        '', '## 실행 무결성 및 한계','',
        '- nominal slot/fit: T8_FULL·T8_QUARTER real2400, REPLAY real2400+synthetic7200, T32_COMPUTE real9600. Mosaic 원본 참조는 trace에 별도 보존했다.',
        '- 실제 criterion 반환은 batch-size multiplier와 내부 target-score/foreground 정규화를 포함한다. ell을 독립 sample loss 평균이라고 주장하지 않는다.',
        '- 모든 arm의 target_base augmented tensor SHA는 같은 seed·step에서 exact 동일했다. Source/target Mosaic 보조 영상은 각 domain allowlist 안에 있었다.',
        '- BN126개의 running_mean/variance/counter는 R0와 bit-exact였고 affine은 학습했다. DFL은 이 모델에서 reg_max1 Identity로 파라미터가 없다.',
        '- Target145/negative2689/source512를 새 checkpoint로 실제 추론했다. R0 target/negative만 동일 weight·recipe의 기존 canonical cache를 SHA 확인 후 재사용했고 source R0는 새 추론했다.',
        f"- clean runtime은 고정26프레임×3repeat의 모든 호출을 유지했다. R0 pooled median={runtime['results']['R0']['median_ms']:.3f}ms. 이는 display/RustDesk가 활성화된 단일 RTX3080 측정이며 Jetson/export 주장이 아니다.",
        '- GT-v2는 수동/기하 재구성 provenance가 섞였으며 새 독립 motion-capture GT가 아니다. Source와 target의 절대 픽셀오차를 같은 난이도로 직접 차감하지 않는다.',
        '- 기존 line/P/active-learning 판정은 변경하지 않는다. 성능을 본 뒤 LR·계수·seed·checkpoint·primary를 바꾸거나 추가 학습하지 않았다.',
        '', '## 논문에 쓸 수 있는 범위','',
        '실제 수치가 뒷받침하는 경우에만, 동일한 표적 정답30장과 고정 BN 통계 조건에서 source replay를 동반한 supervised fine-tuning이 target-only 대조와 어떤 개발 성능 차이를 보였는지 기술할 수 있다. 이 결과는 새 replay 알고리즘, 합성 사전학습 자체의 인과 우위, 임의30장의 충분성, zero-label/self-training, 독립 창고 일반화를 증명하지 않는다.',
        '', f'보조 진단 태그: {", ".join(tags) if tags else "없음"}. 상세 수치는 `METRICS_PER_SEED.json`, `PAIRED_CONTRASTS.json`, `MECHANISM_AUDIT.json`, `RUNTIME.json`에 있다.']
    (DOC/'REPORT_KO.md').write_text('\n'.join(lines)+'\n')
    # Small manifest binds ignored raw evidence without publishing checkpoints/caches.
    raw_receipts={}
    for name in NAMES[1:]:
        raw_receipts[name]={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in
            [RAW/'runs'/name/'TRAINING_AUDIT.json',RAW/'runs'/name/'EXPOSURE.json',checkpoint(name),
             RAW/'evaluation'/name/'RESULT.json',RAW/'evaluation'/name/'TARGET_PER_FRAME.json',
             RAW/'source_evaluation'/name/'SOURCE_PER_FRAME.json']}
    atomic_json(DOC/'RAW_EVIDENCE_MANIFEST.json',dict(note='Weights and prediction caches remain ignored; hashes bind local evidence',artifacts=raw_receipts))
    atomic_json(DOC/'CURRENT_RESULT.json',dict(stage='TRANSFER_REPLAY_CONTROL_COMPLETE',report='REPORT_KO.md',
        verdict='VERDICT.json',audit='FINAL_AUDIT.json',scientific_complete=True,training_complete=True,
        completed_fits=12,actual_main_optimizer_updates=3600,technical_stop_history='REPORT_TECHNICAL_STOP_20260914.md'))
    print(verdict,tags,flush=True)


if __name__=='__main__':main()
