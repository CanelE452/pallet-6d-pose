"""Close the fixed retrospective comparison without tuning or selecting a seed."""
import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('sim_eval',Path(__file__).with_name('simulation_evaluate.py'))
E=importlib.util.module_from_spec(spec);spec.loader.exec_module(E)
S=E.S;RAW,DOC,ROOT,read,write,sha=E.RAW,E.DOC,E.ROOT,E.read,E.write,E.sha

def main():
    E.training_audit();result={};rows={};artifacts={}
    baseline_source=ROOT/'data/pallet/results/pallet_line_pose_v1/baseline/FULL_CANDIDATES.json'
    baseline_cache=read(baseline_source)
    assert baseline_cache['complete'] and baseline_cache['weights_sha256']==sha(S.R0)
    assert baseline_cache['recipe']['input_size']==640 and baseline_cache['recipe']['pad_px']==100
    assert baseline_cache['recipe']['confidence_floor']==.001
    for name in ['R0',*[f'{m}_seed{s}' for s in (1,2,3) for m in S.METHODS]]:
        d=RAW/'evaluation'/name;result[name]=read(d/'RESULT.json');rows[name]=read(d/'PER_FRAME.json')
        assert result[name]['actual_evaluation_positive']==len(read(DOC/'SPLIT.json')['evaluation'])
        matched=[k for k,r in rows[name].items() if r['matched'] and r['errors_px']]
        independent=E.geometry(rows[name],matched)
        assert independent==result[name]['geometry']
        ap=result[name]['metrics']
        assert abs(np.mean(list(ap['box_ap_by_iou'].values()))-ap['box_ap50_95'])<1e-12
        cache=read(d/'PREDICTIONS.json')
        expected=S.R0 if name=='R0' else RAW/'runs'/name/'last.pt'
        assert cache['complete'] and cache['weights_sha256']==sha(expected)
        if name=='R0':
            assert all(v==baseline_cache['frames'][k] for k,v in cache['frames'].items())
        artifacts[name]={str(p.relative_to(ROOT)):sha(p) for p in (
            d/'RESULT.json',d/'PER_FRAME.json',d/'PREDICTIONS.json',
            d/'ACTUAL_POSE_BINDING.json',d/f'POSE_EVALUATION_{name}.json')}
    paired=[];means={};contrasts=[]
    for seed in (1,2,3):
        names=[f'{m}_seed{seed}' for m in S.METHODS]
        common=set.intersection(*[{k for k,r in rows[n].items() if r['matched'] and r['errors_px']} for n in names])
        paired.append(dict(seed=seed,common_frames=len(common),geometry={m:E.geometry(rows[f'{m}_seed{seed}'],common) for m in S.METHODS}))
    for m in S.METHODS:
        rr=[result[f'{m}_seed{s}'] for s in (1,2,3)]
        means[m]=dict(ap50_95=float(np.mean([r['metrics']['box_ap50_95'] for r in rr])),
            ap50=float(np.mean([r['metrics']['box_ap50'] for r in rr])),
            Det=float(np.mean([r['Det'] for r in rr])),auroc=float(np.mean([r['auroc'] for r in rr])),
            fpr95=float(np.mean([r['fpr95'] for r in rr])),pose_coverage=float(np.mean([r['pose']['coverage'] for r in rr])),
            geometry={k:float(np.mean([p['geometry'][m][k] for p in paired])) for k in ('median_px','p90_px','gross20')},
            pose={k:float(np.mean([r['pose']['ALL'][k] for r in rr])) for k in ('translation_median_cm','yaw_median_deg','rotation_median_deg','iou3d_median','add_sym_auc')})
    proposed=means['geometry_weighted_diversity'];gates={}
    for control in ('random','diversity'):
        c=means[control]
        directions=[p['geometry']['geometry_weighted_diversity']['median_px']<p['geometry'][control]['median_px'] for p in paired]
        gates[control]=dict(primary_improves=proposed['geometry']['median_px']<c['geometry']['median_px'],
            two_of_three_seeds=sum(directions)>=2,p90_nonworse=proposed['geometry']['p90_px']<=c['geometry']['p90_px'],
            gross20_nonworse=proposed['geometry']['gross20']<=c['geometry']['gross20'],
            Det_nonworse=proposed['Det']>=c['Det'],
            pose_safe=all(proposed['pose'][k]<=1.1*c['pose'][k] for k in ('translation_median_cm','yaw_median_deg')) and proposed['pose_coverage']>=c['pose_coverage'])
        contrasts.append(dict(control=control,median_delta_px=proposed['geometry']['median_px']-c['geometry']['median_px'],
            p90_delta_px=proposed['geometry']['p90_px']-c['geometry']['p90_px'],seed_primary_improvements=directions))
    passed=all(v for g in gates.values() for v in g.values())
    verdict='RETROSPECTIVE_AL_SIGNAL' if passed else 'RETROSPECTIVE_AL_NO_SIGNAL'
    write(DOC/'RESULTS.json',dict(per_run=result,paired=paired,seed_means=means,contrasts=contrasts))
    write(DOC/'VERDICT.json',dict(verdict=verdict,status='PASS' if passed else 'FAIL',gates=gates,
        evidence='RETROSPECTIVE_DEVELOPMENT',independent_confirmation=False,novelty_claim=False,
        further_training_authorized_by_result=False,R0_replacement=False))
    lock=read(DOC/'PROTOCOL_LOCK.json')
    for p,h in lock['source_bindings'].items():assert sha(ROOT/p)==h
    labels=read(DOC/'LABEL_REVEAL_AUDIT.json')
    for method,rr in labels['bindings'].items():
        for j,r in enumerate(rr):
            assert sha(ROOT/r['source_label'])==r['source_label_sha256']
            exported=RAW/'dataset'/method/'labels'/Path(f"{j:04}_{Path(r['image']).name}").with_suffix('.txt')
            assert sha(exported)==r['exported_label_sha256']
    preserved=read(ROOT/'_docs/experiments/pallet_paper_contribution_screen_v1/PRESERVED_SOURCE_SHA.json')
    assert all(sha(ROOT/p)==h for p,h in preserved.items())
    write(DOC/'EVALUATION_AUDIT.json',dict(status='PASS',artifacts=artifacts,
        R0_cache_source=str(baseline_source.relative_to(ROOT)),R0_cache_source_sha256=sha(baseline_source),
        R0_cache_recipe=baseline_cache['recipe'],R0_cache_subset_exact=True,
        frame_id_binding=read(RAW/'evaluation_inputs/FRAME_ID_BINDING.json'),
        frame_id_binding_sha256=sha(RAW/'evaluation_inputs/FRAME_ID_BINDING.json'),
        implementation_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('simulation*.py')},
        exported_labels_unchanged=True,
        wrapper_correction='Initial pose wrapper stopped at membership assertion before pose scoring: 2D IDs use colon, pose IDs use double underscore. Corrected by one-to-one canonical image path join with frozen image SHA verification. No labels, model weights, split, metric formula or training changed.'))
    write(DOC/'FINAL_AUDIT.json',dict(execution='COMPLETE',integrity='PASS',scientific_verdict=verdict,
        fits=12,updates_per_fit=300,total_optimizer_updates=3600,labels_per_method=30,
        selected_label_union=labels['selected_union'],selection_pool=lock['pool_frames'],
        evaluation_positive=lock['evaluation_positive_frames'],evaluation_negative=lock['negative_frames'],
        actual_augmented_synthetic_parity=True,init_state_parity=True,independent_saved_score_recomputation=True,
        source_labels_unchanged=True,preserved_sources=len(preserved),paper_final_modified=False,
        historical_DEV_repurposing='Only isolated simulator uses its pool subset for training. Existing benchmark artifacts preserved; these students cannot be reported on fullDEV319 as untrained evaluation.',
        original_C_or97frame_pilot_changed=False))
    write(S.AL.DOC/'CURRENT_RESULT.json',dict(stage='RETROSPECTIVE_TRAINING_COMPLETE',
        report='retrospective_v1/REPORT_KO.md',verdict='retrospective_v1/VERDICT.json',
        audit='retrospective_v1/FINAL_AUDIT.json',scientific_verdict=verdict,
        initial_unlabeled_preview='REPORT_KO.md and97-image queue remain historical acquisition-only outputs'))
    lines=['# 모의 active learning — 실제 학습 비교 완료','',
        f'판정: `{verdict}`. 기존 데이터를 활용한 retrospective development 결과이며 독립 일반화·신규성 증명이 아니다.','',
        f"정본319장을 원본 촬영 연결까지 고려해 선택용{lock['pool_frames']}장/평가용{lock['evaluation_positive_frames']}장으로 분리했다. 평가용 GT는 모든 학습 완료 후에만 채점에 사용했다.",
        '네 방법이 각각 기존 정답30장을 선택했고, 선택 합집합83장의 정답만 학습용으로 내보냈다. 4방법×3seed×300회, 총3600 optimizer update를 수행했다. 같은 R0/stock pose loss/학습량/최종 체크포인트 규칙을 사용했다.',
        '실제 synthetic 증강 입력 hash는 같은 seed의 네 방법에서 모두 일치했다. Real은 선택 이미지가 다르지만 nominal batch slots와 위치별 증강 seed는 같았다. Mosaic의 추가 source 사용을 nominal batch-slot 수와 혼동하지 않는다.',
        '', '## 세 seed 평균','',
        '| 선택 방법 | AP50-95 | Det | 공통 kp median/P90 px | translation cm | yaw deg | pose coverage |',
        '|---|---:|---:|---|---:|---:|---:|']
    for m,r in means.items():
        lines.append(f"| {m} | {r['ap50_95']:.6f} | {r['Det']:.6f} | {r['geometry']['median_px']:.4f}/{r['geometry']['p90_px']:.4f} | {r['pose']['translation_median_cm']:.4f} | {r['pose']['yaw_median_deg']:.4f} | {r['pose_coverage']:.4f} |")
    lines+=['','키포인트 primary는 각 seed의 네 방법 공통 검출 프레임에서 계산한 pooled median이다. 세 seed 평균은 해당 통계의 평균이며 독립 세션 수를 세 배로 늘리지 않는다.',
        '', '## 모든 seed','', '| Seed | 방법 | AP50-95 | 공통 median/P90 px | translation cm | yaw deg |','|---|---|---:|---|---:|---:|']
    for p in paired:
        for m in S.METHODS:
            r=result[f"{m}_seed{p['seed']}"];g=p['geometry'][m]
            lines.append(f"| {p['seed']} | {m} | {r['metrics']['box_ap50_95']:.6f} | {g['median_px']:.4f}/{g['p90_px']:.4f} | {r['pose']['ALL']['translation_median_cm']:.4f} | {r['pose']['ALL']['yaw_median_deg']:.4f} |")
    lines+=['', '## 고정 판정 조건','']
    lines += [f'- {c}: '+', '.join(f'{k}={"PASS" if v else "FAIL"}' for k,v in g.items()) for c,g in gates.items()]
    base=result['R0'];lines+=['','## R0 참고값 및 해석 제한','',
        f"같은 평가 subset의 R0: AP50-95={base['metrics']['box_ap50_95']:.6f}, 개별 matched kp median/P90={base['geometry']['median_px']:.4f}/{base['geometry']['p90_px']:.4f}px, translation={base['pose']['ALL']['translation_median_cm']:.4f}cm, yaw={base['pose']['ALL']['yaw_median_deg']:.4f}°. R0 개별 matched 집계와 위 네 방법 공통 집계는 분모가 다르므로 직접 차감하지 않는다.",
        '기존 GT-v2는 수동/기하 재구성 provenance가 섞인 정본이며 새 외부 센서 GT가 아니다. Stock visibility1은 가려진 좌표를 감독하며, 이전 pseudo TRUE_IGNORE loss는 사용하지 않았다. 이미지 밖으로 나간 점은 visibility0으로 내보내고 원본 라벨은 보존했다.',
        '기존 optional Albumentations API 불일치로 해당 optional transform은 미적용이며 모든 arm에서 동일하다. cuBLAS 엄밀 결정론은 보장하지 않는다. 실제 입력 순서의 parity는 직접 확인했다.',
        '평가 세션은 학습/선택에서 분리됐지만 과거 연구에서 이미 본 세션이다. Negatives도 기존 DEV이며 세션 독립 확인을 주장하지 않는다. 3seed 방향은 통계적 유의성이나 독립 일반화 증명이 아니다.',
        '기하 불안정성은 단일 밝기 변화 heuristic이다. CLUE 재현이나 새 방법 novelty를 주장하지 않는다. 이번 실험은 하나의30-label 예산점 비교이며 완전한 label-efficiency curve가 아니다.',
        'Acquisition seed는 하나이며 선택 집합은 학습 seed1/2/3에서 고정이다. 모든 학생의 시작 가중치는 동일한 R0다. 학생 학습 RNG 변동을 반복했을 뿐, 서로 다른 무작위 라벨 선택의 변동성까지 추정하지 않았다.',
        '정본 pose evaluator의 일부 설명문은319장/학습0이라는 역사적 상수를 포함한다. 실제 입력/분모 및 새 checkpoint는 각 ACTUAL_POSE_BINDING.json과 이 보고서를 기준으로 읽는다. 수치 계산은 변경하지 않았다.',
        '이전97장 라벨링 미리보기와 A/B/C/D/E 결과, 논문 final 문서는 변경하지 않았다. 이번 학생은 선택용으로 쓴 원래DEV 프레임을 전체319 평가에 다시 포함해 성능 주장하면 안 된다.',
        '결과에 따른 추가 seed/학습량/밝기/가중치/선택 budget 탐색은 수행하지 않았다.']
    value='\n'.join(lines)+'\n';path=DOC/'REPORT_KO.md'
    if path.exists():assert path.read_text()==value
    else:path.write_text(value)
    print(verdict,means,flush=True)

if __name__=='__main__':main()
