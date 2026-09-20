"""Render the three-arm comparison and audit immutable parent evidence."""
from . import iou_selftrain as I
from . import common as C


def main():
    result=C.read(I.RAW/'SUMMARY.json');old=C.read(I.BASE_DOC/'RESULTS.json')
    names={'R0':'R0 합성만','PLASTIC':'기존 보정 self-training','IOU90':'IoU 0.9 선별 self-training'}
    lines=['# IoU 0.9 선별 후 일반 플라스틱 self-training','',
        '2026-09-20 KST. 일반 플라스틱만 재학습. 이전 결과·모델은 보존했고 자동 최종 모델 교체는 하지 않았다.','',
        '## 학습 조건','',
        '- 249장 후보 중 보정 전후 hull IoU≥0.9 AND 원본/변형 보정 hull 최소 IoU≥0.9인 167장으로 제한했다.',
        '- 기존과 같은 seed9021 복원추출로 실사 512 slots를 구성했다. 실제 고유 노출은 162장(기존 학생 217장)이다. 추가 5장 탈락이 아니라 미샘플링이다.',
        '- 합성 replay는 이전과 동일한 512장, epoch당 합성512+실사512. 5 epoch, batch16/nbs16, 실제 optimizer update320. 고정 last.pt만 평가했다.',
        '- 이전 학생을 이어 학습하지 않고 동일 R0에서 독립 시작했다. AdamW·학습률·증강·true-ignore loss·seed는 이전과 같다.',
        '- 보정 좌표/박스/학습 라벨 바이트는 기존과 동일하다. 수동 검토 JSON 및 코너 제외 마스크는 사용하지 않았다.',
        '- 후보 선별 집합만 바뀌지만 고유 이미지 수와 중복 노출 분포도 함께 바뀐다. 같은 수 무작위 선별 대비가 없어 IoU 기준만의 인과적 우월성 주장은 하지 않는다.','',
        '## 평가 1 — F0 예측 품질','',
        '| 모델 | 유지 | 중앙 오차 px↓ | P90 px↓ | 모든 감독점≤20px 이미지 | Precision↑ |',
        '|---|---:|---:|---:|---:|---:|']
    for arm,r in result['arms'].items():
        f=r['f0'];lines.append(f"| {names[arm]} | {f['accepted']} | {f['pass']['median_px']:.3f} | {f['pass']['p90_px']:.2f} | {f['TP']} | {f['precision']:.4f} |")
    lines+=['','F0는 기존 기본 적격성(검출 및 충분한 confident corner)만 적용한다. Precision은 AP precision이 아니라 모든 감독점이20px 이내인 이미지의 비율이다.','',
        '## 평가 2 — 검출·오검출·keypoint','',
        '| 모델 | AP50–95↑ | AP50↑ | AUROC↑ | FPR95↓ | kp med px↓ | kp P90 px↓ | IoU50 매칭 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,r in result['arms'].items():
        m=r['detection']['report']['metrics']['box_and_keypoint_2d'];a=r['detection']['ranking']
        lines.append(f"| {names[arm]} | {m['box_ap50_95']:.4f} | {m['box_ap50']:.4f} | {a['auroc']:.4f} | {a['fpr95']:.4f} | {m['keypoint_location_median_px']:.3f} | {m['keypoint_location_p90_px']:.2f} | {m['keypoint_matched_frame_count_iou50']}/194 |")
    lines+=['','全194장 일반 플라스틱 + 동일 NEG2689. 보정기/IoU/LOO로 평가 이미지를 걸러내지 않은 학생 단독 추론이다. kp 오차는 IoU50 매칭 이미지의 감독 keypoint pooling이며 F0와 집계 조건이 다르다. FPR95는 각 모델 TPR95% 지점의 배경 FP 비율이다.','',
        '## 평가 3 — frozen MAIN pose','',
        '| 모델 | PoseCov↑ | AxisAcc↑ | R med°↓ | Yaw med°↓ | t med cm↓ | IoU3D↑ | ADDsym AUC↑ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,r in result['arms'].items():
        p=r['pose'];lines.append(f"| {names[arm]} | {p['coverage']:.3f} | {p['axis_accuracy']:.4f} | {p['rotation_median_deg']:.3f} | {p['yaw_median_deg']:.3f} | {p['translation_median_cm']:.3f} | {p['iou3d_median']:.5f} | {p['add_sym_auc']:.5f} |")
    lines+=['','既存 frozen prediction-only selector + SQPnP/RefineLM. GT axis oracle가 아니다. 기준6D는 2D annotation·K·치수 기반 기하 복원 reference이며 외부 측정6D GT가 아니다.','',
        '## 보조 — 전체 분모 symmetry-aware PCK20','',
        '| 모델 | PCK20 |','|---|---:|']
    for arm in names:
        value=result['symmetry']['PCK']['20'] if arm==I.ARM else old['summaries'][arm]['PLASTIC']['PCK']['20']
        lines.append(f'| {names[arm]} | {100*value:.2f}% |')
    lines+=['','허용 whole-object symmetry, 첫8코너, 검출/매칭 실패 penalty 포함. 위 논문 fixed-index9-keypoint median과 다른 지표이다.','',
        '## 한계·재현','',
        '- 단일 seed의 짧은 후속 실험, 반복 사용한 DEV 평가이며 독립 최종 확인이나 통계적 유의성 주장은 없다.',
        '- 평가 결과를 보고 threshold/step/last checkpoint를 다시 선택하지 않는다. 고유 라벨 감소 및 노출 분포 변화도 결과에 포함된다.',
        '- 기존 replay에는 empty-label negative가 없다. 이번에도 추가 negative 학습은 하지 않았다.',
        f"- 학습 완료: {result['fit']['seconds']:.1f}초, 실제320steps, R0 초기화 exact parity와 가중치 변경 검사 통과.",
        '- 원시 결과: `data/pallet/results/pallet_type_selftrain_v1/iou_selftrain/SUMMARY.json`, `metrics/`, `SYMMETRY_METRICS.json`.',
        '- 새 모델: `data/pallet/results/pallet_type_selftrain_v1/iou_selftrain/runs/IOU90/weights/last.pt`.',
        '- 실행: `python -m scripts.research.pallet_type_selftrain_v1.iou_selftrain {prepare,train,eval,negative,score}` 순서. GPU phases는 각각 별도 process로 실행.',
        '- commit/push, 기존 모델 덮어쓰기, 컴퓨터 재부팅 또는 드라이버 변경은 하지 않았다.','']
    report='\n'.join(lines).replace('全194장','전체 194장').replace('既存 frozen','기존 frozen')
    C.write_text(I.RAW/'RESULTS_KO.md',report)
    protocol=C.read(I.DOC/'TRAIN_PROTOCOL.json')
    for b in protocol['inputs']+protocol['sources']+[protocol['initialization']]:C.verify(b)
    prior=C.read(I.BASE_RAW/'paper_metrics_plastic/SUMMARY.json')['arms']
    assert all(result['arms'][a]==prior[a] for a in ['R0','PLASTIC'])
    C.verify(result['fit']['checkpoint'])
    assert result['fit']['optimizer_steps']==320 and result['fit']['initialization_exact'] and result['fit']['changed_tensors']>0
    C.freeze(I.DOC/'FINAL_AUDIT.json',dict(status='PASS',parent_results_unchanged=True,source_hashes_verified=True,
        exact_R0_initialization=True,optimizer_steps=320,epoch_count=5,
        selected_pool=167,sampled_unique=162,model=result['fit']['checkpoint'],
        positive194_negative2689=True,evaluation_filtering=False,GT_used_as_training_target=False,
        artifacts=[C.bound(p) for p in sorted(I.RAW.glob('*.json'))],report=C.bound(I.RAW/'RESULTS_KO.md'),report_code=C.bound(__file__)))
    print(report,flush=True)


if __name__=='__main__':main()
