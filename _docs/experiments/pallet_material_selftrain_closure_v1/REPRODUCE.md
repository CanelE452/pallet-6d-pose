# Material closure 재현

기존 Plastic320-update main과 checkpoint를 덮어쓰지 않는다. 환경은 기존 `pallet-yolo26`이며 패키지/드라이버 변경이나 재부팅은 필요하지 않다. 아래 순서는 저장된 완료 단계를 검증·재사용한다. 학습 명령은 FIT가 있으면 checkpoint를 검증하고 종료하며 미완료 run은 자동 덮어쓰지 않는다. 완전히 새 복제 환경에서 다시 학습하려면 별도 결과 root를 사전 고정해야 한다.

## 입력·계약

`METHOD_LOCK.json`, `INPUT_BINDINGS.json`, `EVAL_POPULATION_LOCK.json`, `WOOD_DATA_INVENTORY.json`, `WOOD_PROVENANCE_AUDIT.json`을 먼저 검증한다. 사적 RGB/타깃/checkpoint는 공개 저장소에 포함하지 않으므로 해당 local artifacts와 SHA가 필요하다. teacher는 Replay9/38 단 하나이며 source-only control은 기존 SYN_LR5를 재사용한다.

## 실행 순서

```bash
python -m scripts.research.pallet_material_selftrain_closure_v1.lock_method
python -m scripts.research.pallet_material_selftrain_closure_v1.wood_inventory_audit
python -m scripts.research.pallet_material_selftrain_closure_v1.prepare_wood
python -m scripts.research.pallet_material_selftrain_closure_v1.pseudo_pool
python -m scripts.research.pallet_material_selftrain_closure_v1.train_pair prepare
python -m scripts.research.pallet_material_selftrain_closure_v1.train_pair WOOD_RAW_LR5
python -m scripts.research.pallet_material_selftrain_closure_v1.train_pair WOOD_REF_LR5
python -m scripts.research.pallet_material_selftrain_closure_v1.infer_eval
python -m scripts.research.pallet_material_selftrain_closure_v1.score_eval
python -m scripts.research.pallet_material_selftrain_closure_v1.material_report
python -m scripts.research.pallet_material_selftrain_closure_v1.figures
python -m scripts.research.pallet_material_selftrain_closure_v1.final_audit build
python -m scripts.research.pallet_material_selftrain_closure_v1.final_audit audit
python -m pytest scripts/research/pallet_material_selftrain_closure_v1 -q
```

`POOL_DECISION.json`의 최초 추가 support gate 오류는 삭제하지 않았다. 실제 학습은 `POOL_DECISION_CORRECTION.json`이 설명하는 `POOL_DECISION_V2.json`을 사용했다. 빈 디렉터리에서 최초 오류 단계를 재현해 새 오류를 만드는 것이 목적은 아니다. 기존 namespace에서는 보존된 원본·정정 결정과 hash를 검증한다. `wood_inventory_audit`의 원 inventory 입력은 `WOOD_INVENTORY_AGENT.json`으로 고정되어 있어야 한다.

추론은 모든 arm과 D9 pose를 먼저 hash lock한 뒤 scoring을 수행한다. `WOOD_PREDICTIONS_LOCK.json.created_at <= WOOD_SCORING_START.json.utc` 및 source bindings를 검사한다. 보고서 생성은 checkpoint/FIT/CSV/실행 상태를 검증하지만 새로운 추론·학습을 하지 않는다.

## 수치 출처

- Plastic: 과거 `pallet_selftraining_paper_closure_v1/CORE_RESULTS.json`의 R0/RAW_LR5/REF_LR5. 후속640-update 모델로 교체하지 않는다.
- Wood: `WOOD_RESULTS.json`, `WOOD_PAIRED_ANALYSIS.json`의 사전 고정45장. 모든 recording/session/사용 가능한 severity를 유지한다.
- 검수 품질: `WOOD_PSEUDO_QUALITY.json`은 trusted0으로 UNRESOLVED. legacy 점수를 verified로 승격하지 않는다.
- 표: `MATERIAL_NUMBER_PROVENANCE.json`의 JSON→MD/TEX hash 매핑.
- 모델: 각 FIT의 checkpoint/initialization/protected-state/optimizer320 증거와 epoch CSV.

## 원고·공개 범위·종료

원고는 `_docs/paper/selftraining_submission_v1/manuscript.tex`이고 기존 빌드 절차를 그대로 사용한다. `material_report`는 실험 보고서와 표만 만들며, `figures`는 MD/원고용 그림을 생성하고 `final_audit build`는 현재 원고의 PDF를 재생성한다. 본 실행은13페이지 PDF·계약테스트30개·자동감사와1/5/6페이지 시각 확인까지 완료했다. 사용자가 재현 메타데이터와 평가 오버레이6장 공개를 승인했다. Private 좌표 배열·원본RGB파일·카메라행렬·가중치는 공개 대상이 아니다. Git 반영은 PUSH_VERIFICATION.json을 참조한다.

새 teacher/threshold/loss/selector/epoch 탐색·추가 수동 레이블·DOPE는 실행하지 않는다. 결과가 불리해도 원고에서 범위를 제한하고 종료한다.

## 실제 학습 완료 trace

```json
{
  "WOOD_RAW_LR5": {
    "checkpoint": {
      "path": "data/pallet/results/pallet_material_selftrain_closure_v1/runs/WOOD_RAW_LR5/weights/last.pt",
      "sha256": "7999e78fb6ff162e60645adc1f05f525a68d658ee1c73d624822b5e274219963",
      "bytes": 6541159
    },
    "seconds": 46.698198260972276,
    "epochs": 5,
    "optimizer_updates": 320,
    "epoch_updates": [
      64,
      128,
      192,
      256,
      320
    ],
    "exact_R0_initialization": true,
    "protected_state_exact": true,
    "results_csv": {
      "path": "data/pallet/results/pallet_material_selftrain_closure_v1/runs/WOOD_RAW_LR5/results.csv",
      "sha256": "4bf523484c4db628959e4a824a5867305af68dd991cf0208a952f9e14204a79a",
      "bytes": 975
    }
  },
  "WOOD_REF_LR5": {
    "checkpoint": {
      "path": "data/pallet/results/pallet_material_selftrain_closure_v1/runs/WOOD_REF_LR5/weights/last.pt",
      "sha256": "56934125bbbfcfce393fb7560ad5ff186a7ccf96fb74f066c9e827479906451e",
      "bytes": 6541159
    },
    "seconds": 46.443398994160816,
    "epochs": 5,
    "optimizer_updates": 320,
    "epoch_updates": [
      64,
      128,
      192,
      256,
      320
    ],
    "exact_R0_initialization": true,
    "protected_state_exact": true,
    "results_csv": {
      "path": "data/pallet/results/pallet_material_selftrain_closure_v1/runs/WOOD_REF_LR5/results.csv",
      "sha256": "c353c5a76aae7f8eba75f8277e69bcc7a4811f2dab10ad2522a12070173f1100",
      "bytes": 975
    }
  }
}
```
