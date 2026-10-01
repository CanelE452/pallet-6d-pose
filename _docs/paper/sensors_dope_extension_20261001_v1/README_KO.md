# Sensors 세 추정기 원고

이 폴더는 기존 YOLO 원고를 보존한 새 revision이며 **미제출 초안**입니다. 현재 수치 삽입 상태는 [DOPE 영수증](DOPE_RESULTS_BINDINGS.json), [ResNet18 영수증](RESNET_RESULTS_BINDINGS.json), [최종 빌드 상태](DRAFT_BUILD.json)를 따릅니다. 현재 준비 단계에서는 두 새 추정기의 실제 결과를 넣지 않았습니다. 결과 삽입과 PDF 빌드는 실험 성공·저자 승인·독립 TEST 확보를 뜻하지 않습니다.

- [본문 PDF](manuscript.pdf), [본문 TeX](manuscript.tex), [보충 PDF](supplementary.pdf), [보충 TeX](supplementary.tex).
- [주장과 한계](CLAIMS_AND_PENDING_KO.md), [저자 검토 항목](AUTHOR_REVIEW.md), [PDF 시각 확인](PDF_VISUAL_REVIEW.json).
- [기존 자산 SHA](HISTORICAL_ASSETS.json), [ResNet18 방법 근거](R18_METHOD_SOURCES.json).

기존 YOLO 결과는 재사용 DEV319장·13세션에서 P의 3-seed 평균 중앙값 6.615678→5.904910 px와 같은 세션의 정확도·비용 관계를 뒷받침합니다. P는 D보다 중앙값이 낮았지만 tail과 비용을 모두 지배하지 않으며, PoseFix-derived prior보다 중앙값이 높았습니다. 이 값을 DOPE/ResNet18 결과나 최근 별도 T/R selector 목표 달성으로 재해석하지 않습니다.

새 원고는 DOPE의 고정 VGG feature와 SimpleBaseline-derived ResNet18의 고정 layer2/layer3 feature에 각각 새 P/D head를 학습하는 계약을 기술합니다. 각 추정기의 전처리·누락·원좌표 복원을 별도로 확인합니다. 동일한 head 노출량은 동일한 backbone 학습 비용을 뜻하지 않습니다. DOPE의 실제 원학습 loop 0…60은 61 passes이며 YOLO는 best checkpoint입니다. ResNet18은 55,980장×60 epochs, 3,358,800 exposures와 209,940 updates를 별도 기록합니다. 각 epoch는 full batch 3,498개와 마지막 12장, 총 3,499 updates입니다.

[방법 근거](R18_METHOD_SOURCES.json)의 실제 첫 TRAIN16장 CPU 입력 검산은 144/144 target 채널, forward/optimizer/GPU 0입니다. 학습·성능 영수증이 아닙니다. 실제 수치를 삽입한 뒤에도 DEV 재사용, geometry-derived pose reference, 독립 TEST 부재는 유지합니다.

## 결과 삽입과 빌드

저장소 루트에서 다음 명령은 가상 자료 검사와 기존 생성물의 CPU 조판만 수행합니다.

```bash
/home/minjae/anaconda3/envs/pallet-pose/bin/python _docs/paper/sensors_dope_extension_20261001_v1/update_dope_results.py --selfcheck
/home/minjae/anaconda3/envs/pallet-pose/bin/python _docs/paper/sensors_dope_extension_20261001_v1/update_resnet_results.py --selfcheck
/home/minjae/anaconda3/envs/pallet-pose/bin/python _docs/paper/sensors_dope_extension_20261001_v1/build.py --render
```

실제 학습·평가·timing·독립 검산이 모두 완료된 뒤 다음 순서로 실행합니다. 삽입기는 모델 실행이나 지표 재평가를 하지 않습니다.

```bash
/home/minjae/anaconda3/envs/pallet-pose/bin/python _docs/paper/sensors_dope_extension_20261001_v1/update_dope_results.py --results-dir _docs/experiments/pallet_dope_refiner_20261001_v1 --runtime _docs/experiments/pallet_dope_refiner_20261001_v1/RUNTIME.json
/home/minjae/anaconda3/envs/pallet-pose/bin/python _docs/paper/sensors_dope_extension_20261001_v1/update_resnet_results.py --results-dir _docs/experiments/pallet_resnet18_refiner_20261001_v1 --runtime _docs/experiments/pallet_resnet18_refiner_20261001_v1/RUNTIME.json
/home/minjae/anaconda3/envs/pallet-pose/bin/python _docs/paper/sensors_dope_extension_20261001_v1/build.py --render
```

각 삽입기는 `DEV_RESULTS`, `DEV_PAIRED_RESULTS`, `TRAINING_COMPLETE`, `SELECTION`, `SYNTHETIC_HELDOUT`의 완료·319장/13세션/2818점·7arms·source 선택·SHA 연결을 검사합니다. ResNet18은 추가로 `BASELINE_TRAINING_COMPLETE`의 60epoch/209940updates/3358800exposures와 모든 epoch loss/mask counts를 검증합니다. 기존 JSON 수치를 출력하며 불리한 수치도 그대로 넣습니다. 조건부 GT 제외 수와 실제 missing point 수는 다릅니다. 서로 다른 support의 bootstrap CI는 `unavailable`입니다.

Timing은 각 추정기 26장×5회×3arms=390 측정과 60warmups, 실패 유지·replay·예측/코드 binding을 확인합니다. 2D/PnP/full 각각의 median을 보고하며 median들을 더해 total latency를 만들지 않습니다. 역사적 YOLO timing과 다른 측정 세션임을 명시합니다.

DOPE 삽입기는 `dope_*.tex` 6개만, ResNet18 삽입기는 `resnet_*.tex` 7개와 실제 완료 시 baseline curve PDF/60행CSV만 소유합니다. 세 번째 추정기 상태 macro는 `resnet_status.tex` 한 곳에 있으므로 완료 후 pending 문구가 남거나 DOPE output hash가 바뀌지 않습니다. 두 영수증은 서로 다른 출력에 SHA를 부여하고, build는 둘의 현재 generated artifact binding을 확인합니다. `--pending`은 미측정 초안 작성 전용이며 실제 결과 삽입 후에는 실행하지 않습니다.

빌드는 기존 로컬 Tectonic과 새 폴더로 복사한 45MB TeX 캐시를 `--only-cached`로 사용합니다. 원본 원고/캐시를 수정하거나 네트워크 패키지를 설치하지 않습니다. `DRAFT_BUILD.json`은 PDF·TeX·표·로그·렌더를 연결합니다. 결과가 갱신되면 이전 PDF 시각 확인은 해당 옛 hash에만 유효하므로 최종 PDF를 다시 확인해야 합니다.

## 게시 자산

| 자산 | 내용 |
|---|---|
| 원고·보충 | TeX/PDF 각 2개, 원래 저자 placeholder 유지 |
| 참고문헌 | 기존 6개 + DOPE와 Simple Baselines 공식 서지 |
| 기존 숫자·그림 | 기존 표10개, PDF그림7개, graphical_abstract PNG; YOLO 전용 원본 bytes |
| 새 표 | dope_*6개, resnet_*7개; 완료 후 실제 baseline curve PDF/CSV |
| 추적 | 양쪽 RESULTS_BINDINGS, HISTORICAL_ASSETS, R18_METHOD_SOURCES, 기존 숫자/그림 manifest |
| 도구·빌드 | update_dope_results.py, update_resnet_results.py, build.py, DRAFT_BUILD와 로그 |

`.gitignore` 때문에 그림/PDF가 일반 추적 목록에 빠질 수 있어 실제 publication 목록과 remote bytes를 확인해야 합니다. 원본 RGB·대형 checkpoint·feature cache·TeX cache·개별 PDF 렌더 PNG는 공개 TeX 의존성이 아닙니다. 저자 이름·기관·funding·제출 동의를 임의로 만들지 않았습니다.
