# V262: 문제별 문헌·공식 코드 검토 — 실험 실행 없음

검토일: 2026-10-26, Asia/Seoul. V261 다음의 **조사 문서**이며 새 학습, 모델 추론, crop 변경, PnP 실행 또는 성능 개선 결과가 아니다. Pro 원문 전체를 새로 읽었다는 뜻도 아니며 사용자가 붙여 준 논의와 기존 인계, 동결된 V261 결과를 대조했다.

**판단: 우선 검증할 가치가 큰 방향은 ‘번호와 근거가 명확한 점의 위치 분포를 찾고, 쓸 수 없는 관측은 보류하는 모듈’이다. 가림 scalar 분류 반복이나 큰 foundation model 교체만으로 해결될 근거는 부족하다.** 이 판단은 연구 우선순위이며 팔레트 성능의 보장이 아니다.

## 1. 지금 실패한 것과 아직 시험하지 않은 것

[V261 전체 결과](../V261_FINAL_PUBLIC_REAL_COMPARISON/README_KO.md)에서 공개 실사 추가 전후 동일 current 1,237점의 보임 정답은 964→959 /1,147, 외부 가림 정답은 68→68 /90이다. 추가군 E precision은 26.56%다. 수동 클릭·보임·원 query 유효·참조 오차≤3px인 376점도 316→308점으로 줄었다. 반면 공개 표면점 AUX DEV에서는 E357/398(89.70%)를 맞혔다.

이것은 **공개 표면점 AUX의 학습이 현재 native MAIN의 전체 개선으로 전이되지 않았다**는 결과다. 표면점과 native 평가의 감독 정의·분모는 다르다. 목표 차이, 외관 차이, 특징 요약, instance/point 대응 각각의 원인 기여율을 이 한 번의 비교로 확정하지 못했다.

V261의 두 출력은 **MAIN/AUX 두 분류 logit**이다. 가시성+위치 두 출력이 아니다. 실제 CODE는 `1186→64 ReLU→2` MLP이며 full CLS, 중앙 고정 patch 평균, centre CLS, 유효율 및 condition32를 받는다. 모든 공간 token으로 위치 heatmap을 예측하는 학습은 하지 않았다. 그러므로 V261의 실패는 Pro가 제안한 spatial localization/joint 학습까지 실패했다는 증거가 아니다. [확인한 로컬 CODE와 수치의 binding](LOCAL_EVIDENCE.json)

## 2. OVD는 되는데 우리는 왜 안 되는가

사용자가 말한 OVD/OVOD는 **open-vocabulary object detection**이다.

Grounding DINO의 기본 입력은 이미지+텍스트이고 출력은 물체 box와 텍스트 대응 점수다. 실제 [`predict`](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/util/inference.py)는 `pred_boxes`와 `pred_logits`를 추출한다. OWLv2도 pseudo-box 기반 대규모 학습과 box detection 평가를 다룬다. ‘새 범주에 zero-shot’이라는 말은 실사 사전학습을 전혀 안 했다는 뜻이 아니다. [Grounding DINO 논문](https://arxiv.org/abs/2303.05499), [OWLv2 논문](https://arxiv.org/abs/2306.09683), [공식 OWL 코드](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)

예를 들어 팔레트의 일부를 보고 물체 상자를 찾을 수 있어도, 그 상자 안의 **지정한 3번 코너가 직접 보이는지**, 같은 색의 다른 팔레트 면을 보고 있는지, 원 query가 옆 배경으로 빗나갔는지는 별도 문제다. referring expression으로 instance를 지정하더라도 기본 출력이 native 코너 identity·가시성 정답으로 바뀌지는 않는다.

Grounding DINO와 우리가 쓴 **DINOv2**도 같은 시스템이 아니다. 전자는 grounded detector이며 후자는 시각 특징 추출기다. DINOv2는 [공식 `forward_features`](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py)에서 patch token을 제공하지만, V261은 그 전체 공간 배열을 위치 decoder에 입력한 실험이 아니다.

SAM의 mask도 예측된 물체 영역이다. mask 밖인 점은 외부 가림 외에도 팔레트 틈, 원래 재료가 없는 가상 주소, 좌표 오차, 다른 instance 선택, 분할 오류일 수 있다. box/mask는 고정 proposal·관측 cue로 사용할 수 있지만 가시성 GT나 코너 대응으로 승격할 수 없다. [SAM 공식 저장소](https://github.com/facebookresearch/segment-anything)

**따라서 OVD의 성공은 물체 수준 의미 단서 활용의 참고이며, 이 native 과제의 가능·불가능을 판정하는 근거는 아니다. 현재 classifier를 큰 OVD로 교체하면 해결된다는 증거도 아니다.** 지정 점의 문제 정의와 감독을 맞추고 공간·대응 정보를 유지해야 한다는 것이 이번 조사에서의 추론이다.

## 3. 문제별 우선순위와 실제로 가져올 아이디어

| 문제 | 판단 | 가장 가까운 근거·아이디어 | 현재 과제에 필요한 조건 |
|---|---|---|---|
| 평가 실사로 맞춘 CAL·공분산·신뢰도·prior 의존 경로 | 출처·설정 정리는 실행 가능한 엔지니어링 작업. 제거 후 성능은 별도 | 학습/선택/평가 출처를 분리하고 통계별 생성 자료를 기록. 합성 노이즈·검증으로 정할 수 있는 값부터 고정 | frozen이라고 기존 CAL 학습 이력이 사라지지 않는다. 과거 OCPR의 의존성이 V261 새 head에도 들어갔다고 혼동하지 않는다 |
| 보이는 점의 위치·가림·관측 적합성 | 작은 모듈로 우선 검증할 가치가 가장 큼 | LUVLi의 위치·불확실성·가시성 분리, PVNet의 위치 분포, FoundPose의 공간 대응 | 물체/코너 ID가 고정돼야 하고 선 하나뿐인 관측·가상 유도 좌표·직접 측정점을 구분해야 함 |
| 90°/폭·깊이/native 대응 및 성공 후보 부재 | 국소 위치 오차보다 범위가 큼. 작은 crop 보정만으로 해결 기대 금지 | EPOS의 one-to-many 대응, SurfEmb의 다봉 대응 분포, FoundPose의 여러 CAD template | CAD/축/치수·instance identity, 전역 형상 증거, 고정 후보 예산 필요. 기하 대칭과 semantic ID 오류는 별도 |
| 성공 후보가 있는데 선택을 틀림 | 후보 생성과 분리하면 연구 질문이 명확함 | SurfEmb처럼 후보 생성과 likelihood 기반 점수화를 구분 | 같은 bank의 재점수화와 후보 추가를 한 비교에서 섞지 않기. 점수가 보정된 확률이라는 보장은 별도 |
| 이미 잘 맞던 결과 손상 | fallback/보류 규칙 구현 가능. 실사 무손상 보장 아님 | 위치 불확실성·관측 지지와 원 결과 유지 비교 | 추론 시 GT 없는 선택 규칙과 coverage/위험을 검증. 성공→실패 수를 반드시 보고 |
| 합성→실사 전이 | 가능한 연구 방향이지만 현재 과제에서 미해결 | BlenderProc4BOP의 재질·조명·배치/가림 다양화, SurfEmb의 synthetic-only 사례 | 외관뿐 아니라 supervision 의미·ID·오차/crop 범위를 맞춰야 한다. 새 촬영 세션 평가가 필요 |

이는 성공 확률 수치가 아닌 우선순위 판단이다. 붙여 준 과거의 **72실패=선택 실패31+성공 후보 부재41**은 특정 historical PRIMARY192 결과이며 current presence 분모가 아니다.

현재 사용자 조건은 **모든 실사 금지**가 아니다. 공개 실사·범용 사전학습은 허용하고 현재 평가 자료로 학습·문턱·설정을 맞추는 것을 금지한다. 공개 학습을 허용하는 실험과 ‘전체 synthetic-only 파이프라인’ 주장은 구분해야 한다.

## 4. 논문과 GitHub를 읽고 고른 것

### LUVLi — 공동 위치·불확실성 추정의 구조를 참고

[논문](https://arxiv.org/abs/2004.02980) · [공식 MERL 저장소](https://github.com/merlresearch/LUVLi)

heatmap으로 위치를 얻고 별도로 covariance/visibility를 추정한다. 위치 NLL의 감독 mask는 예측 visibility가 아닌 **GT v**다. 다만 이 논문의 v=1은 위치 GT가 제공되는 landmark이며 외부 가림도 포함하고 SELF는 제외한다. 따라서 우리 DIRECT_VISIBLE/E와 같은 분류 성능 증거로 쓰면 안 된다. 또한 heatmap 폭·entropy가 곧 보정된 covariance라는 가정도 하지 않는다. 구조와 loss mask를 참고할 수 있으며 얼굴 실사 학습 결과를 팔레트 전이 성능으로 가져올 수는 없다.

### PVNet — 숨은 좌표도 추정할 수 있지만 직접 보임과 구분

[논문](https://arxiv.org/abs/1812.11788) · [저자 코드](https://github.com/zju3dv/pvnet)

여러 object pixel의 방향 투표로 keypoint 위치와 분산을 얻는다. 가림·화면 밖 점의 위치를 추정하는 참고 사례다. 그러나 이것은 직접 관측되는 점만 측정하는 방법이 아니라 **amodal localization**도 포함한다. 원 실험의 실사+렌더/cut-and-paste 사용과, frozen spatial feature의 작은 head 제안을 같은 설정으로 취급하지 않는다.

### FoundPose — 우리 frozen feature 활용에 가장 가까운 구조적 참고

[논문](https://arxiv.org/abs/2311.18809) · [공식 저장소](https://github.com/facebookresearch/foundpose) · [`scripts/infer.py`](https://github.com/facebookresearch/foundpose/blob/main/scripts/infer.py)

단일 RGB와 알려진 객체 CAD, K, 객체 identity/instance mask를 이용해 렌더 template와 DINOv2 patch를 대응시킨다. 대상·pose task별 추가 학습이 없으며, 논문은 중간층 특징의 위치 정보가 대칭·저텍스처에서 중요함을 보인다. 우리에게 직접적인 아이디어는 **공간 정보를 유지한 query↔CAD/구조 참조 대응**이다. 전체 알고리즘을 한꺼번에 이식하자는 제안은 아니다. 공식 README는 공개 구현이 coarse pipeline이며 논문의 featuremetric refinement가 빠져 있음을 명시한다.

### EPOS — 애매한 대응을 하나로 강제하지 않기

[논문](https://arxiv.org/abs/2004.00605) · [공식 저장소](https://github.com/thodan/epos) · [`establish_many_to_many`](https://github.com/thodan/epos/blob/master/epos_lib/corresp.py)

객체 표면 fragment와 픽셀 사이에 여러 가능한 3D 대응을 남기고 robust fitting한다. 90°/부분 대칭 대응을 조기에 하나로 확정하는 문제에 참고할 수 있다. visible mask·surface/pose 감독과 CAD가 필요하며 기존 8점 MLP를 단순 교체하는 수정은 아니다. 공식 inference의 LOCALIZATION benchmark 경로가 GT instance 수를 사용하는 점도 배포 입력과 구분해야 한다.

### SurfEmb — 후보를 만드는 문제와 고르는 문제를 분리

[논문](https://arxiv.org/html/2111.13489v2) · [공식 저장소](https://github.com/rasmushaugaard/surfemb) · [`estimate_pose`](https://github.com/rasmushaugaard/surfemb/blob/master/surfemb/pose_est.py)

RGB와 CAD surface 사이의 다봉 대응 분포를 학습하고 이를 후보 sampling·scoring·refinement에 사용한다. synthetic-only RGB 모델 결과도 있어 합성→실사 연구의 직접적인 사례다. 그렇지만 object-specific dense 학습과 렌더 surface 감독을 사용하는 방법으로, 가상 코너의 직접 가시성을 작은 head로 학습한 성능은 아니다. 고정 bank 재점수화와 bank 확장의 효과를 분리하는 구조를 참고할 수 있다.

### 데이터 생성 및 참고 범위를 제한해야 할 논문

- [BlenderProc 원논문](https://elib.dlr.de/139317/1/denninger.pdf), [공식 BOP 생성 예제](https://dlr-rm.github.io/BlenderProc/examples/datasets/bop_challenge/README.html), [GitHub](https://github.com/DLR-RM/BlenderProc): PBR·재질·조명·배치 다양화와 실사 전이 사례. 그 사례가 native visibility의 해결을 보장하지 않는다.
- [Pix2Pose](https://arxiv.org/abs/1908.07433), [코드](https://github.com/kirumang/Pix2Pose): 좌표 복원·예상 오차·대칭 처리는 참고할 수 있지만 원 실험은 real RGB/pose/mask를 사용한다. textureless CAD 사용 가능과 실사 학습 불필요는 다르다.
- [GDR-Net](https://arxiv.org/abs/2102.12145), [코드](https://github.com/THU-DA-6D-Pose-Group/GDR-Net): visible mask·dense geometry 중간 감독과 GT mask로 좌표 loss를 정하는 구조. end-to-end spatial 학습과 real/synthetic 데이터 설정이 V261과 다르다.
- [ZebraPose](https://arxiv.org/abs/2203.09418), [코드](https://github.com/suyz526/ZebraPose): 계층적 surface code 참고. 공개 solver의 `maximum_model_number=1` 경로를 다중 pose 선택기로 소개하지 않는다.
- [Shi–Tomasi](https://users.cs.duke.edu/~tomasi/papers/shi/TR_93-1399_Cornell.pdf), [OpenCV 공식 구현](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/featureselect.cpp): 국소 영상 변화가 두 방향에서 충분한지 확인하는 기초. OpenCV는 원저자 저장소라는 뜻은 아니다. 일정한 직선 일부만 있으면 선을 따르는 위치가 미결정인 aperture ambiguity가 남으며, 강한 코너 응답도 native ID를 증명하지 않는다.

## 5. 다음 실험을 새로 승인한다면: 최소 분기 설계

아래는 **제안**이며 이번 조사에서 실행하지 않았다. 이전의 마지막 1회 학습 제한을 새 실험 승인으로 해석하지 않는다.

1. **먼저 정답 정의와 저렴한 진단:** 보임, query 픽셀에 대상이 있음, ID가 맞음, 직접 점 위치가 측정 가능함, search 범위 안임을 별도로 기록한다. B=외곽 경계, I=내부 RGB gradient/색 전환이며 I 전체가 교차점은 아니다. 합성에서 원래 결과/가시성만 oracle/직접 관측 위치만 oracle/둘 다 oracle을 같은 고정 후단으로 비교한다. 잘못된 ID나 가상 유도 좌표는 직접 관측점 정답으로 바꾸지 않는다.
2. **위치 보정 단독을 먼저 작게:** 같은 초기 q를 중심으로 공간 token grid→heatmap을 사용한다. 합성 GT의 직접 관측 근거·ID·crop 범위가 유효한 점만 위치 loss를 받고, predicted visibility로 이 loss를 꺼 주지 않는다. 범위 밖을 경계 정답으로 붙이지 않는다. 가상 8코너 좌표도 추정할 수 있지만 직접 측정한 위치와 전체 기하로 추론한 좌표는 별도 평가한다. presence 학습 전체에 물리적 mesh vertex 존재를 필수 조건으로 강제하지 않는다.
3. **추론 시 GT 없는 관측 적합성을 반드시 검증:** no-match/out-of-range 상태, 여러 peak·길게 늘어진 분포·구조 참조 대응을 검토하고 acceptance precision/coverage를 측정한다. heatmap confidence를 보정된 확률로 간주하지 않는다. 선 하나뿐인 관측을 임의의 2D 점으로 만들지 않으며, B/I 어느 쪽이든 위치·ID 근거가 충분하면 사용할 수 있다.
4. **유망할 때만 작은 대조:** 보정 없음/가시성만/위치만/joint를 같은 q·crop·학습량·후단으로 비교한다. crop 선택은 합성 validation에서 고정한다. 처음 seed1개는 가능성 검사이며 이 결과만으로 최종 우수성을 주장하지 않는다. B/I 효과는 이후 동일 구성에서 B/I만 바꾸는 별도 대조가 필요하다.
5. **실사 전이와 자세 효과는 분리:** 반복해서 본 223장은 개발 진단으로 명시하고 새 촬영 세션의 미사용 자료를 별도 보고한다. 목재/정사각형 각각 점 정확도·보류와 성공률·평균/중앙값/p90·큰 방향 오류·성공→실패를 기록한다. 대칭 적용 전/후 pose 오차와 native ID 정답을 구분한다. oracle은 진단일 뿐 배포 성능이 아니다.

등록 치수도 점검 대상이다. 현재 HANDOFF의 목재 X/Z는 0.8/0.59≈1.36, 정사각형은 1.1/1.1=1이다. 실물 실측·CAD 축/scale의 완전 확인과는 별개이므로 ‘매우 긴 목재에서는 방향 모호성이 거의 없을 것’이라는 전제를 수치 확인 없이 쓰지 않는다. 이 등록값이 실패 원인이라고 확정한 것은 아니다.

조사의 결론은 **위치 근거와 대응을 보존하는 쪽으로 문제를 좁혀 검증할 근거가 있다**는 것이다. 가림만 잘 맞히면 자세·후보 선택·90° 오류가 함께 해결된다는 근거는 없으며, 현 결과로 전체 문제를 불가능하다고 판정할 근거도 없다.
