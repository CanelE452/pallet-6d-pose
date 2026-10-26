사용자가 말한 “dclip”의 정확한 논문명은 아직 확인되지 않았다. 아래는 feature map localisation 개선과 관련된 두 후보의 **1차 자료 조사**이며, 새 실험 결과가 아니다. 확인일은 2026-10-26이다.

| 후보 | 특징 위치 구분을 개선하는 방식 | 학습 데이터·공개 구현 |
|---|---|---|
| **DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception, CVPR 2025** | 마지막 attention의 content/context를 분리한다. content는 CLIP crop의 CLS 표현, context는 DINO 등 시각 교사의 토큰 관계를 증류한다. 영역 구분력과 공간 일관성을 각각 학습한다. | 이미지뿐으로 증류하며 논문에서는 COCO를 사용했다. 공식 학습·분할 코드와 EVA02-CLIP-B-16/DINOv2B 기반 가중치가 공개되어 있다. [논문](https://arxiv.org/html/2505.04410v1), [공식 저장소](https://github.com/xiaomoguhz/DeCLIP), [공식 분할 안내](https://raw.githubusercontent.com/xiaomoguhz/DeCLIP/ZSSS/README.md) |
| **DCLIP: Efficiently Disentangling CLIP for Multi-Object Perception, arXiv:2502.02977** | CLIP을 동결하고 이미지·텍스트 projector를 학습한다. MFI loss로 클래스 특징의 과도한 중복을 줄이고 ASL로 이미지와 텍스트를 정렬한다. 국소 특징과 클래스 텍스트의 유사도로 분할한다. | 픽셀 정답 대신 이미지 단위 다중 클래스 존재 라벨을 사용한다. 논문은 ResNet 및 ViT 계열을 다룬다. 이번에 확인한 논문·저자 홈페이지에서는 공식 코드/체크포인트 링크를 확인하지 못했다. 공개물이 전혀 없다는 단정은 아니다. [논문](https://arxiv.org/html/2502.02977v4), [저자 홈페이지](https://samyakr99.github.io/) |

“feature map의 localisation을 개선한다”는 설명에는 첫 번째 DeCLIP이 더 직접적으로 맞는다는 판단이다. 정확한 이름 확인 전 사용자의 의도로 확정하지 않는다. 현재 팔레트 데이터에서 개선된다는 실험 근거는 없다.

DeCLIP은 합성 RGB만으로도 증류 손실을 구성할 수 있다는 점에서 현재 조건에 맞는 후보이다. 이는 적용 가능성에 대한 추론이다. 반면 DCLIP을 논문대로 학습하려면 여러 클래스의 이미지별 존재 라벨이 필요하다. 원본 M4 하나의 여집합은 배경·다른 물체·틈을 함께 포함하므로 “외부 가림 물체”의 단일 의미 라벨로 바꿀 수 없다.

공개 DeCLIP 체크포인트는 실제 파일이 확인되었으며 약1.24GB이다. 공식 분할 설정은 **EVA02-CLIP-B-16, csa 모드**를 사용한다. 현재 OpenAI ViT-B/16 가중치에 바로 덮어쓰는 형태가 아니다. 방법 자체의 효과를 비교하려면 같은 EVA 원본과 EVA-DeCLIP을 맞춰 비교하는 것이 적절하다는 판단이다. [공식 가중치 파일](https://huggingface.co/xiaomoguhzz/DeCLIP_evab_dinov2B_csa_560_0.25_seg/tree/main/checkpoints), [공식 설정](https://raw.githubusercontent.com/xiaomoguhz/DeCLIP/ZSSS/configs/cfg_ade20k.py)

공식 분할 코드는 기본적으로 crop336/stride112의 sliding window를 사용하며, `slide_crop <= 0`일 때 전체 영상 forward 경로도 있다. 따라서 도입 시 전체 영상 입력 범위와 변환을 명시해야 한다. 공개 DeCLIP은 COCO 이미지로 추가 증류한 이력이 있으므로, 이를 사용한 결과는 “공개 실사로 적응된 백본 + 합성 후속 학습”으로 구분해서 기록해야 한다. [공식 분할 구현](https://raw.githubusercontent.com/xiaomoguhz/DeCLIP/ZSSS/declip_segmentor.py), [공식 학습 데이터 설명](https://github.com/xiaomoguhz/DeCLIP)

V267 시각화의 초록 마스크는 **CLIP Surgery 아키텍처 특징 위에 원본 SYN M4로 학습한 이진 마스크 head의 출력**이다. 텍스트 prompt로 클래스들을 비교하고 feature surgery까지 수행한 전체 zero-shot text pipeline의 결과로 부르면 안 된다. V268도 이 동결된 coarse 마스크를 이용한다. 따라서 DeCLIP의 dense 특징 증류는 지금까지의 head/잔차 학습과 다른 변경이다. 기존 실험의 원본 M4 학습 및 전체 영상 범위는 `V267_FULL_IMAGE_CLIP_VISIBLE_MASK/NEW_CONFIG_FROZEN_BEFORE_CURRENT.json`과 V268의 `INDEPENDENT_SCOPE_TRAINING_REVIEW.json`에 기록되어 있다.

특징 위치 구분력이 좋아져도 여러 팔레트 중 어느 것이 대상인지, 외곽 키포인트에서 픽셀 소속과 사람이 정의한 보임이 어떻게 대응하는지는 별도로 검증해야 한다. 기존 실사223장은 계속 평가 전용이며 반복 사용된 개발 평가라는 한계도 유지한다.

이 조사는 문헌·공식 코드 페이지·가중치 파일 목록 확인까지 수행했다. **DeCLIP/DCLIP 새 학습, 추론, 가중치 다운로드는 하지 않았다.**
