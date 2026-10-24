# V219 실제 SOURCE 학습 준비

기존 C61 RAW head를 그대로 사용합니다. 3×3 convolution 6개·stride2 세 번으로 96→12, 4×4 평균 pool로 3×3, shared64를 만듭니다. 기존 support2·owner3와 별도 direct scalar에 native8+mask8+otherXY16의 zero-init residual을 더하는 118359 parameter 구조입니다. 입력은 RGB/255·strict valid·고정 local XY·SAM zero이며, relative XY는 F64 연산 뒤 ±8 clip, FP32입니다. 새 feature 정보나 모델 효과를 주장하지 않습니다.

paired control은 기존 다섯 auxiliary loss의 평균 + synthetic own-image point0 balanced BCE, treatment는 여기에 public expected finite-part weak BCE를 추가합니다. weak .25는 과거와 맞춘 제안이며 실제 값은 root recipe에서 먼저 고정해야 합니다. 초기화1810, old draw1810·strong draw1811은 양쪽 동일, weak draw1812는 독립입니다. warmstart는 금지하며 두 final state가 모두 동결되기 전에 DEV/실사 출력을 내지 않습니다. decoder는 logit>=0 E, <0 D입니다.

synthetic strong은 경로 이름으로 gold라 부르지 않습니다. 실제 source image scene의 saved point0, target-alone D, combined D/E, 원 available·frame/crop 안 reference 조건과 기존 권한 검토를 exact metadata/admission으로 연결해야 합니다. 원7040을 보존하고 제외된 S/U/밖 상태를 E로 채우지 않습니다. 기존6096(D5853/E243)은 선행 source 수치이며 새 entry에서 exact binding과 재검증이 필요합니다.

public family는 consensus와 crop support를 통과한 weak57(FIT45 D39/E6, DEV12 D9/E3)입니다. 이는 57개 독립 frame이 아니며 동일12 frame·candidate·jitter 계보를 공유합니다. q8 자체가 GT-derived 학습 초기화이고 pose/ref/ROI/mask는 추가 배포 입력이 아닙니다. 원 initializer 성능과 정확한 물리 native point gold는 별도입니다. ROI는 감독의 weak bag으로만 보존하고 입력에 넣지 않습니다. 역할이 불분명한 기존 pair9를 native label로 복사하지 않습니다.

새 DEV4 component의 모든 public/CF 후손을 모든 auxiliary loss·sampling·owner class 정규화에서도 제외합니다. 원 component role을 덮어쓰지 않고 별도 fit mask를 교차합니다. lineage 없는 행은 자동 TRAIN으로 넣지 않습니다. 과거 checkpoint는 이 DEV를 보았을 수 있어 fresh init이 필요하며, fresh여도 이미 노출된 작은 SOURCE DEV라 fresh blind 실사 검증이 아닙니다.

현재 실제 입력 배열·PNG·weights·GT·보호 결과를 읽거나 모델을 만들지 않았습니다. root가 exact old6 NPZ와 TRAIN/nominal/component/q8 lineage metadata, point7040 권한, weak57 채택을 명시적으로 바인딩해야 실행됩니다. 최소 파일 목록은 MINIMUM_FUTURE_PAYLOAD_BINDINGS_NO_PAYLOAD_ACCESS.json에 있습니다. source cache COMPLETE의 입력/타깃 pointer는 현재 따라가지 않았습니다.

예산은 null입니다. root가 TRAIN만 사용하는 disposable throughput probe(최대32 step/arm, DEV image/output·checkpoint 저장0)를 승인하고 처리량만으로 예산을 고정한 뒤 새 paired fit을 시작하도록 false entry에 남겼습니다. 4000은 과거 비교 예시이며 이번 예산이 아닙니다. 모델 선택·threshold fit·새 protected evaluation은 runner에 없습니다. 전체 신경망·실데이터 실행 검증은 아직 하지 않았으므로 independent code preflight와 root actual entry가 남아 있습니다.
