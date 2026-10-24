동결 C62/C63 두 팔을 SOURCE TRAIN에서만 진단한다. 원7040 point metadata는 모두 남기고, 사전 admission의 알려진6096(D5853/E243)만 forward한다. 제외944/원inputU64를 E나 음성으로 바꾸지 않는다. weak44(D26/E18)는 별도 권위와 표로 유지한다.

조건은 원입력, RGB127, other-q8 XY0, other-mask+XY0, native-onehot0, 전체condition0의6개다. q/L/native ID·원샘플·valid·가중치는 고정한다. native ID를 다른 값으로 바꾸지 않고 residual 첫Linear 입력의 명시적 조건 열만 scoped hook으로0 처리한다. RGB127은 통계를 맞추지 않은 사전 상수이며 valid/grid/zeroSAM을 남긴다. 이는 OOD 입력 민감도 진단이다.

원 공유NPZ의 전체 압축 바이트는 root 승인 후 opaque SHA로 확인할 수 있지만 NPY에서는 선택 TRAIN6096 RGB/valid 및 원7040 META의 row byte만 배열로 해석한다. HOLD 영상·점수는 해석하지 않으며 상속 full-array loader를 호출하지 않는다. 원 SOURCE GT/config/target NPZ·보호셋·foundation weights도 읽지 않는다. 두 동결 checkpoint와 exact TRAIN-only POINT7040/WEAK44 예측만 root 유한 whitelist에 포함한다.

각 팔·모집단에서 factual forward가 저장 logit과 최대1e-4 이내·sign exact로 일치해야 나머지 조건을 진행한다. 각 조건의 full/native8 CM/P/R/U, logit 변화, sign transition과 정오 변화·원 pair contrast를 모두 공개할 예정이다. score 전 입력동일성을 고정한 pair에서는 nonRGB 동일·RGB 차이가 있는 실제 SOURCE 입력의 logit 대비를 별도 보고한다. variant 이름으로 E/D를 지정하지 않는다.

8개의 표시 PNG는 model score 전에 고정한 point D/E 첫행과 weak factual/cover 첫행의 원96 및 RGB127 복사본이다. 새 cache/학습자료/좌표정답이 아니다. 준비 중 실제 배열·PNG·가중치·점수 접근0이며 authored CPU/NPY/hook/replay10검사만 PASS다. 처음 NumPy scalar를 Torch에 대입한 가상검사 오류는 보존하고 같은127 값을 Python scalar로 대입하도록 수정했다. 실제 forward는 root의 별도 승인 entry 뒤 한 번만 실행한다.
