# V242 독립 SOURCE scene data QA

Root 별도 QA entry 전 실제 dataset/base RGB 접근은0입니다. Root 고정 all FIT D5406 및 기존 donor64/원q8 정책만 검산합니다.

`python -B verify_scene_pairs.py --entry ROOT_ACTUAL_QA_ENTRY.json`

새10NPY+geometryNPZ SHA/header를 확인하고, base804 선언SHA 뒤 PNG 또는 RGB_uint8 단일NPZ멤버만 디코딩합니다. 저장 carrier/alpha로 native composite를 독립 복원하고 producer 운용코드는 import하지 않습니다. V90 crop 및 V107 순수함수만 AST로 추출해 all8와 context를 전량 재샘플합니다. 두64MiB LRU+readonly FD memmap을 사용합니다.

CONSTRUCTED_UNKNOWN도 실제 입력을 비교하며 label만-1입니다. UNCONSTRUCTED만 입력0입니다. 원D와 old39failure를 유지하고 r의continuous positive tap+nearesthalfup3x3를 새로검산합니다. q-r/문자그대로q의opacity/support/동일RGB상반타깃은 진단일뿐 라벨·부모선택을 바꾸지 않습니다. YCBassociated-r는physical pallet native gold가 아닙니다.

Donor PNG/mask는 추가로 열지 않습니다. donor와공유하는61 RGB경로는 승인된base804역할로 읽힐 수 있습니다. rawGT·모델·특징·current/held는 읽지 않으며 새 CF생성·학습도 하지 않습니다. 첫 작은가상 NPY 시험의 private API 오류는 보존했고 공개header API로 교정해 1.0/2.0 둘다 통과했습니다.
