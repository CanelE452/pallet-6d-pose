기존 V239/V245 CODE·schema 준비 점검을 마쳤습니다. 실제 payload·가중치·이미지·배열·교사 행·모형 실행·학습은 0입니다. 선언된 SOURCE 수량을 재집계하거나 보호 평가 결과로 방법을 고르지 않았습니다.

기존 입력은 known7,617/FIT5,876/DEV1,741이며, 원15,040행 중 S/U7,423행은 별도로 보존합니다. output0은 SYN의 saved intended-native witness0, output1은 YCB associated-r surface입니다. 두 목표는 binary BCE를 family별로 적용하며 YCB는 native UNKNOWN/condition32=0입니다. 원 SOURCE DEV는 이미 노출·재사용된 내부 개발 자료입니다.

특징1186은 full96 CLS384 + 고정 중앙 Q384 + centre32 CLS384 + 유효 비율2 + 원 condition32입니다. q8 nodeCLS8과 원 q8 minmax contextCLS를 relational residual에 추가합니다. frozen DINO, 고정 ImageNet 정규화와 L2만 쓰며 fitted normalizer가 없습니다. node_available는 원 availability·central4·any-valid의 입력 조건이고 교사 상태로 마스크하지 않습니다.

V239는 같은 seed17/base 초기값·PCG17 1000×32 draw로 두 head를 각1,000회 학습했습니다. V245는 원 V239 context baseline의 CP/log/CSV를 바이트 재사용(new0)하고 fresh treatment만1,000회 학습하는 data 대조입니다. 각 family8D+8E의 BCE 평균을 반반 합하고 AdamW .001/wd.0001, logit0을 고정합니다. 새 CF는 마지막4D를 같은 부모 SHAM, 대응4E를 COVER로 교체하며 proof 실패 시 양쪽 원 자연 슬롯을 유지합니다. 교사 footprint를 증명할 뿐 q중앙/전체 part/native gold는 아닙니다.

기존 SOURCE gate는 mean family DEV balanced recall +3pp, 각 D/E 하락≤2pp, YCB DEV E≥50%, SYN DEV D/E 각각≥90%입니다. 이는 배포 성공 기준이 아니며 새 AUX recipe로 자동 상속하지 않습니다.

Pallox dense conditional pixel은 원 native8에 끼우면 안 됩니다. 별도 AUX ID·샘플 q/L·certificate/교사 authority·V0/E1/UNKNOWN·native_slot=null·SELF=null·원 TRAIN/group lineage를 기록해야 합니다. 원352 initializer query/qL/U는 그대로 남깁니다. Dense q가 원 detector q라는 주장을 하지 않습니다. SAM negative와 instance 불일치는 E 근거가 아닙니다.

Root가 요청한 공유 stem 접점은 CODE상 가능합니다. 정확 head는 V237_RELATIONAL_QUERY_DESIGN_B/relational_query_head.py(dfd26ab7…): base.head[0]=Linear1186→64, [1]=ReLU, [2]=기존2출력입니다. AUX pixel1154에 zero32를 붙여 동일 [0]/[1]을 통과한 뒤 별도 publicpixel scalar를 둘 수 있습니다. 이 경로는 q8/index/relational을 호출하지 않고 기존 mainforward를 보존합니다. 분리·detach한64-feature를 새 head에만 주면 공유 stem 학습이 아니므로 허용 접점이 아닙니다. AUX readout을 zero-init할 경우 첫 backward의 stem gradient는 0이라는 시점도 공개해야 합니다. 아직 이 후보를 구현하거나 선택하지 않았습니다.

실행 전 필요한 Root 결정은 AUX 역할 분리/그룹 역할·q/L 샘플 규칙·quota·loss weight·초기화·고정 schedule·gate입니다. 기존 trainer는 N/family/32-draw/2출력이 하드코딩되어 직접 append 실행이 불가능합니다. 새 versioned wrapper가 필요하지만 이번 산출물은 interface 제안과 기존 코드 closure까지입니다.

정확한 코드·schema·행 근거는 READINESS.json, path/SHA와 열지 않은 payload dependency는 DEPENDENCY_CLOSURE.json, 새 AUX 제안은 PUBLIC_PIXEL_AUX_INTERFACE.json에 있습니다. 기존 frozen CODE와 자료는 수정하지 않았습니다.
