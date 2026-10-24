# V217 · 고정 12 TRAIN의 GT 바인딩과 선택적 reader 준비

같은 SOURCE 12프레임·96개 원 native 입력만 연결했습니다. 원 q/L/U11, outer TRAIN, 내부 FIT64(U9)/DEV32(U2)를 그대로 보존했습니다. prefix 000000/000002/000005/000006마다 3프레임입니다. 추가 prior text 6개와 scope의 V216 메타데이터 4개만 읽었으며 실제 GT/camera/mask/depth/CAD/RGB/NPZ 파일의 stat·hash·read·decode는 **0**입니다. 모델·점수·학습·투영·PnP·Git도 0입니다.

## 실행 가능한 경로 목록

- obj1 GT-list 후보 인스턴스는 총 15개입니다. 같은 obj1이라도 첫 후보나 q에 가장 가까운 후보를 선택하지 않습니다. 12프레임 모두 선택 인스턴스 대응은 현재 UNKNOWN입니다.
- payload 경로는 총 55개: scene_gt 4 / info 4 / camera 4 / 전체 마스크 15 / 가시 마스크 15 / depth 12 / nominal obj1 CAD 1입니다. RGB·원 NPZ·YCB donor는 포함하지 않았습니다.
- 47개는 과거 영수증에 raw SHA256이 있습니다. info/camera 8개는 V153 writer의 정확한 고정 경로만 확인됐고 **이전 raw SHA는 UNKNOWN**입니다. 미보유라고 결론내리지 않고 현재 실재성도 검사하지 않았습니다.

[12프레임 경로/역할 CSV](FIXED12_TRAIN_GT_BINDING_READINESS.csv), [55개 whitelist](EXACT_PUBLIC_GT_PAYLOAD_WHITELIST.json), [원96 입력·GT 포인터](ALL96_ORIGINAL_INPUTS_AND_GT_POINTERS.json), [SHA 미상 정확8경로](EXACT8_PRIOR_SHA_UNKNOWN.csv)를 제공합니다. 원 슬롯별 availability가 false이면 finite q나 GT가 있어도 U를 보충하지 않습니다.

## 가용성·의존성 분기

1. 기존 SHA가 있는 scene_gt 4개만 새 root entry로 확인할 수 있습니다. camera/info 또는 이에 의존하는 nominal 근거는 UNKNOWN으로 남깁니다. camera SHA가 없다는 이유로 독립적인 기존 mask/depth/scene_gt 근거까지 영구 중단하지 않습니다. 이 reader 자체는 이미지·깊이·CAD를 열지 않습니다.
2. 별도 root entry가 정확8경로의 최초 opaque 존재/바이트 SHA 기록을 명시 승인하면, 기존 로컬 info/camera를 확인하고 **선택 값 parse 전에 실제 SHA 영수증**을 먼저 씁니다. known SHA가 다르면 실패합니다.
3. 파일이 없다면 자동 다운로드/대체 경로를 찾지 않습니다. 별도 승인용으로 pinned commit URL과 새 소유 intake 경로만 미리 제안했습니다. 그 후 새 바인딩·reader entry를 고정해야 합니다. [분기·정확 URL 제안](AVAILABILITY_AND_DEPENDENCY_BRANCHES.json)

## mixed JSON에서 TRAIN 값만 읽기

reader는 허용 프레임 key 문자열을 먼저 확인합니다. 허용된 값 조각만 JSON decoder로 넘기고, 나머지 값은 문자열 escape·중첩 괄호를 고려해 구조 경계만 건너뜁니다. 전체 JSON을 json.loads한 뒤 filter하지 않습니다. 선택 key 누락·중복·선택 값 schema 불일치는 명시 실패이며 다른 frame/동일 obj1로 채우지 않습니다. 건너뛴 값의 의미·라벨·수치 문법을 검증했다고 주장하지 않습니다.

향후 실행은 반드시 새 승인 entry가 필요합니다. 현재 [실행 템플릿](ROOT_ACTUAL_SELECTED_GT_ENTRY_TEMPLATE_APPROVED_FALSE.json)은 approved=false입니다.

```bash
/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B read_fixed12_selected_GT_metadata.py --entry <새 root 승인 entry 절대경로>
```

## 기존 방법과 권위의 경계

V154는 GT-instance/역할 제안과 실제 query의 entity·part 대응을 분리해 연결했고, V159는 nominal 부위/마스크/owner의 한계를 이미 조사했습니다. 이번 준비를 첫 GT 활용으로 부르지 않습니다. 해당 문서에서 SOURCE 계약만 사용했고 과거 held 집계·라벨을 새 분석하지 않았습니다.

원 nominal 방향은 `dynamic[k]=fixed[perm[k]]`이며 이미 dynamic인 배열에 다시 적용하지 않습니다. SOURCE pose의 nominal projection은 별도로 고정한 실제 타깃 진단에서 제안할 수 있지만, 지금 실행하지 않았습니다. 원 q/L/native/U를 옮기거나 GT-best 순열·PnP·좌표 최적화로 맞추지 않습니다. 전체 실루엣·가시 표면·깊이·query owner·의도 부위·native visibility를 각각 남깁니다. 마스크 음성/다른 물체@q/쿼리 OFF가 E는 아닙니다. 완벽한 mesh/정확한 3D corner를 모든 weak 근거의 blanket gate로 요구하지도 않습니다.

[인스턴스·native·증거 계약](INSTANCE_NATIVE_AND_EVIDENCE_CONTRACT.json)과 [실행 전 프로토콜](PROTOCOL_BEFORE_ANY_PUBLIC_GT_PAYLOAD.json)에 이 경계를 고정했습니다. 모든 training admission은 false이며 새 native label은 만들지 않았습니다. DEV는 미래 모든 training branch와 파생 variant에서 제외해야 하며, 이미 노출됐던 source나 과거 all-SOURCE warmstart는 fresh blind/독립성을 자동 보장하지 않습니다.

[23개 authored 기능 검사](INVENTED_FUNCTIONAL_CHECKS_V2.json)는 선택 값만 decode, nested/escape skip, key 누락·중복·경계 실패, all96/all8/U11, role 혼합 거부, known-only 분기, 최초 SHA 승인, FS 전 V112/미선언/변조 거부, symlink 거부를 포함합니다. 실제 payload 실행은 하지 않았습니다.
