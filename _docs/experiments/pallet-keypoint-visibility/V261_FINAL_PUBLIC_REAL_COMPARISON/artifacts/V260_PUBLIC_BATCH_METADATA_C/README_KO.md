# V260 public Pallox batch metadata closure

실행한 것은 **공식 API725973B와 README4874B 두 파일의 metadata-only join**입니다. 새 network/download/decode·pose/GT/body·RGB/depth/mask/mesh·model·protected/current/SourceDEV 접근은0입니다. 모델·문턱·학습 recipe를 선택하거나 채택하지 않았습니다. 완전284 bundle의 정확 fileID/원 publisher path/size/MD5와 directory별 K를 닫았고, 누락 항목도 보존했습니다.

[전체284 manifest](metadata_v1/ALL284_FOUR_WAY_METADATA_MANIFEST.json) · [불완전38 전체](metadata_v1/INCOMPLETE_BUNDLES_RETAINED.json) · [묶음 census/split 제안](metadata_v1/GROUP_CENSUS_AND_SPLIT_PROPOSAL.json) · [실행 영수증](metadata_v1/COMPLETE.json)

| 공식 height/environment 묶음 | RGB | depth | pose | SAM mask | 4-way완전 | 불완전 | hash 제안 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.55m_hall | 48 | 48 | 28 | 30 | 28 | 20 | TRAIN |
| 0.55m_lab | 65 | 65 | 52 | 56 | 52 | 13 | TRAIN |
| 1.23m_hall | 48 | 48 | 48 | 48 | 48 | 0 | TRAIN |
| 1.23m_lab | 65 | 65 | 60 | 60 | 60 | 5 | DEV |
| 1.92m_hall | 40 | 40 | 40 | 40 | 40 | 0 | TRAIN |
| 1.92m_lab | 56 | 56 | 56 | 56 | 56 | 0 | DEV |

RGB/depth stem union322개 중284개가4종을 모두 갖고38개는 불완전합니다. pose38개/mask32개가 없으며 누락 이유와 exact path를 숨기지 않습니다. 완전284에는 이미지별4종1136개와 shared K6개, 총**1142 unique public files / 161,763,540 publisher bytes**가 필요합니다. 이 크기·MD5는 공식 metadata값이며 파일을 새로 읽어 검증한 결과가 아닙니다. occlusion/visibility/modelscore를 보고 제거하거나 고른 항목은 없습니다.

## Capture grouping의 정확한 수준

현재 그룹은 공식 `height_environment` directory 여섯 개입니다. 이는 동일 묶음의 거리·yaw·occluder sibling을 섞지 않기 위한 보수적인 **coarse capture collection**이며, 서로 독립인 실제 acquisition session임을 증명하지 않습니다. pose-file MD5 alias도 metadata로 보존했지만 byte동일pose와 물리적 capture동일성은 다릅니다. 더 세밀한 capture crosswalk는 이 metadata에 없습니다. 같은 directory를 양쪽으로 나누지 않는 것과 clean/freshblind 성능 증명은 구분합니다.

사전 고정 salt `V260_PUBLIC_CAPTURE_GROUP_HASH_V1:`+directory의 SHA256 순위 첫2개를 DEV, 나머지4개를 TRAIN으로 **제안만** 했습니다. 따라서1.23m_lab60+1.92m_lab56, 합계116개가 DEV 제안이며 나머지168개는 TRAIN 제안입니다. frame appearance/label/occlusion/실험 수치를 읽고 조정하지 않았습니다. 실제 역할은 모두 null이고 training_admission=false입니다. Root가 future protocol에서 채택하지 않은 상태입니다.

이미 진단에 사용된 `0.55m_hall/0000`, `0.55m_hall/0011`을 개별 exposed로 표시했고, 전체0.55m_hall 묶음도 known-exposed 표시했습니다. 같은capture/collection을 cleanheldout이라고 부를 수 없습니다. 다른 과거 exposure를 전수 감사했다는 주장도 하지 않습니다.

사람이 촬영한 current223과 Pallox가 무관하다는 사실을 MR6D45와의 비중복 증거로 전용하지 않습니다. 앞선 publicRGB encoded-byte no-hit만으로 reencoding·같은capture를 배제할 수 없고 crosswalk는 UNKNOWN입니다. 이것 때문에 공개 source의 metadata 준비·명시적 source취득을 불가능한 전역증명 대기로 막지는 않습니다. 다만 보호 benchmark 결과를 사용해 clean성능을 주장하거나 training/adoption을 자동 승인하지 않습니다.

## 준비된 deterministic collector / future downloader

`public_batch.py`는 stdlib만 사용합니다. 현재 metadata collector는 아래 entry로1회 실행했으며 terminal a0bf1c exit0/PASS입니다.

```sh
python -B public_batch.py --collect-entry ROOT_METADATA_ONLY_ENTRY.json
```

위 경로에는 원 source의 exact absolute metadata bindings가 있어 동일 snapshot을 갖춘 환경에서만 동작합니다. 결과는 fresh output에만 작성합니다. 새 decode나 download는 없습니다.

`DOWNLOAD_ENTRY_FALSE_TEMPLATE_ALL284.json`은284 exact bundleID와1142개 fileID/path/size/MD5/official HTTPS URL을 모두 열거한 **false** entry입니다. 현재 실행하면 승인을 거부하며 아직 다운로드하지 않았습니다. Root가 future scope/trueentry를 봉인하면 다음 전용 경로를 사용할 수 있습니다.

```sh
python -B public_batch.py --download-entry ROOT_FUTURE_TRUE_DOWNLOAD_ENTRY.json
```

Root가 명시한 complete bundle 집합의4종+K exact allowlist 외 파일은 받을 수 없습니다. 공식 HTTPS exactURL/HTTP200만 허용하고 redirect·자동재시도·기존out/resume·symlink·v112/cache/pyc를 거부합니다. 파일마다 streaming MD5/size와 새 SHA256을 기록하며 원 basename/publisherpath를 보존합니다. 불일치/실패 시partial과FAILURE를 남기고 멈춥니다. decode·pose해석·labelcompiler·모델·평가·학습은 전혀 포함하지 않습니다. Mesh는 이 batch downloader의 대상이 아닙니다.

새 batch teacher/feature/학습 실행은 별도의 Root entry가 필요합니다. 이 준비 기록은 unavailable physical-native support를 source준비의 조건으로 추가하지 않으며, surface AUX와 native/literalqgold를 혼동하지 않습니다.
