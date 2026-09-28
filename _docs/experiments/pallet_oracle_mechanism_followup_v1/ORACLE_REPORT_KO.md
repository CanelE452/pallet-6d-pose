# Oracle를 단계별로 분리한 결과

[확인] R0 / Replay9장38점 teacher / 기존RAW320 / 기존REF320의 고정 출력만 사용했다. 아래 숫자는 현재 후보·참조·목적에 한정되며 서로 더할 수 없다. GT선택은 배포 추론에 들어가지 않는다.

| 재료/모델 | 모집단·참조 | 실제production입력 | production | fixed-set/출력oracle | gap | oracle가추가한정보 | coverage | 분류 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Plastic R0 | 128 / geometry-derived DEV | RGB예측+K/치수 | AUC .337965 | W/D .416000 | .078035 | GT자세로후보선택 | 128/128 | 고정후보ADD상한 |
| Plastic RAW | 같은128 | 같음 | .334723 | .414711 | .079988 | 같음 | 128/128 | 고정후보ADD상한 |
| Plastic REF | 같은128 | 같음 | .359016 | .450094 | .091078 | 같음 | 128/128 | 고정후보ADD상한 |
| Wood R0 | 45 / geometry-derived DEV | 같음 | .670500 | .672756 | .002256 | 같음 | 45/45 | 고정후보ADD상한 |
| Wood RAW | 같은45 | 같음 | .656433 | .657889 | .001456 | 같음 | 45/45 | 고정후보ADD상한 |
| Wood REF | 같은45 | 같음 | .665033 | .667400 | .002367 | 같음 | 45/45 | 고정후보ADD상한 |
| Plastic REF기준/4experts | 128 / 같은pose참조 | 기존4개의전체출력 | .359016 | .441824 | .082809 | GT로frame별모델선택 | 128/128 | whole-pose출력선택상한 |
| Wood REF기준/4experts | 45 / 같은pose참조 | 기존4개의전체출력 | .665033 | .710689 | .045656 | 같음 | 45/45 | whole-pose출력선택상한 |
| Plastic REF기준/4experts | 985 / legacy fixedID | 같은semanticcorner | PCK10 507/985 | whole589 / point613 | +82 /+106점 | 참조점으로모델/점선택 | full985, 원래matchgate유지 | 2D선택상한, rigid6D아님 |
| Wood REF기준/4experts | 346 / legacy fixedID | 같음 | 165/346 | whole210 / point228 | +45 /+63점 | 같음 | full346 | 2D선택상한 |
| Plastic verified66 | 16장 / direct-visible | 같은fixedID | REF43/66 | whole53 / point53 | +10점 | 직접클릭참조로선택 | 66/66 | 검수2D선택상한 |

![단계별headroom](figures/oracle_headroom.png)

Wood REF의 W/D oracle(.667400)는 R0 production(.670500)보다 낮지만, 4개 whole-output oracle에는 여지가 있다. Wood R0 자신의 W/D oracle(.672756)와 혼동하지 않는다. 따라서 'Wood에는 정보가 전혀 없다'와 '현재 REF의 W/D 선택으로 해결된다' 모두 관측과 맞지 않는다. 서로 다른 정보/후보 집합의 gap을 합산하거나 둘 중 큰 값을 새 실제 성능으로 쓰지 않는다.

## 모방·참조입력·정확한감독을 구분

- 실사referencexy→D9는Plastic/Wood AUC1.0이었다. 기존참조의생성과같은기하경로이므로순환적consistency이지독립물리6D가아니다.
- 합성64개 exactprojection→production AUC.905797; C1의6개는180°역할모호성이남았다. 정확한renderer대응까지주면최대재투영오차8.13e-6px. 추가correspondence정보를준 numerical solver검사이지학생달성보장아님.
- 합성상속axis수치.484375는 physical/body와camera-facing역할비교가달라 NA_CONVENTION_MISMATCH로표시했다. 유리한대칭으로GT를바꾸지않았다.
- 확인된가시점만치환하는Plastic7C는이번fixedcandidate원인검사에필수아니어서미실행, Wood는NA_REFERENCE_NOT_VERIFIED다. 없는GTboxcrop도추가하지않았다.
- finiteTRAIN학생capability대조는C3별도실험이며유한모델을수학적upperbound로부르지않는다.

## 검출과 국소 이동

Plastic에서미매칭8장만검사했다. 복수후보6장중3장은다른box로IoU조건을만족할수있지만 REF best-keypoint 선택도507→510/985에그쳤다. Wood45는모두매칭해해당재계산이불필요했다. 이를detector병목이완전히없다는뜻으로확장하지않는다.

GT방향으로r≤8px이동할수있다는낙관적원판가정에서 REF PCK10은Plastic507→700/985, Wood165→252/346다. 이미지단서·강체구조를무시한계산이므로edge/색상으로실제로할수있다는증거가아니다. 결측·미매칭점은원래penalty를유지했다. r=0/2/4/8/12를미리고정했으며튜닝기준으로사용하지않았다.

## 실제회수와재현경로

C1은같은후보·9점·기하penalty에Huber12를적용했지만선택변경0장, ADDsym AUC차이0, 고정후보gap회수0%였다. 높은oracle가강건residual점수만으로회수가능하다는가설은이설정에서지지되지않았다. native출력을바꾸는C2/C3에는이fixed-set회수율을부여하지않는다.

실행코드: `pose_oracle freeze/score/sanity`, `pose_cues`, `pose_sanity_detail`, `coordinate_oracle freeze/score`, `cycle_pose`. 정확한명령·입력hash·전체recording/severity·역순불변성과수치parity는 [자세상세](ORACLE_POSE_REPORT_KO.md), [좌표상세](ORACLE_COORDINATE_REPORT_KO.md), [C1실행](cycles/C1_HUBER_D9/REPORT_KO.md)에있다. 0–.1지름normalized ADD를1001threshold사다리꼴적분했고실패분모를유지했다. 원시좌표/GT선택은private결과namespace에격리했다.
