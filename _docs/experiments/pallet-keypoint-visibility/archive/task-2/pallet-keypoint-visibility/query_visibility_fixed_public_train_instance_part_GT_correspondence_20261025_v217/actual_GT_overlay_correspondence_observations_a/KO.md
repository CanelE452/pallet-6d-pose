# V217 A · 실제 15 GT overlay 독립 관찰

고정15 후보 panel과 별도 원 context7장을 `tools.view_image(detail=original)`로 실제 보았습니다. 모든 view 전에 선언된 owned PNG27개의 encoded SHA/크기를 확인했습니다. 별도 context를 열지 않은 1/3/5/7/8도 panel 안 원 context는 보았지만 standalone view로 세지 않습니다. root 새 관찰은 읽지 않았습니다. A는 renderer 저자이며 V215/V216 SOURCE와 이전 weak 판단/GT 준비에 노출됐으므로 blind·사람 gold가 아닙니다.

| frame/candidate | 물리 문맥 관찰 | 구체 근거와 한계 |
| --- | --- | --- |
| 0/0 | BROAD_VISIBLE_PALLET_ALIGNMENT_SUPPORTED_WEAK | full/visible 색 영역은 화면 아래의 가로 판 팔레트와 넓게 겹치고 visible 영역에서 판의 틈이 구분된다. 오른쪽 판 끝 주변과 정렬되는 근거가 있다. 왼쪽·아래 팔레트가 프레임 밖으로 잘려 전체 형상과 아래쪽 인스턴스 경계를 확인할 수 없다. 넓은 정렬은 detector의 선택 인스턴스 또는 8개 native-part 교차 대응을 인증하지 않는다. |
| 0/1 | PARTIAL_OR_AMBIGUOUS_SECOND_CANDIDATE | full 영역은 아래 왼쪽에서 오른쪽 아래로 비스듬하게 지나고 visible은 일부 가는 판/틈 주변 조각이다. 첫 후보의 넓은 가로 상판과 다른 지지 구조를 가리킬 가능성은 있다. 잘린 아래쪽 구조가 독립 팔레트인지 패널만으로 분리되지 않는다. visible 조각이나 낮은 지원만으로 외부 가림·대상 부재를 결정하지 않는다. |
| 1/0 | NO_INFRAME_PALLET_MASK_ALIGNMENT_TO_CONFIRM | 패널에서 full/visible 색 변화가 보이지 않고 원 영상은 플라스틱 상자·검은 받침을 보여준다. query도 그 상자/받침에 놓인다. 팔레트 candidate의 화면 밖/저장 bbox sentinel/가림/미지원 가능성을 이 view로 구분하지 못한다. 빈 overlay는 native E 근거가 아니다. |
| 1/1 | NO_INFRAME_PALLET_MASK_ALIGNMENT_TO_CONFIRM | 두 번째 후보도 full/visible 색 변화가 보이지 않는다. 보이는 상자와 받침의 물리 구조는 첫 후보 패널과 같다. 후보0/1을 원 영상만으로 고를 근거가 없다. 자동 첫 인스턴스 선택이나 다른 물체를 팔레트로 대체하지 않는다. |
| 2/0 | BROAD_VISIBLE_UPPER_PALLET_ALIGNMENT_SUPPORTED_WEAK | visible 색 영역이 위에 놓인 팔레트의 길쭉한 상판들과 왼쪽 앞면을 넓게 따른다. 원 context에서도 아래 팔레트와 쌓인 구조가 보인다. 판 끝/틈과 겹친 두 팔레트의 가까운 경계는 축소 overlay에서 일부 모호하다. native point/selected detector instance의 전역 대응은 확정하지 않는다. |
| 2/1 | PARTIAL_LOWER_STACK_ALIGNMENT_PLAUSIBLE | full은 위 팔레트에 가려진 영역까지 넓게 덮고 visible은 오른쪽 아래에서 드러난 하단 팔레트의 판/블록 주변과 일부 틈에 나타난다. 아래에 다른 팔레트가 실제로 보이는 것과 일관된다. 많은 query는 위 팔레트에 가까우며 아래 후보에 독점적으로 닿는 query 부분집합은 확인하지 못했다. full 중첩이 실제 visible part라는 뜻은 아니다. |
| 3/0 | BROAD_FRONT_PALLET_ALIGNMENT_SUPPORTED_WEAK | full/visible이 전경 EPAL 팔레트의 상판·블록·아래 판을 넓게 따른다. 위의 파란 상자/목재 상자/회색 물체와 전면 팔레트를 시각적으로 분리할 수 있다. query는 뒤 목재 상자 윗변 쪽이다. 팔레트 mask와 물리 팔레트가 잘 맞아도 현재 q/native가 그 부위를 주소화한다는 증거는 별도다. |
| 4/0 | BROAD_FRONT_PALLET_ALIGNMENT_SUPPORTED_WEAK | full/visible이 왼쪽 EPAL 블록부터 오른쪽 전면까지 이어지는 팔레트와 넓게 맞는다. 상판 위 물체는 visible에서 대체로 분리된다. 작은 가장자리 어긋남·가려진 판/블록의 구체 부위와 n4/n5의 다른 물체 위치를 함께 남긴다. V216 n3의 서로 다른 ROI feature 문제는 이 넓은 mask로 해결되지 않는다. |
| 5/0 | BROAD_PALLET_WITH_OCCLUDERS_ALIGNMENT_SUPPORTED_WEAK | full은 상판/측면의 팔레트 전체 영역을, visible은 위의 파란 상자·목재 상자·회색 통을 제외한 노출된 판을 주로 따른다. 왼쪽 판 끝은 원 context에서 구별된다. full이 상자/통 위를 덮는 부분은 숨겨진 nominal silhouette 문맥이지 그 물체가 팔레트 표면이라는 뜻이 아니다. 각 native의 가림/SELF는 미판정이다. |
| 6/3 | BROAD_UPRIGHT_REAR_PALLET_ALIGNMENT_SUPPORTED_WEAK | full/visible이 뒤에 세운 넓은 판 팔레트를 따른다. visible은 앞 파란 통과 목재 상자에 겹치는 부분을 대체로 제외한다. 앞의 작은 목재 판/큰 상자는 별개 구조다. n5 배경과 n7 판 끝면/아래 변의 차이는 전체 mask만으로 동일 intended feature가 되지 않는다. |
| 7/3 | BROAD_UPRIGHT_REAR_PALLET_ALIGNMENT_SUPPORTED_WEAK | 뒤 팔레트의 가로 판들이 full/visible 영역과 넓게 맞고 앞 파란 통은 visible에서 빠진다. 원 q8은 앞 목재 상자/판 쪽에 있다. 뒤 팔레트와 앞 목재 재질의 유사함 또는 full 겹침만으로 native 대응·E를 만들지 않는다. |
| 8/3 | BROAD_UPRIGHT_REAR_PALLET_ALIGNMENT_SUPPORTED_WEAK | 기울어 보이는 뒤 팔레트의 가로 판과 visible 색 영역이 넓게 일치한다. 앞 통/상자는 별개 전경으로 구분된다. query가 있는 전경 상자 일부에 full 영역이 겹치지만 visible part/native 주소화의 증명이 아니다. 가림 원인의 각 native 상태는 별도다. |
| 9/0 | PARTIAL_WOOD_SUPPORT_CONTEXT_PROXY_ALIGNMENT | full은 아래 목재 지지판/팔레트 후보 영역을 넓게 덮고 visible은 노출된 바깥 목재와 함께 흰 덮개·상자 테두리/안쪽 물체에도 색이 보인다. 물리 팔레트 구조가 대부분 상자·흰 덮개 아래에 있어 정확한 부위를 인증하기 어렵다. visible 색이 입혀진 다른 표면을 target part로 확정하지 않는다. q8은 뒤 손잡이에 있다. |
| 10/0 | PARTIAL_WOOD_SUPPORT_CONTEXT_PROXY_ALIGNMENT | 목재 지지판의 외곽과 넓은 full 영역은 대체로 일관되지만 visible 색은 상자 안쪽/흰 덮개 및 테두리에도 걸린다. 화면 아래 잘림과 덮인 구조 때문에 예상 native part의 가시성은 미확정이다. 빈/겹친 mask·깊이 구멍을 native E로 해석하지 않는다. q8은 카트 손잡이에 있다. |
| 11/0 | PARTIAL_WOOD_SUPPORT_CONTEXT_PROXY_ALIGNMENT | 긴 목재 지지판의 오른쪽 노출 면과 full/visible 외곽이 대략 맞는다. 원 context에서 흰 덮개와 회색 상자가 목재 구조 대부분을 덮는 것을 볼 수 있다. visible 색이 흰 덮개·상자/통 쪽에도 있으므로 mask 점유만으로 같은 팔레트 부위라고 하지 않는다. q5/6 목재 가장자리와 q0 흰 덮개/q4·7 상자 경계를 분리한다. |

## 공통 해석

전경 EPAL 팔레트(3/4/5)와 뒤에 세운 팔레트(6/7/8)에서는 full/visible 영역이 넓은 물리 구조와 일치하는 약한 근거가 있습니다. 겹친 두 팔레트(0/2)는 후보 간 경계·가림·화면 잘림을 남겼습니다. 1은 색 영역이 보이지 않는다고 E를 만들지 않았습니다. 9/10/11은 목재 지지판/팔레트 후보 대부분이 상자와 흰 덮개 아래에 있고 visible 색이 다른 표면에도 보이므로 점유를 동일 intended part로 인증하지 않습니다.

q 묶음이 실제 팔레트 부근인 경우와 상자·카트 손잡이인 경우를 별도 기록했습니다. 특히3의 목재 상자,7/8의 앞 목재 상자,9/10의 손잡이는 선명한 feature여도 GT 팔레트 native 대응이 아닙니다. mask·OOB·깊이 결측·query OFF만으로 native E/SELF를 생성하지 않습니다.

원96/U11·FIT64DEV32를 유지했고 candidate 반복 U는17입니다(이전14 가정과 구분). V216 weakpartD9는 역사적 assistant weak 근거이며 이번에 native label로 복사하지 않았습니다. 모든 selected detector instance는 null/UNKNOWN, native part projection은 아직 미실행, 신규 D/E/SELF gold와 training admission은 없습니다. 완벽한 mesh/corner 인증을 blanket gate로 요구하지 않으면서도 전체 mask 정렬만으로 해당 native part를 확정하지 않습니다.

[전체 관찰 JSON](ALL15_CANDIDATE_AND_FRAME12_OBSERVATIONS.json) · [15행 CSV](ALL15_CANDIDATE_OBSERVATIONS.csv) · [실제 view receipt](ACTUAL_IMAGE_VIEW_RECEIPT.json) · [schema QA](SAVED_OBSERVATION_SCHEMA_QA.json)
