# 비대상 입력 외곽선 기능 시험의 사전 고정 계약

이 문서는 신규 관측/세 경로를 새 MobileSAM 출력과 새 자세 선택을 보기 전에 고정한다. 원결과 main은 `4d509f66e0e544a970cd039afb5d33c94680d394`다. 이전 감사의 범주·좌표·비용차를 추론 입력으로 쓰지 않는다. 여섯 원 RGB만 본 별도 문맥의 모델이 작성한 지시 manifest는 SHA `36a8072c14897acfd29b04e954fb1f33366b2feb27924ffbffe4a6b36b26b030`으로 동결됐다. 사례 선정은 사후이며 독립 블라인드 검증이 아니다.

이번 최소 경로는 ASSISTED6이다. 범주는 `visible traffic cone` 하나다. 명확한 콘이 보인 프레임에만 원좌표 박스와 내부 양성점 하나를 썼다. 나머지는 `NO_ADMISSIBLE_NON_TARGET_EVIDENCE`이며 검증된 배경/가림없음으로 바꾸지 않는다. 입력은 모델 보조이고 SAM이 객체의 의미나 깊이 순서를 인증하지 않는다. 기존 고정 MobileSAM의 box+point, multimask=True를 각 지시당1번 사용한다. RGB embedding은 지시 있는 영상당1번이다. 원 대안·quality를 모두 보존하고 최대 quality, 정확 동률 첫 원순서를 사용한다. 새 pose를 보기 전 mask 전체를 한번 검토하며 명백한 팔레트 혼입이면 미확정으로 되돌린다. 대안 재선택·프롬프트 보완·반복 탐색은 없다.

원 target mask·polygon·clipped finite segments·sample을 보존한다. 신규 부모별 유효성은 **preclip hull edge의 두 원끝점 중 하나라도 동결 non-target binary mask 양성이면0, 나머지1**이다. 최근접 픽셀은 `floor(x+0.5),floor(y+0.5)`이며 원 hull 끝점은 정수 픽셀임을 검사한다. 복원 불가/입력 미확정은 원값을 유지한다. 원 mask 차집합·팽창·새 HSV/명암 임계값·새 closed hull·숨은 물리 경계는 만들지 않는다.

observed→model은 원 표본과 원 가중 기여에서 censored parent만0으로 둔다. model→observed는 **원 전체 finite segments에서 원순서 tie로 찾은 nearest parent**가0일 때 그 표본의 원 가중 기여만0으로 둔다. retained segments로 nearest를 다시 찾지 않는다. 후보별 원 model sample/호 길이·양방향1/2·outline factor·원 normalizer를 유지한다. 기존 weighted contribution에 이미 들어간 가중치를 다시 곱하지 않는다. 같은 원관측은 family/pose와 무관한 지지를 갖는다. 실제 stage 관측이 다르면 각각 연결한다.

| 신규 경로 | 점수 |
|---|---|
| BASELINE_E | 기존 모든 항·기존 계층·기존 tie/자기 fallback |
| SUPPORT_AWARE_RERANK | 위의 양방향 sample 결측 처리; non-outline 항/제약 원값 |
| MASS_MATCHED_CONTROL | 같은 후보·입력의 각 방향 retained 원가중 mass 비율로 원 방향 비용 전체를 일괄 축소 |

새 두 경로는 입력 없음/유효 부모 삭제 없음에서 원 scalar와 순위·최종 pose를 정확히 반환한다. G10 각 family 내부의 원 retained 전부를 정렬하고 원 accepted family comparator로 family를 고정한 뒤, 해당 family의 G13 retained 전부를 정렬한다. G10과 G13 목적함수는 서로 비교하지 않는다. 새로운 pose·seed·최적화·dedup·제약 변경은 없다. Q는 후보의 자기 native basis를 사용한다.

privileged exporter는 기존 후보/비용 자료에서 허용한 pose/cost/geometry 필드만 내보낸다. supplier는 neutral RGB와 동결 지시만, selector는 reference-free neutral bank와 동결 새 입력만 읽는다. reference/GT/audit-label sentinel과 파일 접근 검사를 둔다. BASELINE 재현·순수 geometry·무입력 항등성을 먼저 통과시키고 새 mask를 확정한 뒤 새 예측을 봉인한다. **봉인 후 별도 evaluator만 기존 참조를 읽는다.** 성공은 같은 후보의 위치≤5cm AND physical/C2 회전≤5°이며90° 동등성을 추가하지 않는다.

AUTO는 명시적으로 cone을 지원하는 기존 모델의 API·실제 가중치·라이선스가 확보된 때만 별도 진입한다. COCO80 checkpoint만으로 cone 공급을 가정하지 않는다. 없으면6장 보조 결과와 공백을 보고하고238장 수동 확장을 하지 않는다. 자동 공급이 확보되면 결과 우열과 무관하게 고정 동일 방법으로 PRIMARY192/CAL46 분리 실행한다.

새 학습·외부 모델 설치/다운로드·유료 API·RGB 외부 전송·PnP·수치 최적화·새 후보·가중치/임계값 탐색은0이다. GPU동시1/관련실행30분, CPUworker최대2/분석120분을 넘기지 않는다. 추가 비용은 encoder/decoder·선택을 포함해 계측하되 캐시 시간을 전체 RGB→pose FPS로 부르지 않는다. 이전 결과·상태·가중치·참조·논문·운영 기본값은 불변이다. 새 namespace 게시와 운영 채택은 구분한다.
