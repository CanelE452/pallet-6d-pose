# 시행·구현 수정 기록

기존 실험·실패 저널은 변경하지 않았다. 다음은 이번 작업의 수정과 보존 범위다. 결과가 나쁘다는 이유로 경로나 ID를 제외한 기록은 없다.

| 시행 | 발견 | 조치와 보존 |
|---|---|---|
| 기존 후보 읽기 | G10 8452 outcome 중8353, G13 8456 중8357에 R/t 없음 | 원 bank 전량을 별도 추출. 없는 좌표를 만들지 않고 허용된 원 e 로깅 replay와 분리 |
| 초기 관측 추적 인증 | CAL/TEST12개 보정 좌표/source가 기존 e_trace가 읽는 evaluation_v3와 다름 | 원 e가 실제 사용한 V47 추가복구 저널에 연결. 실패 추출 폴더 보존. 새 solve/NN0 |
| 초기 형식 감사 | 옛 e_trace가 legacy partial72개를 full처럼 표시 | 실제 consumer의 전체 partial-ID 제외 규칙으로 신규 감사 출력. 기존 e_trace 소스 불변 |
| 초기 전이 집계 | N3 source channel 링크와 원 e owner 링크가 다름 | 두 링크를 분리. 원 e owner 기준 partial→post-full381을 사용하고 서로 다른409를 대체값으로 섞지 않음 |
| REOBS 단위 검사 첫 시도 | cache의 `result.observation` 중첩을 fixture가 한 단계 덜 읽음 | fixture 수정 후 통과. 수치 solve0, 새 packet 내용 변경0 |
| TYPED 첫 fit 전 metadata 검사 | 원 e finalizer의 accepted source 태그와 wrapper 중간 태그 차이 | 원 source 태그를 복원하고 전후 소스 SHA receipt 기록. pose 수치·후보 순서·수락 규칙 변경0 |
| REOBS pilot cache 재사용 바인딩 | 사용하지 않는 typed 모듈까지 묶은 과도한 소스 바인딩 | packet 및 실제 소비하는 solver/common 바이트 일치로 재사용. 기존 cache와 원 바인딩 보존. 재계산0 |
| cache 정적 검토 | 재사용 시 명시적 arm/index/id 검사가 부족 | 검사 추가 및 모든 최종 슬롯의 독립 대응 검사. 수학식·기존 cache 값 변경0 |
| 독립 parity 명령의 기존 receipt 재작성 시도 | 독점 생성 `open('x')`가 기존 파일을 보호하며 거부 | 기존 receipt 보존. 새 solver0. 숫자에 맞춘 허용치 완화0 |
| 독립 재집계 첫 시도 | FIXED_REFERENCES 파일의 `references` wrapper를 한 단계 덜 읽음 | 구조 읽기를 수정하고 실패 로그 보존. 거리·회전 공식과 허용치 변경0, 새 fit0 |
| 후보 완전성 범위 정정 | 과거 부모 단계의 탈락 후보를 원 e의 실제 consumer 후보와 혼동 | 실제 consumer로 전달된 시작·미보정 시작·최적화 종점을 구분. 과거 전체 bank 완전성은 주장하지 않으며 이번 consumer는238/238 확보 |
| CAL1 시작점 로깅 공백 | 기존 e에서 생성한 P3P 시작의 R/t가 원 저널에 없음 | 지정된 원 generator만 호출해 원51/교환50 시작의 이름·순서·검색 metadata 일치 확인. 새 seed·fit·top-level slot0. 원 저널과 supplement 각각 보존 |
| scoring metadata 운반 | TS의 SINGLE 범위·E0 초기화, TM의 E0 비사용과 NOT_ISOLATED 표시 누락 | score_v1 보존 후 score_v2에 metadata만 정정. 4522 metric dict·76 집계·80 짝대비/CI exact 동등성을 별도 bridge로 검산 |
| 갤러리 첫 렌더 | 여백과 겹치는 라벨이 많음 | 첫 레이아웃을 private에 보존하고 동일 저장 좌표로 최종8사례 재배치. 새 NN·PnP·fit0 |
| 보고서 직접 생성 | 기존 REPRODUCE의 독점 생성 보호 장치가 작동 | 쓰기 전 중단, private의 새 출력 폴더로 생성 후 신규 REPORT_KO만 복사. 기존 재현 문서 보존 |
| 보고서 링크 검사 | score seal 링크가 실제 metrics 위치와 다름 | 링크만 정정해 별도 report attempt에 재생성. 수치·분석 결과 변경0 |
| 공개본 점검 | 테스트 영수증에 로컬 Python 절대경로 포함 | private 원본 보존, 공개본은 TASK_PYTHON 변수로 치환한 파생본임을 표시하고 parent SHA 연결 |
| 소스 공개 경로 점검 | frozen-cache test의 기본값에 개인 절대경로 포함 | 사용자 home 기준 기본값과 환경변수 override로 이식성을 확보. 최종 전체72검사 통과, 이전71검사와 보고서8검사 영수증은 보존 |

파일럿은 참조 없이 고정 CAL6을 사용했다. 각 경로의 동일 결과를 전체 실행에 재사용한다. 시행 중 정밀 계측이 없었던 초기 추출 실패의 CPU 시간을 발명하지 않는다. 최종 비용에는 계측 범위와 전체 작업 wall을 함께 명시한다. 이번 작업의 새 수치 적합은 승인된 네 pose 진단과 공유 로깅 범위만 수행한다.
