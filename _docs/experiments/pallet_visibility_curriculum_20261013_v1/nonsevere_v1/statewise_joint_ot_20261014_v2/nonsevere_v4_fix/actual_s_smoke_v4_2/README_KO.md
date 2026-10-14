# V4.2 실제 CAL RGB 초기 S 좌표 교란 검증

실제 CAL 사진 4개에서 현재 `StatewiseIntervalV4_2.refine`를 각 2회 실행했습니다. 실제 Tiny/SAM provider와 고정 λ1·β1·56 P3P·24 global starts·250 joint iterations를 사용했습니다. 이전 세 사진(night, outside, wood432)에 조건부 T 보류 경로를 검증하는 고정 cad304118 사진을 추가했습니다.

초기 S raw 좌표만 약 1e8/−1e9로 바꿔도, 저장된 S raw 메타데이터를 제외한 전체 runtime이 4쌍 모두 정확히 일치했습니다. 실제 Tiny/SAM 마스크·프롬프트·quality/provenance, B/I 관측과 유효 공분산, O 영역, nonS scale, 등록 Q, 9번째 메타데이터, pose와 interval 모델 바인딩이 보존됐습니다. S 지역 이미지 질의·마스크 프롬프트·보정·관측·영역·pose 이미지항은 모두 0입니다.

Cad304118에서는 native1·2의 조건부 T가 두 실행 모두 U로 보류됐습니다. Night의 BI2·O3 수락은 모호한 pose family이며 고유 pose나 정확한 숨은 좌표를 증명하지 않습니다. Wood432는 알려진 native 기준 불일치 CAL stress 사례로, 이 검사는 정확도 주장이 아닙니다.

8회 실행 후 prediction을 봉인하고 쌍별 출력을 감사했습니다. TEST 파일이나 GT 좌표·pose를 추론에 사용하지 않았습니다. 이미 봉인된 CAL 수락 점수/모델을 검증하는 runtime guard는 확인했습니다. 모든 source·model·input·image·보호 파일 SHA는 전후 일치했습니다. 기존 실행 소스와 선택 설정은 변경하지 않았습니다.

이 검사는 고정 초기 기하 S의 제외를 증명합니다. 실제 human SELF를 모두 맞게 분류했다는 증거나 새 환경 일반화 증거는 아닙니다.
