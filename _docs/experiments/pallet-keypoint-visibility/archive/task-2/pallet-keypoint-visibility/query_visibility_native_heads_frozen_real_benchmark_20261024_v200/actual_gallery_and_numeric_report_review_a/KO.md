# V200 fixed68 gallery 실제 확인

**PASS.** A는 이 gallery 코드의 저자이며 독립 코드 검토를 주장하지 않는다. 수치 원본은 별도 C의 독립 저장결과 감사 PASS와 연결했다. 원 RGB cache·NPZ header/body·가중치·모델은 열지 않았다.

원 manifest 순서에서 root가 본 0·2·256·300·840·996을 제외한 첫6개 **1·3·4·5·7·8** 카드를 tools.view_image로 실제 보았다. 10개 모델 행과 q/L/native·human·reference·validity·footer는 읽을 수 있었고 겹침/잘림을 발견하지 못했다. 여섯 카드 모두 저장 human D, C58/C59 E 응답이다. 모두 앞부분 WOOD 사례이므로 전체 성능/재질/실패 원인의 대표 표본이 아니다.

- **original 1**: 목재 구조의 모서리와 회색 바닥이 함께 보이며 q cross는 그 경계 부근이다. 국소 crop만으로 선택 instance 전체나 native 끝의 의미를 새로 인증할 수 없다.
- **original 3**: 목재 블록/판재 가장자리와 회색 바닥이 보이고 q는 아래 경계 부근이다. 보이는 구조를 실제 외부가림 원인이나 정확한 끝점 gold로 바꾸지 않는다.
- **original 4**: 오른쪽 아래 목재 판재 가장자리와 넓은 바닥 영역이 보인다. q 주변의 국소 외형만으로 native4 부위 정체성/가림 원인을 확정할 수 없다.
- **original 5**: 목재 판재와 판 사이 어두운 틈이 함께 보인다. 기존 reference 오차1.81743px는 이 caption의 관측 연관이며 물리 좌표나 가림 원인 인증이 아니다.
- **original 7**: 세로로 보이는 여러 목재 판재 가장자리와 회색 바닥이 보인다. reference와의19.1141px 차이는 기존 proxy 수치이고 정확 위치 오류/가림 원인으로 확정하지 않는다.
- **original 8**: 목재 상판/블록 면과 바닥이 함께 보인다. 어두운 경계/틈을 다른 물체의 앞가림이라고 단정하지 않는다.

68개 ID·순서·기존 cell membership은 V195와 같았다. 각 ID의 clean96와 valid96를 V195 PNG와 실제 decode해 비교해 전부 픽셀 byte 일치를 확인했다. 이 계산은 전체68카드를 의미 판독했다는 뜻이 아니다. 원 q/L 보존은 C 감사와 봉인 영수증에 의존하며 원 cache를 다시 읽어 재검산하지 않았다.

저장 current metric에서 30개 primary 행의 CM/P/R/U를 별도 scalar로 집계해 모두 일치했고, material/native16행·public weak9행·SELF40행도 일치했다. primary는1237(D1147/E90), full1784 원 inputU45이며 SELF paired21과 full same-slot136을 분리했다. Public360의 known center204/UNKNOWN156·inputU108와 legacy 없음도 유지했다. MD176개/HTML141개 상대 링크와 모든 표 셀을 확인했다. 브라우저 렌더는 하지 않았다.

국소 목재 외형·어두운 틈·q 근접성을 실제 외부가림/정확한 native 끝/좌표 gold로 해석하지 않는다. 기존 reference≤3px와 manualnear376은 다른 집단이다. 새 라벨·문턱·학습자료·모델 선택은 없고 미채택·전체 목표 미해결을 유지한다.

[전체 고정 gallery](../fixed_native_real_visual_v1/PORTABLE_GALLERY_KO.md) · [저장 수치 독립 감사](../independent_saved_result_review_c/actual_review_v1/REVIEW.json)
