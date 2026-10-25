# V225 고정15 source panel 실제 관찰

고정 manifest SHA `e84947db6bca86645d23bdb640ec780da220a9396e6290b4b9b7f60881624d2e`의15 PNG를 `view_image(detail=original)`로 모두 실제 열었다. E대표15개와 같은 frame D14개, 총29질문을 원순서대로 관찰했다. 원frame은 202, 153, 204, 149, 155, 220, 209, 207, 205, 206, 125, 126, 217, 134, 133다. header·원q/L·native8 crop·fullcontext가 읽혔고 다른 사진/GT overlay는 열지 않았다.

**문맥의 가림 개연성과 crop 단독 판별 가능성이 다르다.** E13개는 선언 역할을 전제로 전체 pallet/적재 관계에서 외부cover가 개연적이나 native 대응은 약하다. frame155/E4와217/E4는 그 대응 자체를 UNKNOWN으로 남겼다. E crop은7개 ambiguous,8개 target-specific 근거 absent다. absent는 inputU나 semantic E를 새로 부여한 것이 아니다. 기존 E 라벨의 무효화·재인증도 아니다.

D14개는 fullcontext에서 외곽 노출 역할이 개연적이었다.13개는 crop의 deck 끝/블록 면/바닥 경계로 coarse visibility 근거가 있고,217/D0은 작은 흐린 목재 조각만 보여 crop 단독은 ambiguous다. 여기서 strong은 coarse 역할 연결이며 물리 native 좌표·3px 정답 인증이 아니다.

- frame149/D1: 회색 통 일부가 crop에 들어와도 앞쪽 목재 외곽 끝은 노출된다. foreign presence만으로 E라 할 수 없다.
- frame125/E5·134/E5: wood와 회색 통/다른 물체가 함께 보인다. wood presence만으로 해당 hidden role을 D라 할 수 없다.
- frame155/E4: 파란 KLT가 crop을 채우고 팔레트 외곽 일부가 원화면 밖으로 이어져, 특정 hidden role 대응은 패널만으로 닫기 어렵다.
- frame217: E4 crop은 흰 포장면, D0 crop은 흐린 목재 조각이다. fullcontext 없이는 두 질문의 대상부위 주소와 원query 위치 관계를 충분히 알 수 없다.

이것은 기존 약판독을 알고 수행한 assistant의 정성 시각점검이다. Root의 새 V225 관찰은 읽지 않았으나 과거 V224 텍스트와 기존 weak 설명에는 노출됐다. 독립 사람 gold/블라인드 합의·44E전수검증·물리장면독립성·일반화율·모델성능 결과로 확대하지 않는다. exact native에서 SELF인지 외부인지 닫히지 않은 경우 E를 확정하지 않는다. 원라벨·query/crop·admission을 바꾸지 않았고 모델/학습/score/protected 접근은0이다.

[패널별 실제 관찰과 원 PNG bindings](OBSERVATIONS.json)
