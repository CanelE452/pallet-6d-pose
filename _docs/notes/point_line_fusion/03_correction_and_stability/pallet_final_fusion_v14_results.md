# V14 완료: 최종 점·선 구조와 기존 방법 비교

요청한 최종 후보를 통합 구현하고 동일 평가 데이터에서 비교했다. 결론은 `IMPLEMENTED_COMPARED_NOT_ACCEPTED`: 통합 구현·검산은 통과했지만 안정적 추가 개선 기준을 통과하지 못해 배포를 교체하지 않았다.

설계: [상세 아키텍처](pallet_final_fusion_v14_architecture.md).
코드: `scripts/research/pallet_final_fusion_v14/`.
결과: `data/pallet/results/pallet_final_fusion_v14/`의 `index.html`, `REPORT_KO.md`, `SUMMARY.json`, `AUDIT.json`, `FAILURE_DIAGNOSIS.json`, `INTEGRATION_CHECK.json`.

## 실행

- 고정 YOLO/V10/V12/V11 부모 +5후보 평가기. V14 evaluator46,425파라미터, 전체 YOLO 이후473,804파라미터.
- insample_fusion / heldout_points / heldout_fusion 각3시드·2000업데이트, 총18,000업데이트. 실제 학습0장, 평가 기반 체크포인트 선택0회.
- 기존 부모 학습 이미지253장(1464검출 관측)과 학습에 포함되지 않은 former calibration253장(1462관측), 각6조건. 후자는 이번부터 학습 데이터. 부모 refinement train/calibration/synth_val ID 및 이미지 SHA 비중복. YOLO 사전학습 미노출까지 보장하지 않는다.
- 평가: 재사용 real297/9sessions/manual1251points, synth_val505,6조건. 같은 원래GT ID·검출 실패를 공유.

## 실제 주요 결과

3개 시드 평균px. clean은 프레임 평균, 나머지는 지정 대상 점 평균.

| 대상 | YOLO | V10 local | V11 joint | V12 gate | V12 amodal | V13 scorer | V14 points | V14 fusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean |5.816782|5.534459|5.477749|5.537562|5.542445|5.539674|5.552705|5.527641|
| corner20 hidden280 |16.446099|17.039815|17.045310|16.539330|13.554489|13.637115|13.748346|13.767414|
| crop20 outside558 |12.335344|12.530962|12.661467|12.372119|16.496415|16.748662|16.238791|16.258101|
| crop40 outside571 |27.771420|27.836821|27.757752|27.770936|19.921341|20.222629|20.336811|20.526354|

세 시드 모두 clean<=1% 기준 및 corner20 복원 이득80% 유지 통과. 세 시드 모두 crop20<=1% 기준 실패. seed3 crop40 복원 이득80% 유지 실패. 주요4조건 모두 Hough 추가 이득(각시드>=1%와 CI 상한<0) 기준 실패. V14 원본 평균은 점 대조군보다0.025px 좋으나 한 시드는 악화하고1% 기준 미달.

이미지 분리만으로 문제가 해결되지 않았다. crop20 실제 전체 복원 이득은 -4.554..-3.613px인데 V14는 +0.853..+1.123px로 예상했다. 기존 학습 이미지 융합 대비 분리 융합의 crop20 변화 -0.205px, 세션95%CI -0.595..+0.188로 불확실하고 원본/심한 잘림 평균은 악화했다.

선 후보의 보정량 자체도 작았다. corner20/crop20/crop40 대상 점에서 평균 선 이동량0.140/0.274/0.328px. 점 후보 GT oracle에 선 후보를 추가해도 추가 이득0.0168/0.0166/0.0275px뿐이다. 이는 고정된 이번 후보 집합의 특권 상한으로, DHT 원리적 불가능성이나 실제 도달 성능의 증명이 아니다. 선 입력이 평가기의 amodal 선택을 돕는 가능성은 이 후보 상한 하나로 배제하지 않는다.

## 검증·산출물

- 독립 audit178,044예측 프레임 행,69,600선택 프레임 행. 후보 재구성 최대6.1035e-5px, 지표 재계산 최대7.1054e-15. 기존 부모 해시·학습 소스 해시·18000유한 업데이트·수동GT 해시/원본좌표이동·합성GT이동 확인.
- `FinalPointLineNetwork`: 각3시드×4조건×8프레임 통합/캐시 출력 및argmax 일치, 최대3.0518e-5px.
- 통합 experimental checkpoint `export/final_seed{1,2,3}.pt`. YOLO 이후 batch1 전체 경로8.05~8.78ms(10예열+30동기화측정), YOLO/로딩/전송 제외. 전체FPS나 비교모델 속도우위 주장 없음.
- HTML: 상세 구조도, 모든 방법/도메인/시드/지표 선택표,12개 예시에서 실제 DHT 연관 가중치·후보 위치·예상/실제 개선량 시각화. 예시·점 선택은 사후 GT 설명용.
- 자연 가림unknown, 재사용DEV 인공스트레스. 수동GT 재클릭 없음, 최종6D/이전41.49px와 다른 지표. 기존 배포 수정 없음.

재현 순서(새 결과 디렉터리 사용 필요, 기존 학습 덮어쓰기 거부): `prepare14.py --phase extract` → `prepare14.py --phase parents` → `train14.py` → `eval14.py --phase inference` → `eval14.py --phase score` → `network14.py` → `audit14.py` → `diagnose14.py` → `report14.py`.
Python: `/home/minjae/anaconda3/envs/pallet-yolo26/bin/python`. 추출/부모/학습/추론/통합 확인은 CUDA 필요.

HTML 실제 창 열기 및 완료-only Discord 전송은 결과 폴더의 delivery receipts에 기록한다. 새로운 모델 선택/임계값 튜닝/공동 학습으로 자동 확장하지 않았다.
