# V33 실행 인계

사용자가 V32 후속 점별 선 구간/끝점 직접 대응 실험을 승인했다.
V32 결과 및 데이터는 읽기 전용으로 유지하며 새 결과 폴더에 저장한다.

- 코드: `scripts/research/pallet_correspondence_v33/`
- 결과: `data/pallet/results/pallet_correspondence_v33/`
- 동일 V32 실제 CAD 가시 경계/끝점 감독, 같은153원본+렌더 관측, 배치/시드/2000업데이트.
- pooled / query / explicit geometry 제거 / line-only / endpoint-only / physical aux-only,6대조×3시드=18모델36,000업데이트 계획.
- 원래 점/강한 복원점 각각 query. 점에 연결되는 cuboid 역할의 Hough 상위4bin/가시 support 중심·범위 및 공간 끝점 셀을 참조한다. Null token으로 근거 부재 표현. 최종 mix+2D잔차는 V32 유지.
- 점 GT, 물리 경계 라벨, 카메라/좌표 정의는 V32와 동일하다. 역할은 CAD 체인과 가상 cuboid 근접 관계로 완전한 인간 의미 라벨이 아니다.
- 주판정: query가 pooled 및 end-only 대비 실사 잘림20에서 모든 시드1% 이상 개선,9세션 차이95%상한<0. V32와 같은 전체 안정성/기존 이득 보존 기준도 동시 통과해야 성공이다.
- 실사297/9세션은 사용자 승인한 재사용DEV 계보. 정본140/논문/봉인 평가와 구분한다. 수동GT 변경/실사 학습/임계값튜닝 없음.
- 단계: precheck→train→모든 기존 분리 평가→기여 개입→GT/수치/파일 감사→MD/그림/HTML 실제 열기→사용자가 요청한 완료 한 문장 Discord 알림.
- 상태/완료 여부는 결과 JSON을 확인한다. 완료 전 성공으로 주장하지 않는다.

## 학습·평가 완료 결과

- 18모델·36,000업데이트 완료. AUDIT.json: 546예측 파일,36회 GPU 재현,수동GT297×8행 좌표/valid/hash,같은 초기값/배치 및 기존 입력/결과 무변경 통과.
- 잘림20 pooled14.714 / query15.004 / no_geometry14.869 / line_only14.090 / end_only15.800 / point_aux15.408px. Query vs end_only 기준은 통과, vs pooled와 전체 안정성은 실패.
- 탐색 line_only vs pooled 시드 개선6.947/2.048/3.556%, 세션95%차이[-.861,-.359]px. 기준점12.372/V30 13.385보다 나쁘므로 안정적 교체 모델이 아니다.
- Query line_off16.624 / end_off14.827 / both_off16.559 / cross_donor14.994. 선 경로 활성과 정확한 선 위치의 효용을 구분한다. Seed1 crop20 donor mapL1 .067, top4bin 겹침13.3%.
- 코너 가림 query13.596 vs strong11.907, crop40 query18.452 vs strong19.382. 좋은 부모의 모든 조건 보존은 미달.
- 전역 대조 반복은 V32와 공통 초기값·첫 손실이 같으나13/32/34step에서 약1e-8 차이가 시작됐다. V32 crop20 14.343 vs V33 pooled14.714. Deterministic algorithms 강제는 꺼져 있었고, 단일 grid_sample의 deterministic backward는 실행됐다. 전체 결정적 반복 검증은 미실행이다.
- 고정 첫 혼합 배치 query 공유trunk gradient cosine 평균+.309. 지속 충돌/PCGrad 효과의 근거로 일반화하지 않는다. PCGrad 미실행.
- 상세 원인/한계: 결과 폴더의 INTERPRETATION.md, POSTHOC_DIAGNOSTICS.json, REPORT_KO.md. 실제 attention과 고정 사례3/극단 개선·악화는 figures 및 CASES.json.
- HTML 실제 창·해시·Discord 상태는 DELIVERY.json / COMPLETION.json / DISCORD_NOTIFICATION.json 참조. 이전 V32 알림은 재전송하지 않는다.

다음 우선순위: 선 단독의 결정적 반복→선 위치 이동/교체 반사실 검사→물리 끝점과 가상 점의 역할/offset 검수 뒤 끝점 하나만 추가. 전역압축만 바꾸면 해결된다는 가설은 이번 예산에서 지지되지 않았다.
