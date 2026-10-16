# C — 실제 원고·전체 참고문헌 정리

[확인] 최신 remaining_evidence_connection 원고를 새 paper_updated/에 복사해 Abstract, Introduction, Related Work, Method, Results, Discussion/Limitations, Conclusion, 비교·절제·손상 caption을 실제 수정했어. 기존 N3 evidence 42개 파일의 SHA256은 원본과 같아. 코드의 기본 N3 동작이나 원결과를 바꾸지 않았어. 현재 검증과 빌드 상태는 C_status.json, source/결과 해시는 C_manifest.json과 paper_updated/audit/HANDOFF_BUILD.json이 기준이야.

[확인] 주장 범위를 등록 치수·초기 코너가 있는 국소 정제로 고정했어. Base→N3 전체 이득, N0→N2 약 −0.1662px, N2→N3 약 +0.0004884px를 구분했고 대칭을 주된 원인으로 쓰지 않았어. 정준 xyz=[W,H,D]의 수직 Y축에서 C2는 180도, C4는 90도 간격의 전체 회전이고, W=D만으로 물리·외관·포크 작업 대칭을 승인하지 않았어. center8·객체 선택·박스·점수·마스크 보존과 검출 실패 복구의 부재를 명시했어.

[확인] 319장/13세션은 재사용 DEV(개발자료)로 유지했고 정사각형 119장/한 세션·602/600코너와 이전 128/194장 패널을 합산하지 않았어. 같은 코너 레이블의 Perspective-n-Point(PnP) 재구성 자세 참조를 독립 물리 계측으로 쓰지 않아. 64/2,445 cap 도달과 이미 완료된 cap 탐색, 2D 개선 뒤 이동 악화 98/78/100장(2D 매칭 공동 분석311장 분모; 전체 YOLO Base/N3 자세 산출·위치/회전 통계는319/319장), 큰 오류 복구와 좋은 점 손상을 함께 넣었어.

[확인] Base/N3/PoseFix 전체 방법 비교는 입력·손실·감독의 차이를 명시했어. PoseFix 방식의 더 낮은 중앙값과 N3의 더 낮은 P90을 모두 보존했어. 새 같은 경계의 벽시계 시간 중앙값은 11.056/14.470/25.239ms, P90은 13.081/17.170/27.311ms야. 각130회 측정에서 자세 산출130/130이었고 GPU(그래픽 처리 장치)는 같은 NVIDIA RTX3080·pallet-yolo26 환경이야. 13세션/26프레임·arm별 준비20회·5반복이며 모든 전처리·검출기·shared feature·보정·최종 prediction-only PnP를 포함하고 decoding/loading은 제외했어. N3/PoseFix 정확도 표는 세 seed 통계 평균, 비용은 사전 고정 seed1이므로 같은 통계량으로 섞지 않았어. 원본 수치 계약이 달랐던 첫450호출은 폐기했고 검산된450호출만 재사용했어. Base/N3 좌표차0px, PoseFix≤0.000122px(허용0.001px)와 source/model 해시를 RUNTIME_VERIFICATION.json에 연결했어. 기존 세 기반 CUDA event 표는 보충자료의 다른 환경 패널로 보존했어.

[확인] bibliography 실제 초기42개를 전부 파싱했고 현재46개(main45/supplement1)의 모든 실제 citation 위치·문장을 REFERENCES_AUDIT.tsv 새 스키마로 기록했어. 기존42개는 제목·판본·저자순서·연도·주장 범위를 동일 primary lock과 대조해 재사용했고 그중16개의 출판사 DOI deposit을 추가 확인했어. 현재 metadata 확인46개, 기존 수정15개(서지 표현/page/full name12개 + BB8/Simple Baselines/Knitt 실제 문장3개), 문헌 항목 자체 미확인0개야. 새4개는 ResNet 몸체, 선택 편향, 측정 불확실성, PERM 원리의 실제 문장을 뒷받침하므로 추가했어. 초록/서지만 확인한 기존 항목을 전문 구현 검증으로 승격하지 않았고 확인하지 않은 DOI·쪽수는 빈칸이야. IEEE 숫자형 최초 등장 순서, 미사용/undefined/중복 key도 검산해.

[확인] BB8의 optional 2D corner projection refinement를 명시했고 Simple Baselines를 팔레트 head adaptation 근거로 유지했어. ResNet 몸체 원전과 구분했어. Knitt의 optical MoCap 실제 센서 평가와 현재 독립성 제한을 본문에 넣었어. 필수 1차 전문을 직접 읽은 section/page와 원문 SHA256은 PRIMARY_SOURCE_READ_NOTES_KO.md/PRIMARY_SOURCE_BINDINGS.json에 있어. 저작권 PDF는 저장소 밖 /tmp에만 뒀어.

[확인] 같은 상대 자세의 clean/실물 가림과 독립 참조가 연결된 실제 pair는0개야. B 취득 패킷 구현 완료와 독립 평가 BLOCKED_DATA를 구분했고, 9,029개 관측의 독립 물리 참조0개, CAN 중립 명령32구간/8,263프레임의 실제 상대 정지 미확인, 원래12장/96점과 미제출72점 버전을 그대로 제한했어. A 새 실험은 기존 N3를 덮어쓰지 않는 탐색적 보충 섹션으로 실제 완료한 최종 수치·NoOp/F/oracle/I/J/PERM 한계·손상·비용을 넣었어. A_PAPER_INTEGRATION이 원 snippet과 편집 변환, 최종 A_summary·대조·oracle gap·runtime 및 삽입 파일 SHA를 묶어.

[확인] 격리 GNU Tectonic0.15.0을 /tmp/pallet-paper-tools/gnu15/tectonic에 설치해 원래 LaTeX 도형3개를 실제 PDF로 만들었어. Tectonic0.17 musl의 Korean ICU 오류와 최신GNU의 host glibc 불일치는 호환 GNU0.15로 해결했어. 폰트 smallcaps 호환과 추가 영수증 텍스트의 underscore를 수정해 본문/보충자료를 컴파일했어. 현재 페이지 수·PDF해시·최종 undefined/glyph 상태는 HANDOFF_BUILD.json에 기록해. 원고의 새 빌드 영수증은 과거 source_previous_snapshot 영수증과 구분해.

[확인] 2026-10-06 공식 IEEE Sensors Journal 저자 안내의 double-column·graphical abstract와 보통8페이지, February2025 심사 지침의 novelty/appropriateness를 확인했어. 기존 graphical_abstract.tex를 사용해. 현재 한국어 검토 원고는 저자·소속 미확정이며 8페이지보다 길어서 완성된 투고/채택으로 표시하지 않아. 남은 편집과 영문 제출은 과학적 독립 평가 완료와도 별개야.

[추정] 지금까지 원고에서 허용되는 결론은 주어진 초기 출력·등록 치수·재사용 개발참조에서 국소 중심부 개선과 꼬리/손상/비용의 손익이 있다는 거야. 치수·대칭 약효를 근거로 모델 용량 확대를 확정하지 않아. 독립 계측은 추가 모델 변경으로 대체할 수 없어.

재개 명령은 실제 --help로 확인했어.

```bash
python3 -m unittest scripts.research.pallet_joint_action_handoff_20261006_v1.test_paper
python3 scripts/research/pallet_joint_action_handoff_20261006_v1/paper.py validate
python3 scripts/research/pallet_joint_action_handoff_20261006_v1/paper.py build --engine /tmp/pallet-paper-tools/gnu15/tectonic
python3 scripts/research/pallet_joint_action_handoff_20261006_v1/paper.py finalize
```

실제로 완료된 A snippet을 아래처럼 삽입했어. 후속 A 결과를 바꿀 때에는 연결된 분석·비용 표도 함께 다시 검토한 뒤 빌드·finalize해야 해.

```bash
python3 scripts/research/pallet_joint_action_handoff_20261006_v1/paper.py integrate-a --a-snippet _docs/experiments/pallet_joint_action_handoff_20261006_v1/A_PAPER_SNIPPET.tex
python3 scripts/research/pallet_joint_action_handoff_20261006_v1/paper.py build --engine /tmp/pallet-paper-tools/gnu15/tectonic
python3 scripts/research/pallet_joint_action_handoff_20261006_v1/paper.py finalize
```

[확인] C 빌드 검사 코드는 실제 input/include에서 사용하는 .cls/.sty/.bst와 그림 PNG/PDF·logo EPS/PDF 및 standalone 그림 TeX까지 연결해. 그림 생성 전에는 생성 입력을, 생성 후부터 main/supplement 빌드 완료까지는 새 그림 PDF도 hash로 고정해. 실제 사용 스타일·그림이 바뀌면 예전 빌드를 current로 재사용하지 않아. aux/out/bbl/blg/log 등의 로컬 TeX 임시물은 C_manifest의 배포 변경 파일에서 제외하고 delivery_required=false인 진단 목록으로 분리해. 이 변경과 기존 검증의 fixture 검사 5개가 통과했고 최종 A 삽입 뒤 클래스·스타일·그림까지 포함한 81개 계약 입력으로 실제 다시 빌드했어. 생성자산의 합법적 PDF SHA 갱신은 허용하되 생성 입력이나 main/supplement 빌드 중의 변경은 0개였어.

[확인] A 최종 추가 학습은 GEO/PERM 각 세 seed의 6 fit·36,000 update·576,000 노출이며 원 N3 가중치와 headline을 보존했어. 실사 GEO 코너 중앙값은 seed별6.429/6.461/6.491px, PERM6.627/6.766/6.564px야. GEO−N3/PoseFix 정규화 오차 차이와 세션 구간은 악화 방향이고, RAW/PERM 대비 음의 구간도 좋은 코너 손상으로 고정 보존 조건 미충족이야. 손상은 frame이 아닌 같은 정준 GT 코너 개수로 RAW6/6/5, N3 17/15/19, PoseFix28/26/26, PERM5/15/22점을 실제 표에 넣었어. 이동/회전 paired 평균 차이와 중앙값 차이, 311매칭/319pose 분모를 구별했어. 합성 전체1,985장 중 참조 코너 없는7장의 non-evaluable 상태와1,978장 코너 구간도 명시했어. 모든 대조는 탐색이며 다중비교 보정이나 독립 확증을 주장하지 않아.

[확인] 새로운 조용한 A 비용 패널은 Base/N3/동결GEO-J/학습GEO-J/학습PERM-J 중앙값11.080/14.433/56.663/56.541/56.667ms, P90 13.428/16.636/59.235/59.174/59.837ms야. 각130/130pose이고 준비100+측정650=750호출의 원출력 좌표 차이는 모두0px, action/finalW-D/coverage도 정확히 같아. fresh RGB 특징 점수·실제201후보 생성·hardJ·선택후최종F를 포함하며 전체oracle F탐색을 배포 경계에 넣지 않았어. CPU집계와 겹친 앞750호출은 비용에서 전부 폐기했고 조용한 측정만 보고해. 원 Base/N3/PoseFix 패널과 새 패널의 값·호출·메모리 한계를 섞지 않았어.

[확인] 최종 main15쪽/supplement14쪽의81개 입력 계약 빌드가 DONE이며 마지막 인용·참조/glyph 오류0개야. 실제29쪽 raster에서 같은23쪽은 기존 검토를 재사용하고 바뀐6쪽은 재확인했어. 새4개표와J식은180dpi로 확인했으며 표 잘림·폰트 누락·수식/본문 겹침의 material issue0개, 모든59/24글꼴embedded·Unicode/페이지밖단어0개야. 세부 SHA·화면 검토 범위는 C_VISUAL_REVIEW.json에 있어. 참고문헌46개/미확인0개/수정기존15개와 기존N3evidence42개동일SHA를 최종 검산했어.

[확인] measurement_b의 독립 원고 숫자 검산45/45도 PASS야. 6seed수치·손상·paired평균T/R구간·E_sym·NoOp/W-D/oracle gap·5armruntime와모든분모·단위를 실제최종영수증에 대조했고 material issue0개야. 결과는 results/INDEPENDENT_A_PAPER_NUMBER_REVIEW.json이며 새CNN/PnP/optimizer는각0회야.

[확인] root의 정상 unified resume 뒤 실제6 fit를 포함한25방법 subgroup을 저장된 원행만으로 독립 검산했어. canonical2499대응·grade/distance 분모·seed 통계 평균·NoOp147/141/167에 대한77,837 scalar/구조검사가 PASS이고 CSV396/198행을 확인했어. INDEPENDENT_SUBGROUP_REVIEW는 최종 code/output SHA와25방법을 묶어. sorted 최종 runtime verification SHA 및45/45원고QA도 마지막 통합 metadata에 갱신했으며 원고·PDF·성능 숫자는 바꾸지 않았어.
