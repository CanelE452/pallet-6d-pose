# V241 검토문과 V251–V253 후속 기록의 차이 — V254 초안

첨부는 V241 시점의 검토 자료이며 실행 승인이 아니다. 지정한 6·8·9·11·12절의 SOURCE 진단 제안만 읽었고, 문서의 현재·held 평가 집계는 제외했다. 최신 최소 상태는 main `49256c1c…`, 목표 미해결이다. 아래 판단에 새 숫자 payload나 모형 실행은 없다.

| 항목 | 이미 실행·확인한 범위 | 아직 남은 것 / V254의 질문 |
|---|---|---|
| 분기별 기여 | V251 frozen core+reference residual 대조와 V252 direct-reference 대조는 두 팔 학습·freeze까지 완료됐다. | V251은 V240 자체 base/relational 분해와 동일한 진단이 아니다. CONTROL 차이를 TREAT 내부 residual로 부르지 않는다. 이 초안에서 posthoc 수치를 재검산하지 않았다. |
| 생성 쌍 행동 | 문서의 oldV235 쌍별 행동 공백은 이후 V244 replay·실제 draw 노출 join으로 다뤄졌다(Root 봉인 기록 재인용). | 학습에 뽑힌 쌍과 augmentation pool은 다르다. 생성쌍 반응은 자연·배포 전이의 증명이 아니다. |
| coherent scene CF | V241 순수 준비 뒤 실제 scene composition·proof·고정 SOURCE trial까지 진행됐다. V245 완료 기록이 있으므로 미실행 준비처럼 되돌려 쓰지 않는다. | 이번에는 이미 존재하는 V129 fullscene variant와 저장 V252 점수의 연결을 검산한다. 새 CF 생성·encoder·학습은 이번 진단의 목적이 아니다. |
| 목표 의미 | V253에서 SYN point0는 선택된 native record의 첫 witness임을 CODE로 확인했다. 전체 pallet ray에서 foreign BOX 앞hit만 E, 자기 기하 앞hit는 S다. | witness reference와 q-pixel owner, wholepart, 실사 native-instance는 별개다. YCB associated-r AUX도 분리한다. |
| 입력·실패 해석 | q8·다른 node·문맥이 없다는 설명은 기존 구현과 맞지 않는다. reference 계열도 이미 시험했다. | 형태·재질·주소 편중이 실패 원인이라는 결론, target 식별불가능성, 배포 성공 모두 아직 확정할 수 없다. |

V254는 네 variant `BEFORE/COVER/NEAR_SHAM/FAR_SHAM`과 `SELF nominal/BEFORE-anchor`를 별도 층으로 유지한다. 전체 group을 먼저 완성한 다음 원 SYN의 D/E/S/U와 masked 행을 보존하고 저장 점수를 붙인다. 같은 base나 상속 label만으로 같은 질문이라고 보지 않는다. 원 q/L/native/U, q8/L8/availability8·box·condition이 정확히 같은지 확인하고, 아니면 unmatched로 남긴다. 두 SHAM 중 잘 나온 쪽을 고르지 않는다.

첨부의 핵심 경계는 여전히 유효하다. 준비·toy, 실제 데이터 proof, 실제 fit, SOURCE admission, 배포 성공을 구분해야 한다. q 근처 표면만으로 E 라벨 오류를 주장하거나, point0를 native0로 바꾸거나, SOURCE DEV를 보고 문턱·epoch·seed를 고르지 않는다. 이번 초안은 새 recipe나 모델 선택을 하지 않는다.

근거의 정확한 binding과 직접 읽은 metadata/Root 기존 receipt 재인용 구분은 [RECONCILIATION.json](RECONCILIATION.json)에 기록했다.
