# V221 root 보고서 QA

**PASS.** 여섯 모델의 D1147/E90/N1237 혼동행렬·재현율·E 정밀도·보류율이 CSV/STATUS 및 기존 B V220 수치 감사와 일치한다. D 정밀도도 같은 CM에서 직접 계산해 대조했다.

| 비교 | D 정답 변화 | E 정답 변화 | D 재현율 변화 | E 재현율 변화 |
|---|---:|---:|---:|---:|
| C55→C65 | +70 | −23 | +6.10%p | −25.56%p |
| C63→C65 | +58 | −5 | +5.06%p | −5.56%p |

C55 총정답868→915, C65의 E→D76/D→E246, reference≤3px D461 중 E93도 기존 완료 감사와 맞다. 이는 이미 공개한 집계의 산술 검산이며 새 실사 평가가 아니다.

그림을 `tools.view_image`로 실제1회 열었다. 여섯 모델명과 D/E 막대의 분자·분모·백분율, legacy U177/1237=14.31% 및 반복 진단·새 fit/추론/선택0 주석이 정확하고 읽힌다. 원 실사 crop을 새로 판독하지 않았다.

V95 actual2000/cue-set control과 V163 390query·88frame·ROI22/지원17·새native−1·fit0은 A 완료 감사에 맞다. DINO/context/query 및 C14 24쌍 BCE+0.25hinge의 중복은 본인 코드 감사와 맞다. SOURCE 권한 차이와 실사 집계를 구분하며, GT 유래 q8와 원 initializer q8·weak region과 primary native gold를 섞지 않는다. public-real/GT task fit은 현재 허용이고 첨부의 구 synth-only 전제는 명령으로 취급하지 않는다. 새 학습·model forward·PnP·adoption0/goal미해결 및 보호 평가를 선택에 쓰지 않는 경계가 명시됐다.

최종 MD의 상대 링크8개가 일반파일로 존재하며 symlink를 따르지 않았다. C9분류는 root 본문4+2+3과 nativeDE−1 설명의 정합만 확인했다. 행별 C 분류·ROI·validity는 이 제한된 QA에서 재집계하지 않았고 별도 authority/addendum에 귀속한다. 구 원문 ‘근거부족5’가 root 결론의 수량·admission으로 쓰이지 않는다.

입력은 root 보고서·상태·CSV·생성 chart 및 기존 완료 A/B 감사 텍스트다. private 원장·원 GT·배열·weights·원영상·oldVAL/HOLD 본문은 열지 않았다. browser/HTML render 검사를 했다는 주장도 없다. 자체 checker의 링크수9 가정 실패는 tool trace에 보존한 뒤 실제8로만 고쳤으며, 과학값·보고서 변경은 없었다. 정확 최종 SHA·계산·실제 view·한계는 [QA.json](QA.json)에 있다.
