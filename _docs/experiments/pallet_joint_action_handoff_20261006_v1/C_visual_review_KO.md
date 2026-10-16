# 최종 원고 PDF 화면 검토

[확인] 실제 A 6 fit·최종 집계·조용한 전체 경계 runtime을 삽입한 main15쪽/supplement14쪽의 모든29쪽을95dpi로 raster했어. 이전에 확인한23쪽(main1–12/supplement1–11)은 raster SHA가 정확히 같아 검토를 재사용했고, 달라진 main13–15/supplement12–14쪽은 다시 봤어. main13쪽과 supplement12·13·14쪽은180dpi로 추가 확인했어. 현재 PDF SHA와 모든 raster/font/bounds 검산은 C_VISUAL_REVIEW.json에 있어.

[확인] 새 보충 표 SXVII의6seed·0–1PCK·311/2445/2499/319분모, SXVIII의 정준 GT 코너 손상과 paired평균T/R구간, SXIX의NoOp·생성/최종W-D·oracle gap(m), SXX의 별도 A 비용(ms)과130/130가용성을 확인했어. 표 열·숫자·caption이 잘리거나 옆 열을 덮는 material issue는 찾지 못했어. supplement12쪽의J평균 점수와중심회전 식도 페이지 안에 들어가. main13쪽의 추가 부정 결과·비용 문장과결론, main14–15쪽의45개 참고문헌도 정상 표시돼.

[확인] 사용 글꼴 main59개/supplement24개가 모두embedded·Unicode mapping이 있고 pdftotext-bbox의 페이지 밖 단어는 모두0개야. 마지막TeXlog의 미해결 인용/참조·Missing character도0개야. 기존 IEEEclass의output단계Overfull hbox/vbox 경고는 남아 있으나 실제 표·수식 절단이나 본문 겹침은 나타나지 않았어. 경고 수는 영수증에 남겼어.

[확인] 보충6쪽의 기존float여백과7–10쪽의전체12장패널을 보존했어. 새14쪽에도gap/runtime표 주변여백이 있어. 작은 기존 plot 축 글자는PDF확대가 필요해. 이 한국어15쪽검토 원고는 저자·소속/영문/통상8쪽의 최종 투고 요건 완료를 주장하지 않아. raster/font dump는 저장소 밖 /tmp/pallet-final-paper-visual에만 두었어.
