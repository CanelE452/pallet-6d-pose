# Guarded A/B 이전의 14손상 분석 기록

이 묶음은 초기 최소 B/I 결과에서 발생한 14손상과 예측 기반 특징을 분석한 당시 기록이다. FINDINGS의 후속 제안은 이후 guarded A/B 실험 이전의 제안이며, 현재 채택 권고는 [최종 guarded 권고](../../pallet_guarded_bi_final_v1/FINAL_RECOMMENDATION_KO.md)를 따른다. 당시 실패·결과·분석 원본은 바꾸지 않았다.

[FINDINGS_KO.md](FINDINGS_KO.md), 원본 특징·score join JSON, [FEATURE_SEAL.json](FEATURE_SEAL.json)을 함께 보존했다. `.json.gz`를 같은 이름의 `.json`으로 압축 해제하면 원본 바이트와 SHA256이 복원된다. 원본 해시와 저장 파일 해시는 이번 공개 manifest에 별도로 기록했다.

[원작성 코드](archive/audit.py.txt)는 원본 바이트 그대로 보관한 역사 자료이며 현재 재현 CLI가 아니다. 고정된 로컬 경로와 기존 출력 쓰기 방식을 포함하므로 이번 게시에서 실행하거나 변경하지 않았다. 현재 최소 B/I 재현은 [기존 재현 문서](../REPRODUCE.md)를 이용한다. 새 학습·추론·PnP·정답 수정은 없다.
