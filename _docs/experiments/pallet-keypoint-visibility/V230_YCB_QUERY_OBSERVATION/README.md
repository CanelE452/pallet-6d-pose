# V230 공개 YCB q/r 관측 진단

[보고서와 분모별 표](PORTABLE_REPORT_KO.md) · [원 crop6개](figures/PORTABLE_GALLERY_KO.html) · [기존 YCB 실제 loss 경로](CONSUMER_LOSS_AUTHORITY.json)

원8000 teacher D/E/S/U를 보존하고 추정 crop 중심 q의 관측을 별도로 집계했습니다. q-owner는 새 native 가림 라벨이 아닙니다. 새 학습·현재 평가셋 사용·PnP·성능 개선은 없습니다.

패키지 루트에서 `python -B VERIFY_SAVED_QUERY_OBSERVATIONS.py`를 실행하면 표준 Python으로 압축 원문SHA,8000행과표/분모/좌표·소유자 산술을 재검산합니다. 원 depth/mask·GT 권위·모델 성능을 독립 재인증하지 않습니다. 이 실행은 V229 전에 고정된 V228 계획을 재개한 기록이며 기존 PREP_ONLY 게시물과 실패 결과는 그대로 보존했습니다.
