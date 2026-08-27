# 무어노 근접 프레임 attention / belief — PURPOSE

상위 폴더 `../PURPOSE.md` 의 연장. 대상만 다르다: **GT 가 없는 근접 세션**.

**[소비처]** 방향결정 — 정본 평가(161장)에서 **빠져 있는 근접 레짐**에서 최종
모델이 무엇을 근거로 답하는지 확인한다. 이 레짐은 `paper_strategy_master.md:132`
가 "근접 캘리브성 촬영, 덱 꽉참/코너 잘림 → 분포 이질"이라며 명시적으로 드롭한
구간이고, memory `challenge-ft-negative-fp-suppression` 의 미해결 항목("잘린/원거리
팔레트가 죽는다")과 같은 축이다. 부수적으로 논문 정성 그림 후보.

**[문장]** "근접에서 belief 가 무너질 때 line branch 는 여전히 팔레트를 보고
있는가" — 두 경로가 같은 코너를 가리키면 서로 확증하는 것이고, 갈라지면 최소한
하나는 틀린 것이다. 화면 점유가 커질수록 이 불일치가 커지는지가 판정 지표다.

## 학습 아님 · GT 없음

추론 + 시각화만 한다. 읽는 체크포인트는 상위 PURPOSE 와 동일.
소스: `data/pallet/raw_data/outside/capturepalletcad/rgb` 1,179장, **JSON 0개**.

GT 가 없으므로 **계산하지 않는 것**을 먼저 못 박는다 — theta/rho 오차, 코너
오차, on-edge mass, direct 대 CIGM 우열. 전부 GT 를 기준으로 정의된 값이다.
여기서 나오는 수치는 **자기일관성 하나뿐**이다: belief 최대점과 그 코너에 붙은
엣지 3개의 교점(CIGM)이 몇 px 어긋나는가.

## 판정 지표

```
선정      "가까이" 는 사람이 고르지 않는다.  검출 상자 대각 / 이미지 대각 상위.
          단 score_4kp >= 0.3 과 2점 이상 검출로 먼저 거른다(빈 프레임 배제).
관측      corner-line 불일치 중앙값(px), 검출 수 분포, corr(화면점유, 불일치)
```

## 읽을 때의 단서

전처리는 정본과 같은 plain squash 다. memory `dope-inference-needs-reflect-padding`
이 "plain squash 는 근접·truncation 에서 체계적 과소검출"이라고 기록해 두었으므로,
검출이 낮게 나오면 모델 실패가 아니라 **전처리 탓일 수 있다**. 이 그림 하나로
"근접에서 모델이 실패한다"고 결론내지 말 것.
