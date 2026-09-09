# PURPOSE — backbone_compare_v1

[소비처] 논문 §experiments 의 백본 비교표. "왜 YOLO26 인가" 에 답하는 유일한 통제 비교.
[문장]   "같은 학습 데이터·같은 60 epoch·같은 seed 로 학습한 DOPE 와 YOLO26n 을
          같은 PAPER_EVAL 319 · 같은 evaluator 로 나란히 놓으면 백본 차이만 남는다."

`backbone_dope_final_v1` 은 학습만 끝내고 `pose_metrics: NOT_MEASURED` 로 남아 있었다
(착수 시점에 평가셋 미완성). 지금은 평가셋이 있으므로 **추론 + 기존 evaluator** 만으로
그 칸을 채운다.  새 학습 0 · 새 synthetic 0 · solver 변경 0.
