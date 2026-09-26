"""Public aggregate-only progress report; no private RGB or exact coordinates."""
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import common as C

def render():
    if (C.DOC/'DECISION.json').exists():
        from .final_report import render as final_render
        return final_render()
    a=C.read(C.DOC/'CANDIDATE_POOL_AUDIT.json');q=C.read(C.DOC/'DIFFICULTY_QUEUE_LOCK.json');state=C.state()
    figures=C.DOC/'figures';figures.mkdir(parents=True,exist_ok=True)
    recs=sorted(a['eligible_recordings']);fig,ax=plt.subplots(figsize=(10,4))
    bottom=[0]*len(recs)
    for i in (1,2,3):
        vals=[q['round_counts'][str(i)].get(r,0) for r in recs]
        ax.bar(recs,vals,bottom=bottom,label=f'Round {i}');bottom=[a+b for a,b in zip(bottom,vals)]
    ax.set_ylabel('Raw-only review frames');ax.set_title('Locked temporal coverage queue, max48 per recording');ax.legend()
    fig.tight_layout();fig.savefig(figures/'01_tagging_queue_recordings.png',dpi=150);plt.close(fig)
    summarypath=C.DOC/'DIFFICULTY_TAG_SUMMARY_PUBLIC.json'
    summary=C.read(summarypath) if summarypath.exists() else dict(total=0,counts={s:0 for s in C.TAGS},hard_recordings=0)
    fig,ax=plt.subplots(figsize=(10,3.5))
    if summary['total']:
        ax.bar(list(summary['counts']),list(summary['counts'].values()));ax.set_ylabel('Human tagged frames')
    else:
        ax.axis('off');ax.text(.5,.6,'WAITING FOR HUMAN DIFFICULTY TAGS',ha='center',va='center',fontsize=18)
        ax.text(.5,.35,'No human tags yet. Hard prevalence is UNKNOWN, not zero.',ha='center',fontsize=12)
    fig.tight_layout();fig.savefig(figures/'02_human_difficulty_distribution.png',dpi=150);plt.close(fig)
    table='\n'.join(f'|{r}|{a["source_recordings"][r]}|{a["eligible_recordings"][r]}|'+ '|'.join(str(q['round_counts'][str(i)].get(r,0)) for i in (1,2,3))+'|' for r in recs)
    counts={k:sum(v.values()) for k,v in q['round_counts'].items()}
    report=f'''# Minimal hard supervision A/B — 준비 및 사람 입력 대기

## 1. 한 줄 결론

**{state['status']}**. 현재는 Phase1–2 준비 단계이며 새 모델 학습·A/B 평가를 하지 않았다. BASE S1+GEO_LINEAR를 교체하지 않았다. 기존 RGB {a['source_frames']}장 → 미사용/중복 제외 후 **{a['eligible_frames']}장 /{len(recs)} recordings** → 첫 라운드 **{counts['1']}장**을 고정했다.

## 2. 왜 이 실험을 했나

이전 preservation adapter는 clean을 회복했지만 severe 이득을 보존하지 못했다. Frozen S1과 teacher가 모두 >10px인 verified visible점8개가 있으나, 이전8031 RGB에는 model-independent hard 태그가 없었다. 이번에는 사람이 원본 영상만 보고 sampling용 난도를 새로 태깅한다. 이전 오류점이나 모델 confidence로 후보를 고르지 않는다.

## 3. Model-blind difficulty tagging

|recording|원본|중복·기존 풀 제외 후|Round1|Round2|Round3|
|---|---:|---:|---:|---:|---:|
{table}

제외 내역: `{a['exclusions']}`. 과거 balanced PLASTIC pool1000장은 실제 학습된 하위 집합을 prediction으로 역추적하지 않고 전체를 보수적으로 제외했다. **1000장 모두 과거에 학습했다는 뜻은 아니다.** 현재 S1 학습 이미지522개(합성512+clean10), split 학습 후보, anchor, HELDOUT 및 reserved identity도 검사했다.

예약 recording의 **전체 원본 RGB**와 평가/anchor/학습 split 보호 영상 총 {a['protected_thumbnails']}개 썸네일에 MAD 검사를 했다. 보존 후보의 평가/예약 SHA 및 MAD 중복0. 풀 내부에서도 SHA 순서로 중복을 제거하여 남은 모든 쌍의 grayscale64×48 MAD>2/255를 보장한다. MAD는 중복 검사에만 쓰고 hard 판정에 쓰지 않았다.

각 recording을 시간순 최대48 bin으로 나눠 `sha256("hard-tag-v1:"+frame_id)` 최소 한 장을 뽑는다. Round1=bin0,3,6…; Round2=1,4,7…; Round3=2,5,8…. 태깅 전에 private queue SHA를 공개 lock에 고정했다. 모델/teacher/좌표 GT/PnP 결과를 파싱하거나 GUI에 표시하지 않았다. 과거 결과 파일은 provenance 보존을 위해 바이트 hash만 검증했다.

![촬영별 고정 태깅 큐](figures/01_tagging_queue_recordings.png)

현재 사람 태깅 수 **{summary['total']}**. 아직 입력하지 않은 영상을 clean이나 hard로 가정하지 않는다.

![사람 난도 입력 상태](figures/02_human_difficulty_distribution.png)

### 지금 하는 조작

```bash
python -m scripts.research.pallet_min_hard_ab_v1.tag_difficulty
```

- **C** 깨끗함: 가림/잘림이 없거나 매우 경미하고 주요 모서리·앞면 구조가 명확함.
- **M** 중간: 일부 모서리는 가려지거나 애매하지만 대부분의 구조와 일부 직접 보이는 점은 확실함.
- **S** 심함: 여러 주요 모서리가 손실되어 구조 복원이 어렵지만 일부 직접 보이는 점은 남음.
- **U** 모호함: 난도나 front/near 역할을 확신 못함.
- **X** 사용 불가: 팔레트가 없거나 흐림/손상으로 주석 불가.
- **Z** 직전 응답 취소. 키를 입력하면 저장하고 바로 다음 이미지로 이동. 중간에 닫아도 이어서 진행.

지금은 **키포인트/박스를 찍지 않는다**. 난도만 입력한다. 첫 라운드 완료 후:

```bash
python -m scripts.research.pallet_min_hard_ab_v1.cli resume
```

Hard≥12, ≥3 recording, recording당 최대3장으로 initial8+reserve2 구성 가능하면 태깅을 중단한다. Severe≥2는 선호이며 강제 변경하지 않는다. 부족할 때만 다음 라운드, 최대 {q['total']}장. Round3까지 부족하면 모델 실패 영상으로 채우지 않고 종료한다.

## 4. Manual annotation — 아직 미수행

태깅 완료 후 hard 인간 태그·recording·고정 SHA만으로 initial8+reserve≤2를 선정한다. 전체에서 recording당 최대3, initial에 최소3 recording. 4M/4S 선호. 원본+정적 역할 안내만 표시하는 전용 입력기로 수동 bbox와 직접 보이는 P0..7만 입력한다. P8/숨은점/PnP 보완은 감독하지 않는다. 역할 불확실은 제외한다. 6 usable/24 clicks/3개 corner 각3점/3recording 기준을 검사한다.

## 5. Causal training contract — 아직 미수행

원래 S1 args를 그대로 hash-bind했다: 5epochs/320updates/batch16/seed42, AdamW lr0=0.0001, cosine/lrf0.1. 이전 adapter의 lr0.001과 다르다. BASE는 기존 S1 재사용. 새 H_MANUAL/H_PSEUDO는 원래 init에서 학습하고 synthetic512/epoch는 동일, real512 중 clean448+hard64로만 대체한다. Hard natural RGB에 추가 가림을 넣지 않는다.

두 H arm은 RGB/bbox/augmentation/occurrence/support mask가 동일하고, 직접 보이는 점의 xy 값만 manual vs frozen teacher로 달라야 한다. Teacher inference는 **사람 label lock 이후에만** 허용한다. Partial hard slot의 box/cls/dfl/visibility loss는0. 남은 BASE slot 수식은 유지한다. 결과를 보고 예산/seed/selector를 조정하지 않는다.

## 6–13. 평가 결과 — NOT_RUN

CLEAN29/MODERATE21/SEVERE78/ALL128, verified FINAL_V2, source256, per-recording, pseudo vs manual, oracle vs current 수치와 학습 적합도는 아직 없다. 예측 lock 전에 평가 GT를 읽지 않는 후속 학습·평가 실행 단계가 남아 있다. 준비 보고서를 최종 A/B 결과로 해석하면 안 된다.

## 14. Objective decision

**NOT_EVALUATED**. 성공/실패를 미리 선언하지 않는다. hard 양쪽과 clean 유지 + PCK/oracle 개선 여부를 요청한 고정 규칙대로 판정한다. 무이득이면 추가20/50장 어노테이션을 자동 요청하지 않는다.

## 15. 다음 단계

지금 사용자는 첫 라운드 난도 태깅만 하면 된다. 사람 태그 → selection lock → 직접 보이는 점·bbox 입력 → QA/label lock 순서를 지킨다. **학습 실행부의 pair-integrity 검증과 실제 fit/evaluation은 label lock 뒤 다음 작업 단계에서 이어서 완료한다.** 현재 prepare/resume은 사람 입력과 label lock까지 처리하며, 그 뒤 `HARD_LABELS_LOCKED_TRAINING_PENDING`에서 안전하게 멈춘다. 사람 입력을 대신 만들거나 자동 학습 성공을 기록하지 않는다.

## 16. 한계

Already-viewed HELDOUT128 DEV이며 독립 TEST가 아니다. 태깅 표본은 시간/중복/기존 노출 제외로 선택되므로 hard prevalence population estimate가 아니다. 후속실험도6–10장, single seed, plastic only, partial manual-visible GT, geometry-derived real6D reference다. Camera-facing 역할과 물리 C2 동치는 다르다. Frozen GEO_LINEAR가 새 모델에 최적이라는 보장은 없고 independent final test는 아직 없다.

[후보 감사](CANDIDATE_POOL_AUDIT.json) · [큐 lock](DIFFICULTY_QUEUE_LOCK.json) · [프로토콜](PROTOCOL_LOCK.json) · [코드/실행 안내](../../../scripts/research/pallet_min_hard_ab_v1/README.md)

준비 감사: 17개 체크와 11개 단위 테스트 PASS. 전체 고정 큐368장의 평가·예약/큐 내부 MAD를 독립 재계산했다. 실제 Tk 창에서 마우스 클릭 없이 키 입력, 즉시 다음 이동, 길게 누름 방지, undo, 저장 후 resume을 임시 태그로 검증했다. 사람 태그/실제 annotation/학습/평가 테스트는 NOT_RUN이다. [준비 테스트](PREPARATION_TESTS.json).
'''
    if (C.DOC/'HARD_SELECTION_LOCK.json').exists():
        from .selection_report import enrich
        report=enrich(report,summary)
    C.save(C.DOC/'REPORT_KO.md',report)

if __name__=='__main__':render()
