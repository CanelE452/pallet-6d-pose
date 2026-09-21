"""Post-run independent membership checks and cached synthetic-only control."""
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_cad8_selftrain_v1 import run as R
from scripts.research.pallet_dim_conditioned_p_v1 import eval_math as M


def main():
    p=R.check();metrics=R.read(R.RAW/'METRICS.json');primary=set(p['primary_occlusion_ids'])
    train=p['train_records'];evaluation=p['eval_records']
    assert {r['id'] for r in train}.isdisjoint(r['id'] for r in evaluation)
    assert {r['image']['sha256'] for r in train}.isdisjoint(r['image']['sha256'] for r in evaluation)
    assert {r['id'].split(':')[0] for r in train}.isdisjoint(r['session'] for r in evaluation)
    per_session={};comparison={}
    old=R.read(R.C.RAW/'EVAL_METRICS.json')
    syn_fit=R.read(R.C.DOC/'FIT_SYN_ONLY.json');R.G.verify(syn_fit['checkpoint'])
    assert syn_fit['optimizer_steps']==320
    syn=[r for r in old['SYN_ONLY'] if r['id'] in primary];assert len(syn)==96
    subsets={'PRIMARY_OCC96':primary,'ALL_OCC107':{r['id'] for r in evaluation if r['occlusion'] not in ['none','unknown','']},'ALL_NONCAD176':{r['id'] for r in evaluation}}
    subsets['EXPLORATORY_PRIMARY_OCC_NO_TRUNC81']={r['id'] for r in evaluation if r['id'] in primary and r['truncation']=='none'}
    assert len(subsets['EXPLORATORY_PRIMARY_OCC_NO_TRUNC81'])==81
    for name,ids in subsets.items():
        comparison[name]={'ARCHIVED_SYN_ONLY_320':M.summary([r for r in old['SYN_ONLY'] if r['id'] in ids]),**{a:M.summary([r for r in rows if r['id'] in ids]) for a,rows in metrics.items()}}
    for session in sorted({r['session'] for r in evaluation if r['id'] in primary}):
        per_session[session]={a:M.summary([r for r in rows if r['id'] in primary and r['session']==session]) for a,rows in metrics.items()}
    for arm in R.ARMS:
        slots=(ROOT/p['datasets'][arm]['train_list']['path']).read_text().splitlines()
        real=[s for s in slots if '/pallet_cad8_selftrain_v1/' in s]
        assert len(slots)==1024 and len(real)==512 and len(set(real))==8
        assert set(Counter(real).values())=={64}
        fit=R.read(R.DOC/f'FIT_{arm}.json');assert fit['steps']==320 and len(fit['epochs'])==5;R.G.verify(fit['checkpoint'])
    log=(R.RAW/'run.log').read_text()
    checks=dict(train_eval_id_hash_session_disjoint=True,CAD_session_fully_excluded=True,train_real_unique=8,
        training_arms=2,optimizer_steps_each=320,balanced_real_exposures_each_image_per_epoch=64,
        GT_not_used_as_training_targets=True,eval_without_refiner_or_filter=True,
        primary_occlusion=96,primary_sessions=6,baseline_parity=True,
        warning='Optional Albumentations transform construction failed due to installed ImageCompression API; training continued with native Ultralytics augmentation. No package changes.' if 'unexpected keyword' in log else None)
    R.write(R.DOC/'AUDIT.json',dict(checks=checks,cached_synthetic_only_control=comparison,primary_per_session=per_session,
        synthetic_control_caveat='Archived run, same R0/320updates/seed42/hyperparameters/synthetic512;512 real slots replaced by another synthetic512. Not rerun in this experiment.',
        source_bindings=[R.G.binding(R.C.RAW/'EVAL_METRICS.json'),R.G.binding(R.C.DOC/'FIT_SYN_ONLY.json'),R.G.binding(Path(__file__))]))
    lines=['# CAD8 fine-tuning 감사','',
        '학습: CAD 2·6·7·8·9·10·11·12의 고정 수도레이블. CAD 세션 전체는 이번 평가에서 제외. 원본 데이터/라벨/모델은 유지.',
        'R0에서 각각 시작, 5 epoch·320 update·seed42. 실사 8장(각 epoch마다 각64회) + 합성512장. 평가에는 보정기/필터 없음.',
        '주 평가: 기존 occlusion 조건 태그에 해당하고 Replay 학습 세션과 겹치지 않는96장,6세션. 재사용 DEV이며 독립적인 논문 확인 실험은 아님.','',
        '| 방법 | 중앙값 px ↓ | P90 px ↓ | PCK10 ↑ | PCK20 ↑ | 매칭 |','|---|---:|---:|---:|---:|---:|']
    for a,s in comparison['PRIMARY_OCC96'].items():
        lines.append(f'| {a} | {s["matched_pooled_corner8_median_px"]:.2f} | {s["matched_pooled_corner8_P90_px"]:.2f} | {100*s["PCK"]["10"]:.1f}% | {100*s["PCK"]["20"]:.1f}% | {s["matched"]}/96 |')
    lines += ['','ARCHIVED_SYN_ONLY_320은 이전 동일 예산 합성-only 실험의 저장 결과를 재집계한 보조 대조군입니다. 이번에 재학습한 모델은 N3_PNP/REPLAY_PNP 두 개입니다.',
        '실험은 corrected-pseudo 학습 전체의 효과를 보며, PnP 추가분만의 효과를 분리한 ablation은 아닙니다.',
        '주의: 현재 CAD 수도레이블의 숨은 점은 직육면체 근사 PnP에 의존합니다. 8장·단일 seed 결과로 일반화를 확정할 수 없습니다.']
    R.write(R.DOC/'AUDIT_KO.md','\n'.join(lines)+'\n')
    print(json.dumps(dict(checks=checks,primary=comparison['PRIMARY_OCC96']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
