"""Generate final-method tables from frozen historical data, without training."""
import json
import numpy as np
from PIL import Image
from scripts.evaluation import final_dimension_release as R


def main():
    lock=R.checked_lock(); E=R.setup()
    from dev_evaluate import population_metadata,iou
    from eval_math import measure,summary,contrast,damage
    pe,pop=population_metadata()
    protocol=R.read(E.LINE/'BASELINE_PROTOCOL.json');pe.assert_sources(protocol)
    baseline=R.read(E.LINE/'baseline/FULL_CANDIDATES.json')
    groups={g['object_type']:g for g in R.read(E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')['objects']}
    metrics=R.read(E.RAW/'REAL_DEV_METRICS.json')
    control={r['id']:r for r in metrics['N0_BASE_REPLAY_seed1']}
    rows=[]
    for item,meta in pop:
        target=pe.E._legacy_forbidden_target(item)
        candidates=baseline['frames'][pe.canonical_key(item.image)]
        selected=int(np.argmax([c['score'] for c in candidates])) if candidates else None
        c=None if selected is None else candidates[selected]
        with Image.open(R.ROOT/item.image) as im: w,h=im.size
        points=np.full((9,2),np.nan) if c is None else c['keypoints_xy']
        matched=c is not None and iou(c['box_xyxy'],target.box_xyxy)>=.5
        m=measure(points,target.keypoints_xy,target.keypoint_supervision_mask,
            groups[meta['object_type']]['permutations'],(h,w),matched,selected is not None)
        assert m['canonical_valid']==control[item.frame_id]['canonical_valid']
        assert m['matched']==control[item.frame_id]['matched']
        rows.append(dict(id=item.frame_id,session=meta['session_id'],**m))
    assert [r['id'] for r in rows]==[r['id'] for r in metrics['N0_BASE_REPLAY_seed1']]
    reports=R.read(R.DCP/'REAL_DEV_RESULTS.json')['summary']
    table=[dict(model='R0',aggregation='single',**summary(rows))]
    keys=['E_sym','matched_pooled_corner8_median_px','matched_pooled_corner8_P90_px','gross20','coverage']
    for arm in ['OLD_P','N0_BASE_REPLAY','N2_DIM_ONLY']:
        rr=[reports[f'{arm}_seed{s}'] for s in (1,2,3)]
        table.append(dict(model=arm,aggregation='three-seed mean',
            **{k:float(np.mean([r[k] for r in rr])) for k in keys},
            PCK={t:float(np.mean([r['PCK'][t] for r in rr])) for t in ('5','10','20')}))
    table.append(dict(model='N2_DIM_ONLY',aggregation='fixed seed1',**reports['N2_DIM_ONLY_seed1']))
    a=[[r['E_sym'] for r in metrics[f'N2_DIM_ONLY_seed{s}']] for s in (1,2,3)]
    comparisons={}
    for arm in ['R0','N0_BASE_REPLAY']:
        b=[[r['E_sym'] for r in (rows if arm=='R0' else metrics[f'{arm}_seed{s}'])] for s in (1,2,3)]
        comparisons[arm]=contrast(a,b,[r['session'] for r in rows])
    results=dict(table=table,paired_session_E_sym=comparisons,
        source_bindings=[R.binding(R.DCP/'REAL_DEV_RESULTS.json'),R.binding(E.RAW/'REAL_DEV_METRICS.json'),
            R.binding(E.LINE/'BASELINE_PROTOCOL.json'),R.binding(E.LINE/'baseline/FULL_CANDIDATES.json'),R.binding(pe.POS)],
        R0_recalculated_using_identical_corner8_contract=True,original_GT_sources_verified=True,
        population='Existing DEV319, 13 reused sessions; NOT new confirmation',
        new_training=0,new_green_inference=0,
        damage_N2_vs_N0={str(s):damage(metrics[f'N0_BASE_REPLAY_seed{s}'],metrics[f'N2_DIM_ONLY_seed{s}']) for s in (1,2,3)})
    R.freeze(R.DOC/'DEV_COMPARISON.json',results)
    R.freeze(R.RAW/'R0_CORNER8_METRICS.json',dict(records=rows))
    md=['# 고정 최종 모델 — 동일 DEV319 비교', '',
        '모든 행은 동일한 corner8 symmetry-aware 지표다. 과거 9-point 중앙오차와 직접 섞지 않는다.',
        '기존 라벨/평가 코드의 고정 해시를 검증하고 R0만 같은 지표로 재계산했다. 신규 데이터 평가가 아니다.', '',
        '| 모델 | 집계 | 중앙오차 px ↓ | P90 px ↓ | PCK10 % ↑ | E_sym ↓ |',
        '|---|---|---:|---:|---:|---:|']
    tex=['\\begin{tabular}{llrrrr}', '\\toprule',
         'Model & Seeds & Median & P90 & PCK10 (\\%) & $E_{sym}$ \\\\', '\\midrule']
    labels={'R0':'R0','OLD_P':'Original P','N0_BASE_REPLAY':'P replay','N2_DIM_ONLY':'Dimension P'}
    for row in table:
        med=row['matched_pooled_corner8_median_px'];p90=row['matched_pooled_corner8_P90_px'];pck=100*row['PCK']['10'];e=row['E_sym']
        md.append(f'| {row["model"]} | {row["aggregation"]} | {med:.4f} | {p90:.4f} | {pck:.2f} | {e:.8f} |')
        seeds='1--3 mean' if row['aggregation']=='three-seed mean' else '1' if row['aggregation']=='fixed seed1' else '--'
        tex.append(f'{labels[row["model"]]} & {seeds} & {med:.3f} & {p90:.3f} & {pck:.2f} & {e:.5f} \\\\')
    md += ['', '## 불확실성과 제한', '']
    for arm,c in comparisons.items():
        md.append(f'- N2 − {arm}, 평균 E_sym 차이 {c["delta"]:.8f}, 세션 bootstrap 95% [{c["CI95"][0]:.8f}, {c["CI95"][1]:.8f}].')
    md += ['- seed 평균은 ensemble이나 배포 seed1의 성능이 아니다.',
           '- N2−N0 평균 개선과 별개로 good<5→bad>10 1건 때문에 기존 엄격 안전 판정은 UNRESOLVED다.',
           '- 20,259 대 18,962 파라미터로 용량이 달라 치수 정보만의 효과를 완전히 분리하지 못했다.',
           '- 기존 PoseFix 비교는 다른 9-point 표에서 P보다 정확했다. N2의 선행 대비 우월성을 새로 주장하지 않는다.',
           '- N0/N4 과거 시간 측정치를 N2 지연으로 재사용하지 않는다. 새 실측은 RUNTIME_RESULTS.json/처리시간 표에 별도 기록한다.',
           '- 정사각형 단일 치수 집단만으로 치수 조건화의 효용을 식별할 수 없다.', '']
    (R.DOC/'DEV_COMPARISON.md').write_text('\n'.join(md))
    (R.DOC/'dev_table.tex').write_text('\n'.join(tex+['\\bottomrule','\\end{tabular}','']))
    runtime_path=R.DOC/'RUNTIME_RESULTS.json'
    if runtime_path.exists():
        runtime=R.read(runtime_path)
        rmd=['# 동일 세션 처리시간 — 재학습 없음', '',
             '| 모델 (seed1) | RGB→2D 중앙 ms | PnP 포함 중앙 ms | PnP 포함 P90 ms |',
             '|---|---:|---:|---:|']
        rtex=['\\begin{table}[t]\\centering',
              '\\caption{Same-session desktop timing (ms), seed 1; images already in RAM.}',
              '\\begin{tabular}{lrrr}\\toprule',
              'Model & RGB-to-2D & Full & Full P90 \\\\ \\midrule']
        for arm in ['R0','N0_BASE_REPLAY','N2_DIM_ONLY']:
            s=runtime['summary'][arm]
            a=s['RGB_to_2D_ms']['median'];b=s['full_ms']['median'];c=s['full_ms']['P90']
            rmd.append(f'| {arm} | {a:.3f} | {b:.3f} | {c:.3f} |')
            rtex.append(f'{labels[arm]} & {a:.3f} & {b:.3f} & {c:.3f} \\\\')
        rmd += ['',f'각 모델 {runtime["samples_per_arm"]}개 유효 표본 전부 사용. R0와 두 head 상주 상태의 process 범위 메모리.',
                '이미지 디코딩/카메라 캡처 제외. 공유 데스크톱 RTX3080; Jetson/독립 process/전력 측정 아님. GPU/CPU 동기화 및 모든 raw 표본 보존.', '']
        rtex += ['\\bottomrule\\end{tabular}', '\\end{table}',
                 f'The measurement retains all {runtime["samples_per_arm"]} samples per model with 20 warm-up calls and five repeats. '
                 'R0 and both heads share the GPU process. Thread counts, parity checks, and raw samples are recorded; no faster subset is selected.', '']
        (R.DOC/'RUNTIME.md').write_text('\n'.join(rmd))
        (R.DOC/'runtime_table.tex').write_text('\n'.join(rtex))
    print('\n'.join(md))


if __name__=='__main__': main()
