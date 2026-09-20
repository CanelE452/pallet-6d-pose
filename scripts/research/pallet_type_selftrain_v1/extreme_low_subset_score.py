"""Reaggregate saved corner8 metrics on a previously frozen RGB-only subset."""
import numpy as np
from . import common as C
from .extreme_low_subset import DOC, OUT
from scripts.research.pallet_dim_conditioned_p_v1.eval_math import summary


def ordered(rows, ids):
    by_id = {r['id']: r for r in rows}
    assert len(by_id) == len(rows)
    assert set(ids) <= set(by_id)
    return [by_id[i] for i in ids]


def main():
    lock = C.read(DOC/'LOCK_COMPLETE.json')
    C.verify(lock['manifest'])
    manifest = C.read(DOC/'SUBSET_MANIFEST.json')
    for b in manifest['sources']:
        C.verify(b)
    decisions = manifest['decisions']
    ids = [r['id'] for r in decisions]
    kept = [r['id'] for r in decisions if r['decision']=='keep']
    excluded = [r['id'] for r in decisions if r['decision']=='exclude']
    assert len(ids)==194 and len(kept)==159 and len(excluded)==35
    assert not set(kept)&set(excluded) and set(kept)|set(excluded)==set(ids)
    paths = [C.RAW/'EVAL_METRICS.json',
        C.ROOT/'data/pallet/results/final_dimension_v1/R0_CORNER8_METRICS.json',
        C.ROOT/'data/pallet/results/pallet_dim_conditioned_p_v1/REAL_DEV_METRICS.json',
        C.RAW/'selftrain_recovery_v1/pose_only/SCREEN_REF_LR5.json']
    r0 = ordered(C.read(paths[0])['R0'],ids)
    paper_r0 = ordered(C.read(paths[1])['records'],ids)
    # Independently cached paper R0 must agree with the self-training R0 contract.
    max_error_delta = 0.
    for a,b in zip(r0,paper_r0):
        for key in ('evaluable','canonical_valid','matched','detected','corners'):
            assert a[key]==b[key], (a['id'],key)
        delta = float(np.max(np.abs(np.array(a['errors'])-np.array(b['errors']))))
        max_error_delta = max(max_error_delta,delta)
        assert np.allclose(a['errors'],b['errors'],rtol=0,atol=1e-5),a['id']
    arms = {'R0': r0,
        'N2_DIM_ONLY_seed1':ordered(C.read(paths[2])['N2_DIM_ONLY_seed1'],ids),
        'SELFTRAIN_REF_LR5':ordered(C.read(paths[3])['metrics'],ids)}
    for rows in arms.values():
        for a,b in zip(r0,rows):
            assert a['canonical_valid']==b['canonical_valid']
            assert a['evaluable']==b['evaluable']
    results={}
    for name,rows in arms.items():
        results[name]={key:summary(ordered(rows,subset)) for key,subset in
            [('full194',ids),('retained159',kept),('excluded35',excluded)]}
    # Verify source labels/images remain untouched, including excluded evaluation.
    original=C.read(C.DOC/'EVAL_PROTOCOL.json')
    for r in original['records']:
        C.verify(r['image']); C.verify(r['annotation'])
    result=dict(status='SAVED_METRICS_REAGGREGATED_NO_TRAINING_NO_INFERENCE',
        subset=C.bound(DOC/'SUBSET_MANIFEST.json'),sources=[C.bound(p) for p in paths]+[C.bound(__file__)],
        results=results,R0_cross_cache_max_error_delta_px=max_error_delta,
        original469_images_and_annotations_verified=True,
        metric='Symmetry-aware corner8: matched pooled errors for median/P90, all evaluable corners with missing/mismatch penalty for PCK20. Not historical 9-point metrics or AP.',
        caveat='Posthoc evaluation scope restriction, not recovery or model improvement; no independent confirmation and no fresh model selection.',
        unchanged_types=['GREEN150','WOOD125','negatives'])
    C.freeze(DOC/'RESULTS.json',result)
    lines=['# 극저각 범위 제외 — 일반 플라스틱 평가', '',
        '일반 플라스틱 **194 → 159장**: 극저각 35장 제외(18.04%). 초록150·목재125 유지, 양성 전체469 → 434장. Negative는 변경하지 않았다.', '',
        'RGB만 보고 전체194장을 정성 검토했다. LOW 태그만으로 제외하지 않았으며 경계 사례는 유지했다. 수치 각도 임계값으로 정의한 자동 필터가 아니다.',
        '제외: eval_outside 3장, eval_pallet09 31장, plastic_night_01 1장. 프레임별 사유·해시·유지 목록은 SUBSET_MANIFEST.json에 고정했다. 목록 고정 후 기존 저장 지표만 다시 집계했다.', '',
        '## 동일 corner8 지표', '',
        '| 모델 | 평가 범위 | 중앙오차 px ↓ | P90 px ↓ | PCK20 % ↑ | 매칭/전체 |',
        '|---|---|---:|---:|---:|---:|']
    for name,rr in results.items():
        for key in ('full194','retained159','excluded35'):
            s=rr[key]
            lines.append(f'| {name} | {key} | {s["matched_pooled_corner8_median_px"]:.3f} | {s["matched_pooled_corner8_P90_px"]:.3f} | {100*s["PCK"]["20"]:.2f} | {s["matched"]}/{s["total_frames"]} |')
    lines += ['', '## 해석 및 사용', '',
        '- 원래 전체194 결과를 함께 보존한다. 이것은 사후 평가 범위 축소이며, 새 독립 테스트나 모델 개선·큰 오차 복구를 의미하지 않는다.',
        '- R0, 논문 고정 N2 seed1, 기존 보수적 self-training REF_LR5를 동일한159장으로 비교했다. 새 학습·추론·모델 채택은 하지 않았다.',
        '- 중앙/P90은 매칭된 예측의 관측 가능한8개 코너 오차를 모은 값이다. PCK20은 매칭 실패 패널티를 포함한다. 과거9-point/AP 표와 직접 비교하지 않는다.',
        '- 모든 원본 이미지·주석469개 해시를 다시 검증했다. 원본 split과 기존 평가 기본값·논문 표는 덮어쓰지 않았다.',
        '- 후속 평가는 이 디렉터리의 SUBSET_MANIFEST.json records를 명시적으로 사용한다. 제외35장도 계속 평가 출신으로 취급하고 학습으로 옮기지 않는다.',
        '- 기존 개발 과정에서 본 평가이므로 posthoc 제한을 논문에 명시한다. 양각 제한 없이 일반적 강건성을 입증했다는 주장은 할 수 없다.', '',
        '[제외35장 원본 이미지](../../../../../outputs/pallet_type_selftrain_v1/large_corner_recovery_v1/extreme_low_subset_v1/excluded.html)', '']
    C.write_text(OUT/'RESULTS_KO.md','\n'.join(lines))
    print('\n'.join(lines))


if __name__=='__main__':main()
