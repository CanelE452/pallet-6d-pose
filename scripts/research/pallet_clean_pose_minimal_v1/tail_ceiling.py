"""기존 두 후보의 T-only oracle로 선택기만의 tail 개선 상한을 진단한다."""
from collections import Counter
from pathlib import Path
import numpy as np
from . import common as C

ARMS = ('R0_GEO', 'OLD_RAW_GEO', 'OLD_REF_GEO', 'REF_CLEAR_S42_GEO', 'REF_OCC_S42_GEO', 'REF_OCC_S43_GEO')


def run():
    output = C.DOC/'CANDIDATE_TAIL_CEILING.json'
    if output.exists():
        for binding in C.read(output)['sources']:
            C.verify(binding)
        return C.read(output)
    metadata_path = C.OLD.RAW/'evaluation/S42/METADATA.json'
    rows = C.read(metadata_path); ids = [r['id'] for r in rows if r['severity'] != 'CLEAN']
    assert len(ids) == 99
    recordings = {r['id']: r['recording'] for r in rows}
    paths = [metadata_path, C.RAW/'CURRENT_GEO_FRAME_METRICS_PRIVATE.json', C.RAW/'CURRENT_GEO_CASE_BINDINGS_PRIVATE.json',
             C.RAW/'HISTORICAL_RAW_FRAME_METRICS_PRIVATE.json', C.RAW/'HISTORICAL_RAW_CASE_BINDINGS_PRIVATE.json', Path(__file__)]
    metrics = C.read(paths[1]); association = C.read(paths[2]); metrics.update(C.read(paths[3])); association.update(C.read(paths[4]))
    result, private, cache = {}, {}, {}
    for arm in ARMS:
        binding = association[arm]
        C.verify(binding['candidates'])
        candidate_path = C.ROOT/binding['candidates']['path']
        metric_path = candidate_path.parent/'ORACLE_METRICS_SELECTIONS_PRIVATE.json'
        if metric_path not in cache:
            cache[metric_path] = C.read(metric_path)['candidate_metrics']; paths.append(metric_path)
        options = cache[metric_path][binding['candidate_arm']]
        current = metrics[arm]
        best = {}
        for fid in ids:
            valid = [r for r in options[fid] if r['metric']['available']]
            assert current[fid]['available'] and valid
            chosen = min(valid, key=lambda r: (r['metric']['translation_cm'], r['name']))
            best[fid] = chosen
        worst = sorted(ids, key=lambda f: (-current[f]['translation_cm'], f))[:10]
        selected_t = [current[f]['translation_cm'] for f in ids]
        oracle_t = [best[f]['metric']['translation_cm'] for f in ids]
        residual = [current[f]['translation_cm']-best[f]['metric']['translation_cm'] for f in worst]
        assert min(residual) >= -1e-7
        result[arm] = dict(frames=99, valid=99, current_T_P90_cm=float(np.quantile(selected_t, .9)),
            T_only_oracle_P90_cm=float(np.quantile(oracle_t, .9)),
            same_selected_worst10=dict(frames=10, already_T_best=sum(abs(v) < 1e-7 for v in residual),
                selected_T_median_cm=float(np.median([current[f]['translation_cm'] for f in worst])),
                oracle_T_median_cm=float(np.median([best[f]['metric']['translation_cm'] for f in worst])),
                paired_possible_T_reduction_median_cm=float(np.median(residual)),
                recordings=dict(Counter(recordings[f] for f in worst))),
            current_equals_T_oracle_P90=bool(np.isclose(np.quantile(selected_t, .9), np.quantile(oracle_t, .9), atol=1e-9, rtol=0)))
        private[arm] = dict(selected_worst10=worst, T_only_oracle={f: dict(name=best[f]['name'], translation_cm=best[f]['metric']['translation_cm']) for f in ids})
    private_path = C.RAW/'CANDIDATE_TAIL_CEILING_PRIVATE.json'; C.save(private_path, private, True)
    value = dict(created_at=C.now(), status='POSTHOC_DIAGNOSTIC_ONLY', arms=result,
        selection_rule='각 고정 deployable 출력의 T 큰 순 top10, ID tie. 같은10을 두 후보 T-only oracle와 비교; oracle로 집합을 다시 고르지 않음.',
        oracle_definition='각 프레임의 저장된 두 W/D 최종 후보 중 T 최소. 평가 참조 사용 진단이며 배포/학습/라우팅이 아님.',
        interpretation_ko='명시한 6개 설정은 현재 T P90과 T-only 후보 oracle P90이 같다. 현재 후보 안에서 선택기만 바꾸는 것으로 그 T P90을 낮출 여지가 없다.',
        limitations=['이는 모든 향후 후보 생성/좌표 추정 방법이 불가능하다는 뜻이 아니다.',
            '큰 T 오류의 point/geometry/reference 원인은 이 비교만으로 확정하지 않는다.',
            'T oracle와 R oracle의 개별 최솟값을 하나의 실현 가능한 자세로 합치지 않았다.',
            '6개 명시 설정의 사후 진단이며 전체33개를 검사했다고 주장하지 않는다.'],
        new_fits=0, optimizer_updates=0, GPU_seconds=0, private_artifact=C.bind(private_path),
        sources=[C.bind(p) for p in dict.fromkeys(paths)])
    C.save(output, value, True)
    lines = ['# 선택기만으로 복구 가능한 위치 tail 상한', '', value['interpretation_ko'], '',
        '| 설정 | 현재 T P90 cm | T-only oracle P90 cm | 선택된 worst10 중 이미 T최선 | 가능한 T감소량 중앙값 cm |',
        '|---|---:|---:|---:|---:|']
    for arm, r in result.items():
        s = r['same_selected_worst10']
        lines.append(f"| {arm} | {r['current_T_P90_cm']:.4f} | {r['T_only_oracle_P90_cm']:.4f} | {s['already_T_best']}/10 | {s['paired_possible_T_reduction_median_cm']:.4f} |")
    lines += ['', value['selection_rule'], '', '각 설정의 worst10은 REC_007 2장, REC_022 5장, REC_027 3장이다. 평균 난도 중앙값을 계산한 것이 아니라 같은99프레임의 오차를 직접 집계했다.', '',
        value['oracle_definition'], '', *['- '+x for x in value['limitations']], '', '[출처·전체 수치](CANDIDATE_TAIL_CEILING.json)', '']
    C.save(C.DOC/'CANDIDATE_TAIL_CEILING_KO.md', '\n'.join(lines), True)
    print('CANDIDATE_TAIL_CEILING_COMPLETE', len(result), flush=True)
    return value


if __name__ == '__main__':
    run()
