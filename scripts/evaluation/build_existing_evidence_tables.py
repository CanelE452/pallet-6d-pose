"""Format completed DEV evidence without inference, training, or new scoring."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / '_docs/paper/final_dimension_v1'


def main():
    source = DOC / 'DEV_COMPARISON.json'
    data = json.loads(source.read_text())
    for binding in data['source_bindings']:
        raw = (ROOT / binding['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding['sha256']:
            raise ValueError('Historical source changed: ' + binding['path'])
    rows = data['table']
    labels = ['R0', 'Original P', 'P replay', 'Dimension P', 'Dimension P (seed 1)']
    assert [r['model'] for r in rows] == ['R0', 'OLD_P', 'N0_BASE_REPLAY', 'N2_DIM_ONLY', 'N2_DIM_ONLY']
    latex = [r'\begin{table}[t]\centering',
             r'\caption{Completed DEV319: PCK and matching coverage (\%). Refiner rows are means of three seed-level scores except the fixed seed-1 row.}',
             r'\begin{tabular}{lrrrr}\toprule',
             r'Model & PCK5 & PCK10 & PCK20 & Coverage \\ \midrule']
    md = ['# 기존 결과만으로 구성한 추가 실험표', '',
          '신규 평가·학습 없이 DEV_COMPARISON.json을 표로 변환했다. 원본 수치 및 source SHA를 확인했다.',
          '새 초록 150장은 포함하지 않는다. 재사용 DEV이며 독립 평가라고 이름을 바꾸지 않는다.', '',
          '| 모델 | PCK5 % | PCK10 % | PCK20 % | 매칭 coverage % |', '|---|---:|---:|---:|---:|']
    for label, row in zip(labels, rows):
        values = [100 * row['PCK'][str(t)] for t in (5, 10, 20)] + [100 * row['coverage']]
        assert all(0 <= x <= 100 for x in values)
        latex.append(label + ' & ' + ' & '.join(f'{v:.2f}' for v in values) + r' \\')
        md.append('| ' + label + ' | ' + ' | '.join(f'{v:.2f}' for v in values) + ' |')
    latex += [r'\bottomrule\end{tabular}', r'\label{tab:coverage}', r'\end{table}',
              'All arms retain the same detections and boxes: 319 detections and 311 matched frames. '
              'The complete supervised denominator has 2,499 corners; the matched summaries contain 2,445 corners.',
              r'\begin{table*}[t]\centering',
              r'\caption{Completed DEV319 paired N2--N0 diagnostics. Negative $\Delta E_{sym}$ means improvement. Improved/worsened/unchanged count frames; good-to-bad counts canonical corners with error $<5$ to $>10$ pixels.}',
              r'\begin{tabular}{lrrrrr}\toprule',
              r'Seed & $\Delta E_{sym}$ & Improved & Worsened & Unchanged & Good-to-bad \\ \midrule']
    md += ['', '| Seed | ΔE_sym (N2−N0) | 개선 프레임 | 악화 프레임 | 동일 프레임 | good→bad 코너 |',
           '|---|---:|---:|---:|---:|---:|']
    for i, delta in enumerate(data['paired_session_E_sym']['N0_BASE_REPLAY']['per_seed_delta'], 1):
        r = data['damage_N2_vs_N0'][str(i)]
        counts = [r[k] for k in ('improved_frames', 'harmed_frames', 'unchanged_frames', 'good5_to_bad10')]
        assert sum(counts[:3]) == 319
        values = [str(i), f'{delta:.8f}'] + list(map(str, counts))
        latex.append(' & '.join(values) + r' \\')
        md.append('| ' + ' | '.join(values) + ' |')
    latex += [r'\bottomrule\end{tabular}', r'\label{tab:damage}', r'\end{table*}',
              'All three seeds improve the mean normalized error, but none improves every frame. '
              'Seed 1 has one good-to-bad corner; the reverse threshold transition is zero for all three seeds. '
              'This descriptive check is not a calibrated task-safety threshold.']
    md += ['', 'seed는 독립 촬영 표본이 아니다. 평균 개선과 모든 프레임의 개선은 구분한다.',
           'good→bad는 연구 내부 보조 기준이며 실작업 안전성 인증이 아니다.']
    (DOC / 'existing_evidence_tables.tex').write_text('\n'.join(latex) + '\n')
    (DOC / 'EXISTING_EVIDENCE_TABLES.md').write_text('\n'.join(md) + '\n')
    audit = dict(status='EXISTING_RESULTS_FORMATTED_NOT_NEW_EXPERIMENT',
                 source=dict(path=str(source.relative_to(ROOT)), sha256=hashlib.sha256(source.read_bytes()).hexdigest()),
                 new_training=0, new_inference=0, green_frames_included=0,
                 tables=['dev_table.tex', 'existing_evidence_tables.tex', 'runtime_table.tex'],
                 source_bindings_verified=True, pdf_compiled=False)
    (DOC / 'EXISTING_EVIDENCE_CLOSEOUT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
