"""Format immutable GREEN150 scores and provenance into paper/review tables."""
from pathlib import Path
import json
from scripts.evaluation import final_dimension_release as F
from scripts.evaluation import green_saved_labels_v1 as G

LABELS = {'R0': 'YOLO', 'N0_BASE_REPLAY': 'Basic P', 'N2_DIM_ONLY': 'Dimension P'}


def average(values):
    return None if not values or any(v is None for v in values) else sum(values) / len(values)


def aggregate(summaries, arm):
    rows = [summaries['R0']] if arm == 'R0' else [summaries[f'{arm}_seed{s}'] for s in (1, 2, 3)]
    return dict(model=arm, aggregation='single' if arm == 'R0' else 'three-seed mean',
                median=average([r['matched_pooled_corner8_median_px'] for r in rows]),
                P90=average([r['matched_pooled_corner8_P90_px'] for r in rows]),
                E_sym=average([r['E_sym'] for r in rows]),
                PCK={str(t): average([r['PCK'][str(t)] for r in rows]) for t in (5, 10, 20)},
                coverage=average([r['coverage'] for r in rows]),
                frames=rows[0]['total_frames'], corners=rows[0]['corners'],
                detected=rows[0]['detected'], matched=rows[0]['matched'])


def num(x, digits=3, scale=1):
    return '--' if x is None else f'{scale*x:.{digits}f}'


def main():
    snapshot = G.load_dataset()
    path = G.RAW / 'METRICS.json'
    result = F.read(path)
    F.verify(result['predictions'])
    tables = {mode: [aggregate(info['summary'], arm) for arm in LABELS] for mode, info in result['modes'].items()}
    md = ['# 초록 팔레트 150장 — 현재 저장 라벨 기준 실제 평가', '',
          'R0와 기존 N0/N2 seed1/2/3 GPU 추론 완료. 새 학습/모델 변경/라벨 수정 없음.',
          '사용자 요청으로 초록을 포함했다. 최종 사람 검수 완료 또는 독립 테스트로 간주하지 않는다.',
          '잘림47 / 잘리지 않은 다각도73 / 저녁30. 조건별 표는 중복 조건을 허용하므로 합산하지 않는다.',
          '128장의 카메라 값 불일치 때문에 직접 클릭한 681개 코너를 주 분석, PnP 생성점519개 포함 1200개를 보조 분석으로 분리했다.',
          '이 변경은 모델 추론 전에 명시했다. 매칭용 GT bbox는 기존 전체 알려진 점으로 만들므로 자동 생성점의 영향을 받을 수 있다.',
          '초록 수치는 2D 관측점 평가이지 물리적 6D 정확도나 치수 입력의 인과 효과 검증이 아니다.', '']
    tex = []
    for mode, title, caption in (
        ('manual_only', '직접 클릭 코너 681개 — 주 분석', 'Green150 saved-label analysis on 681 manually clicked corners.'),
        ('all_known_proxy', '자동 생성점 포함 코너 1200개 — 보조 분석', 'Green150 proxy analysis on 1,200 known corners, including 519 PnP-derived points.'),
    ):
        md += [f'## {title}', '', '| 모델 | 중앙 px | P90 px | PCK5 % | PCK10 % | PCK20 % | E_sym | coverage % |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
        tex += [r'\begin{table*}[t]\centering', '\\caption{' + caption + ' Median/P90 are matched-only. PCK and normalized error retain the full denominator. Refiner rows average three seed-level scores.}',
                r'\begin{tabular}{lrrrrrrr}\toprule', r'Model & Median (px) & P90 (px) & PCK5 (\%) & PCK10 (\%) & PCK20 (\%) & $E_{sym}$ & Coverage (\%) \\ \midrule']
        for row in tables[mode]:
            values = [LABELS[row['model']], num(row['median']), num(row['P90']),
                      *(num(row['PCK'][str(t)], 2, 100) for t in (5, 10, 20)), num(row['E_sym'], 6), num(row['coverage'], 2, 100)]
            md.append('| ' + ' | '.join(values) + ' |')
            tex.append(' & '.join(values) + r' \\')
        tex += [r'\bottomrule\end{tabular}', '\\label{tab:green-' + mode.replace('_', '-') + '}', r'\end{table*}']
        row = tables[mode][0]
        md += ['', f"검출 {row['detected']}/150, 매칭 {row['matched']}/150; 검출/박스/매칭은 모든 모델에서 동일.", '']
    groups = {'truncation': 'Truncation', 'handheld_multiview': 'Handheld',
              'evening_capture_candidate': 'Evening', 'darker_dusk_candidate': 'Darker dusk subset'}
    md += ['## 조건별 — 직접 클릭 코너', '', '| 조건 | 장수 | 모델 | 중앙 px | PCK10 % | E_sym | coverage % |', '|---|---:|---|---:|---:|---:|---:|']
    tex += [r'\begin{table*}[t]\centering', r'\caption{Green150 manual-corner condition breakdown. Conditions overlap; the seven darker-dusk frames are descriptive only. Refiner results are three-seed means.}',
            r'\begin{tabular}{lrlrrrr}\toprule', r'Condition & Frames & Model & Median (px) & PCK10 (\%) & $E_{sym}$ & Coverage (\%) \\ \midrule']
    for group, title in groups.items():
        for arm in LABELS:
            row = aggregate(result['modes']['manual_only']['condition'][group], arm)
            values = [title, str(row['frames']), LABELS[arm], num(row['median']), num(row['PCK']['10'], 2, 100), num(row['E_sym'], 6), num(row['coverage'], 2, 100)]
            md.append('| ' + ' | '.join(values) + ' |')
            tex.append(' & '.join(values) + r' \\')
    tex += [r'\bottomrule\end{tabular}', r'\label{tab:green-conditions}', r'\end{table*}']
    md += ['', '## Seed별 N2−N0 — 직접 클릭 코너', '',
           '| Seed | ΔE_sym | 개선 프레임 | 악화 프레임 | 동일 | good→bad 코너 |', '|---|---:|---:|---:|---:|---:|']
    tex += [r'\begin{table*}[t]\centering', r'\caption{Green150 manual-corner N2--N0 paired diagnostics by training seed. No independent-capture confidence interval is asserted.}',
            r'\begin{tabular}{lrrrrr}\toprule', r'Seed & $\Delta E_{sym}$ & Improved frames & Worsened frames & Unchanged frames & Good-to-bad corners \\ \midrule']
    for seed, row in result['modes']['manual_only']['damage_and_delta']['N0_BASE_REPLAY'].items():
        values = [seed, num(row['delta_E_sym'], 8)] + [str(row[k]) for k in ('improved_frames', 'harmed_frames', 'unchanged_frames', 'good5_to_bad10')]
        md.append('| ' + ' | '.join(values) + ' |')
        tex.append(' & '.join(values) + r' \\')
    tex += [r'\bottomrule\end{tabular}', r'\label{tab:green-seeds}', r'\end{table*}']
    md += ['', '## 해석', '',
           '직접 클릭점 기준 N2는 R0보다 중앙오차가 낮지만 PCK10은 낮아졌다. N2가 기본 P보다 좋다는 일관된 근거는 없다.',
           'PnP점을 포함하면 N2의 중앙오차는 N0보다 낮지만 E_sym/PCK10은 조금 나쁘다. 유리한 중앙값만 골라 성공으로 판정하지 않는다.',
           '세 seed의 차이는 원본 METRICS.json에 모두 보존했다. 촬영 폴더를 독립 표본으로 취급한 유의성 검정은 하지 않았다.',
           '기존 DEV319와 초록은 라벨/분모/매칭 계약이 달라 하나의 합산 성능으로 섞지 않는다. 기존 모델과 데이터는 변경하지 않는다.']
    (G.DOC / 'RESULTS_KO.md').write_text('\n'.join(md) + '\n')
    (F.DOC / 'green_tables.tex').write_text('\n'.join(tex) + '\n')
    (G.DOC / 'TABLES.json').write_text(json.dumps(dict(tables=tables, source=F.binding(path),
                  snapshot=F.binding(G.DATASET), exporter=F.binding(Path(__file__))), indent=2) + '\n')
    combined = ['# 논문 실험표 모음 — 기존 319장 + 초록 150장', '',
                '기존 319장과 초록 150장은 서로 다른 라벨/매칭 조건으로 별도 표기한다. 469장 합산 결과가 아니다.',
                '초록 결과는 현재 저장 라벨의 추가 2D 평가이며 최종 사람 검수/독립 테스트 인증은 아니다.', '',
                (F.DOC / 'DEV_COMPARISON.md').read_text(),
                (F.DOC / 'EXISTING_EVIDENCE_TABLES.md').read_text(),
                '\n'.join(md), (F.DOC / 'RUNTIME.md').read_text()]
    (F.DOC / 'ALL_EXPERIMENT_TABLES.md').write_text('\n\n'.join(combined) + '\n')
    print(json.dumps(tables['manual_only'], indent=2))


if __name__ == '__main__':
    main()
