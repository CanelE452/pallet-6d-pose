"""Compare the full e method with existing baselines using sealed image scores."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np

from .scatter import COLORS, NAMES, MARKERS, METRICS


ARMS = ('E0', 'N0', 'POSEFIX_RAW', 'N1')
COHORTS = ('FULL192', 'RETAINED121', 'EXCLUDED71')
COHORT_NAMES = {
    'FULL192': '전체 192장 · 제외 없음',
    'RETAINED121': '공통 제외 후 121장 · 사후 보조 분석',
    'EXCLUDED71': '공통 제외된 71장',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        stream.write(value if isinstance(value, str) else
                     json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def build(source, output):
    start = time.monotonic()
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError('Preserve earlier scatter artifacts')
    seal = json.loads((source.parent / 'SEAL.json').read_text())
    if seal['assets'][source.name] != sha(source):
        raise ValueError('Saved plot data seal mismatch')
    data = json.loads(source.read_text())
    points = data['points']
    if len(points) != 192 or [p['index'] for p in points] != list(range(1, 193)):
        raise ValueError('Preserve every original image and its x coordinate')
    if len({p['id'] for p in points}) != 192:
        raise ValueError('Image IDs must be unique')
    populations = {}
    stats = {}
    paired = {}
    for cohort, expected in zip(COHORTS, (192, 121, 71)):
        ids = set(data['cohorts'][cohort])
        selected = [p for p in points if p['id'] in ids]
        if len(selected) != expected or len(ids) != expected:
            raise ValueError('Use the same fixed images for every arm')
        populations[cohort] = selected
        stats[cohort] = {}
        paired[cohort] = {}
        for arm in ARMS:
            values = np.array([[p['metrics'][arm][m] for m in METRICS] for p in selected])
            if not np.isfinite(values).all():
                raise ValueError('Do not omit unavailable errors')
            stats[cohort][arm] = {
                'n': len(selected),
                'success': int(np.all(values <= 5, axis=1).sum()),
                **{m: {'mean': float(values[:, j].mean()),
                       'variance_sample': float(values[:, j].var(ddof=1))}
                   for j, m in enumerate(METRICS)},
            }
        for arm in ARMS[1:]:
            paired[cohort][arm] = {}
            for metric in METRICS:
                delta = np.array([p['metrics'][arm][metric] - p['metrics']['E0'][metric]
                                  for p in selected])
                paired[cohort][arm][metric] = {
                    'baseline_lower_error': int((delta < -1e-9).sum()),
                    'e_lower_error': int((delta > 1e-9).sum()),
                    'tie': int((np.abs(delta) <= 1e-9).sum()),
                    'mean_baseline_minus_e': float(delta.mean()),
                }
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    font = Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus': False})
    for metric, (label, unit, stem) in METRICS.items():
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, layout='constrained')
        fig.suptitle(f'우리 전체 방법 e vs N3→SubPix + F · {label}\n'
                     '점 하나 = 이미지 한 장의 개별 오차 · 낮을수록 좋음 · 점선 = 방법별 평균', fontsize=17)
        for ax, cohort in zip(axes, COHORTS[:2]):
            selected = populations[cohort]
            for j, group in enumerate(data['session_blocks']):
                if j % 2:
                    ax.axvspan(group['first']-.5, group['last']+.5,
                               color='#eaf0f5', alpha=.8, zorder=0)
                if j:
                    ax.axvline(group['first']-.5, color='#bdc8d2', lw=.7)
            for arm in ('E0', 'N0'):
                mean = stats[cohort][arm][metric]['mean']
                name = '우리 전체 방법 e' if arm == 'E0' else NAMES[arm]
                ax.scatter([p['index'] for p in selected],
                           [p['metrics'][arm][metric] for p in selected],
                           color=COLORS[arm], marker=MARKERS[arm], s=32, alpha=.7,
                           linewidths=.7, label=f'{name} · 평균 {mean:.2f}{unit}', zorder=3)
                ax.axhline(mean, color=COLORS[arm], linestyle='--', lw=1.4)
            ax.set(title=COHORT_NAMES[cohort], ylabel=f'이미지별 {label} ({unit})',
                   xlim=(.5, 192.5), ylim=(0, None))
            ax.legend(loc='upper left', framealpha=.95, fontsize=11)
            ax.grid(axis='y', alpha=.2)
        axes[1].set_xlabel('이미지 순번 · 원래 192장 순서 유지 (제외 후에도 순번 유지)')
        fig.savefig(output / f'{stem}_e_vs_n0.png', dpi=150)
        plt.close(fig)

    payload = {
        'points': points, 'cohorts': data['cohorts'], 'cohort_names': COHORT_NAMES,
        'statistics': stats, 'paired': paired, 'session_blocks': data['session_blocks'],
        'names': {a: ('우리 전체 방법 e' if a == 'E0' else NAMES[a]) for a in ARMS},
        'colors': {a: COLORS[a] for a in ARMS},
        'point_definition': 'one image, individual error; fixed x in original PRIMARY192',
        'paired_direction': 'baseline minus E0; lower error counted with tolerance 1e-9',
    }
    from plotly.offline import get_plotlyjs
    template = Path(__file__).with_suffix('.html')
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    page = template.read_text().replace('__PLOTLY_JS__', get_plotlyjs()).replace('__PLOT_DATA__', encoded)
    write(output / 'index.html', page)
    write(output / 'PLOT_DATA.json', payload)
    write(output / 'PAIRED_STATS.json', {'statistics': stats, 'paired': paired})
    write(output / 'README_KO.md', '''# 우리 전체 방법 e와 기존 비교 방법의 이미지별 산점도

[대화형 산점도](index.html)

기본 비교는 우리 전체 방법 e와 N3→SubPix+F다. HTML에서 PoseFix 또는 최소 B/I를 선택하고, 겹쳐 보기와 나란히 보기를 전환할 수 있다. 위치(cm)와 회전(°)을 별도로 비교하며, 점 하나는 이미지 한 장의 개별 오차다. 점선은 현재 선택한 같은 영상 집합에서의 방법별 평균이다. x는 기존 PRIMARY192 순서를 유지한다. 나란히 보기의 좌우 y축 범위도 같다.

전체 192장이 기본이며 실패를 제거하지 않는다. 공통 제외 후 121장과 제외된 71장은 이전과 동일한 영상 집합이다. 다섯 경로 중 하나라도 N0 유래 IQR 상한(위치 18.728249cm 또는 회전 7.221970°)을 넘으면 모든 경로에서 그 영상을 공통 제외한 사후 보조 분석이며, 전체 결과를 대체하지 않는다. CAL46은 별도다.

N3→SubPix+F는 전체 192장 평균 위치·회전 오차가 e보다 낮지만, 공동 성공 수는 e가 더 많다. 121장에서는 e의 평균 오차와 공동 성공 수가 가장 좋다. 비교 방법이 모든 영상에서 더 좋다는 뜻은 아니다. 성공은 위치 ≤5cm 및 회전 ≤5°를 동시에 만족하는 경우다.

파랑 사각형이 우리 전체 방법 e, 주황 원이 N3→SubPix+F다. PNG 두 개는 각각 위에 전체 192장, 아래에 공통 제외 후 121장을 표시한다. 이미지는 제외 후에도 번호를 다시 매기지 않는다. 선택 필터는 양쪽 방법에 동일하게 적용한다. 기존 고정 평가참조와 C2 물리 회전 평가를 그대로 사용했다. 새 학습·seed·데이터·추론·PnP·정답 수정·운영 기본값 변경은 없다.
''')
    write(output / 'RECEIPT.json', {
        'source_sha256': sha(__file__), 'template_sha256': sha(template),
        'input_plot_data_sha256': sha(source), 'images': 192,
        'cohort_sizes': {c: len(p) for c, p in populations.items()},
        'default_comparison': ['E0', 'N0'], 'available_baselines': list(ARMS[1:]),
        'independent_y_metrics': list(METRICS), 'fixed_x_indices': True,
        'new_training_or_prediction_or_PnP_or_reference_fit': 0,
        'wall_seconds': time.monotonic()-start,
    })
    write(output / 'SEAL.json', {'assets': {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    return output / 'index.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source, args.output))
