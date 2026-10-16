"""Plot each image at (translation error, physical rotation error)."""
from pathlib import Path
import argparse
import json
import time

import numpy as np

from .comparison_scatter import ARMS, COHORT_NAMES, sha, write
from .scatter import COLORS, MARKERS, PLOTLY_MARKERS


def build(source, output):
    start = time.monotonic()
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError('Preserve earlier scatter artifacts')
    seal = json.loads((source.parent / 'SEAL.json').read_text())
    if sha(source) != seal['assets'][source.name]:
        raise ValueError('Input plot data seal mismatch')
    data = json.loads(source.read_text())
    if len(data['points']) != 192 or len({p['id'] for p in data['points']}) != 192:
        raise ValueError('Keep all192 images')
    cohorts = ('FULL192', 'RETAINED121')
    populations = {}
    quadrants = {}
    for cohort, expected in zip(cohorts, (192, 121)):
        ids = set(data['cohorts'][cohort])
        selected = [p for p in data['points'] if p['id'] in ids]
        if len(ids) != expected or len(selected) != expected:
            raise ValueError('Same image populations required')
        populations[cohort] = selected
        quadrants[cohort] = {}
        for arm in ARMS:
            xy = np.array([[p['metrics'][arm]['translation_cm'],
                            p['metrics'][arm]['rotation_deg']] for p in selected])
            if not np.isfinite(xy).all() or (xy < 0).any():
                raise ValueError('Do not omit invalid scores')
            t, r = xy[:, 0] <= 5, xy[:, 1] <= 5
            quadrants[cohort][arm] = {
                'joint_success': int((t & r).sum()),
                'position_only_above5': int((~t & r).sum()),
                'rotation_only_above5': int((t & ~r).sum()),
                'both_above5': int((~t & ~r).sum()),
            }
            if quadrants[cohort][arm]['joint_success'] != data['statistics'][cohort][arm]['success']:
                raise ValueError('Success region must match prior joint scores')
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.patches import Rectangle
    font = Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.4), layout='constrained')
    fig.suptitle('위치 오차 × 회전 오차 · 우리 전체 방법 e vs N3→SubPix + F\n'
                 '방법별 이미지 한 장당 점 하나 · 왼쪽 아래일수록 좋음 · 녹색 영역 = 5cm·5° 공동 성공', fontsize=16)
    for ax, cohort in zip(axes, cohorts):
        selected = populations[cohort]
        ax.add_patch(Rectangle((0, 0), 5, 5, color='#79c698', alpha=.22, zorder=0))
        ax.axvline(5, color='#37865b', ls='--', lw=1, alpha=.7)
        ax.axhline(5, color='#37865b', ls='--', lw=1, alpha=.7)
        for arm in ('E0', 'N0'):
            q = quadrants[cohort][arm]
            ax.scatter([p['metrics'][arm]['translation_cm'] for p in selected],
                       [p['metrics'][arm]['rotation_deg'] for p in selected],
                       color=COLORS[arm], marker=MARKERS[arm], s=34, alpha=.7,
                       linewidths=.7, label=f"{data['names'][arm]} · 성공 {q['joint_success']}/{len(selected)}", zorder=3)
        ax.set(title=COHORT_NAMES[cohort], xlabel='위치 오차 (cm)', ylabel='회전 오차 (°)',
               xlim=(0, None), ylim=(0, None))
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14), fontsize=10.5, framealpha=.95)
        ax.grid(alpha=.2)
    fig.savefig(output / 'position_rotation_e_vs_n0.png', dpi=150)
    plt.close(fig)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=.10,
                        subplot_titles=[COHORT_NAMES[c] for c in cohorts])
    for col, cohort in enumerate(cohorts, 1):
        selected = populations[cohort]
        fig.add_shape(type='rect', x0=0, y0=0, x1=5, y1=5,
                      fillcolor='rgba(80,180,110,.18)', line_width=0,
                      layer='below', row=1, col=col)
        fig.add_vline(x=5, line_color='#37865b', line_dash='dash', line_width=1, row=1, col=col)
        fig.add_hline(y=5, line_color='#37865b', line_dash='dash', line_width=1, row=1, col=col)
        for arm in ARMS:
            fig.add_trace(go.Scatter(
                x=[p['metrics'][arm]['translation_cm'] for p in selected],
                y=[p['metrics'][arm]['rotation_deg'] for p in selected], mode='markers',
                marker=dict(color=COLORS[arm], symbol=PLOTLY_MARKERS[arm], size=8, opacity=.7),
                name=data['names'][arm], legendgroup=arm, showlegend=col == 1,
                visible=True if arm in ('E0', 'N0') else 'legendonly',
                customdata=[[p['id'], p['index'],
                             '공동 성공' if p['metrics'][arm]['translation_cm'] <= 5
                             and p['metrics'][arm]['rotation_deg'] <= 5 else '기준 초과'] for p in selected],
                hovertemplate=(data['names'][arm] + '<br>%{customdata[0]} · 원래 이미지 #%{customdata[1]}'
                               '<br>위치: %{x:.3f}cm<br>회전: %{y:.3f}°<br>%{customdata[2]}<extra></extra>'),
            ), row=1, col=col)
        fig.update_xaxes(title_text='위치 오차 (cm)', rangemode='tozero', row=1, col=col)
        fig.update_yaxes(title_text='회전 오차 (°)', rangemode='tozero', row=1, col=col)
    fig.update_layout(height=740, template='plotly_white', margin=dict(l=70, r=25, t=120, b=70),
                      legend=dict(orientation='h', y=1.16, x=0, groupclick='togglegroup'),
                      hovermode='closest')
    plot = fig.to_html(full_html=False, include_plotlyjs=True, div_id='joint-scatter',
                       config={'responsive': True, 'displaylogo': False,
                               'toImageButtonOptions': {'format': 'png', 'scale': 2}})
    table = '<table><tr><th>방법</th><th>전체192장 성공</th><th>유지121장 성공</th></tr>'
    for arm in ARMS:
        table += f"<tr><td>{data['names'][arm]}</td><td>{quadrants['FULL192'][arm]['joint_success']} / 192</td><td>{quadrants['RETAINED121'][arm]['joint_success']} / 121</td></tr>"
    table += '</table>'
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>위치 × 회전 오차 산점도 · 우리 e 비교</title>
<style>body{font-family:"Nanum Gothic",sans-serif;margin:24px;color:#183047}h1{font-size:24px}p{line-height:1.65}table{border-collapse:collapse}td,th{padding:9px 20px;border-bottom:1px solid #dbe3ea;text-align:left}.note{font-size:13px;color:#526b7e}</style>
<h1>x = 위치 오차(cm), y = 회전 오차(°)</h1>
<p>각 방법에서 이미지 한 장을 (위치 오차, 회전 오차) 한 점으로 표시합니다. <b>왼쪽 아래일수록 좋고, 녹색 영역은 위치 ≤5cm 및 회전 ≤5° 공동 성공</b>입니다.<br>
파랑 사각형은 우리 전체 방법 e, 주황 원은 N3→SubPix+F입니다. 범례에서 PoseFix·최소 B/I도 켜서 비교할 수 있습니다. 마우스를 점에 올리면 이미지 ID와 두 오차가 표시됩니다. 드래그해서 확대하고 두 번 클릭하면 전체 범위로 돌아갑니다.</p>''' + plot + table + '''
<p class="note">왼쪽은 전체192장, 오른쪽은 이전과 동일하게 공통 제외 후 남은121장입니다. 두 패널의 표시 범위는 서로 다릅니다. 오른쪽121장은 N0 유래 IQR 상한(위치18.728cm 또는 회전7.222°)을 다섯 경로에 적용해 공통 제외한 사후 보조 분석이며 전체 결과를 대체하지 않습니다. CAL46은 별도입니다.<br>
기존 고정 평가참조·물리C2 평가를 유지했습니다. 실패와 이전 산점도를 보존했고 새 학습·추론·PnP·정답 수정은 하지 않았습니다.</p></html>'''
    write(output / 'index.html', page)
    write(output / 'PLOT_DATA.json', {'points': data['points'], 'cohorts': data['cohorts'],
                                     'quadrants': quadrants, 'x': 'translation_cm', 'y': 'rotation_deg',
                                     'names': data['names'], 'default_methods': ['E0', 'N0']})
    write(output / 'README_KO.md', '''# 위치 오차 × 회전 오차 산점도

[대화형 산점도](index.html)

x는 위치 오차(cm), y는 물리 C2 회전 오차(°)다. 방법별로 이미지 한 장당 점 하나이며 두 오차를 평균내지 않는다. 왼쪽 아래가 좋고, 0≤x≤5 및 0≤y≤5의 녹색 사각형은 공동 성공 영역이다. PNG와 HTML 왼쪽은 전체192장, 오른쪽은 공통 제외 후121장이다. 패널의 표시 범위는 다르다. HTML 범례에서 PoseFix·최소 B/I도 켤 수 있고, 드래그로 확대할 수 있다.

192/121장 집합과 저장 오차를 그대로 사용했다. 121장은 기존 N0 유래 IQR 상한을 다섯 경로에 공통 적용한 사후 보조 분석이며 전체 결과를 대체하지 않는다. CAL46은 별도다. 모든 실패와 이전 결과를 보존하고 새 학습·추론·PnP·정답 수정·운영 기본값 변경은 하지 않았다.
''')
    write(output / 'RECEIPT.json', {'source_sha256': sha(__file__), 'input_sha256': sha(source),
                                   'x': 'translation_cm', 'y': 'rotation_deg',
                                   'cohort_sizes': {c: len(p) for c, p in populations.items()},
                                   'default_visible_points': 626, 'all_methods_points': 1252,
                                   'new_training_or_inference_or_PnP_or_reference_fit': 0,
                                   'wall_seconds': time.monotonic()-start})
    write(output / 'SEAL.json', {'assets': {p.name: sha(p) for p in output.iterdir() if p.is_file()}})
    return output / 'index.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source, args.output))
