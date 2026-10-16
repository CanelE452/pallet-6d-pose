"""Show three user-pointed blue scatter outliers and their paired baseline images."""
from pathlib import Path
import argparse
import copy
import json
import re
import shutil
import time

import numpy as np

from .comparison_scatter import sha, write
from .gallery import EDGES
from .scatter import COLORS


IDS = ('wood_day_01:005249', 'wood_night_01:031426', 'wood_night_01:031344')
TEMPLATE = Path(__file__).with_name('parallel_gallery.html')


def build(source_root, private_root, output):
    start = time.monotonic()
    source_root, private_root, output = map(Path, (source_root, private_root, output))
    if output.exists():
        raise FileExistsError('Preserve prior image galleries')
    audit = private_root / 'pallet-posefix-gap-audit-20261016-v1'
    gallery = audit / 'parallel_gallery_v2'
    plot_path = audit / 'joint_scatter_v2/PLOT_DATA.json'
    plot = json.loads(plot_path.read_text())
    sealed = json.loads((plot_path.parent / 'SEAL.json').read_text())
    if sealed['assets'][plot_path.name] != sha(plot_path):
        raise ValueError('Original joint scores changed')
    original_gallery = json.loads((gallery / 'DATA.json').read_text())
    frames = json.loads((private_root / 'pallet-guarded-bi-final-20261016-v1/e_inputs/INPUTS.json').read_text())['frames']
    cases = []
    records = []
    for fid in IDS:
        point = next(p for p in plot['points'] if p['id'] == fid)
        case = copy.deepcopy(next(c for c in original_gallery['cases'] if c['id'] == fid))
        frame = next(f for f in frames if f['id'] == fid)
        for arm in ('E0', 'N0'):
            for metric in ('translation_cm', 'rotation_deg'):
                if point['metrics'][arm][metric] != case['metrics'][arm][metric]:
                    raise ValueError('Scatter and actual image gallery must agree')
        raw_path = source_root / frame['image']
        if sha(raw_path) != frame['image_sha256']:
            raise ValueError('Actual original RGB changed')
        xy = np.concatenate([np.asarray(case['methods'][a]['final_xy']) for a in ('N0', 'E0')])
        x0, y0, x1, y1 = case['roi']
        # Include both saved projected boxes, including the day case above its old ROI.
        roi = [max(0, int(np.floor(min(x0, xy[:, 0].min()) - 16))),
               max(0, int(np.floor(min(y0, xy[:, 1].min()) - 16))),
               min(case['raw_hw'][1], int(np.ceil(max(x1, xy[:, 0].max()) + 16))),
               min(case['raw_hw'][0], int(np.ceil(max(y1, xy[:, 1].max()) + 16)))]
        record = {'id': fid, 'original_image_index': point['index'],
                  'original_excluded_gallery_number': case['number'],
                  'original_roi': case['roi'], 'expanded_shared_roi': roi,
                  'source_rgb_sha256': sha(raw_path), 'metrics': point['metrics'],
                  'scatter_x': 'translation_cm', 'scatter_y': 'rotation_deg',
                  'N0_final_xy': case['methods']['N0']['final_xy'],
                  'E0_final_xy': case['methods']['E0']['final_xy'],
                  'all_boxes_are_saved_predictions': True, 'no_GT_overlay': True}
        case['roi'] = roi
        case['methods'] = {a: case['methods'][a] for a in ('N0', 'E0')}
        cases.append(case)
        records.append((record, raw_path))
    output.mkdir(parents=True)
    (output / 'rgb').mkdir()
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    from matplotlib.patches import Rectangle
    from PIL import Image
    font = Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(),
                         'axes.unicode_minus': False})
    for i, ((record, raw_path), case) in enumerate(zip(records, cases), 1):
        rgb = Image.open(raw_path).convert('RGB')
        if rgb.size != (case['raw_hw'][1], case['raw_hw'][0]):
            raise ValueError('Do not rescale points relative to original image')
        raw_copy = output / 'rgb' / f'{i:02d}{raw_path.suffix.lower()}'
        shutil.copy2(raw_path, raw_copy)
        record['rgb_copy'] = str(raw_copy.relative_to(output))
        record['comparison_png'] = f'case_{i:02d}.png'
        fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.8), layout='constrained')
        fig.suptitle(f"{i}. {record['id']} · 원래 이미지 #{record['original_image_index']}\n"
                     '원본 영상과 같은 확대 범위의 최종 예측 · 모든 상자는 저장된 예측이며 GT 아님', fontsize=16)
        axes[0].imshow(rgb)
        x0, y0, x1, y1 = case['roi']
        axes[0].add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False,
                                    edgecolor='#f4c538', lw=2))
        axes[0].set_title('원본 전체 영상 · 노란 사각형 = 오른쪽 확대 범위')
        axes[0].axis('off')
        for ax, arm in zip(axes[1:], ('N0', 'E0')):
            ax.imshow(rgb, interpolation='nearest')
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            xy = np.asarray(case['methods'][arm]['final_xy'])
            for a, b in EDGES:
                ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=COLORS[arm], lw=2)
            m = record['metrics'][arm]
            name = 'N3→SubPix + F (노랑)' if arm == 'N0' else '우리 전체 방법 e (파랑)'
            ax.set_title(f"{name}\n위치 {m['translation_cm']:.2f}cm / 회전 {m['rotation_deg']:.2f}°", fontsize=13)
            ax.tick_params(labelsize=8)
        fig.savefig(output / record['comparison_png'], dpi=140)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 7.5), layout='constrained')
    offsets = {'E0': [(10, -15), (-95, 16), (-150, -25)],
               'N0': [(9, 12), (-125, -30), (-85, 18)]}
    for i, (record, _) in enumerate(records):
        e, n = (record['metrics'][a] for a in ('E0', 'N0'))
        ax.annotate('', (e['translation_cm'], e['rotation_deg']),
                    (n['translation_cm'], n['rotation_deg']),
                    arrowprops={'arrowstyle': '->', 'color': '#8997a4', 'lw': 1.5})
        for arm, marker in [('N0', 'o'), ('E0', 's')]:
            m = record['metrics'][arm]
            ax.scatter(m['translation_cm'], m['rotation_deg'], color=COLORS[arm],
                       marker=marker, s=65, label=('우리 e' if arm == 'E0' else 'N3→SubPix + F') if i == 0 else None)
            ax.annotate(f"{i+1}: {record['id'].split(':')[1]}\n({m['translation_cm']:.2f}, {m['rotation_deg']:.2f})",
                        (m['translation_cm'], m['rotation_deg']),
                        xytext=offsets[arm][i], textcoords='offset points', fontsize=10,
                        color=COLORS[arm])
    ax.set(xlim=(0, 285), ylim=(0, 176), xlabel='위치 오차 (cm)', ylabel='회전 오차 (°)',
           title='지목한 3장의 대응 점 · 화살표는 같은 이미지의 N3→SubPix+F → 우리 e')
    ax.grid(alpha=.2)
    ax.legend(loc='upper right')
    fig.savefig(output / 'SELECTED_SCATTER.png', dpi=150)
    plt.close(fig)

    # Reuse the existing six actual stages, restricted to the selected two methods.
    process = output / 'process'
    process.mkdir()
    derived = copy.deepcopy(original_gallery)
    derived['cases'] = cases
    derived['ordered_ids'] = list(IDS)
    derived['method_order'] = ['N0', 'E0']
    for case in cases:
        urls = [case['image']]
        for method in case['methods'].values():
            urls.extend(method.get('masks', {}).values())
        for url in set(urls):
            if not url:
                continue
            if Path(url).is_absolute() or '..' in Path(url).parts:
                raise ValueError('Only existing local relative assets')
            dest = process / url
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gallery / url, dest)
    template = TEMPLATE.read_text()
    template = template.replace('제외71장 · 방법별 병렬 과정', '지목한3장 · 두 방법의 실제 과정')
    template = template.replace('다섯 방법', '두 방법').replace('repeat(5,minmax(285px,1fr))', 'repeat(2,minmax(420px,1fr))').replace('min-width:1610px', 'min-width:980px')
    template, replaced = re.subn(r'^const METHODS=\[.*\];$',
                                "const METHODS=[['N0','N3 → SubPix + F','#dc8b12'],['E0','우리 전체 방법 e','#136eaa']];",
                                template, count=1, flags=re.M)
    if replaced != 1:
        raise ValueError('Existing gallery template contract changed')
    write(process / 'DATA.json', derived)
    encoded = json.dumps(derived, ensure_ascii=False, allow_nan=False).replace('</', '<\\/')
    write(process / 'index.html', template.replace('__PAYLOAD__', encoded))

    table = '<table><tr><th>이미지</th><th>우리 e 위치/회전</th><th>N3→SubPix+F 위치/회전</th></tr>'
    blocks = ''
    for i, (r, _) in enumerate(records, 1):
        e, n = (r['metrics'][a] for a in ('E0', 'N0'))
        table += f"<tr><td>{i}. {r['id']}</td><td>{e['translation_cm']:.2f}cm / {e['rotation_deg']:.2f}°</td><td>{n['translation_cm']:.2f}cm / {n['rotation_deg']:.2f}°</td></tr>"
        blocks += f"<section id='case{i}'><h2>{i}. {r['id']}</h2><a href='{r['comparison_png']}'><img src='{r['comparison_png']}' alt='{r['id']} 실제 영상과 예측 비교'></a><p><a href='{r['rgb_copy']}'>실제 원본 이미지</a></p></section>"
    table += '</table>'
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>파란 고오차 3장 · 실제 이미지와 대응 비교</title>
<style>body{font-family:sans-serif;margin:24px;color:#183047}main{max-width:1800px;margin:auto}h1{font-size:25px}h2{font-size:21px}p{line-height:1.6}img{max-width:100%;height:auto}table{border-collapse:collapse}th,td{padding:9px 20px;border-bottom:1px solid #dbe3ea;text-align:left}section{margin-top:30px;border-top:1px solid #ced9e4;padding-top:15px}a{color:#1169a5}</style><main>
<h1>산점도에서 지목한 파란 고오차 3장</h1>
<p>파랑은 우리 전체 방법 e, 노랑은 N3→SubPix+F입니다. 아래 화살표는 같은 이미지의 노란 점에서 파란 점으로 향합니다. 두 야간 사례는 N3→SubPix+F도 이미 큰 오차가 있으며 e에서 위치 오차가 더 커졌습니다. 상자는 모두 저장된 최종 예측이며 GT가 아닙니다.</p>
<p><a href="process/index.html">두 방법의 전체 실제 과정 보기: 입력 → 보정 → 상태 → 마스크·선 → 자세 선택 → 최종 자세</a></p>''' + table + '<p><img src="SELECTED_SCATTER.png" alt="동일 이미지의 노란 점과 파란 점 대응"></p>' + blocks + '''
<p>그림은 원래 RGB와 저장된 최종 투영 좌표를 사용했습니다. 같은 확대 범위를 적용했고, 005249의 상단 예측이 잘리지 않도록 기존 ROI를 확장했습니다. 기존 고정 평가참조와 C2 지표를 유지했으며 새 추론·PnP·학습·정답 수정은 없습니다. 원래192장 결과와 기존 갤러리는 보존했습니다.</p></main></html>'''
    write(output / 'index.html', page)
    write(output / 'POINTED_CASES.json', [r for r, _ in records])
    write(output / 'README_KO.md', '# 산점도에서 지목한 파란 고오차 3장\n\n[실제 이미지와 대응 비교](index.html) · [두 방법의 전체 실제 과정](process/index.html)\n\n'
          '선정 기준은 사용자가 지정한 산점도 좌표다: e (약54cm,159°), 약208cm, 약260cm. 각 사례에 같은 이미지의 N3→SubPix+F 결과를 짝지었다. 두 야간 사례는 N0도 이미 큰 실패이므로 e만 실패한 영상으로 해석하지 않는다. '
          '모든 상자는 저장된 예측이며 GT가 아니다. 원래 RGB SHA와 저장 오차를 대조했고, 같은 확대 범위에 두 저장 투영을 그렸다. 005249의 원래 ROI 위로 나간 예측도 보이도록 범위를 확장했다. '
          '기존 실제 6단계 렌더러를 3장·2방법으로 제한해 함께 제공한다. 원래71장 갤러리와192장 평가를 보존했고 새 학습·추론·PnP·정답 수정·운영 기본값 변경은 없다.\n')
    write(output / 'RECEIPT.json', {'source_sha256': sha(__file__), 'template_sha256': sha(TEMPLATE),
                                   'scatter_input_sha256': sha(plot_path), 'gallery_input_sha256': sha(gallery / 'DATA.json'),
                                   'ids': list(IDS), 'methods': ['N0', 'E0'], 'same_expanded_roi': True,
                                   'saved_actual_stages_per_method': 6, 'new_fit_or_inference_or_training': 0,
                                   'wall_seconds': time.monotonic()-start})
    write(output / 'SEAL.json', {'assets': {str(p.relative_to(output)): sha(p)
                                          for p in sorted(output.rglob('*')) if p.is_file()}})
    return output / 'index.html'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'private-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source_root, args.private_root, args.output))
