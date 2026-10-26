"""Saved-output development diagnostic; no model, fitting, threshold selection or PnP.

Replay with Python -B and numpy/matplotlib. Packaged sibling owner folders preserve
the same relative layout. All input bindings are checked before reading arrays.
"""
from pathlib import Path
import csv
import hashlib
import html
import json
import platform
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


HERE = Path(__file__).absolute().parent
ROOT = HERE.parent
OUT = HERE / 'heatmap_visibility_diagnostic_v1'


def safe(path):
    path = Path(path)
    if '..' in path.parts or any(s in str(path).lower() for s in ('v112', 'cache', '.pyc')):
        raise ValueError(str(path))
    for ancestor in reversed((path, *path.parents)):
        if ancestor.is_symlink():
            raise ValueError(str(ancestor))
    return path


def digest(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def checked(relative, sha256):
    path = safe(ROOT / relative)
    if digest(path) != sha256:
        raise ValueError('Input binding mismatch: ' + str(path))
    return path


def rank_auc(score, mask, visible):
    d, e = score[mask & visible], score[mask & ~visible]
    if not len(d) or not len(e):
        return None
    pairs = d[:, None] - e[None, :]
    return float(((pairs > 0).sum() + .5 * (pairs == 0).sum()) / pairs.size)


def main():
    source_comp = checked('V263_SPATIAL_HEAD_C/current223_frozen_inference_v1/COMPLETE.json',
                          '8fcfa2761b26e3f2fa236f525bf98267d4964d7a650fef5bceb99b7d2ac3e881')
    eval_comp = checked('V263_CURRENT_EVALUATION_A/actual_evaluation_v1/COMPLETE.json',
                        'fd2b427399c82967005afde092ad74c8f9db5bf327b08676c590dcfa053287c4')
    sc = json.loads(source_comp.read_text())
    ec = json.loads(eval_comp.read_text())
    paths = {
        'heatmaps': checked('V263_SPATIAL_HEAD_C/current223_frozen_inference_v1/JOINT_SELECTED_HEATMAPS.npz',
                            sc['heatmaps']['JOINT_SELECTED']['binding']['sha256']),
        'joined_rows': checked('V263_CURRENT_EVALUATION_A/actual_evaluation_v1/ALL1784_CLASSIFICATION_LOCATION_JOIN.csv',
                               ec['outputs']['joined_rows']['sha256']),
        'support': checked('V263_SPATIAL_HEAD_C/current223_frozen_inference_v1/TOKEN_SUPPORT.npy',
                           sc['outputs']['TOKEN_SUPPORT.npy']['sha256']),
        'source_complete': source_comp,
        'evaluation_complete': eval_comp,
    }
    with np.load(paths['heatmaps'], allow_pickle=False) as data:
        probability = data['probability'].astype(np.float64).reshape(1784, 256)
    support = np.load(paths['support'], allow_pickle=False).reshape(1784, 256).astype(bool)
    with paths['joined_rows'].open(newline='') as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1784
    assert all(int(row['global_index']) == i for i, row in enumerate(rows))
    visible = np.array([r['human_state'] == 'DIRECT_VISIBLE' for r in rows])
    selected = np.array([
        r['primarypairedbool'] == 'True' and r['JOINT_SELECTED__available'] == 'True'
        and r['human_state'] in ('DIRECT_VISIBLE', 'EXTERNAL_OCCLUDED') for r in rows])
    assert selected.sum() == 1237 and (selected & visible).sum() == 1147
    assert np.allclose(probability[selected].sum(1), 1, atol=1e-5)
    assert not np.any(probability[~support] != 0)
    peak = probability.max(1)
    entropy = -(probability * np.log(np.maximum(probability, 1e-300))).sum(1)
    support_n = support.sum(1)
    normalized_entropy = entropy / np.log(np.maximum(support_n, 2))
    feature_scores = {'peak_higher_is_D': peak, 'entropy_lower_is_D': -entropy,
                      'normalized_entropy_lower_is_D': -normalized_entropy}
    stats = []
    for group in ('ALL', 'WOOD_PRIMARY105', 'SQUARE118'):
        group_mask = np.array([group == 'ALL' or r['group'] == group for r in rows])
        for subset, extra in [('PRIMARY_supported', np.ones(1784, bool)),
                              ('PRIMARY_supported_full256', support_n == 256)]:
            mask = selected & group_mask & extra
            item = {'group': group, 'subset': subset, 'D': int((mask & visible).sum()),
                    'E': int((mask & ~visible).sum()),
                    'AUC': {k: rank_auc(v, mask, visible) for k, v in feature_scores.items()}}
            for state, label in [(visible, 'D'), (~visible, 'E')]:
                m = mask & state
                item[label + '_quantiles25_50_75'] = {
                    k: np.quantile(v[m], [.25, .5, .75]).tolist() if m.any() else None
                    for k, v in [('peak', peak), ('entropy_nats', entropy),
                                 ('normalized_entropy', normalized_entropy), ('support_cells', support_n)]}
            stats.append(item)
    report = {
        'schema': 'V263_SAVED_HEATMAP_VISIBILITY_DIAGNOSTIC_V1',
        'sources': {k: {'relative_path': str(v.relative_to(ROOT)), 'sha256': digest(v)}
                    for k, v in paths.items()},
        'replay_code_sha256': digest(Path(__file__)),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                    'matplotlib': matplotlib.__version__},
        'model': 'JOINT_SELECTED', 'selected_rows': int(selected.sum()),
        'label_authority': 'Existing PRIMARY D/E annotations, unchanged; disputed cases retained.',
        'feature_directions': 'Fixed hypotheses: D higher peak, lower entropy; no threshold search.',
        'stats': stats,
        'interpretation': {
            'not_accuracy': True, 'not_fresh_blind_evaluation': True,
            'no_training_or_inference_or_PnP': True, 'no_deployment_rule_selected': True,
            'no_claim_of_statistical_independence_between_points_or_frames': True,
            'current_VIS_reads_uniform_spatial_mean_not_heatmap_weighted_features': True,
            'softmax_peak_exists_for_occluded_crops': True,
            'E_heatmap_has_no_location_target': True,
        },
        'user_visual_feedback': {
            '996': 'Existing E versus predicted D; user sees visible point. Annotation/native correspondence requires review; no automatic relabel.',
            '981': 'Error 1.46 to 3.65px crosses 3px rule; does not establish a large visual or deployment failure.',
            '200': 'Error 1.77 to 4.57px crosses 3px rule; does not establish a large visual or deployment failure.',
        },
    }
    safe(OUT).mkdir(exist_ok=True)
    safe(OUT / 'RESULTS.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with safe(OUT / 'PRIMARY1237_FEATURES.csv').open('w', newline='') as file:
        fields = ['global_index', 'row_id', 'group', 'existing_human_state', 'peak',
                  'entropy_nats', 'normalized_entropy', 'support_cells']
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for i in np.flatnonzero(selected):
            r = rows[i]
            writer.writerow(dict(zip(fields, [i, r['row_id'], r['group'], r['human_state'],
                                             peak[i], entropy[i], normalized_entropy[i], support_n[i]])))
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
    for ax, value, title, feature in [
        (axes[0], peak, '봉우리 높이', 'peak_higher_is_D'),
        (axes[1], entropy, '분포의 퍼짐 · entropy (낮을수록 집중)', 'entropy_lower_is_D')]:
        limits = value[selected]
        bins = np.linspace(limits.min(), limits.max(), 23)
        for label, mask, color in [('보임 D · 1,147점', selected & visible, '#1266a8'),
                                   ('외부 가림 E · 90점', selected & ~visible, '#dd6b20')]:
            ax.hist(value[mask], bins=bins, density=True, histtype='step', linewidth=2,
                    color=color, label=label)
        ax.set_title(title + f"\nAUC {stats[0]['AUC'][feature]:.3f}", fontproperties=font)
        ax.set_ylabel('클래스별 확률 밀도', fontproperties=font)
        ax.legend(prop=font, fontsize=9)
        ax.grid(alpha=.15)
    fig.suptitle('저장 heatmap에 보임/가림 신호는 있지만 두 분포가 겹칩니다', fontproperties=font)
    fig.text(.5, .02, '기존 주석 기준 개발 진단 · AUC는 정확도가 아님 · 학습/임계값 선택 없음',
             ha='center', fontproperties=font, fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .92))
    fig.savefig(safe(OUT / 'DISTRIBUTIONS_KO.png'))
    plt.close(fig)
    body = []
    for row in stats:
        a = row['AUC']
        body.append('<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in [
            row['group'], row['subset'], row['D'], row['E'],
            f"{a['peak_higher_is_D']:.3f}" if a['peak_higher_is_D'] is not None else 'N/A',
            f"{a['entropy_lower_is_D']:.3f}" if a['entropy_lower_is_D'] is not None else 'N/A',
            f"{a['normalized_entropy_lower_is_D']:.3f}" if a['normalized_entropy_lower_is_D'] is not None else 'N/A']) + '</tr>')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Heatmap으로 가림 판단하기 · 저장 출력 진단</title>
<style>body{font:17px/1.7 sans-serif;max-width:1150px;margin:36px auto;padding:0 20px;color:#172431}img{width:100%}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccd5dd;padding:7px}aside{background:#f1f5f9;padding:16px}code{background:#eef2f6}</style>
<h1>Heatmap이 찾은 위치를 보임/외부 가림 판단에 이용할 수 있습니다</h1>
<p>현재 VIS head는 crop 전체 특징의 균등 평균을 읽습니다. 위치 heatmap으로 점 주변의 특징을 골라 읽는 구조는 아직 사용하지 않았습니다.
다음 개선 가설은 <b>heatmap → 위치 주변 특징 + 주변 맥락 → D/E 판단</b>입니다. 이 구조의 성능은 아직 검증하지 않았습니다.</p>
<p>현재 heatmap은 각 crop에서 합이 1인 softmax이며, E에는 위치 heatmap 정답을 주지 않았습니다. 따라서 가려져도 봉우리가 생깁니다.
봉우리 자체를 보임 확률로 사용할 수는 없습니다. 위치를 못 찾음/검색 범위 밖도 외부 가림과 구분해야 합니다.</p>
<img src="DISTRIBUTIONS_KO.png" alt="D와 E heatmap 분포가 겹치는 정도">
<p>기존 PRIMARY 주석의 D 1,147점/E 90점 기준입니다. AUC는 D/E 한 쌍의 순서를 점수가 맞추는 정도이며 분류 정확도나 보증된 성능이 아닙니다.
목재/정사각형 및 256칸 모두 유효한 crop에서도 별도로 집계했습니다. 데이터 일부의 주석·native 대응은 재검토가 필요합니다.</p>
<table><tr><th>그룹</th><th>집계</th><th>D</th><th>E</th><th>높은 peak AUC</th><th>낮은 entropy AUC</th><th>정규화 entropy AUC</th></tr>ROWS</table>
<aside><b>사용자 시각 검토 반영</b><p>G996은 기존 E 주석과 모델 D가 불일치한 사례로, 명백한 모델 실패라고 단정하지 않습니다.
G981과 G200은 각각 1.46→3.65px, 1.77→4.57px로 3px 기준을 넘은 경우입니다. 큰 시각적 실패나 실제 자세 손상을 뜻하지 않습니다.
원 주석과 기존 집계는 바꾸지 않았습니다.</p></aside>
<p>이번 작업: 저장 배열·주석 재집계만 수행. 새 학습, 모델 추론, PnP, 임계값 선택 없음. 반복 확인한 기존 개발 평가 자료이며 독립 검증 결과가 아닙니다.</p>
<p><a href="RESULTS.json">입력 SHA·결과 JSON</a> · <a href="PRIMARY1237_FEATURES.csv">1,237점 수치</a> ·
<a href="../diagnose_saved_heatmap_visibility.py">재현 코드</a> · <a href="../visual_success_failure_v1/INDEX_KO.html">원 사례 갤러리</a></p></html>'''
    safe(OUT / 'INDEX_KO.html').write_text(page.replace('ROWS', ''.join(body)))
    output_files = ['RESULTS.json', 'PRIMARY1237_FEATURES.csv', 'DISTRIBUTIONS_KO.png', 'INDEX_KO.html']
    safe(OUT / 'COMPLETE.json').write_text(json.dumps({
        'completed': True, 'input_SHA_verified': True,
        'outputs': {name: {'sha256': digest(OUT / name), 'size_bytes': (OUT / name).stat().st_size}
                    for name in output_files},
        'new_optimizer_updates': 0, 'new_model_inference': 0, 'PnP': 0,
        'threshold_search': 0, 'annotation_changes': 0,
    }, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': str(OUT), 'stats': stats}, ensure_ascii=False))


if __name__ == '__main__':
    main()
