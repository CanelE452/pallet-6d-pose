"""Saved V163 weak observations -> scientific count plots and input-card gallery.

No RGB, crop packets, features, model outputs, weights or training are opened.
"""
from pathlib import Path
from collections import Counter
import csv, hashlib, html, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

V = Path(__file__).resolve().parent.parent
O = Path(__file__).resolve().parent
JOIN = V / 'join_observations/ALL390_REVIEW_BARRIER_AND_SUMMARY.json'
FONT = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
PV = ['VISIBLE_SUPPORTED', 'EXTERNAL_BLOCK_SUPPORTED', 'SELF_HIDE_SUPPORTED',
      'BOUNDARY_OR_GAP_AMBIGUOUS', 'UNKNOWN']
QR = ['ON_INTENDED_PART_SUPPORTED', 'NEAR_INTENDED_PART_SAME_SELECTED_PALLET',
      'OTHER_SELECTED_PALLET_PART', 'FOREIGN_OBJECT', 'BACKGROUND_OR_GAP', 'UNKNOWN']
PL = ['부위 노출 지원', '외부 물체 가림 지원', '자체 구조 숨김 지원', '경계·틈 모호', 'UNKNOWN']
QL = ['의도 부위 위', '인접 같은 팔레트', '선택 팔레트\n다른 부위', '다른 물체', '배경·틈', 'UNKNOWN']
COL = ['#2A9D8F', '#E76F51', '#7661A8', '#E9C46A', '#ADB5BD']
NAMES = ['앞 상단 좌', '앞 상단 우', '앞 하단 우', '앞 하단 좌',
         '뒤 상단 좌', '뒤 상단 우', '뒤 하단 우', '뒤 하단 좌']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p): return {'path': str(p), 'sha256': sha(p)}
def read(p): return json.loads(Path(p).read_text())
def readb(b):
    assert sha(b['path']) == b['sha256']
    return read(b['path'])
def rowsb(b):
    assert sha(b['path']) == b['sha256']
    return [json.loads(x) for x in Path(b['path']).read_text().splitlines()]
def write(p, d):
    assert not p.exists(), f'Preserve previous artifact: {p}'
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
def rel(p): return os.path.relpath(p, O)

def main():
    assert sha(JOIN) == '986b53690ba6c6ccbd91a6dc8cbb04ab914558fa6a1e5f45826372c1f1f75945'
    j = read(JOIN); assert j['completed'] and j['count'] == 390
    rows = rowsb(j['all390']); cross = rowsb(j['cross_witnesses'])
    assert len(rows) == len({r['case_id'] for r in rows}) == 390 and len(cross) == 40
    assert all(r['split'] == 'SOURCE_DEVELOPMENT' and not r['training_admitted'] for r in rows)
    assert all(r['all_new_native_visibility_target'] == -1 and r['all_new_point_quality_target'] == -1 for r in rows)
    evidence_binding = read(V / 'ROOT_ACTUAL_BLINDED_RGB_REVIEW_TRIGGER.json')['all390']
    evidence = {r['case_id']: r for r in rowsb(evidence_binding)}
    cb = {r['case_id']: r for r in cross}
    single = [r for r in rows if r['case_id'] not in cb]
    assert len(single) == 350
    matrix = np.zeros((5, 6), dtype=int)
    native = np.zeros((8, 5), dtype=int)
    for r in rows:
        p = r['primary_observation']; matrix[PV.index(p['native_part_visibility']), QR.index(p['query_relation'])] += 1
        native[r['native_slot'], PV.index(p['native_part_visibility'])] += 1
    assert matrix.sum() == native.sum() == 390
    assert native.sum(1).tolist() == [51, 62, 56, 29, 80, 87, 2, 23]
    assert matrix.sum(1).tolist() == [299, 45, 1, 35, 10]
    assert matrix.sum(0).tolist() == [24, 28, 17, 80, 197, 44]
    assert matrix[0, 4] == 169  # Relation observation, never a coordinate-error count.
    cross_disagreement = {k: [r['case_id'] for r in cross if not r['consensus'][k]['unanimous']]
                          for k in ['selected_instance_correspondence', 'native_part_visibility', 'query_relation', 'part_cover_readiness']}
    assert {k: len(v) for k, v in cross_disagreement.items()} == {
        'selected_instance_correspondence': 0, 'native_part_visibility': 15,
        'query_relation': 8, 'part_cover_readiness': 2}
    font = FontProperties(fname=str(FONT))
    plt.rcParams.update({'font.family': font.get_name(), 'font.size': 12,
                         'axes.unicode_minus': False, 'savefig.facecolor': 'white'})
    # Korean font is registered locally; no network or source image processing.
    from matplotlib import font_manager
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font.get_name()
    fig, ax = plt.subplots(figsize=(15, 8.2), dpi=150)
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=max(matrix.max(), 1), aspect='auto')
    for y in range(5):
        for x in range(6):
            ax.text(x, y, str(matrix[y, x]), ha='center', va='center', fontsize=19,
                    fontweight='bold', color='white' if matrix[y, x] > 85 else '#16324A')
    ax.set_xticks(range(6), [f'{name}\nN={matrix[:, x].sum()}' for x, name in enumerate(QL)])
    ax.set_yticks(range(5), [f'{name}  (N={matrix[y].sum()})' for y, name in enumerate(PL)])
    ax.set_xlabel('원래 q의 관계 관찰 · 위치 변경 없음', labelpad=12)
    ax.set_ylabel('의도된 선택 인스턴스 구조 부위의 가시성 관찰', labelpad=15)
    ax.set_title('부위 가시성 × 원래 q 관계 — 관찰 교차표', fontsize=21, loc='left', pad=25)
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True); ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2); ax.tick_params(which='minor', bottom=False, left=False)
    fig.colorbar(im, ax=ax, fraction=.026, pad=.025, label='primary 관찰 수')
    fig.text(.035, .034, '390개 원래 TRAIN 입력 · 각 사례 primary 1개: 단독 약관찰 350 + 독립 교차관찰 40\n'
             'assistant 약관찰의 교차표이며 정확도·정답 혼동행렬 아님. 교차40의 4개 표를 추가 표본으로 세지 않음.', fontsize=12)
    fig.subplots_adjust(left=.24, right=.93, top=.86, bottom=.22)
    fig.savefig(O / 'PART_VISIBILITY_QUERY_RELATION_OBSERVATION_COUNTS.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(14.5, 8.7), dpi=150)
    left = np.zeros(8)
    for k, (name, color) in enumerate(zip(PL, COL)):
        bars = ax.barh(range(8), native[:, k], left=left, label=name, color=color, edgecolor='white', height=.65)
        for i, n in enumerate(native[:, k]):
            if n >= 3: ax.text(left[i] + n / 2, i, str(n), ha='center', va='center', fontsize=13, fontweight='bold')
        left += native[:, k]
    for i, total in enumerate(left):
        ax.text(total + .7, i, f'N={int(total)}' + (' · 노출 2 / 표본 희소' if i == 6 else ''), va='center', fontsize=12)
    ax.set_yticks(range(8), [f'native{i} · {NAMES[i]}' for i in range(8)])
    ax.invert_yaxis(); ax.set_xlim(0, 113)
    ax.set_xlabel('primary 약관찰 수 · 비율 정규화 없음', labelpad=12)
    ax.set_title('native8별 부위 가시성 관찰과 분모', fontsize=21, loc='left', pad=25)
    ax.grid(axis='x', color='#DCE2E7'); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.1), ncol=3, frameon=False, fontsize=12)
    fig.text(.035, .035, '전체 N=390 · native6은 N=2 · Euro WOOD만 포함, SQUARE 및 다른 WOOD 형상은 없음.\n'
             'native는 카메라 기준 역할이며 고정 실물 꼭짓점 아님. 350개 단독 약관찰과 교차40을 분리해 갤러리에서 확인 가능.', fontsize=12)
    fig.subplots_adjust(left=.18, right=.97, top=.87, bottom=.24)
    fig.savefig(O / 'NATIVE8_PART_VISIBILITY_OBSERVATION_COUNTS.png'); plt.close(fig)
    # Every case retains its original primary vote; disagreement is an additional field, never majority replacement.
    display = []
    for r in rows:
        p = r['primary_observation']; e = evidence[r['case_id']]
        assert all(r[k] == e[k] for k in ['global_index', 'query_id', 'native_slot', 'query_xy', 'L', 'selected_gt_instance_index'])
        card = e['images']['card']; assert Path(card['path']).is_file() and sha(card['path']) == card['sha256']
        c = cb.get(r['case_id'])
        votes = None if c is None else {who: {k: v[k] for k in ['native_part_visibility', 'query_relation', 'reason_KO', 'ambiguity_KO']}
                                          for who, v in c['original_votes'].items()}
        display.append(dict(case_id=r['case_id'], query_id=r['query_id'], native_slot=r['native_slot'], native_name=r['native_name'],
            source_id=r['source_id'], scene=r['source_id'].split(':')[0], query_xy=r['query_xy'], L=r['L'],
            support='CROSS40_FOUR_WAY' if c else 'SINGLE350_WEAK', primary_reviewer=p['reviewer'],
            part=p['native_part_visibility'], q=p['query_relation'], reason_KO=p['reason_KO'], ambiguity_KO=p['ambiguity_KO'],
            readiness=p['part_cover_readiness'], card=rel(card['path']), card_binding=card,
            effective=r['effective_observation'], four_way=votes,
            disagreement_axes=[] if c is None else [k for k, v in c['consensus'].items() if not v['unanimous']]))
    history = None; hist_complete = V / 'historical_label_diagnosis/COMPLETE.json'
    if hist_complete.exists():
        assert sha(hist_complete) == '2f73c7dad115527dedad7566696ac58cfe4d25970577a573ceb2090acd78f48a'
        hd = read(hist_complete); assert hd['completed'] and hd['PASS']
        hs = readb(hd['outputs']['summary'])
        assert hd['outputs']['summary']['sha256'] == '687383fb136008284e6a63a6960a19ecaabae3f3bced0243a51d4f74876b048f'
        cohorts = hs['cohorts']
        decomp = {k:cohorts[k]['historical_TARGET53']['relation_decomposition']
                  for k in ['primary390','mixed350_single_plus_cross40_unanimity']}
        assert list(decomp['primary390'].values()) == [12,1,40]
        assert list(decomp['mixed350_single_plus_cross40_unanimity'].values()) == [12,2,39]
        old_role_part = {k:v['observations']['native_part_visibility']
                        for k,v in cohorts['primary390']['old_role_x_new_observation'].items()}
        assert sum(old_role_part['D'].values()) == 341 and sum(old_role_part['E'].values()) == 49
        history = {'complete': bind(hist_complete), 'summary': hd['outputs']['summary'],
                   'target53_decomposition': decomp, 'primary_old_role_x_part': old_role_part,
                   'cohorts': {k:{'N':v['N'],'old_roles':v['old_roles'],'old_owners':v['old_owners']}
                               for k,v in cohorts.items()}}
    data = dict(completed=True, N=390, single350=350, cross40=40, part_order=PV, q_order=QR,
                observation_cross_table=matrix.tolist(), native_visibility_counts=native.tolist(),
                native_denominators=native.sum(1).tolist(), cross_disagreement_cases=cross_disagreement,
                historical_diagnosis=history, cases=display)
    write(O / 'READOUT_DATA.json', data)
    with (O / 'ALL390_PRIMARY_OBSERVATION_TABLE.csv').open('x') as f:
        w=csv.writer(f); w.writerow(['case_id','native_slot','scene','source_id','support','primary_reviewer','part_visibility','query_relation','cross_disagreement_axes','card_path'])
        for r in display: w.writerow([r[k] for k in ['case_id','native_slot','scene','source_id','support','primary_reviewer','part','q']] + [';'.join(r['disagreement_axes']),r['card']])
    template = (O / 'gallery_template.html').read_text()
    serial = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    (O / 'index.html').write_text(template.replace('__INLINE_DATA__', serial))
    manifest = dict(completed=True, PASS=True, code=bind(__file__), HTML_template=bind(O/'gallery_template.html'),
        source_join=bind(JOIN), all390=j['all390'], cross40_input=j['cross_witnesses'], evidence=evidence_binding,
        historical_diagnosis=None if history is None else {k:v for k,v in history.items() if k!='content'},
        outputs={n:bind(O/n) for n in ['PART_VISIBILITY_QUERY_RELATION_OBSERVATION_COUNTS.png',
            'NATIVE8_PART_VISIBILITY_OBSERVATION_COUNTS.png','READOUT_DATA.json','ALL390_PRIMARY_OBSERVATION_TABLE.csv','index.html']},
        fixed_explanatory_cases=['Q1608','Q0094','Q1213','Q1026'], all390_retained=True, primary_count=390,
        single350=350,cross40=40,cross_votes_not_counted_as_new_cases=True, majority_resolution=False,
        plot_semantics='Assistant weak observation counts; 관찰 교차표, not accuracy or confusion matrix.',
        native6_n=2,SQUARE_examples=0,training_admitted=False,original_labels_modified=False,
        source_RGB_crop_features_weights_models_or_training_read_or_called=False,
        HTML_browser_rendered=False,actual_new_plot_visual_QA_pending=True)
    write(O/'GENERATION_COMPLETE.json',manifest)
    print(json.dumps({'completed':True,'manifest':bind(O/'GENERATION_COMPLETE.json'),'historical_complete_integrated':history is not None}))

if __name__ == '__main__': main()
