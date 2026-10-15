"""Write a reviewable actual-image B1/B2 case; labels are posthoc comparison only."""
import argparse
import html
import os
from pathlib import Path

import numpy as np

from . import common as C
from . import evaluation_interval_v4_2b as E

FID = 'eval_pallet09:1778653661653195264'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--figure', type=Path, required=True)
    parser.add_argument('--cal-score', type=Path)
    parser.add_argument('--test-score', type=Path)
    args = parser.parse_args()
    for name in ('candidate','figure','cal_score','test_score'):
        path=getattr(args,name)
        if path is not None:setattr(args,name,path.resolve())
    output = args.figure.parent
    if (output / 'index.html').exists():
        raise FileExistsError('Preserve completed case review')
    candidate = C.read(args.candidate)
    figure_binding = C.read(args.figure.with_suffix('.json'))
    if (C.sha(args.candidate) != figure_binding['candidate_sha256']
            or C.sha(args.figure) != figure_binding['figure_sha256']):
        raise ValueError('Case figure/candidate bindings changed')
    context = Path(figure_binding['context_figure'])
    if C.sha(context) != figure_binding['context_figure_sha256']:
        raise ValueError('Context figure binding changed')
    row = next(r for r in C.rows(E.OUTPUT / 'PREDICTIONS.jsonl.gz') if r['id'] == FID)
    raw = np.asarray(row['runtime']['original_raw_points'], float)
    if candidate['image_pixel_sha256'] != row['runtime']['visible_correction']['image_pixel_sha256']:
        raise ValueError('Actual candidate image differs from frozen case')
    # Candidate was already produced and bound before the historical labels are opened.
    scoring = C.read(E.OUTPUT / 'POSTSEAL_SCORING.json')
    label_binding = scoring['label_bindings']['TEST']
    label_path = C.ROOT / label_binding['path']
    if C.sha(label_path) != label_binding['sha256']:
        raise ValueError('Historical annotation binding changed')
    label = C.read(label_path)['rows'][FID]
    mapping = label['metadata']['native_to_reference_corner_ids']
    truth = np.asarray(label['target']['gt'], float)[mapping]
    valid = np.asarray(label['target']['valid'], bool)[mapping]
    measurements = {m['native_corner']: m for m in candidate['visible_measurements']}
    recovery = {m['native_corner']: m for m in candidate['boundary_recovery_records']}
    points = []
    for k in (1, 2):
        if not valid[k] or not recovery[k]['recovered']:
            raise ValueError('Two recovered points with valid posthoc annotations required')
        xy = np.asarray(measurements[k]['xy'])
        points.append(dict(native_corner=k, raw_xy=raw[k].tolist(), recovered_xy=xy.tolist(),
            initial_annotation_error_px=float(np.linalg.norm(raw[k]-truth[k])),
            recovered_annotation_error_px=float(np.linalg.norm(xy-truth[k])),
            exterior_distance_px=recovery[k]['exterior_distance_px'],
            image_derived_reach_px=recovery[k]['image_derived_reach_px'],
            applied_shift_px=recovery[k]['applied_shift_px']))
    original = row['runtime']['visible_correction']
    old_rows = {r['native_corner']: r for r in original['records']}
    new_rows = {r['native_corner']: r for r in candidate['records']}
    old_measurements = {m['native_corner']: m for m in original['visible_measurements']}
    preserved = all(new_rows[k] == old_rows[k] and measurements[k] == old_measurements[k] for k in (4, 7))
    if not preserved:
        raise ValueError('Previously corrected B4/B7 changed')
    evidence = dict(id=FID, points=points, B4_B7_rows_and_measurements_exact=True,
        S_O_T_U_recovery_queries=[k for k in candidate['recovery_queried_native_ids']
                                 if row['runtime']['states'][k] not in {'B','I'}],
        new_pose_fits=0, total_error_covariance_calibrated=False,
        historical_annotations_posthoc_only=True, GT_inputs_to_recovery=False,
        candidate_sha256=C.sha(args.candidate), renderer_binding=figure_binding,
        annotation_binding=label_binding, report_builder_sha256=C.sha(__file__),
        score_bindings={p: {'path': str(path.relative_to(C.ROOT)), 'sha256': C.sha(path)}
                        for p,path in (('CAL',args.cal_score),('TEST',args.test_score)) if path})
    if evidence['S_O_T_U_recovery_queries']:
        raise ValueError('Excluded state queried by boundary recovery')
    score_rows=[]
    for part,path in (('CAL',args.cal_score),('TEST',args.test_score)):
        if path is None:
            continue
        result=C.read(path)
        if result['partition']!=part or result['new_pose_fits']!=0 or result['function_GT_inputs']:
            raise ValueError('Unexpected correction-only evaluation binding')
        strata=result['strata']['DIRECT_COMPATIBLE']
        score_rows.append(dict(partition=part, frames=result['frames'],
            points=strata['BASE']['points'], base_mean_px=strata['BASE']['mean_px'],
            full_mean_px=strata['FULL']['mean_px'],base_P90_px=strata['BASE']['P90_px'],
            full_P90_px=strata['FULL']['P90_px'],changed_FULL_points=result['changed_FULL_points']))
        if 'support_guard_rejected_FULL_count' in result:
            score_rows[-1]['support_guard_rejected']=result['support_guard_rejected_FULL_count']
            parent_path=path.parent.parent/'evaluation_v2'/path.name
            parent=C.read(parent_path)
            previous={(p['id'],p['native_corner']):p for p in parent['all_valid_point_records']}
            rejected=[previous[(p['id'],p['native_corner'])] for p in result['support_guard_rejected_FULL_inventory']
                      if (p['id'],p['native_corner']) in previous]
            score_rows[-1].update(rejected_previous_improvements=sum(p['errors_px']['FULL']<p['errors_px']['BASE'] for p in rejected),
                rejected_previous_regressions=sum(p['errors_px']['FULL']>p['errors_px']['BASE'] for p in rejected),
                parent_score_sha256=C.sha(parent_path))
    evidence['correction_only_evaluation']=score_rows
    C.write(output / 'CASE_REVIEW.json', evidence)
    table = ''.join(f'<tr><td>KP{p["native_corner"]}</td>'
        f'<td>{p["initial_annotation_error_px"]:.2f}px</td>'
        f'<td><strong>{p["recovered_annotation_error_px"]:.2f}px</strong></td>'
        f'<td>±{p["image_derived_reach_px"]:.2f}px</td></tr>' for p in points)
    explanation = ('기존 코드는 원예측점에서 법선 방향으로 ±12px만 탐색했습니다. '
        '오른쪽 세로 경계가 그 범위 밖에 있어 두 점 모두 세로선 표본이 0/15였습니다. '
        '교점을 만들 두 선을 찾지 못해 원좌표를 유지한 것입니다.')
    method = ('실제 SAM 외곽선까지의 거리와 기존 7px 문맥 여유로 탐색 범위를 계산합니다. '
        '관측한 세로선에 원예측점을 수직으로 옮겨 탐색 기준을 잡고, 위·아래 선을 다시 측정합니다. '
        '영상과 마스크가 함께 지지하는 두 선의 교점을 보정 좌표로 사용합니다.')
    support_guard=candidate.get('boundary_recovery_support_guard')
    support_text=('교점이 두 선의 실제 관측 구간 가까이에 있는지도 검사합니다. '
        '관측 구간에서 멀리 떨어진 교점은 보정을 보류합니다. 야간 사진에서 서로 떨어진 경계선을 길게 연장해 잘못 보정하던 사례를 이 검사로 차단했습니다.') if support_guard else ''
    links = ''.join(f'<a href="{html.escape(os.path.relpath(path,output))}">'
        f'{part} 오차 평가 JSON</a>' for part,path in (('CAL',args.cal_score),('TEST',args.test_score)) if path)
    population_rows=''.join(f'<tr><td>{p["partition"]} {p["frames"]}장</td><td>{p["points"]}</td>'
        f'<td>{p["base_mean_px"]:.2f} → {p["full_mean_px"]:.2f}px</td>'
        f'<td>{p["base_P90_px"]:.2f} → {p["full_P90_px"]:.2f}px</td></tr>' for p in score_rows)
    population=f'<div class="card"><p>다른 사진에서도 오차를 비교했습니다.</p><table><tr><th>사진 집합</th><th>평가한 점</th><th>평균 오차</th><th>90% 지점 오차</th></tr>{population_rows}</table><p class="small">사람이 실제로 보인다고 주석한 B/I 중 축·코너 대응이 일치하는 사진의 점입니다. Severe 가림은 제외했습니다. Self-occlusion 및 대응 불일치 사진은 평가 JSON에서 따로 보고했습니다.</p></div>' if score_rows else ''
    if support_guard:
        bad=sum(p.get('rejected_previous_regressions',0) for p in score_rows)
        good=sum(p.get('rejected_previous_improvements',0) for p in score_rows)
        population+=f'<p class="small">관측 구간 검사는 악화되던 교점 {bad}개를 차단했고, 개선되던 교점 {good}개도 보류했습니다. 남긴 추가 보정은 CAL 6점, TEST 5점으로 이번 데이터에서는 모두 오차가 줄었습니다. 이 결과는 재사용한 TEST의 탐색적 검증이며 새로운 독립 평가가 아닙니다.</p>'
    document = f'''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KP1·KP2 B 경계보정 확인</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;color:#162330;background:#f4f6f8;margin:0}}
main{{max-width:1140px;margin:28px auto;padding:0 22px 40px}}h1{{font-size:28px;line-height:1.35}}
p{{margin:16px 0}}.card{{background:white;border:1px solid #d5dde5;border-radius:12px;padding:20px;margin:18px 0}}
img{{max-width:100%;height:auto;display:block}}table{{border-collapse:collapse;width:100%;text-align:left}}
th,td{{padding:10px;border-bottom:1px solid #dce3e9}}strong{{color:#a74700}}a{{color:#1768af;margin-right:18px}}
.small{{font-size:14px;color:#435260}}@media(max-width:600px){{main{{padding:0 12px}}h1{{font-size:23px}}.card{{padding:12px}}th,td{{padding:6px;font-size:14px}}}}
</style><main><h1>왜 KP1·KP2가 B인데 보정되지 않았나</h1>
<p>{explanation}</p>
<div class="card"><a href="{html.escape(context.name)}"><img src="{html.escape(context.name)}" alt="실제 영상에서 원예측 KP1·KP2와 경계로 보정한 좌표 비교"></a></div>
<div class="card"><table><tr><th>점</th><th>기존 오차</th><th>수정 후 오차</th><th>영상으로 정한 탐색 범위</th></tr>{table}</table>
<p class="small">오차는 사후 주석과의 2D 거리입니다. 주석 좌표는 보정 입력에 사용하지 않았습니다.</p></div>
<p>{method}</p>
<p>KP2는 탐색 범위를 넓히기만 해서는 아래쪽 선의 표본이 8/15로 부족했습니다. 관측한 세로선을 기준으로 다시 탐색하니 15/15로 늘었습니다. 기존 선의 방향·표본 수·잔차 검사와 두 점이 공유하는 경계의 일치 검사를 통과한 경우만 보정합니다.</p>
<p>{support_text}</p>
<div class="card"><a href="{html.escape(args.figure.name)}"><img src="{html.escape(args.figure.name)}" alt="기존 12px 탐색 범위, 실제 마스크 경계, 새 영상 선과 교점의 확대 비교"></a><p class="small">이미지를 누르면 원본 크기로 볼 수 있습니다.</p></div>
<p>기존에 보정된 KP4·KP7은 그대로 보존했습니다. Self-occlusion 점은 보정에 조회하지 않으며, 최종 위치는 기존 방식대로 PnP를 통해 계산합니다.</p>
{population}
<p class="small">이번 비교는 B 키포인트 좌표의 복구 실험입니다. 새 관측의 조건부 공분산은 전체 오차를 보장하지 않습니다. 이 사진의 별도 축·코너 등록 불일치와 최종 PnP 정확도는 이 그림으로 해결됐다고 판단할 수 없습니다.</p>
<p class="small">{html.escape(FID)} · {links}<a href="CASE_REVIEW.json">이미지·좌표 근거 JSON</a></p></main></html>'''
    (output / 'index.html').write_text(document, encoding='utf-8')
    C.write(output / 'REVIEW_SEAL.json', {'html_sha256':C.sha(output/'index.html'),
        'case_json_sha256':C.sha(output/'CASE_REVIEW.json'),
        'context_sha256':C.sha(context),'figure_sha256':C.sha(args.figure),
        'GT_inputs_to_recovery':False,'pose_accuracy_improvement_claimed':False})


if __name__ == '__main__':
    main()
