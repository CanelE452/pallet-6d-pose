"""Interpret sealed fixed-pose forensics; independent dot-product t check.

Reads only the already sealed three-case forensic artifact, not images,
references, model inputs, fitted models or any optimizer.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
from .common import read,sha,write


def check_queries(context):
    differences=[];inside=outside=0
    for q in context['queries']:
        if not q['candidate_geometry_valid']:continue
        a=q['projected_endpoint_a'];b=q['projected_endpoint_b'];p=q['intersection_xy']
        v=[b[i]-a[i] for i in (0,1)];w=[p[i]-a[i] for i in (0,1)]
        t=math.fsum(w[i]*v[i] for i in (0,1))/math.fsum(x*x for x in v)
        differences.append(abs(t-q['segment_parameter_t']))
        finite=0.<=t<=1.;inside+=finite;outside+=not finite
        if finite is not q['inside_finite_segment']:raise ValueError('Independent finite-segment classification differs')
    return dict(valid=len(differences),inside=inside,outside=outside,
                maximum_independent_t_difference=max(differences,default=None))


def run(run_dir,*,forensic_GO=False):
    if forensic_GO is not True:raise PermissionError('Explicit forensic GO required before IO')
    folder=Path(run_dir)/'boundary_extreme_forensics_v1';sp=folder/'FORENSIC_SEAL.json';seal=read(sp)
    path=folder/'BOUNDARY_EXTREME_FORENSICS.json'
    if sha(path)!=seal['artifacts'][path.name]:raise ValueError('Original fixed-pose forensic numeric artifact changed')
    original=read(path);facts=[]
    for c in original['cases']:
        selected=c['poses']['SELECTED'];family=c['selected_family'];mode=c['mode'];score=selected['stored_scores'][family]
        h=score['audits'][mode]['owner']['global_owner_argmax'] if mode=='AB' else 0
        context=selected['finite_edge_by_owner_h'][str(h)];check=check_queries(context)
        fact=dict(id=c['id'],arm=c['arm'],selected_family=family,selected_scoring_h=h,
            stored_owner_responsibility=context.get('stored_owner_responsibility'),finite_segment_check=check,
            optimizer_success=c['optimizer_record']['optimizer_success'],optimizer_status=c['optimizer_record']['optimizer_status'],
            optimizer_feasible=c['optimizer_record']['feasible'],optimizer_minimum_slack=c['optimizer_record']['minimum_constraint_slack'],
            selected_bbox_px=selected['projected_bbox_extent_px'],t_min=context['t_min'],t_max=context['t_max'],
            same_family_same_mode_stored_cost={name:p['stored_scores'][family]['scores'][mode] for name,p in c['poses'].items()},
            selected_fixed_cost_audit=score['audits'][mode])
        if c['AB_optimizer_context']:
            origin=c['AB_optimizer_context'];fact['separate_origin_optimizer_context']=dict(family=origin['family'],
                conditional_h=origin['conditional_optimizer_h'],finite_segment_check=check_queries(origin['segment_diagnosis']))
        facts.append(fact)
    value=dict(schema='FINITE_PHYSICAL_SUPPORT_CONTRACT_VALIDITY_NOTE_V1',complete=True,cases=facts,
        confirmed='All three selected scoring contexts use valid infinite supporting-line intersections outside every finite projected edge segment:0/1,0/2,0/6.',
        classification='finite physical support unguarded / contract validity limitation',
        distinction={'observations':'Original fixed RGB samples, supportIDs, distributions and costs are preserved; no new pixels or boundary observations were manufactured.',
            'linkage':'The model extrapolates candidate supporting lines beyond the finite edge. Those profile intersections cannot certify that the candidate finite physical/model edge traverses the observed support.',
            'explicit_code_contract':'No t-in[0,1] gate was implemented or declared numerically in the frozen config. The forensic classification does not claim an existing numerical t bound was violated.',
            'wording_contract':'METHOD_B and candidate_energy say physical/projected edge. Actual evidence supports an infinite-line approximation, so finite physical support interpretation has a demonstrated validity gap, beyond a mere untested performance claim.',
            'source_material':'Original cuboid-to-material owner was already unverified. The finite-segment counterexample occurs even within the registered cuboid model, before claiming a material match.'},
        causal_scope='These outcome-selected3cases demonstrate this missing validity condition. They do not isolate its contribution across all frames or establish that a new guard/selector would improve performance.',
        baseline_counterexamples='Some original N0/E0 contexts also intersect outside finite segments (see original full numeric artifact); no claim that adding a segment check alone repairs these cases.',
        no_B2_or_selector_or_candidate_change=True,no_new_pose_model_or_optimization_calls=True,
        input_bindings=dict(original_forensic=sha(path),original_forensic_seal=sha(sp)),source_sha256=sha(__file__))
    write(folder/'CONTRACT_VALIDITY_NOTE.json',value)
    lines=['# 유한 physical support의 계약 유효성 한계','',
        '**확인된 분류: finite physical support unguarded / contract validity limitation.** 단순히 효과가 아직 검증되지 않았다는 수준이 아니다. '
        '실제로 선택한 pose의 scoring context에서 모든 지원 normal-line 교차가 finite projected edge 밖이었다. 원 관측·동결모델·예측·선택은 그대로 보존한다.','',
        '| 사례/arm | 실제 선택 family/h | 유한변 내부/유효 교차 | t 범위 | optimizer success / feasible | 같은family 비용: 선택 / N0 / E0 |','|---|---|---:|---|---|---|']
    for c in facts:
        z=c['finite_segment_check'];v=c['same_family_same_mode_stored_cost']
        lines.append(f"| {c['id']} / {c['arm']} | {c['selected_family']} / {c['selected_scoring_h']} | {z['inside']}/{z['valid']} | [{c['t_min']:.6g}, {c['t_max']:.6g}] | {c['optimizer_success']} / {c['optimizer_feasible']} | {v['SELECTED']:.6g} / {v['N0']:.6g} / {v['E0']:.6g} |")
    lines += ['',
        'B_FULL에서는 full:7을 단1개의 supported profile로 대체했다. 남은 partial 비용과 B region 비용은 각각 약1.34e-10·2.30e-9지만 교차 t=-18.4다. '
        '선택 pose가 원 모델의 비용을 거의0으로 설명한다는 사실은 실제 finite edge가 관측 profile에 존재한다는 인증이 되지 않는다.','',
        'EDGE 선택은 iteration limit(status9,80회)로 optimizer_success=False지만 hard-constraint feasible이었다. 원 selector는 finite score와 feasible을 사용하며 optimizer_success를 별도 기록한다. '
        '선택 bbox는 약0.00142×0.00248px로 붕괴했지만 두 지지선 연장의 offset 비용은 낮게 남았다. 이를 수렴한 최적해라고 부르지 않는다.','',
        'CAL AB는 SWAPPED_WD/h3에서 생성됐지만 최종 common-score 선택은 ORIGINAL_WD/h2다. 선택 context의 유일 feasible h2 책임은1이고 finite edge 교차는0/6이다. '
        '생성 SWAPPED/h3를 별도로 검사해도0/6이다. 남은 full4그룹의 none 책임은 거의1이므로 full 위치항은 bounded none 비용으로 포화하며, B의 지지선 연장과 결합해 극단 깊이 pose가 더 낮은 총비용으로 남을 수 있었다. '
        'conditional 생성비용380.772와 최종 owner-mixture 비용383.545를 섞지 않는다.','',
        '**관측을 새로 만들어 낸 것과 관측-모델 연결의 유효성은 구분한다.** 원 RGB profile과 supportID는 실재하며 후보마다 새로 query하거나 삭제하지 않았다. '
        '그러나 후보의 무한 지지선을 유한 변 밖으로 연장해 얻은 교차를 finite physical/model edge의 지지로 해석할 수는 없다. '
        '따라서 관측하지 않은 후보 경계를 계속 이어 설명하는 연결이 허용돼 있었다는 점에서 instruction13.1.5의 유효성 점검에 걸리는 한계다.','',
        '동결 config/원 제약에는 명시적인 t∈[0,1] bound가 없었다. 존재하던 수치 bound를 실행 코드가 무시했다고 주장하는 것은 아니다. '
        '동시에 METHOD_B와 docstring의 physical projected edge라는 표현보다 실제 구현이 보장하는 범위는 약하므로, 이를 단순한 성능 불확실성으로 축소해서도 안 된다. '
        '기존 cuboid/material owner 미인증과 별도로, 등록 cuboid의 유한 변 자체도 이 교차를 지지하지 않는 것이 확인됐다.','',
        '세 사례는 결과를 보고 고른 극단 포렌식이다. 원 N0/E0에도 finite-segment 밖 교차가 있는 context가 있으므로 segment 검사 하나로 전체가 회복된다고 주장하지 않는다. '
        '새 B2·가중치·임계값·후보·선택기·추론·최적화는 만들거나 실행하지 않았다. 모든 원 numeric/seal은 보존했고 이 문서는 별도 해석 기록이다.']
    (folder/'CONTRACT_VALIDITY_NOTE_KO.md').write_text('\n'.join(lines)+'\n')
    write(folder/'CONTRACT_NOTE_SEAL.json',dict(complete=True,input_bindings=value['input_bindings'],source_sha256=sha(__file__),
        artifacts={n:sha(folder/n) for n in ('CONTRACT_VALIDITY_NOTE.json','CONTRACT_VALIDITY_NOTE_KO.md')}))
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--forensic-GO',action='store_true');a=p.parse_args()
    print(run(a.run_dir,forensic_GO=a.forensic_GO)['classification'])


if __name__=='__main__':main()
