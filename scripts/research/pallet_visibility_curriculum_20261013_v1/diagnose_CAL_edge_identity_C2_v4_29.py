"""Post-top3 CAL diagnosis: two coherent C2 references, never edgewise choice."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from . import diagnose_CAL_geometry_guided_noise_v4_29 as FIRST

C,D=FIRST.C,FIRST.D
OUTPUT=D.OUTPUT.parent/'CAL_geometry_guided_C2_identity_diagnosis_v4_29'
PLAN=OUTPUT.parent/'CAL_geometry_guided_C2_identity_plan_v4_29/PREDECLARATION.json'


def alternate_reference(xyz,Q,R,t):
    xyz,Q,R,t=map(np.asarray,(xyz,Q,R,t))
    alternate=R@Q@np.diag([-1.,1.,-1.])@Q.T
    center=R@xyz.mean(0)+t
    return alternate,center-alternate@xyz.mean(0)


def line_errors(projected,component):
    k,j=component['native_corner'],component['native_neighbor']
    direction=projected[j]-projected[k];direction=direction/np.linalg.norm(direction)
    normal=np.array([-direction[1],direction[0]])
    n=np.asarray(component['normal_unit_xy'])
    sine=float(np.cross(n,normal))
    return dict(normal_px=float(normal@(np.asarray(component['support_anchor_xy'])-projected[k])),
        angle_deg=float(np.degrees(np.arcsin(np.clip(abs(sine),0.,1.)))))


def declare():
    if PLAN.exists():raise FileExistsError('Preserve post-top3 C2 diagnostic declaration')
    paths=[FIRST.OUTPUT/'DIAGNOSIS_DATA.json',FIRST.OUTPUT/'RENDER_SEAL.json',D.LABELS,
        D.OUTPUT/'CAL_AUTHENTICATED_INPUTS.jsonl.gz',D.OUTPUT/'MODEL_SEAL.json']
    paths.extend(D.OUTPUT/p/'CAL_ERRORS.jsonl.gz'for p in D.M.SOURCE_POLICIES)
    sources={str(Path(__file__).resolve().relative_to(C.ROOT)):C.sha(__file__),
        str(Path(__file__).with_name('test_diagnose_CAL_edge_identity_C2_v4_29.py').resolve().relative_to(C.ROOT)):
        C.sha(Path(__file__).with_name('test_diagnose_CAL_edge_identity_C2_v4_29.py'))}
    PLAN.parent.mkdir(parents=True)
    C.write(PLAN,dict(schema='POST_TOP3_CAL_COHERENT_C2_IDENTITY_DIAGNOSTIC_PLAN_V29',UTC=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources,input_sha256={str(p.relative_to(C.ROOT)):C.sha(p)for p in paths},
        frames='same original fixed-top3-per-policy union; no reselection',
        hypotheses='fixed aligned original reference and Rnative@Q@diag(-1,1,-1)@Q.T, same camera model center',
        comparison='all active physical edges of each policy in each selected frame evaluated under BOTH single-frame reference hypotheses; no per-edge reference choice',
        purpose='diagnose coherent C2/native identity mismatch vs different physical boundary; never new CAL target/model/selector',
        current_reference_models_runtime_and_primary_labels_unchanged=True,new_fits=0))
    print(dict(plan_sha256=C.sha(PLAN)))


def render():
    plan=C.read(PLAN)
    for key in('source_sha256','input_sha256'):D._verify(plan[key])
    if OUTPUT.exists():raise FileExistsError('Preserve C2 diagnosis')
    original=C.read(FIRST.OUTPUT/'DIAGNOSIS_DATA.json')['records'];ids=list(dict.fromkeys(x['id']for x in original))
    prepared={r['id']:r for r in C.rows(D.OUTPUT/'CAL_AUTHENTICATED_INPUTS.jsonl.gz')}
    labels=C.read(D.LABELS)['rows']
    if len(prepared)!=46 or set(labels)!=set(prepared):raise ValueError('Isolated CAL46 only')
    errors={p:{r['id']:r for r in C.rows(D.OUTPUT/p/'CAL_ERRORS.jsonl.gz')}for p in D.M.SOURCE_POLICIES}
    OUTPUT.mkdir(parents=True);results=[]
    for ident in ids:
        geometry=prepared[ident]['initial_geometry'];xyz=np.asarray(geometry['native_xyz']);Q=np.asarray(geometry['physical_registration_Q']);K=np.asarray(geometry['K'])
        alignment=D.OLD.ALIGN._reference_alignment({'runtime':{'initial_geometry':geometry}},labels[ident])
        if not alignment['compatible']:raise ValueError('Fixed compatible CAL alignment')
        R=np.asarray(alignment['R_reference_native']);t=np.asarray(alignment['t_reference']);R2,t2=alternate_reference(xyz,Q,R,t)
        q=[C.project(xyz,r,tr,K)for r,tr in((R,t),(R2,t2))]
        native_alias=(xyz-xyz.mean(0))@Q@np.diag([-1.,1.,-1.])@Q.T+xyz.mean(0)
        distances=np.linalg.norm(native_alias[:,None,:]-xyz[None,:,:],axis=2)
        permutation=distances.argmin(1)
        if distances.min(1).max()>1e-10:raise ValueError('Declared physical C2 must preserve unordered cuboid')
        record=dict(id=ident,C2_native_permutation=permutation.tolist(),same_center=np.allclose(R@xyz.mean(0)+t,R2@xyz.mean(0)+t2,atol=1e-12,rtol=0),
            fixed_alignment_phase=alignment['phase'],policies={})
        for policy in D.M.SOURCE_POLICIES:
            groups=errors[policy][ident]['groups'];edges=[]
            for group in groups:
                component=group['components'][0]['source_component']
                e=[line_errors(projected,component)for projected in q]
                edges.append(dict(native_edge=group['physical_native_edge'],component_id=component['component_id'],reference0=e[0],referenceC2=e[1]))
            summaries=[]
            for key in ('reference0','referenceC2'):
                normal=[abs(e[key]['normal_px'])for e in edges];angle=[e[key]['angle_deg']for e in edges]
                summaries.append(dict(edge_count=len(edges),mean_absolute_normal_px=float(np.mean(normal))if normal else None,
                    maximum_absolute_normal_px=max(normal,default=None),mean_angle_deg=float(np.mean(angle))if angle else None,maximum_angle_deg=max(angle,default=None)))
            record['policies'][policy]=dict(edges=edges,reference0_summary=summaries[0],referenceC2_summary=summaries[1],
                edges_smaller_normal_under_C2=sum(abs(e['referenceC2']['normal_px'])<abs(e['reference0']['normal_px'])for e in edges),
                reference_not_selected_or_changed=True)
        results.append(record)
    C.write(OUTPUT/'COHERENT_C2_DATA.json',dict(records=results,original_top3_unchanged=True,per_edge_reference_choice=False))
    text='# CAL 상위 잔차 프레임의 일관된 C2 가설 진단\n\n첫 진단의 상위3선/세 프레임을 유지한다. 각 정책의 모든 활성선을 **하나의 원래 기준 자세**와 **하나의 C2 자세**에 각각 평가했다. 선마다 더 잘 맞는 기준을 섞거나 CAL 기준·모델을 바꾸지 않았다. 동일 중심/동일 무순서 cuboid이며 native 대응만 달라지는 명시적 두 진단 가설이다.\n\n'
    text+='| 프레임 | 정책 | 선 수 | 원기준 normal 평균/최대(px) | C2 normal 평균/최대(px) | 원기준 각도 평균/최대(°) | C2 각도 평균/최대(°) |\n|---|---|---:|---:|---:|---:|---:|\n'
    for row in results:
        for policy,p in row['policies'].items():
            a,b=p['reference0_summary'],p['referenceC2_summary']
            text+=f'| {row["id"]} | {policy} | {a["edge_count"]} | {a["mean_absolute_normal_px"]:.3f}/{a["maximum_absolute_normal_px"]:.3f} | {b["mean_absolute_normal_px"]:.3f}/{b["maximum_absolute_normal_px"]:.3f} | {a["mean_angle_deg"]:.3f}/{a["maximum_angle_deg"]:.3f} | {b["mean_angle_deg"]:.3f}/{b["maximum_angle_deg"]:.3f} |\n'
    text+='\n[개별 선 전체 잔차](COHERENT_C2_DATA.json)를 함께 보아야 한다. 평균 감소만으로 모든 선의 native identity가 C2 하나로 설명된다는 뜻은 아니다. 이 표는 기준 재선정/오라클 routing/새 적합 근거로 사용하지 않는다.\n'
    (OUTPUT/'REPORT_KO.md').write_text(text)
    for key in('source_sha256','input_sha256'):D._verify(plan[key])
    C.write(OUTPUT/'DIAGNOSIS_SEAL.json',dict(plan_sha256=C.sha(PLAN),source_sha256=plan['source_sha256'],input_sha256=plan['input_sha256'],
        assets={p.name:C.sha(p)for p in OUTPUT.iterdir()if p.is_file()},current_models_runtime_reference_changed=False,new_fits_RGB_NN_queries=0,primary_labels_decoded=0))
    print(dict(records=len(results),seal_sha256=C.sha(OUTPUT/'DIAGNOSIS_SEAL.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('declare','render'));args=parser.parse_args()
    declare()if args.stage=='declare'else render()
