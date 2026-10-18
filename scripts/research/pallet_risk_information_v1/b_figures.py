"""Post-seal geometry-only B diagnostic panels; no synthetic RGB publication.

Selection rules are fixed in SELECT_RULES before rendering. This module reads
archived diagnostic poses/observations only; no fitting or new cost evaluation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time
import resource
import numpy as np

from .common import read, write, rows, sha, csv_write

FAMILY='ORIGINAL_WD'
SELECT_RULES=dict(
    incompatible='ORIGINAL_WD, B present, every permitted raw-reference phase rejected; maximize minimum rejected-profile count over permitted phases, then minimum SHA256(frame ID)',
    representation='ORIGINAL_WD, phase0 raw finite reject and another permitted phase passes; minimum SHA256(frame ID), then first passing phase',
    score_conflict='same ORIGINAL_WD original fixed G13 package; reference feasible at a permitted phase, failed actual E0 available and same-package feasible, numeric E0 cost strictly lower than minimum feasible reference cost; maximize reference-minus-E0 gap then minimum SHA256(frame ID)',
    no_case='explicit NOT_APPLICABLE, never substitute a different rule',
    geometry='registered renderer reference diagnostic only, raw R unchanged; full off-canvas geometry retained; RGB omitted')


def identity_key(row):return hashlib.sha256(row['id'].encode()).hexdigest()


def family_of(phase):return next(f for f in phase['families'] if f['family']==FAMILY)


def fixed_comparison(family):
    return next(c for c in family['same_objective_comparisons'] if c['objective']=='G13_FIXED_POSE_fixed')


def select_cases(records,tolerance):
    """Use the same-family/stage/objective only; no cross-objective ranking."""
    bad=[];representation=[];conflict=[]
    for row in records:
        phases=row['registered_phases'];families=[family_of(p) for p in phases]
        aa=[f['raw_physical_finite_support'] for f in families]
        if aa and aa[0] and aa[0]['fixed_profile_count']:
            if all(not a['finite_support'] for a in aa):
                unavoidable=min(a['fixed_profile_count']-a['supported_profiles'] for a in aa)
                best=min(range(len(aa)),key=lambda i:(aa[i]['fixed_profile_count']-aa[i]['supported_profiles'],i))
                worst=max(range(len(aa)),key=lambda i:(aa[i]['fixed_profile_count']-aa[i]['supported_profiles'],-i))
                bad.append((unavoidable,row,best,worst))
            if not aa[0]['finite_support']:
                passing=[i for i,a in enumerate(aa) if i and a['finite_support']]
                if passing:representation.append((row,passing[0]))
        if not row['selected_E0_available'] or row['selected_E0_metric_diagnostic_only']['success_5cm5deg']:
            continue
        comparisons=[fixed_comparison(f) for f in families]
        good=[i for i,c in enumerate(comparisons) if c['reference_original_constraints_feasible'] and
              c['cost_comparison_valid'] and c['selected_h_min_slack'] and
              max(c['selected_h_min_slack'])>=-tolerance]
        if not good:continue
        best=min(good,key=lambda i:(comparisons[i]['reference_cost'],i))
        c=comparisons[best];gap=c['reference_cost']-c['selected_E0_cost']
        if gap>0:conflict.append((gap,row,best))
    cases=[]
    if bad:
        n,r,b,w=min(bad,key=lambda x:(-x[0],identity_key(x[1])))
        cases.append(dict(case='incompatible',status='AVAILABLE',row=r,phase_indices=[b],
            selection_value=n,selection_unit='minimum rejected profiles across all permitted phases'))
    else:cases.append(dict(case='incompatible',status='NOT_APPLICABLE'))
    if representation:
        r,k=min(representation,key=lambda x:identity_key(x[0]))
        cases.append(dict(case='representation',status='AVAILABLE',row=r,phase_indices=[0,k],selection_value=None))
    else:cases.append(dict(case='representation',status='NOT_APPLICABLE'))
    if conflict:
        gap,r,k=min(conflict,key=lambda x:(-x[0],identity_key(x[1])))
        cases.append(dict(case='score_conflict',status='AVAILABLE',row=r,phase_indices=[k],
            selection_value=gap,selection_unit='same G13 ORIGINAL_WD fixed objective energy'))
    else:cases.append(dict(case='score_conflict',status='NOT_APPLICABLE'))
    return cases


def _frame_index(cache):
    found={}
    for seal in sorted((Path(cache)/'frames').glob('*/FRAME_SEAL.json')):
        r=read(seal)
        if r['id'] in found:raise ValueError('Duplicate frame ID')
        found[r['id']]=(seal.parent,r)
    return found


def _edges(xyz):
    # Discover actual cuboid adjacency, never invent an index table.
    xyz=np.asarray(xyz)
    return [(a,b) for a in range(len(xyz)) for b in range(a+1,len(xyz))
            if np.count_nonzero(xyz[a]!=xyz[b])==1]


def _selected_projection(row,f,K):
    """Same original R→Rodrigues→R representation, no pose optimization."""
    from scipy.spatial.transform import Rotation
    p=row['selected_E0_pose'];Q=np.asarray(f['physical_registration_Q']);xyz=np.asarray(f['native_xyz'])
    R=np.asarray(p['R_physical'])@Q.T
    center=np.asarray(p['centroid']);t=center-R@xyz.mean(0)
    R=Rotation.from_rotvec(Rotation.from_matrix(R).as_rotvec()).as_matrix()
    scale=max(float(np.linalg.norm(np.ptp(xyz,axis=0))),1e-3)
    t=(t/scale)*scale
    camera=xyz@R.T+t;hom=camera@np.asarray(K).T
    z=hom[:,2];den=np.where(abs(z)<1e-10,np.where(z<0,-1e-10,1e-10),z)
    return hom[:,:2]/den[:,None],camera[:,2]


def _profile_reports(bundle,projected,depth):
    # Rendering uses the original immutable finite predicate only; independent
    # numerical verification remains a separate receipt.
    from scripts.research.pallet_boundary_action_repair_v1 import finite_boundary as finite
    return finite.support_audit(bundle,projected,depth)['profiles']


def panel_records(case,bundle,packet):
    row=case['row'];records=[]
    for index in case['phase_indices']:
        phase=row['registered_phases'][index];f=family_of(phase)
        records.append(dict(name=f'GT raw phase {index}',phase_index=index,family=f,
            q=np.asarray(f['raw_physical_projection_native']),profiles=f['raw_physical_finite_support']['profiles'],
            is_reference=True,representation='unchanged raw renderer R_fixed'))
    if case['case']=='incompatible':
        index=case['phase_indices'][0];f=family_of(row['registered_phases'][index])
        records.append(dict(name=f'GT backend Rodrigues phase {index}',phase_index=index,family=f,
            q=np.asarray(f['reference_projected_native']),profiles=f['finite_support']['profiles'],
            is_reference=True,representation='stored original backend parameterized reference expression'))
    if case['case']=='score_conflict':
        f=family_of(row['registered_phases'][case['phase_indices'][0]])
        q,depth=_selected_projection(row,f,packet['solver_input']['K'])
        records.append(dict(name='Stored actual E0 / inherited Rodrigues expression',phase_index=None,
            family=f,q=q,profiles=_profile_reports(bundle,q,depth),is_reference=False,
            representation='stored selected E0, original parameterization; no new pose solve'))
    return records


def _draw(ax,record,bundle,image_hw,*,full_geometry):
    from matplotlib.patches import Rectangle
    q=record['q'];f=record['family'];H,W=image_hw
    ax.add_patch(Rectangle((0,0),W,H,fill=False,color='.6',linestyle=':',linewidth=1,label='Image canvas (no RGB)'))
    for a,b in _edges(f['native_xyz']):
        ax.plot(q[[a,b],0],q[[a,b],1],color='.55',lw=1,alpha=.65)
    states=bundle.get('original_states',['?']*8)
    for k,point in enumerate(q):
        ax.text(*point,f'{k}:{states[k]}',fontsize=7,color='.15',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    allpoints=[q,np.array([[0.,0.],[W,H]])]
    observations=f['observations']
    for observation in observations['full_points']:
        p=np.asarray(observation['original_observation']['xy'])
        active=observation['likelihood_active']
        ax.scatter(*p,s=18,marker='o',facecolors='#1674b7' if active else 'none',
            edgecolors='#1674b7' if active else '.55',zorder=4)
        allpoints.append(p[None,:])
    for observation in observations['partial']:
        for line in observation['original_observation']['lines']:
            n=np.asarray(line['normal_unit_xy']);t=np.asarray(line['tangent_unit_xy'])
            low,high=line['support_tangent_interval_px'];margin=line['support_extrapolation_px']
            points=np.asarray([-line['offset_px']*n+(low-margin)*t,-line['offset_px']*n+(high+margin)*t])
            ax.plot(points[:,0],points[:,1],color='#00a5a8',lw=1.7);allpoints.append(points)
    queries={p['support_id']:p for g in bundle.get('groups',[]) for p in g['queries'] if p.get('supported')}
    for report in record['profiles']:
        p=queries[report['support_id']];a,b=np.asarray(report['projected_endpoints_xy']);c=np.asarray(p['center_xy'])
        color='#25843b' if report['finite_support'] else '#bd3542'
        ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=1.2,alpha=.65)
        ax.scatter(*c,s=8,color='#de9a20',zorder=3);allpoints.append(c[None,:])
        n=np.asarray(p['outward_normal_xy']);normal=np.asarray([c-10*bundle['pixel_scale']*n,c+10*bundle['pixel_scale']*n])
        ax.plot(normal[:,0],normal[:,1],color='#de9a20',lw=.65,linestyle=':');allpoints.append(normal)
        if report['intersection_xy'] is not None:
            point=np.asarray(report['intersection_xy']);allpoints.append(point[None,:])
            ax.plot([c[0],point[0]],[c[1],point[1]],color='#de9a20',lw=.65,linestyle=':')
            ax.scatter(*point,s=13,marker='x',color=color,zorder=5)
            lam=report['lambda_segment']
            if lam is not None and (lam<0 or lam>1):
                end=a if lam<0 else b
                ax.plot([end[0],point[0]],[end[1],point[1]],color=color,lw=.8,linestyle='--')
    if full_geometry:
        allpoints=np.concatenate(allpoints);allpoints=allpoints[np.isfinite(allpoints).all(1)]
        lo=allpoints.min(0);hi=allpoints.max(0);margin=.06*np.maximum(hi-lo,1.)
        ax.set_xlim(lo[0]-margin[0],hi[0]+margin[0]);ax.set_ylim(hi[1]+margin[1],lo[1]-margin[1])
    else:
        ax.set_xlim(-.04*W,1.04*W);ax.set_ylim(1.04*H,-.04*H)
    statuses=Counter(p['geometry_status'] for p in record['profiles'])
    lam=[p['lambda_segment'] for p in record['profiles'] if p['lambda_segment'] is not None]
    label=('Unclipped geometry view' if full_geometry else 'Image-canvas overview; outside geometry appears below')
    ax.set_title(record['name']+'\n'+label,fontsize=9,pad=11)
    ax.set_xlabel('Original image x [px]');ax.set_ylabel('Original image y [px]')
    ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.12)
    return dict(status_counts=dict(statuses),lambda_min=min(lam) if lam else None,lambda_max=max(lam) if lam else None)


def render(args):
    go=read(args.render_go)
    if (go.get('all_registered_predictions_sealed') is not True or
        go.get('prediction_manifest_sha256')!=sha(args.prediction_manifest)):
        raise ValueError('Explicit root post-seal render GO and immutable prediction manifest required')
    seal=read(args.diagnostic_seal)
    if sha(args.diagnostic_rows)!=seal['rows_sha256']:raise ValueError('B diagnostic rows changed')
    if seal.get('complete') is not True:raise ValueError('Whole1003 diagnostic required')
    cfg=read(args.tolerance_receipt)
    # Permit actual producer schema through a CLI that explicitly binds the
    # frozen tolerance; caller must supply the verified value, no new search.
    tolerance=args.original_tolerance
    expected={str(tolerance):1003}
    if (tolerance<=0 or cfg.get('rows')!=1003 or cfg.get('missing_explicit_values')!=0 or
        cfg.get('unique_values_and_counts')!=expected):
        raise ValueError('Tolerance must equal the actual independent1003 packet receipt')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scripts.research.pallet_boundary_action_repair_v1.finite_boundary import universe
    started,cpu=time.monotonic(),time.process_time()
    data=list(rows(args.diagnostic_rows))
    if len(data)!=1003:raise ValueError('Exact1003 rows required')
    cases=select_cases(data,tolerance);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    index=_frame_index(args.cache);selections=[];profile_rows=[];artifacts={}
    for case in cases:
        name=case['case']
        if case['status']!='AVAILABLE':
            selections.append(dict(case=name,status='NOT_APPLICABLE',selection_rule=SELECT_RULES[name]));continue
        row=case['row'];folder,fs=index[row['id']]
        if sha(folder/'BOUNDARY_BUNDLE.jsonl.gz')!=fs['boundary_bundle_sha256'] or sha(folder/'E_RUN.jsonl.gz')!=fs['e_run_sha256']:
            raise ValueError('Original observation bytes changed')
        bundle=next(rows(folder/'BOUNDARY_BUNDLE.jsonl.gz'));er=next(rows(folder/'E_RUN.jsonl.gz'))
        pp=panel_records(case,bundle,er['packet']);fig,axes=plt.subplots(2,len(pp),figsize=(6.8*len(pp),10.8),squeeze=False)
        numeric=[]
        for j,record in enumerate(pp):
            _draw(axes[0,j],record,bundle,bundle['image_hw'],full_geometry=False)
            numeric.append(_draw(axes[1,j],record,bundle,bundle['image_hw'],full_geometry=True))
            for p in record['profiles']:
                profile_rows.append(dict(case=name,id=row['id'],panel=j,panel_name=record['name'],
                    family=FAMILY,phase_index=record['phase_index'],is_reference=record['is_reference'],
                    representation=record['representation'],support_id=p['support_id'],native_corner=p['native_corner'],
                    native_neighbor=p['native_neighbor'],lambda_segment=p['lambda_segment'],
                    finite_support=p['finite_support'],geometry_status=p['geometry_status'],
                    intersection_x=None if p['intersection_xy'] is None else p['intersection_xy'][0],
                    intersection_y=None if p['intersection_xy'] is None else p['intersection_xy'][1],
                    off_canvas_intersection=p['off_canvas_intersection'],
                    projected_endpoints_xy=json.dumps(p['projected_endpoints_xy']),
                    original_bundle_sha256=fs['boundary_bundle_sha256']))
        f=family_of(row['registered_phases'][case['phase_indices'][0]])
        mapping=f['physical_mapping_audit']
        caption=('Reference diagnostic only; not an inference candidate, material owner UNVERIFIED.\n'
            'Blue circle: active full image observation; hollow: archived/revoked full likelihood; cyan: actual partial tangent interval.\n'
            'Green/red: supported/rejected finite edge; dashed red: extension; orange: original fixed query normal; X: actual intersection.\n'
            'S and all cuboid vertices are model projections, not image observations. Raw lambda and off-canvas points retained in CSV.\n'
            f'R raw→backend max|Δ|={mapping["inherited_Rodrigues_rotation_max_abs"]:.8g}; '
            f'projection max|Δ|={mapping["raw_physical_vs_inherited_projected_xy_max_abs"]:.8g} px; '
            f'original 1e-10 assertion PASS={mapping["original_roundtrip_1e_10_pass"]}.')
        if name=='score_conflict':
            cc=fixed_comparison(f);metric=row['selected_E0_metric_diagnostic_only']
            caption+=(f'\nSame G13 ORIGINAL_WD fixed energy: GT={cc["reference_cost"]:.8g}, '
                f'E0={cc["selected_E0_cost"]:.8g}, GT−E0={case["selection_value"]:.8g}; '
                f'stored E0 error={metric["translation_cm"]:.8g} cm / {metric["rotation_deg"]:.8g} deg.')
        elif name=='incompatible':
            caption+=f'\nMinimum rejected profile count over all registered phases: {case["selection_value"]}.'
        fig.suptitle(f'B {name}: {row["id"]} / {FAMILY} / registered C{row["symmetry_order"]}',fontsize=12,y=.99)
        fig.text(.02,.015,caption,fontsize=8,va='bottom',bbox=dict(facecolor='white',edgecolor='.8'))
        fig.subplots_adjust(left=.06,right=.98,bottom=.19,top=.92,hspace=.40,wspace=.22)
        path=out/(name+'.png');fig.savefig(path,dpi=130,facecolor='white');plt.close(fig)
        artifacts[path.name]=sha(path)
        selections.append(dict(case=name,status='AVAILABLE',id=row['id'],ID_sha256=identity_key(row),
            phase_indices=case['phase_indices'],family=FAMILY,selection_rule=SELECT_RULES[name],
            selection_value=case['selection_value'],selection_unit=case.get('selection_unit'),
            raw_numeric_panel_summaries=numeric,original_bundle_sha256=fs['boundary_bundle_sha256'],
            bundle_universe=universe(bundle),actual_selected_E0_metric=row['selected_E0_metric_diagnostic_only']))
    csv_write(out/'B_CASES.csv',[{k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v
        for k,v in s.items()} for s in selections])
    if profile_rows:csv_write(out/'B_PROFILE_GEOMETRY.csv',profile_rows)
    write(out/'B_CASE_SELECTION.json',dict(selection_rules=SELECT_RULES,cases=selections,
        tolerance_value=tolerance,tolerance_receipt_sha256=sha(args.tolerance_receipt),
        RGB_published=False,reference_diagnostic_not_inference=True,material_owner='UNVERIFIED'))
    for path in out.iterdir():
        if path.is_file():artifacts[path.name]=sha(path)
    write(out/'B_FIGURE_MANIFEST.json',dict(schema='B_REFERENCE_GEOMETRY_FIGURES_V1',complete=True,
        source_sha256=sha(__file__),diagnostic_seal_sha256=sha(args.diagnostic_seal),
        diagnostic_rows_sha256=sha(args.diagnostic_rows),prediction_manifest_sha256=sha(args.prediction_manifest),
        render_go_sha256=sha(args.render_go),artifacts=artifacts,
        wall_seconds=time.monotonic()-started,CPU_seconds=time.process_time()-cpu,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        new_cost_evaluations=0,new_pose_solves=0,new_candidates=0,new_model_forwards=0,
        pixel_review='NOT_PERFORMED_BY_RENDERER',scope='geometry only, synthetic RGB not published'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic-rows','diagnostic-seal','cache','tolerance-receipt','prediction-manifest','render-go','output'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--original-tolerance',type=float,required=True,help='Exact original packet value, bound to the independent receipt; no tuning')
    render(p.parse_args())


if __name__=='__main__':main()
