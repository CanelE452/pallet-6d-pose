"""Independent numerical diagnosis of sealed predictions; no fitting or tuning."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(ROOT))
import gzip
import html
import json
from collections import Counter
import hashlib

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager

from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C
from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_joint_visualize_v2 as V

OUT = Path(__file__).resolve().parent
STUDY = OUT.parent.parent
RUN = STUDY / "evaluation_v3"
COHORT = STUDY.parent
TOL = 1e-6
SELECT = (
    "eval_cad:1778653018878592768",
    "eval_night08:1779449501478488320",
    "eval_pallet07:1778652170735118080",
    "wood_183705:000706",
    "eval_night08:1779449488068910592",
    "eval_pallet09:1778653661653195264",
)


def summary(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    return dict(n=len(x), mean=float(x.mean()) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,
                P90=float(np.quantile(x,.9)) if len(x) else None,
                maximum=float(x.max()) if len(x) else None)


def project(xyz, R, t, K):
    camera = xyz @ np.asarray(R).T + np.asarray(t)
    pixel = camera @ np.asarray(K).T
    return pixel[:,:2]/pixel[:,2,None], camera[:,2]


def in_canvas(q, hw, closed=True):
    h,w=hw
    return bool(q[0]>=-TOL and q[1]>=-TOL and q[0]<=w+TOL and q[1]<=h+TOL) if closed else bool(0<=q[0]<w and 0<=q[1]<h)


def geometry(xyz,R,t):
    # Native geometry is the fixed registered axis-aligned cuboid, with 3 faces
    # incident at every vertex. This does not classify physical RGB ownership.
    ray = (-np.asarray(R).T @ np.asarray(t))[None]-xyz
    cosines = np.max(np.sign(xyz)*ray/np.maximum(np.linalg.norm(ray,axis=1)[:,None],1e-12),axis=1)
    return cosines


def domain_check(domain,q,hw):
    if q is None or not np.isfinite(q).all():return None
    checks=[]; radius=None
    if domain['shape']=='ellipse':
        angle=np.deg2rad(domain.get('angle_deg',0)); co,si=np.cos(angle),np.sin(angle)
        basis=np.array([[co,-si],[si,co]])
        radius=float(np.linalg.norm((basis.T@(q-np.asarray(domain['center_xy'])))/np.asarray(domain['half_axes_px'])))
        checks.append(('ellipse',1-radius**2))
    elif domain['shape'] in ('box','axis_aligned_box'):
        b=np.asarray(domain.get('xyxy',domain.get('bounds_xyxy')),float)
        checks.extend(zip(('left','top','right','bottom'),[*((q-b[:2])/(b[2:]-b[:2])),*((b[2:]-q)/(b[2:]-b[:2]))]))
    for i,(a,b,c) in enumerate(domain.get('halfplanes',[])):
        checks.append(('halfplane'+str(i),float((c-a*q[0]-b*q[1])/np.hypot(a,b))))
    if domain['state']=='T':
        h,w=hw; m=domain.get('outside_margin_px',0)
        choices=domain.get('outside_canvas_side')
        choices=list(choices) if isinstance(choices,(tuple,list)) else [choices] if choices in ('left','right','top','bottom') else ['left','right','top','bottom']
        alternatives={'left':-m-q[0],'right':q[0]-(w+m),'top':-m-q[1],'bottom':q[1]-(h+m)}
        checks.append(('outside_canvas_union',float(max(alternatives[s] for s in choices))))
    return dict(point=q.tolist(),contains=bool(all(float(v)>=-TOL for _,v in checks)),
                normalized_ellipse_radius=radius,inequalities=[dict(name=k,slack=float(v)) for k,v in checks])


def measurement_check(m,q,cosine,hw):
    residual=np.asarray(q)-np.asarray(m['xy']); covariance=np.asarray(m['covariance_px2'],float)
    maha=float(np.sqrt(max(0,float(residual@np.linalg.solve(covariance,residual)))))
    return dict(residual_px=float(np.linalg.norm(residual)),mahalanobis=maha,
                within_3sigma=maha<=3*np.sqrt(1+TOL),
                geometric_visibility_pass=bool(cosine+np.sin(np.deg2rad(2))>=-TOL) if cosine is not None else None,
                outward_ray_max_cosine=float(cosine) if cosine is not None else None,
                inside_closed_canvas=in_canvas(q,hw), covariance_is_calibrated_probability=False)


def reference_alignment(row,label):
    truth=label['target']['truth']; mapping=np.asarray(label['metadata']['native_to_reference_corner_ids'],int)
    geom=row['runtime']['initial_geometry'];xyz=np.asarray(geom['native_xyz'],float)
    Q=np.asarray(geom['physical_registration_Q'])
    Qref=np.asarray(truth['body_R']).T@np.asarray(truth['R'])
    source_physical=xyz@Q
    mapped=C.cuboid(*truth['body_xyz'])[mapping]@Qref
    phases=[]
    for k in range(truth['order']):
        angle=k*2*np.pi/truth['order'];co,si=np.cos(angle),np.sin(angle)
        symmetry=np.array([[co,0,si],[0,1,0],[-si,0,co]])
        gap=float(np.linalg.norm(source_physical@symmetry.T-mapped,axis=1).max())
        phases.append((gap,k,symmetry))
    gap,k,sym=min(phases,key=lambda v:v[0])
    compatible=gap<1e-6
    ref,_=project(C.cuboid(*truth['body_xyz']),truth['body_R'],truth['t'],row['K'])
    ref=ref[mapping]
    R=np.asarray(truth['R'])@sym@Q.T
    native,depth=project(xyz,R,truth['t'],row['K'])
    return dict(compatible=compatible,physical_native_gap_m=gap,alignment_phase=k,
                alignment_uses_geometry_only_not_pose_metric=True,
                reference_native_projection=ref,
                registered_model_reference_projection=native if compatible else None,
                registered_reference_R_cf=R if compatible else None,
                registered_reference_centroid=np.asarray(truth['t']) if compatible else None,
                reference_depth=depth if compatible else None,
                mapping_projection_max_gap_px=float(np.linalg.norm(ref-native,axis=1).max()) if compatible else None,
                independent_physical_pose_truth=bool(label['metadata'].get('independent_physical_pose_truth',False)))


def audit_row(row,score,label):
    rt=row['runtime']; states=rt['states'];hw=row['raw_hw'];xyz=np.asarray(rt['initial_geometry']['native_xyz'],float)
    gt=np.asarray(score['native_gt'],float);valid=np.asarray(score['native_gt_valid'],bool);human=score['human_states_native']
    alignment=reference_alignment(row,label); ref=alignment['reference_native_projection']; native=alignment['registered_model_reference_projection']
    ref_cos=geometry(xyz,alignment['registered_reference_R_cf'],alignment['registered_reference_centroid']) if alignment['compatible'] else None
    raw=np.asarray(rt['original_raw_points'],float)[:8]
    final=np.asarray(rt['keypoint_readout']['rigid_model_corners'],float)[:8]
    final_pose=rt['actual_pose'];final_cos=geometry(xyz,final_pose['R_cf'],final_pose['centroid'])
    final_check,depth=project(xyz,final_pose['R_cf'],final_pose['centroid'],row['K'])
    bi=[]
    for m in rt['frozen_visible_measurements']:
        k=m['native_corner'];g=gt[k] if valid[k] else None
        gt_check=measurement_check(m,g,None,hw) if g is not None else None
        ref_check=measurement_check(m,native[k],ref_cos[k],hw) if native is not None else None
        final_support=measurement_check(m,final[k],final_cos[k],hw)
        raw_error=float(np.linalg.norm(raw[k]-g)) if g is not None else None
        corrected_error=float(np.linalg.norm(np.asarray(m['xy'])-g)) if g is not None else None
        bi.append(dict(native_id=k,state=m['state'],human_source_state=human[k],raw_xy=raw[k].tolist(),corrected_xy=m['xy'],source=m.get('source'),
            raw_annotation_error_px=raw_error,corrected_annotation_error_px=corrected_error,
            corrected_minus_raw_error_px=corrected_error-raw_error if g is not None else None,
            direct_annotation_in_canvas=bool(g is not None and human[k]=='DIRECT_VISIBLE' and in_canvas(g,hw,False)),
            strict_compatible_visible_pixel_eligible=bool(g is not None and human[k]=='DIRECT_VISIBLE' and alignment['compatible'] and row['cohort']=='PRIMARY_NONSEVERE'),
            annotation_measurement_budget=gt_check,registered_reference_pose_support=ref_check,final_pose_support=final_support))
    domains=[]
    for d in rt['ot_domains']['domains']:
        k=d['native_corner'];g=gt[k] if valid[k] else None
        domains.append(dict(native_id=k,state=d['state'],human_source_state=human[k],
            annotation=domain_check(d,g,hw),reference_native_projection=domain_check(d,ref[k],hw),
            registered_model_reference_projection=domain_check(d,native[k],hw) if native is not None else None,
            final_pose_projection=domain_check(d,final[k],hw),independent_precise_support=d.get('independent_precise_support',False),
            predicted_domain=d))
    mismatches=dict(false_O_inframe_DIRECT=[k for k in range(8) if states[k]=='O' and human[k]=='DIRECT_VISIBLE' and valid[k] and in_canvas(gt[k],hw,False)],
        O_but_source_visible_actual_T=[k for k in range(8) if states[k]=='O' and human[k]=='DIRECT_VISIBLE' and valid[k] and not in_canvas(gt[k],hw,False)],
        false_S_source_nonself=[k for k in range(8) if states[k]=='S' and human[k] in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED')],
        missed_external_O=[k for k in range(8) if human[k]=='EXTERNAL_OCCLUDED' and states[k]!='O'],
        false_T_annotated_inframe=[k for k in range(8) if states[k]=='T' and valid[k] and human[k] in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED') and in_canvas(gt[k],hw,False)],
        conditional_T_unannotated=[k for k in range(8) if states[k]=='T' and not valid[k]],
        BI_but_human_SELF=[k for k in range(8) if states[k] in ('B','I') and human[k]=='SELF_OCCLUDED'],
        source_visible_actual_T=[k for k in range(8) if human[k]=='DIRECT_VISIBLE' and valid[k] and not in_canvas(gt[k],hw,False)],
        source_DIRECT_dropped=[k for k in range(8) if human[k]=='DIRECT_VISIBLE' and states[k] not in ('B','I')])
    supported=alignment['compatible'] and all(x['registered_reference_pose_support']['within_3sigma'] and x['registered_reference_pose_support']['geometric_visibility_pass'] and x['registered_reference_pose_support']['inside_closed_canvas'] for x in bi)
    reference_ot=alignment['compatible'] and all(d['registered_model_reference_projection']['contains'] for d in domains)
    direct_errors=[float(np.linalg.norm(ref[k]-gt[k])) for k in range(8) if valid[k] and human[k]=='DIRECT_VISIBLE']
    integer=[bool(np.max(np.abs(gt[k]-np.round(gt[k])))<1e-8) for k in range(8) if valid[k] and human[k]=='DIRECT_VISIBLE']
    a=score['metrics']['ORIGINAL_PNP'];b=score['metrics']['FROZEN_BI_ONLY_PNP'];c=score['metrics']['ALL_OT_JOINT_SE3']
    seed=rt['joint_pose'].get('corrected_BI_seed')
    seed_delta=float(np.linalg.norm(np.asarray(seed['t'])-np.asarray(final_pose['centroid']))) if seed and seed.get('t') is not None else None
    hypotheses=[]
    if len(bi)<=3:hypotheses.append('few_visible_anchors_exact_reprojection_does_not_establish_accurate_pose')
    if any(not x['registered_reference_pose_support']['within_3sigma'] for x in bi) if alignment['compatible'] else False:hypotheses.append('derived_reference_pose_excluded_by_BI_covariance_budget')
    if any(not x['registered_model_reference_projection']['contains'] for x in domains) if alignment['compatible'] else False:hypotheses.append('derived_reference_pose_excluded_by_OT_domain')
    if mismatches['false_O_inframe_DIRECT'] or mismatches['false_T_annotated_inframe'] or mismatches['false_S_source_nonself']:hypotheses.append('state_semantic_mismatch')
    if not alignment['compatible']:hypotheses.append('registered_source_model_and_fixed_native_reference_phase_incompatible')
    if any(x['corrected_minus_raw_error_px']>1 for x in bi if x['strict_compatible_visible_pixel_eligible']):hypotheses.append('some_BI_image_correction_harmful_in_pixel_metric')
    return dict(id=row['id'],source_record_id=row['source_record_id'],partition=row['partition'],cohort=row['cohort'],image=row['image'],states=states,
        human_source_states=human,metrics=score['metrics'],baseline_success_to_current_failure=bool(a['success_5cm_5deg'] and not c['success_5cm_5deg']),
        delta_current_minus_original_cm=c['translation_cm']-a['translation_cm'],delta_current_minus_BI_only_cm=c['translation_cm']-b['translation_cm'],
        accepted=rt['accepted'],status=rt['joint_pose']['status'],BI_count=len(bi),OT_count=len(domains),material_ambiguity=rt['ambiguity_audit']['material_ambiguity'],
        alignment=alignment,semantic_mismatches=mismatches,BI_measurements=bi,OT_domains=domains,
        derived_reference_pose_BI_support_all=bool(supported),derived_reference_pose_OT_support_all=bool(reference_ot),
        derived_reference_pose_satisfies_all_current_constraints=bool(supported and reference_ot and alignment['compatible'] and np.min(alignment['reference_depth'])>=1e-6),
        human_DIRECT_annotation_vs_derived_reference_projection_px=summary(direct_errors),human_DIRECT_integer_coordinate_count=sum(integer),human_DIRECT_coordinate_count=len(integer),
        final_pose_rigid_projection_self_check_max_px=float(np.linalg.norm(final-final_check,axis=1).max()),
        final_projection_min_depth_m=float(depth.min()),final_pose_source=final_pose['source'],final_fallback=final_pose['fallback'],
        final_equals_initial_pose=bool(np.linalg.norm(np.asarray(final_pose['R_cf'])-np.asarray(row['initial_pose']['R_cf']))<1e-8 and np.linalg.norm(np.asarray(final_pose['centroid'])-np.asarray(row['initial_pose']['centroid']))<1e-8),
        final_equals_BI_only_pose=not score['joint_changes_BI_only_pose'],final_minus_corrected_BI_seed_translation_m=seed_delta,
        conditional_S_is_initial_model_geometry_not_physical_RGB_certificate=True,S_image_queries=rt['joint_pose'].get('final_model_visibility_audit',{}).get('S_image_queries',0),
        hypotheses=hypotheses,hypotheses_are_associations_not_causal_proof=True)


def render(row,audit,output):
    image=cv2.imread(str(ROOT/row['image']));rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);rt=row['runtime'];states=rt['states'];h,w=image.shape[:2]
    score=SCORING[row['id']];gt=np.asarray(score['native_gt']);valid=score['native_gt_valid'];raw=np.full((8,2),np.nan)
    for k,s in enumerate(states):
        if s!='S':raw[k]=rt['original_raw_points'][k]
    fig,axes=plt.subplots(1,4,figsize=(20,6.2));titles=['Raw non-S queries + postseal annotation','Frozen B/I + all O/T domains','Original coarse PnP projection','Current joint projection / saved fallback']
    if audit['material_ambiguity']:titles[-1]='Possible pose samples · not one precise solution'
    limits=V._roi(image,{'states':states,'original_raw_points':raw})
    for ax,title in zip(axes,titles):
        ax.imshow(rgb);ax.set_xlim(limits[:2]);ax.set_ylim(limits[2:]);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,fontsize=9,loc='left')
        if 'T' in states:ax.set_facecolor('#ffe8e8');ax.add_patch(Rectangle((0,0),w,h,facecolor='none',edgecolor=V.COLORS['T'],lw=1.3))
    for k,s in enumerate(states):
        if s!='S':V._mark(axes[0],raw[k],s,k,marker='x')
    V._domains(axes[1],rt['ot_domains']['domains'])
    for m in rt['frozen_visible_measurements']:
        k=m['native_corner'];V._mark(axes[1],m['xy'],m['state'],k,label_suffix=' fixed')
        if np.isfinite(raw[k]).all():axes[1].annotate('',m['xy'],xytext=raw[k],arrowprops=dict(arrowstyle='->',color='white',lw=1))
    original=np.asarray(rt['initial_geometry']['initial_projection_xy']);current=np.asarray(rt['keypoint_readout']['rigid_model_corners'])
    for ax,p in [(axes[2],original),(axes[3],current)]:
        for a,b in V.EDGES:
            if np.isfinite(p[[a,b]]).all():ax.plot(p[[a,b],0],p[[a,b],1],color='#8dc5fb',lw=.9,alpha=.8)
        for k,s in enumerate(states):
            if s!='U':V._mark(ax,p[k],s,k,marker='s' if s=='S' else 'D',hollow=s=='S',label_suffix=' pose only' if s=='S' else '')
    if audit['material_ambiguity']:
        samples=rt['joint_pose']['pose_family_candidates']
        for sample in samples[:12]:
            p=np.asarray(sample['projected_native'])
            for a,b in V.EDGES:
                if 'U' not in (states[a],states[b]):axes[3].plot(p[[a,b],0],p[[a,b],1],color='#ccc',lw=.7,alpha=.4)
    for n,ax in enumerate(axes):
        for k,p in enumerate(gt):
            if valid[k] and score['human_states_native'][k]!='SELF_OCCLUDED' and (n>=2 or states[k]!='S'):
                ax.scatter(*p,marker='+',s=35,color='#ffff00',lw=1,zorder=8)
    for ax in axes[:2]:ax.text(.015,.02,'S coordinates/queries absent',transform=ax.transAxes,color='white',fontsize=8,bbox=dict(facecolor='#111',alpha=.8,edgecolor='none'))
    for ax,arm in zip(axes[2:],['ORIGINAL_PNP','ALL_OT_JOINT_SE3']):
        m=audit['metrics'][arm];ax.text(.015,.02,'%.2f cm / %.2f deg · success=%s'%(m['translation_cm'],m['rotation_deg'],m['success_5cm_5deg']),transform=ax.transAxes,color='white',fontsize=8,bbox=dict(facecolor='#111',alpha=.8,edgecolor='none'))
    fig.suptitle(row['id']+'  ·  '+row['partition']+' postseal diagnosis\n'+row['image'],fontsize=10,x=.015,ha='left',y=.98)
    flags=audit['semantic_mismatches'];note='States: '+' '.join(states)+f" · B/I={audit['BI_count']} · accepted={audit['accepted']} · native model compatible={audit['alignment']['compatible']}"
    note+='\nFalse O='+str(flags['false_O_inframe_DIRECT'])+'; false conditional T='+str(flags['false_T_annotated_inframe'])+'; source DIRECT lost='+str(flags['source_DIRECT_dropped'])+'. Yellow +: postseal coordinate annotation.'
    note+='\nDerived reference pose passes all current constraints='+str(audit['derived_reference_pose_satisfies_all_current_constraints'])+'. Historical derived reference is not independent physical ground truth.'
    if audit['material_ambiguity']:note+='\n가능한 자세 집합·단일 정밀해 아님 · finite saved search samples, not entire feasible set.'
    fig.text(.015,.04,note,fontsize=9,linespacing=1.6);fig.subplots_adjust(left=.015,right=.995,bottom=.21,top=.86,wspace=.07)
    fig.savefig(output.with_suffix('.png'),dpi=210,bbox_inches='tight');fig.savefig(output.with_suffix('.svg'),bbox_inches='tight');plt.close(fig)


def main():
    global SCORING
    seal=C.read(RUN/'PREDICTION_SEAL.json');assert seal['before_scoring'] and C.sha(RUN/'PREDICTIONS.jsonl.gz')==seal['predictions_sha256']
    score=C.read(RUN/'POSTSEAL_SCORING.json');assert score['prediction_seal_sha256']==C.sha(RUN/'PREDICTION_SEAL.json')
    sources=C.read(RUN/'PROTOCOL.json')['source_sha256'];assert all(C.sha(ROOT/p)==v for p,v in sources.items())
    paths=[COHORT/'OOF_pose_v1/ISOLATED_CAL_SCORING.json',COHORT/'test_reused_v1/ISOLATED_TEST_SCORING.json']
    labels={};labels.update(C.read(paths[0])['rows']);labels.update(C.read(paths[1])['rows'])
    rows=list(C.rows(RUN/'PREDICTIONS.jsonl.gz'));SCORING={r['id']:r for r in score['rows']}
    protocol=dict(schema='POSTSEAL_FAILURE_DIAGNOSIS_PROTOCOL_V1',prediction_cache_sha256=C.sha(RUN/'PREDICTIONS.jsonl.gz'),prediction_seal_sha256=C.sha(RUN/'PREDICTION_SEAL.json'),postseal_score_sha256=C.sha(RUN/'POSTSEAL_SCORING.json'),label_sha256={str(p.relative_to(ROOT)):C.sha(p) for p in paths},source_sha256=sources,diagnostic_source_sha256=C.sha(__file__),no_training_or_tuning=True,no_PnP_fit=True,no_GT_reindex=True,source_model_alignment_uses_fixed_mapping_and_geometry_only=True,frame_selection='all primary baseline-success to current-failure; night OT coverage counterexample; largest TEST source-model-incompatible translation outlier',tolerance=TOL,BI_budget=3,geometry_tolerance_deg=2)
    if (OUT/'FAILURE_AUDIT.json').exists():raise FileExistsError('Preserve completed audit')
    C.write(OUT/'PROTOCOL.json',protocol)
    audited=[audit_row(r,SCORING[r['id']],labels[r['source_record_id']]) for r in rows]
    reports={}
    for part in ('CAL','TEST'):
        group=[a for a in audited if a['partition']==part and a['cohort']=='PRIMARY_NONSEVERE'];comp=[a for a in group if a['alignment']['compatible']]
        pixel=[m for a in group for m in a['BI_measurements'] if m['strict_compatible_visible_pixel_eligible']]
        true_O=[d for a in group for d in a['OT_domains'] if d['state']=='O' and d['human_source_state']=='EXTERNAL_OCCLUDED' and d['annotation'] is not None]
        reports[part]=dict(frames=len(group),source_model_compatible_frames=len(comp),baseline_success_to_current_failure_ids=[a['id'] for a in group if a['baseline_success_to_current_failure']],semantic_mismatch_counts={k:sum(len(a['semantic_mismatches'][k]) for a in group) for k in group[0]['semantic_mismatches']},
            compatible_derived_reference_all_constraint_pass=sum(a['derived_reference_pose_satisfies_all_current_constraints'] for a in comp),compatible_derived_reference_BI_pass=sum(a['derived_reference_pose_BI_support_all'] for a in comp),compatible_derived_reference_OT_pass=sum(a['derived_reference_pose_OT_support_all'] for a in comp),
            strict_visible_raw_error_px=summary([m['raw_annotation_error_px'] for m in pixel]),strict_visible_corrected_error_px=summary([m['corrected_annotation_error_px'] for m in pixel]),
            visible_correction_favourable=sum(m['corrected_minus_raw_error_px']<-.1 for m in pixel),visible_correction_harmful=sum(m['corrected_minus_raw_error_px']>.1 for m in pixel),visible_correction_unchanged=sum(abs(m['corrected_minus_raw_error_px'])<=.1 for m in pixel),
            true_external_O_produced_domain_count=len(true_O),true_external_O_annotation_covered=sum(d['annotation']['contains'] for d in true_O),true_external_O_coverage_rows=[dict(frame=a['id'],native_id=d['native_id'],annotation_contains=d['annotation']['contains'],radius=d['annotation']['normalized_ellipse_radius'],reference_contains=d['reference_native_projection']['contains']) for a in group for d in a['OT_domains'] if d['state']=='O' and d['human_source_state']=='EXTERNAL_OCCLUDED' and d['annotation'] is not None],
            accepted_pose_with_3_or_fewer_BI=sum(a['accepted'] and a['BI_count']<=3 for a in group),derived_reference_not_independent_physical_truth=sum(not a['alignment']['independent_physical_pose_truth'] for a in group))
    C.write(OUT/'FAILURE_AUDIT.json',dict(schema='FRAMEWISE_POSTSEAL_FAILURE_AUDIT_V1',summary=reports,rows=audited,causality='Observed stage association and deterministic constraint exclusion; no controlled causal intervention performed',ground_truth_limit='Historical reference poses derived from annotation; not independent physical metrology. Native human coordinate and reference projection may disagree. No coordinate relabel, reindex, or target fit performed.'))
    font=Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf');font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name();plt.rcParams['axes.unicode_minus']=False
    by={r['id']:r for r in rows};aby={r['id']:r for r in audited};figures=[]
    for fid in SELECT:
        row=by[fid];expected=next(f['image_sha256'] for f in C.read(COHORT/'PREDICTION_INPUTS.json')['frames'] if f['id']==row['source_record_id'])
        assert C.sha(ROOT/row['image'])==expected
        stem=fid.replace(':','__').replace('|','__');render(row,aby[fid],OUT/stem);figures.append(dict(id=fid,png=stem+'.png',svg=stem+'.svg',png_sha256=C.sha(OUT/(stem+'.png'))))
    write_readme(audited,reports,figures)
    assert all(C.sha(ROOT/p)==v for p,v in sources.items()) and C.sha(RUN/'PREDICTIONS.jsonl.gz')==seal['predictions_sha256']
    C.write(OUT/'DIAGNOSIS_SEAL.json',dict(schema='POSTSEAL_DIAGNOSIS_SEAL_V1',rows=len(audited),frozen_inference_sources_unchanged=len(sources),prediction_cache_unchanged=True,scoring_unchanged=C.sha(RUN/'POSTSEAL_SCORING.json')==protocol['postseal_score_sha256'],no_models_or_inference_sources_modified=True,protocol_sha256=C.sha(OUT/'PROTOCOL.json'),audit_sha256=C.sha(OUT/'FAILURE_AUDIT.json'),figures=figures))
    print(json.dumps(reports,ensure_ascii=False))


def write_readme(audited,reports,figures):
    primary=[r for r in audited if r['cohort']=='PRIMARY_NONSEVERE'];reg=[r for r in primary if r['baseline_success_to_current_failure']]
    lines=['# 봉인된 current v3 실패 원인 사후 진단','',
        '모든 예측과 기존 사후 평가가 봉인된 뒤 CAL/재사용 TEST를 열어 독립 산술 검사를 했습니다. 상태·B/I·영역·solver·모델은 수정하지 않았고 새 PnP fit, 학습, TEST 기반 튜닝을 하지 않았습니다. 원인 후보의 동시 발생은 인과 증명이 아닙니다.', '',
        '“제약을 추가하면 좋아진다”는 전제부터 점검했습니다. 고정된 B/I 3σ budget, O/T Ω, 가시성·canvas 제약 안에 실제 표적 자세가 있어야 합니다. 여기서는 사람이 정한 native 좌표와 그로부터 만든 역사적 reference pose를 따로 검사합니다. Reference pose는 독립 물리 ground truth가 아닙니다. Ω의 경험적 반경과 B/I covariance도 확률적으로 보증된 범위가 아닙니다.', '',
        '| 분할 | 프레임 / 모델 대응 호환 | 호환 reference pose가 모든 현재 제약 통과 | BI budget·geometry·canvas 통과 | OT 통과 | true O annotation 영역 포함 |','|---|---:|---:|---:|---:|---:|']
    for part,s in reports.items():lines.append(f"| {part} | {s['frames']} / {s['source_model_compatible_frames']} | {s['compatible_derived_reference_all_constraint_pass']} | {s['compatible_derived_reference_BI_pass']} | {s['compatible_derived_reference_OT_pass']} | {s['true_external_O_annotation_covered']}/{s['true_external_O_produced_domain_count']} |")
    lines+=['','모델 대응이 호환되지 않는 프레임은 고정 native phase와 등록된 물리 모델의 꼭짓점 위치가 맞지 않습니다. 그 프레임의 pose score는 역사적 대칭 평가대로 유지하되, native pixel 개선이나 reference pose가 current native solver에 들어갈 수 있다는 주장은 분리해야 합니다. 대응을 유리하게 바꾸지 않았습니다.','','기존 성공→current 실패의 primary 4건:','','| 프레임 | 원본 / BI-only / current cm | 원본 / BI-only / current deg | 상태 | 핵심 후보 |','|---|---|---|---|---|']
    for r in reg:
        ms=[r['metrics'][a] for a in ('ORIGINAL_PNP','FROZEN_BI_ONLY_PNP','ALL_OT_JOINT_SE3')]
        lines.append('| '+r['id']+' | '+' / '.join('%.2f'%m['translation_cm'] for m in ms)+' | '+' / '.join('%.2f'%m['rotation_deg'] for m in ms)+' | '+' '.join(r['states'])+' | '+', '.join(r['hypotheses'])+' |')
    lines+=['','이 중 최종 자세가 BI-only와 동일한 사례는 OT가 정확도를 떨어뜨렸다고 해석할 수 없습니다. sparse/오차 있는 B/I 관측과 상태에 따라 제외된 DIRECT 점을 먼저 분리해야 합니다. 특히 3개 관측은 P3P가 잔차 0으로 맞춰도 정확한 물리 자세가 보증되지 않습니다. Wood 000706의 OT는 102° BI-only 회전 오류를 약 5.93°로 줄였지만 5° 성공 기준을 넘었습니다.','','B/I 픽셀 보정은 호환 모델의 실제 DIRECT 점만 집계했습니다:','','| 분할 | raw / corrected 평균 px | 유리 / 불리 / 변화 ≤0.1px |','|---|---|---|']
    for part,s in reports.items():lines.append(f"| {part} | {s['strict_visible_raw_error_px']['mean']:.3f} / {s['strict_visible_corrected_error_px']['mean']:.3f} | {s['visible_correction_favourable']} / {s['visible_correction_harmful']} / {s['visible_correction_unchanged']} |")
    lines+=['','점 평균 개선과 특정 sparse pose의 악화를 구분해야 합니다. 보정이 움직이지 않은 cad 사례도 악화했으므로 모든 악화를 보정기 하나의 인과 효과로 돌릴 수 없습니다. 최종 B/I residual이 작은 것은 입력 관측에 맞는다는 뜻이며 semantic corner 또는 역사적 reference pose에 맞는다는 증거는 아닙니다.','','S는 초기 coarse geometry에서 조건부로 제외되었고 이후 영상 query가 없다는 보장은 그대로 통과합니다. 그러나 물리적으로 실제 self-occluded라는 증명은 아니며, human DIRECT/EXTERNAL과 충돌하는 false-S는 별도 기록했습니다. Natural T의 UNANNOTATED target은 독립 실제 절단 label로 단정하지 않았습니다. 실제 crop의 source-visible T4/T5 누락과 false T0/T3도 별도로 남겼습니다.','','Reference 좌표의 정수화 정도와 reference pose 투영 오차를 프레임별 JSON에 기록했습니다. 좌표가 반올림되거나 실제 pallet의 모서리가 이상적 cuboid와 다르거나 고정 phase가 틀리면, 모든 2D 주석과 하나의 rigid pose가 완전히 일치하지 않을 수 있습니다. 이번 검사는 불일치를 측정했으며 그 원인을 독립적으로 확정하지 않았습니다.','','다음 controlled ablation 후보는 상태 유지/교체, raw-BI/corrected-BI, BI covariance budget, raw-centered Ω의 표적 포함, sparse-P3P branch와 가족 representative 정책입니다. 이번 진단 수치로 TEST에 맞춰 모델을 조정하지 않았습니다.','','실제 사진 비교 (S는 첫 두 패널에서 제외, 최종 pose 투영에만 표시):','']
    for f in figures:lines.append(f"- [{f['id']}]({f['png']}) · [SVG]({f['svg']})")
    lines+=['','[프레임별 JSON](FAILURE_AUDIT.json) · [진단 protocol](PROTOCOL.json) · [봉인 및 source 불변 확인](DIAGNOSIS_SEAL.json)']
    (OUT/'README_KO.md').write_text('\n'.join(lines)+'\n')
    html_lines=["<!doctype html><html lang='ko'><meta charset='utf-8'><title>Current v3 postseal failure audit</title><style>body{font:16px sans-serif;margin:24px auto;max-width:1750px;line-height:1.5}img{width:100%}article{margin:40px 0}</style>",'<h1>Current v3 봉인 후 실패 진단</h1><p>모델·추론 변경 없음. 사후 CAL/재사용 TEST 진단이며 인과 증명이나 TEST 튜닝이 아닙니다. 노란 +는 사후 native 주석, 파란 선은 저장된 자세 투영입니다.</p><p><a href="README_KO.md">한국어 보고서</a> · <a href="FAILURE_AUDIT.json">프레임별 JSON</a></p>']
    for f in figures:html_lines.append('<article><h2>'+html.escape(f['id'])+'</h2><a href="'+f['svg']+'">SVG</a><img src="'+f['png']+'"></article>')
    (OUT/'index.html').write_text('\n'.join(html_lines)+'</html>\n')


if __name__=='__main__':main()
