"""Read-only CAL46 truncation evidence diagnosis; no pose solves or TEST reads.

Run from the repository root with pallet-yolo26 Python.  Source/models/old
artifacts are protected.  This script reports hypotheses, never adopts a gate.
"""
from __future__ import annotations
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.research.pallet_visibility_curriculum_20261013_v1 import statewise_uncertainty_calibration_v4 as C

ROOT=C.ROOT
F=C.N/'nonsevere_v4_fix'
OUT=F/'truncation_diagnosis_CAL_v1'
CAL=F/'fresh_observations_v2/CAL_OBSERVATIONS.jsonl.gz'
LABEL=C.BASE/'OOF_pose_v1/ISOLATED_CAL_SCORING.json'
MODEL=F/'calibration/final_adapter_CAL46/MODEL.json'
SOURCE=Path(__file__).resolve()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):return {'path':str(Path(p).relative_to(ROOT)),'sha256':sha(p),'bytes':Path(p).stat().st_size}
def write(name,data):
    p=OUT/name
    if p.exists():raise RuntimeError(f'refusing to overwrite {p}')
    p.write_text(json.dumps(C._jsonable(data),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def canvas(q,hw):
    h,w=hw;x,y=np.asarray(q,float)
    side={'left':float(-x),'right':float(x-(w-1)),'top':float(-y),'bottom':float(y-(h-1))}
    closest=np.array([np.clip(x,0,w-1),np.clip(y,0,h-1)])
    return {'inside_closed_canvas':bool(0<=x<=w-1 and 0<=y<=h-1),
            'signed_inside_margin_px':float(-max(side.values())),
            'outside_side_margin_px':{s:v for s,v in side.items() if v>0},
            'euclidean_distance_to_closed_canvas_px':float(np.linalg.norm(np.asarray(q)-closest))}

def pose_from_prior(g,xyz,v):
    R0=np.asarray(g['R_cf'],float);c=xyz.mean(0)
    c0=R0@c+np.asarray(g['centroid'],float)
    R=Rotation.from_rotvec(np.asarray(v)[:3]).as_matrix()@R0
    z=c0[2]*np.exp(v[5])
    center=np.r_[c0[:2]/c0[2]+np.asarray(v)[3:5],1.]*z
    return R,center-R@c

def prior_jacobian(g,xyz,K,k,mu):
    def f(v):
        R,t=pose_from_prior(g,xyz,v)
        return C._project(xyz[[k]],R,t,K)[0]
    eps=1e-6;J=np.empty((2,6))
    for j in range(6):
        dv=np.zeros(6);dv[j]=eps
        J[:,j]=(f(mu+dv)-f(mu-dv))/(2*eps)
    return f(mu),J

def quantile_rank(vals,alpha=.05):
    a=sorted(map(float,vals));rank=math.ceil((len(a)+1)*(1-alpha))
    return {'n':len(a),'one_based_rank':rank,'alpha':alpha,
            'radius_normalized':a[rank-1] if rank<=len(a) else None,
            'infinite_if_rank_exceeds_n':rank>len(a),
            'exchangeable_frame_guarantee_claimed':False}

def line_evidence(rt,k):
    lines=[]
    for r in rt['visible_correction']['records']:
        src=int(r['native_corner'])
        if rt['initial_geometry']['initial_self_hidden'][src]:continue
        if r['state'] not in ('B','I') or not r.get('pixel_queried'):continue
        for ray in r.get('rays',[]):
            if int(ray.get('native_neighbor',-1))!=k:continue
            keep=['edge','endpoints','native_corner','native_neighbor','supported','reason','samples',
                  'transition_sample_count','normal','offset','inlier_count','inlier_fraction',
                  'inlier_span_fraction','inlier_rms_px','angle_change_deg','guide_source','measurement_source']
            lines.append(dict(BI_source_native_id=src,BI_source_state=r['state'],
                              BI_corrected=bool(r.get('corrected')),
                              **{key:ray[key] for key in keep if key in ray}))
    supported=[x for x in lines if x.get('supported') and 'normal' in x and 'offset' in x]
    intersection=None
    if len(supported)>=2:
        A=np.array([x['normal'] for x in supported]);b=np.array([-x['offset'] for x in supported])
        if np.linalg.matrix_rank(A)==2:
            q=np.linalg.lstsq(A,b,rcond=None)[0]
            intersection={'xy':q.tolist(),'condition':float(np.linalg.cond(A)),
                          'line_residual_rms_px':float(np.sqrt(np.mean((A@q-b)**2))),
                          'canvas':canvas(q,rt['initial_geometry']['image_hw']),
                          'independent_physical_T_certificate':False}
    return {'adjacent_BI_ray_count':len(lines),'supported_adjacent_ray_count':len(supported),
            'rays':lines,'supported_line_intersection':intersection,
            'T_corner_RGB_queried':False,
            'limitation':'Real BI image transition samples; edge association/direction is initial-model guided. No calibrated extrapolation uncertainty or independent corner identity certificate.'}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    os=json.loads((F/'fresh_observations_v2/OBSERVATION_SEAL.json').read_text())
    if sha(CAL)!=os['CAL_observations_sha256']:raise RuntimeError('CAL cache seal mismatch')
    modelseal=json.loads((F/'calibration/final_adapter_CAL46/SEAL.json').read_text())
    # Every diagnostic uses only the sealed full-CAL model; never the TEST score.
    assert sha(MODEL)==modelseal['files']['MODEL.json']
    model=json.loads(MODEL.read_text());assert model['fit_partition']=='CAL' and not model['TEST_labels_read']
    prior=np.asarray(model['prior_covariance6']);mu=np.asarray(model['prior_fit']['median_bias6'])
    protected=json.loads((F/'evaluation_bias_v4_1/PROTOCOL.json').read_text())['protected_user_sha256']
    sources=json.loads((F/'evaluation_bias_v4_1/PROTOCOL.json').read_text())['source_sha256']
    before={p:sha(ROOT/p) for p in {**sources,**protected}}
    assert all(before[p]==h for p,h in {**sources,**protected}.items())
    protocol={'schema':'CAL_ONLY_CONDITIONAL_TRUNCATION_DIAGNOSIS_V1',
              'input_bindings':[binding(p) for p in [CAL,LABEL,MODEL,SOURCE,F/'fresh_observations_v2/OBSERVATION_SEAL.json',F/'calibration/final_adapter_CAL46/SEAL.json']],
              'population':'CAL46 only; incompatible native/reference geometry reported separately',
              'projection_population':'every initial nonS native corner, independent of predicted B/I/O/T/U or human visibility',
              'scale':'sealed nonself_prediction_scale_px; no S coordinates',
              'diagnostic_recipes':['full-CAL pooled max raw-to-valid-human DIRECT/EXTERNAL normalized error disk','full-CAL max(frame max normalized corner error) disk','rank-corrected 95% frame-max disk with LOGO descriptive checks'],
              'centers':['initial projection','projection after full-CAL median prior bias'],
              'LOGO':'envelope only; full-CAL upstream heads and optional median/prior model remain conditional inputs',
              'whole_domain_test':'closed disk disjoint from closed canvas iff distance(center,canvas)>radius; otherwise U hypothesis',
              'Jacobian':'optional local projection propagation J covariance6 J.T after CAL mean; not adopted or a coverage certificate',
              'TEST_labels_or_features_read':False,'new_model_inference':False,'new_pose_solves':False,
              'source_or_frozen_artifact_edits':False,'gate_adopted':False,'physical_probability_guarantee':False}
    write('PROTOCOL.json',protocol)
    rows=[json.loads(x) for x in gzip.open(CAL,'rt')]
    labeldoc=json.loads(LABEL.read_text());labels=labeldoc['rows']
    assert len(rows)==46 and len(labels)==46 and labeldoc['TEST_rows']==0 and labeldoc['FINAL_rows']==0
    assert set(r['id'] for r in rows)==set(labels)
    original_domain_hashes={r['id']:hashlib.sha256(json.dumps(r['runtime']['ot_domains'],sort_keys=True,separators=(',',':')).encode()).hexdigest() for r in rows}
    points=[];frames=[];Ts=[];incompatible=[];raw_points=[];phase_audits=[]
    for row in rows:
        assert row['partition']=='CAL' and labels[row['id']]['metadata']['partition']=='CAL'
        rt=row['runtime'];g=rt['initial_geometry'];L=labels[row['id']]
        ids=[i for i,s in enumerate(g['initial_self_hidden']) if not s]
        assert all(rt['states'][i]!='S' for i in ids)
        alignment=C._reference_alignment(row,L);scale=float(rt['ot_domains']['nonself_prediction_scale_px'])
        if not alignment['compatible']:
            incompatible.append({'id':row['id'],'geometry_gap_m':alignment['geometry_gap_m'],'phase_gaps_m':alignment['phase_gaps_m'],'T_ids':[i for i in ids if rt['states'][i]=='T'],'reason':'no native reference projection error fit; annotation/status disclosure retained'})
        cur=[]
        truth=L['target']['truth'];Q=np.asarray(g['physical_registration_Q']);Rtruth=np.asarray(truth['R']);R0=np.asarray(g['R_cf'])
        phases=[]
        for phase in range(int(truth.get('order',1))):
            sym=Rotation.from_rotvec([0,phase*2*np.pi/int(truth.get('order',1)),0]).as_matrix()
            Rnative=Rtruth@sym@Q.T
            phases.append({'phase':phase,'geometry_gap_m':alignment['phase_gaps_m'][phase],
                           'camera_rotation_distance_from_coarse_deg':float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(Rnative@R0.T).as_rotvec())))})
        closest=min(phases,key=lambda p:p['camera_rotation_distance_from_coarse_deg'])['phase']
        phase_audits.append({'id':row['id'],'compatible':alignment['compatible'],'fixed_native_phase':alignment['phase'],
                            'closest_coarse_C2_phase':closest,'closest_equals_fixed':closest==alignment['phase'],'phases':phases,
                            'fixed_native_phase_chosen_from_geometry_only':True,'native_annotation_mapping_changed':False})
        for k in ids:
            initial=np.asarray(g['initial_projection_xy'][k]);raw=np.asarray(rt['original_raw_points'][k])
            p={'id':row['id'],'session':row['session'],'native_id':k,'state':rt['states'][k],
               'initial_role':g['initial_roles'][k],'human_state':L['metadata']['human_states_native'][k],
               'compatible':alignment['compatible'],'scale_px':scale,'initial_xy':initial.tolist(),
               'raw_query_xy':raw.tolist(),'initial_canvas':canvas(initial,g['image_hw']),
               'raw_query_canvas':canvas(raw,g['image_hw'])}
            ref_id=int(alignment['mapping'][k]);p['annotation_valid']=bool(L['target']['valid'][ref_id])
            p['annotation_xy']=L['target']['gt'][ref_id]
            p['annotation_canvas']=canvas(p['annotation_xy'],g['image_hw'])
            if alignment['compatible'] and p['annotation_valid'] and p['human_state'] in ('DIRECT_VISIBLE','EXTERNAL_OCCLUDED'):
                raw_error=raw-np.asarray(p['annotation_xy'])
                p.update(raw_minus_annotation_xy_px=raw_error.tolist(),raw_annotation_error_px=float(np.linalg.norm(raw_error)),
                         raw_annotation_error_normalized=float(np.linalg.norm(raw_error)/scale))
                raw_points.append(p)
            if alignment['compatible']:
                ref=C._project(alignment['xyz'][[k]],alignment['R_reference_native'],alignment['t_reference'],row['K'])[0]
                mean,J=prior_jacobian(g,alignment['xyz'],row['K'],k,mu);V=J@prior@J.T
                error=ref-initial;mean_error=ref-mean
                p.update(reference_xy=ref.tolist(),reference_canvas=canvas(ref,g['image_hw']),
                         reference_minus_initial_xy_px=error.tolist(),error_px=float(np.linalg.norm(error)),
                         error_normalized=float(np.linalg.norm(error)/scale),mean_prior_xy=mean.tolist(),
                         reference_minus_mean_xy_px=mean_error.tolist(),mean_error_normalized=float(np.linalg.norm(mean_error)/scale),
                         covariance_projection_after_CAL_mean_px2=V.tolist(),projection_J_2x6=J.tolist(),
                         mean_canvas=canvas(mean,g['image_hw']),projection_sigma_xy_px=np.sqrt(np.diag(V)).tolist(),
                         mean_reference_mahalanobis2=float(mean_error@np.linalg.pinv(V)@mean_error))
                cur.append(p)
            points.append(p)
            if rt['states'][k]=='T':
                srec=rt['state_observation']['records'][k]
                assert not srec.get('rgb_evaluated')
                Ts.append(dict(p,state_record=srec,adjacent_line_evidence=line_evidence(rt,k)))
        if cur:
            frames.append({'id':row['id'],'session':row['session'],'nonS_count':len(cur),'scale_px':scale,
                           'max_error_normalized':max(p['error_normalized'] for p in cur),
                           'max_mean_error_normalized':max(p['mean_error_normalized'] for p in cur),
                           'native_id_at_max':max(cur,key=lambda p:p['error_normalized'])['native_id']})
    assert len(frames)==44 and len(incompatible)==2 and len(Ts)==13
    raw_max=max(p['raw_annotation_error_normalized'] for p in raw_points)
    raw_classes={}
    for c in ['DIRECT_VISIBLE','EXTERNAL_OCCLUDED']:
        cp=[p for p in raw_points if p['human_state']==c]
        raw_classes[c]={'points':len(cp),'frames':len(set(p['id'] for p in cp)),
                        'max_error_normalized':max(p['raw_annotation_error_normalized'] for p in cp),
                        'argmax':max(cp,key=lambda p:p['raw_annotation_error_normalized'])}
    raw_folds=[]
    for session in sorted(set(p['session'] for p in raw_points)):
        train=[p for p in raw_points if p['session']!=session];held=[p for p in raw_points if p['session']==session]
        rad=max(p['raw_annotation_error_normalized'] for p in train)
        ids=sorted(set(p['id'] for p in held));missed=[p for p in held if p['raw_annotation_error_normalized']>rad]
        raw_folds.append({'heldout_session':session,'train_points':len(train),'heldout_points':len(held),
            'train_ids':sorted(set(p['id'] for p in train)),'heldout_ids':ids,'pooled_max_radius_normalized':rad,
            'covered_points':len(held)-len(missed),'covered_frames':len(ids)-len(set(p['id'] for p in missed)),
            'missed_point_ids':[[p['id'],p['native_id']] for p in missed],
            'per_class_coverage':{c:{'points':sum(p['human_state']==c for p in held),
                'covered_points':sum(p['human_state']==c and p['raw_annotation_error_normalized']<=rad for p in held)}
                for c in ['DIRECT_VISIBLE','EXTERNAL_OCCLUDED']}})
    recipes={}
    for center,key in [('initial','max_error_normalized'),('mean','max_mean_error_normalized')]:
        vals=[f[key] for f in frames];recipes[center]={'max_radius_normalized':max(vals),'rank95':quantile_rank(vals)}
    folds=[]
    for session in sorted(set(f['session'] for f in frames)):
        train=[f for f in frames if f['session']!=session];test=[f for f in frames if f['session']==session]
        fold={'heldout_session':session,'train_ids':[x['id'] for x in train],'heldout_ids':[x['id'] for x in test],'recipes':{}}
        for center,key,pkey in [('initial','max_error_normalized','error_normalized'),('mean','max_mean_error_normalized','mean_error_normalized')]:
            q=quantile_rank([x[key] for x in train]);radii={'max':max(x[key] for x in train),'rank95':q['radius_normalized']}
            pp=[p for p in points if p['compatible'] and p['session']==session]
            fold['recipes'][center]={}
            for name,radius in radii.items():
                misses=[x['id'] for x in test if radius is not None and x[key]>radius]
                fold['recipes'][center][name]={'radius_normalized':radius,'heldout_frames':len(test),
                    'covered_frames':len(test)-len(misses),'missed_frame_ids':misses,
                    'heldout_nonS_points':len(pp),'covered_nonS_points':sum(radius is None or p[pkey]<=radius for p in pp)}
        folds.append(fold)
    for t in Ts:
        t['envelope_hypothesis']={}
        for center,xykey in [('initial','initial_xy'),('mean','mean_prior_xy')]:
            xy=t.get(xykey)
            if xy is None:continue
            dist=canvas(xy,next(r['raw_hw'] for r in rows if r['id']==t['id']))['euclidean_distance_to_closed_canvas_px']
            t['envelope_hypothesis'][center]={}
            for name,r in [('max',recipes[center]['max_radius_normalized']),('rank95',recipes[center]['rank95']['radius_normalized'])]:
                radius=r*t['scale_px'] if r is not None else None
                outside=radius is not None and dist>radius
                t['envelope_hypothesis'][center][name]={'radius_px':radius,'distance_to_canvas_px':dist,
                    'domain_wholly_outside_canvas':outside,'hypothetical_state':'T' if outside else 'U',
                    'physical_truncation_certified':False}
        rawdist=t['raw_query_canvas']['euclidean_distance_to_closed_canvas_px']
        fold=next(x for x in raw_folds if x['heldout_session']==t['session'])
        t['raw_localization_envelope_hypothesis']={}
        for name,r in [('full_CAL_pooled_max',raw_max),('heldout_session_train_pooled_max',fold['pooled_max_radius_normalized'])]:
            radius=r*t['scale_px'];outside=rawdist>radius
            t['raw_localization_envelope_hypothesis'][name]={'radius_normalized':r,'radius_px':radius,
                'raw_center_xy':t['raw_query_xy'],'distance_to_canvas_px':rawdist,'domain_wholly_outside_canvas':outside,
                'hypothetical_state':'T' if outside else 'U','physical_truncation_certified':False}
    crop_protocol=json.loads((F/'fresh_observations_v2/PROTOCOL.json').read_text())
    crops=[]
    for path,expected in crop_protocol['crop_source_sha256'].items():
        if not path.endswith('_PREDICTION_INPUT.json'):continue
        assert sha(ROOT/path)==expected
        d=json.loads((ROOT/path).read_text());assert d['partition']=='CAL'
        tr=d['transform']
        crops.append({'id':d['record_id'],'source_record_id':d['source_record_id'],'binding':binding(ROOT/path),
            'actual_RGB_crop':tr['actual_RGB_crop'],'crop_xyxy':tr['crop_xyxy'],'known_K_transform':True,
            'source_inframe_native_ids':tr.get('source_inframe_native_ids'),
            'source_inframe_evidence':tr.get('source_inframe_evidence'),
            'source_RGB_validated':tr.get('source_RGB_validated'),
            'candidate_T_native_ids':d['candidate_T_native_ids'],
            'physical_T_confirmed':d['physical_T_confirmed'],
            'independent_per_corner_certificate_available':bool(d['physical_T_confirmed'])})
    compatible_points=[p for p in points if p['compatible']]
    result={'schema':'CAL_TRUNCATION_EVIDENCE_DIAGNOSIS_V1','CAL_frames':46,'compatible_frames':44,
            'incompatible_frames':incompatible,'initial_nonS_points':len(points),'compatible_nonS_points':len(compatible_points),
            'T_count':13,'T_frame_count':len(set(t['id'] for t in Ts)),
            'T_independently_certified':sum(bool(t['state_record'].get('physical_truncation_certified')) for t in Ts),
            'T_human_state_counts':dict(collections.Counter(t['human_state'] for t in Ts)),
            'T_reference_inside':sum(t.get('reference_canvas',{}).get('inside_closed_canvas',False) for t in Ts),
            'T_valid_annotation_inside':sum(t['annotation_valid'] and t['annotation_canvas']['inside_closed_canvas'] for t in Ts),
            'T_with_supported_adjacent_BI_ray':sum(t['adjacent_line_evidence']['supported_adjacent_ray_count']>0 for t in Ts),
            'T_with_two_nonparallel_supported_line_intersection':sum(t['adjacent_line_evidence']['supported_line_intersection'] is not None for t in Ts),
            'max_error_frame':max(frames,key=lambda x:x['max_error_normalized']),
            'normalized_frame_error_quantiles':{str(q):float(np.quantile([f['max_error_normalized'] for f in frames],q)) for q in [.5,.8,.9,.95,1.]},
            'recipes_not_adopted':recipes,'LOGO_folds':folds,'available_CAL_real_crop_metadata':crops,
            'raw_localization_envelope':{'fit_population':'native-compatible44; initial nonS; valid human DIRECT+EXTERNAL annotation; no trimming',
                'points':len(raw_points),'frames':len(set(p['id'] for p in raw_points)),
                'pooled_max_radius_normalized':raw_max,'argmax':max(raw_points,key=lambda p:p['raw_annotation_error_normalized']),
                'per_class':raw_classes,'LOGO_folds':raw_folds,
                'full_CAL_T_retained_count':sum(t['raw_localization_envelope_hypothesis']['full_CAL_pooled_max']['domain_wholly_outside_canvas'] for t in Ts),
                'heldout_session_T_retained_count':sum(t['raw_localization_envelope_hypothesis']['heldout_session_train_pooled_max']['domain_wholly_outside_canvas'] for t in Ts)},
            'fixed_native_vs_closest_C2_phase_mismatches':[p for p in phase_audits if p['compatible'] and not p['closest_equals_fixed']],
            'original_state_and_OT_domain_bindings':[{'id':r['id'],'states':r['runtime']['states'],
                'original_domain_sha256':r['runtime']['ot_domains']['domains_sha256'],
                'nonself_prediction_scale_px':r['runtime']['ot_domains']['nonself_prediction_scale_px'],
                'diagnostic_full_RAW_max_hypothesis_states':['U' if s=='T' and not next(t for t in Ts if t['id']==r['id'] and t['native_id']==k)['raw_localization_envelope_hypothesis']['full_CAL_pooled_max']['domain_wholly_outside_canvas'] else s for k,s in enumerate(r['runtime']['states'])]}
                for r in rows],
            'S_image_coordinate_or_query_reads':0,'TEST_files_read':False,'pose_solves':0,
            'gate_adopted':False,'physical_probability_guarantee':False,
            'limits':['CAL4sessions are not independent exchangeable frames; LOGO coverage is descriptive.',
                      'Derived historical pose reference is not independent physical metrology.',
                      'All-CAL median/prior and upstream heads make mean/Jacobian LOGO conditional, not full-pipeline OOF.',
                      'A domain-outside test remains conditional on localization-domain coverage, never an independent T certificate.',
                      'Existing raw OOB and initial projection share the detector/model/pose errors.',
                      'Model-guided image-line association has no fitted corner extrapolation uncertainty.',
                      'All O and genuinely independent-certified T must remain simultaneous hard constraints if retained by producer.']}
    write('ALL_INITIAL_NON_S_POINTS.json',points);write('CAL_FRAME_MAX_ERRORS.json',frames);write('ALL_13_T_AUDIT.json',Ts);write('RAW_ANNOTATION_ERRORS.json',raw_points);write('NATIVE_PHASE_AUDIT.json',phase_audits);write('DIAGNOSIS.json',result)
    plot(rows,points,Ts,frames,recipes,folds)
    readme(result,Ts,folds)
    after={p:sha(ROOT/p) for p in before};assert before==after
    assert sha(CAL)==os['CAL_observations_sha256']
    assert original_domain_hashes=={r['id']:hashlib.sha256(json.dumps(r['runtime']['ot_domains'],sort_keys=True,separators=(',',':')).encode()).hexdigest() for r in rows}
    checks={'CAL_cache_seal_matches':True,'CAL_label_ids_exact46':True,'native_compatible44_incompatible2_disclosed':True,
            'initial_S_coordinate_and_image_query_inputs':0,'all13T_audited':True,'TEST_labels_or_features_read':False,
            'source32_and_protected6_pre_post_unchanged':before==after,'source_input_hashes':before,
            'all_original_states_and_OT_domains_and_nonS_scale_unchanged':True,
            'new_pose_solves':0,'new_gate_adopted':False}
    write('CHECKS.json',checks)
    write('SEAL.json',{'schema':'CAL_TRUNCATION_DIAGNOSIS_SEAL_V1','files':{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},'checks':checks})
    print(json.dumps({k:result[k] for k in ['CAL_frames','compatible_frames','T_count','T_independently_certified','T_human_state_counts','T_reference_inside','T_valid_annotation_inside','T_with_supported_adjacent_BI_ray','T_with_two_nonparallel_supported_line_intersection','recipes_not_adopted']}))
    print(json.dumps({'raw_max':raw_max,'raw_points':len(raw_points),'full_CAL_T_retained':result['raw_localization_envelope']['full_CAL_T_retained_count'],'LOGO_T_retained':result['raw_localization_envelope']['heldout_session_T_retained_count'],'phase_mismatches':result['fixed_native_vs_closest_C2_phase_mismatches']}))

def plot(rows,points,Ts,frames,recipes,folds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    fig,ax=plt.subplots(figsize=(8,5))
    for key,label in [('max_error_normalized','initial'),('max_mean_error_normalized','after CAL median')]:
        x=np.sort([f[key] for f in frames]);ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',label=label)
    ax.set_xscale('log');ax.set_xlabel('Frame maximum nonS projection error / nonS query scale');ax.set_ylabel('CAL44 empirical fraction');ax.grid(alpha=.25);ax.legend();ax.set_title('CAL-only descriptive errors; no predictive coverage guarantee')
    fig.tight_layout();fig.savefig(OUT/'CAL_ERROR_CDF.png',dpi=180);fig.savefig(OUT/'CAL_ERROR_CDF.svg');plt.close(fig)
    for row in rows:
        ts=[t for t in Ts if t['id']==row['id']]
        if not ts:continue
        assert sha(ROOT/row['image'])==row['image_sha256']
        img=np.asarray(Image.open(ROOT/row['image']).convert('RGB'));h,w=img.shape[:2]
        fig,axs=plt.subplots(1,2,figsize=(14,6));ids=[p for p in points if p['id']==row['id']]
        for ax in axs:ax.imshow(img);ax.set_xlim(-30,w+70);ax.set_ylim(h+55,-30);ax.axvline(0,c='.5',lw=.5);ax.axvline(w-1,c='.5',lw=.5);ax.axhline(0,c='.5',lw=.5);ax.axhline(h-1,c='.5',lw=.5)
        for p in ids:
            k=p['native_id'];color='tab:orange' if p['state']=='T' else 'tab:blue'
            q=np.array(p['raw_query_xy']);z=np.array(p['initial_xy']);axs[0].scatter(*q,c=color,marker='x',s=35);axs[0].scatter(*z,c=color,marker='o',s=20);axs[0].text(*z,f" {k}:{p['state']}",color=color,fontsize=8)
            if 'reference_xy' in p:
                ref=np.array(p['reference_xy']);axs[0].plot([z[0],ref[0]],[z[1],ref[1]],color='.6',lw=.8);axs[0].scatter(*ref,c='limegreen',marker='+',s=50)
            if p['annotation_valid'] and p['human_state']=='DIRECT_VISIBLE':axs[0].scatter(*p['annotation_xy'],facecolors='none',edgecolors='cyan',marker='s',s=50)
        for r in row['runtime']['visible_correction']['records']:
            k=int(r['native_corner'])
            if row['runtime']['initial_geometry']['initial_self_hidden'][k]:continue
            if r['state'] not in ('B','I') or not r.get('pixel_queried'):continue
            for ray in r.get('rays',[]):
                if not any(int(ray.get('native_neighbor',-1))==t['native_id'] for t in ts):continue
                q=np.asarray(ray.get('observed_points',[]))
                if len(q):axs[1].scatter(q[:,0],q[:,1],s=7,c='limegreen' if ray.get('supported') else 'tab:red')
        for t in ts:
            z=np.asarray(t['initial_xy']);axs[1].scatter(*z,c='tab:orange',marker='o');axs[1].text(*z,f" T{t['native_id']}",c='tab:orange',fontsize=9)
        axs[0].set_title('nonS only: raw x, coarse o, reference +, DIRECT annotation square')
        axs[1].set_title('Actual BI adjacent-ray samples: supported green / unsupported red')
        fig.suptitle(row['id']+'\nAll T conditional and uncertified; S coordinates absent; no new image inference')
        fig.tight_layout();fig.savefig(OUT/(row['id'].replace(':','_')+'.png'),dpi=160);plt.close(fig)

def readme(d,Ts,folds):
    lines=['# CAL-only T 근거 진단','',f"CAL46 중 호환44의 초기 nonS 코너 {d['compatible_nonS_points']}개를 검사했다. 3D 대응 불일치 CAL2는 별도로 기록했으며, S 좌표·영상 query와 TEST 자료는 사용하지 않았다. 모델·솔버·봉인 예측은 수정하지 않았다.",'',
           'T 13개/7프레임은 모두 initial_pose_projection에 의한 조건부 판정이며 독립 증명은 0개다. raw query와 초기 투영이 함께 영상 밖이어도, 초기 pose는 같은 검출기 관측과 모델로 만든 값이므로 독립 증거 두 개가 아니다.',
           '',f"사람 상태는 {d['T_human_state_counts']}. 참조 재투영이 화면 안인 T는 {d['T_reference_inside']}개, 유효 annotation이 화면 안인 T는 {d['T_valid_annotation_inside']}개다. UNANNOTATED는 물리적 T 정답 상태로 간주하지 않는다.",'',
           f"모든13개에 대해 원 좌표·화면 거리·사람 상태·annotation/reference·조건부 상태 레코드·인접 B/I 영상 선분 근거를 [ALL_13_T_AUDIT.json](ALL_13_T_AUDIT.json)에 기록했다. 실제 B/I에서 T 쪽으로 향한 supported ray가 있는 T는 {d['T_with_supported_adjacent_BI_ray']}개, 비평행 supported선 교점이 계산되는 T는 {d['T_with_two_nonparallel_supported_line_intersection']}개다. 선 방향·대응이 초기 모델을 사용하고 외삽 오차가 교정되지 않았으므로 T certificate로 승격하지 않는다.",'',
           '## CAL에서 확인한 가능한 규칙','',
           '주된 raw localization 가설은 native-compatible44, initial nonS, 유효 human DIRECT/EXTERNAL annotation에서 raw query 오차의 정규화 최댓값을 구한다. raw query를 중심으로 r=이 CAL 최댓값×현재 nonS scale인 원 Ω를 놓는다. 닫힌 화면과 Ω가 완전히 분리되는 경우에만 조건부 T를 유지한다. Ω가 화면과 겹치면 U로 유보한다. 독립적으로 증명된 T는 우선하며, producer가 유지한 O 및 T는 공동 hard constraint로 남긴다. 여기서는 어떤 규칙도 채택하거나 pose solve하지 않았다.','',
           '초기 투영(또는 CAL median을 적용한 투영)을 중심으로 한 다음 범위들은 보조 진단이다. raw localization 범위의 대체물로 선택하지 않았다.','',
           '1. 전체 CAL frame-max의 최댓값을 반지름으로 고정: 관측한 모든 호환 CAL 오차를 포함하지만 미래 오차를 보장하지 않는다.',
           '2. frame-max 95% rank 보정 분위수 ceil((n+1)×.95): 더 작은 범위의 가설이며 LOGO 누락을 그대로 공개한다. 같은 session의 프레임 상관 및 CAL 4개 session 때문에 95% 물리적/미래 coverage라고 부를 수 없다.',
           '3. CAL prior covariance6를 mean 위치에서 2D Jacobian으로 전파: 국소 선형 민감도 보조 자료다. P3P/phase 다중성·모델/semantic 오차를 모두 포괄하는 범위라고 볼 수 없어 독립 채택하지 않는다.','']
    for c,v in d['recipes_not_adopted'].items():
        q=v['rank95'];lines.append(f"- {c} 중심: full-CAL max={v['max_radius_normalized']:.8f}, rank95={q['radius_normalized']:.8f} (rank {q['one_based_rank']}/{q['n']}).")
    lines+=['',f"최대 오차 프레임은 {d['max_error_frame']['id']}이며 normalized={d['max_error_frame']['max_error_normalized']:.8f}다. 같은 CAL에서 참조/model 오차가 큰 사례를 제거하여 범위를 줄이지 않았다. 초기 중심 frame-max 중앙값 {d['normalized_frame_error_quantiles']['0.5']:.8f}, 보간95% {d['normalized_frame_error_quantiles']['0.95']:.8f}다.",'',
            '| held-out session | train-max initial frame coverage | rank95 initial frame coverage |','|---|---:|---:|']
    for f in folds:
        a=f['recipes']['initial'];lines.append(f"| {f['heldout_session']} | {a['max']['covered_frames']}/{a['max']['heldout_frames']} | {a['rank95']['covered_frames']}/{a['rank95']['heldout_frames']} |")
    raw=d['raw_localization_envelope']
    lines+=['','## Raw query localization 범위','',
            f"유효 DIRECT/EXTERNAL {raw['points']}점/{raw['frames']}프레임의 full pooled max={raw['pooled_max_radius_normalized']:.8f}. 최대점은 {raw['argmax']['id']} native{raw['argmax']['native_id']} ({raw['argmax']['raw_annotation_error_px']:.3f}px)이다. 풀 CAL max 원을 쓰면 기존 T13 중 {raw['full_CAL_T_retained_count']}개 유지, 나머지는 U 가설이다. held-out session별 train-only max로는 {raw['heldout_session_T_retained_count']}개 유지한다. O/domain/scale 숫자는 바꾸지 않았다.",'',
            '| human class | 점 수 | normalized max |','|---|---:|---:|']
    for c,v in raw['per_class'].items():lines.append(f"| {c} | {v['points']} | {v['max_error_normalized']:.8f} |")
    lines+=['','| held-out session | train max | point coverage | frame coverage |','|---|---:|---:|---:|']
    for f in raw['LOGO_folds']:lines.append(f"| {f['heldout_session']} | {f['pooled_max_radius_normalized']:.8f} | {f['covered_points']}/{f['heldout_points']} | {f['covered_frames']}/{len(f['heldout_ids'])} |")
    lines+=['','CAL full pooled max는 관측 CAL 안에서 유효 점들을 포함한다는 경험적 규칙이다. 미래 native point coverage 또는 독립적 physical T를 보장하지 않는다. EXTERNAL annotation 또한 가림 뒤 좌표의 인간 추정 오차를 가진다.','',
            '## Native phase와 C2 구별','',
            'calibrator는 xyz/Q/native-to-reference의 고정 3D 대응으로 phase를 정하고 2D 오차나 추정 pose로 phase를 고르지 않는다. 물리 C2 pose 점수의 가장 가까운 phase는 native KP ID 대응과 별개다. 두 phase가 다른 CAL 사례를 [NATIVE_PHASE_AUDIT.json](NATIVE_PHASE_AUDIT.json)에 기록했다. 가까운 C2 phase로 바꾸어 큰 native 오차를 줄이지 않았다.','']
    for p in d['fixed_native_vs_closest_C2_phase_mismatches']:lines.append(f"- {p['id']}: fixed native phase {p['fixed_native_phase']} / closest coarse C2 phase {p['closest_coarse_C2_phase']}.")
    lines+=['','LOGO는 envelope를 held-out session 없이 fit했다. mean/Jacobian은 이미 모든 CAL로 fit된 μ/Σ를 사용하므로 별도 조건부 기술 통계이며 full pipeline OOF가 아니다. 초기 중심의 envelope LOGO도 upstream state/geometry가 all-CAL fit인 조건부 입력이라는 제한을 유지한다.','',
            '## 실제 crop 메타데이터','',
            'CAL3 실제 crop의 RGB·K 이동은 알려져 있으나 metadata의 physical_T_confirmed=false, source_inframe_native_ids=[]이고 source_RGB_validated=false다. source corner 위치가 독립적으로 확인되지 않아 특정 T 코너의 증명이 되지 않는다. 이 진단은 crop 좌표의 사후 정답을 로드하여 certificate를 만들어 내지 않았다.','',
            '## 경계 사례','',
            '- eval_cad:1778653041187027200 T2: raw/coarse는 오른쪽 밖, DIRECT annotation 및 reference는 안.','- eval_night08:1779449501478488320 T0/T3: raw/coarse는 왼쪽 밖, EXTERNAL annotation 및 reference는 안.','- eval_cad:1778653055734035712 T0: DIRECT annotation x=1은 안, derived reference x≈−0.417은 밖. 1픽셀 근처의 reference/annotation 차이는 물리적 정확도의 한계를 드러낸다.','',
            '[전체 수치·LOGO·crop evidence](DIAGNOSIS.json) · [CAL 오차 CDF](CAL_ERROR_CDF.png) · [변경·입력 검증](CHECKS.json)','']
    (OUT/'README_KO.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
