"""Saved-source split/teacher/query diagnostics. No RGB, model, or feature read."""
from pathlib import Path
import collections, hashlib, json
import numpy as np

H=Path(__file__).resolve().parent
P=H.parent.parent
V129=P/'query_visibility_registered_native_diversity_20261023_v129'
V132=P/'query_visibility_actual_initializer_source_20261024_v132'
V133=P/'query_visibility_native_geometry_control_20261024_v133'
CFG=V129/'render_preparation/BASE_CONFIGS192_BEFORE_RGB.json'
FACTS=V132/'frozen_C16_actual_initializer_v1/INPUT_ROWS_AND_CROP_FACTS.json'
LEDGER=V132/'saved_initializer_diagnosis_v1/ALL6144_SLOTS_TWO_CONVENTIONS_GT_ONLY.jsonl'
N=6144

def read(p):return json.loads(Path(p).read_text())
def bind(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def hist(x):return dict(sorted(collections.Counter(map(str,x)).items()))
def stats(x):
    a=np.asarray([z for z in x if z is not None and np.isfinite(z)],np.float64)
    if not len(a):return dict(N=0)
    return dict(N=len(a),minimum=float(a.min()),q25=float(np.quantile(a,.25)),median=float(np.median(a)),q75=float(np.quantile(a,.75)),maximum=float(a.max()))
def key(r):
    f=r['factors']
    fields=['geometry_family','aspect','material','signed_quarter','foreign_family','target_role','distance_scale']
    return tuple(f[k]for k in fields)+(int(np.sign(f['camera_pair_deg'][0])),)+(tuple(r['sham_direction_unit_xy']),)+(r['selected_CF_index'],r['foreign_extent_factor'],r['foreign_margin_px'],r['foreign_depth_fraction'],r['foreign_roll_deg'])
def contains(outer,inner):
    return bool(np.all(np.asarray(inner)[:2]>=np.asarray(outer)[:2])and np.all(np.asarray(inner)[2:]<=np.asarray(outer)[2:]))
def metric(rows,pred,indices):
    ix=np.asarray(indices,np.int64);states=np.array([rows[i]['teacher_state']for i in ix]);pp=pred[ix]
    cm=[[int(((states==s)&(pp==p)).sum())for p in(0,1,-1)]for s in'DE']
    correct=lambda j:cm[j][j]/sum(cm[j])if sum(cm[j])else None
    return dict(slots=len(ix),teacher=hist(states),CM_DE_rows_DEU_columns=cm,D_recall=correct(0),E_recall=correct(1),
                teacher_S_responses=hist(pp[states=='S']),teacher_U_responses=hist(pp[states=='U']))

def run():
    configs=read(CFG)['rows'];facts=read(FACTS)['rows'];ledger=[json.loads(x)for x in LEDGER.read_text().splitlines()if x]
    rows=[r for r in ledger if r['convention']=='CF_PRIMARY_DECLARED']
    assert len(configs)==192 and len(facts)==len(rows)==N
    for i,(r,f)in enumerate(zip(rows,facts)):
        assert r['scene_index']==i//8 and r['native_slot']==i%8 and r['query_id']==f['occurrence_id']
        assert r['query_xy']==f['query_xy']
    scores={};score_bindings={}
    score_paths=dict(C16=V132/'frozen_C16_actual_initializer_v1/C16_PREDICTIONS_INPUT_ONLY.npz',
       C24=V133/'runs/C24_C25_MATCHED_G42_FIXED8000/C24_ZERO_GEOMETRY_CONTROL/FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz',
       C25=V133/'runs/C24_C25_MATCHED_G42_FIXED8000/C25_NATIVE_GEOMETRY42/FINAL_NATIVE6144_SCORES_INPUT_AND_OUTPUT.npz')
    for name,path in score_paths.items():
        with np.load(path,allow_pickle=False)as f:
            scores[name]=f['prediction'].copy();available=f['query_available'].copy();ids=f['occurrence_id'].copy()
        assert available.shape==scores[name].shape==(N,)
        assert available.tolist()==[bool(r['query_available'])for r in facts]
        assert ids.tolist()==[r['query_id']for r in rows]
        score_bindings[name]=bind(path)

    split={};config_comparison={};train=[r for r in configs if r['split']=='TRAIN'];val=[r for r in configs if r['split']=='VALIDATION']
    categorical=('geometry_family','aspect','material','signed_quarter','foreign_family','target_role','distance_scale','camera_pair_deg','asset_seed')
    for fold,rr in [('TRAIN',train),('VALIDATION',val)]:
        uniquegeometry={r['geometry_group_id']:r for r in rr};geometry=list(uniquegeometry.values())
        split[fold]=dict(bases=len(rr),scenes=4*len(rr),native_slots=32*len(rr),unique_positive_scaled_geometry_groups=len(geometry),
            template_bindings=[dict(path=p,sha256=s)for p,s in sorted({(r['template']['path'],r['template']['sha256'])for r in rr})],
            factor_counts={k:hist(r['factors'][k]for r in rr)for k in categorical},
            placement_schedule_counts={k:hist(r[k]for r in rr)for k in('foreign_extent_factor','foreign_margin_px','foreign_depth_fraction','foreign_roll_deg','selected_CF_index')},
            dimensions_xyz_uniquegeometry={axis:stats(r['dimensions_xyz_m'][j]for r in geometry)for j,axis in enumerate('XYZ')},
            camera_distance_m=stats(r['camera']['distance']for r in rr),
            reference_bbox_native_width=stats(np.ptp(np.asarray(r['native8_CF_xy']),axis=0)[0]for r in rr),
            reference_bbox_native_height=stats(np.ptp(np.asarray(r['native8_CF_xy']),axis=0)[1]for r in rr),
            material_seed_count=len({r['seeds']['material']for r in rr}),light_seed_count=len({r['seeds']['light']for r in rr}),
            background_seed_count=len({r['seeds']['background']for r in rr}))
    match=[]
    for v in val:
        matched=[t['base_index']for t in train if key(t)==key(v)]
        assert matched
        match.append(dict(VAL_base=v['base_index'],matching_TRAIN_discrete_configuration_bases=matched))
    config_comparison=dict(all_VAL64_have_seen_categorical_configuration=len(match)==64,
       matching_key='Geometry/aspect/material/quarter/foreign/role/distance schedule/camera sign/sham direction/selectedCF/extent/margin/depth/roll; excludes camera angle magnitude, actual dimensions, appearance seeds and asset/fold.',
       matched_VAL_to_TRAIN=match,heldout_numeric_camera_pairs=[list(x)for x in sorted({tuple(r['factors']['camera_pair_deg'])for r in val})],
       exact_template_bindings_shared_between_folds=True,new_geometry_topology_count_in_VAL=0,
       no_factor_isolated_by_split='Camera angles, positive axis dimensions, and appearance/light/background seeds all change together; causal attribution to one factor is not identified.')

    teacher_audit=collections.Counter();details=[];base_bindings=[]
    for cfg in configs:
        path=V129/'actual_corpus_v1/bases'/cfg['base_id']/'SCENES_GT_ONLY.json';packet=read(path);base_bindings.append(bind(path))
        assert len(packet['rows'])==4
        for scene in packet['rows']:
            si=scene['scene_index'];assert si//4==cfg['base_index']
            for k,record in enumerate(scene['records_CF']):
                i=si*8+k;r=rows[i];f=facts[i]
                assert record['state']==r['teacher_state']==scene['state8_CF'][k]
                assert record['canonical_index']==cfg['permCF_to_canonical'][k]
                assert np.array_equal(record['projected_reference_xy'],r['reference_xy'])
                assert np.array_equal(scene['native8_CF_xy'][k],r['reference_xy'])
                active=np.asarray(record['active_witness_indices'],np.int64)
                before=np.asarray(record['target_witness_states']);combined=np.asarray(record['combined_witness_states'])
                before_state=record['target_alone_region_state'];state=r['teacher_state']
                expected=('U'if not record['all_region_witnesses_in_frame']else'S'if before_state=='S'else'U'if before_state!='D'else'E'if np.all(combined[active]=='E')else'D'if np.all(combined[active]=='D')else'U')
                assert expected==state
                if state in'DE':assert before_state=='D'and len(record['front_face_axes'])>=2 and np.all(before[active]=='D')and 0 in active
                if state=='S':assert np.all(before=='S')and record['geometric_S']
                if state=='U':
                    reason='OFFFRAME'if not record['all_region_witnesses_in_frame']else'TARGET_ALONE_U'if before_state!='D'else'PARTIAL_OR_AMBIGUOUS_EXTERNAL'
                else:reason=state
                teacher_audit[(r['split'],reason)]+=1
                bbox=np.asarray(record['projected_active_witness_bbox_xyxy'],np.float64)
                width,height=bbox[2:]-bbox[:2];d=dict(index=i,active_count=len(active),active_bbox_xyxy=bbox.tolist(),
                    active_bbox_native_width=float(width),active_bbox_native_height=float(height),teacher_U_reason=reason,
                    front_faces=len(record['front_face_axes']),canonical_index=record['canonical_index'],
                    nearest_background_all37=scene['nearest_pixel_diagnostic_CF'][k]['nearest_background'],
                    unique_active_nearest_pixels=scene['nearest_pixel_diagnostic_CF'][k]['active_unique_nearest_pixels'],
                    full96_sample_centre_bbox_contains_active_region=None,central32_sample_centre_bbox_contains_active_region=None,
                    target_context_bbox_contains_active_region=None,active_bbox_RGB96_width=None,active_bbox_RGB96_height=None)
                if r['query_xy']is not None:
                    q=np.asarray(r['query_xy']);L=r['roi_side'];fullhalf=(.5-.5/96)*L;corehalf=(15.5/96)*L
                    d.update(full96_sample_centre_bbox_contains_active_region=contains(np.r_[q-fullhalf,q+fullhalf],bbox),
                        central32_sample_centre_bbox_contains_active_region=contains(np.r_[q-corehalf,q+corehalf],bbox),
                        target_context_bbox_contains_active_region=contains(f['context_box_xyxy'],bbox),
                        active_bbox_RGB96_width=float(width*96/L),active_bbox_RGB96_height=float(height*96/L))
                details.append(d)
    assert len(details)==N and [d['index']for d in details]==list(range(N))
    details_by_index={r['index']:r for r in details}
    groups={}
    for fold in('TRAIN','VALIDATION'):
        for boundary in('ALL','NONBOUNDARY','BOUNDARY'):
            for band in('ALL','LE3','GT3','GT8','NO_QUERY'):
                ix=[]
                for i,r in enumerate(rows):
                    if r['split']!=fold:continue
                    if boundary=='NONBOUNDARY'and r['correspondence_boundary']or boundary=='BOUNDARY'and not r['correspondence_boundary']:continue
                    error=r['error_px']
                    if band=='LE3'and(error is None or error>3)or band=='GT3'and(error is None or error<=3)or band=='GT8'and(error is None or error<=8)or band=='NO_QUERY'and error is not None:continue
                    ix.append(i)
                groups['::'.join((fold,boundary,band))]=ix
    for fold in('TRAIN','VALIDATION'):
        for field in('camera_pair_deg','geometry_family','aspect','material','foreign_family','target_role','distance_scale','signed_quarter'):
            vals=sorted({str(r['factors'][field])for r in rows if r['split']==fold})
            for value in vals:
                for boundary in('ALL','NONBOUNDARY'):
                    groups['::'.join((fold,boundary,field,value))]=[i for i,r in enumerate(rows)if r['split']==fold and str(r['factors'][field])==value and(boundary=='ALL'or not r['correspondence_boundary'])]
    reports={name:{m:metric(rows,pred,ix)for m,pred in scores.items()}for name,ix in groups.items()}
    # Keep nested strata needed to separate smallq errors from a placement effect.
    strata={}
    for fold in('TRAIN','VALIDATION'):
        for variant in('BEFORE','COVER','NEAR_SHAM','FAR_SHAM'):
            for near in(False,True):
                ix=[i for i,r in enumerate(rows)if r['split']==fold and r['variant']==variant and not r['correspondence_boundary']and(not near or r['error_px']is not None and r['error_px']<=3)]
                strata[fold+'::NONBOUNDARY::'+variant+('::LE3'if near else'')]=dict(models={m:metric(rows,p,ix)for m,p in scores.items()},error_px_knownD=stats(rows[i]['error_px']for i in ix if rows[i]['teacher_state']=='D'))
    support={};failures={}
    for model,pred in scores.items():
        fail=[i for i,r in enumerate(rows)if r['split']=='VALIDATION'and not r['correspondence_boundary']and r['teacher_state']=='D'and pred[i]==1]
        small=[i for i in fail if rows[i]['error_px']is not None and rows[i]['error_px']<=3]
        failures[model]=dict(nonboundary_D_to_E=len(fail),smallerror_LE3_D_to_E=len(small),
             LE3_variant=hist(rows[i]['variant']for i in small),LE3_camera=hist(rows[i]['factors']['camera_pair_deg']for i in small),
             LE3_native_slot=hist(rows[i]['native_slot']for i in small),LE3_geometry=hist(rows[i]['factors']['geometry_family']for i in small),
             LE3_material=hist(rows[i]['factors']['material']for i in small),LE3_foreign_family=hist(rows[i]['factors']['foreign_family']for i in small),
             LE3_full96_contains=sum(details_by_index[i]['full96_sample_centre_bbox_contains_active_region']for i in small),
             LE3_core32_contains=sum(details_by_index[i]['central32_sample_centre_bbox_contains_active_region']for i in small),
             LE3_context_contains=sum(details_by_index[i]['target_context_bbox_contains_active_region']for i in small),
             LE3_region_RGB96_width=stats(details_by_index[i]['active_bbox_RGB96_width']for i in small),
             LE3_region_RGB96_height=stats(details_by_index[i]['active_bbox_RGB96_height']for i in small),
             indices=fail,smallerror_indices=small)
    for fold in('TRAIN','VALIDATION'):
        for state in'DESU':
            ix=[i for i,r in enumerate(rows)if r['split']==fold and r['teacher_state']==state]
            support[fold+'::'+state]=dict(N=len(ix),active_count=hist(details_by_index[i]['active_count']for i in ix),
                active_bbox_native_width=stats(details_by_index[i]['active_bbox_native_width']for i in ix),
                active_bbox_native_height=stats(details_by_index[i]['active_bbox_native_height']for i in ix),
                unique_active_nearest_pixels=stats(details_by_index[i]['unique_active_nearest_pixels']for i in ix),
                active_bbox_RGB96_width=stats(details_by_index[i]['active_bbox_RGB96_width']for i in ix),
                active_bbox_RGB96_height=stats(details_by_index[i]['active_bbox_RGB96_height']for i in ix))
    dimension_overlap=[]
    for v in val[::16]:
        tt=[t for t in train if t['factors']['geometry_family']==v['factors']['geometry_family']and t['factors']['aspect']==v['factors']['aspect']]
        xyz=np.asarray([t['dimensions_xyz_m']for t in tt]);point=np.asarray(v['dimensions_xyz_m'])
        dimension_overlap.append(dict(VAL_geometry_group=v['geometry_group_id'],TRAIN_same_family_aspect_xyz_min=xyz.min(0).tolist(),TRAIN_same_family_aspect_xyz_max=xyz.max(0).tolist(),VAL_xyz=point.tolist(),VAL_outside_TRAIN_dimension_range=( (point<xyz.min(0)) | (point>xyz.max(0))).tolist()))
    out=dict(completed=True,PASS=True,goal_status='active',fully_solved=False,source_only=True,split=split,
      config_comparison=config_comparison,dimension_overlap=dimension_overlap,
      teacher_state_derivation_all6144_exact=True,teacher_GT_reference_matches_declared_CF_permutation_all6144=True,
      teacher_counts_and_U_reasons={fold:{reason:count for(f,reason),count in teacher_audit.items()if f==fold}for fold in('TRAIN','VALIDATION')},
      reports=reports,variant_nonboundary_strata=strata,false_hidden_visible_summary=failures,support_summary=support,
      semantic_scope='Constructed continuous endpoint + camera-facing active subset of37registeredwitnesses; D/E are not query-pixel occupancy. All37-nearest-background metadata cannot identify which active witness aliases.',
      crop_extent_scope='Only analytic containment of projected active-witness bbox in input sampling-centre extents; not evidence pixels/colors/features preserve the witness.',
      no_single_cause_proven=True,actions=dict(RGB_reads=0,mixed_buffer_reads=0,feature_reads=0,model_calls=0,fit=0,current223=0,V112=0,PnP=0),
      evidence=dict(code=bind(Path(__file__)),configs=bind(CFG),input_facts=bind(FACTS),GT_ledger=bind(LEDGER),source_scene_GT=base_bindings,predictions=score_bindings,
        source_policy=bind(V129/'ROOT_CORPUS_POLICY_BEFORE_PIXELS.json'),source_math=bind(V129/'render_preparation/corpus_math.py'),teacher_code=bind(P/'query_visibility_spatial_stencil_control_20261023_v127/registered_native_teacher_feasibility/build_registered_pallet.py')))
    path=H/'SOURCE_CONTRACT_AND_SPLIT_REVIEW.json'
    assert not path.exists();path.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    rowsout=H/'ALL6144_TEACHER_SUPPORT_METADATA.json';assert not rowsout.exists();rowsout.write_text(json.dumps(dict(completed=True,rows=details),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(result=bind(path),support_rows=bind(rowsout),false_hidden_summary=failures,teacher_counts=out['teacher_counts_and_U_reasons']),ensure_ascii=False),flush=True)

if __name__=='__main__':run()
