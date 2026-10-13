"""Inventory existing RGB-bound backbone caches; never infer or fit a pose."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[7]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.research.pallet_visibility_curriculum_20261013_v1 import common as C

OUT=Path(__file__).resolve().parent
STUDY=OUT.parent.parent
COHORT=STUDY.parent
RAW=ROOT/'data/pallet/results'
DOC=ROOT/'_docs/experiments'


def bound(path):
    p=Path(path)
    return dict(path=str(p.relative_to(ROOT)),sha256=C.sha(p),bytes=p.stat().st_size)


def stats(values):
    x=np.asarray(values,float)
    return dict(n=len(x),mean=float(x.mean()) if len(x) else None,median=float(np.median(x)) if len(x) else None,
                P90=float(np.quantile(x,.9)) if len(x) else None)


def coverage(rows,frames,*,id_key='frame_id',hash_key='image_sha256',path_key='image_key'):
    by={r[id_key]:r for r in rows};common=[fid for fid in frames if fid in by]
    bad=[fid for fid in common if by[fid].get(hash_key)!=frames[fid]['image_sha256'] or by[fid].get(path_key)!=frames[fid]['image']]
    return dict(cached_rows=len(rows),current95_ID_matches=len(common),CAL_matches=sum(frames[i]['partition']=='CAL' for i in common),TEST_matches=sum(frames[i]['partition']=='TEST' for i in common),missing_current_ids=[i for i in frames if i not in by],RGB_path_or_sha_mismatch_ids=bad,RGB_hash_bound=(len(common)==95 and not bad))


def main():
    if (OUT/'INVENTORY.json').exists():raise FileExistsError('Preserve completed inventory')
    inp=COHORT/'PREDICTION_INPUTS.json';frames={f['id']:f for f in C.read(inp)['frames']};assert len(frames)==95
    labels={};label_paths=[COHORT/'OOF_pose_v1/ISOLATED_CAL_SCORING.json',COHORT/'test_reused_v1/ISOLATED_TEST_SCORING.json']
    for p in label_paths:labels.update(C.read(p)['rows'])
    oldgt_path=RAW/'paper_pose_metric_closure_v1/GEOMETRY_RESOLVED_POSE_GT.json';axis_path=RAW/'paper_pose_metric_closure_v1/AXIS_REVIEW_MANIFEST.json'
    oldgt=C.read(oldgt_path)['frames'];axis={r['image']:r for r in C.read(axis_path)['frames_list']};bindings=[]
    for fid,f in frames.items():
        a=axis[f['image']];g=oldgt[a['frame_id']];t=labels[fid]['target']['truth'];d=g['physical_dimensions_m'];body=np.array([d['across'],d['height'],d['along']]);R=np.asarray(g['R_gt_representative']);tr=np.asarray(g['t_gt'])
        quarter=np.eye(3) if abs(body[0]-t['xyz'][0])<1e-6 else np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        assert np.allclose(R,t['body_R'],atol=1e-12,rtol=0) and np.allclose(tr,t['t'],atol=1e-12,rtol=0)
        assert np.allclose(body,t['body_xyz'],atol=1e-12,rtol=0) and np.allclose(R@quarter,t['R'],atol=1e-12,rtol=0) and t['order']==2
        raw=C.read(ROOT/a['annotation'])['camera_data']['intrinsics'];K=np.array([[raw['fx'],0,raw['cx']],[0,raw['fy'],raw['cy']],[0,0,1]])
        assert np.allclose(K,f['K'],atol=1e-12,rtol=0) and C.sha(ROOT/f['image'])==f['image_sha256']
        bindings.append(dict(id=fid,partition=f['partition'],image=f['image'],image_sha256=f['image_sha256'],K=f['K'],old_pose_frame_id=a['frame_id'],old_vs_current_reference_body_R_t_dimensions_and_physical_Q_match=True,proper_group_order=2))
    inventory=[];metric_rows=[]
    for experiment in ('pallet_dope_refiner_20261001_v1','pallet_resnet18_dim_refiner_20261002_v1'):
        result_path=DOC/experiment/'DEV_RESULTS.json';r=C.read(result_path);pred_path=ROOT/r['predictions']['path'];p=C.read(pred_path)
        assert C.sha(pred_path)==r['predictions']['sha256']
        cov=coverage(p['frames'],frames)
        artifacts=[]
        for arm,artifact in r['pose_artifacts'].items():
            path=ROOT/artifact['path'];assert C.sha(path)==artifact['sha256'];cached=C.read(path)
            match=[rr for rr in cached['all_frame_rows'] if rr['frame_id'] in frames]
            artifacts.append(dict(arm=arm,artifact=bound(path),current_IDs=len(match),current_OK_count=sum(x['status']=='OK' for x in match),raw_pose_R_t_saved=False))
        inventory.append(dict(experiment=experiment,predictions=bound(pred_path),coverage=cov,pose_artifacts=artifacts,
            fair_current_physical_rotation_metric_reuse=False,
            reason='Canonical MAIN scores R_cf against camera-facing R_gt_representative; current scores R_cf@Q in fixed physical frame. Raw R/t and per-prediction Q are not saved, so physical metric cannot be reconstructed from canonical scalar errors.',
            same_scorer_R_t_recomputation_performed=False,new_RGB_inference=False))
    # These are already direct-physical C2 endpoints on the exact same target.
    # No R/t is imputed: only scalar unit conversion and <5cm/<5deg success are
    # derived. Missing poses remain unavailable and fail full-population success.
    for backbone in ('dope','resnet18'):
        pred_path=RAW/'pallet_n3_completion_v3/predictions'/f'{backbone}_DEV319.json';evaluation_path=RAW/'pallet_n3_completion_v3/evaluation'/f'{backbone}.json'
        p=C.read(pred_path);e=C.read(evaluation_path);cov=coverage(p['frames'],frames,id_key='id');assert cov['RGB_hash_bound']
        assert e['inputs']['predictions'][0]['sha256']==C.sha(pred_path)
        methods=[]
        for method,v in e['methods'].items():
            if not v.get('result'):continue
            by={r['id']:r for r in v['result']['pose_rows']};assert set(frames)<=set(by)
            method_name=backbone.upper()+'_'+method.upper()
            for fid,f in frames.items():
                r=by[fid];available=bool(r['available']);entry=dict(id=fid,partition=f['partition'],method=method_name,available=available,
                    translation_cm=r['translation_cm'] if available else None,rotation_deg=r['rotation_deg'] if available else None,
                    ADDsym_cm=float(r['ADDsym_m'])*100 if available else None,success_5cm_5deg=bool(available and r['translation_cm']<5 and r['rotation_deg']<5),
                    physical_C2_scalar_cache_reused=True,raw_R_t_recomputed=False,missing_pose_not_imputed=True)
                metric_rows.append(entry)
            methods.append(method_name)
        model_hashes=sorted({r['predictions']['base']['base_checkpoint_sha256'] for r in p['frames'] if r['id'] in frames})
        inventory.append(dict(experiment='pallet_n3_completion_v3',backbone=backbone,
            actual_model='DOPE VGG final epoch60' if backbone=='dope' else 'ResNet18 DSNT CONSTANT epoch10 (distinct from SimpleBaseline epoch60 and DSNT FULL)',
            predictions=bound(pred_path),evaluation=bound(evaluation_path),coverage=cov,methods=methods,base_checkpoint_sha256=model_hashes,
            cached_endpoint_formula_and_reference_compatible_with_current_physical_C2=True,
            raw_pose_R_t_saved=False,same_scorer_R_t_recomputation_performed=False,scalar_endpoint_conversion_and_success_threshold_reaggregation_only=True,
            source_contract=dict(physical_rotation='R_cf@Q',Q='identity if selected CF width agrees with fixed registry X else Ry90',translation='cuboid centroid, m',proper_group='C2, corresponding cuboid vertices',K='native original image annotation camera K; verified exact match',reference='historical geometry-derived body reference; verified exact body_R/t/xyz/physical-Q match on95'),
            limitation='Physical scalar cache reuse is justified by matched RGB/ref/K and exact declared metric formula; without saved R/t a fresh vector-level scorer recomputation is unavailable. No new PnP or model forward was performed.'))
    # DSNT direct metrics exist, but their raw records contain no persistent RGB
    # bytes/path hash. Report availability without promoting them to strict SHA
    # matched comparisons through a merely similar online decode.
    for arm in ('CONSTANT','SHAPE','FULL'):
        pred_path=RAW/'pallet_resnet18_dimension_20261002_v1'/f'DSNT_FULL_{arm}_REAL_PREDICTIONS.json';metric_path=DOC/'pallet_resnet18_dimension_20261002_v1'/f'DSNT_FULL_{arm}_POSE_RESULTS.json';p=C.read(pred_path);m=C.read(metric_path)
        ids={r['id'] for r in p['predictions']['DEV319']};metric_ids={r['id'] for r in m['rows']};common=set(frames)&ids&metric_ids
        inventory.append(dict(experiment='pallet_resnet18_dimension_20261002_v1',arm=arm,predictions=bound(pred_path),pose_metric_cache=bound(metric_path),current_ID_matches=len(common),CAL_matches=sum(frames[i]['partition']=='CAL' for i in common),TEST_matches=sum(frames[i]['partition']=='TEST' for i in common),
            raw_pose_R_t_saved=False,persistent_per_image_RGB_path_or_sha_saved=False,fair_strict_current_comparison_promoted=False,
            reason='Direct physical C2 scalar metrics exist for95, but these specific raw records lack persisted RGB paths/SHA and raw R/t; current RGB identity cannot be independently bound from the cache alone. CONSTANT N3 cache above separately provides exact95 hashes; FULL refiner cache is hash bound but uses canonical MAIN rotation.'))
    simple=RAW/'pallet_resnet18_refiner_20261001_v1';runtime=RAW/'pallet_three_backbone_runtime_20261002_v1/runtime/attempt_0001/ROWS.jsonl'
    cache_names=sorted(p.name for p in simple.glob('*'))
    inventory.append(dict(experiment='pallet_resnet18_refiner_20261001_v1',baseline_training_receipt=bound(DOC/'pallet_resnet18_refiner_20261001_v1/BASELINE_TRAINING_COMPLETE.json'),available_top_level_artifacts=cache_names,
        complete95_frame_pose_or_prediction_cache_found=False,fair_current_comparison_possible=False,
        reason='SimpleBaseline epoch60 training completion exists, but no DEV95/319 pose or prediction cache exists here. Baseline failure-diagnostic NPZ is not a complete RGB-bound current pose population. No model forward was run.'))
    timed=list(C.rows(runtime)) if runtime.suffix=='.gz' else [__import__('json').loads(line) for line in runtime.open() if line.strip()]
    timed=[r for r in timed if 'frame_id' in r]
    inventory.append(dict(experiment='pallet_three_backbone_runtime_20261002_v1',timing_rows=bound(runtime),unique_frame_IDs=len({r['frame_id'] for r in timed}),current_ID_overlap=len(set(frames)&{r['frame_id'] for r in timed}),raw_R_t_saved=False,
        fair_current_pose_comparison_possible=False,reason='26-frame runtime/parity timing subset; timing records do not contain full current95 R/t. The three-estimator publication snapshot is progress metadata, not an additional complete model/prediction method.'))
    summaries={}
    current_score=C.read(STUDY/'evaluation_v3/POSTSEAL_SCORING.json');current_rows={r['id']:r for r in current_score['rows'] if r['cohort']=='PRIMARY_NONSEVERE'}
    for fid,r in current_rows.items():
        for arm in ('ORIGINAL_PNP','ALL_OT_JOINT_SE3'):
            metric_rows.append(dict(id=fid,partition=r['partition'],method='CURRENT_'+arm,available=True,**r['metrics'][arm],physical_C2_scalar_cache_reused=True,raw_R_t_recomputed=False))
    for method in sorted({r['method'] for r in metric_rows}):
        summaries[method]={}
        for part in ('CAL','TEST','ALL95'):
            rows=[r for r in metric_rows if r['method']==method and (part=='ALL95' or r['partition']==part)];available=[r for r in rows if r['available']]
            summaries[method][part]=dict(all_frames=len(rows),available_poses=len(available),missing_poses=len(rows)-len(available),full_population_success_5cm_5deg=sum(r['success_5cm_5deg'] for r in rows),full_population_success_fraction=sum(r['success_5cm_5deg'] for r in rows)/len(rows),translation_cm_conditional_on_available=stats([r['translation_cm'] for r in available]),rotation_deg_conditional_on_available=stats([r['rotation_deg'] for r in available]),ADDsym_cm_conditional_on_available=stats([r['ADDsym_cm'] for r in available]))
    protocol=dict(schema='CACHED_BACKBONE_INVENTORY_PROTOCOL_V1',created_by='read-only inventory; no learned model inference, no pose solve, no training',current95_input=bound(inp),current_score=bound(STUDY/'evaluation_v3/POSTSEAL_SCORING.json'),legacy_GT=bound(oldgt_path),legacy_axis=bound(axis_path),current_label_bindings=[bound(p) for p in label_paths],scorer_contract_source=bound(ROOT/'scripts/research/pallet_dim_conditioned_p_v1/pose.py'),current_scorer_source=bound(ROOT/'scripts/research/pallet_visibility_curriculum_20261013_v1/scoring.py'),native_R_t_recomputation_not_possible_when_vectors_not_cached=True,missing_pose_policy='retain all95; success false; no numeric pose-error imputation; available-only means explicitly conditional',no_external_SOTA_or_other_dataset_mix=True)
    C.write(OUT/'PROTOCOL.json',protocol);C.write(OUT/'INVENTORY.json',dict(schema='CACHED_BACKBONE_CURRENT95_INVENTORY_V1',scope='current exact CLEAN/MODERATE95, CAL46/TEST49; postseal reused data',methods=inventory,per_frame_reference_binding=bindings))
    C.write(OUT/'DIRECT_PHYSICAL_C2_CACHE_SUMMARY.json',dict(schema='EXACT_REFERENCE_PHYSICAL_C2_CACHED_SCALAR_REAGGREGATION_V1',summary=summaries,rows=metric_rows,
        scalar_scores_reused=True,new_R_t_scorer_recomputation=False,vector_level_reproduction_unavailable=True,ordinary_same_population_full_success_comparison=True,conditional_error_means_must_not_replace_all_frame_denominator=True,backbone_changes_do_not_test_current_method_on_other_backbones=True))
    lines=['# 기존 DOPE / ResNet 캐시의 current95 비교 가능성','',
        '새 모델 forward·학습·PnP solve 없이 저장 캐시만 확인했습니다. current CLEAN/MODERATE 95 ID(CAL46/TEST49)의 실제 RGB byte SHA, camera K, 역사적 reference body_R/t/dimensions 및 physical Q/C2 계약을 대조했습니다. 외부 논문 SOTA나 다른 데이터셋 수치는 섞지 않았습니다.', '',
        'DOPE20261001와 DSNT FULL refiner20261002의 319-frame 예측에는 current95가 모두 있고 RGB path/SHA가 일치합니다. 그러나 canonical MAIN은 R_cf와 camera-facing GT를 비교하며, current는 R_cf@Q인 physical rotation을 비교합니다. 이 canonical scalar rotation·ADD를 같은 열에서 섞지 않았습니다. 해당 pose 산출물은 raw R/t도 저장하지 않아 새 physical 점수를 벡터에서 재계산할 수 없습니다.', '',
        '별도의 N3 completion v3 DOPE 및 ResNet18 CONSTANT epoch10 캐시는 95개 RGB SHA를 모두 저장하고, 현재와 동일한 physical C2 scalar pose metric을 이미 제공합니다. 95개 reference body_R/t/dimensions/physical-Q 및 K가 정확히 일치하고 두 scorer의 T/R/대응점 C2-ADD 식이 같습니다. 아래 표는 **이 캐시의 scalar 지표를 재사용**하고 ADD m→cm와 5cm/5deg 성공을 재집계한 결과입니다. raw R/t 재계산 또는 새 추론 결과로 부르지 않습니다.', '',
        '| 방법 | TEST 전체 / available pose | 전체 성공 <5cm,<5deg | available-only 평균 T cm | available-only 평균 R deg |','|---|---:|---:|---:|---:|']
    for method,v in summaries.items():
        s=v['TEST'];lines.append(f"| {method} | {s['all_frames']} / {s['available_poses']} | {s['full_population_success_5cm_5deg']}/{s['all_frames']} | {s['translation_cm_conditional_on_available']['mean']:.3f} | {s['rotation_deg_conditional_on_available']['mean']:.3f} |")
    lines+=['','DOPE의 pose miss는 실패로 유지했습니다. Available-only 평균 오차는 current의 all49 평균과 같은 분모가 아니므로 우월성 비교의 근거로 바꾸지 않았습니다. N3 세 seed를 모두 공개하며 TEST 성능으로 best seed를 선택하지 않았습니다. 기존 current method는 YOLO 입력으로만 실행됐으므로 DOPE/ResNet 위에서도 같은 방법의 보정 효과가 입증됐다는 뜻은 아닙니다.', '',
        '공정 비교를 보류한 범위: SimpleBaseline ResNet18 epoch60은 training receipt만 있고 완전한 DEV95 pose cache가 없었습니다. DSNT CONSTANT/SHAPE/FULL direct raw 캐시는 95 IDs를 포함하지만 해당 cache에 실제 image path/SHA와 R/t가 저장되지 않아 strict byte-bound 새 물리 재계산으로 승격하지 않았습니다. 세 백본 runtime은 26프레임 타이밍/parity 부분집합이며, three-estimator publication snapshot은 진행 상태 문서입니다.', '',
        '[전체 캐시 목록·이유](INVENTORY.json) · [동일 reference/C2 scalar 재집계](DIRECT_PHYSICAL_C2_CACHE_SUMMARY.json) · [검증 protocol](PROTOCOL.json)']
    (OUT/'README_KO.md').write_text('\n'.join(lines)+'\n')
    C.write(OUT/'INVENTORY_SEAL.json',dict(schema='READ_ONLY_BACKBONE_INVENTORY_SEAL_V1',current95_RGB_K_reference_bound=True,new_model_forwards=0,new_PnP_solves=0,new_training=0,old_artifacts_modified=False,protocol_sha256=C.sha(OUT/'PROTOCOL.json'),inventory_sha256=C.sha(OUT/'INVENTORY.json'),metric_summary_sha256=C.sha(OUT/'DIRECT_PHYSICAL_C2_CACHE_SUMMARY.json')))
    print({method:s['TEST'] for method,s in summaries.items()})


if __name__=='__main__':main()
