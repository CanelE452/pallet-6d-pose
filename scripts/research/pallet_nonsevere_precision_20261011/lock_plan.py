"""Recreate the declared five-arm plan after all original prerequisite gates.

The declarative configuration matches the original PLAN_LOCK. Only creation UTC,
the explicitly supplied main SHA, and code/input hashes are refreshed. Existing
outputs are never overwritten. No model, optimizer, inference, or solver runs.
"""
import argparse,copy,hashlib,json,math,re,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

CONFIG={
 'status':'LOCKED_BEFORE_FITS',
 'arms':{'M0':'REPLAY CE','M1':'CE+native SmoothL1 expectedcoordinate','M2':'M1+fixed E_cov','M3':'M1+native RGB candidate evidence','M4':'M3+fixed E_cov'},
 'seeds':[1,2],'planned_updates_per_arm_seed':3000,'maximum_updates':30000,'batch':16,'backbone_training':False,
 'optimizer':{'name':'AdamW','lr':.001,'betas':[.9,.999],'weight_decay':.0001,'gradient_clip_norm':5.,'warmup_steps':100,'cosine_final_lr_fraction':.1,'cosine_horizon':3000},
 'common_source_split':{'train':55980,'calibration':1004,'selection':1031,'heldout':1985},
 'real':{'frames':110,'clean':68,'moderate':42,'severe':0,'sessions':9,'primary_seeds':[1,2],'historical_existing_seeds':[1,2,3]},
 'selection':{'criterion':'finalSubPix5cm5deg maximum; tieTmedian thenactiveRGBcost thenarmname','C2_fixed_rows':782,'C1_physical_rotation_excluded':249,'never_REAL_selection':True},
 'finalcap':'allSP: uncapped expectation→cornerSubPix→one Base-anchored1%native diagonal cap; allstandalone originalcap exceptcanonicalPoseFix RAW uncapped',
 'training_aux':{'temperature':1.,'lambda':1.,'coordinate_beta_native_px':1.,'eligibility':'valid/support,13-directionconvexhull,native1%capreachability; commonM0..M4','geometry_rank':6,'rcond':1e-6,'damping':0.,'centroid_excluded':True,'aux_weights':'one-time32TRAINminibatches logitsgradientnorm0.25CE; zero/nonfinite explicitlyblocked'},
 'RGB':{'stride':1,'convs':'3→16→16,3x3SiLU','native_tile_context_pixels':4,'resize':False,'align_corners':False,'pixel_index_grid_offset':.5,'padding':'mean-zeroRGB outsideoriginalimage; batchmaxpad preservespixelpitch','GT_crop':False,'final_residual_zero_init':True,'all_arm_parameter_slots_identical':True},
 'precision':{'FP32':True,'AMP':False,'TF32_matmul':False,'TF32_cudnn':False,'accuracy':'freshfrozenYOLOsinglepass+canonicalOLD_P/N3/PoseFix seeds1/2/3 allcommon110, originalcachedhistoricalpreserved'},
 'augmentation':{'geometry':False,'color':False,'source_precomputed_neck_fixed':True},
 'checkpoint':'lastcommonstep only; noREALstopping, no replacementseeds',
 'budgets':{'device_seconds':7200,'training_reserved_seconds':6200,'CPU_analysis_wall_seconds':3600,'first200':'actualupdates includeall5arms; commonstep floor100(min3000 capacity safety15%) noreset/noreshuffle/noreschedule','runtime_gpu_jobs':1},
 'bootstrap':{'unit':'session','draws':10000,'seed':20260917,'seed_bound_within_frame':True,'primary_family':'fiveM0..M4_SUBPIX minusPOSEFIX_SUBPIX strict5cm5deg','primary_holm_family_size':5,'mechanism_family':['M1-M0','M2-M1','M3-M1','M4-M3'],'mechanism_holm_family_size':4,'exposed_development_only':True},
 'prohibitions':['FINAL/scoring or protectedRGB','Severe REAL','backbonetrain','newsyntheticgeneration','hyperparametersearch','manuscript/LaTeX/PDF modifications','reset/stash/forcepush','externalprocesskill','packageupgrade']}

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4194304),b''):h.update(block)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def validate(args):
    doc=Path(args.public_dir);private=Path(args.private_dir);source=Path(args.source_root)
    required=['DATA_CONTRACT.json','SOURCE_BINDINGS.json','GEOMETRY_AUDIT.json','GEOMETRY_CONTRACT_AUDIT.json','DECODER_AUDIT.json','AUX_WEIGHT_LOCK.json','SYNTHETIC_SELECTION_LOCK.json']
    v={n:read(doc/n) for n in required}
    contract=v['DATA_CONTRACT.json'];assert contract['status']=='LOCKED_COMMON_NONSEVERE110'
    assert contract['frames']==110 and len(set(contract['allowed_ids']))==110
    assert contract['grade_counts']=={'clean':68,'moderate':42} and len(contract['session_counts'])==9
    assert contract['current_protected_per_ID_access']==0 and contract['Severe_excluded_by_user_scope'] and contract['original_RECT245_not_reused_as_new_denominator']
    assert contract['excluded_physical_FINAL_count']==146
    assert re.fullmatch('[0-9a-f]{40}',args.main_sha)
    owners={'historical_source':source,'published_source':Path(args.published_root),'wd_published_source':Path(args.wd_published_root),'private_new':private,'new_worktree':Path(__file__).resolve().parents[3]}
    for b in v['SOURCE_BINDINGS.json']['bindings']:
        p=owners[b['owner']]/b['path'];assert p.is_file() and p.stat().st_size==b['bytes'] and sha(p)==b['sha256'],('SOURCE_BINDING_CHANGED',b['owner'],b['path'])
    geo=v['GEOMETRY_AUDIT.json'];assert geo['status']=='PASS_PARTIAL_ELIGIBILITY' and geo['rows']==60000 and geo['geometry_detached']
    assert geo['rank_gate']==6 and geo['damping']==0 and geo['minimum_auxiliary_corners']==4 and geo['centroid_excluded']
    assert geo['fd_actual_eligible_frames']==32 and geo['fd_J2d']['median']<1e-4 and geo['fd_J2d']['p99']<1e-2 and geo['fd_J3d24']['p99']<1e-4
    assert max(geo['covariance_residual_gradient_fd_relative_errors'])<1e-4 and geo['exact_zero_loss']==0 and geo['exact_zero_gradient_finite'] and geo['exact_zero_gradient_norm']==0
    assert geo['source_bindings_unchanged'] and geo['pose_solver_calls']==geo['new_model_inference_calls']==geo['optimizer_updates']==geo['real_reads']==0
    locks=list(doc.glob('GEOMETRY*LOCK*.json'));assert locks
    operative=[read(p) for p in locks if read(p).get('code_sha256')==sha(Path(__file__).parent/'geometry.py')]
    assert len(operative)==1
    pars=operative[0]['parameters'];assert pars['rank_required']==6 and pars['svd_relative_cutoff']==1e-6 and pars['maximum_condition']==1e6 and pars['damping']==0 and pars['minimum_auxiliary_corners']==4 and pars['projection_max_residual_gate_px']==.5 and not pars['centroid_included']
    for b in operative[0]['bindings']:assert sha(source/b['path'])==b['sha256'],('GEOMETRY_SOURCE_CHANGED',b['path'])
    original_config=read(source/'_docs/experiments/pallet_final_ml_contribution_test_v1/B_line_vs_point/PARAMETER_BUDGET_LOCK.json')['config']
    dimension_config=read(source/'_docs/experiments/pallet_dim_conditioned_p_v1/TRAIN_PROTOCOL_LOCK.json')['config']
    assert original_config==dimension_config==dict(c3=64,c4=128,hidden=16,encoded=24,stencil_fraction=.1310373991727829)
    cache=private/'GEOMETRY_CACHE.npz';assert sha(cache)==geo['private_cache_sha256']
    z=np.load(cache);valid=z['valid'];assert z['A24'].shape==(60000,24,16) and z['eligible'].shape==(60000,8) and z['branch'].shape==(60000,) and int(valid.sum())==58354
    gc=v['GEOMETRY_CONTRACT_AUDIT.json'];assert gc['status']=='PASS' and gc['source_row_join']['same_unique_id_sets']==60000 and gc['rectangular_quarter_turn_rejected'] and gc['reflection_rejected']
    assert gc['projection_category_clarification']['finite_rows_exceeding_gate']==0
    decoder=v['DECODER_AUDIT.json'];assert decoder['status']=='PASS' and decoder['target_rows']==60000 and decoder['actual_train_probability_rows']==512 and decoder['actual_validation_probability_rows']==8040
    assert decoder['half_pixel']['source_feature_center_max_error']==0 and decoder['half_pixel']['native_integer_center_max_error']==0
    grad=decoder['actual_train_logit_coordinate_gradient'];assert grad['status']=='PASS' and grad['gradient_norm']>0 and max(grad['relative_errors'])<1e-4
    assert decoder['whole_object_branch_and_mask_cache_match'] and decoder['geometry_detached'] and not decoder['global_1_33_scaling']
    aux=v['AUX_WEIGHT_LOCK.json'];assert aux['status']=='TRAIN_ONLY_ONE_TIME_LOCK' and aux['minibatches']==32 and aux['batch_size']==16 and aux['gradient_target_ratio']==.25
    assert aux['shared_step0_logits_exact'] and aux['real_reads']==aux['optimizer_updates']==0
    assert all(math.isfinite(aux['weights'][k]) and aux['weights'][k]>0 for k in ('COORD','GEO'))
    sel=v['SYNTHETIC_SELECTION_LOCK.json'];assert sel['status']=='PASS' and sel['physical_rotation_eligible_C2_rows']==782 and sel['excluded_C1_rows']==249 and sel['excluded_C4_rows']==0 and sel['exact_GT_roundtrip_PASS']
    assert all(c['available'] and c['T_cm']<.02 and c['R_deg']<.02 for c in sel['exact_GT_production_F_roundtrip'])
    source_manifest=read(source/'data/pallet/results/pallet_line_pose_v1/SOURCE_MANIFEST.json')
    counts={p:sum(r['partition']==p for r in source_manifest['records']) for p in CONFIG['common_source_split']}
    assert counts==CONFIG['common_source_split'] and all(r['source_kind']=='synthetic' for r in source_manifest['records'])
    side=np.load(source/'data/pallet/results/pallet_dim_conditioned_p_v1/DIMENSION_SIDECAR.npz');indices=np.load(source/'data/pallet/results/pallet_line_pose_v1/cache/record_index.npy');assert np.array_equal(indices,side['record_index'])
    selected=np.array([i for i,j in enumerate(indices) if source_manifest['records'][int(j)]['partition']=='selection' and int(side['order'][i])==2])
    assert len(selected)==782 and hashlib.sha256(selected.astype('<i8').tobytes()).hexdigest()==sel['row_sha256']
    for seed in (1,2):
        order=np.load(source/f'data/pallet/results/pallet_final_ml_contribution_test_v1/B_line_vs_point/order_seed{seed}.npy',mmap_mode='r')[:3000]
        assert order.shape==(3000,16) and all(source_manifest['records'][int(indices[r])]['partition']=='train' for r in np.unique(order))
    return v,required

def run(args):
    start=time.monotonic();doc=Path(args.public_dir);target=doc/'PLAN_LOCK.json';assert not target.exists(), 'Existing PLAN_LOCK is preserved; use a fresh namespace'
    v,required=validate(args);plan=copy.deepcopy(CONFIG)
    plan['created_utc']=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ');plan['source_main_sha']=args.main_sha
    plan['training_code_sha256']={n:sha(Path(__file__).parent/n) for n in ('precision_model.py','train.py')}
    plan['geometry_cache_sha256']=sha(Path(args.private_dir)/'GEOMETRY_CACHE.npz')
    for key,name in [('data_contract_sha256','DATA_CONTRACT.json'),('geometry_audit_sha256','GEOMETRY_AUDIT.json'),('synthetic_selection_lock_sha256','SYNTHETIC_SELECTION_LOCK.json')]:plan[key]=sha(doc/name)
    write(target,plan)
    write(doc/'PLAN_REPLAY_PREREQUISITES.json',dict(status='PASS_ALL_ORIGINAL_GATES',configuration_sha256=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest(),configuration_matches_original_declared_plan=True,creator_code_sha256=sha(__file__),prerequisites={n:sha(doc/n) for n in required},geometry_config_unchanged=True,synthetic_partition_rows=CONFIG['common_source_split'],synthetic_usable_train_rows=v['GEOMETRY_CONTRACT_AUDIT.json']['partition_usable_counts']['train']['usable_correspondence_rows'],raw_rgb_reads=0,new_model_inferences=0,solver_calls=0,optimizer_updates=0,source_mutations=0,wall_seconds=time.monotonic()-start))
    print('PLAN_REPLAY_LOCKED',target.name,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--private-dir',required=True);p.add_argument('--public-dir',required=True);p.add_argument('--published-root',required=True);p.add_argument('--wd-published-root',required=True);p.add_argument('--main-sha',required=True);run(p.parse_args())
