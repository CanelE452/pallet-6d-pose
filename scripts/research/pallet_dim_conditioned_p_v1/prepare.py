"""Phase 0: actual 60k provenance joins and immutable protocol; no DEV performance."""
from collections import Counter
import hashlib,subprocess
import numpy as np
import torch
import dcp_env as E
from refiner import dimension_features,model
from data import validate_group
from scripts.research.pallet_symmetry_dht_local_v1.symdht_local.export import metadata,legacy_metadata,symmetry,SYMMETRY_CONTRACT,LEGACY_METADATA

def main():
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='main'
    assert E.sha(E.R0)==E.R0_SHA
    d=E.old('train').FeatureDataset(E.LINE,E.LINE/'cache');rows=d.source['records'];assert len(rows)==60000
    assert len({r['id'] for r in rows})==60000 and np.array_equal(d.indices,np.arange(60000))
    oldside={r['id']:r for r in E.read(E.SYM_RAW/'A/rect_target_sidecar.json')['records']}
    legacy=legacy_metadata();contract=E.read(SYMMETRY_CONTRACT);records=[];dims=[];orders=[];perms=[];group=[];sources={};axis_differences=Counter()
    geometry=E.ROOT/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'
    geo=np.load(geometry);gi={str(fid):i for i,fid in enumerate(geo['stems'])}
    builder=E.ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/build_probe_metadata.py'
    fixed_builder=E.C.module('dcp_prepare_fixed_dimensions',builder)
    for i,r in enumerate(rows):
        dim,asset,origin=metadata(r,legacy);order,basis,p=symmetry(asset,contract);s=oldside[r['id']]
        assert order==s['group_order'] and p==s['permutations'] and asset==s['asset']
        # The example exporter returns camera-facing G38 W/D, forbidden as
        # conditioning input. Recover fixed object axes OFFLINE and compare the
        # immutable fixed-XYZ table, never a GT pose/branch at inference.
        xyz=geo['dims'][gi[r['id']]]
        if r['source']=='G38':
            obj=E.read(E.ROOT/r['renderer_annotation_locator_provenance_only'])['objects'][0]
            _,fixed,_,case=fixed_builder.fixed_dimensions(obj,r['id'])
            assert np.array_equal(fixed,xyz);axis_differences[case]+=1
            dim=xyz[[0,2,1]].tolist()
            origin='GEOMETRY_SIDETABLE.dims: audited fixed renderer XYZ; full G38 reconstruction cross-check'
        else:assert np.array_equal(np.array(dim),xyz[[0,2,1]])
        assert np.allclose(xyz,s['dimensions_xyz_m'],rtol=1e-6,atol=1e-7)
        if i==0 or p!=perms[-1][:order]:validate_group(p,order)
        path=LEGACY_METADATA if r['source'] in ['P0','TEX'] else geometry
        if str(path) not in sources:sources[str(path)]=E.bound(path)
        records.append(dict(frame_id=r['id'],record_index=i,partition=r['partition'],source_split=r['source_split'],
          canonical_WDH=dim,dimension_source=origin,dimension_source_file=sources[str(path)]['path'],dimension_source_sha256=sources[str(path)]['sha256'],
          source_asset=asset,symmetry_order=order,allowed_permutations=p,contract_basis=basis,scenario_id=r['scenario_id']))
        dims.append(dim);orders.append(order);perms.append(p+[list(range(9))]*(4-order));group.append([j<order for j in range(4)])
        if i%10000==0:print('DIMENSION_AUDIT',i,60000,flush=True)
    dims=np.array(dims);orders=np.array(orders);assert np.isfinite(dims).all() and (dims>0).all()
    E.write(E.RAW/'DIMENSION_SIDECAR.json',dict(records=records,GT_pose_used=False,canonical_order='W,D,H = X,Z,Y'))
    np.savez(E.RAW/'DIMENSION_SIDECAR.npz',record_index=d.indices,dimensions=dims,order=orders,permutations=np.array(perms),group_valid=np.array(group,bool))
    train=d.partitions=='train';f=dimension_features(dims[train]);mean=f.mean(0);std=f.std(0)
    E.freeze(E.DOC/'DIM_NORMALIZATION_LOCK.json',dict(mean=mean.tolist(),scale=np.where(std<1e-8,1,std).tolist(),std=std.tolist(),
      input=['logW','logD','logH','log(W/D)','log(H/sqrt(WD))'],population='all original paper TRAIN rows only; same fixed scaling reused for square/mixed',rows=int(train.sum()),
      training_record_indices_sha256=hashlib.sha256(d.indices[train].astype('<i8').tobytes()).hexdigest(),calibration_selection_DEV_included=False))
    config=E.read(E.C.B/'PARAMETER_BUDGET_LOCK.json')['config'];protocol=E.read(E.LINE/'TRAIN_PROTOCOL.json');sel=E.read(E.C.B/'P_SELECTION.json')
    files=[E.R0,E.LINE/'SOURCE_MANIFEST.json',E.LINE/'TRAIN_PROTOCOL.json',E.LINE/'cache/CACHE_MANIFEST.json',E.LINE/'cache/CACHE_COMPLETE.json',
      SYMMETRY_CONTRACT,LEGACY_METADATA,geometry,builder,E.ROOT/'challenge/yolo_pose_one_model/spatial_concat_scratch/PROBE_METADATA_60K_AUDIT.json',E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json',E.SYM_RAW/'A/rect_target_sidecar.json',
      E.ROOT/'challenge/config/SQUARE_PALLET_SYMMETRY_CONTRACT.json',E.ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json',
      E.C.B/'P_SELECTION.json',E.C.B/'PARAMETER_BUDGET_LOCK.json',E.C.B/'INIT_ORDER_PARITY.json',
      *[E.OLD_CODE/f for f in ['generic_point_refiner.py','point_inference.py','preflight.py','train_point.py','select_point.py']],E.ROOT/'scripts/research/pallet_line_pose_v1/features.py']
    seed_orders={}
    for seed in [1,2,3]:
        p=E.C.BRAW/f'order_seed{seed}.npy';order=np.load(p);assert order.shape==(6000,16) and np.isin(order,d.train_rows).all()
        h=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest();assert h==E.read(E.C.B/'INIT_ORDER_PARITY.json')['seeds'][str(seed)]['batch_order_sha256']
        ck=E.C.BRAW/f'runs/seed{seed}/last.pt';assert E.sha(ck)==sel['checkpoints'][str(seed)];files.extend([p,ck]);seed_orders[str(seed)]=dict(file=E.bound(p),batch_order_sha256=h)
    bindings=[E.bound(p) for p in files]
    cache_stats={k:dict(path=str((d.directory/v['file']).relative_to(E.ROOT)),bytes=(d.directory/v['file']).stat().st_size,mtime_ns=(d.directory/v['file']).stat().st_mtime_ns) for k,v in d.manifest['arrays'].items()}
    E.freeze(E.DOC/'SOURCE_BINDINGS.json',dict(files=bindings,cache_stat=cache_stats,cache_arrays_open_read_only=all(not a.flags.writeable for a in d.arrays.values())))
    E.write(E.DOC/'DIMENSION_INPUT_PROVENANCE.json',dict(complete=True,source_rows=60000,missing=0,bijective_record_join=True,
      source_dimension_files=len(sources),source_digest=hashlib.sha256(''.join(sorted(v['sha256'] for v in sources.values())).encode()).hexdigest(),
      sidecar=E.bound(E.RAW/'DIMENSION_SIDECAR.json'),registry=E.bound(E.ROOT/'challenge/config/CHALLENGE_OBJECT_GEOMETRY_REGISTRY.json'),
      no_GT_pose_or_clicked_keypoints_input=True,no_framewise_WD_swap=True,DEV='deployment-known object type -> fixed registry WDH; unknown object type explicitly rejected; annotation object type is an oracle-availability assumption, not an object classifier',
      normalization=E.bound(E.DOC/'DIM_NORMALIZATION_LOCK.json'),prior_pose_frame_axis_differences=dict(axis_differences),
      axis_audit='Fixed renderer XYZ matches immutable geometry for all rows. G38 camera-facing dimensions_m is NOT used as input; original fixed_dimensions reconstructs canonical axes offline from renderer index metadata.',
      fixed_dimension_source=E.bound(geometry),fixed_dimension_builder=E.bound(builder),G38_rows_reconstructed=40000,
      offline_dimension_reconstruction_uses_renderer_index_metadata=True,inference_receives_fixed_XYZ_only=True))
    E.write(E.DOC/'SYMMETRY_CONTRACT_AUDIT.json',dict(counts={part:dict(Counter('C'+str(x) for x in orders[d.partitions==part])) for part in np.unique(d.partitions)},
      dimension_only_C4_promotion=False,reflection_allowed=False,center8_fixed=True,edge_topology=True,source_contract=E.bound(SYMMETRY_CONTRACT),real_contract=E.bound(E.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json')))
    E.write(E.DOC/'SPLIT_AND_ORDER_AUDIT.json',dict(partitions=dict(Counter(d.partitions.tolist())),matched_train_rows=len(d.train_rows),orders=seed_orders,read_only=True,
      no_real_train_or_selection=True,missing_unmatched_train_retained_in_audit=int(train.sum())-len(d.train_rows)))
    pars={}
    for arm in E.ARMS:
        torch.manual_seed(1);head=model(arm,config);n=sum(p.numel() for p in head.parameters());pars[arm]=dict(total=n,added=n-18962,increase_fraction=n/18962-1)
    E.freeze(E.DOC/'MODEL_AND_PARAMETER_AUDIT.json',dict(config=config,parameters=pars,visual_parameter_shapes_unchanged=True,metadata_final_layer_zero=True,
      missing_geometry_policy='explicit ValueError; no silent zero-fill',null_flag=True,role_embedding='existing shared P embedding8'))
    E.freeze(E.DOC/'TRAIN_PROTOCOL_LOCK.json',dict(start_SHA=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      paper_arms=E.ARMS,square_arms=E.SQUARE_ARMS,mixed_arms=E.MIXED_ARMS,seeds=[1,2,3],steps=6000,batch=16,optimizer=protocol['optimizer'],
      expected_fits=30,expected_main_updates=180000,smoke_updates=200,config=config,temperature_grid=protocol['calibration']['temperature_grid'],
      calibration_target='original fixed-index P Gaussian candidate CE for ALL arms; same objective and grid, no calibration GT phase assignment',
      selected_rule={k:sel['selected_rule'][k] for k in ['lam','max_move_image_diagonal_fraction']},checkpoint='last6000 only',
      no_new_lambda_cap_sweep=True,FP32=True,AMP=False,TF32='retain existing PyTorch defaults; record actual environment, do not claim deterministic CUDA grid_sample backward',
      train_phase='one whole-object GT tuple and mask, identity-first tie, frozen R0 residual mean/bbox diagonal, missing prediction normalized penalty1',
      branch_in_inference=False,normalization='paper TRAIN-only fixed stats shared across all tracks',
      square_temperature='T=1 fixed; no square DEV calibration; original P lambda/cap',mixed_temperature='same original P fixed-target CE on synthetic calibration only',
      primary_contrasts=[['N1_SYM_ONLY','N0_BASE_REPLAY'],['N2_DIM_ONLY','N0_BASE_REPLAY'],['N3_DIM_SYM','N2_DIM_ONLY'],['N4_META_SYM','N3_DIM_SYM'],['N4_META_SYM','N0_BASE_REPLAY']],
      bootstrap=dict(seed=20260917,resamples=10000,real_unit='session',synthetic_unit='frame plus scenario secondary'),
      parity_tolerance=dict(initial_logits_maxabs=0,detector_candidates_maxabs=0,oldP_replay_Esym_absolute=1e-4,oldP_replay_pooled_median_px_absolute=.1),
      safety='No severe damage: proposed-minus-control good5_to_bad10 count <=0 and gross20 delta <=0; canonical GT identity aligned. Otherwise UNRESOLVED even if CI supports improvement.',
      verdict='CI lower>0 WORSENED; CI upper<0 plus >=2/3 better and preservation+safety SUPPORTED; otherwise UNRESOLVED; integrity failure stops execution',
      no_multiplicity_adjusted_confirmatory_claim=True,square_mixed_claim='secondary/diagnostic only, domain and group confounded'))
    E.DOC.mkdir(parents=True,exist_ok=True)
    (E.DOC/'PURPOSE_AND_HYPOTHESES.md').write_text('# Dimension-conditioned symmetry-aware local P\n\nFrozen R0, unchanged 222 local candidates; only P head parameters train. N1−N0 target effect, N2−N0 dimensions, N3−N2 target with dimensions, N4−N3 explicit code, N4−N0 package. OLD_P is an immutable external reference. Paper, square real-supervised and mixed-domain diagnostic are separate tracks.\n\nCanonical dimensions and approved symmetry must be supplied externally at deployment. The experiment cannot prove this information is always known or free. Unknown metadata is an explicit failure, never guessed from GT pose. No Hough path, FINAL access, original checkpoint/source edits, real-based selection or radius sweep.\n')
    (E.DOC/'SOURCE_AND_DIMENSION_AUDIT.md').write_text('# Source audit\n\n60,000 source records directly joined to unchanged P cache indices; no missing dimensions. G38 uses audited fixed renderer XYZ from GEOMETRY_SIDETABLE, fully cross-checked against the original fixed_dimensions builder. Its raw dimensions_m is camera-facing and MUST NOT be a model input. Legacy uses fixed renderer XYZ. Both inputs are WDH=XYZ[0,2,1], never an inference-time pose-based swap. Full source hashes are in the sidecar.\n\nRegistry dimensions match the requested values. Real C2 is the pre-existing benchmark convention (wood physically unreviewed); square C4 is the approved task contract. Unknown source assets remain C1 even if square.\n\nOnly 55,915 of 55,980 paper training records are matched/usable, exactly as OLD_P. All missing/unmatched rows remain in evaluation/audits. Original source image/cache files are not copied or rewritten.\n')
    print('PHASE0_SOURCE_COMPLETE',pars,flush=True)
if __name__=='__main__':main()
