"""B source, prediction-only geometric evidence, parameter and order locks."""
import numpy as np
import torch
from common import *
from generic_point_refiner import GenericPointRefiner

def dataset():return old('train').FeatureDataset(LINE,LINE/'cache')

def run():
    assert (A/'TASK_RISK_ROBUSTNESS_VERDICT.json').exists()
    protocol=read(LINE/'TRAIN_PROTOCOL.json');selection=read(LINE/'SELECTION.json')
    for p,h in protocol['source_sha256'].items():assert sha(p)==h,p
    data=dataset();legacy=old('train');lm=old('model');cfg=protocol['model_config']
    source=data.source
    assert protocol['steps']==6000 and protocol['batch']==16 and protocol['arms']['image_line_only']['corner_weight']==0
    for a in data.arrays.values():assert not a.flags.writeable
    # Match the median length of L's spatial evidence using TRAIN prediction
    # geometry only. No matching mask or GT target/error participates.
    train_idx=np.flatnonzero(data.partitions=='train')
    points=np.asarray(data.arrays['points'][train_idx]);boxes=np.asarray(data.arrays['boxes'][train_idx]);valid=np.asarray(data.arrays['point_valid'][train_idx])
    ends=points[:,np.array(lm.SIDE_EDGES)];length=np.linalg.norm(ends[:,:,1]-ends[:,:,0],axis=-1)
    diag=np.maximum(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=-1),1)
    usable=valid[:,np.array(lm.SIDE_EDGES)].all(-1)&np.isfinite(length)&(length>=2)&np.isfinite(diag[:,None])
    fractions=length/diag[:,None];stencil=float(np.median(fractions[usable]))
    config=dict(c3=cfg['c3'],c4=cfg['c4'],hidden=16,encoded=24,stencil_fraction=stencil)
    model=GenericPointRefiner(**config);line=lm.PalletLinePoseHead(**cfg)
    np_=sum(p.numel() for p in model.parameters());nl=sum(p.numel() for p in line.parameters())
    assert abs(np_/nl-1)<=.1
    bindings={};perseed={};orders={}
    for seed in protocol['seeds']:
        p=LINE/'runs'/f'image_line_only_seed{seed}'/'last.pt';ck=torch.load(p,map_location='cpu',weights_only=False)
        assert ck['step']==6000 and ck['complete'] and ck['baseline_checkpoint_sha256']==R0_SHA
        assert sha(p)==read(p.parent/'COMPLETION.json')['checkpoint_sha256']
        sampler=legacy.ShuffledRows(data.train_rows,seed)
        order=np.stack([sampler.take(protocol['batch']) for _ in range(protocol['steps'])])
        state=sampler.state();saved=ck['sampler_state']
        assert np.array_equal(state['order'],saved['order']) and state['position']==saved['position'] and state['epochs_started']==saved['epochs_started'] and state['rng']==saved['rng']
        h=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest()
        orders[str(seed)]=dict(batch_order_sha256=h,steps=6000,batch=16,exposures=96000,
            old_final_sampler_state_exact=True,old_order_proof='Replayed unchanged deterministic ShuffledRows from same train rows/seed; exact saved final permutation/position/epoch/RNG state. Old per-step row traces were not separately saved.')
        BRAW.mkdir(parents=True,exist_ok=True)
        path=BRAW/f'order_seed{seed}.npy'
        if path.exists():assert np.array_equal(np.load(path),order)
        else:
            with path.open('xb') as f:np.save(f,order)
        rule=selection['selected_rules']['image_line_only']
        perseed[str(seed)]=dict(checkpoint=str(p.relative_to(ROOT)),checkpoint_sha256=sha(p),
            temperature=selection['temperatures']['image_line_only'][str(seed)]['temperature'],
            lam=rule['lam'],cap=rule['max_move_image_diagonal_fraction'],params=nl)
        bindings[str(p.relative_to(ROOT))]=sha(p)
    for p in [LINE/'TRAIN_PROTOCOL.json',LINE/'SOURCE_MANIFEST.json',LINE/'SELECTION.json',LINE/'cache/CACHE_MANIFEST.json',LINE/'cache/CACHE_COMPLETE.json',R0,*LINE_CODE.glob('*.py')]:bindings[str(p.relative_to(ROOT))]=sha(p)
    write(B/'LINE_SOURCE_BINDING.json',dict(paths=bindings,seeds=perseed,old_L_retrained=False,original_image_joint_verdict_unchanged=True))
    write(B/'PRE_FLIGHT_AUDIT.json',dict(status='PASS',train_protocol=protocol,
        actual_train_records=len(train_idx),matched_usable_train_records=len(data.train_rows),
        partitions={p:int(np.sum(data.partitions==p)) for p in ('train','calibration','selection','heldout')},
        cache_read_only=True,augmentation='None added: reuse exact existing cached spatial features and canonical target coordinates',
        symmetry='Source/cache canonical index assignment is fixed; existing L cache/train has NO C2 min-over-permutation reassignment. P reuses those exact targets. C1/C2 diagram permutations are checked as semantic coordinate transforms, not a new training loss.',
        coordinates='Same label-affine letterboxed input pixels internally; original-image displacement=delta/gain; exact baseline coordinate preservation through restore_refinement',
        valid='Same matched train rows and per-point cached gt_valid/predicted finite sentinel mask. P has no edge endpoint/length mask since it has no edge target; per-frame supported-item reduction reused.',
        real_DEV_selection=False))
    write(B/'GENERIC_CANDIDATE_GEOMETRY.json',dict(non_null=221,null=1,directions=13,radii=17,
        r_max_bbox_diagonal_fraction=.08,derivation='Existing max offset fraction. Conservative envelope; does not cover extra angular/tangent-induced WLS motion. Never tuned against GT coverage.',
        sigma_radial_spacing=.08/17,stencil_shape=[4,8],stencil_diameter_bbox_diagonal_fraction=stencil,
        stencil_derivation='Median existing L along-line span / bbox diagonal on all original TRAIN predicted valid edges only, no GT/matching/error input',
        predicted_geometry_rows=len(train_idx),valid_spans=int(usable.sum()),
        P_forward_uses_line_topology=False,stencil_choice_used_GT=False))
    write(B/'PARAMETER_BUDGET_LOCK.json',dict(config=config,P_params=np_,L_params=nl,relative_difference=np_/nl-1,
        tolerance=.1,within5percent=abs(np_/nl-1)<=.05,no_performance_used=True))
    write(B/'EVIDENCE_BUDGET_AUDIT.json',dict(L=dict(roles=8,non_null_candidates=221,null=1,samples=32,feature_channels=[64,128]),
        P=dict(roles=8,non_null_candidates=221,null=1,samples=32,feature_channels=[64,128]),
        spatial_reads_each=8*221*32*2,null_evidence='Both pool non-null evidence without separate null spatial reads',
        sample_count_exact=True,parameter_matching_not_runtime_matching=True,
        body_difference='L two1D kernel3 convolutions32->32->32; P square4x8 patch,2D kernel3 then1x1,32->24->24; local generic shared scorer. No claim FLOPs equal.'))
    write(B/'INIT_ORDER_PARITY.json',dict(status='PASS',seeds=orders,train_rows_sha256=hashlib.sha256(data.train_rows.astype('<i8').tobytes()).hexdigest(),R0_SHA=sha(R0),L_checkpoints_frozen=True))
    print('B_PREFLIGHT_PASS',np_,nl,'stencil fraction',stencil,'usable train',len(data.train_rows),flush=True)

if __name__=='__main__':run()
