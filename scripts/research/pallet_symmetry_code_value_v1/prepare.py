"""Freeze source integrity, code-only protocol, initial states and donor maps."""
from collections import Counter
import subprocess
import numpy as np
import torch
import cv_env as E
from code_adapter import model,PaperData,MixedData
from refiner import dimension_features
from data import validate_group
from inference import registry_input

def counts(a):return {'C'+str(g):int((np.asarray(a)==g).sum()) for g in [1,2,4]}
def main():
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='main'
    assert E.sha(E.D.R0)==E.D.R0_SHA
    D=E.D;old=E.read(D.DOC/'SOURCE_BINDINGS.json');E.verify(old['files'])
    proof=E.read(D.DOC/'DIMENSION_INPUT_PROVENANCE.json');assert proof['inference_receives_fixed_XYZ_only']
    E.verify([proof[k] for k in ['sidecar','normalization','fixed_dimension_source','fixed_dimension_builder','registry']])
    assert E.read(D.DOC/'CANONICAL_DIMENSION_CORRECTION.json')['complete']
    paper=PaperData();mixed=MixedData();square=mixed.square
    dims=paper.side['dimensions'];geo=np.load(E.ROOT/proof['fixed_dimension_source']['path']);gi={str(s):i for i,s in enumerate(geo['stems'])}
    assert len(set(paper.indices))==len(paper.indices)
    geometry_dims=geo['dims']
    fixed=np.array([geometry_dims[gi[r['id']]][[0,2,1]] for r in paper.source['records']])
    assert np.array_equal(dims,fixed)
    train=paper.partitions=='train';f=dimension_features(dims[train]);norm=paper.norm
    assert np.allclose(f.mean(0),norm['mean'],rtol=0,atol=1e-12) and np.allclose(f.std(0),norm['std'],rtol=0,atol=1e-12)
    for g in np.unique(paper.side['order']):
        ps=paper.side['permutations'][paper.side['order']==g,:g];assert (ps==ps[0]).all();validate_group(ps[0],int(g))
    validate_group(square.perms,4)
    dev=E.read(D.DOC/'DEV_CACHE_COMPLETE.json');devrows=[torch.load(E.ROOT/r['path'],map_location='cpu',weights_only=False) for r in dev['records']]
    for row in devrows:
        d,g=registry_input(row['object_type']);assert np.array_equal(d,row['dimensions']) and g==row['order'] and not row['GT_input']
    groups={part:counts(paper.side['order'][paper.partitions==part]) for part in np.unique(paper.partitions)}
    audit=dict(complete=True,paper=groups,paper_usable_train=counts(paper.side['order'][paper.train_rows]),paper_DEV=counts([r['order'] for r in devrows]),
      square=dict(train=len(square.rows)-len(square.validation_rows),usable_train=len(square.train_rows),DEV=len(square.validation_rows),group='C4'),
      canonical_fixed_XYZ_all_rows_exact=True,TRAIN_normalization_exact=True,dimensions_missing=0,
      real_metadata='externally known object type -> fixed canonical registry. Unknown object type not supported; no GT pose/branch/axis input.',
      wood='Physical symmetry UNREVIEWED; evaluator retains existing C2 benchmark-equivalence convention, not a physical symmetry claim.',
      inference_group_from_dimensions=False,DEV_C2_only=all(r['order']==2 for r in devrows),A_C4_absent=groups['train']['C4']==0,
      B_domain_group_confounded=True,old_splits_unchanged=True)
    E.write(E.DOC/'GROUP_AND_DIMENSION_AUDIT.json',audit);print('ACTUAL_GROUP_COUNTS',audit,flush=True)
    orders={};initial={};(E.RAW/'orders').mkdir(parents=True,exist_ok=True);(E.RAW/'initial').mkdir(parents=True,exist_ok=True)
    for seed in [1,2,3]:
        a=np.load(D.C.BRAW/f'order_seed{seed}.npy');assert a.shape==(6000,16) and np.isin(a,paper.train_rows).all()
        p=D.old('train').ShuffledRows(paper.train_rows,seed);s=D.old('train').ShuffledRows(square.train_rows,seed)
        b=np.stack([np.r_[p.take(8),s.take(8)+60000] for _ in range(6000)])
        assert np.array_equal(b,np.load(D.RAW/f'orders/MIXED_seed{seed}.npy'))
        for track,order in [('A',a),('B',b)]:
            dst=E.RAW/f'orders/{track}_seed{seed}.npy'
            if dst.exists():assert np.array_equal(np.load(dst),order)
            else:np.save(dst,order)
            orders[f'{track}_seed{seed}']=dict(**E.bound(dst),sequence_sha256=E.order_sha(order))
        torch.manual_seed(seed);m=model();torch.manual_seed(seed);other=model()
        assert all(torch.equal(v,other.state_dict()[k]) for k,v in m.state_dict().items())
        dst=E.RAW/f'initial/seed{seed}.pt'
        if dst.exists():assert E.state_sha(torch.load(dst,map_location='cpu',weights_only=False))==E.state_sha(m.state_dict())
        else:torch.save(m.state_dict(),dst)
        initial[str(seed)]=dict(file=E.bound(dst),full_state_sha256=E.state_sha(m.state_dict()),params=sum(p.numel() for p in m.parameters()))
    lock=E.read(D.DOC/'TRAIN_PROTOCOL_LOCK.json');assert lock['selected_rule']==dict(lam=1.,max_move_image_diagonal_fraction=.01)
    E.freeze(E.DOC/'PROTOCOL_LOCK.json',dict(start_SHA=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),arms=E.ARMS,
      seeds=[1,2,3],steps=6000,batch=16,optimizer=lock['optimizer'],config=lock['config'],context_dim=8,neutral=[1/3]*3,
      only_input_difference='context[:,5:8]',target='existing DCP whole-object local_phase for both arms, TRAIN-only',normalization=E.bound(D.DOC/'DIM_NORMALIZATION_LOCK.json'),
      T=1.,rule=lock['selected_rule'],temperature_calibration=False,FP32=True,AMP=False,checkpoint='last6000',main_fits=12,main_updates=72000,smoke_updates=200,
      C_optimizer_updates=0,bootstrap=dict(resamples=10000,seed=20260918),initial=initial,orders=orders,
      safety='Both A/B paper DEV: net canonical good<5->bad>10 (forward minus reverse) <=0 AND gross20 delta<=0. Same strict diagnostic guard as DCP.',
      decision_operationalization=dict(benefit='CI upper<0 and >=2/3 improved seeds. A SYNTH overall OR >=2 of B SYNTH C1, B SYNTH C2, B SQUARE C4.',
        perturbation_advantage='For neutral AND wrong on the SAME multi-group SYNTH population: perturb-minus-correct CI lower>0, >=2/3 positive seeds, mean coordinate change>=0.01px. Fixed before new performance.',
        practical_output_threshold_px=.01,KEEP='benefit AND A/B real safety AND B C4 not CI lower>0 AND perturbation_advantage',
        DROP='No consistent matched-capacity benefit in ANY A/B overall/group contrast AND no significant correct-code performance advantage vs neutral/wrong in ANY evaluated population (with >=0.01px change). Output dependence alone is not value.',
        otherwise='UNRESOLVED: code-free main method recommendation; no automatic sweep'),
      decision_limitations='No multiplicity-adjusted confirmation; reused DEV; B domain/group confounding; all-C2 DEV is constant-token sensitivity, not multi-group routing',
      hardware_changes=False,new_data=False,FINAL_access=False))
    E.freeze(E.DOC/'CAPACITY_PARITY.json',dict(complete=True,params={a:initial['1']['params'] for arms in E.ARMS.values() for a in arms},
      initial=initial,full_state_exact=True,same_8D_architecture=True,only_final3_context_channels_vary=True))
    # Freeze donors using only IDs and approved metadata, never prediction/GT scores.
    held=np.flatnonzero(paper.partitions=='heldout')
    pops={'SYNTH':[(paper.source['records'][i]['id'],int(paper.side['order'][i])) for i in held],
      'DEV':[(r['id'],r['order']) for r in devrows],'SQUARE':[(square.rows[i]['id'],4) for i in square.validation_rows]}
    donors={};rng=np.random.default_rng(20260918)
    for pop,rows in pops.items():
        if len({g for _,g in rows})<2:donors[pop]=dict(applicable=False,reason='Only one group; shuffle cannot remove group information',records=[]);continue
        ids=[r[0] for r in rows];g=np.array([r[1] for r in rows]);ix=rng.permutation(len(rows))
        donors[pop]=dict(applicable=True,same_code=int((g==g[ix]).sum()),different_code=int((g!=g[ix]).sum()),same_frame=int((ix==np.arange(len(ix))).sum()),
          records=[dict(recipient=ids[i],donor=ids[j],donor_group=int(g[j])) for i,j in enumerate(ix)])
    E.freeze(E.DOC/'CODE_DONOR_LOCK.json',dict(seed=20260918,GT_scores_used=False,populations=donors))
    # Full hashes of old code/docs and reused raw artifacts, plus read-only cache stats.
    paths=[*D.HERE.glob('*.py'),*D.DOC.rglob('*')];paths=[p for p in paths if p.is_file()]
    paths += [D.RAW/'DIMENSION_SIDECAR.npz',D.RAW/'DIMENSION_SIDECAR.json',*[E.ROOT/r['path'] for r in dev['records']],
      *[E.ROOT/r['path'] for r in E.read(D.DOC/'SYNTH_DETECTION_AUDIT.json')['files']],
      *[E.ROOT/r['path'] for r in E.read(D.DOC/'SQUARE_CACHE_COMPLETE.json')['cache_bindings']],
      *[E.ROOT/r['path'] for r in E.read(D.DOC/'SQUARE_CACHE_COMPLETE.json')['files']],
      *[D.RAW/f'orders/MIXED_seed{s}.npy' for s in [1,2,3]],D.R0,D.SYM_DOC/'OBJECT_EQUIVALENCE_AND_INDEX_CONTRACT.json',
      D.SYM_RAW/'A/square_membership.json',D.SYM_RAW/'A/square_annotation_target_view.json']
    E.freeze(E.DOC/'SOURCE_BINDING.json',dict(files=[E.bound(p) for p in sorted(set(paths))],original_source_files=old['files'],paper_cache_stat=old['cache_stat'],
      original_DCP_read_only=True,feature_arrays_read_only=all(not a.flags.writeable for a in paper.arrays.values()) and all(not a.flags.writeable for a in square.arrays.values())))
    (E.DOC/'PURPOSE.md').write_text('# Explicit group-code value isolation\n\nBoth arms use the identical existing 8-D DCP model, canonical dimensions, whole-object symmetry-aware target, initialization, order, optimizer and fixed T=1 decode. Only context[:,5:8] is neutral1/3 or approved one-hot. A: C1/C2 source. B: mixed-domain diagnostic only, no separate square training. C: inference perturbations only. Original DCP code/docs/raw artifacts are read-only.\n\nReal DEV is conditional on externally known registry object type. Its C2 code is constant; it cannot demonstrate cross-group routing. Wood C2 is a benchmark convention, not reviewed physical symmetry. No FINAL, new data, new architecture, extra sweep or deployment overwrite. Decision thresholds are locked before this experiment produces performance.\n')
    print('PREFLIGHT_COMPLETE',flush=True)
if __name__=='__main__':torch.set_num_threads(2);main()
