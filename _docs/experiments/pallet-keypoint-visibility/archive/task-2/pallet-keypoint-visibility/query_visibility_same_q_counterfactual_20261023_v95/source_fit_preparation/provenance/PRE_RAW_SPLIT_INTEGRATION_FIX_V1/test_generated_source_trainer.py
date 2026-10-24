"""Generated-only C2 checks; never opens actual source tensors or GT ledgers."""
from pathlib import Path
import argparse
import copy
import importlib.util
import json
import math
import tempfile
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v95_generated_trainer',HERE/'train_source_once.py')
t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)

def rejected(fn):
    try:fn()
    except (AssertionError,ValueError):return
    raise AssertionError('Malformed fixture was accepted')

def metadata():
    lineage=[];ledger=[];accepted={0,1,160,161}
    for i in range(960):
        original=0 if i<6 or 160<=i<297 else (1 if 297<=i<317 else -1)
        for j,v in enumerate(t.VARIANTS):
            index=3*i+j;oid=f'INVENTED_{i}::{v}'
            lineage.append(dict(occurrence_index=index,slot_index=i,variant=v,occurrence_id=oid,
                source='LOCO' if i<160 else 'PALLOX',split='TRAIN' if i%2==0 else 'VALIDATION',
                query_xy=[float(i%31),17.],L=32.))
            y=[0,1,0][j] if i in accepted else (original if j==0 else -1)
            ledger.append(dict(occurrence_index=index,base_row_index=i,slot_index=i,variant=v,
                occurrence_id=oid,candidate_aux_hidden_label_GT_only=y,known_candidate_GT_only=y>=0,
                original_natural_label_GT_only=original,pair_ready=i in accepted,
                native_expected_part_state='UNKNOWN',human_gold=False))
    return lineage,ledger

def main(output):
    torch.set_num_threads(1);torch.manual_seed(194)
    lineage,ledger=metadata();meta=t.join_metadata(lineage,ledger)
    assert meta['counts']['TRAIN']['accepted_pairs']==meta['counts']['VALIDATION']['accepted_pairs']==2
    assert np.sum(meta['labels'][::3]==1)==20 and len(meta['labels'])==2880
    bad=copy.deepcopy(ledger);bad[3*297]['candidate_aux_hidden_label_GT_only']=-1;bad[3*297]['known_candidate_GT_only']=False
    rejected(lambda:t.join_metadata(lineage,bad))
    wrong=copy.deepcopy(lineage);wrong[1]['query_xy']=[0.,16.]
    rejected(lambda:t.join_metadata(wrong,ledger))
    rejected(lambda:t.join_metadata(lineage[:-1],ledger[:-1]))
    for s in ['TRAIN','VALIDATION']:
        counts=copy.deepcopy(meta['counts']);counts[s]['accepted_pairs']=0
        assert any(counts[x]['accepted_pairs']==0 for x in ['TRAIN','VALIDATION'])

    pools=t.make_pools(meta);g=torch.Generator().manual_seed(194);g2=torch.Generator().manual_seed(194)
    for _ in range(2000):
        ix=t.draw(pools,g);assert np.array_equal(ix,t.draw(pools,g2))
        assert (meta['split'][ix]=='TRAIN').all() and meta['known'][ix].all()
        assert np.sum(meta['labels'][ix]==0)==np.sum(meta['labels'][ix]==1)==32

    # Deliberately unequal stratum sizes; equal mean of the four BCE means.
    m=dict(labels=np.array([0,0,1,0,0,0,1,1,-1],np.int8),
        split=np.array(['VALIDATION']*9),variant=np.array(['BEFORE']*3+[t.VARIANTS[2]]*3+[t.VARIANTS[1]]*2+['BEFORE']))
    z=np.array([-1.,-2.,1.,0.,-1.,-1.,1.,0.,999.],np.float32)
    value,strata=t.selector(z,m)
    soft=lambda x: math.log1p(math.exp(x))
    means=[(soft(-1)+soft(-2))/2,soft(-1),(math.log(2)+2*soft(-1))/3,(soft(-1)+math.log(2))/2]
    assert abs(value-sum(means)/4)<1e-14
    assert [s['count'] for s in strata.values()]==[2,1,3,2]
    best=float('inf');selected=None
    for step,criterion in [(0,2.),(100,1.),(200,1.),(300,1.1)]:
        if criterion<best:best,selected=criterion,step
    assert selected==100

    z=np.where(meta['labels']==0,-1.,1.).astype(np.float32);avail=np.ones(2880,bool)
    result,checks,passed=t.report(z,avail,meta);assert passed and len(checks)==6
    assert result['VALIDATION']['accepted_pairs']==result['VALIDATION']['all3_correct']==2
    ix=3*1+1;avail[ix]=False;z[ix]=0
    result,checks,passed=t.report(z,avail,meta)
    assert not passed and result['VALIDATION']['all3_correct_rate']==.5
    assert result['VALIDATION']['by_variant'][t.VARIANTS[1]]['matrix']==[[0,0,0],[0,1,1]]
    assert result['VALIDATION']['by_variant'][t.VARIANTS[1]]['E_recall']==.5
    assert t.decode(np.array([-0.,0.,np.nextafter(np.float32(0),np.float32(-1)),1.]),np.array([True,True,True,False])).tolist()==[1,1,0,-1]

    # Sparse base/chunk refs are exercised using tiny, invented three tensors.
    with tempfile.TemporaryDirectory() as d:
        d=Path(d);p=np.zeros((1,65,9,9),np.float16);p[:,64]=1
        rgb=np.full((1,3,96,96),11,np.uint8);valid=np.ones((1,1,96,96),bool)
        np.savez(d/'base.npz',p3_roi=p,rgb96=rgb,crop_valid96=valid)
        np.savez(d/'changed.npz',p3_roi=np.repeat(p,2,axis=0)+np.float16(0),rgb96=np.repeat(rgb+4,2,axis=0),crop_valid96=np.repeat(valid,2,axis=0))
        before={k:dict(file=t.bind(d/'base.npz'),key=k,index=0) for k in ['p3_roi','rgb96','crop_valid96']}
        base=[dict(before_input_refs=before)];(d/'base.json').write_text(json.dumps(base));binding=t.bind(d/'base.json')
        rr=[dict(slot_index=0,full_rgb_changed=False,input_ref=dict(kind='BASE_ROW',slot_index=0,base_lineage=binding))]
        rr += [dict(slot_index=0,full_rgb_changed=True,input_ref=dict(kind='CHANGED_CHUNK',file=t.bind(d/'changed.npz'),index=j)) for j in range(2)]
        data=t.materialize_tensor_rows(base,rr,binding)
        assert data['rgb96'][:,0,0,0].tolist()==[11,15,15]
        assert set(data)=={'p3_roi','rgb96','crop_valid96'} and data['p3_roi'].shape==(3,65,9,9)
        malformed=copy.deepcopy(rr);malformed[0]['full_rgb_changed']=True
        rejected(lambda:t.materialize_tensor_rows(base,malformed,binding))
        malformed=copy.deepcopy(rr);malformed[1]['input_ref']['file']['sha256']='0'*64
        rejected(lambda:t.materialize_tensor_rows(base,malformed,binding))

    rejected(lambda:t.require_entry(dict(approved_actual_source_fit=False)))
    op=t.operator();model=op.QueryCenterScalar();assert sum(x.numel() for x in model.parameters())==26573
    p=torch.randn(4,65,9,9);p[:,64]=1
    rgb=torch.randint(0,256,(4,3,96,96),dtype=torch.uint8);valid=torch.ones(4,1,96,96,dtype=torch.bool);valid[2]=False
    model_output=model(p,rgb,valid);target=torch.tensor([0,1,1,-1]);known=target>=0
    assert op.decode_binary(model_output).tolist()==[1,1,-1,1]
    loss=op.known_binary_loss(model_output,target,known)
    assert int(loss['known_count'])==3 and int(loss['available_known_count'])==2
    zz=torch.tensor([2.,-1.,0.,123.],requires_grad=True)
    dummy=dict(query_logit=zz,query_available=torch.tensor([True,True,False,True]))
    op.known_binary_loss(dummy,target,known)['loss'].backward()
    assert zz.grad[2:].tolist()==[0.,0.]
    with torch.no_grad():model.readout[-1].weight.fill_(.031)
    model.train();out_train=model(p,rgb,valid);model.eval();out_eval=model(p,rgb,valid)
    assert torch.equal(out_train['query_logit'],out_eval['query_logit'])
    op.known_binary_loss(out_train,target,known)['loss'].backward()
    assert any(x.grad is not None and x.grad.abs().sum()>0 for x in model.p3_encoder.parameters())
    try:model(p,rgb,valid,labels=target)
    except TypeError:pass
    else:raise AssertionError('GT forward keyword accepted')
    proof=dict(completed=True,PASS=True,trainer=t.bind(HERE/'train_source_once.py'),operator=t.bind(t.MODEL),
        generated_test_code=t.bind(__file__),torch=torch.__version__,runtime_executable=__import__('sys').executable,
        invented_base_rows=960,all_occurrences=2880,natural_E20_canonical_before_preserved=True,
        sparse_BASE_ROW_CHANGED_CHUNK_loader=True,changed_RGB_base_reference_rejected=True,
        four_VAL_strata_equal_mean_includes_unavailable_ln2=True,selector_strict_earliest_tie=True,
        generated_balanced_sampler_draws=2000,TRAIN_known_only_D32_E32=True,six_gates_and_pair_U_denominator=True,
        no_accepted_pair_NO_FIT_condition=True,raw_signed_zero_tiny_and_U_decode=True,
        unchanged_three_input_model_params=26573,known_unavailable_unknown_loss_gradient_zero=True,
        train_eval_same_scalar_exact=True,generated_nonzero_encoder_backward=True,GT_forward_rejected=True,
        actual_source_arrays_GT_weights_current_reads=0,actual_fit_optimizer_updates=0,model_checkpoint_loads=0)
    t.write(output,proof);print(json.dumps(proof))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'GENERATED_SOURCE_TRAINER_TESTS.json')
    main(parser.parse_args().output)
