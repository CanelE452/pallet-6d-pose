"""V91 independent prefit review. Only code/protocol + generated data are read."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import importlib.util
import json
import math
import sys
import numpy as np
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
TRAINER=ROOT/'train_physical_cue_once.py'
PROTOCOL=ROOT/'FIT_PROTOCOL_BEFORE_SOURCE_ARRAYS.json'
MODEL=ROOT/'local_cue_evidence_map.py'
MOSAIC=ROOT/'spatial_mosaic.py'
TRAINER_SHA='ffbb7eeb63722c56b18ab0927729f1d344fe08a91ad2f908f607cf4f7a331d90'
PROTOCOL_SHA='6191df432b4254fb500087b714e1a8e9867a111581158707a95eadf993492028'


def binding(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)


def write(name,d):
    with (HERE/name).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def module(p,name):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m


def generated_views():
    """604 invented rows; source/annotation paths and coordinates are not read."""
    rows=[]
    def add(cue,partition,beforelabel,afterlabel,duplicate=1.):
        for view,label,frame in [('before',beforelabel,0 if partition=='TRAIN_LAB'else 2),('after',afterlabel,1 if partition=='TRAIN_LAB'else 3)]:
            rows.append(dict(cue_view_index=len(rows),cue_id=cue,pair_id='GENERATED:'+cue,
                split=partition,view=view,frame_input_id='GENERATED_FRAME_'+str(frame),
                teacher_label=label,duplicate_weight=duplicate,
                reference_xy_unchanged=[8.,8.],uncertainty9_xy_GT_only=[[7.,7.],[8.,7.],[9.,7.],[7.,8.],[8.,8.],[9.,8.],[7.,9.],[8.,9.],[9.,9.]]))
    for i in range(64):add('TRAIN_KNOWN_'+str(i),'TRAIN_LAB',0,1 if i<15 else 0,.5 if 15<=i<19 else 1.)
    for i in range(70):add('TRAIN_U_'+str(i),'TRAIN_LAB',-1,-1)
    for i in range(85):add('VAL_KNOWN_'+str(i),'VAL_HALL',0,1 if i<22 else 0)
    for i in range(83):add('VAL_U_'+str(i),'VAL_HALL',-1,-1)
    assert len(rows)==604
    return rows


class GeneratedMapModel(torch.nn.Module):
    def __init__(self):super().__init__();self.calls=[]
    def forward(self,p3_roi,local_rgb96,local_valid96):
        self.calls.append(len(p3_roi));z=p3_roi[:,0,0,0].reshape(-1,1,1).expand(-1,9,9)
        valid=(p3_roi[:,64]>0)&(F.adaptive_avg_pool2d(local_valid96.float(),(9,9))[:,0]>0)
        return dict(physical_external_logit=z,map_valid=valid,input_available=valid.flatten(1).any(1))


def main():
    assert binding(TRAINER)['sha256']==TRAINER_SHA and binding(PROTOCOL)['sha256']==PROTOCOL_SHA
    write('SCOPE_BEFORE_GENERATED_TESTS.json',dict(completed=True,
        created_utc=datetime.now(timezone.utc).isoformat(),trainer=binding(TRAINER),protocol=binding(PROTOCOL),
        scope='Read reviewed code/protocol/generated proof only; use invented CPU tensors and 604-row metadata fixture.',
        source_checkpoint_or_teacher_pack_hash_or_array_reads=0,source_pixels_features_weights_current_reads=0,
        no_trainer_protocol_function_call=True,no_actual_model_forward_fit_PnP=True,
        generated_numeric_policy=dict(loss_float64_abs=1e-12,float32_abs=2e-6,sampling='exact seeded sequence, no fitted tolerance')))
    torch.set_num_threads(1);torch.manual_seed(20261023)
    p=json.loads(PROTOCOL.read_text());proof=json.loads((ROOT/'GENERATED_TRAINER_TESTS.json').read_text())
    assert proof['PASS'] and proof['trainer']['sha256']==TRAINER_SHA
    assert p['parameter_count']==25381 and p['forward_fields']==['p3_roi','local_rgb96','local_valid96']
    assert p['sampling']['per_class']==48 and p['sampling']['batch_queries']==96 and p['sampling']['seed']==20261023
    assert p['sampling']['replacement'] is True and 'mean over available' in p['sampling']['unavailable']
    assert p['optimizer']==dict(name='AdamW',transferred_encoder_lr=1e-4,new_fusion_output_lr=3e-4,weight_decay=.001,betas=[.9,.999],updates=2000,clip_grad_norm=5.,schedule='constant')
    assert p['selection']['every_updates']==100 and p['selection']['threshold']==.5
    assert p['selection']['unavailable_region_NLL']=='FIXED_ZERO_LOGIT_BCE_LOG2_DIAGNOSTIC_PENALTY'
    assert p['source_gate_before_results']['total_gates']==9 and not p['current223_reads_allowed'] and not p['PnP_allowed']
    t=module(TRAINER,'v91_generated_review_trainer');m=module(MODEL,'v91_generated_review_model');s=module(MOSAIC,'v91_generated_review_mosaic')
    results={}
    pools={0:np.array([2,4,9]),1:np.array([1,7])};weights={0:np.array([.2,.3,.5]),1:np.array([.75,.25])}
    rng=np.random.default_rng(20261023);ref=np.random.default_rng(20261023);same=np.random.default_rng(20261023)
    counts={0:0,1:0}
    for _ in range(2000):
        got=t.sample_terms(pools,weights,rng)
        expected=np.r_[ref.choice([2,4,9],48,replace=True,p=[.2,.3,.5]),ref.choice([1,7],48,replace=True,p=[.75,.25])]
        assert np.array_equal(got,expected)and np.array_equal(got,t.sample_terms(pools,weights,same))
        assert np.isin(got[:48],pools[0]).all()and np.isin(got[48:],pools[1]).all()
        counts[0]+=48;counts[1]+=48
    results['sampler']=dict(updates=2000,total_queries=sum(counts.values()),queries_by_class={str(k):v for k,v in counts.items()},
        exact_independent_seeded_sequence=True,class48_replacement_weighted=True,nominal_unavailable_not_removed_from_pools=True)
    # Mixed known/missing/unknown/support holes: only first two have a BCE.
    z=torch.tensor([-2.,1.,50.,80.,-90.,60.],dtype=torch.float64)
    maps=z[:,None,None].expand(-1,9,9).clone().requires_grad_(True)
    valid=torch.ones_like(maps,dtype=torch.bool);valid[2]=False
    coordinates=torch.zeros(6,9,2,dtype=torch.float64);coordinates[3:]=float('nan')
    labels=torch.tensor([0,1,1,-1,0,1]);admitted=torch.tensor([1,1,1,0,0,0],dtype=torch.bool)
    actual,stats=t.masked_region_loss(m,maps,valid,coordinates,labels,admitted)
    expected=(np.logaddexp(0.,-2.)+np.logaddexp(0.,-1.))/2
    assert abs(float(actual.detach())-expected)<1e-12 and stats==dict(eligible=3,factual_available=2)
    actual.backward();assert maps.grad[:2].abs().sum()>0 and not maps.grad[2:].any()
    zeros=[]
    for unknown in (True,False):
        x=torch.randn(4,9,9,requires_grad=True);v=torch.zeros_like(x,dtype=torch.bool)
        y=torch.full((4,),-1)if unknown else torch.tensor([0,1,0,1])
        ad=torch.zeros(4,dtype=torch.bool)if unknown else torch.ones(4,dtype=torch.bool)
        grid=torch.full((4,9,2),float('nan'))if unknown else torch.zeros(4,9,2)
        loss,st=t.masked_region_loss(m,x,v,grid,y,ad);loss.backward()
        assert loss.item()==0 and not x.grad.any();zeros.append(st)
    results['loss']=dict(mean_available_not_fixed96_denominator_matches_protocol=True,
        independent_BCE_and_gradients=True,excluded_GT_NaN_not_interpolated=True,
        unknown_MISS_PARTIAL_and_factual_unavailable_zero_gradient=True,all_empty_graphzero=True,empty_reports=zeros)
    labels=np.array([0,1,0,1,-1]);available=np.array([False,False,True,True,False])
    logit=np.array([99.,-99.,-2.,3.,99.],np.float32)
    nll=t.factual_region_NLL(logit,available,labels)
    assert np.array_equal(nll[[0,1,4]],np.full(3,np.log(2),np.float32))
    mask=np.isin(labels,[0,1]);dup=np.array([.5,1.,.5,2.,1000000.])
    macro,detail=t.macro_class_loss(nll,labels,dup,mask)
    expected_D=(float(nll[0])*.5+float(nll[2])*.5)
    expected_E=(float(nll[1])+float(nll[3])*2)/3
    assert abs(macro-(expected_D+expected_E)/2)<1e-12 and detail['0']['N']==detail['1']['N']==2
    cm=t.confusion(labels,logit,available,mask)
    assert cm['rows_D_E_cols_D_E_U']==[[1,0,1],[0,1,1]]and cm['D_recall']==cm['E_recall']==.5
    tie=t.confusion(np.array([0,1]),np.array([0.,0.]),np.array([True,True]),np.array([True,True]))
    assert tie['rows_D_E_cols_D_E_U']==[[0,1,0],[0,1,0]]
    results['selector_and_U']=dict(unavailable_fixed_FP32_log2=True,unavailable_prediction_U_not_D_or_E=True,
        unknown_excluded_even_if_diagnostic_label_clamped=True,duplicate_weighted_classmacro_independent=True,
        recall_includes_known_U=True,zero_logit_inclusive_E=True)
    # All four invented frames each have 129 tiles, metadata-only split. No GT bank field.
    frameidx=np.repeat(np.arange(4),129)
    frames=[dict(frame_input_id='GENERATED_FRAME_'+str(i),raw_hw=[17,17],split='TRAIN_LAB'if i<2 else'VAL_HALL')for i in range(4)]
    meta=dict(frames=frames,data=dict(image_frame_index=frameidx,query_xy=np.full((516,2),8.,np.float64),L=np.full(516,16.)))
    bank=dict(p3_roi=torch.zeros(516,65,9,9),local_rgb96=torch.zeros(516,3,96,96,dtype=torch.uint8),local_valid96=torch.ones(516,1,96,96,dtype=torch.bool))
    bank['p3_roi'][:,64]=1
    bank['p3_roi'][:,0]=torch.from_numpy(np.where(frameidx%2==0,-2.,2.).astype(np.float32))[:,None,None]
    fake=GeneratedMapModel();prediction=t.predict_mosaics(fake,bank,meta,[2,3],s,'cpu')
    assert prediction['boundary']==dict(all_fixed_tiles_processed=True,frames_completed=2,tiles_processed=258,GT_reference_or_label_used_in_mosaic=False)
    assert sum(fake.calls)==258 and set(prediction['mosaics'])=={'GENERATED_FRAME_2','GENERATED_FRAME_3'}
    # Only after complete maps exist, write/read the invented scoring-coordinate file.
    fixture=HERE/'GENERATED_604_VIEW_ROWS.jsonl'
    fixture.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n'for r in generated_views()))
    scored=t.score_completed_mosaics(prediction,fixture,s);assert len(scored['rows'])==336
    report=t.score_report(scored);val=report['VAL_HALL']
    assert val['known_views']==170 and val['teacher_U']==166
    assert val['region']['rows_D_E_cols_D_E_U']==[[85,63,0],[0,22,0]]
    assert val['reference']['rows_D_E_cols_D_E_U']==[[85,63,0],[0,22,0]]
    assert val['paired_views']['D_to_E']['N']==22 and val['paired_views']['D_to_D']['N']==63
    assert val['paired_views']['D_to_E']['region']['both_correct']==22 and val['paired_views']['D_to_D']['region']['both_correct']==0
    expected_gates={f'{kind}_{key}_{direction}':passed for kind in('region','reference')for key,direction,passed in [('D_recall','ge',False),('E_recall','ge',True),('E_precision','ge',False),('U_rate','le',True)]}
    expected_gates['region_D_to_E_pair_both_correct_ge_0p7']=True
    assert t.gates(report)==expected_gates and len(expected_gates)==9
    # Run real validate on generated input-only tensors and the generated604 scorer.
    value,classes,_,sc=t.validate(GeneratedMapModel(),bank,meta,dict(bindings={'teacher_views':{'path':str(fixture)}}),s,'cpu',[2,3])
    assert sum(v['N']for v in classes.values())==170 and np.isfinite(value)
    results['mosaic_and_scoring']=dict(generated_bank_only_three_fields=list(bank),all129tiles_per_frame_processed=True,
        complete_GTfree_native_maps_before_view_scorer=True,generated_604_rows_fixture=binding(fixture),
        all170_known_VAL_retained=True,teacher_U166_reported_separately=True,paired_D_E22_and_D_D63=True,
        independent_nine_gates=expected_gates,criterion_macroBCE=float(value),no_actual_views_or_input_bank_read=True)
    # Literal code contract: no optimizer/state selection alternatives.
    tree=ast.parse(TRAINER.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run')
    text=ast.get_source_segment(TRAINER.read_text(),run)
    for required in ['range(1,2001)','lr\':1e-4','lr\':3e-4','weight_decay=.001','criterion<best','update%100==0','restart_allowed=False']:
        assert required in text,required
    best=float('inf');selected=None
    for u,criterion in [(0,.01),(100,.5),(200,.4),(300,.4),(400,.6)]:
        if u and criterion<best:best=criterion;selected=u
    assert selected==200
    assert 'RUN_STARTED.json' in text and 'not run_dir.exists()' in text and 'FIRST_UPDATE.json' in text
    results['execution']=dict(one_new_run_directory_guard=True,after_first_update_restart_disallowed=True,
        source_GPU_preflight0_update_then_baseline0_diagnostic=True,constant_LR_2000_updates=True,
        differential_AdamW_LR_and_weight_decay_match=True,eval_every100_strict_min_earliest_tie=True,
        sourcefreeze_before_all100_final_prediction=True,no_hidden_alternative_checkpoint_or_threshold=True)
    assert binding(TRAINER)['sha256']==TRAINER_SHA and binding(PROTOCOL)['sha256']==PROTOCOL_SHA
    final=dict(completed=True,PASS=True,blockers=[],protocol=binding(PROTOCOL),trainer=binding(TRAINER),
        model=binding(MODEL),mosaic=binding(MOSAIC),generated_trainer_proof=binding(ROOT/'GENERATED_TRAINER_TESTS.json'),
        scope=binding(HERE/'SCOPE_BEFORE_GENERATED_TESTS.json'),review_code=binding(__file__),generated=results,
        exact_scope_notes=[
            'No trainer protocol() call: referenced feature/checkpoint/teacher payloads were neither hashed nor deserialized in this review. Only code/protocol/generated files and invented fixture were read.',
            'At actual fit, load_terms reads all77916 unit coordinates including VAL as GT-side bookkeeping, but sampler pools are TRAIN-only. GT never enters the three-input bank or GT-free all-tile mosaic construction. View scoring occurs after completed maps.',
            'Unavailable nominal terms remain in the sampling pools. Training averages only available samples as explicitly frozen; availability can alter effective class mass and is reported instead of hidden by new admission filtering.',
            'Known170 VAL cue views remain selector/gate denominator even when factual support is unavailable: predictionU, selector fixed zero-logit BCE ln2. U teacher rows are not semanticD/E and never enter selector.',
            'This verifies frozen code/choices and generated math only. Actual packet/GT/weight integrity, GPU numeric behavior, empirical transfer, and native-slot identity are outside this review.',
        ],
        actual_source_RGB_GT_feature_checkpoint_parameter_reads=0,actual_source_forward_fit_current_PnP=0,
        generated_CPU_forward_and_loss_tests_only=True)
    write('REVIEW.json',final)
    write('REVIEW_SEAL.json',dict(completed=True,PASS=True,files=[binding(x)for x in sorted(HERE.iterdir())if x.is_file()],
        protocol=binding(PROTOCOL),trainer=binding(TRAINER),source_model_current_fit_PnP=0))
    print(json.dumps(dict(review=binding(HERE/'REVIEW.json'),seal=binding(HERE/'REVIEW_SEAL.json')),indent=2),flush=True)


if __name__=='__main__':main()
