"""Bind the two prospectively fixed SYN-only refiners before any new fit."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).absolute().parent
PREV=HERE.parent/'V267_FULL_IMAGE_CLIP_VISIBLE_MASK'


def safe(p):
    p=Path(p)
    assert p.is_absolute()and not any(x in str(p).lower()for x in('v112','cache','.pyc','..')),str(p)
    for a in reversed((p,*p.parents)):assert not a.is_symlink(),str(a)
    return p


def binding(p):
    p=safe(p);h=hashlib.sha256()
    with p.open('rb')as f:
        for raw in iter(lambda:f.read(8*1024*1024),b''):h.update(raw)
    return dict(path=str(p),sha256=h.hexdigest(),size_bytes=p.stat().st_size)


def checked(b):
    assert binding(b['path'])==b
    return safe(b['path'])


def load(p):
    return json.loads(safe(p).read_bytes())


def main():
    out=safe(HERE/'PROTOCOL_BEFORE_FIT.json');assert not out.exists()
    prep=load(HERE/'syn_refine_inputs_v1/COMPLETE.json');assert prep['PASS']and prep['FIT_DEV']==[768,235]
    assert prep['current_or_real_reads']==prep['M0_or_query_or_pose_reads']==prep['optimizer_updates']==0
    assert not safe(HERE/'syn_training_v1').exists()
    oldfreeze=load(PREV/'CURRENT_PREDICTIONS_FROZEN.json')
    oldcfg=load(checked(oldfreeze['configuration']))
    oldprotocol=load(checked(oldcfg['protocol']))
    b=prep['outputs']
    sources=dict(RGB256=b['RGB256_FP16.npy'],tokensCS=prep['source_tokens'],coarse_logits=b['COARSE_LOGITS_CS.npy'],
        targets=b['MASK_TARGETS.npz'],draws=b['FIT_DRAWS1000x8.npy'])
    for source in sources.values():checked(source)
    d=dict(completed=True,PASS=True,schema='V268_TWO_FINAL1000_FULL_IMAGE_REFINERS',
        design=binding(HERE/'DESIGN_BEFORE_PREPARATION.json'),input_preparation=binding(HERE/'syn_refine_inputs_v1/COMPLETE.json'),
        model_code=binding(HERE/'model_mask_refine_v268.py'),training_code=binding(HERE/'train_mask_refine_v268.py'),
        inference_code=binding(HERE/'infer_fullframe_v268.py'),evaluation_code=binding(HERE/'evaluate_fullframe_v268.py'),
        sealing_code=binding(Path(__file__).absolute()),train_arms=['HIGHRES_REFINE','BOUNDARY_REFINE'],
        seed=17,optimizer_updates_each=1000,batch_size=8,optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001),
        boundary_radius=2,boundary_loss_weight=.5,delta_bound=6.,threshold=.5,
        FIT_foreground_frequency=prep['FIT_foreground_frequency'],positive_weight=prep['positive_weight'],negative_weight=prep['negative_weight'],
        source_bindings=sources,supervision='OriginalbinaryvisibleM4only;rawmask→full512AA→pool2soft256.NoM0/query/native/imagecrop.',
        loss_shared='FITclassbalancedweightedBCE+softDice',loss_only_arm_difference='BOUNDARY_REFINE adds.5balancedpositive/negativeboundaryBCE;GTbandradius2at256doesnotaltertarget.',
        architecture='FrozenCS32tokens+frozenV267coarse32+fullRGB256;26721oldCSheadnotoptimized;fullframerefinerzerooutputinit; total=bilinearcoarse256+6tanh(rawdelta/6).',
        model_parameters_init_and_draws_identical=True,frozen_backbone_or_coarse_head_updates=0,
        no_checkpoint_threshold_or_recipe_selection=True,checkpoint_policy='Bothfinal1000firstsealedbeforeSYNDEV;bothreported.Noextraautomaticfits.',
        current_input_helper=oldprotocol['current_input_helper'],current_input_complete=oldprotocol['current_input_complete'],
        current_CS_tokens=oldfreeze['feature_bindings']['FULL_CLIP_SURGERY_ARCH'],V267_current_frame_masks=oldfreeze['frame_masks'],
        prior_V267_predictions=oldfreeze['predictions'],prior_V267_configuration=oldfreeze['configuration'],
        current_source_bindings_from_prior_sealed_metadata_only=True,
        query_readout='Afterbothfullmapssaved;sameinitialnativeq→raw256logitbilinearalignFalseborder;D>=0/E<0,45oldunavailableU;selfseparate.',
        no_op_resampling_diagnostic='Report32directlookupvs32→256→q numericaldifferencebeforeGT;notaselectablepolicy.',
        success_assessment='CompareDmiss552recovery/lossandretentionof84priorEcorrect;EandDprecision/recall/Ubothgroups;manualDinitial<=3subset;nonewgatefitted.',
        current223_fit_or_threshold_or_checkpoint_selection=0,current223_is_repeated_development_evaluation=True,
        new_location_training_or_PnP=0,mask_presence_is_not_external_cause=True,real_pixel_GT_unavailable=True)
    assert prep['model_code']==d['model_code']
    with out.open('x')as f:json.dump(d,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,protocol=binding(out))),flush=True)


if __name__=='__main__':main()
