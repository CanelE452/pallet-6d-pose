"""Freeze historical panel and prior contracts before any new network training."""
from env import *
def run():
    start=now()
    if complete('BIND_COMPLETE'): verify(); print('BIND_REUSED'); return
    assert subprocess.check_output(['git','branch','--show-current'],text=True).strip()=='main'
    old_binding=read(OLD_DOC/'SOURCE_BINDING.json')
    files={r['path']:r for r in old_binding['files']}
    for root in (OLD_DOC,OLD_CODE,ROOT/'_docs/paper/sensors_refinement_closeout_v1'):
        for p in sorted(root.rglob('*')):
            if p.is_file() and '.git' not in p.parts: files[str(p.relative_to(ROOT))]=bound(p)
    for p in sorted(OFFICIAL.rglob('*.py')): files[str(p.relative_to(ROOT))]=bound(p)
    freeze(DOC/'SOURCE_BINDING.json',dict(start_main=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),files=list(files.values()),cache_arrays=old_binding['cache_arrays'],reused=['R0/P/L/D training and raw DEV/pose outputs','P-R0/P-D paired statistics','historical runtime retained'],new=['PoseFix-derived pallet9','new prior DEV comparison','same-session runtime if prior ready','confirmation if independently supplied','compiled manuscript']))
    verify()
    freeze(DOC/'MODEL_PANEL_LOCK.json',dict(existing_method=read(OLD_DOC/'METHOD_FREEZE.json'),D=read(OLD_DOC/'D_SELECTION.json'),L_selection=bound(LINE/'SELECTION.json'),representative_P_seed=1,PRIOR_seeds=[1,2,3],PRIOR_status='UNTRAINED_NO_CONFIRMATION_UNBLINDING',existing_retraining=0))
    freeze(DOC/'PRIOR_PROTOCOL_LOCK.json',dict(name='PoseFix-derived pallet9, matched-exposure adaptation',source_commit='5556364bb0f43b0743a5fcd820de48f34b3d4360',prior_design=read(OLD_DOC/'PREDECLARED_PRIOR_BASELINE_PROTOCOL.json'),seeds=[1,2,3],steps=6000,batch=16,source_order='exact P order_seed*.npy',lr=.0005,betas=[.9,.999],epsilon=1e-8,L2_conv_weight_penalty_half_scale=1e-5,decay_updates=[3858,5143],clip='NOT_USED',augmentation='fixed source-image crop; no human pose noise/flip/scale/rotation; matched P exposures',crop=[384,288],heatmap=[96,72],input_maps_amplitude=255,input_sigma=9,RGB=True,center_loss=False,center_restored=True,scalar_atol=1e-5,scalar_rtol=1e-4,network_atol=3e-4,network_rtol=3e-4,network_relative_l2_max=1e-4,smoke_max_updates=100,checkpoint_policy='last6000 only; durable optimizer/RNG/order cursor',no_DEV_selection=True,amendments=['Official gen_batch.py currently leaves BGR conversion commented out; requested RGB is an explicit purpose adaptation, not exact loader reproduction.','Official trainer includes L2 over conv weights, not AdamW weight decay.','TF1 Adam epsilon is applied to uncorrected sqrt second moment; preserve its update formula.','Safe bilinear targets retain only in-range scatter indices and normalize; crop-outside/nonfinite GT gets no training supervision, never removed from evaluation GT.']))
    p=DOC/'BIND_COMPLETE.json'
    receipt('BIND_COMPLETE',[HERE/'preflight.py'],[DOC/'SOURCE_BINDING.json',DOC/'MODEL_PANEL_LOCK.json',DOC/'PRIOR_PROTOCOL_LOCK.json'],start,preserved_files=len(files))
    print('BOUND',len(files),flush=True)
if __name__=='__main__': run()
