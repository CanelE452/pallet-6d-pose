from context24 import *
def main():
    for f in ('AUDIT.json','TINY_AUDIT.json','INTEGRATION_CHECK.json','DELIVERY_CHECK.json'):assert read(RUN/f)['status']=='PASS'
    assert read(RUN/'BROWSER_OPEN.json')['status']=='OPEN_CONFIRMED';s=read(RUN/'SUMMARY.json');t=read(RUN/'TINY_COMPLETE.json');cl=read(RUN/'CLASSIFICATION.json');assert cl['bootstrap_known_vector_pass']
    neural=read(RUN/'TRAINING_COMPLETE.json')['updates']+t['neural_updates'];assert neural==54000 and t['privileged_lookup_updates']==6000
    for seed in SEEDS:
        for a in ARMS:assert (RUN/'models'/f'{a}_seed{seed}'/'final.pt').exists() and (RUN/'export'/f'{a}_seed{seed}.pt').exists()
        for a in TINY_ARMS:assert (RUN/'tiny_models'/f'{a}_seed{seed}'/'final.pt').exists()
    conclusion=dict(stable_fusion_resolved=bool(s['stable_arms']),stable_arms=s['stable_arms'],tiny_memorization_all_pass=t['all_memorization_pass'],real_crop20_direction_signal=cl['gates']['crop20'],scope='Small-data learning succeeds; direction discrimination fails transfer to real mild cropping under this probe. No information-theoretic impossibility or elimination of all large-data optimization concerns.',operating_model_changed=False)
    write(RUN/'CONCLUSION.json',conclusion)
    files=list(KIT.glob('*.py'))+list(KIT.glob('*.md'))+list((RUN/'models').glob('*/final.pt'))+list((RUN/'tiny_models').glob('*/final.pt'))+list((RUN/'tiny_models').glob('*/early2000.pt'))+list((RUN/'export').glob('*.pt'))+list(RUN.glob('LOOKUP_seed*.pt'))+list((RUN/'figures').glob('*.png'))+[RUN/f for f in ('PROTOCOL.json','TRAIN_PROTOCOL.json','TINY_TRAIN_PROTOCOL.json','TRAINING_COMPLETE.json','TINY_COMPLETE.json','SUMMARY.json','CLASSIFICATION.json','CONCLUSION.json','AUDIT.json','TINY_AUDIT.json','INTEGRATION_CHECK.json','EVALUATION_BINDINGS.json','REPORT_KO.md','DESIGN_KO.md','INTERPRETATION_KO.md','index.html','effects.png','GALLERY.json','REPORT_COMPLETE.json','DELIVERY_CHECK.json','BROWSER_OPEN.json')]
    write(RUN/'COMPLETION.json',dict(complete=True,experiment_completed=True,neural_models=15,neural_updates=neural,privileged_lookup_controls=3,privileged_lookup_updates=6000,real_GT_training=False,new_real_holdout=False,html_open_confirmed=True,**conclusion,files={str(p.relative_to(ROOT)):sha(p) for p in files}));print('COMPLETION',len(files),'bound files',flush=True)
if __name__=='__main__':main()
