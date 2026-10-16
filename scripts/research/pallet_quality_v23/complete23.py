"""Record experiment completion separately from unresolved scientific objective."""
from context23 import *
from eval23 import ARMS

def main():
    for f in ('AUDIT.json','INTEGRATION_CHECK.json','DELIVERY_CHECK.json'):assert read(RUN/f)['status']=='PASS'
    assert read(RUN/'BROWSER_OPEN.json')['status']=='OPEN_CONFIRMED';assert read(RUN/'REPORT_COMPLETE.json')['complete'];assert read(RUN/'DIAGNOSIS.json')['weight_independent_replay']=='PASS'
    total=sum(read(RUN/f)['updates'] for f in ('TRAINING_COMPLETE.json','MASS_TRAINING_COMPLETE.json'));assert total==36000
    for seed in SEEDS:
        for arm in ARMS:assert (RUN/'models'/f'{arm}_seed{seed}'/'final.pt').exists() and (RUN/'export'/f'{arm}_seed{seed}.pt').exists()
    s=read(RUN/'SUMMARY.json');files=list(KIT.glob('*.py'))+list(KIT.glob('*.md'))+list((RUN/'models').glob('*/final.pt'))+list((RUN/'export').glob('*.pt'))+list((RUN/'figures').glob('*.png'))+[RUN/f for f in ('index.html','effects.png','REPORT_KO.md','DESIGN_KO.md','INTERPRETATION_KO.md','PROTOCOL.json','MASS_PROTOCOL.json','TRAIN_PROTOCOL.json','TRAINING_COMPLETE.json','MASS_TRAINING_COMPLETE.json','TRAIN_QUALITY.json','MASS_WEIGHTS.json','SUMMARY.json','AUDIT.json','DIAGNOSIS.json','MATCHED_DIAGNOSIS.json','INTEGRATION_CHECK.json','SPLIT_AUDIT.json','TARGET_AUDIT.json','IMAGE_BINDINGS.json','EVALUATION_BINDINGS.json','GALLERY.json','REPORT_COMPLETE.json','DELIVERY_CHECK.json','BROWSER_OPEN.json')]
    write(RUN/'COMPLETION.json',dict(complete=True,experiment_completed=True,stable_fusion_resolved=bool(s['stable_arms']),stable_arms=s['stable_arms'],models=len(ARMS)*len(SEEDS),updates=total,new_source_originals=640,new_train_requested=512,new_synthetic_check=128,training_originals_usable=764,observations=4409,real_GT_training=False,operating_model_changed=False,html_open_confirmed=True,interpretation='Data expansion/count balance insufficient; movement coefficient balance partially improves mild crop but worsens large crop. Direction distribution shift persists after posthoc point-ID/5px-error-bin standardization. Training accurate points still harmed. No universal impossibility claim.',next_experiments_executed=False,files={str(p.relative_to(ROOT)):sha(p) for p in files}));print('COMPLETION',len(files),'bound files',flush=True)
if __name__=='__main__':main()
