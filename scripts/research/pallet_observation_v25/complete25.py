"""Separate completion of the registered experiment from solving stable fusion."""
from context25 import *

def main():
    for name in ('AUDIT.json','PROVENANCE_CHECK.json','INTEGRATION_CHECK.json','DELIVERY_CHECK.json','DOCS_CHECK.json'):
        assert read(RUN/name)['status']=='PASS'
    assert read(RUN/'REPORT_COMPLETE.json')['complete'];assert read(RUN/'TRAINING_COMPLETE.json')['updates']==42000
    s=read(RUN/'SUMMARY.json');files=list(KIT.glob('*.py'))+list(KIT.glob('*.md'))+list((RUN/'models').glob('*/final.pt'))+list((RUN/'figures').glob('*.png'))+[RUN/name for name in ('PROTOCOL.json','TRAIN_PROTOCOL.json','TRAINING_COMPLETE.json','EVAL_PROTOCOL.json','INFERENCE_COMPLETE.json','SUMMARY.json','DIAGNOSIS.json','INTERVENTION_DIAGNOSIS.json','AUDIT.json','PROVENANCE_CHECK.json','INTEGRATION_CHECK.json','REPORT_KO.md','REPORT_COMPLETE.json','index.html','DELIVERY_CHECK.json','BROWSER_OPEN.json','GALLERY.json','FIGURE_DATA.json','DOCS_CHECK.json')]+[NOTE]
    write(RUN/'COMPLETION.json',dict(complete=True,registered_experiment_complete=True,stages_executed=['condition diagnosis','actual opposite-side crop intervention','matched gate/vector point-line comparison'],stable_fusion_resolved=bool(s['stable_arms']),stable_arms=s['stable_arms'],models=21,updates=42000,new_image_observations=640,no_real_GT_training=True,operating_model_changed=False,html_open_confirmed=read(RUN/'BROWSER_OPEN.json')['status']=='OPEN_CONFIRMED',not_executed=['new camera-pose rendering','renderer-visible edge/endpoint ground truth supervision','new independent real capture holdout'],interpretation='Opposite-side crops did not shift synthetic inward-correction demand toward real. Direction-capable correction improves hard cases but still harms mild crop. Small extra line+pixel benefit on mild crop below preregistered every-seed1% threshold; no stable arm.',files={str(p.relative_to(ROOT)):sha(p) for p in files}));record('V25 등록 실험 완료. 실험 완료와 안정적 융합 해결을 분리한 COMPLETION.json 생성. 카메라 신규 렌더·실제 가시 경계GT·새 실사 최종평가는 미실행으로 명시.');print('COMPLETE25',len(files),'files',s['stable_arms'],flush=True)

if __name__=='__main__':main()
