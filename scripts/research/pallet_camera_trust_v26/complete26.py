"""Seal diagnosis completion without claiming a trained improvement."""
from common26 import *

def main():
    for n in ('AUDIT.json','TARGET_AUDIT.json','DELIVERY_CHECK.json','DOCS_CHECK.json'):assert read(RUN/n)['status']=='PASS'
    for n in ('REPORT_COMPLETE.json','RENDER_COMPLETE.json','INFERENCE_COMPLETE.json','TRUST_SUMMARY.json','CAMERA_SUMMARY.json'):assert read(RUN/n)['complete']
    files=list(KIT.glob('*.py'))+list(KIT.glob('*.md'))+list((RUN/'figures').glob('*.png'))+list((RUN/'predictions').glob('*.npz'))+list((RUN/'parents').glob('*.pt'))+list((RUN/'cache').glob('*'))+[NOTE]
    files += [p for p in RUN.iterdir() if p.is_file() and p.suffix in ('.json','.jsonl','.md','.html','.png') and p.name not in ('COMPLETION.json','DISCORD_NOTIFICATION.json') and '.pending.' not in p.name]
    files += list((RUN/'renders').glob('*'))
    write(RUN/'COMPLETION.json',dict(complete=True,registered_diagnosis_complete=True,stable_fusion_resolved=False,new_models_trained=0,new_training_updates=0,rendered_images=180,requested_observations=360,detected_observations=172,seeds=[1,2,3],GT_used_for_inference=False,GT_used_for_oracle_diagnosis=True,operating_model_changed=False,html_open_confirmed=read(RUN/'BROWSER_OPEN.json')['status']=='OPEN_CONFIRMED',not_executed=['visible physical edge/endpoint GT supervision','new direction-specific utility training','new independent real holdout evaluation'],interpretation='Correct compatible lines may leave large tangent errors; existing association is not calibrated correction utility. Current predicted candidates retain some privileged oracle headroom, not an achieved improvement. Actual camera probe has limited renderer-domain detection coverage.',files={str(p.relative_to(ROOT)):sha(p) for p in files}));record('V26 등록 진단 완료. 실제 카메라180렌더/360관측·고정모델/신뢰도 진단·GT/지표 감사·MD/HTML을 완료하고 해시를 봉인. 신규 학습 및 안정적 융합 해결과 구분.');print('COMPLETE26',len(set(files)),flush=True)

if __name__=='__main__':main()
