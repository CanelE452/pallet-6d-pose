"""Seal completed training/evaluation separately from stable-improvement success."""
from context27 import *

def main():
    for n in ('PRECHECK.json','AUDIT.json','DELIVERY_CHECK.json','DOCS_CHECK.json'):assert read(RUN/n)['status']=='PASS'
    for n in ('TRAINING_COMPLETE.json','INFERENCE_COMPLETE.json','SUMMARY.json','TRAIN_DIAGNOSIS.json','UTILITY_DIAGNOSIS.json','REPORT_COMPLETE.json'):assert read(RUN/n)['complete']
    assert read(RUN/'TRAINING_COMPLETE.json')['updates']==24000;s=read(RUN/'SUMMARY.json');files=list(KIT.glob('*.py'))+list(KIT.glob('*.md'))+[NOTE]
    files +=[p for p in RUN.rglob('*') if p.is_file() and p.name not in ('COMPLETION.json','DISCORD_NOTIFICATION.json') and '.pending.' not in p.name]
    write(RUN/'COMPLETION.json',dict(complete=True,registered_training_validation_complete=True,trained_models=12,training_updates=24000,stable_arms=s['stable_arms'],accepted_line_arms=s['accepted_line_arms'],stable_fusion_resolved=bool(s['accepted_line_arms']),operating_model_changed=False,new_sealed_real_evaluation=False,html_open_confirmed=read(RUN/'BROWSER_OPEN.json')['status']=='OPEN_CONFIRMED',not_executed=['DHT backbone retraining','physical visible boundary/endpoint supervision','cross-fitted out-of-parent-training residual dataset','new independent real final holdout'],interpretation='Candidate oracle headroom remains, but learned mixture retains point-proposal dominance and mild-crop harm; small line effect and variable seed gains do not pass preregistered stability.',files={str(p.relative_to(ROOT)):sha(p) for p in files}));record('V27 전체 완료:새 구조12모델/24,000업데이트 학습·고정평가·원인분석·통합검증·시각화·문서 완료. 안정적 선 융합 성공 여부는 accepted_line_arms로 별도 표시.');print('COMPLETE27',len(files),s['accepted_line_arms'],flush=True)

if __name__=='__main__':main()
