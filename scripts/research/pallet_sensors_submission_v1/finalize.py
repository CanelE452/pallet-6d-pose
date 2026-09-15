"""Separate execution integrity, scientific evidence, and human dependencies."""
import numpy as np
from env import *

def run():
    start=now();verify();checks={};errors={}
    for label in ('BIND','IMPLEMENTATION','TRAIN','DEV','EVALUATE','MANUSCRIPT'):
        try:checks[label]=complete(label+'_COMPLETE')
        except (AssertionError,OSError) as e:checks[label]=False;errors[label]=str(e)
    pdf=read(DOC/'PDF_CHECKS.json') if (DOC/'PDF_CHECKS.json').exists() else {}
    visual=read(DOC/'PDF_VISUAL_REVIEW.json') if (DOC/'PDF_VISUAL_REVIEW.json').exists() else {}
    checks['PDF_VISUAL']=bool(visual.get('complete')) and all(sha(ROOT/b['path'])==b['sha256'] for b in visual.get('files',[])) and len(visual.get('reviewed_pages',[]))>0
    checks['PDF_VISUAL']=checks['PDF_VISUAL'] and all(sha(ROOT/r['render']['path'])==r['render']['sha256'] for r in visual.get('reviewed_pages',[]))
    training=read(DOC/'TRAINING_AUDIT.json') if checks['TRAIN'] else None
    if training:
        assert training['updates']==18000 and training['R0_P_D_L_retraining']==0
        for r in training['runs']:
            metrics=read(RAW/f"runs/PRIOR{r['seed']}/STEP_METRICS.json")
            assert len(metrics)==6000 and [x['step'] for x in metrics]==list(range(1,6001))
            assert r['exposures']==96000 and r['last_only'] and r['real_training']==0
            assert [p['step'] for p in r['probes']]==list(range(0,6001,500))
            assert all(np.isfinite(x['actual_update_norm']) and x['actual_update_norm']>0 for x in metrics)
            assert all(x['lr']==.0005*(.1**sum(x['step']>=i for i in (3858,5143))) for x in metrics)
    if checks['EVALUATE']:
        rt=read(DOC/'RUNTIME_PANEL.json');assert len(rt['records'])==13*26*5
        assert rt['accuracy_parity'] and rt['canonical_MAIN_pose_parity']
        for s in (1,2,3):
            assert complete(f'PRIOR{s}_INFERENCE') and complete(f'PRIOR{s}_SCORED') and complete(f'PRIOR{s}_raw_SCORED')
    c=read(DOC/'CONFIRMATION_COMPLETE.json') if (DOC/'CONFIRMATION_COMPLETE.json').exists() else {}
    conf='COMPLETE' if c.get('complete') and c.get('status')=='FULL_REFERENCE_COMPLETE' else c.get('status','AWAITING_INDEPENDENT_DATA')
    evidence={k:read(p) for k,p in {'P_minus_R0':OLD_DOC/'P_VS_R0_PAIRED.json','P_minus_D':OLD_DOC/'P_VS_D_PAIRED.json','P_minus_PRIOR':DOC/'P_VS_PRIOR_PAIRED.json'}.items() if p.exists()}
    interpretation={}
    for name,r in evidence.items():
        if 'session' in r:
            q=r['session'];label='P_LOWER_DEV_MEDIAN' if q['high']<0 else 'P_HIGHER_DEV_MEDIAN' if q['low']>0 else 'INTERVAL_INCLUDES_ZERO_DIFFERENCE_UNRESOLVED'
            interpretation[name]=dict(status=label,scope='Reused development data; secondary contrasts unadjusted; not a converged-method, independent-generalization, equivalence or noninferiority claim')
        else:interpretation[name]=dict(status=r.get('status','SUPPORT_REVIEW_REQUIRED'))
    status=dict(EXECUTION=('COMPLETE' if conf=='COMPLETE' else 'COMPLETE_FOR_AVAILABLE_INPUTS') if all(checks.values()) else 'PARTIAL_TECHNICAL',PRIOR='MATCHED_BUDGET_COMPLETE' if checks['TRAIN'] and checks['EVALUATE'] else 'TRAINING_INCOMPLETE' if checks['IMPLEMENTATION'] else 'IMPLEMENTATION_BLOCKED',CONFIRMATION=conf,EVIDENCE=evidence,MANUSCRIPT='FULL_DRAFT_WITH_DECLARED_GAPS',PUBLICATION='NOT_SUBMITTED',git_publication='Verify final main SHA with git rev-parse HEAD / origin/main / ls-remote; receipt is outside commit to avoid self-reference',checks=checks,errors=errors,independent_physical_6D='NOT_MEASURED',Jetson='NOT_MEASURED',human_handoff='HUMAN_INPUTS_REQUIRED.txt',author_review='_docs/paper/sensors_submission_v1/AUTHOR_REVIEW.md')
    status['EVIDENCE_INTERPRETATION']=interpretation
    status['PRIOR']='MATCHED_BUDGET_COMPLETE' if checks['TRAIN'] and checks['DEV'] else status['PRIOR']
    status['COST']='COMPLETE_SAME_SESSION_EXPANDED_PANEL' if checks['EVALUATE'] else 'DEFERRED_FOREIGN_GPU_COMPUTE' if (DOC/'EVALUATE_BLOCKED.json').exists() and 'foreign_compute' in read(DOC/'EVALUATE_BLOCKED.json').get('error','') else 'NOT_COMPLETE'
    status['DEV']='COMPLETE' if checks['DEV'] else 'INCOMPLETE'
    if not checks['EVALUATE']:
        status['runtime_handoff']=dict(owner='GPU task owner / next CLI session',action='Allow other compute to finish; do not kill it. Re-run evaluate, confirmation, manuscript, inspect all current PDF pages, audit, publish.',command='python scripts/research/pallet_sensors_submission_v1/run.py evaluate',note='Completed DEV receipts skip training, image inference and bootstrap calculations. No background continuation is left running.')
    write(DOC/'FINAL_STATUS.json',status)
    past=[]
    for p in sorted(DOC.glob('*_BLOCKED.json')):
        b=read(p);done=DOC/(p.name.replace('_BLOCKED','_COMPLETE'));r=read(done) if done.exists() else {}
        past.append(dict(failure=bound(p),resolved_by_later_completion=bool(r.get('complete')) and r.get('end','')>b.get('time',''),completion=str(done.relative_to(ROOT)) if done.exists() else None))
    write(DOC/'FINAL_AUDIT.json',dict(complete=all(checks.values()),checks=checks,errors=errors,historical_failures_retained=past,old_files_verified=len(read(DOC/'SOURCE_BINDING.json')['files']),training=training,changed_prior_inputs_only=True,scientific_success_not_an_execution_gate=True,PDF_compile=pdf.get('compile',False),PDF_visual=checks['PDF_VISUAL']))
    lines=['# Sensors 원고 마감 결과','',f"- 시작 main: `{read(DOC/'SOURCE_BINDING.json')['start_main']}`.",'- 게시 SHA는 작업 완료 메시지와 ignored raw/PUBLISH_COMPLETE.json에서 확인한다. 자기 SHA를 커밋 내부에 억지로 넣지 않는다.',f"- 실행: {status['EXECUTION']}; 선행 비교: {status['PRIOR']}; 독립 확인: {conf}.",'- 기존 R0/P/D/L의 학습·가중치·선택·평가 결과는 그대로 보존했다. 기존 모델 추가 학습은 0회다.', '- 선행은 실제 공식 TF1 CPU 네트워크를 기준으로 연산/가중치를 검증한 PyTorch PoseFix-derived pallet9다. RGB 및 pallet9 target/support 변경, BN의 실제 epsilon 보정을 공개했다. torchvision 대체나 D 재명명 결과가 아니다.','']
    if training:
        lines+=['## 새 학습','',f"실제 {training['updates']} updates, disposable smoke {training['smoke_updates']} update(s).",'']
        for r in training['runs']:lines.append(f"- seed{r['seed']}: {r['updates']}회 / 노출 {r['exposures']} / 재개 {r['resumes']} / {r['elapsed_seconds']:.1f}초.")
        lines+=['','실제 학습 코드 SHA는 TRAIN_CODE_LOCK.json에 기록했다. 6,000회 최종 checkpoint만 사용했으며 원 논문의 140 epochs 수렴 성능을 재현했다고 주장하지 않는다.']
        lines+=['']+[f"- `{n}`: `{h}`" for n,h in read(DOC/'TRAIN_CODE_LOCK.json')['code'].items()]
    lines+=['','## 결과와 한계','','DEV319/13 sessions 및 negative2689이며 기존 조건부 매칭311장/2,756점, 전체 GT2,818점 분모를 보존했다. 새 prior의 실제 분모·비유한값·pose coverage는 UNIFIED_DEV_RESULTS.json에 따로 기록했다.']
    if not checks['EVALUATE']:
        lines+=['','새 전체 panel 속도 측정은 외부 finetune_green_zip.py의 GPU 점유 때문에 보류했다. GPU는 정상 동작 중이며 no-CUDA/드라이버 오류가 아니다. 지시문의 반복 확인 상한에 따라 다른 작업을 종료하거나 무기한 대기하지 않았다. 기존 속도는 역사적 기록으로만 보존하며 PRIOR와의 동일 세션 속도 비교라고 하지 않는다.', '', 'GPU 작업 담당자가 다른 학습 종료를 확인한 뒤 `python scripts/research/pallet_sensors_submission_v1/run.py evaluate`로 비용 단계부터 재개한다. 이어 confirmation → manuscript → 모든 새 PDF 페이지 검사/visual receipt → audit → publish를 실행한다. 완료된 18,000 updates와 DEV 추론/통계를 재수행하지 않는다. 백그라운드 대기 작업은 남기지 않았다.']
    lines+=['']
    for name,r in evidence.items():
        if 'session' in r:
            q=r['session'];lines.append(f"- {name}: {q['delta']:.6f}px, paired-session95% interval [{q['low']:.6f}, {q['high']:.6f}]. 음수는 P의 낮은 오차를 뜻한다.")
        else:lines.append(f'- {name}: 공통 지원집합에 관한 상태를 원 JSON에서 확인한다. 임의 교집합으로 주 비교를 바꾸지 않았다.')
    lines+=['','P가 모든 지표에서 최상은 아니다. 기존 D/L의 관측 P90는 P보다 낮으며 D의 지연도 더 낮았다. eval_noapril 세션에서는 P−R0의 중앙값이 악화됐다. 개발 자료 재사용 및 보정하지 않은 복수 secondary interval이므로 독립 확인이나 수렴된 방법 우월성으로 확대하지 않는다.','', 'R0 초기 checkpoint에는 COCO-pose 사전학습 이력이 있다. 합성 전용 주장은 추가 P 학습에 한정한다. cm→mm로 표시한 pose 오차 통계 변화는 독립 물리 6D 실측이나 삽입 성공률이 아니다.','', '## 산출물과 다음 의존성','', '- 영문 전체 원고: `_docs/paper/sensors_submission_v1/manuscript.pdf` 및 `.tex`.','- 보조자료: 같은 디렉터리의 `supplementary.pdf` 및 `.tex`.','- JSON 기반 숫자/도표와 원천 hash: `NUMBER_SOURCES.json`, `NUMBERS_MANIFEST.json`.','- 현재 미완료 의존성은 외부 사람 검증 자료다. 데이터 담당자는 HUMAN_INPUTS_REQUIRED.txt의 촬영/블라인드 이중 어노테이션/치수·카메라 검증 자료를 지정 incoming 경로에 제공하고, 공저자는 AUTHOR_REVIEW.md의 저자·연구비·권리·동의 항목을 검토해야 한다.','- 자료 도착 후 `run.py confirmation --panel <검증된 panel.json>` → `run.py manuscript` → PDF 각 페이지 재검토 → `run.py audit` → `run.py publish`.','- 독립 새 자료가 없는 상태를 모든 실험 완료나 투고 준비 완료라고 부르지 않는다. 학술지 제출은 하지 않았다.','']
    if checks['DEV']:
        result=read(DOC/'UNIFIED_DEV_RESULTS.json')['methods'];runtime=read(DOC/'RUNTIME_PANEL.json')['summary'] if checks['EVALUATE'] else {}
        lines+=['## 동일 DEV 및 같은 세션 속도 요약','', '| 모델 | median px | P90 px | 전체 PCK10 % | translation cm | pose coverage | 대표 full-path ms |','|---|---:|---:|---:|---:|---:|---:|']
        for fam in ('R0','P','D','L','PRIOR'):
            names=['R0'] if fam=='R0' else [f'{fam}{s}' for s in (1,2,3)];rep=names[0]
            m=lambda key:float(np.mean([result[n][key] for n in names]));p=lambda key:float(np.mean([result[n]['pose'][key] for n in names]))
            pck=float(np.mean([result[n]['ALL_GT_PCK']['10'] for n in names]))*100
            latency=f"{runtime[rep]['end_to_end_ms']['median']:.3f}" if rep in runtime else 'NM (새 panel 보류)'
            lines.append(f"| {fam} {'single' if fam=='R0' else 'seed mean'} | {m('median_px'):.6f} | {m('p90_px'):.6f} | {pck:.4f} | {p('translation_median_cm'):.6f} | {p('coverage'):.4f} | {latency} |")
        lines+=['','정확도는 seed별 통계의 평균이고 속도 대표는 seed1이다. 같은 열을 ensemble 또는 seed1 정확도로 해석하지 않는다. Raw prior는 UNIFIED_DEV_RESULTS와 보조자료의 별도 행이다. Rotation/yaw/IoU3D/ADDsym AUC, 전체 frame/point 분모와 모든 latency 반복도 해당 JSON/PDF에 보존했다.','']
    if interpretation.get('P_minus_PRIOR',{}).get('status')=='P_HIGHER_DEV_MEDIAN':
        lines+=['','이번 고정 예산의 주 중앙오차 비교에서는 PRIOR가 P보다 낮은 오차를 보였다. P를 가장 정확한 비교군으로 결론내리지 않는다. 알려진 파라미터 수와 미측정인 새 동일 세션 속도를 구분해야 한다.','']
    (DOC/'FINAL_REPORT_KO.md').write_text('\n'.join(lines))
    receipt('AUDIT_COMPLETE',[DOC/'FINAL_AUDIT.json'],[DOC/'FINAL_STATUS.json',DOC/'FINAL_REPORT_KO.md'],start,all_available_work_complete=all(checks.values()))
    print(json.dumps({k:status[k] for k in ('EXECUTION','PRIOR','DEV','COST','CONFIRMATION','MANUSCRIPT','PUBLICATION')},ensure_ascii=False),flush=True)

if __name__=='__main__':run()
