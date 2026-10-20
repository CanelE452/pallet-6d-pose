"""Evidence-bound paper tables only; no manuscript edits or new experiments."""
import argparse,csv,json,math,shutil
from pathlib import Path
from collections import Counter
import numpy as np
from analyze import read,rows,index,sha,write,csvwrite,summary,metric,SEEDS,PF,POPS,transition
from make_report import table

def rr(p):return list(csv.DictReader(p.open()))
def run(public,data,repo):
    out=public/'paper_tables';out.mkdir(exist_ok=False);sources={};checks=[]
    def bind(alias,p):sources[alias]=dict(basename=p.name,sha256=sha(p),bytes=p.stat().st_size);return p
    refdir=data/'pallet-statewise-correspondence-diagnosis-20261016-v1/score_v2';refp=bind('fixed_references',refdir/'FIXED_REFERENCES.json');refs=read(refp)['references']
    assert sha(refp)==read(refdir/'SCORE_SEAL.json')['artifacts'][refp.name]
    bases=rows(bind('original_baselines',refdir/'BASELINE_PREDICTIONS.jsonl.gz'));assert len(index(bases))==238
    assert sha(refdir/'BASELINE_PREDICTIONS.jsonl.gz')==read(refdir/'SCORE_SEAL.json')['artifacts']['BASELINE_PREDICTIONS.jsonl.gz']
    # Trace exact native single-stage names to the source producer, not N0 relabeling.
    sourcecode=bind('baseline_stage_mapping',repo/'scripts/research/pallet_statewise_correspondence_diagnosis_v1/analysis.py')
    code=sourcecode.read_text();assert "N3_ONLY=deepcopy(s['predictions']['N3_F'])" in code and "SUBPIX_ONLY=deepcopy(s['predictions']['SUBPIX_F'])" in code
    extra=[]
    for r in bases:
        x={k:r[k] for k in ('id','partition','session','cluster_id')};x['metrics']={}
        for a in ['R0','N3_ONLY','SUBPIX_ONLY','POSEFIX_RAW']:
            x['metrics'][a]=metric(r['predictions'][a]['actual_pose'],refs[r['partition']][r['id']])
        extra.append(x)
    saved_original=list(csv.DictReader(bind('original_row_metrics',refdir/'ROW_METRICS.csv').open()))
    # The authoritative scalar CSV is also compared below using its exact field schema.
    base_extra=[]
    for pop,parts in POPS.items():
        rs=[r for r in extra if r['partition'] in parts]
        for a in ['R0','N3_ONLY','SUBPIX_ONLY','POSEFIX_RAW']:base_extra.append(dict(population=pop,**summary(rs,a),status='VERIFIED_SAVED_OUTPUT',definition={'R0':'raw YOLO + original F','N3_ONLY':'N3_F, no SubPix','SUBPIX_ONLY':'SUBPIX_F, no N3','POSEFIX_RAW':'PoseFix first uncapped PASS1 + F'}[a]))
    csvwrite(out/'ADDITIONAL_BASELINES.csv',base_extra)
    es={r['id']:r for r in extra}
    # Baseline source CSV uses columns inspected at runtime; fail rather than guess.
    fields=set(saved_original[0]);assert {'id','arm','translation_cm','rotation_deg','success_le5'}.issubset(fields),fields
    for r in saved_original:
        if r['arm'] not in ['R0','N3_ONLY','SUBPIX_ONLY','POSEFIX_RAW']:continue
        m=es[r['id']]['metrics'][r['arm']]
        assert abs(float(r['translation_cm'])-m['translation_cm'])<1e-8 and abs(float(r['rotation_deg'])-m['rotation_deg'])<1e-8
        assert (r['success_le5']=='True')==m['success'];checks.append('original_baseline_row')
    allsum=rr(public/'tables/all_saved_methods_summary.csv');sm={(r['population'],r['arm']):r for r in allsum}
    tab2=[]
    for arm in ['R0','SUBPIX_ONLY','N3_ONLY','POSEFIX_RAW',*PF,'N0','E0',*[f'D1_H1_MLP_s{s}' for s in SEEDS]]:
        for pop in POPS:
            r=next((x for x in base_extra if x['arm']==arm and x['population']==pop),None)
            if r is None:r=dict(sm[pop,arm],status='VERIFIED_SAVED_OUTPUT',definition={'N0':'N3 + SubPix + original F','E0':'original D1 geometry before H1'}.get(arm,arm))
            tab2.append(r)
    csvwrite(out/'TABLE_02_PRIMARY_AND_PARTITIONS.csv',tab2)
    protocol=[dict(item='PRIMARY192',n='192',role='TEST49 + REMAINDER143; repeatedly used development real frames',contract='same ID/hash/K/known dimensions/physical axes/C2; full binary denominator'),dict(item='CAL46',n='46',role='existing calibration; separate reporting',contract='not independent validation or test'),dict(item='D1 synthetic train',n='768',role='original tiny H1 training',contract='same frozen original3seed normalizer and checkpoints'),dict(item='D1 synthetic validation',n='235',role='already reused development selection',contract='not fresh confirmation'),dict(item='PF',n='seed1/step6000',role='synthetic-only PRIOR1 first uncapped PASS1',contract='two fixed cap anchors; neither selected by PRIMARY outcome'),dict(item='current three-backbone transfer',n='3backbones/3heads',role='same original YOLO-derived CAL/state/mask/prior/H1 transferred',contract='no own-backbone H1/CAL fit; upstream training budgets differ'),dict(item='independent real physical GT',n='0',role='BLOCKED',contract='existing references are geometric reconstructions'),dict(item='independent real test',n='0',role='NOT_RUN',contract='all current real frames already used for development')]
    csvwrite(out/'TABLE_01_DATA_AND_PROTOCOL.csv',protocol)
    tri=data/'pallet-d1-three-backbones-20261020-v1';t=tri/'public_bundle_v1/results/tables'
    completion=read(bind('three_backbone_completion',tri/'TASK_COMPLETION_V1.json'));assert completion['complete'] is True and completion['status']=='PASS'
    rmetrics=rr(bind('three_backbone_rows',t/'ROW_METRICS.csv'));alltri=rr(bind('three_backbone_metrics',t/'ALL_POPULATION_METRICS.csv'));pairedtri=rr(bind('three_backbone_paired',t/'PAIRED_COUNTS.csv'))
    # Independent scalar reaggregation; no new 3-backbone inference or optimization.
    for r in alltri:
        ss=[x for x in rmetrics if x['backbone']==r['backbone'] and x['arm']==r['arm'] and x['partition'] in POPS[r['population']]]
        assert len(ss)==int(r['denominator']) and len({x['id'] for x in ss})==len(ss)
        assert sum(x['success']=='True' for x in ss)==int(r['successes'])
        assert sum(x['pose_available']=='False' for x in ss)==int(r['no_pose'])
        for short,key in [('T_cm','translation_cm'),('R_deg','rotation_deg')]:
            v=[float(x[key]) for x in ss if x[key] not in ('','None','null')];assert len(v)==int(r[short+'_n'])
            for name,z in [('mean',np.mean(v)),('median',np.median(v)),('P90',np.quantile(v,.9)),('P95',np.quantile(v,.95)),('max',max(v)),('sample_variance',np.var(v,ddof=1))]:
                assert abs(float(z)-float(r[short+'_'+name]))<1e-8;checks.append('three_backbone_metric')
    for r in pairedtri:
        parts=POPS[r['population']];bef={x['id']:x for x in rmetrics if x['backbone']==r['before_backbone'] and x['arm']==r['before_arm'] and x['partition'] in parts};aft={x['id']:x for x in rmetrics if x['backbone']==r['after_backbone'] and x['arm']==r['after_arm'] and x['partition'] in parts};assert set(bef)==set(aft)
        c=Counter(transition(bef[k]['success']=='True',aft[k]['success']=='True') for k in bef)
        for k,alias in [('RECOVER','RECOVERY'),('DAMAGE','DAMAGE'),('BOTH_SUCCESS','BOTH_SUCCESS'),('BOTH_FAILURE','BOTH_FAILURE')]:assert c[k]==int(r[alias]);checks.append('three_backbone_pair')
    for name in ['ALL_POPULATION_METRICS.csv','PAIRED_COUNTS.csv','CANDIDATE_DIAGNOSTICS.csv','CANDIDATE_ROWS.csv','SUPPLEMENTAL_COMMON_INITIAL_AVAILABLE_PRIMARY146.csv','COST_COMPONENTS.csv']:
        q=bind('three_backbone_'+name,t/name);shutil.copyfile(q,out/('TABLE_03_'+name if not name.startswith('COST') else 'TABLE_07_'+name))
    shutil.copyfile(t/'ROW_METRICS.csv',out/'TABLE_03_ROW_METRICS.csv')
    original=rows(bind('D1_original_ablation',data/'pallet-boundary-action-repair-20261018-v1/evaluation_v1/SCORED_ROWS.jsonl.gz'))
    p0=rows(bind('current_baseline',data/'pallet-bi-full-execution-20261019-v1/P0_BASELINE_REAGGREGATION/SCORED_ROWS.jsonl.gz'));pidx=index(p0)
    for r in original:
        for arm in ('N0','E0',*[f'D1_H1_MLP_s{s}' for s in SEEDS]):
            for key in ('translation_cm','rotation_deg'):assert r['metrics'][arm][key]==pidx[r['id']]['metrics'][arm][key]
    tab4=[]
    for a in ['N0','E0',*[f'D1_H0_MLP_s{s}' for s in SEEDS],*[f'D1_H1_MLP_s{s}' for s in SEEDS]]:
        for pop,parts in POPS.items():
            ss=[r for r in original if r['partition'] in parts];v=dict(population=pop,**summary(ss,a),contrast='N0_to_stage',same_saved_D1_bank='True' if a.startswith('D1_H') else 'N/A')
            c=Counter(transition(r['metrics']['N0']['success'],r['metrics'][a]['success']) for r in ss);v.update(recover_vs_N0=c['RECOVER'],damage_vs_N0=c['DAMAGE']);tab4.append(v)
    csvwrite(out/'TABLE_04_STAGE_AND_FIXED_BANK.csv',tab4)
    groups=[]
    for stem,kind,control in [('S3_COMMON','generic_support_common','D1_H1_MLP'),('S3_BI','generic_support_role_policy','S3_COMMON'),('F_COMMON','matched_fullnormal_common','D1_H1_MLP'),('F_BI','matched_fullnormal_role_policy','F_COMMON'),('H1_NO_B_FEATURE','original_bank_feature_ablation','D1_H1_MLP')]:
        for s in SEEDS:
            arm=stem+'_s'+str(s);before=control+'_s'+str(s)
            for pop in POPS:
                groups.append(dict(comparison_kind=kind,before=before,**sm[pop,arm],not_all_BI_information_removed=True,note='no-B blocks72/188 features on original bank; F/S3 change candidate/observation contracts; do not merge their causal interpretations'))
    csvwrite(out/'TABLE_05_COMMON_BI_VS_NO_B.csv',groups)
    for src,name in [('all_saved_methods_summary.csv','TABLE_06_ALL74_POPULATIONS.csv'),('paired_seed_summary.csv','TABLE_06_D1_PF_SEEDS.csv'),('recording_summary.csv','TABLE_06_RECORDING.csv'),('session_summary.csv','TABLE_06_SESSION.csv'),('success_failure_strata.csv','TABLE_06_TRANSITIONS.csv')]:shutil.copyfile(public/'tables'/src,out/name)
    cost=read(bind('three_backbone_typed_cost',tri/'public_bundle_v1/results/COST_ACCOUNTING.json'))
    write(out/'TABLE_07_TYPED_COST_ACCOUNTING.json',cost)
    pfseal=read(bind('PF_postprocessing_cost_seal',data/'pallet-accuracy-closeout-20261017-v1/posefix192_v1/PREDICTION_SEAL.json'))
    costsummary=[dict(component='PF saved neural outputs -> SubPix/F',wall_seconds=pfseal['timing']['wall_seconds'],CPU_seconds=pfseal['timing']['process_cpu_seconds'],scope='238frames ×3PF paths; no neural forward; postprocessing only',fair_end_to_end_comparison=False),dict(component='current mean-gap CPU saved-pose audit',wall_seconds=read(public/'tables/VALIDATION.json')['wall_seconds'],CPU_seconds='UNKNOWN_NOT_MEASURED',scope='hash checks and all saved scalar/bank analysis; no inference',fair_end_to_end_comparison=False),dict(component='equal training budget backbone comparison',wall_seconds='NOT_RUN',CPU_seconds='NOT_RUN',scope='upstream pretrained histories differ; fixed transfer comparison only',fair_end_to_end_comparison=False),dict(component='end-to-end detector to final pose/Jetson FPS',wall_seconds='NOT_RUN',CPU_seconds='NOT_RUN',scope='cache reuse costs cannot establish system FPS',fair_end_to_end_comparison=False)]
    csvwrite(out/'TABLE_07_FAIR_COST_SCOPE.csv',costsummary)
    missing=[dict(cell='new independent real test',status='NOT_RUN',reason='current192 repeatedly exposed development data'),dict(cell='independent physical pose GT / signed axes / exact real CAD',status='BLOCKED',reason='existing reconstructed references; no independently certified real registration'),dict(cell='physical owner/edge correspondence',status='BLOCKED',reason='independent real labels absent; current scope prohibits new manual labels/data'),dict(cell='own-backbone D1 H1/CAL retraining for DOPE/ResNet18',status='NOT_RUN',reason='completed comparison transfers the same original YOLO-derived frozen H1/CAL'),dict(cell='equal training/compute backbone causal comparison',status='NOT_RUN',reason='pretraining/history/budget differ'),dict(cell='whole-system detector->D1 latency and Jetson power/FPS',status='NOT_RUN',reason='typed cached postprocessing costs are not end-to-end'),dict(cell='new mean-winning limited intervention',status='NOT_RUN',reason='two hypotheses stopped: insufficient global-bias evidence and missing physical correspondence'),dict(cell='professor feedback / changed bibliography reconciliation',status='BLOCKED',reason='those separate GPT conversation sources are not supplied to this environment; no invented attribution')]
    csvwrite(out/'MISSING_CELLS.csv',missing)
    report='''# 논문 대화용 검증 표 패키지

원고를 편집하지 않고 같은 검증 출력에서 표를 조립했다. **평균 우위 확보 실패**를 포함한 부정 결과를 보존한다. 모든 수치는 PRIMARY192와 별도CAL46/TEST49/REMAINDER143으로 나뉘며 이전245장 N3 결과를 섞지 않는다.

| 표 | 파일 | 범위 |
|---|---|---|
|1|[자료·계약](TABLE_01_DATA_AND_PROTOCOL.csv)|교정/학습/평가·고정 transfer·독립GT 공백|
|2|[전체 추정 비교](TABLE_02_PRIMARY_AND_PARTITIONS.csv)|실제 R0/SubPix_ONLY/N3_ONLY/PoseFix_RAW/PF+SubPix/N0/E0/D1|
|3|[현재 세 backbone](TABLE_03_ALL_POPULATION_METRICS.csv)|완료된 frozen original D1 재사용;60행/미산출 유지|
|4|[단계·고정 은행](TABLE_04_STAGE_AND_FIXED_BANK.csv)|N0/E0/D1-H0/D1-H1, N0대비회복손상|
|5|[COMMON/BI와 no-B 구별](TABLE_05_COMMON_BI_VS_NO_B.csv)|동일정보 matched 정책과 특징차단의 해석 분리|
|6|[전체분포](TABLE_06_ALL74_POPULATIONS.csv)|평균/중앙/P90/P95/미산출/성공; 별도seed/recording/session/전이표|
|7|[공정 비용 범위](TABLE_07_FAIR_COST_SCOPE.csv)|실제 typed CPU/벽시간과 미계측FPS·불일치학습예산 구별|

SubPix_ONLY는 실제 저장된 SUBPIX_F, N3_ONLY는 N3_F다. N0=N3+SubPix+F를 단독으로 바꿔 부르지 않았다. 원48/95 또는245장 수치를 누락 셀에 넣지 않았다. R0/SubPix/N3/PoseFix_RAW는 원봉인 반환 자세에서 독립 재계산하고 원 scalar와 검산했다. 세 backbone표는 오늘 완료된 출력3570행/60집계/96전이를 재집계하여 재사용했으며 추가 forward/fit/PnP0이다.

DOPE는PRIMARY192중146자세 산출/46미산출이다. 성공은192분모, 연속평균은146조건부이므로 YOLO/ResNet18의192평균과 단순 우열로 해석하지 않는다. 보조공통146은 기존 초기가용성으로 고정된 별도집합이며 전체192를 대체하지 않는다. PRIMARY46미산출은 검출실패3/<6지원코너13/현재F입력계약거부30이다. 마지막30을 수치PnP실패라고 이름 바꾸지 않는다. frozen F가 rectangular 입력에서9개슬롯 모두유한을 요구한 기존 진단이며 이번에 고치지 않았다.

COMMON/BI표의 F는 새 fullnormal bank/동일cue 정책 비교이며 모든B/I 제거군이 아니다. no-B는 원 D1 bank에서 B유래 특징72/188을0으로 차단한 작은선택기 ablation이다. 목적과 은행이 달라 같은 B효과라고 합치지 않는다. 각행의 same-seed counterpart와 원참조를 따른다.

원자료·모델·R/t는 넣지 않았고 source SHA는 [SOURCE_MANIFEST](SOURCE_MANIFEST.json), 누락은 [MISSING_CELLS](MISSING_CELLS.csv)다. 별도GPT에서 교수님 의견/기존 자료를 합칠 때는 다음 사실을 유지해야 한다. 문제는 가림/관측대응과 선택의 실패이며 성공률 우위가 평균정확도 우위를 뜻하지 않는다. 목표는 같은정보·참조·분모에서 한최종자세의 평균위치/회전·성공/tail을 함께 개선하는 것이다. 현재 입증된 기여는 상태별 관측/후단/고정후보 선택의 명시적 구현·진단과 검증 프로토콜 범위이며, 물리owner 해결/평균우월성/새실사일반화는 입증하지 않았다. 교수님 의견·논문 reference 원문은 이 실행에 제공되지 않아 임의로 반영했다고 쓰지 않았다. 논문구성과 문헌 수정은 별도GPT 대화에서 이어간다.
'''
    primary=[r for r in tab2 if r['population']=='PRIMARY192']
    report+='\n## 기본 비교의 PRIMARY192 요약\n'
    report+=table(['경로','성공/192','T평균 cm','R평균 °','T/R 산출'],[[r['arm'],str(r['successes'])+'/192',f"{float(r['T_mean_cm']):.6f}",f"{float(r['R_mean_deg']):.6f}",str(r['T_n'])+'/'+str(r['R_n'])] for r in primary])
    report+='\n## 세 backbone의 현재D1 대표seed18\n'
    report+=table(['backbone','성공/192','미산출','T/R 가용 n','T평균 cm','R평균 °'],[[r['backbone'],r['successes']+'/192',r['no_pose'],r['T_cm_n']+'/'+r['R_deg_n'],f"{float(r['T_cm_mean']):.6f}",f"{float(r['R_deg_mean']):.6f}"] for r in alltri if r['population']=='PRIMARY192' and r['arm']=='D1_H1_MLP_s20261018'])
    (out/'README_KO.md').write_text(report)
    # Readable Markdown and simple tabular LaTeX accompany CSV without inventing styling.
    for filename in ['TABLE_01_DATA_AND_PROTOCOL.csv','TABLE_02_PRIMARY_AND_PARTITIONS.csv','TABLE_04_STAGE_AND_FIXED_BANK.csv','TABLE_05_COMMON_BI_VS_NO_B.csv','TABLE_07_FAIR_COST_SCOPE.csv']:
        rs=rr(out/filename);keys=list(rs[0]);(out/filename.replace('.csv','.md')).write_text(table(keys,[[r[k] for k in keys] for r in rs]))
    write(out/'SOURCE_MANIFEST.json',sources)
    write(out/'VALIDATION.json',dict(status='PASS',checks=dict(Counter(checks)),new_backbone_forwards=0,new_fits=0,new_pose_solves=0,original_baseline_rows=len(extra),completed_three_backbone_rows=len(rmetrics),three_backbone_summaries=len(alltri),three_backbone_paired=len(pairedtri),source_reference_sha256=sha(refp)))
    shutil.copyfile(Path(__file__),public/'paper_tables.py')
    with (public/'README.md').open('a') as f:f.write('\n추가 요청: [논문 대화용7종 검증 표/누락 셀/읽는 법](paper_tables/README_KO.md). 원고는 수정하지 않았다.\n')
    print(json.dumps(dict(status='PASS',checks=Counter(checks),tables=str(out)),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--source-repo',type=Path,required=True);a=p.parse_args();run(a.public,a.data_root,a.source_repo)
