"""고정4팔 결과의 한국어 원인분해: fit 선택/실행 없이 근거와 공백만 보고한다."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import common as C
from . import eval_student as E


def delta(before, after):
    result={}
    for key in ('translation_cm','rotation_deg'):
        result[key]={stat:after['conditional'][key][stat]-before['conditional'][key][stat]
            if after['conditional'][key][stat] is not None and before['conditional'][key][stat] is not None else None
            for stat in ('median','P90')}
    result.update(valid_before=before['valid_pose'],valid_after=after['valid_pose'],frames=before['frames'],
        PCK10_pp=100*(after['twoD']['PCK']['10']-before['twoD']['PCK']['10']),
        ADDsym_AUC=after['ADDsym_AUC']-before['ADDsym_AUC'])
    return result


def selector_evidence(result, oracle):
    """조건부 compatibility 검사 신호; 새selector fit이나성공 판정이 아님."""
    if oracle is None:
        return dict(status='PENDING_CANDIDATE_ORACLE',fit_selected=False)
    group=result['groups'][E.PRIMARY]
    before,after=group['CLEAN_REF_CLEAR'],group['CLEAN_REF_OCC']
    full=result['classification']['corrected_occlusion_value']
    old,new=[oracle['groups'][E.PRIMARY][arm] for arm in ('CLEAN_REF_CLEAR','CLEAN_REF_OCC')]
    less=lambda left,right:left is not None and right is not None and left<right
    signals=dict(
        localization_PCK10_increases=after['twoD']['PCK']['10']>before['twoD']['PCK']['10'],
        T_optimal_T_median_decreases=less(new['translation_optimal']['conditional']['translation_cm']['median'],old['translation_optimal']['conditional']['translation_cm']['median']),
        R_optimal_R_median_decreases=less(new['rotation_optimal']['conditional']['rotation_deg']['median'],old['rotation_optimal']['conditional']['rotation_deg']['median']),
        final_joint_gain=full['eligible_joint'],
        current_single_candidate_joint_headroom=new['frames_with_same_candidate_joint_gain'],
        baseline_single_candidate_joint_headroom=oracle['groups'][E.PRIMARY]['CLEAN_REF_OCC_joint_candidate_vs_baseline']['CLEAN_REF_CLEAR'])
    possible=(signals['localization_PCK10_increases'] and signals['T_optimal_T_median_decreases']
        and signals['R_optimal_R_median_decreases'] and not signals['final_joint_gain']
        and signals['current_single_candidate_joint_headroom']>0)
    return dict(status='ZERO_FIT_SELECTOR_COMPATIBILITY_SIGNAL' if possible else 'NO_COMPLETE_SELECTOR_TRIGGER',
        observations=signals,fit_selected=False,
        interpretation='두개별oracle최솟값은단일실현가능pose아님. 실제한후보의joint headroom을별도보고. 관찰신호만으로selector원인확정불가.',
        automatic_fit_or_change_authorized_here=False)


def analyze(result, oracle):
    comparisons={g:{after+'-minus-'+before:delta(result['groups'][g][before],result['groups'][g][after])
        for before,after in E.PAIRS} for g in ('NATURAL99','CLEAN29','FULL128','MOD21','SEV78')}
    pose=result['classification']
    return dict(status='COMPLETE_WITH_ORACLE' if oracle else 'EVALUATION_COMPLETE_ORACLE_PENDING',
        primary_population='자연Moderate21+Severe78=99, 기존고정재사용DEV',
        comparisons=comparisons,primary_classification=pose,
        clean_preservation=comparisons['CLEAN29'],selector_diagnostic=selector_evidence(result,oracle),
        candidate_oracle=oracle['groups'] if oracle else None,
        source_rules=['6개사전비교를모두공개; 좋은팔/하위집합을새주설정으로선택하지않음',
            'median차이와개별paired차이median은다름; 원집계와paired9분류는EVAL_RESULTS에있음',
            '2D/AUC향상만으로T/R동시개선판정하지않음',
            '평가99와TRAIN78의타깃추종을혼동하지않음',
            '새독립검증이아니며원인확정/새fit/selector변경을자동실행하지않음'],
        new_fits=0,optimizer_updates=0,selector_fits=0)


def number(value):
    return 'NA' if value is None else f'{value:.4f}'


def markdown(report, result, seed, figure, json_name):
    lines=['# Clean→자연 가림 자세 전이: 고정2×2 결과 원인분해','',
        f'상태: `{report["status"]}`. 이 분석기에서 새 학습/optimizer/selector fit은0회다.','',
        f'![동일99장과Clean29의T/R 관계]({figure})','',
        '## 자연 가림99: 사전 지정6개 비교','',
        '음수 ΔT/ΔR은 오차 감소다. P90 손익과valid분모를 함께 읽는다. PCK/AUC만으로성공이라고판정하지않는다.','',
        '| 비교 (후−전) | ΔT median/P90 cm | ΔR median/P90 ° | valid 전→후 | ΔPCK10 %p | ΔAUC |',
        '|---|---:|---:|---:|---:|---:|']
    for key,row in report['comparisons']['NATURAL99'].items():
        lines.append(f'| {key} | {number(row["translation_cm"]["median"])}/{number(row["translation_cm"]["P90"])} | {number(row["rotation_deg"]["median"])}/{number(row["rotation_deg"]["P90"])} | {row["valid_before"]}→{row["valid_after"]}/99 | {number(row["PCK10_pp"])} | {number(row["ADDsym_AUC"])} |')
    lines+=['','## 난도·전체 성능과Clean손상','',
        '| 집합 | 모델 | T median/P90 cm | full R median/P90 ° | valid/분모 | PCK10 % | AUC |',
        '|---|---|---:|---:|---:|---:|---:|']
    for group in ('CLEAN29','MOD21','SEV78','NATURAL99','FULL128'):
        for arm in (*E.BASELINES,*E.ARMS):
            row=result['groups'][group][arm]
            values=['/'.join(number(row['conditional'][metric][q]) for q in ('median','P90')) for metric in ('translation_cm','rotation_deg')]
            lines.append(f'| {group} | {arm} | {values[0]} | {values[1]} | {row["valid_pose"]}/{row["frames"]} | {100*row["twoD"]["PCK"]["10"]:.3f} | {row["ADDsym_AUC"]:.5f} |')
    lines+=['','## 3층 해석','',
        '1. Localization: 같은128/99의legacy PCK·tail·fixed-ID 보조값을보고한다. 이는독립verified physicalaccuracy가아니다. TRAIN residual은별도TRAIN_TARGET_FOLLOWING에서측정하며정답오차로부르지않는다.',
        '2. Candidate quality: 같은frozen좌표가생성하는기존D9 두W/D후보만본다. T-optimal과R-optimal은각각완전한pose지만그최솟값끼리합친가상pose는배포가능하지않다.',
        '3. Selector: 동일한한후보가현재선택보다T/R모두좋은프레임수를확인한다. 후보headroom이있어도일반화가능한selector를학습할수있다는보장은없다.','',
        f'조건부신호: `{report["selector_diagnostic"]["status"]}`. 이문서는후속fit을선택하거나실행하지않았다.','']
    if report['candidate_oracle']:
        lines+=['| 자연99 후보집합 | 실제 T/R med | T-optimal의 T/R med | R-optimal의 T/R med | 같은한후보 joint개선수 |',
            '|---|---:|---:|---:|---:|']
        for arm in (*E.BASELINES,*E.ARMS):
            row=report['candidate_oracle']['NATURAL99'][arm]
            vals=['/'.join(number(row[key]['conditional'][metric]['median']) for metric in ('translation_cm','rotation_deg'))
                for key in ('current','translation_optimal','rotation_optimal')]
            lines.append(f'| {arm} | {vals[0]} | {vals[1]} | {vals[2]} | {row["frames_with_same_candidate_joint_gain"]}/99 |')
        lines+=['','oracle는GT기반사후진단전용이며최종D9성능을대체하지않는다.']
    lines+=['','## 근거의 범위','',
        '자연99오차를모아median/P90을계산했으며난도별median을평균하지않았다. Clean29손상을삭제하지않는다. 참조6D는기존annotation과known dimensions의기하재구성이며독립물리측정이아니다. 반복DEV결과로독립확인을완료했다고쓰지않는다.', '',
        f'[전체평가·recording·paired·LORO](EVAL_RESULTS_S{seed}.json) · [분석 JSON]({json_name})', '']
    train=report.get('train_target_following')
    if train:
        lines+=['## TRAIN 타깃 전달','',
            '실제clean78의고정native입력에서측정한의사타깃추종이다. 물리정답정확도나실제가림/affine입력TRAIN loss로해석하지않는다.','',
            '| 모델 | raw타깃 평균px | corrected타깃 평균px | corrected타깃 occurrence가중평균px |',
            '|---|---:|---:|---:|']
        for arm,values in train['groups']['ALL78'].items():
            lines.append(f'| {arm} | {number(values["raw_target"]["unique_corner_pooled"]["mean_px"])} | {number(values["ref_target"]["unique_corner_pooled"]["mean_px"])} | {number(values["ref_target"]["occurrence_weighted_corner_pooled"]["mean_px"])} |')
        lines+=['',f'[TRAIN 좌표계·support·recording별분석](TRAIN_TARGET_FOLLOWING_S{seed}.md)','']
    else:
        lines+=['TRAIN native 타깃 추종은 아직미실행이다. 잔차나최적화부족을추정하지않는다.','']
    return '\n'.join(lines)


def figure(result, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'R0':'R0','OLD_REF':'Old REF','CLEAN_RAW_CLEAR':'RAW clear','CLEAN_REF_CLEAR':'REF clear','CLEAN_RAW_OCC':'RAW occ','CLEAN_REF_OCC':'REF occ'}
    colors={'R0':'#777777','OLD_REF':'#b45a55','CLEAN_RAW_CLEAR':'#7ba5c8','CLEAN_REF_CLEAR':'#266a9a','CLEAN_RAW_OCC':'#d1ad58','CLEAN_REF_OCC':'#298a54'}
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for axis,group,title in zip(axes,('NATURAL99','CLEAN29'),('Natural occlusion99','Clean29 preservation')):
        for index,arm in enumerate((*E.BASELINES,*E.ARMS)):
            row=result['groups'][group][arm]
            x,y=[row['conditional'][k]['median'] for k in ('translation_cm','rotation_deg')]
            if x is None or y is None:
                axis.text(.02,.98-.055*index,labels[arm]+': no valid pose',transform=axis.transAxes,va='top',fontsize=8)
                continue
            axis.scatter(x,y,color=colors[arm],marker='x' if arm in E.BASELINES else 'o',s=70)
            axis.annotate(labels[arm],(x,y),xytext=(5,5 if index%2 else -12),textcoords='offset points',fontsize=8)
        axis.set(xlabel='T median (cm), lower is better',ylabel='Full R median (deg), lower is better',title=title)
        axis.margins(.22)
        axis.grid(alpha=.2)
    fig.suptitle('Frozen clean-only 2 x 2: unchanged D9, reused DEV')
    fig.tight_layout()
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=150)
    plt.close(fig)


def main(seed=42):
    p=E.paths(seed)
    if not p['results'].exists():
        print(json.dumps(dict(status='PENDING_EVALUATION',fit_selected=False)),flush=True)
        return
    result=C.read(p['results'])
    for b in result['scoring_sources']+result['private_artifacts']:
        C.verify(b)
    oracle=C.read(p['oracle_result']) if p['oracle_result'].exists() else None
    if oracle:
        for b in oracle['sources']+oracle['private_artifacts']:
            C.verify(b)
    report=analyze(result,oracle)
    train_path=C.DOC/f'TRAIN_TARGET_FOLLOWING_S{seed}.json'
    train=C.read(train_path) if train_path.exists() else None
    if train:
        for b in train['inputs']:
            C.verify(b)
    report['train_target_following']=train
    suffix=('' if oracle else '_EVAL_ONLY')+('' if train else '_TRAIN_PENDING')
    destination=C.DOC/f'PRIMARY_ANALYSIS{suffix}_S{seed}.json'
    if destination.exists():
        for b in C.read(destination)['inputs']:
            C.verify(b)
        return
    f=C.DOC/f'figures/primary_pose_transfer_S{seed}.png'
    figure(result,f)
    report.update(inputs=[C.bind(p['results']),C.bind(Path(__file__))]+([C.bind(p['oracle_result'])] if oracle else [])+([C.bind(train_path)] if train else []),figure=C.bind(f))
    C.save(destination,report,True)
    C.save(destination.with_suffix('.md'),markdown(report,result,seed,str(f.relative_to(C.DOC)),destination.name),True)
    print('PRIMARY_ANALYSIS_WRITTEN',destination,report['selector_diagnostic']['status'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=42)
    main(parser.parse_args().seed)
