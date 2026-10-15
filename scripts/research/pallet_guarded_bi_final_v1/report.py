"""Generate Korean method/results from closed saved results only.

No reference, image, observation, network, optimizer, PnP or inference module
is imported or opened. Reporting never creates or changes a method choice.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ARMS = ('R0','R1','RA','RB','N0','N1','NA','NB','E0','E1','EA','EB')
POPULATIONS = ('PRIMARY','TEST','REMAINDER','CAL')
NAMES = {
    'R0':'RAW + F', 'R1':'RAW + 무조건 B/I', 'RA':'RAW + 규칙 A', 'RB':'RAW + 규칙 B',
    'N0':'N3·SubPix + F', 'N1':'N3·SubPix + 무조건 B/I',
    'NA':'N3·SubPix + 규칙 A', 'NB':'N3·SubPix + 규칙 B',
    'E0':'기존 고정 e', 'E1':'e 초기화 + 새 B/I',
    'EA':'e 초기화 + 규칙 A', 'EB':'e 초기화 + 규칙 B',
}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        stream.write(value)


def write_json(path,value):
    write(path,json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def number(value,digits=3):
    return 'NA' if value is None else f'{value:.{digits}f}'


def mean_sd(summary):
    return f"{number(summary['mean'])}±{number(summary['sample_std'])} (n={summary['n']})"


def paragraphs(blocks):
    """Keep Markdown table rows contiguous while separating prose blocks."""
    out=''
    previous=''
    for block in blocks:
        if out:
            out+='\n' if block.startswith('|') and previous.endswith('|') else '\n\n'
        out+=block
        previous=block
    return out+'\n'


def table(population):
    lines=['| 경로 | 공동 성공/전체 | 위치 cm 평균±SD | 최대 cm | 회전 ° 평균±SD | 최대 ° | no-pose | fallback |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        s=population['arms'][arm]
        fallback=str(s['fallback_n']) if s['acceptance_known_n'] else 'NA'
        lines.append(f"| {arm} · {NAMES[arm]} | {s['successes_le5']}/{s['denominator']} | "
            f"{mean_sd(s['errors']['translation_cm'])} | {number(s['errors']['translation_cm']['max'])} | "
            f"{mean_sd(s['errors']['rotation_deg'])} | {number(s['errors']['rotation_deg']['max'])} | "
            f"{s['no_pose_n']} | {fallback} |")
    return '\n'.join(lines)


def cells(rows,new,before):
    groups={k:[] for k in ('failure_to_success','success_to_failure','both_success','both_failure','reference_unresolved')}
    for row in rows:
        a,b=(row['metrics'][x]['success_le5'] for x in (new,before))
        key=('reference_unresolved' if a is None or b is None else
            'failure_to_success' if a and not b else 'success_to_failure' if b and not a else
            'both_success' if a else 'both_failure')
        groups[key].append(row['id'])
    return dict(counts={k:len(v) for k,v in groups.items()},ids=groups,full_denominator=len(rows))


def recommendation(results,selection,rows):
    """Describe the sealed choice; never replace it with the PRIMARY winner."""
    chosen=selection['selected_arm']
    primary=[r for r in rows if r['partition']!='CAL']
    test=[r for r in rows if r['partition']=='TEST']
    versus_e=cells(primary,chosen,'E0')
    versus_e49=cells(test,chosen,'E0')
    s=results['populations']['PRIMARY']['arms'][chosen]
    e=results['populations']['PRIMARY']['arms']['E0']
    if chosen=='E0':
        label='NO_NEW_METHOD_ADOPTED'
        text='CAL에서 기존 고정 e(E0)가 선택됐다. 새 B/I 보정 후보는 최종 방법으로 채택하지 않는다.'
    elif (s['successes_le5']<e['successes_le5'] or
          versus_e['counts']['success_to_failure'] or versus_e49['counts']['success_to_failure']):
        label='CAL_SELECTED_BUT_PRIMARY_NOT_ADOPTABLE'
        text=(f'CAL 선택은 {chosen}였지만 PRIMARY에서 기존 e 대비 손상 또는 성능 저하가 확인됐다. '
              '새 방법 마무리 후보로서는 실패·미채택이며 운영 기본값 교체를 권하지 않는다.')
    else:
        label='CAL_SELECTED_RESEARCH_CANDIDATE_ONLY'
        text=(f'CAL에서 선택된 {chosen}는 현재 PRIMARY에서 기존 e의 성공 사례를 보존했다. '
              '반복 개발 자료의 연구 후보이며 독립 검증 전 운영 채택을 확정하지 않는다.')
    return dict(label=label,selected_arm=chosen,text=text,PRIMARY_versus_E0=versus_e,
        TEST_versus_E0=versus_e49,default_adoption_recommended=False,
        all192_solved=s['successes_le5']==s['denominator'],
        primary_failures=s['denominator']-s['successes_le5'])


def paired_table(population):
    lines=['| 대비 | 성공률 차이 %p | 95% 군집 CI %p | 회복 | 손상 | 양쪽 성공 | 양쪽 실패 | 미확정 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,d in population['comparisons'].items():
        r=d['success_rate'];c=d['transition']['counts'];ci=r.get('CI95')
        interval='NA' if ci is None else f"[{number(100*ci[0])}, {number(100*ci[1])}]"
        difference=None if r.get('mean_difference') is None else 100*r['mean_difference']
        lines.append(f"| {name} | {number(difference)} | {interval} | {c['failure_to_success']} | "
            f"{c['success_to_failure']} | {c['both_success']} | {c['both_failure']} | {c['reference_unresolved']} |")
    return '\n'.join(lines)


def _closed_inputs(folder,prior):
    paths={name:folder/relative for name,relative in {
        'method_protocol':'METHOD_PROTOCOL.json','selection':'SELECTION_SEAL.json',
        'score_seal':'score/SCORE_SEAL.json','results':'score/RESULTS.json',
        'scored_rows':'score/SCORED_ROWS.json','row_metrics':'score/ROW_METRICS.csv',
        'diagnostics':'diagnostics/DIAGNOSTICS.json','prediction_seal':'PREDICTION_SEAL.json',
        'E_input_contract':'e_inputs/CONTRACT.json','E_input_seal':'e_inputs/INPUT_SEAL.json',
        'timing':'timing/TIMING.json','resource_ledger':'RESOURCE_LEDGER.jsonl',
        'prior_results':'__prior__',}.items()}
    paths['prior_results']=prior/'score_v2/RESULTS.json'
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError('Required closed result absent: '+path.name)
    protocol,selection,seal,results,scored,diag=(read(paths[k]) for k in
        ('method_protocol','selection','score_seal','results','scored_rows','diagnostics'))
    if (selection['method_protocol_sha256']!=sha(paths['method_protocol']) or
        selection['prediction_seal_sha256']!=sha(paths['prediction_seal']) or
        selection['before_new_PRIMARY_scoring'] is not True or selection['PRIMARY_used_for_selection'] is not False or
        selection['selection_rule']!=protocol['selection_rule'] or
        seal['pre_PRIMARY_selection_seal_sha256']!=sha(paths['selection']) or
        seal['prediction_seal_sha256']!=sha(paths['prediction_seal']) or
        results['selection_seal_sha256']!=sha(paths['selection']) or
        results['selected_arm']!=selection['selected_arm'] or
        results['default_changed'] is not False or scored['arm_rows']!=len(ARMS)*len(scored['rows'])):
        raise ValueError('Closed selection/prediction/result ancestry mismatch')
    for name,h in seal['artifacts'].items():
        if sha(folder/'score'/name)!=h:
            raise ValueError('Saved score artifact changed: '+name)
    for pop in POPULATIONS:
        d=results['populations'][pop]
        if d['rows']!=protocol['populations'][pop] or set(d['arms'])!=set(ARMS):
            raise ValueError('Complete frozen population/all12 arms required')
    if len(scored['rows'])!=sum(protocol['populations'][p] for p in ('CAL','TEST','REMAINDER')):
        raise ValueError('Every original238 row required')
    for arm in ARMS:
        failure=[r['id'] for r in scored['rows'] if r['partition']!='CAL' and
                 r['metrics'][arm]['success_le5'] is not True]
        if failure!=diag['all_failures'][arm]:
            raise ValueError('Failure inventory changed: '+arm)
    receipt_paths=sorted(folder.glob('independent*/RECEIPT.json'))
    if not receipt_paths:
        raise FileNotFoundError('Independent saved-result check receipt is required')
    attempts=[]
    for path in receipt_paths:
        r=read(path)
        attempts.append(dict(attempt=path.parent.name,sha256=sha(path),checks=r['checks'],
            failure_count=r['failure_count'],prediction_seal_matches=r['prediction_seal_sha256']==sha(paths['prediction_seal']),
            score_seal_matches=r['score_seal_sha256']==sha(paths['score_seal'])))
        paths['independent_'+path.parent.name]=path
    if not any(x['failure_count']==0 and x['prediction_seal_matches'] and x['score_seal_matches'] for x in attempts):
        raise ValueError('No passing independent receipt matches the current sealed results')
    return paths,protocol,selection,results,scored['rows'],diag,attempts


def _csv(path,rows,fields):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def generate(folder,prior,output):
    """Render reviewable result documents without any new scientific operation."""
    folder,prior,output=Path(folder),Path(prior),Path(output)
    paths,protocol,selection,results,rows,diag,attempts=_closed_inputs(folder,prior)
    prior_results=read(paths['prior_results']);e_contract=read(paths['E_input_contract'])
    timing=read(paths['timing'])
    ledger=[json.loads(line) for line in paths['resource_ledger'].read_text().splitlines()]
    if (timing['all_outputs_checked'] is not True or timing['full_RGB_to_pose_seconds'] is not None or
        len(timing['samples'])!=len(timing['panel_ids'])*timing['repeats_per_image'] or
        not all(s['output_matches_accuracy'] for s in timing['samples'])):
        raise ValueError('Closed actual correction-block timing and output checks required')
    decision=recommendation(results,selection,rows)
    chosen=selection['selected_arm'];primary=results['populations']['PRIMARY'];test=results['populations']['TEST']
    owned=['RESULTS_KO.md','METHOD_KO.md','README_KO.md','REPRODUCE.md','REPORT_INPUTS.json','REPORT_MANIFEST.json',
        'diagnostics/ALL_FAILURES.csv','diagnostics/CHANGED_CASES.csv','diagnostics/PRIMARY_TAILS.csv',
        'diagnostics/PAIRED_TRANSITIONS.json']
    if any((output/p).exists() for p in owned):
        raise FileExistsError('Preserve previous generated report; choose a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    failures=[];tails=[]
    for r in rows:
        if r['partition']=='CAL':
            continue
        for arm in ARMS:
            value=dict(id=r['id'],partition=r['partition'],session=r['session'],cluster_id=r['cluster_id'],
                arm=arm,**r['metrics'][arm],**r['prediction_status'][arm])
            tails.append(value)
            if r['metrics'][arm]['success_le5'] is not True:
                failures.append(value)
    fields=list(tails[0])
    _csv(output/'diagnostics/ALL_FAILURES.csv',failures,fields)
    tails.sort(key=lambda r:(ARMS.index(r['arm']),r['translation_cm'] is None,
        -(r['translation_cm'] or 0),r['id']))
    _csv(output/'diagnostics/PRIMARY_TAILS.csv',tails,fields)
    changes=[];paired={}
    byid={r['id']:r for r in rows}
    for pop in POPULATIONS:
        paired[pop]={name:record['transition'] for name,record in results['populations'][pop]['comparisons'].items()}
        for contrast,d in paired[pop].items():
            new,before=contrast.split('-')
            for category in ('failure_to_success','success_to_failure'):
                for ident in d['ids'][category]:
                    r=byid[ident]
                    changes.append(dict(population=pop,contrast=contrast,category=category,id=ident,
                        partition=r['partition'],new_translation_cm=r['metrics'][new]['translation_cm'],
                        before_translation_cm=r['metrics'][before]['translation_cm'],
                        new_rotation_deg=r['metrics'][new]['rotation_deg'],before_rotation_deg=r['metrics'][before]['rotation_deg']))
    _csv(output/'diagnostics/CHANGED_CASES.csv',changes,['population','contrast','category','id','partition',
        'new_translation_cm','before_translation_cm','new_rotation_deg','before_rotation_deg'])
    write_json(output/'diagnostics/PAIRED_TRANSITIONS.json',paired)
    write_json(output/'REPORT_INPUTS.json',dict(schema='GUARDED_BI_SAVED_RESULT_REPORT_INPUTS_V1',
        input_sha256={role:sha(path) for role,path in paths.items()},
        reporter_source_sha256=sha(__file__),numeric_GT_or_reference_opens=0,new_image_NN_or_PnP_calls=0,
        decision=decision,independent_attempts=attempts,selection_created_or_changed=False))

    e0=primary['arms']['E0'];selected=primary['arms'][chosen]
    primary_transition=decision['PRIMARY_versus_E0']['counts'];test_transition=decision['TEST_versus_E0']['counts']
    text=[f"# 상태별 B/I 보정의 제한된 최종화 실험\n\n**{decision['label']} — {decision['text']}**",
        f"CAL에서 봉인한 최종 선택은 **{chosen} · {NAMES[chosen]}**이다. PRIMARY에서 다시 고르지 않았다. "
        f"선택 경로는 공동 성공 {selected['successes_le5']}/{selected['denominator']}, 실패 {decision['primary_failures']}장이다. "
        f"기존 e는 {e0['successes_le5']}/{e0['denominator']}이며 전체192장 해결 방법이 아니다. "
        f"e 위치 최대 {number(e0['errors']['translation_cm']['max'])}cm·회전 최대 {number(e0['errors']['rotation_deg']['max'])}°를 보존했다.",
        'CAL 선택과 운영 채택은 다르다. 이 자료는 이전 개발과 방법 선택에 반복 노출됐고 새 독립 평가가 아니다. '
        '운영 기본값·main·논문·정답은 이 실험에서 자동 변경하지 않았다.',
        '성공은 물체 중심 위치≤5cm와 고정 physical/C2 회전≤5°의 동시 충족이다. '
        '±SD는 표본 산포(ddof=1)이며 평균의 신뢰구간이 아니다. no-pose는 성공 분모에 실패로 남고, '
        '연속 오차는 가용 n을 병기했다. strict <5/<5와 정확한 경계 건수도 저장 결과에 보존했다. '
        '추가143장의 참조는 기존 주석에서 재구성한 값이며 독립 실측 GT가 아니다. 참조 정책·좌표·크기를 수정하지 않았다.']
    for pop in POPULATIONS:
        text.extend([f"## {pop} · {results['populations'][pop]['rows']}장",table(results['populations'][pop])])
    text.extend(['기준선0의 fallback/accepted가 저장 API에서 미기록이면 NA다. '
        'A/B의 fallback은 원자세 유지이며 pose 없음과 다르다. A/B accepted는 보정 교체 선택을 뜻하고, '
        '1의 accepted는 B/I solver 제약 수락이다. 어느 것도 정답 정확도 확률이 아니다.',
        '## CAL에서 무엇을 선택했는가',
        f"초기화별 선택은 RAW={selection['per_initialization']['R']}, N3/SubPix={selection['per_initialization']['N']}, "
        f"기존 e={selection['per_initialization']['E']}였다. A/B는 자기 기준선 대비 CAL 손상0·회복1 이상일 때만 후보가 된다. "
        '성공 수→손상 수→위치 평균→회전 평균→단순한 모드 순으로 초기화별 후보를 고르고, '
        '최종 초기화는 CAL 성공 수와 고정 E>N3>RAW 동점 순서로 고른다. 무조건 보정1은 선택 후보가 아니다.',
        '| 후보 | CAL 자격 | 자기 기준선 회복 | 자기 기준선 손상 |','|---|---:|---:|---:|'])
    for arm,d in selection['eligibility'].items():
        text.append(f"| {arm} | {'통과' if d['eligible'] else '제외'} | {d['recovery']} | {d['damage']} |")
    text.extend(['## 봉인 후 PRIMARY 대비',paired_table(primary),
        f"선택 경로 대 e는 회복{primary_transition['failure_to_success']}·손상{primary_transition['success_to_failure']}, "
        f"원 TEST49에서는 회복{test_transition['failure_to_success']}·손상{test_transition['success_to_failure']}이다. "
        'TEST49 손상이나 e 대비 손상이 남으면 운영 기본값 교체 권고 대상이 아니다. '
        '전체표의 다른 행이 더 좋아 보인다는 이유로 CAL 선택을 바꾸지 않았다.',
        f"원녹화 군집 {primary['bootstrap']['units']}개를 통째로 {primary['bootstrap']['resamples']:,}회 재표집했다"
        f"(기존 통계 seed {primary['bootstrap']['seed']}). 동일 draw와 원영상 수 가중 분모를 모든 대비에 공유한다. "
        '소수 군집·반복 개발 노출의 제한이 있다. 기존 성공률 가설검정 정의가 검증되지 않아 p/Holm p는NA이며 '
        'CI에서 p를 역산하거나 비유의성을 동등성으로 해석하지 않았다.',
        '## 모든 실패와 꼬리 사례',
        '[전체2856행](metrics/ROW_METRICS.csv), [PRIMARY 실패 전체](diagnostics/ALL_FAILURES.csv), '
        '[전체 꼬리 순서](diagnostics/PRIMARY_TAILS.csv), [회복·손상 ID](diagnostics/CHANGED_CASES.csv), '
        '[네 전환 셀 전체](diagnostics/PAIRED_TRANSITIONS.json)를 보존했다. '
        '어려운 영상·fallback·큰 유한 오차를 제거하거나 성공 프레임 평균으로 바꾸지 않았다.',
        '| 경로 | 위치 최대 사례 | 위치 cm | 회전 ° | 공동 성공 |','|---|---|---:|---:|---:|'])
    for arm in ARMS:
        available=[r for r in tails if r['arm']==arm and r['translation_cm'] is not None]
        if available:
            r=available[0]
            text.append(f"| {arm} | {r['id']} | {number(r['translation_cm'])} | {number(r['rotation_deg'])} | {r['success_le5']} |")
    text.extend(['## 보정 손상을 얼마나 막았는가',
        '| 후보 | PRIMARY 보정 선택 | 원자세 유지 | 무조건 보정 손상 차단 | 남은 손상 | 원래 회복 보존 | 회복 포기 |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for arm in ('RA','RB','NA','NB','EA','EB'):
        d=diag['arms'][arm]['PRIMARY']
        text.append(f"| {arm} | {d['accepted']} | {d['retained_baseline']} | "
            f"{len(d['original_proposal_damage_blocked'])} | {len(d['original_proposal_damage_still_accepted'])} | "
            f"{len(d['original_proposal_recovery_retained'])} | {len(d['original_proposal_recovery_discarded'])} |")
    text.extend(['규칙 B의 선 검사는 보정에 이미 쓴 영상 지지를 다시 보는 자기 일관성 검사다. '
        '수락·국소 rank6·ambiguity flag 부재·선 비용 개선으로 정답 또는 전역 유일성을 증명하지 않는다. '
        '[단계 진단 원문](diagnostics/DIAGNOSTICS.json)과 [실제 과정 갤러리](gallery/GALLERY_KO.md)를 함께 본다.',
        '선택 경로가 기존 e의 성공/실패를 바꾼 PRIMARY 사례를 모두 표시한다. '
        '선 지지에 잘 맞는 자세가 올바른 3D 자세라는 보장은 없으며 큰 회전 손상도 그대로 남겼다.',
        '| 전환 | 사례 | e 위치 cm → 선택 위치 cm | e 회전 ° → 선택 회전 ° | 선택 이유 |',
        '|---|---|---:|---:|---|'])
    for category in ('failure_to_success','success_to_failure'):
        for ident in decision['PRIMARY_versus_E0']['ids'][category]:
            r=byid[ident];before=r['metrics']['E0'];after=r['metrics'][chosen]
            text.append(f"| {'회복' if category=='failure_to_success' else '손상'} | {ident} | "
                f"{number(before['translation_cm'],4)} → {number(after['translation_cm'],4)} | "
                f"{number(before['rotation_deg'],4)} → {number(after['rotation_deg'],4)} | "
                f"{r['prediction_status'][chosen]['reason']} |")
    text.extend([
        '## N3/SubPix와 방법 마무리',
        f"앞선 동일 입력 ablation에서 RAW+B/I(R1)는 {prior_results['populations']['PRIMARY']['arms']['R1']['successes_le5']}/"
        f"{prior_results['populations']['PRIMARY']['rows']}, N3/SubPix+B/I(N1)는 "
        f"{prior_results['populations']['PRIMARY']['arms']['N1']['successes_le5']}/{prior_results['populations']['PRIMARY']['rows']}였다. "
        '현재 후속 실험은 N3/SubPix를 개별 제거한 재학습 비교가 아니며 제거의 안전성을 새로 증명하지 않는다. '
        '기존 묶음을 유지하는 권고를 보존한다. N3와 SubPix 각각의 기여는 이 실험으로 분리하지 않는다.',
        decision['text'],
        '기존 E0는 W/D를 고정하는 기존 전체 e 방법의 저장 예측이다. E1/EA/EB는 그 예측을 초기화로 '
        '자기 geometry·state·mask·B/I를 새로 계산한 별도 시험 경로다. 새 E 관측 결과가 E0를 생성했다고 설명하지 않는다.',
        '## 실제 비용과 검산',
        '이번 A/B guard는 저장 자기 관측과 두 자세를 비교하며 추가 NN·영상 질의·PnP가 없다. '
        '다음 시간은 이미 생성한 후보 뒤의 guard 선택 연산만이다.',
        '| 후보 | guard 평균 ms | 최대 ms | 계측 n |','|---|---:|---:|---:|'])
    for arm in ('RA','RB','NA','NB','EA','EB'):
        d=diag['arms'][arm]['PRIMARY']['actual_guard_ms']
        text.append(f"| {arm} | {number(d['mean'])} | {number(d['max'])} | {d['n']} |")
    text.extend(['별도로 같은 기존 CAL 패널에서 E의 실제 state/mask/B/I→pose→guard 보정 블록을 재실행했다. '
        f"패널{len(timing['panel_ids'])}장·batch{timing['batch']}·예열{timing['warmups_count']}회·영상당{timing['repeats_per_image']}회이며 "
        'A/B는 같은 observation·solve를 공유한 뒤 각각 guard를 측정했다.',
        '| 실제 E 보정 블록 | 평균±SD ms | 중앙값 ms | p95 ms | 계측 n |',
        '|---|---:|---:|---:|---:|'])
    for mode in ('A','B'):
        d=timing['summary'][mode]
        text.append(f"| E+{mode} | {number(d['mean_ms'])}±{number(d['sample_std_ms'])} | "
                    f"{number(d['median_ms'])} | {number(d['p95_ms'])} | {d['n']} |")
    text.extend([f"NN/model loading과 파일 decode는 계측 밖이고 CUDA 동기화={timing['CUDA_synchronized']}, "
        f"threads={timing['environment']['threads']}다. 모든 계측 출력의 pose·수락·fallback을 정확도 실행과 비교했다. "
        f"E/YOLO/N3/SubPix 앞단은 재실행하지 않아 전체 RGB 지연은NA다. "
        '[원시 시간과 환경](evidence/TIMING.json)을 남겼으며 캐시 시간을 합쳐 전체 FPS·Jetson 성능으로 부르지 않는다.',
        '| 자원 ledger 단계 | 종류 | 상태 | stage elapsed 초 | 합산 worker CPU 초 |',
        '|---|---|---|---:|---:|'])
    for entry in ledger:
        text.append(f"| {entry['stage']} | {entry['kind']} | {entry['status']} | "
            f"{number(entry['elapsed_seconds'])} | {number(entry.get('summed_worker_cpu_seconds'))} |")
    text.extend(['위 stage elapsed와 합산 worker CPU 시간은 서로 다른 양이다. 초기 모델 다운로드·캐시 생성 비용을 포함한 '
        '제품 전체 비용으로 해석하지 않는다.',
        f"e 입력은 물리 vertex 재번호 {e_contract['nonidentity_permutation_frames']}장, 최대 매칭오차 "
        f"{number(e_contract['maximum_physical_vertex_error_m'],16)}m였다. 원예측 좌표를 재번호했고 모델 투영점을 영상 관측으로 쓰지 않았다.",
        '| 독립 검산 시도 | 검사 수 | 실패 수 | 현재 봉인 일치 |','|---|---:|---:|---:|'])
    for attempt in attempts:
        text.append(f"| {attempt['attempt']} | {attempt['checks']} | {attempt['failure_count']} | "
                    f"{attempt['prediction_seal_matches'] and attempt['score_seal_matches']} |")
    text.extend(['검산은 저장 결과 산술·source/input 바인딩과 현재 guard/CAL 선택을 확인한다. '
        'hash만으로 기록되지 않은 모든 IO의 부재나 독립 일반화를 증명했다고 주장하지 않는다.',
        '[방법 정의](METHOD_KO.md), [재현 안내](REPRODUCE.md), [보고서 입력 SHA](REPORT_INPUTS.json)를 남겼다. '
        '새 학습·새 seed·새 데이터·정답 수정·운영 기본값 자동 교체는 없었다.'])
    write(output/'RESULTS_KO.md',paragraphs(text))

    methods=f'''# 현재 방법과 제한된 두 보정 규칙

이번 CAL 봉인 선택은 **{chosen} · {NAMES[chosen]}**다. {decision['text']}

기존 e(E0)는 초기 N3/SubPix와 상태별 관측, W/D 분기·전역 외곽·부분 경계 및 기존 영역 제약을 이용한 과거 전체 방법의 고정 예측이다. 과거 수치 자세를 그대로 보존했다. 전체192장 모두 성공했다는 뜻은 아니다. 새 E1/EA/EB의 state·mask·B/I로 E0를 다시 만들지 않았다.

후속 시험은 같은 RGB와 각자의 초기화를 쓴다. R은 RAW YOLO+자기 F, N은 N3/SubPix+자기 F, E는 위 고정 e다. R/N의 자기 관측은 앞선 봉인 실행을 재사용했다. E는 N3 실제 관측을 고정 physical vertex bijection으로 재번호하고, 자기 pose에서 geometry·state·SAM prompt/mask·B/I를 새로 계산했다. e 모델 투영점이나 혼합 readout은 영상 관측이 아니다.

| 상태 | 의미 | 이번 새 B/I 후단의 처리 |
|---|---|---|
| S | 초기 자세 조건부 자체 가림 | 영상/마스크 prompt/2D PnP 관측 제외, 최종 pose의 모델 투영만 |
| B | 보이는 물체-배경 경계 코너 | 실제 자기 경계선 관측으로 보정, 실패하면 자기 원좌표 |
| I | 보이는 물체 내부 코너 | 실제 자기 내부 RGB 접합 관측으로 보정, 실패하면 자기 원좌표 |
| O | 외부 가림으로 분류 | 새 정확한 2D 대응점에 넣지 않음 |
| T | 화면 밖 절단으로 분류 | 새 정확한 2D 대응점에 넣지 않음 |
| U | 판정 보류 | 새 정확한 2D 대응점에 넣지 않음 |

S와 B/I 기하 역할은 자기 초기 pose에 조건화된다. 실제 외부 가림과 보임은 고정 RGB 상태 모델이 따로 판단한다. 초기 F에 숨은 채널의 영향이 없었다거나 S 판정이 항상 정답이라고 주장하지 않는다. 상태 모델은 기존 CAL 실사 감독을 사용한 이력이 있다.

새 후보1은 자기 B/I만 동일 공분산·지지·고정 solver로 다시 푼다. 보정 영상 좌표와 최종 모델 투영을 구분한다. 자기 초기화 이외의 과거 후보 bank는 추가하지 않았다. E는 이름 그대로 e 초기화 하나를 사용하며 no-e 방법으로 부르지 않는다.

규칙 A는 기존 solver 수락, 고유 B/I≥{protocol['A']['minimum_unique_BI']}, visible rank={protocol['A']['visible_rank']}, ambiguity/local-underdetermined/두 점 pose-family flag 부재, 실제 RGB 보정 이동≥{protocol['A']['genuine_RGB_moved_BI_minimum']}개를 모두 요구한다. 이동 판정은 {protocol['A']['moved_tolerance_px']}px다. 조건이 부족하면 자기 기준 자세를 정확히 유지한다. rank6는 국소 식별성 조건이며 전역 유일성 증명이 아니다.

규칙 B는 A에 실제 저장 ray 점과 confidence의 선 일관성을 추가한다. 각 ray의 동일 native corner-neighbor 모델 투영 무한 직선까지 수직거리 Huber 비용(delta={protocol['B']['huber_delta_px']}px)을 비교한다. 각 ray가 초기 비용+1e−9보다 나쁘지 않고 전체 가중 평균이1e−9보다 엄격히 개선돼야 한다. 실제 유효 ray가 필요하다. 같은 fitting evidence를 재사용한 검사이며 독립 영상 검증·다른 물체의 깊이 소유권 확인이 아니다.

이후 후보를 늘리거나 임계값을 탐색하지 않았다. CAL46에서 각 자기 기준선 손상0·회복≥1인 A/B만 남겨 공동 성공→손상→위치 평균→회전 평균→단순한 모드 순으로 선택한다. 초기화 전체에서는 공동 성공과 E>N3>RAW 동점 규칙을 쓴다. CAL 선택을 봉인한 뒤 PRIMARY192를 평가하며 ID·TEST/추가143별로 방법을 바꾸지 않는다.

{decision['text']} 이 보고서는 방법 정의와 실패 근거를 남기는 별도 연구 문서다. 운영 코드·기존 실험·논문 원문을 덮어쓰지 않았다. [전체 결과](RESULTS_KO.md)와 [실제 과정](gallery/GALLERY_KO.md)에 연결한다.
'''
    write(output/'METHOD_KO.md',methods)
    readme=f'''# 상태별 B/I 보정 마무리 실험

**{decision['label']}**. CAL 선택은 {chosen}; PRIMARY 공동 성공은 {selected['successes_le5']}/{selected['denominator']}다. {decision['text']}

기존 e·실패·전체 분모를 보존했다. 새 독립 평가나 운영 기본값 교체가 아니다.

- [전체 결과표·채택 권고](RESULTS_KO.md)
- [방법 정의](METHOD_KO.md)
- [실제 단계별 갤러리](gallery/GALLERY_KO.md)
- [전체2856행](metrics/ROW_METRICS.csv)
- [PRIMARY 실패 전체](diagnostics/ALL_FAILURES.csv)
- [회복·손상 및 네 전환 셀](diagnostics/PAIRED_TRANSITIONS.json)
- [재현 안내](REPRODUCE.md)
- [보고서 입력 SHA](REPORT_INPUTS.json)
'''
    write(output/'README_KO.md',readme)
    reproduce='''# 재현 안내

이 실험은 기존 고정 가중치·입력·캐시를 사용한다. 새 환경 설치/학습/seed/데이터/정답 변경 없이 원래 검증 환경에서 실행한다. SOURCE_ROOT는 원 자산 저장소, PRIOR는 앞선 minimal-bi-ablation 보존본, FOLDER는 새 후속 실험 폴더, REPORT는 새 보고서 폴더다. 기존 봉인 폴더를 덮어쓰지 않는다. 공개 작은 결과만으로 대형 입력·모델·마스크 원시 캐시가 포함됐다고 주장하지 않는다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -m scripts.research.pallet_guarded_bi_final_v1.e_inputs --prior-input-dir "$PRIOR/inputs_v2" --source-root "$SOURCE_ROOT" --output-dir "$FOLDER/e_inputs"
python -m scripts.research.pallet_guarded_bi_final_v1.freeze --folder "$FOLDER" --prior "$PRIOR"
python -m scripts.research.pallet_guarded_bi_final_v1.run observe --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_guarded_bi_final_v1.run solve --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_guarded_bi_final_v1.run guard --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_guarded_bi_final_v1.score cal --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_guarded_bi_final_v1.score primary --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_guarded_bi_final_v1.analyze --folder "$FOLDER"
python -m pytest -q scripts/research/pallet_guarded_bi_final_v1
```

독립 검산은 현재 PREDICTION_SEAL.json과 score/SCORE_SEAL.json의 전체 SHA256을 명시해야 한다. 해당 값을 계산해 PREDSHA/SCORESHA 변수에 두고 새 검산 폴더로 실행한다. source/input이 달라졌거나 과거 실패가 있으면 그 시도를 보존한다.

```bash
python -m scripts.research.pallet_guarded_bi_final_v1.independent --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT" --output "$FOLDER/independent_v1" --prediction-seal-sha256 "$PREDSHA" --score-seal-sha256 "$SCORESHA"
python -m scripts.research.pallet_guarded_bi_final_v1.timing --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_guarded_bi_final_v1.report --folder "$FOLDER" --prior "$PRIOR" --output "$REPORT"
```

보고서 생성기는 score/RESULTS.json·SCORED_ROWS.json·DIAGNOSTICS.json·SELECTION_SEAL.json·독립 검산 영수증·실제 시간 결과와 ledger만 읽는다. 정답·원영상·관측·모델을 열거나 새 pose를 풀지 않는다. 보고서 입력과 소스 SHA는 REPORT_INPUTS.json에 있다. 공개 metrics·diagnostics·gallery 파일은 root의 별도 패키징 단계에서 원본 결과와 SHA를 확인해 복사한다.

성공 기준·physical/C2 alignment·전체238장·세부192/49/143/46 분모·가용 n·모든 no-pose/fallback을 유지한다. CAL 선택을 바꾸거나 PRIMARY 결과를 보고 새 후보/임계값을 추가하면 이 봉인 실험의 재현이 아니다.
'''
    write(output/'REPRODUCE.md',reproduce)
    write_json(output/'REPORT_MANIFEST.json',dict(schema='GUARDED_BI_GENERATED_REPORT_MANIFEST_V1',
        assets={name:sha(output/name) for name in owned if name!='REPORT_MANIFEST.json'},
        from_closed_saved_results_only=True,GT_or_reference_or_image_opens=0,new_NN_or_PnP_calls=0))
    return dict(output_files=owned,decision=decision,input_sha256={k:sha(p) for k,p in paths.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('folder','prior','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    result=generate(args.folder,args.prior,args.output)
    print(json.dumps({k:result['decision'][k] for k in ('label','selected_arm','primary_failures')},
                     ensure_ascii=False),flush=True)
