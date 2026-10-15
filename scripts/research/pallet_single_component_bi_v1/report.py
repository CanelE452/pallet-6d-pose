"""Saved-result-only report for four initializers x two fixed backends.

No selection, reference/image decode, model loading or pose computation exists
here. A report is generated only after all1904 rows and independent checks close.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from scripts.research.pallet_guarded_bi_final_v1.report import (
    sha,read,write,write_json,number,mean_sd,paragraphs,_csv)


ARMS=('R0','R1','N3_F','N3_BI','SUBPIX_F','SUBPIX_BI','N0','N1')
NAMES=dict(R0='RAW + F',R1='RAW + B/I',N3_F='N3 단독 + F',N3_BI='N3 단독 + B/I',
    SUBPIX_F='SubPix 단독 + F',SUBPIX_BI='SubPix 단독 + B/I',N0='N3·SubPix + F',N1='N3·SubPix + B/I')
POPS=('PRIMARY','TEST','REMAINDER','CAL')
MEANING={
    'R1-R0':'RAW 뒤 B/I 전체의 추가 효과', 'N1-N0':'묶음 뒤 B/I 전체의 추가 효과',
    'N3_F-R0':'B/I 이전 N3 단독', 'SUBPIX_F-R0':'B/I 이전 SubPix 단독',
    'N0-N3_F':'N3 위에 SubPix 추가 · F', 'N0-SUBPIX_F':'SubPix에 앞서 N3 추가 · F',
    'N3_BI-N3_F':'N3 단독 뒤 B/I 추가', 'SUBPIX_BI-SUBPIX_F':'SubPix 단독 뒤 B/I 추가',
    'N1-N3_BI':'N3 위에 SubPix 추가 · B/I 후단', 'N1-SUBPIX_BI':'SubPix에 앞서 N3 추가 · B/I 후단',
    'N3_BI-R1':'같은 B/I 앞 N3 단독', 'SUBPIX_BI-R1':'같은 B/I 앞 SubPix 단독'}


def table(pop):
    out=['| 경로 | 공동 성공/전체 | 위치 cm 평균±SD | 최대 cm | 회전 ° 평균±SD | 최대 ° | accepted | fallback | no-pose |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a in ARMS:
        d=pop['arms'][a];known=d['acceptance_known_n']
        out.append(f"| {a} · {NAMES[a]} | {d['successes_le5']}/{d['denominator']} | "
            f"{mean_sd(d['errors']['translation_cm'])} | {number(d['errors']['translation_cm']['max'])} | "
            f"{mean_sd(d['errors']['rotation_deg'])} | {number(d['errors']['rotation_deg']['max'])} | "
            f"{d['accepted_n'] if known else 'NA'} | {d['fallback_n'] if known else 'NA'} | {d['no_pose_n']} |")
    return '\n'.join(out)


def paired_table(pop):
    out=['| 대비 | 질문 | 성공률 차이 %p | 95% 군집 CI %p | 회복 | 손상 | 양쪽 성공 | 양쪽 실패 | 미확정 |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,d in pop['comparisons'].items():
        s=d['success_rate'];c=d['transition']['counts'];ci=s.get('CI95')
        interval='NA' if ci is None else f"[{number(100*ci[0])}, {number(100*ci[1])}]"
        diff=None if s.get('mean_difference') is None else s['mean_difference']*100
        out.append(f"| {key} | {MEANING.get(key,'기존 주대비')} | {number(diff)} | {interval} | "
            f"{c['failure_to_success']} | {c['success_to_failure']} | {c['both_success']} | "
            f"{c['both_failure']} | {c['reference_unresolved']} |")
    return '\n'.join(out)


def effect_text(pop,key):
    d=pop['comparisons'][key];c=d['transition']['counts'];s=d['success_rate'];ci=s.get('CI95')
    diff=None if s.get('mean_difference') is None else s['mean_difference']*100
    interval='NA' if ci is None else f"[{number(ci[0]*100)}, {number(ci[1]*100)}]%p"
    return (f"{key}: 성공률 차이 {number(diff)}%p, 95% 군집 CI {interval}, "
            f"회복{c['failure_to_success']}·손상{c['success_to_failure']}")


def closed(folder,prior):
    paths={k:folder/v for k,v in dict(protocol='PROTOCOL.json',predseal='PREDICTION_SEAL.json',
        scoreseal='score/SCORE_SEAL.json',results='score/RESULTS.json',rows='score/SCORED_ROWS.json',
        inputseal='inputs_v1/INPUT_SEAL.json',contract='inputs_v1/CONTRACT.json',ledger='RESOURCE_LEDGER.jsonl',
        compliance='compliance_v1/COMPLIANCE_AUDIT.json',legacy='compliance_v1/LEGACY245_110_SOURCE_ID_AUDIT.json',
        old_main_observation='compliance_v1/MAIN_PUBLICATION_OBSERVATION.json').items()}
    paths.update({k:folder/v for k,v in dict(stage_diagnostics='diagnostics/STAGE_DIAGNOSTICS.json',
        CAL_state_diagnostics='diagnostics/CAL_STATE_DIAGNOSTICS.json',
        diagnosticseal='diagnostics/DIAGNOSTIC_SEAL.json').items()})
    paths['prior_results']=prior/'score_v2/RESULTS.json'
    for p in paths.values():
        if not p.is_file():raise FileNotFoundError('Closed report input missing: '+p.name)
    protocol,pred,seal,result,saved=(read(paths[k]) for k in ('protocol','predseal','scoreseal','results','rows'))
    if (pred['arms']!=list(ARMS) or pred['rows']!=protocol['rows'] or pred['arm_rows']!=protocol['arm_rows'] or
        pred['protocol_sha256']!=sha(paths['protocol']) or seal['prediction_seal_sha256']!=sha(paths['predseal']) or
        result['all_rows_kept'] is not True or result['no_method_selection'] is not True or
        result['default_changed'] is not False or result['old_four_arm_poses_and_metrics_exact'] is not True):
        raise ValueError('Fixed eight-arm result/protocol ancestry changed')
    for n,h in seal['artifacts'].items():
        if sha(folder/'score'/n)!=h:raise ValueError('Saved score changed: '+n)
    ds=read(paths['diagnosticseal'])
    if ds['prediction_seal_sha256']!=sha(paths['predseal']) or ds['score_seal_sha256']!=sha(paths['scoreseal']):
        raise ValueError('Postscore diagnostic ancestry changed')
    for n,h in ds['assets'].items():
        if sha(folder/'diagnostics'/n)!=h:raise ValueError('Saved diagnostic changed: '+n)
    for pop in POPS:
        if (result['populations'][pop]['rows']!=protocol['populations'][pop] or
            set(result['populations'][pop]['arms'])!=set(ARMS) or
            set(result['populations'][pop]['comparisons'])!={a+'-'+b for a,b in protocol['contrasts']}):
            raise ValueError('Full population or fixed contrasts changed')
    if (len(saved['rows'])!=protocol['rows'] or saved['arm_rows']!=protocol['arm_rows'] or
        [r['id'] for r in saved['rows']]!=protocol['ordered_ids']):raise ValueError('All ordered rows required')
    attempts=[]
    for p in sorted(folder.glob('independent*/RECEIPT.json')):
        d=read(p);match=(d['prediction_seal_sha256']==sha(paths['predseal']) and d['score_seal_sha256']==sha(paths['scoreseal']))
        attempts.append(dict(attempt=p.parent.name,checks=d['checks'],failure_count=d['failure_count'],
            same_seals=match,sha256=sha(p)));paths['independent_'+p.parent.name]=p
    if not any(a['failure_count']==0 and a['same_seals'] for a in attempts):
        raise ValueError('Passing independent receipt matching all current rows required')
    return paths,protocol,result,saved['rows'],attempts


def compliance_final(original,publication,*,rows,checks):
    out=deepcopy(original)
    out['schema']='MINIMAL_BI_COMPLIANCE_ADDITIVE_FINAL_VIEW_V1'
    out['original_v1_audit_preserved']=True
    out['new_individual_component_arm_rows']=rows
    out['new_independent_checks']=checks
    out['method_or_default_selection']=False
    out['new_publication_status']='Final new-results main push is recorded separately by root; this generator does not claim it.'
    for item in out['requirements']:
        if item['id']=='single_components':
            item['status']='COMPLETE_NEW_INDIVIDUAL_COMPONENT_EXPERIMENT'
            item['finding']='이번 추가8경로에서 N3단독/SubPix단독의자기F및자기B/I를전체238장1904행으로비교하고독립검산했다. 이전묶음비교에서이미완료했다고소급하지않는다.'
        elif item['id']=='new_main_push':
            item['status']='PRIOR_WORK_MAIN_PUBLISHED_NEW_RESULT_PUSH_SEPARATELY_RECORDED'
            item['finding']='이전minimal/guarded main반영은read-only원격확인기록에있다. 현재추가실험최종mainpush는root의별도게시영수증으로확인하며이생성기는완료를선언하지않는다.'
    out['publication_observation']=publication
    out['remaining_limits']=['일부초기읽기/짧은probe/프로세스시작종료의정밀시간은회복불가',
        '확인된245예측행=49unique×5,110=동일192중성공분자. 원문이다른245/110unique영상표를뜻했는지는확정불가',
        '전체YOLO→N3/SubPix→F→B/I시간/FPS미측정(원문NA허용)',
        'p/Holm은기존검정정의미검증으로NA(원문허용)',
        '반복개발자료이며새독립촬영평가/운영기본교체는수행하지않음']
    return out


def generate(folder,prior,output):
    folder,prior,output=map(Path,(folder,prior,output))
    paths,protocol,result,rows,attempts=closed(folder,prior)
    primary=result['populations']['PRIMARY'];contract=read(paths['contract']);old=read(paths['prior_results'])
    compliance=compliance_final(read(paths['compliance']),read(paths['old_main_observation']),
        rows=protocol['arm_rows'],checks=sum(a['checks'] for a in attempts if not a['failure_count'] and a['same_seals']))
    owned=['README.md','RESULTS_KO.md','REPRODUCE.md','COMPONENT_DECISIONS.md','COMPLIANCE_FINAL_KO.md',
        'COMPLIANCE_AUDIT_FINAL.json','REPORT_INPUTS.json','REPORT_MANIFEST.json',
        'diagnostics/ALL_FAILURES.csv','diagnostics/PRIMARY_TAILS.csv','diagnostics/PAIRED_TRANSITIONS.json']
    if any((output/x).exists() for x in owned):raise FileExistsError('Preserve every prior report; choose new output')
    output.mkdir(parents=True,exist_ok=True)
    flat=[]
    for r in rows:
        if r['partition']=='CAL':continue
        for a in ARMS:
            flat.append(dict(id=r['id'],partition=r['partition'],session=r['session'],cluster_id=r['cluster_id'],
                arm=a,**r['metrics'][a],**r['prediction_status'][a]))
    fields=list(flat[0]);fail=[r for r in flat if r['success_le5'] is not True]
    _csv(output/'diagnostics/ALL_FAILURES.csv',fail,fields)
    tail=sorted(flat,key=lambda r:(ARMS.index(r['arm']),r['translation_cm'] is None,-(r['translation_cm'] or 0),r['id']))
    _csv(output/'diagnostics/PRIMARY_TAILS.csv',tail,fields)
    write_json(output/'diagnostics/PAIRED_TRANSITIONS.json',{pop:{k:v['transition'] for k,v in
        result['populations'][pop]['comparisons'].items()} for pop in POPS})
    write_json(output/'COMPLIANCE_AUDIT_FINAL.json',compliance)
    write_json(output/'REPORT_INPUTS.json',dict(schema='SINGLE_COMPONENT_SAVED_REPORT_INPUTS_V1',
        inputs_sha256={k:sha(p) for k,p in paths.items()},reporter_source_sha256=sha(__file__),
        no_reference_image_model_or_PnP_opens=True,no_method_selection=True,independent_attempts=attempts))

    first=['# N3 단독·SubPix 단독과 B/I 후단의 전체 비교',
        '**N3 단독과 SubPix 단독까지 전체238장·8경로 평가와 독립 검산을 완료했다.** '
        '이전952행은 RAW 대 N3→SubPix 묶음 비교였으며 개별 성분을 이미 확인한 것으로 설명하면 안 된다. '
        '이번에 그 공백을 별도 실험으로 보충했다.',
        f"고정 원영상 {protocol['rows']}장에 8경로를 적용해 {protocol['arm_rows']}행을 보존했다. "
        '주평가는 TEST49+추가143=192장이고 CAL46은 별도다. 새 학습·seed·정답 수정·방법 선택·운영 기본값 교체는 없다. '
        '이 자료는 반복 개발에 노출됐고 새 독립 평가가 아니다.',
        '성공은 같은 모델 중심 위치≤5cm AND physical/C2 회전≤5°다. ±SD는 표본 산포(ddof=1)이며 평균의 CI가 아니다. '
        '큰 유한 오차를 절단하지 않았고 no-pose는 성공 분모의 실패로 남긴다. 연속 오차의 가용 n을 표기한다. '
        '추가143 참조는 기존 주석 재구성 값이며 독립 실측 GT가 아니다.']
    for pop in POPS:first += [f"## {pop} · {result['populations'][pop]['rows']}장",table(result['populations'][pop])]
    first += ['F 기준선의 accepted/fallback 기록이 없으면 NA다. B/I accepted는 제약 수락이지 정답 성공이 아니다. '
        'fallback은 자기 기준 자세 복귀이며 no-pose와 구분한다. strict<5/<5와 정확한 경계 건수는 원결과에 보존했다.',
        '## 각각 무엇을 얻고 잃었는가',
        effect_text(primary,'N3_F-R0')+'; '+effect_text(primary,'N3_BI-R1')+'.',
        effect_text(primary,'SUBPIX_F-R0')+'; '+effect_text(primary,'SUBPIX_BI-R1')+'.',
        effect_text(primary,'N0-N3_F')+'; '+effect_text(primary,'N1-N3_BI')+'.',
        effect_text(primary,'N0-SUBPIX_F')+'; '+effect_text(primary,'N1-SUBPIX_BI')+'.',
        '처음 두 줄은 RAW에 각 성분을 단독으로 붙인 효과다. 뒤 두 줄은 기존 N3→SubPix 순서에서 한 성분이 더해진 조건부 효과다. '
        '앞단이 바뀌면 자기 F·상태·프롬프트·마스크·B/I 관측도 바뀌므로 이를 좌표 한 단계의 순수 인과 기여로 부르지 않는다.',
        '## 사전 고정12개 대비',paired_table(primary)]
    boot=primary['bootstrap']
    first += [f"원녹화 군집 {boot['units']}개를 통째로 {boot['resamples']:,}회 재표집했다(기존 통계 seed {boot['seed']}). "
        '모든 방법의 짝과 영상 수 가중 추정 대상을 유지한다. 소수 군집·과거 개발 노출의 제한이 있다. '
        'p/Holm은 검정 정의를 검증하지 못해 NA이며 CI에서 역산하지 않았다. 비유의/CI의 0 포함은 동등성·비열등성 증거가 아니다.',
        '원 최소 실험의 교차 주대비 R1−N1과 R1−N0는 새12대비에 추가하지 않고 이전 봉인 결과로 분리해 보존한다. '
        '기존4경로의 예측과 지표는 이번에도 정확히 같다.',
        paired_table(dict(comparisons={k:old['populations']['PRIMARY']['comparisons'][k] for k in ('R1-N1','R1-N0')})),
        '## 입력의 실제 의미',
        f"N3단독: {contract['N3_ONLY']}. SubPix단독: {contract['SUBPIX_ONLY']}. "
        f"묶음: {contract['N3_SUBPIX']}. N3 단독의 기존 float32 상한 roundoff도 보존하고 추가 상한을 조용히 적용하지 않았다.",
        '과거20261008의319장 캐시에 BASE/N3/SUBPIX/N3_SUBPIX 초기 좌표는 이미 있었다. '
        '그것은 현재192장의 각 초기화 뒤에 같은 B/I를 독립 계산한8경로 비교를 완료했다는 뜻은 아니다. '
        '이번에는 정확한238장 부분집합과 원래 선택한 YOLO 실행/좌표를 연결하고 누락된 자기 F만 계산한 뒤, '
        'N3 단독/SubPix 단독의 상태·마스크·B/I·pose를 각자 새로 계산했다.',
        f"자기 F 준비는 총 {contract['resources']['F_calls']}회, 기존 cache 대조 {contract['resources']['cached_F_parity_calls']}회,"
        f"누락계산{contract['resources']['new_missing_F']}회다. 새 YOLO/N3/SubPix forward는0이다.",
        'S는 자기 초기 자세 조건부 기하 판정이며 국소 RGB·마스크 prompt·B/I 2D 대응에서 제외한다. '
        '최종 S는 모델 투영으로만 보고하고 O/T/U도 새 정확 2D 대응에 넣지 않는다. '
        '보정 실제 관측과 모델 투영은 다르며 투영을 다시 관측으로 넣지 않았다.',
        '## 자기 관측과 CAL 상태 진단',
        '네 초기화의 상태·마스크·B/I 관측과 solver 기록을 별도로 보존했다. 다음 CAL 사람 상태 비교는 '
        '기존 physical vertex bijection으로 번호를 옮긴 진단이며 실제 물리 코너 의미의 일치까지 증명하지 않는다. '
        'CAL은 기존 상태 모델의 감독 자료이므로 독립 가림 성능으로 일반화하지 않는다.',
        '| 초기화 | DIRECT 전체 / B·I 판정 | EXTERNAL 전체 / B·I 판정 | SELF 전체 / S 판정 | UNANNOTATED |',
        '|---|---:|---:|---:|---:|']
    human=read(paths['CAL_state_diagnostics'])
    for branch,matrix in human['matrices'].items():
        total=lambda label:sum(x['count'] for x in matrix if x['actual']==label)
        mapped=lambda label,predictions:sum(x['count'] for x in matrix if x['actual']==label and x['predicted'] in predictions)
        first.append(f"| {branch} | {total('DIRECT_VISIBLE')} / {mapped('DIRECT_VISIBLE',('B','I'))} | "
            f"{total('EXTERNAL_OCCLUDED')} / {mapped('EXTERNAL_OCCLUDED',('B','I'))} | "
            f"{total('SELF_OCCLUDED')} / {mapped('SELF_OCCLUDED',('S',))} | {total('UNANNOTATED')} |")
    first += [f"사람 B/I 역할 정답: {human['human_B_vs_I_GT']}; 절단 계측 정답: {human['human_T_metrology_GT']}. "
        f"상태 transport 불가 기록은 {len(human['unavailable'])}개다. 미주석 상태는 제거하지 않았다.",
        '| 초기화 · PRIMARY | B/I 관측 | 실제 새 교점 | 원예측 유지 | 실제 이동 |',
        '|---|---:|---:|---:|---:|']
    stages=read(paths['stage_diagnostics'])
    for branch in ('RAW_YOLO','N3_ONLY','SUBPIX_ONLY','N3_SUBPIX'):
        rr=[r['diagnostics'] for r in stages['rows'] if r['branch']==branch and r['partition']!='CAL']
        first.append('| '+branch+' | '+' | '.join(str(sum(r[k] for r in rr)) for k in
            ('measured_BI_n','new_intersection_n','retained_backbone_n','moved_n'))+' |')
    first += ['[저장 단계 진단](diagnostics/STAGE_DIAGNOSTICS.json.gz)과 '
        '[CAL 상태 행렬](diagnostics/CAL_STATE_DIAGNOSTICS.json)은 점수 봉인 후 만든 분석이며 새 학습·보정 입력으로 쓰지 않았다.',
        '## 모든 실패·회복·손상',
        '[전체1904행](metrics/ROW_METRICS.csv), [전체회복·손상](metrics/CHANGED_CASES.csv), '
        '[네전환셀과전체ID](diagnostics/PAIRED_TRANSITIONS.json), '
        '[PRIMARY실패전부](diagnostics/ALL_FAILURES.csv), [전체꼬리순서](diagnostics/PRIMARY_TAILS.csv)를 남겼다.',
        '| 경로 | 위치최대사례 | 위치 cm | 회전 ° |','|---|---|---:|---:|']
    for a in ARMS:
        avail=[r for r in tail if r['arm']==a and r['translation_cm'] is not None]
        if avail:
            r=avail[0];first.append(f"| {a} | {r['id']} | {number(r['translation_cm'])} | {number(r['rotation_deg'])} |")
    first += ['## 권고와 남은 범위',
        '이 표는 성분별 효과를 분리한 대조실험이다. CAL 또는 PRIMARY에서 가장 높은 행을 골라 새 운영 기본값으로 승격하지 않는다. '
        '공동 성공 증가와 손상·평균·꼬리·fallback·자기 상태 모델 전이 제한을 함께 본다. '
        '현재 자료의 좋은 점추정만으로 모든 경로에서 N3나 SubPix가 필수라고 하거나 안전하게 제거 가능하다고 단정하지 않는다.',
        '[성분별 판정](COMPONENT_DECISIONS.md)에 위 대비를 함께 정리했다. '
        '운영 e 교체·논문 수정 전에는 고정한 경로의 손상과 대응 의미 문제가 독립 촬영 군집에서도 재현되는지 확인해야 한다. '
        '기존 guarded 마무리 실험의 실패와 기존 e의 큰 오차도 덮어쓰지 않았다.',
        '## 검산·비용·이행 감사',
        '| 독립검산시도 | 검사수 | 실패수 | 현재봉인일치 |','|---|---:|---:|---:|']
    for a in attempts:first.append(f"| {a['attempt']} | {a['checks']} | {a['failure_count']} | {a['same_seals']} |")
    ledger=[json.loads(line)for line in paths['ledger'].read_text().splitlines()]
    first += ['검산은 실제 원 초기 캐시·기존4경로 불변·새476자기 관측과 pose·물리/C2 지표·전환·통계와 CSV를 대조한다. '
        '새 독립 GT를 획득한 것은 아니다. source/artifact 해시는 기록되지 않은 모든 IO 부재의 절대 증명은 아니다.',
        '| 실행단계 | 종류 | stage elapsed 초 | 합산worker CPU 초 | 상태 |','|---|---|---:|---:|---|']
    for e in ledger:first.append(f"| {e['stage']} | {e['kind']} | {number(e['elapsed_seconds'])} | {number(e.get('summed_worker_cpu_seconds'))} | {e['status']} |")
    first += ['이번 추가 실험에서 각 성분의 전체 앞단 지연이나 전체 RGB→pose/FPS를 실측하지 않았다. '
        '원 지시문은 전체 호출을 연결하지 못했을 때 NA를 허용하지만, 그 NA를 전체 속도 감소의 증거로 바꾸지 않는다. '
        '앞선 minimal의5회 반복 보정 블록 시간과 guarded의 별도 시간은 그대로 보존한다.',
        '[원문이행감사와남은공백](COMPLIANCE_FINAL_KO.md), [245/110표기출처](LEGACY245_110_SOURCE_ID_AUDIT_KO.md), '
        '[재현](REPRODUCE.md), [보고서입력SHA](REPORT_INPUTS.json)를함께남겼다. '
        '이 보고서 생성기는 main 게시를 실행하지 않으며 최종 게시 확인은 root의 별도 영수증을 따른다.']
    write(output/'RESULTS_KO.md',paragraphs(first))

    components=['# N3와 SubPix 성분별 판정',
        '방법을 선택하거나 기본값을 교체하지 않는다. 아래는 전체 PRIMARY의 짝지은 대비이며 CAL 선택 성능이 아니다.']
    for name,keys in [('N3 단독',('N3_F-R0','N3_BI-R1')),('SubPix 단독',('SUBPIX_F-R0','SUBPIX_BI-R1')),
        ('N3 위 SubPix 추가',('N0-N3_F','N1-N3_BI')),('SubPix 앞 N3 추가',('N0-SUBPIX_F','N1-SUBPIX_BI')),
        ('각 단독 성분 뒤 B/I 추가',('N3_BI-N3_F','SUBPIX_BI-SUBPIX_F'))]:
        components += ['## '+name]+[effect_text(primary,k)+'.' for k in keys]
    components += ['RAW와 각 단독 초기화는 같은 모델 가중치·임계값·B/I/solver를 쓰되 자기 기하·상태·mask를 갖는다. '
        '따라서 한 성분 제거의 전체 경로 효과이며 관측 좌표만의 순수 인과 기여나 독립 일반화 보장이 아니다.',
        '남긴 것은 원 YOLO 선택 출력·고정 N3/SubPix 정의·상태 ResNet/Tiny/MobileSAM·선 지지/covariance·동일 solver다. '
        '최소 후단에서 U 표시·OT 영역 생성·V33 비용/과거 bank·보고서 의존을 제거한 결정은 앞선 동등성 실험 그대로다. '
        '이 추가 실험은 불리한 성분이나 실패 기록을 삭제하지 않는다.',
        '남은 최소 확인은 고정 경로의 회복·손상 및 물리 코너 대응 의미를 독립 촬영 군집에서 검증하는 것이다. '
        '이번 결과를 보고 새 학습·seed·임계값 탐색을 자동 시작하지 않았다.']
    write(output/'COMPONENT_DECISIONS.md',paragraphs(components))
    audit=['# 원문 이행 감사: 추가 실험 후 상태',
        '원문497줄 전체와 산출물을 대조한 v1 감사를 보존하고 이번 추가 실험 상태만 덧붙였다. '
        '**이전에도 모든 세부가 다 끝났다고 말하면 과장이었다.** 원4경로 핵심은 완료했지만 개별 성분과 기록상의 공백을 구분해야 한다.',
        '원문166줄은 N3/SubPix 묶음 제거 설계다. 따라서 N3 단독/SubPix 단독은 이전952행의 완료 항목이 아니었고, '
        f"현재 새 요청에서 전체{protocol['arm_rows']}행과 독립 검산으로 보충했다. 과거319장의4초기 캐시 존재와 현재192장 B/I 교차 평가는 다르다.",
        '원문403줄은 전체 호출 연결 불가 시 전체 시간 NA,348줄은 기존 검정 정의 미검증 시 p/Holm NA를 허용한다. '
        '두 NA 자체가 지시 위반은 아니지만, 전체 속도와 통계적 유의성을 측정했다는 뜻으로 포장하지 않는다.',
        '245/110는 이전 contract에서 미확정이었다. 이번 감사로 확인된245행은49unique×5방법이고,110은 동일192장의 성공 분자다. '
        '원문이 따로 다른245/110unique 영상 표를 가리켰는지는 여전히 확정되지 않았다. 전체 ID 차집합·출처 SHA를 별도 추가 문서에 남겼다.',
        '일부 초기 읽기·짧은 probe·프로세스 시작/종료의 정밀 시간은 회복할 수 없다. '
        '기존 기록은 주요 stage와 보수적 포함 구간의 상한 확인이지 모든 개별 초수의 완전 계측은 아니다.',
        '새 사용자의 main 게시 지시는 원문의 별도 브랜치 지침을 대체한다. 이전 minimal/guarded 작업의 main 반영은 감사의 read-only 원격 확인에 기록했다. '
        '이번 추가 결과의 최종 main push 완료는 별도 root 게시 영수증으로 확인하며 보고서 생성 시 완료를 미리 선언하지 않는다.',
        '| 감사 항목 | 현재 판정 |','|---|---|']
    for x in compliance['requirements']:audit.append('| '+x['id']+' · §'+x['section']+' | '+x['status']+' |')
    audit += ['[전체30항목기계판독감사](COMPLIANCE_AUDIT_FINAL.json), [245/110근거](LEGACY245_110_SOURCE_ID_AUDIT_KO.md). '
        '기존 봉인 파일·사용자 수정·실패·논문을 고치지 않았다. 새 독립 촬영 평가·운영 기본값 교체는 수행하지 않았다.']
    write(output/'COMPLIANCE_FINAL_KO.md',paragraphs(audit))
    write(output/'README.md',paragraphs(['# N3 단독·SubPix 단독의 추가 비교',
        f"같은{protocol['rows']}장·8경로·{protocol['arm_rows']}행을전체평가하고독립검산했다. 이전실험은묶음제거비교였고, 이번은개별성분을분리한추가실험이다. 방법선택·운영기본교체는없다.",
        '- [전체결과·짝지은회복/손상/CI](RESULTS_KO.md)\n- [성분별판정](COMPONENT_DECISIONS.md)\n- [원문이행감사](COMPLIANCE_FINAL_KO.md)\n- [전체지표](metrics/ROW_METRICS.csv)\n- [실패전부](diagnostics/ALL_FAILURES.csv)\n- [재현](REPRODUCE.md)']))
    reproduce='''# 재현

기존환경과확정가중치·정확한원캐시를사용한다. PRIOR는앞선minimal보존본, SOURCE_ROOT는원자산저장소, FOLDER는새출력폴더, REPORT는새보고서폴더다. 기존봉인을덮어쓰지않는다.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -m scripts.research.pallet_single_component_bi_v1.inputs --prior-input-dir "$PRIOR/inputs_v2" --source-root "$SOURCE_ROOT" --output "$FOLDER/inputs_v1"
python -m scripts.research.pallet_single_component_bi_v1.run freeze --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_single_component_bi_v1.run observe --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_single_component_bi_v1.run solve --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_single_component_bi_v1.score --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT"
python -m scripts.research.pallet_single_component_bi_v1.diagnostics --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT" --output "$FOLDER/diagnostics"
python -m pytest -q scripts/research/pallet_single_component_bi_v1
```

독립검산은실제완료PREDICTION_SEAL과SCORE_SEAL의전체SHA를PREDSHA/SCORESHA로명시한다. 새검산폴더를사용하고실패시도를보존한다.

```bash
python -m scripts.research.pallet_single_component_bi_v1.independent --folder "$FOLDER" --prior "$PRIOR" --source-root "$SOURCE_ROOT" --output "$FOLDER/independent_v1" --prediction-seal-sha256 "$PREDSHA" --score-seal-sha256 "$SCORESHA"
python -m scripts.research.pallet_single_component_bi_v1.report --folder "$FOLDER" --prior "$PRIOR" --output "$REPORT"
```

감사의기존v1JSON·245/110출처감사·원격main관찰은FOLDER/compliance_v1에보존된메타데이터다. 보고서생성기는완료점수·독립검산·감사만읽으며정답·영상·모델을열거나pose를풀지않는다. 초기화별mask관측476개와solver후보등대형원배열은비공개보존본에있고공개작은파일이전체자산을포함한다고주장하지않는다.

재개는동일source/input/model/config와자기관측binding을통과한완료행만재사용한다. 기준선R0/R1/N0/N1예측과지표의정확한불변,새N3/SubPix초기캐시재현,모든실패/분모/물리C2정의를검산한다. source해시·정확명령·완료seal은evidence/metrics원결과와REPORT_INPUTS.json에있다. 사용자의최신지시에따라최종main게시는root가별도수행·검증한다. 운영기본값과논문변경은없다.
'''
    write(output/'REPRODUCE.md',reproduce)
    write_json(output/'REPORT_MANIFEST.json',dict(schema='SINGLE_COMPONENT_GENERATED_REPORT_MANIFEST_V1',
        assets={x:sha(output/x) for x in owned if x!='REPORT_MANIFEST.json'},
        no_reference_image_model_or_PnP_calls=True,no_selection_or_default_change=True))
    return dict(rows=protocol['rows'],arm_rows=protocol['arm_rows'],output_files=owned,
        primary_successes={a:primary['arms'][a]['successes_le5'] for a in ARMS},method_selected=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('folder','prior','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();result=generate(args.folder,args.prior,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='output_files'},ensure_ascii=False),flush=True)
