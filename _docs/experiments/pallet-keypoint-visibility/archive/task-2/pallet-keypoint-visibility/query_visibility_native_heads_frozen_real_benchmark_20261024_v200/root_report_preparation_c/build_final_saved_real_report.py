#!/usr/bin/env python3
"""Assemble only closed numerical and gallery evidence into exclusive reports."""
import hashlib
import html
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

V = Path(__file__).parent.parent
PREP = Path(__file__).parent
SCOPE = V / 'ROOT_FINAL_REAL_BENCHMARK_REPORT_ASSEMBLY_SCOPE_C.json'
SCOPE_SHA = '446e434771a90f6f9badda0ed86054e919c94a33e17eac4e6e9286c112dd46eb'
NUM = V / 'numeric_readout_v1'
CLOSED = V / 'numeric_report_preparation_c/NUMERIC_READOUT_COMPLETE.json'
CLOSED_SHA = 'd5ad7b62e10db83d4ea396cee9a6b80f2766ba6fb18be65637debe1c23c5cb3d'
COMPARISON = V.parent / 'query_visibility_source_group_split_direct_end_training_20261024_v194/external_review_received/RECEIVED_REVIEW_AND_ROOT_COMPARISON.json'
COMPARISON_SHA = '4a466b7cfa41e01a9c85b5c568ebfb9ef5be8357ad48ce8c57851f420534c314'


def safe(p):
    p = Path(p)
    assert not any(re.search(r'(^|_)v112($|_|\.)', x, re.I) for x in p.parts)
    for x in [p, *p.parents]:
        assert not x.is_symlink(), str(x)
    return p


def sha(p):
    return hashlib.sha256(safe(p).read_bytes()).hexdigest()


def bind(p):
    return {'path': str(p), 'sha256': sha(p), 'size_bytes': safe(p).stat().st_size}


def read(b):
    p = safe(b['path'])
    assert p.suffix == '.json'
    assert sha(p) == b['sha256'], str(p)
    return json.loads(p.read_text())


def rel(p):
    return Path(os.path.relpath(p, V)).as_posix()


def dump(p, x):
    with p.open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def rebase(ref):
    assert not ref.startswith(('/', 'http:', 'https:', 'file:'))
    return Path(os.path.normpath(Path('numeric_readout_v1') / ref)).as_posix()


def mdtable(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n|' + '|'.join(['---'] * len(headers)) + '|\n' + ''.join('| ' + ' | '.join(map(str, r)) + ' |\n' for r in rows)


def run():
    assert sha(SCOPE) == SCOPE_SHA
    scope = json.loads(SCOPE.read_text())
    assert scope['approved_saved_evidence_report_assembly_only']
    expected = [V/n for n in ['ROOT_RESULTS_KO.json','ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html']]
    assert [str(x) for x in expected] == scope['allowed_new_files']
    assert str(PREP) == scope['allowed_builder_directory']
    assert all(not p.exists() for p in expected), 'No overwrites'
    assert sha(CLOSED) == CLOSED_SHA
    closed = json.loads(CLOSED.read_text())
    assert closed['completed'] and closed['PASS'] and closed['figure_tool_view_pending'] is False
    numerical_b = closed['output_files26']['RESULTS_KO.json']
    numeric = read(numerical_b)
    evidence = {k: read(scope[k]) for k in ['saved_evaluation_complete','independent_numeric_audit','fixed_gallery_complete','root6_actual_view','other6_actual_view_and_saved_gallery_QA','root_chart_visual_QA','source_phase_freeze']}
    assert all(x['completed'] and x['PASS'] for x in evidence.values())
    ev, gallery = evidence['saved_evaluation_complete'], evidence['fixed_gallery_complete']
    rootqa, otherqa = evidence['root6_actual_view'], evidence['other6_actual_view_and_saved_gallery_QA']
    seal_b = ev['prediction_seal']
    seal = read(seal_b)
    fit_b = evidence['source_phase_freeze']['source_fit_complete']
    fit = read(fit_b)
    assert seal['source_fit_complete'] == fit_b
    assert fit['all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs'] is True
    assert seal['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads'] is True
    assert seal['original_rows2144'] == 2144 and seal['model_domain_rows4288'] == 4288
    assert fit['parameters_each'] == 116407 and fit['steps_each'] == 8000 and fit['total_updates'] == 16000
    assert sha(COMPARISON) == COMPARISON_SHA
    comparison = json.loads(COMPARISON.read_text())
    assert comparison['instructions_in_document_are_advice_not_new_user_request']
    comparison_md = safe(comparison['comparison_memo']['path'])
    assert sha(comparison_md) == comparison['comparison_memo']['sha256']
    comparison_text = comparison_md.read_text()
    assert '공개 실사' in comparison_text and '엄격 synthetic-only' in comparison_text
    checks = []
    def check(name, condition):
        assert condition, name
        checks.append(name)
    check('numeric all10 unchanged vs independent actual audit', numeric['primary'] == evidence['independent_numeric_audit']['current']['primary']['ALL'])
    check('original U45/1784 distinct primary', numeric['original_input_U'] == {'N':1784,'U':45,'rate':45/1784})
    check('same68 original crop/valid parity from bound gallery QA', gallery['selected_unique_cases'] == 68 and gallery['all_selected_original_RGB96_valid96_png_parity'] and otherqa['checks']['all68_clean_and_valid_PNG_decoded_pixel_parity_vs_V195'])
    rootids, otherids = rootqa['original_indices'], otherqa['actual_viewed_ids']
    check('root6 A6 disjoint union12; not68 toolview', len(rootids) == len(otherids) == 6 and len(set(rootids+otherids)) == 12 and not rootqa['all68_or1784_toolview_claim'] and not otherqa['all68_cards_tool_view_claim'])
    check('A renderer authorship disclosed', otherqa['own_renderer_authorship_disclosed'] and not otherqa['independent_of_own_renderer_code'])
    cases = {x['original_global_index']:x for x in rootqa['cases']}
    image_bindings = [closed['output_files26']['PRIMARY_ALL10_COUNTS.png']]
    for k in [840,996]:
        b = cases[k]['card'];check(f'root example case{k} stored image hash', sha(b['path']) == b['sha256']);image_bindings.append(b)
    check('source all2 final state barrier and both frozen prediction closure', fit['all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs'] and seal['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads'])

    artifact_pairs = [
        ('수치표 전체 JSON', NUM/'RESULTS_KO.json'), ('수치표·분모 설명', NUM/'REPORT_KO.md'), ('수치 HTML', NUM/'index.html'),
        ('10모델 주평가 CSV', NUM/'PRIMARY_ALL10.csv'), ('same-slot 별도 CSV', NUM/'SAME_SLOT_ALL10.csv'),
        ('material20행 CSV', NUM/'PRIMARY_MATERIAL_ALL10.csv'), ('native8×10×3 CSV', NUM/'SAVED_PRIMARY_NATIVE8_D_E_WITH_U.csv'),
        ('SELF 두 cohort CSV', NUM/'SELF_TWO_COHORTS_ALL10.csv'), ('reference461/manual376 CSV', NUM/'REFERENCE461_AND_MANUAL376_ALL10.csv'),
        ('public weak9 CSV', NUM/'PUBLIC_WEAK_BINARY9.csv'), ('public owner3 CSV', NUM/'PUBLIC_WEAK_OWNER3_8.csv'),
        ('고정68 portable gallery', gallery['gallery']['path']), ('고정68 HTML', gallery['HTML']['path']), ('gallery 봉인', scope['fixed_gallery_complete']['path']),
        ('root6 실제 card 관찰', scope['root6_actual_view']['path']), ('A6 실제 card·68 PNG parity/table QA', scope['other6_actual_view_and_saved_gallery_QA']['path']),
        ('root count 그림 실제 관찰', scope['root_chart_visual_QA']['path']), ('C numeric CLOSED', CLOSED),
        ('V199 source 보고서', Path(fit_b['path']).parents[2]/'PORTABLE_REPORT_KO.md'),
        ('V199 source fit COMPLETE', fit_b['path']), ('V199 source 최종8000 동결 장벽', fit['all2_freeze']['path']),
        ('V199 source trainer 코드', fit['code']['path']), ('V199 source trainer protocol', fit['protocol']['path']),
        ('V199 source 감사', seal['source_fit_integrity_review']['path']),
        ('V200 두 팔 inference 코드', seal['code']['path']), ('V200 inference protocol', seal['protocol']['path']),
        ('V200 prediction seal', seal_b['path']), ('V200 저장 evaluator 코드', ev['code']['path']), ('V200 저장 evaluator protocol', ev['protocol']['path']),
        ('V200 저장 평가 COMPLETE', scope['saved_evaluation_complete']['path']), ('V200 독립 실제 수치감사', scope['independent_numeric_audit']['path']),
        ('첨부 조언 원문·현재 권한 비교 영수증', COMPARISON), ('현재 권한·recipe 비교 메모', comparison_md), ('받은 조언 원문 보존본', comparison['received_copy']['path'])]
    links = [{'label':label,'relative':rel(path)} for label,path in artifact_pairs]
    for x in links:
        check('artifact exists '+x['relative'], safe(V/x['relative']).is_file())
    intro = '''# V200 동결 native 조건 head 실사 비교

이번 시도에서 native 번호 조건을 추가한 C59는 실사 개선을 보이지 않았다. 고정 주평가1237(D1147/E90)에서 C58 zero8 control은 D650·E29, C59 native onehot은 D603·E39를 맞혔다. 이전 C57은 D729·E52로 새 두 팔보다 두 클래스 정답 수가 모두 많다. C55 대비 C59는 E39대37로2개 많지만 D603대831로228개 적으므로 종합 개선이라고 부를 수 없다. 어떤 모델도 선택·채택하지 않았고 전체 목표는 계속 미해결이다.

이번에 완료한 것은 source의 두 최종8000 상태를 동결한 뒤 같은 원초기 q/L/native·저장 crop에서 두 팔 모두 평가하고, 기존8모델과 동일 분모의 수치·고정68 gallery를 연결한 절차다. 새2모델×(current1784+public360)=4288행을 봉인한 후 별도 평가했다. current의 기존 인간 semantic D/E/S 기준은 유효한 authority로 유지한다. 반복 노출된 cohort이므로 fresh/blind 또는 자연실사 일반화 증명으로 해석하지 않는다.

새 두 팔은 모두 RAW다. C58은 zero8 control, C59는 native onehot이며 각각116407 stored parameters·8000steps다. 같은 parameter 수가 같은 유효 함수공간을 뜻하지 않고 shared backbone도 바뀔 수 있다. 기존5loss와 별도 direct objective, 원좌표·query crop·고정 decoder를 유지한 결과이며 이 수치로 threshold·모델·입력·architecture를 고르지 않았다.

'''
    numeric_md_b = closed['output_files26']['REPORT_KO.md']
    assert sha(numeric_md_b['path']) == numeric_md_b['sha256']
    numeric_md = Path(numeric_md_b['path']).read_text()
    numeric_rebased = re.sub(r'\]\(([^)]+)\)', lambda m: ']('+rebase(m.group(1))+')', numeric_md)
    numeric_rebased = numeric_rebased.replace('# V200 저장 수치 readout', '## 저장 수치 readout', 1)
    case_rows = [[f'CASE_{k:04d}', cases[k]['observations']] for k in [840,996]]
    visual = '\n## 같은 고정68 gallery와 실제 관찰의 범위\n\n'
    visual += '이전 고정68개를 동일 순서로 재사용했다. 새 점수로 좋은/나쁜 사례를 다시 고르지 않았다.68 clean96/valid96 PNG의 기존 V195 pixel/encoded byte parity와 표·링크는 A의 봉인 QA에 근거한다. 이는68개의 semantic 의미를 도구로 모두 판독했다는 뜻이 아니다. root가 [0,2,256,300,840,996] 여섯 card, A가 [1,3,4,5,7,8] 여섯 card를 실제로 보아 union12개다. A는 해당 renderer 저자임을 밝혔으므로 자기 코드와 독립된 QA라고 주장하지 않는다. 실제6개 A 관찰은 초반 WOOD humanD에 치우쳐 전체 재질·실패율을 대표하지 않는다. C는 count chart를 실제 확인했고 원 crop/card를 새로 열지 않았다. HTML browser 렌더 검증은 하지 않았다.\n\n'
    visual += mdtable(['root 실제 관찰 사례', '기존 human/reference 및 저장 응답에 근거한 관찰'],case_rows)+'\n'
    for k in [840,996]:
        visual += f"![CASE_{k:04d}: root 실제 card 관찰을 인용한 고정 예시]({rel(cases[k]['card']['path'])})\n\n"
    visual += 'CASE0840은 humanD·manual reference2.7108px인데 C58/C59가 E, CASE0996은 humanE·manual reference1.13822px인데 두 팔이 D다. 근접 reference 사례에서도 이 오류가 남는다. 인접 재료·외부물체 단서와 의도한 native 가시성을 구분하는 문제가 남아 있다는 기술적 관찰이며, local crop만으로 selected instance·물리 native 정체·차폐 원인·실패 원인 빈도를 인증하지 않는다. 위 두 예시는 root의 실제 관찰 영수증을 인용했으며 보고서 작성자가 새 semantic label을 만들지 않았다.\n\n'
    advice = '''## 기존 조언 문서와 현재 허용 범위

V194에 보존한 첨부 V193 검토 문서는 조언이며 새 실행·중단 지시로 가져오지 않았다. 해당 문서의 엄격 synthetic-only 전제와 현재 사용자의 공개실사 학습 허용을 구분한다. 이번 source recipe는 합성+공개실사 weak end를 사용했고 current223/publicheld45·기존 평가 cohort를 학습·보정·threshold·모델 선택에 쓰지 않았다. 문서가 지적한 native와 finite-end의 불일치 가능성, target 식별 가능성, U0의 해석 문제는 미해결 한계로 남긴다. prior한계와 이번 오류를 연결해 단일 인과를 확정하지 않는다.

## 현재까지 닫힌 것과 남은 목표

원 q/L/native와 기존 초기 crop을 유지한 source 학습·동결·같은 분모 실사 비교·수치 검산·고정 시각 자료 연결을 완료했다. 번호 조건 추가가 이번 source HOLDOUT 및 실사에서 개선을 주었다는 근거는 없다. source direct 감독은 weak finite-end 권위이며 physical native gold로 승격하지 않았다. native6·7 direct known0, 공개실사 end HOLDOUT known0, 반복 노출 조건과 source TRAIN apparent fit의 한계가 남는다. 단순 주변 재료/물체 존재와 의도한 끝부위의 외부가림을 안정적으로 구분하는 전체 사용자 목표는 미해결이다. 후속 원인·자료·모델 변경은 이 보고서가 선택하거나 실행하지 않는다.

## 코드·실행·감사·표·시각화 계보

'''
    doc = intro + numeric_rebased + visual + advice + ''.join(f"- [{x['label']}]({x['relative']})\n" for x in links)
    with (V/'ROOT_REPORT_KO.md').open('x') as f: f.write(doc)
    with (V/'PORTABLE_REPORT_KO.md').open('x') as f: f.write(doc)

    numeric_html_b = closed['output_files26']['index.html']
    assert sha(numeric_html_b['path']) == numeric_html_b['sha256']
    page = Path(numeric_html_b['path']).read_text()
    page = re.sub(r'((?:href|src)=")([^"]+)(")', lambda m:m.group(1)+rebase(m.group(2))+m.group(3), page)
    page = page.replace('<title>V200 저장 수치 readout</title>','<title>V200 동결 native 조건 head 실사 비교</title>')
    lead = '<h1>V200 동결 native 조건 head 실사 비교</h1><p>'+html.escape(intro.split('\n\n')[1])+'</p><p>'+html.escape(intro.split('\n\n')[2])+'</p><p>'+html.escape(intro.split('\n\n')[3])+'</p><h2>저장 수치 readout</h2>'
    page = page.replace('<h1>V200 저장 수치 readout</h1>',lead,1)
    appendix = '<h2>고정68 gallery와 실제 관찰 범위</h2><p>'+html.escape(visual.split('\n\n')[1])+'</p>'
    for k in [840,996]:
        appendix += '<figure><a href="'+rel(cases[k]['card']['path'])+'"><img style="width:100%;max-width:820px;height:auto" src="'+rel(cases[k]['card']['path'])+'" alt="root 실제 관찰 고정 사례 CASE_'+str(k).zfill(4)+'"></a><figcaption>'+html.escape(cases[k]['observations'])+'</figcaption></figure>'
    appendix += '<p>'+html.escape(visual.split('\n\n')[-2])+'</p>'
    appendix += '<h2>기존 조언·source 권한·미해결 목표</h2><p>'+html.escape(advice.split('\n\n')[1])+'</p><p>'+html.escape(advice.split('\n\n')[3])+'</p><h2>코드·감사·그림 계보</h2><ul>'+''.join('<li><a href="'+x['relative']+'">'+html.escape(x['label'])+'</a></li>' for x in links)+'</ul>'
    page = page.replace('</body></html>',appendix+'</body></html>')
    with (V/'ROOT_READOUT_KO.html').open('x') as f:f.write(page)

    inputs = {k:scope[k] for k in ['saved_evaluation_complete','independent_numeric_audit','fixed_gallery_complete','root6_actual_view','other6_actual_view_and_saved_gallery_QA','root_chart_visual_QA','source_phase_freeze']}
    inputs.update(numeric_CLOSED=bind(CLOSED), numeric_data=numerical_b, source_fit_complete=fit_b, prediction_seal=seal_b, prior_advice_receipt=bind(COMPARISON), prior_advice_comparison=comparison['comparison_memo'])
    result = dict(completed=True,PASS=True,phase=V.name,result_kind='FROZEN_REAL_EVALUATION_WITH_SAVED_NUMERICAL_AND_FIXED_GALLERY_EVIDENCE',created_UTC=datetime.now(timezone.utc).isoformat(),inputs=inputs,
                  all10_models=numeric['models'], primary1237=numeric['primary'], same_slot1248=numeric['same_slot'], primary_material20=numeric['material'], native240=numeric['native240'],
                  SELF_all136_paired21=numeric['self_two_cohorts'], existing_reference461=numeric['existing_reference461'], manual376=numeric['manual376'], publicweak9=numeric['public_weak9'], publicweak_owner8=numeric['public_weak_owner8'], public_coverage=numeric['public_coverage'],public_legacy_missing=numeric['public_legacy_missing'],
                  original_all1784_inputU45=numeric['original_input_U'],source_TRAIN_and_HOLDOUT=numeric['source_train_holdout'], source_HOLDOUT_bases=numeric['source_HOLDOUT_bases'], source_real_end_HOLDOUT_known=0, source_native6_7_known=0,
                  new_frozen_models={'C58':{'RAW':True,'native_condition':'zero8 control'},'C59':{'RAW':True,'native_condition':'onehot'}}, source_steps_each8000=fit['steps_each'],stored_parameters_each116407=fit['parameters_each'],equal_effective_capacity_claim=False,
                  new_model_domain_predictions4288=4288,new_forward_during_reporting=0,
                  gallery={'selected68':68,'same_previous_fixed68':True,'selected_inputU11':otherqa['selected_U_count'],'all68_clean_valid_parity_authority':scope['other6_actual_view_and_saved_gallery_QA'],'root_toolview_ids':rootids,'A_toolview_ids':otherids,'actual_card_toolview_union12':sorted(set(rootids+otherids)),'all68_or1784_semantic_toolview_claim':False,'renderer_authorship_disclosed':True,'C_crop_card_toolviews':0,'chart_actual_QA':scope['root_chart_visual_QA'],'root_case_examples':{str(k):cases[k] for k in [840,996]}},
                  conclusion='native onehot C59와 zero8 C58 모두 이번 고정 실사 비교에서 이전 C57보다 D/E 정답 수가 적다. 이번 시도 개선 미확인, 원인 미확정, 미채택, 전체 목표 미해결.',
                  advice_is_not_new_instruction=True,strict_synthetic_only_claim=False,publicreal_source_allowed=True,human_semantic_authority_preserved=True,repeated_development_exposure_not_freshblind=True,
                  no_current_or_publicheld_for_training_or_selection=True,original_qL_native_inputs_and_fixed_decoders_preserved=True,
                  no_new_ML_crop_SAM_PnP_threshold_or_data_architecture_selection=True,limits=numeric['limits'],links=links,image_bindings=image_bindings,selected_model=None,adopted=False,fully_solved=False,goal_active=True,final_report_QA_pending_root=True,phase_result_or_freeze_written=False)
    # Report prose is Korean; the numerical tables are copied from the closed readout.
    dump(V/'ROOT_RESULTS_KO.json',result)
    check('root MD and portable bytes identical', (V/'ROOT_REPORT_KO.md').read_bytes() == (V/'PORTABLE_REPORT_KO.md').read_bytes())
    check('all10 numerical primary preserved', result['primary1237'] == numeric['primary'])
    for m in numeric['models']:
        check(f'{m} literal CM retained', str(numeric['primary'][m]['CM_D_E_by_D_E_U'][0]).replace(' ','') in doc.replace(' ',''))
    references = []
    for fn in ['ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html']:
        text = (V/fn).read_text()
        refs = re.findall(r'\]\(([^)]+)\)',text) if fn.endswith('.md') else re.findall(r'(?:href|src)="([^"]+)"',text)
        for ref in refs:
            check(f'relative existing link {fn} {ref}', not ref.startswith(('/', 'http:','https:','file:')) and safe(V/ref).is_file())
            target = safe(V/ref)
            if target.suffix in ['.json','.md','.html','.csv','.py','.jsonl']:
                check(f'publication direct text <=2MiB {fn} {ref}', target.stat().st_size <= 2*1024*1024)
            references.append({'page':fn,'reference':ref})
    outputs = {x.name:bind(x) for x in expected}
    dump(PREP/'ROOT_REPORT_BUILD_COMPLETE.json',dict(completed=True,PASS=True,code=bind(Path(__file__)),scope=bind(SCOPE),inputs=inputs,outputs=outputs,checks=len(checks),check_names=checks,relative_link_occurrences=len(references),links=references,image_bindings_without_decode=image_bindings,source_arrays_weights_models_forward_fit_PnP_SAM_or_crops_read=0,own_actual_source_card_toolviews=0,ROOT_PHASE_RESULT_or_freeze_writes=0,root_final_QA_pending=True))
    print(json.dumps({'completed':True,'PASS':True,'checks':len(checks),'relative_link_occurrences':len(references),'outputs':outputs,'build_complete':bind(PREP/'ROOT_REPORT_BUILD_COMPLETE.json')},ensure_ascii=False))


if __name__ == '__main__':
    run()
