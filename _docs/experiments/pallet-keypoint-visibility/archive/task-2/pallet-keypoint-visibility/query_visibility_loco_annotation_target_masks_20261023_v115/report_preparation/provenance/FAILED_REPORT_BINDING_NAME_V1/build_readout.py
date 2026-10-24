"""Korean V115 readout from saved metadata; no source/model/GT array decoding."""
from pathlib import Path
import hashlib
import html
import json
from html.parser import HTMLParser

V = Path(__file__).resolve().parents[1]
P = V.parent
V114 = P / 'query_visibility_material_augmented_query_head_20261023_v114'
SOURCES = {
    'execution': V / 'ROOT_ACTUAL_SAM_CANDIDATE_EXECUTION.json',
    'summary': V / 'source_sam_candidates_v1/SUMMARY.json',
    'output_seal': V / 'source_sam_candidates_v1/SAM_CANDIDATE_OUTPUT_SEAL.json',
    'saved_audit': V / 'independent_saved_mask_audit/completed_audit_v1/REVIEW.json',
    'saved_audit_seal': V / 'independent_saved_mask_audit/completed_audit_v1/SEAL.json',
    'metadata_review': V / 'independent_metadata_review/REVIEW.json',
    'root_visual_decision': V / 'ROOT_ACTUAL_FIXED_VISUAL_REVIEW_AND_DECISION.json',
    'prior_public_feasibility': P / 'query_visibility_ycb_wood_foam_source_20261023_v113/public_pallet_source_feasibility_next/SOURCE_FEASIBILITY_NEXT.json',
    'public_C12_diagnosis': V114 / 'public_source_regression_inventory/COMPACT_RESULT.json',
    'wood_TRAIN_feasibility': V114 / 'next_wood_foreground_source_feasibility/RESULT.json',
    'latest_current_C12_report': V114 / 'ROOT_READOUT_KO.html',
}


def bind(p):
    b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def load(p):
    return json.loads(p.read_text())


def verify(b):
    p = Path(b['path']); actual = bind(p)
    assert actual['sha256'] == b['sha256'], str(p)
    if 'bytes' in b:
        assert actual['bytes'] == b['bytes']


def link(p, label):
    return '<a href="%s">%s</a>' % (html.escape(str(p), quote=True), html.escape(label))


def figure(b, caption):
    p = b['path']
    return '<figure><a href="%s"><img src="%s" alt="%s" loading="lazy"></a><figcaption>%s</figcaption></figure>' % (
        html.escape(p, quote=True), html.escape(p, quote=True), html.escape(caption, quote=True), html.escape(caption))


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__(); self.paths = []
    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if k in ('href', 'src') and v and not v.startswith('#'):
                assert v.startswith('/'), 'Only explicit local evidence links'
                assert Path(v).is_file(), v
                self.paths.append(v)


def main():
    output_json = V / 'ROOT_RESULTS_KO.json'; output_html = V / 'ROOT_READOUT_KO.html'
    assert not output_json.exists() and not output_html.exists(), 'No report overwrite'
    s = {k: bind(p) for k, p in SOURCES.items()}
    execution = load(SOURCES['execution']); summary = load(SOURCES['summary'])
    audit = load(SOURCES['saved_audit']); meta = load(SOURCES['metadata_review'])
    visual = load(SOURCES['root_visual_decision']); diag = load(SOURCES['public_C12_diagnosis'])
    wood = load(SOURCES['wood_TRAIN_feasibility'])
    for k in ('entry', 'policy', 'independent_metadata_review', 'output_seal', 'summary'):
        verify(execution[k])
    for k in ('render_entry', 'render_receipt', 'render_seal', 'saved_numerical_audit'):
        verify(visual[k])
    for b in visual['viewed_figures']:
        verify(b)  # Hash saved display files; no RGB/image decoding.
    for b in diag['bindings'].values():
        verify(b)  # Public saved diagnostic only, no private/GT arrays.
    assert execution['terminal_exit_code'] == 0 and summary['PASS'] and audit['PASS'] and meta['PASS']
    assert execution['actual_counts']['primary_masks'] == summary['primary_masks'] == audit['counts']['independently_read_bool_masks'] == 33
    assert summary['set_image_calls'] == 20 and summary['total_predict_calls'] == 34
    assert summary['original_factual_counts'] == audit['factual_input_counts']
    assert audit['counts']['factual_U'] == 38 and summary['original_query_slots'] == 160
    assert visual['all20_overview_frames_actually_viewed'] and visual['all_first6_detailed_cards_actually_viewed']
    assert len(visual['viewed_figures']) == 8
    assert audit['eligible122_per_role_mask_readouts']['TARGET'] == dict(MASK_FALSE=119, MASK_TRUE=3)
    assert audit['frame_pairs_with_positive_mask_overlap'] == 9
    assert visual['descriptive_counts_not_accuracy']['both_true_at_query'] == 0
    regressions = diag['all5_VAL_new_E_regressions']
    assert len(regressions) == 5 and all(x['donor_object_id'] == 16 for x in regressions)
    assert wood['old_public_TRAIN_controlled_object16_E_role']['absent'] is True
    assert wood['new_candidate_inventory']['instances'] == 200
    images = visual['viewed_figures']; details = images[2:]
    counts = dict(frames=20, target_masks=20, competitor_masks=13, primary_masks=33,
        request_slots=40, missing_competitor_no_calls=7, repeat_calls=1, total_predict_calls=34,
        set_image_calls=20, original_queries=160, eligible_queries=122, factual_U=38,
        noquery_U=16, OOB_or_invalid_U=22, target_mask_true=3, target_mask_false=119,
        competitor_mask_true=1, competitor_mask_false=73, competitor_missing_at_query=48,
        both_masks_true_at_query=0, positive_full_mask_overlap_pairs=9, present_competitor_pairs=13)
    results = {
        'schema': 'V115_PUBLIC_SAM_PILOT_ROOT_READOUT_KO_V1', 'completed': True,
        'headline_KO': '33개 마스크를 실행·검산했지만, 원래160개 query의 자동 D/E/S teacher로 채택하지 않았습니다.',
        'decision_KO': '초기 추론 대상과 대표 annotation의 대응이 미확인이고, 고정 사례에서 서로 다른 팔레트를 가리킵니다. 분할의 경계·구멍·소유권도 불확실하므로 candidate mask로만 보존합니다.',
        'counts': counts, 'execution_seconds': summary['seconds'], 'GPU_peak_bytes': summary['GPU_peak_bytes'],
        'repeat_mask_exact': audit['repeat']['mask_arrays_bitexact'],
        'repeat_quality_abs_difference': audit['repeat']['quality_abs_difference'],
        'producer_model_state_unchanged': audit['producer_bound_certificates']['state_unchanged'],
        'verification_KO': '독립 감사는 저장마스크와 모든메타데이터·query readout을 재계산했습니다. RGB검증·호출수·모델상태불변은 producer에 결합된 실행기록이며, 감사에서 RGB나모델을 다시 실행하지 않았습니다.',
        'not_accuracy_KO': '유효122개 중 target 마스크값 true3/false119는 대응 미확인 annotation mask를 원래q에서 읽은 기술통계입니다. 3/122를 정확도로 해석하지 않습니다.',
        'semantic_state_KO': '원래160개 semantic UNKNOWN,16noquery+22OOB factualU 보존. Mask부재를 E/S로 바꾸지 않으며 q/L/slot/fold를 변경하지 않았습니다.',
        'overlap_KO': '13개 target/competitor 쌍 중9개 전체마스크가 겹칩니다. 중복annotation/분할혼합의 가능성을 보여줄 뿐, 실제앞뒤소유권이나 단일원인을 증명하지 않습니다.',
        'visual_review': {'root_actually_viewed_all20_overviews_and_fixed_first6': True,
            'figures': images, 'first6_observations_KO': visual['per_frame_first6_KO'],
            'overview_observations_KO': visual['overview_observations_KO']},
        'separate_PUBLIC_C12_source_diagnosis': {
            'scope_KO': '이전 PUBLIC 저장 source 진단이며 V115 학습·private 평가 결과가 아닙니다.',
            'old_public_VAL_C11': diag['old_public_VALIDATION']['C11'],
            'old_public_VAL_C12': diag['old_public_VALIDATION']['C12'],
            'cover_TRAIN_C11_C12_correct': [75, 75, 75],
            'cover_VAL_C11_C12_correct': [116, 111, 119],
            'all5_new_regressions_donor_object_id': 16,
            'regression_bases': diag['clusters_and_limits']['regressions_query_bases'],
            'regression_images': diag['clusters_and_limits']['regressions_images'],
            'old_TRAIN_object16_foreground_donors': 0,
            'source_cluster_KO': 'Accepted controlled pairs는 PALLOX 한 물리 bin에서 나왔고 TRAIN75개는10query/6images, VAL119개는18query/9images에 군집됩니다. 개별rank를 독립실험으로 세지 않습니다.',
            'relative_vs_absolute_KO': '새5회귀의 cover-sham 상대차이는 모두2이상이지만 cover절대logit은 D였습니다. 상대차이만으로 외부가림 판정이 보장되지 않습니다.',
            'causality_KO': 'Object16 집중과 TRAINforeground 부재는 관측된 coverage차이입니다. 재질·형상·배치·반복장면·데이터·선택된가중치가 함께 달라져 원인을 단독확정하지 않습니다.',
            'wood_TRAIN_nextstep': {'status': 'ACTIVE_METADATA_CODE_FEASIBILITY_ONLY', 'instances': 200,
                'donors_selected': 0, 'new_cutout_pixels_or_opaque_feasibility_tested': False,
                'new_teacher_or_fit_executed': False,
                'meaning_KO': 'V113공개TRAIN woodblock16의200개 후보가 기존object16 VAL donor images와 분리됨을 메타데이터로 확인했습니다. 원래TRAIN 이미지의 다른역할노출은 있으며 freshholdout이 아닙니다.'}},
        'latest_actual_current_report': s['latest_current_C12_report'],
        'latest_current_scope_KO': '최신 C12 current 보고서는 별도 기존결과 링크입니다. V115에서는 새private 평가·학습·current GT/배열열람을 하지 않았습니다.',
        'bindings': s, 'training_admission': False, 'semantic_D_E_S_labels_created': 0,
        'native_mapping': 'UNKNOWN', 'human_gold': False, 'new_private_evaluation': False,
        'new_model_training': False, 'coordinate_changes': 0, 'PnP': 0,
        'goal_status': 'active', 'fully_solved': False,
        'report_render_QA_by_root': 'PENDING',
        'report_builder': bind(Path(__file__)),
    }
    with output_json.open('x') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')
    overview_html = ''.join(figure(b, '원래 순서 전체20프레임: ' + ('00–09' if i == 0 else '10–19') + '. Raw RGB와 별도의 mask/box/query 표시. 색상은 candidate annotation 분할이며 정답 소유권이 아닙니다.') for i,b in enumerate(images[:2]))
    detail_html = ''.join(figure(b, 'Frame %02d · %s' % (i, visual['per_frame_first6_KO'][str(i)])) for i,b in enumerate(details))
    evidence = [
        ('실행 기록', SOURCES['execution']), ('SAM 출력 seal', SOURCES['output_seal']),
        ('독립 저장값 감사', SOURCES['saved_audit']), ('독립 metadata 검토', SOURCES['metadata_review']),
        ('root 실제 시각 검토·결정', SOURCES['root_visual_decision']), ('전체 수치 JSON', output_json),
    ]
    evidence_html = ' · '.join(link(p, label) for label,p in evidence)
    doc = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V115 · 공개 LOCO SAM candidate mask pilot</title><style>
:root{color-scheme:light;font-family:system-ui,-apple-system,"Noto Sans KR",sans-serif;color:#182331;background:#f4f6f8}body{margin:0}main{max-width:1180px;margin:auto;padding:34px 24px 54px}h1{font-size:30px;line-height:1.4;margin:12px 0}h2{font-size:21px;margin:32px 0 12px}p{line-height:1.75;margin:10px 0}a{color:#155ea8;overflow-wrap:anywhere}.eyebrow{font-size:13px;color:#526579;letter-spacing:.06em}.lead{background:#fff;padding:24px 28px;border:1px solid #d7e1ec;border-left:6px solid #2f647e;border-radius:10px}.lede{font-size:18px}.status{display:inline-block;color:#7b390b;background:#fff1d9;border:1px solid #e3bc7c;padding:5px 10px;border-radius:5px;font-size:13px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}.metric{background:#fff;padding:15px;border:1px solid #dce3e9;border-radius:8px}.metric strong{display:block;font-size:28px;margin-bottom:5px}.muted{color:#536476;font-size:14px}.note{background:#eaf2f6;padding:14px 18px;border-radius:6px}.warning{background:#fff6e8;padding:15px 18px;border-radius:6px}table{border-collapse:collapse;width:100%;background:#fff;margin:12px 0;font-size:15px}th,td{border:1px solid #dce3e9;padding:10px 12px;text-align:left}th{background:#eaf0f4}figure{margin:0;background:#fff;border:1px solid #dce3e9;border-radius:7px;overflow:hidden}img{display:block;width:100%;height:auto}figcaption{padding:12px 14px;font-size:14px;line-height:1.65;color:#364a5b}details{margin-top:18px}summary{cursor:pointer;font-weight:650;padding:14px 16px;background:#e5edf3;border-radius:6px}details .grid{margin-top:14px}.evidence{font-size:13px;line-height:1.9;margin-top:24px}.footer{border-top:1px solid #d6dfe7;margin-top:28px;padding-top:15px;font-size:13px;color:#526579}@media(max-width:750px){.grid{grid-template-columns:1fr}.metrics{grid-template-columns:1fr 1fr}h1{font-size:25px}main{padding:22px 14px}.lead{padding:18px}}
</style><main>
<div class="eyebrow">V115 · PUBLIC SOURCE SAM PILOT · 2026-10-23</div>
<div class="lead"><span class="status">자동 D/E/S teacher 미채택 · 목표 계속 진행</span>
<h1>33개 마스크를 실행·검산했지만,<br>원래 query의 가림 정답으로 채택하지 않았습니다.</h1>
<p class="lede">초기 추론이 선택한 대상과 대표 annotation이 일부 고정 사례에서 다른 팔레트를 가리킵니다. 분할의 경계·구멍·소유권에도 불확실성이 있어, 마스크는 <b>annotation entity에 조건부인 candidate</b>로만 보존합니다.</p>
<p class="muted">V115에서 새 private 평가·모델 학습·좌표 보정은 수행하지 않았습니다. Native mapping과 human gold는 UNKNOWN입니다.</p></div>
<div class="metrics"><div class="metric"><strong>33</strong>실제 primary mask<br><span class="muted">20 target + 13 competitor</span></div><div class="metric"><strong>160</strong>원래 q/L·slot·fold 보존<br><span class="muted">122 eligible + 38 factual U</span></div><div class="metric"><strong>3 / 122</strong>target mask true<br><span class="muted">기술통계이며 정확도 아님</span></div><div class="metric"><strong>0</strong>새 D/E/S teacher 채택<br><span class="muted">학습·current 평가도 0</span></div></div>
<h2>실행과 검산에서 확인한 것</h2>
<p>고정20개 canonical RGB에 set_image를 한 번씩 수행했습니다. Box-only primary33회와 첫 target의 고정 repeat1회, 총34 predict를 실행했습니다. 누락 competitor7개는 null no-call로 남겼고 실제 빈 마스크는0개입니다. 실행시간은5.918초, repeat 마스크·quality는 동일했고 저장된 모델상태 digest도 전후 일치했습니다.</p>
<p class="muted">독립 저장값 감사는 모든40 요청·33 native bool mask·160 query의 nearest/3×3·면적·box 교차·pair IoU를 재계산했습니다. RGB 검증·호출수·모델상태 불변은 실행 producer에 결합된 기록이며, 감사에서 RGB나모델을 재실행하지 않았습니다.</p>
<div class="grid"><div><table><tr><th>원래 입력 상태</th><th>개수</th></tr><tr><td>유효 query</td><td>122</td></tr><tr><td>no-query factual U</td><td>16</td></tr><tr><td>OOB/invalid factual U</td><td>22</td></tr><tr><td>총계</td><td>160</td></tr></table></div><div><table><tr><th>유효122개에서의 mask readout</th><th>true / false / missing</th></tr><tr><td>Target</td><td>3 / 119 / 0</td></tr><tr><td>Competitor</td><td>1 / 73 / 48</td></tr><tr><td>두 mask 모두 true</td><td>0</td></tr><tr><td>전체 mask가 겹치는 쌍</td><td>9 / 13</td></tr></table></div></div>
<div class="warning"><b>3/122는 정확도가 아닙니다.</b> 초기 추론 대상↔annotation 대응이 미확인인 상태의 mask 샘플입니다. 9/13 overlap도 중복 annotation·분할 혼합 가능성의 진단이며 실제 앞뒤 소유권이나 단일 원인을 증명하지 않습니다. Mask absence는 background·query 오차·own hole·self·foreign cover·분할실패를 구분하지 못합니다. 원래160개 semantic UNKNOWN과38개 factual U를 유지합니다.</div>
<h2>실제 검토한 전체20프레임</h2><p>Root는 아래 overview2장과 사전고정 첫6개 상세카드를 모두 실제로 검토했습니다. 예시를 교체하지 않았습니다. 노란 초기 추론 box/q와 annotation target의 위치 차이를 함께 확인할 수 있습니다.</p>
<div class="grid">''' + overview_html + '''</div>
<details><summary>사전고정 Frame00–05 상세카드와 관찰6개 펼치기</summary><div class="grid">''' + detail_html + '''</div></details>
<p class="muted">같은 ann_id에 명시적으로 연결된 기존 약한 visible cue10개는5프레임에만 있습니다. 정확한 center pixel은 UNKNOWN이므로 positive point prompt로 승격하지 않았습니다. 전체20의 물리 ID를 새로 인증한 결과도 아닙니다.</p>
<h2>별도 PUBLIC source 진단: 다음 데이터 작업의 근거</h2>
<p>이전 C11→C12의 저장 PUBLIC source 진단에서 새 external E 회귀5개는 모두 donor object16(woodblock)이었습니다. 기존 TRAIN controlled foreground에는 object16 donor가0개였고, VAL에는35개 cover가 있었습니다. C12 material source에서 object16을 <em>가려지는 target</em>으로 본 것과, 팔레트 cue를 가리는 <em>foreign foreground</em>로 본 것은 다른 역할입니다.</p>
<table><tr><th>PUBLIC source 고정 집단</th><th>C11</th><th>C12</th></tr><tr><td>TRAIN accepted COVER E 정답</td><td>75 / 75</td><td>75 / 75</td></tr><tr><td>VAL accepted COVER E 정답</td><td>116 / 119</td><td>111 / 119</td></tr><tr><td>VAL old-public D 정답</td><td>147 / 152</td><td>147 / 152</td></tr><tr><td>VAL old-public E 정답</td><td>118 / 121</td><td>113 / 121</td></tr></table>
<p>회귀5개는5 query base·3 image에 군집됩니다. Accepted pair는 PALLOX 한 물리 bin에서 나왔고, VAL119개도18 query·9 image입니다. 새5회귀 모두 cover−sham 상대차이가2이상이지만 cover 절대logit은 D였습니다. 재질·형상·배치·반복장면·학습·선택이 함께 달라 <b>object16 하나를 원인으로 확정하지 않습니다.</b></p>
<p class="note"><b>다음 source 작업은 준비 단계입니다.</b> 공개 V113 TRAIN woodblock16의200개 후보가 기존 object16 VAL donor 이미지와 분리됨을 메타데이터로 확인했습니다. 실제 cutout 선택·opaque/placement 검사·새 teacher 생성·fit은 아직 없습니다. 기존 TRAIN의 다른 source 역할 노출이 있어 fresh holdout으로 부르지 않습니다.</p>
<p>''' + link(SOURCES['public_C12_diagnosis'], 'PUBLIC source 진단 원수치') + ' · ' + link(SOURCES['wood_TRAIN_feasibility'], 'TRAIN wood foreground feasibility') + '''</p>
<h2>최신 current 결과는 별도 보고서</h2><p>''' + link(SOURCES['latest_current_C12_report'], '기존 C12 current 결과 보고서 열기') + ''' — V115의 SAM pilot 수치와 별도입니다. 이번 단계에서 current GT·배열을 다시 읽거나 새 private 평가를 실행하지 않았습니다.</p>
<div class="evidence">''' + evidence_html + '''</div>
<div class="footer">목표 상태: active · fully_solved=false · source candidate mask만 보존 · training admission=false · 보고서의 root render QA와 최종 phase seal은 별도 후속 단계입니다.</div>
</main></html>'''
    with output_html.open('x') as f:
        f.write(doc)
    parser = LocalLinks(); parser.feed(doc)
    validation = dict(completed=True, PASS=True, html=binding(output_html), results=binding(output_json),
        verified_local_link_references=len(parser.paths), unique_local_link_targets=len(set(parser.paths)),
        saved_display_image_bindings_verified=8, all_linked_files_exist=True,
        source_bindings=s, image_decodes=0, source_or_current_GT_array_reads=0, model_calls=0,
        root_render_QA_complete=False, phase_seal_written=False, fully_solved=False)
    vp = V / 'report_preparation/REPORT_LINK_AND_BINDING_VALIDATION.json'
    with vp.open('x') as f:
        json.dump(validation, f, indent=2, ensure_ascii=False); f.write('\n')
    print(json.dumps(dict(html=binding(output_html), results=binding(output_json), validation=binding(vp)), indent=2))


if __name__ == '__main__':
    main()
