"""Source/link/count checks; visual findings follow two actual view_image calls."""
from pathlib import Path
from html.parser import HTMLParser
from collections import Counter
from datetime import datetime, timezone
import hashlib, json, re, shutil, subprocess

O = Path(__file__).resolve().parent
def bind(p):
    p = Path(p)
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p): return json.loads(Path(p).read_text())
g = read(O/'GENERATION_COMPLETE.json')
d = read(O/'READOUT_DATA.json')
h = (O/'index.html').read_text()
checks = []
def ck(name, condition):
    assert condition, name
    checks.append(dict(check=name, PASS=True))
ck('all390_unique_preserved', len(d['cases']) == len({r['case_id'] for r in d['cases']}) == 390)
ck('350_single_and40_cross_separate', Counter(r['support'] for r in d['cases']) == {'SINGLE350_WEAK':350,'CROSS40_FOUR_WAY':40})
ck('native8_denominators', d['native_denominators'] == [51,62,56,29,80,87,2,23])
ck('primary_part_margins', [sum(r) for r in d['observation_cross_table']] == [299,45,1,35,10])
ck('primary_q_margins', [sum(r[k] for r in d['observation_cross_table']) for k in range(6)] == [24,28,17,80,197,44])
ck('visible_x_BG169_relation_only', d['observation_cross_table'][0][4] == 169 and '좌표 오류 169개를 뜻하지 않습니다' in h)
ck('cross_axis_disagreements_preserved', [len(x) for x in d['cross_disagreement_cases'].values()] == [0,15,8,2])
ck('fixed_four_explanation_cases_present', all(x in {r['case_id'] for r in d['cases']} for x in ['Q1608','Q0094','Q1213','Q1026']))
ck('existing390_card_hash_bindings', all((O/r['card']).resolve() == Path(r['card_binding']['path']).resolve()
    and bind(O/r['card'])['sha256'] == r['card_binding']['sha256'] for r in d['cases']))
class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links = []
    def handle_starttag(self, tag, attrs):
        self.links += [val for key,val in attrs if key in ['href','src'] and val and not val.startswith('#')]
p = Links(); p.feed(h)
ck('all_static_HTML_links_exist', all((O/x).is_file() for x in p.links))
script = re.search(r'<script>([\s\S]*)</script>', h).group(1)
node = shutil.which('node'); assert node
pr = subprocess.run([node, '--check'], input=script, text=True, capture_output=True)
ck('inline_JS_syntax_pass', pr.returncode == 0)
ck('historical_completed_exact_SHA', d['historical_diagnosis']['complete']['sha256'] == '2f73c7dad115527dedad7566696ac58cfe4d25970577a573ceb2090acd78f48a')
ck('historical_TARGET53_primary40_12_1_mixed39_12_2', d['historical_diagnosis']['target53_decomposition'] == {
    'primary390': {'OTHER_SELECTED_PALLET_PART':12,'UNKNOWN_OR_DISAGREEMENT':1,'VISIBLE_PART_AND_Q_ON_OR_NEAR':40},
    'mixed350_single_plus_cross40_unanimity': {'OTHER_SELECTED_PALLET_PART':12,'UNKNOWN_OR_DISAGREEMENT':2,'VISIBLE_PART_AND_Q_ON_OR_NEAR':39}})
ck('historical_roles_weak_provenance_and_no_full_failure_explanation', '공식 사람별 keypoint 가시성 gold가 아닙니다' in h
    and '실제 전이·일반화 실패가 모두 설명되었다고 결론내리지 않습니다' in h)
ck('no_new_native_quality_gold_or_training_admission', 'UNKNOWN(-1)' in h and not g['training_admitted'] and not g['original_labels_modified'])
ck('HTML_render_claim_absent', g['HTML_browser_rendered'] is False and '브라우저 렌더링을 확인했다고 주장하지 않습니다' in h)
ck('all_generation_output_hashes_match', all(bind(x['path']) == x for x in g['outputs'].values()))
findings = [
    ('PART_VISIBILITY_QUERY_RELATION_OBSERVATION_COUNTS.png', 'All 30 cells, primary marginals, Korean axes, weak-count disclaimer and 350+40 composition are legible; titled 관찰 교차표.'),
    ('NATIVE8_PART_VISIBILITY_OBSERVATION_COUNTS.png', 'Native8 labels, actual denominators, legend and Euro WOOD/no SQUARE/native6 N2 limits are legible. Tiny count1/2 segments retain exact values in bound data/CSV.')]
qa = dict(completed=True, PASS=True, UTC=datetime.now(timezone.utc).isoformat(), source_only=True,
    generation=bind(O/'GENERATION_COMPLETE.json'), QA_code=bind(__file__),
    inputs={k:g[k] for k in ['source_join','all390','cross40_input','evidence','historical_diagnosis']},
    code=bind(O/'build_readout.py'), HTML_template=bind(O/'gallery_template.html'), outputs=g['outputs'], checks=checks,
    actual_new_plot_visual_review=[dict(**bind(O/name), actually_viewed=True, tool='tools.view_image', readable=True, findings=note) for name,note in findings],
    actual_plot_QA_pending=False, HTML_browser_rendered=False,
    HTML_validation='Source, inline JavaScript syntax and local link/hash checks only; no browser rendering claim.',
    plot_semantics='Assistant weak observation counts; not accuracy/confusion matrix/coordinate error or native visibility gold.',
    visible_part_and_q_BG169_interpretation='부위 보임과 중심 재료/관계 차이; not 169 coordinate errors. Corners are material boundaries and nominal/raster uncertainty remains.',
    all390_retained=True, single350=350, cross40=40, cross_votes_not_extra_cases=True, majority_resolution=False,
    all_new_native_visibility_targets=-1, all_new_point_quality_targets=-1,
    training_admitted=False, original_labels_modified=False, source_pixels_models_encoder_or_training_accessed=False,
    model_performance_or_generalization_improvement_claimed=False,
    limitations=['Assistant weak observations are not human/native/pixel gold.', '350 single-reviewer cases remain single weak witnesses.',
        'Euro WOOD only; SQUARE absent; native6 N2; full WOOD/SQUARE/native8 objective remains unresolved.',
        'Prior D/E and owner provenance remain weak; task alignment differences do not explain all actual generalization failure.', 'HTML not browser-rendered.'],
    generation_note='Initial launch had a compile-time duplicate keyword, corrected before artifact generation. Successful generation completed once; no annotations/pixels changed. Initial QA path-string equality was corrected to resolved path plus exact hash; no generated output changed.')
out = O/'READOUT_COMPLETE.json'; assert not out.exists()
out.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(dict(PASS=True, checks=len(checks), readout_complete=bind(out), gallery=bind(O/'index.html')), ensure_ascii=False))
