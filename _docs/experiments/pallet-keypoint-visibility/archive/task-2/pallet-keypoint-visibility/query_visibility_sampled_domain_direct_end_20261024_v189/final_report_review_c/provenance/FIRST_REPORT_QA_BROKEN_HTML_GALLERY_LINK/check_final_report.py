"""Independent final source-report QA: text metadata, links, four PNG SHA only."""
from pathlib import Path
from collections import Counter
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import csv, hashlib, json, os, posixpath, re, stat
H=Path(__file__).absolute().parent;V=H.parent

def safe(p):
    p=Path(p);assert p.is_absolute() and '..' not in p.parts and all('v112' not in x.casefold() for x in p.parts)
    assert p.is_relative_to(V)
    cur=Path(p.anchor)
    for part in p.parts[1:]:
        cur=cur/part;assert not stat.S_ISLNK(cur.lstat().st_mode)
    return p
def sha(p):
    p=safe(p);assert p.suffix.lower() not in ('.npz','.npy','.pt','.pth','.safetensors')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(safe(p).read_text())
def binding(b):assert sha(b['path'])==b['sha256'];return Path(b['path'])
def records(p):return [json.loads(s) for s in safe(p).read_text().splitlines() if s]
def table(p):
    with safe(p).open(newline='') as f:return list(csv.DictReader(f))
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ('href','src') and v)

def run():
    checks=[]
    def ok(name,v):assert bool(v),name;checks.append({'name':name,'PASS':True})
    phase=read(V/'ROOT_PHASE_RESULT.json');results=read(binding(phase['results']))
    ok('frozen_phase_binding',sha(V/'ROOT_PHASE_RESULT.json')=='7923d699403d174e1f2f9777d9b79ec1c78adeb3a82e40ff82c068a65804e4b8')
    for key in ['report','readout','portable']:binding(phase[key])
    ok('report_readout_portable_results_SHA_closure',True)
    for b in results['receipts'].values():binding(b)
    ok('all13_report_receipts_exact_and_phase_receipt_map_equal',phase['source_audits_and_preparation']==results['receipts'] and len(results['receipts'])==13)
    tc=read(binding(results['receipts']['source_tables']));summary=read(binding(tc['summary']))
    for b in tc['tables']+tc['inputs']:binding(b)
    ok('source_table_summary_complete_SHA_and_results_copy',tc['completed'] and tc['PASS'] and summary==results['source_counts'])
    syn=records(tc['inputs'][0]['path']);real=records(tc['inputs'][1]['path']);audit=read(tc['inputs'][2]['path']);admission=read(tc['inputs'][3]['path'])
    known=lambda r:r['direct_end_states']==[0,1,0]
    ok('all1024_synthetic_128bases_native8_U16_preserved',len(syn)==1024 and Counter(r['native_slot'] for r in syn)==Counter({k:128 for k in range(8)}) and len({r['base_index'] for r in syn})==128 and sum(not r['original_input_flags']['query_available'] for r in syn)==16)
    ok('synthetic97_known_927_UNKNOWN_no_coverage_increase',sum(known(r) for r in syn)==97 and results['source_counts']['supported_triplets']['synthetic']==97 and results['source_counts']['V188_to_V189_supported_count_change']['synthetic']==0)
    fc=Counter(tuple(r['factors'][k] for k in ('material','aspect','geometry_family')) for r in syn)
    fk=Counter(tuple(r['factors'][k] for k in ('material','aspect','geometry_family')) for r in syn if known(r))
    fu=Counter(tuple(r['factors'][k] for k in ('material','aspect','geometry_family')) for r in syn if not r['original_input_flags']['query_available'])
    for row in table(tc['tables'][0]['path']):
        key=tuple(row[k] for k in ('material','aspect','geometry_family'));assert [int(row[k]) for k in ('attempts','supported_triplets','unknown','original_inputU')]==[fc[key],fk[key],fc[key]-fk[key],fu[key]]
    ok('all8_factor_CSV_denominators_and_UNKNOWN_independent',len(fc)==8 and set(fc.values())=={128})
    kn=Counter(r['native_slot'] for r in syn if known(r));un=Counter(r['native_slot'] for r in syn if not r['original_input_flags']['query_available'])
    for row in table(tc['tables'][1]['path']):
        k=int(row['native_slot']);assert [int(row[x]) for x in ('attempts','supported_triplets','unknown','original_inputU','direct_visible_rows','direct_external_rows')]==[128,kn[k],128-kn[k],un[k],2*kn[k],kn[k]]
    ok('native8_CSV_and_zero_native6_7_explicit',[kn[k] for k in range(8)]==results['synthetic_native_supported']==[29,24,17,23,2,2,0,0])
    cases={r['case_id'] for r in real};rs=Counter(r['status'] for r in real)
    ok('all22_real66_and5_supported43geometry18UNKNOWN',len(real)==66 and len(cases)==22 and all(sum(r['case_id']==k for r in real)==3 for k in cases) and sorted(rs.values())==[5,18,43] and sum(r['PASS'] for r in real)==5)
    ok('real_known_only_two_frames_native0',all(r['original_input']['native_slot']==0 for r in real if r['PASS']) and {r['case_id'] for r in real if r['PASS']}=={'Q1088','Q1624'})
    for row in table(tc['tables'][2]['path']):
        rr=[r for r in real if r['case_id']==row['case_id']];s=audit['per_case_independent_positive_raw_support'][row['case_id']]
        assert [int(row[x]) for x in ('attempts','supported_triplets','geometry_failure','unknown')]==[3,sum(r['PASS'] for r in rr),sum(r['status']=='GEOMETRY_FAILURE_RETAINED' for r in rr),sum(r['status']=='UNKNOWN_RETAINED' for r in rr)]
        assert row['source_locality']==s['status']
        eligible=s.get('V189_candidate_sampled_eligible','UNKNOWN');assert row['sampled_eligible']==str(eligible)
        if eligible is True:assert row['continuous_extent_violation']=='0' and all(q['inside'] for q in s['per_candidate_sampled_relation'])
    ok('real22_CSV_full_denominators_and_locality_metadata',True)
    new=[r for r in audit['all66_independent_checks'] if r['V188_status']=='UNKNOWN_RETAINED' and r['V189_status']=='GEOMETRY_FAILURE_RETAINED']
    reasons=Counter(k for r in new for k in r['independent_geometry_failure_reasons']);wr=Counter()
    for r in new:
        for k in ('positive_original96_tap','outside_candidate','outside_cover_bbox','outside_sham_bbox'):
            if any(not w[k] for w in r['chosen_witness_checks']):wr[k]+=1
    expectedw={k:wr[k] for k in ('positive_original96_tap','outside_candidate','outside_cover_bbox','outside_sham_bbox')}
    ok('newly_entered33_failure_and_overlapping12_5_3_recount',len(new)==33 and dict(reasons)==summary['real_newly_entered_failure_reasons'] and expectedw==summary['real_newly_entered_witness_reasons'] and expectedw=={'positive_original96_tap':12,'outside_candidate':0,'outside_cover_bbox':5,'outside_sham_bbox':33})
    packet_rows=[r for r in audit['all198_variant_statuses'] if r['constructed_packet']]
    ok('real58_physical_packets464rows_U60',len(packet_rows)==58 and sum(r['original_native_rows'] for r in packet_rows)==464 and sum(r['original_input_U'] for r in packet_rows)==60==summary['real_original_U_rows'])
    gs=admission['groups'];counts=Counter(g['admitted_target'] for g in gs);known_direct=counts[0]+counts[1]
    ok('admission349_2792_D204_E102_UNKNOWN2486',len(gs)==349 and counts==Counter({0:204,1:102,-1:43}) and 8*len(gs)-known_direct==2486)
    ok('full3270_variant26160_native_and2921_missing_null',results['full_requested_variant_census']==3*(len(syn)+len(real))==3270 and results['full_requested_native_status_rows']==8*3270 and results['missing_variant_cache_mapping']==3270-len(gs)==2921)
    ok('no_native_quality_owner_or_broad_target_alias',all(r['new_native_visibility_target']==r['new_point_quality_target']==-1 and not r['training_admission'] for r in syn) and all(r['new_native_quality_owner_all8']==-1 and not r['training_admission'] for r in real))
    pngs=[]
    for b in phase['exact4_source_figure_bindings']:pngs.append(dict(b,size_bytes=safe(binding(b)).stat().st_size))
    ok('exact4_existing_source_figure_SHA_without_decode_or_view',len(pngs)==4)
    pages=[V/'ROOT_REPORT_KO.md',V/'PORTABLE_REPORT_KO.md',V/'ROOT_READOUT_KO.html',V/'synthetic_end_review_visuals_b/PORTABLE_GALLERY_KO.md',V/'independent_actual_real_review_a/actual_saved_real_review_v1/PORTABLE_GALLERY_KO.md']
    links=[]
    for page in pages:
        text=safe(page).read_text();assert '/home/' not in text and 'file://' not in text
        refs=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
        if page.suffix=='.html':parser=Links();parser.feed(text);refs+=parser.links
        for ref in refs:
            u=urlsplit(ref);assert not u.scheme and not u.netloc
            if not u.path:continue
            relative=posixpath.normpath(posixpath.join(str(page.parent.relative_to(V)),unquote(u.path)))
            assert not relative.startswith('../') and not Path(relative).is_absolute()
            target=V/relative;assert target.suffix.lower() not in ('.npz','.npy','.pt','.pth','.safetensors');safe(target);assert target.is_file()
            links.append(dict(page=str(page.relative_to(V)),reference=ref,target=relative))
    ok('all_relative_report_and_portable_gallery_links_exist_without_binary_payload_reads',bool(links))
    doc=(V/'PORTABLE_REPORT_KO.md').read_text();html=(V/'ROOT_READOUT_KO.html').read_text()
    ok('source_weak_limited_coverage_and_no_real_improvement_claim',all(x in doc for x in ['더 좋아진 새 모델은 아직 검증되지','중복 집계','데이터 구성 실패','native0','원래 native/위치품질/중심소유자/주변재질 라벨과 별개','실사 전이 성공으로 해석하지','실제 SAM 입력 계산과 새 학습은 V190']))
    ok('phase_complete_is_source_only_and_goal_unsolved',phase['completed'] and phase['PASS'] and phase['production_fit_or_eval_in_this_phase']==0 and results['new_production_forward_in_V189']==results['new_fit_in_V189']==results['new_model_eval_in_V189']==0 and results['latest_actual_real_model_benchmark']=='V184' and results['latest_real_model_improvement_since_V184'] is None and phase['selected_model'] is None and not phase['adopted'] and not phase['fully_solved'])
    ok('original192_direct32_and_no_eval_selection_contract_clear',all(x in doc for x in ['192개 draw','각16개','가중치1','같은 초기값·8000단계','두 최종 모델','선택에 쓰지 않습니다','PnP와 좌표 보정도 하지 않습니다']) and results['V182_COMMON_fit_admission'] is False)
    ok('HTML_is_offline_source_only_no_browser_claim','<img' in html and len(re.findall(r'<img\b',html))==4)
    result=dict(completed=True,PASS=True,reviewer='C',checks=checks,number_checks=len(checks),phase=binding_dict(V/'ROOT_PHASE_RESULT.json'),results=binding_dict(V/'ROOT_RESULTS_KO.json'),source_table_complete=binding_dict(V/'root_source_tables/COMPLETE.json'),checked_source_rows=dict(synthetic1024=len(syn),real66=len(real),newly_entered33=len(new)),relative_links_checked=len(links),unique_local_targets=len({r['target'] for r in links}),PNG_SHA_checks=pngs,report_files=[binding_dict(p) for p in pages],actual_image_decodes_or_views=0,HTML_browser_rendered=False,source_NPZ_NPY_weights_features_models_or_evaluation_reads=0,source_or_report_modified=False,source_phase_frozen_after_this_receipt=True,limitations=['Numerical/report/relative-link consistency only; no physical real-native gold or transfer/model success proof.','Root/A/B actual visual receipts are bound; C performed only four PNG byte hashes here.','V190 actual cache/fit/evaluation are a separate phase and not re-read in this QA.'])
    with (H/'REVIEW.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,checks=len(checks),path=str(H/'REVIEW.json'),sha256=sha(H/'REVIEW.json'),links=len(links))))
def binding_dict(p):return dict(path=str(p),sha256=sha(p))
if __name__=='__main__':run()
