"""Final saved V216 report facts/links QA; no producer, image or GT execution."""
from pathlib import Path
from collections import Counter
from html.parser import HTMLParser
import hashlib,json,os,re,stat,datetime,traceback

V=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_public_train_part_query_support_contract_20261025_v216')
OLD=V.parent/'query_visibility_public_train_visible_part_coverage_expansion_20261025_v215'
O=V/'final_scientific_report_review_c'
def guard(p,link=False):
    p=Path(p);assert 'v112' not in str(p).casefold() and p.is_absolute() and '..' not in p.parts
    assert V in p.parents or (link and OLD in p.parents)
    for q in [p,*p.parents]:assert not q.is_symlink()
    if not link:assert p.suffix in ('.json','.csv','.md','.html','.py')
    return p
def read(p):return guard(p).read_bytes()
def sha(b):return hashlib.sha256(b).hexdigest()
def bind(p):b=read(p);return dict(path=str(p),sha256=sha(b),size_bytes=len(b))
def bound(b):
    raw=read(b['path']);assert sha(raw)==b['sha256']
    if 'size_bytes' in b:assert len(raw)==b['size_bytes']
    return raw
def js(b):return json.loads(bound(b))
def bid(b):return b['path'],b['sha256']
def put(name,x):
    p=O/name
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
    return bind(p)
class HTML(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.imgs=[];self.rows=[];self.row=None;self.cell=None
    def handle_starttag(self,t,attrs):
        self.links.extend(v for k,v in attrs if k in ('href','src') and v)
        if t=='img':self.imgs.extend(v for k,v in attrs if k=='src')
        if t=='tr':self.row=[]
        if t in ('td','th'):self.cell=[]
    def handle_data(self,d):
        if self.cell is not None:self.cell.append(d)
    def handle_endtag(self,t):
        if t in ('td','th') and self.cell is not None:self.row.append(''.join(self.cell));self.cell=None
        if t=='tr' and self.row is not None:self.rows.append(self.row);self.row=None

def main():
    newp=V/'report_wording_repair_a_v2/FINAL_READY.json';new=js(dict(path=str(newp),sha256='46d250b8e16e07ad8e8d3ba2bcea864e12727995b108bc42165c8c9c9f87499e'))
    original=js(new['historical_original_READY']);repair=js(new['repair_receipt'])
    docs={name:bound(b) for name,b in new['current_report_bindings'].items()}
    preserved={k:bound(v) for k,v in new['original_preservation'].items()}
    old_sentence='새 SOURCE label/admission/paired controls/fit은 하지 않았습니다.'
    new_sentence='새 SOURCE weak 관찰은 기록했지만, 새 학습 타깃 채택·paired controls·fit은 하지 않았습니다.'
    oldhtml=preserved['original_HTML'].decode();html=docs['ROOT_READOUT_KO.html'].decode()
    assert oldhtml.count(old_sentence)==1 and html.count(new_sentence)==1 and old_sentence not in html
    assert oldhtml.replace(old_sentence,new_sentence,1)==html and html.replace(new_sentence,old_sentence,1)==oldhtml
    assert preserved['original_READY']==bound(new['historical_original_READY']) and preserved['original_generator']==bound(new['original_report_builder_is_historical_pre_wording_repair'])
    for name in ('ROOT_RESULTS_KO.json','ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md'):assert docs[name]==bound(original['outputs'][name])
    assert docs['ROOT_REPORT_KO.md']==docs['PORTABLE_REPORT_KO.md']
    provisional=json.loads(read(O/'ROOT_RECORD_CONSISTENCY.json'));assert provisional['completed'] and provisional['PASS']
    result=json.loads(docs['ROOT_RESULTS_KO.json']);root=js(original['input_bindings']['root_semantic']);joined=js(original['input_bindings']['joined_rows'])['rows96'];side=js(original['input_bindings']['exact_OR'])['rows96'];closed=js(original['input_bindings']['independent_join_QA']);calc=js(closed['independent_recomputation'])
    rr=root['rows96'];assert len(rr)==len(joined)==len(side)==96 and result['all96_root_semantic_rows']==rr
    def stats(ii):
        rows=[rr[i] for i in ii];jj=[joined[i] for i in ii];ss=[side[i] for i in ii]
        valid=Counter('no_ROI' if s['candidate_union_all_valid96'] is None else 'valid' if s['candidate_union_all_valid96'] else 'invalid_preserved' for s in ss)
        return dict(N=len(ii),frames=len({r['frame_position'] for r in rows}),metadata_components=len({r['input']['component_id'] for r in jj}),original_inputU=sum(r['original_inputU'] for r in rows),reader_A_D=sum(r['reader_A_axes']['visibility']['state']=='D' for r in jj),reader_C_D=sum(r['reader_C_axes']['visibility']['state']=='D' for r in jj),raw_native_weak_D=sum(r['raw_native_weak_candidate']=='D' for r in rows),raw_spatial_witness_candidates=sum(r['raw_spatial_witness_candidate'] for r in rows),root_weak_same_pallet_part_D=sum(r['root_weak_part_state']=='D' for r in rows),root_native_UNRESOLVED=sum(r['root_weak_part_state']=='UNKNOWN' for r in rows),native_external_E_confirmed=sum(r['root_weak_part_state']=='E' for r in rows),native_SELF_confirmed=sum(r['root_weak_part_state']=='SELF' for r in rows),query_support_agreement=dict(Counter(r['raw_query_support_agreement'] for r in rows)),exact_OR_validity={k:valid.get(k,0) for k in ('valid','invalid_preserved','no_ROI')},root_review_status=dict(Counter(r['root_semantic_review_status'] for r in rows)))
    role_indices={role:[i for i,r in enumerate(rr) if role=='ALL' or r['internal_source_role']==role] for role in ('ALL','SOURCE_FIT','SOURCE_DEV')}
    counts={role:stats(ii) for role,ii in role_indices.items()}
    detail={role:{str(n):stats([i for i in ii if rr[i]['native_slot']==n]) for n in range(8)} for role,ii in role_indices.items()}
    assert result['counts_by_role']==counts and result['counts_by_role_and_native']==detail
    assert result['withheld_raw_D_cases']==provisional['withheld_rows']
    for role in role_indices:
        assert result['raw_reader_visibility_counts'][role]==calc['all_role_native_reader_counts'][role]['readers']
        assert result['all_axis_disagreements'][role]==calc['all_role_native_reader_counts'][role]['join']['categorical_axis_disagreements']
    for f in result['frame_readout']:
        pos=f['position'];assert f['counts']==stats(list(range(pos*8,pos*8+8))) and f['root_observation_ko']==root['all12_actual_viewed'][pos]['manual_root_observation_ko'] and bid(f['overlay'])==bid(root['all12_actual_viewed'][pos]['actual_overlay_view'])
    assert result['authority']['physical_native_point_visibility_gold']=='UNKNOWN' and result['authority']['primary_DE_training_admission'] is False and result['authority']['auxiliary_training_admission'] is False
    assert result['invariants']['original_outer_role_TRAIN'] is True and result['invariants']['future_DEV_all_loss_branch_sampler_and_normalization_exclusion_required'] is True and result['new_fit_forward_evaluation_model_or_threshold_change'] is False
    assert result['GT_readiness']['current_GT_payload_access']==0 and result['GT_readiness']['payload_current_reverification'] is False and result['GT_readiness']['current_teacher_or_training_admission'] is False
    assert result['execution_history']['v1_runtime_execution_count']==0 and result['execution_history']['v2_actual_execution_count']==1
    expect={}
    expect['roles']=[[role,str(s['frames'])+'/'+str(s['N']),s['original_inputU'],s['raw_native_weak_D'],s['root_weak_same_pallet_part_D'],s['root_native_UNRESOLVED'],s['raw_spatial_witness_candidates']] for role,s in counts.items()]
    expect['native_all']=[[n,s['N'],s['original_inputU'],s['raw_spatial_witness_candidates'],s['raw_native_weak_D'],s['root_weak_same_pallet_part_D'],s['root_native_UNRESOLVED'],'/'.join(str(s['query_support_agreement'].get(k,0)) for k in ('ON','NEAR','OFF','UNKNOWN'))] for n in range(8) for s in [detail['ALL'][str(n)]]]
    expect['role_native']=[[role,n,s['N'],s['original_inputU'],s['raw_native_weak_D'],s['root_weak_same_pallet_part_D'],s['root_native_UNRESOLVED'],s['raw_spatial_witness_candidates']] for role in ('SOURCE_FIT','SOURCE_DEV') for n in range(8) for s in [detail[role][str(n)]]]
    expect['query_and_exact_OR']=[[role,*[s['query_support_agreement'].get(k,0) for k in ('ON','NEAR','OFF','UNKNOWN')],'/'.join(str(s['exact_OR_validity'].get(k,0)) for k in ('valid','invalid_preserved','no_ROI'))] for role,s in counts.items()]
    md=docs['ROOT_REPORT_KO.md'].decode();mdrows=[[x.strip() for x in line.strip().strip('|').split('|')] for line in md.splitlines() if line.startswith('|')]
    h=HTML();h.feed(html)
    for key,rows in expect.items():
        assert result['tables'][key]['rows']==rows
        for row in rows:assert list(map(str,row)) in mdrows and list(map(str,row)) in h.rows
    gt=js(original['input_bindings']['GT_text_readiness']);gtko=bound(original['input_bindings']['GT_text_KO']).decode()
    assert gt['completed'] is True and gt['PASS'] is True
    # Human-readable source interpretation below was read directly; exact text
    # assertions are limited to the one approved literal repair, not prose style.
    links={};targets=set()
    for name in ('ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html'):
        text=docs[name].decode()
        if name.endswith('.html'):hp=HTML();hp.feed(text);urls=hp.links
        else:urls=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
        for url in urls:
            assert not url.startswith(('/','file:','http:','https:')) and '://' not in url
            target=guard(Path(os.path.normpath(str(V/url))),link=True)
            assert target.suffix in ('.json','.md','.html','.csv','.py','.png')
            st=target.lstat();assert stat.S_ISREG(st.st_mode);targets.add(str(target))
        links[name]=len(urls)
    assert sum(links.values())==60 and len(targets)==27
    desired_overlays={str(V/'joined_part_query_support_v2'/('FIXED_FRAME_%02d_TWO_READER_OVERLAY.png'%n)) for n in (0,6,10)}
    desired_clean={x['binding']['path'] for x in root['additional_two_actual_clean96_views']}
    assert {os.path.normpath(str(V/p)) for p in h.imgs}==desired_overlays|desired_clean
    for path in desired_overlays:assert path in targets
    provenance=dict(completed=True,PASS=True,original_HTML_hash=sha(preserved['original_HTML']),current_HTML_hash=sha(docs['ROOT_READOUT_KO.html']),original_READY=new['historical_original_READY'],wording_blocker='Old HTML could incorrectly imply no new weak SOURCE observations were created.',approved_one_literal_replacement=True,reverse_replacement_exact=True,other_three_reports_byte_exact=True,generator_original_unchanged=True,repair_ready=bind(newp),repair_receipt=new['repair_receipt'],initial_C_report_read_chunk='c85d7a',source_annotation_join_metric_or_model_changes=0)
    finding=put('FIRST_REPORT_WORDING_FINDING_PRESERVED.json',provenance)
    checks=put('FINAL_REPORT_NUMERIC_LINK_CHECKS.json',dict(completed=True,PASS=True,counts_by_role=counts,counts_by_role_and_native=detail,all_4_JSON_tables_MD_HTML_exact=True,all96_root_rows_and_2_withheld_exact=True,raw_reader_and_allaxis_counts_exact=True,links=links,local_unique_targets=len(targets),fixed_0_6_10_and_two_withheld_images_as_bound=True,wording_repair=finding,C_new_image_or_HTML_browser_views=0))
    review=put('REVIEW.json',dict(completed=True,PASS=True,current_report_READY=bind(newp),current_four_reports=new['current_report_bindings'],historical_original_report_READY=new['historical_original_READY'],builder_original=original['builder'],root_record_consistency=bind(O/'ROOT_RECORD_CONSISTENCY.json'),closed_saved_join_QA=original['input_bindings']['independent_join_QA'],numeric_and_link_checks=checks,wording_finding_and_exact_repair=finding,code=bind(__file__),facts_KO=['전체96/U11/FIT64DEV32 및 raw D11과 root weak part D9(FIT7/DEV2)를 구분했다. 2개 보류만 별도이며 raw 관찰/ROI/기계 후보는 바꾸지 않았다.','3역할×native8=24 JSON strata와 표시된 native8/role-native16/role3/query3 수표, 원reader·불일치·96 root 행이 저장 원문과 일치한다.','local witness62, query ON24/NEAR17/OFF11/UNKNOWN44, exactOR79valid/4invalid/13noROI를 native/gold/alpha proof로 오인하지 않으며 4invalid를 보존한다.','실제 root view12overlay+2clean과 C의 앞선 독립관찰24view는 각자 봉인영수증의 권위다. 이 최종QA는 새 이미지 view나 HTML browser render를 수행하지 않았다.','공개 GT의 과거 보존 영수증과 V154/V159 prior use는 A의 text-only 준비도에 근거한다. 현재 GT payload 존재/바이트/교사 적격성을 재검증한 주장이 아니며 처음 GT 활용도 아니다.','다음 한 가지 SOURCE TRAIN correspondence/mask/depth bounded audit를 기술하며 user 승인 대기나 완벽mesh gold를 영구 진전의 gate로 제시하지 않는다. 실제 접근 전 exact TRAIN scope를 두는 내부 실행 조건이다.','HTML의 새 약관찰 부재 오해 문장만 교정되었다. weak 관찰 생성과 학습 타깃 채택/paired control/fit 미실행을 분명히 구분하며 새 학습·실사효과·adoption·목표완료 주장이 없다.'],limits_KO=['C는 root qualitative withholding의 저장 일관성을 검산했다. 새 semantic relabeling이나 root overlay 재관찰·3D gold 인증을 하지 않았다.','GT text-readiness의 기존20 local 문서·외부웹을 재조사하거나 GT source payload를 stat/hash/read하지 않았다. 현재 공개 복사나 Git 상태도 검증 범위가 아니다.','old READY/generator는 역사적 최초 build이며 current HTML의 정확한 재현은 별도 단일 문장 repair provenance와 새 READY를 함께 사용한다.'],access=dict(new_source_RGB_or_overlay_views=0,new_valid96_decode=0,GT_array_model_weights_scores_or_Git=0,source_or_producer_modifications=0),own_review_frozen_no_further_writes=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(completed=True,PASS=True,REVIEW=review,checks=checks,repair_provenance=finding,links=links,unique_targets=len(targets)),ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except BaseException:
        put('FAILURE_PRESERVED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),source_edits=0,automatic_retry=False));raise
