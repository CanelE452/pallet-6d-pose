"""Final V187 source report QA; metadata/text only, no new raw image views."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import hashlib,json,os,re,stat

H=Path(__file__).absolute().parent;V=H.parent
EXPECTED={'ROOT_RESULTS_KO.json':'f820cb3c5932d0bbc1bfdcac7481c18abeab7fa448095df13c1d65e99bbe01c7',
 'PORTABLE_REPORT_KO.md':'57b649a1f25334ffdae9cfb613bb2ba30cdd2a272548c33a093a9c8330d192c2',
 'ROOT_READOUT_KO.html':'7eede49846d1dbee6766fd1ae2895639a5bb0f2166f2a25182917840df8c4b0f'}

def need(ok,msg):
    if not ok:raise ValueError(msg)
def check_name(path):
    p=Path(path);need(p.is_absolute() and V in p.parents and '..' not in p.parts and all('v112' not in s.lower() for s in p.parts),'Forbidden/out-of-scope name precheck');return p
def bind(path):
    p=check_name(path);need(p.suffix in ('.json','.jsonl','.csv','.py','.md','.html'),'Only source/report metadata text')
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(path):return json.loads(check_name(path).read_text())
def assert_binding(b):need(bind(b['path'])==b,'Frozen report/source metadata binding drift')

class ReportHTML(HTMLParser):
    def __init__(self):super().__init__();self.targets=[];self.tables=[];self.table=None;self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if value and key in ('href','src'):self.targets.append(value)
        if tag=='table':self.table=[]
        elif tag=='tr' and self.table is not None:self.row=[]
        elif tag in ('td','th') and self.row is not None:self.cell=''
    def handle_data(self,value):
        if self.cell is not None:self.cell+=value
    def handle_endtag(self,tag):
        if tag in ('td','th') and self.cell is not None:self.row.append(self.cell.strip());self.cell=None
        elif tag=='tr' and self.row is not None:self.table.append(self.row);self.row=None
        elif tag=='table' and self.table is not None:self.tables.append(self.table);self.table=None

def markdown_tables(text):
    tables=[];current=[]
    for line in text.splitlines()+['']:
        if line.startswith('|'):
            row=[c.strip() for c in line.strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',c or ' ') for c in row):current.append(row)
        elif current:tables.append(current);current=[]
    return tables

def local_link_exists(target):
    u=urlsplit(target);need(not u.scheme and not u.netloc,'Offline source report relative links only')
    if not u.path:return
    rel=Path(unquote(u.path));need(not rel.is_absolute() and '..' not in rel.parts and all('v112' not in p.lower() for p in rel.parts),'Unsafe report link precheck')
    parts=rel.parts;fd=os.open(V,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for p in parts[:-1]:
            new=os.open(p,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=new
        if parts:
            info=os.stat(parts[-1],dir_fd=fd,follow_symlinks=False)
            need(not stat.S_ISLNK(info.st_mode),'Report link symlink rejected')
    finally:os.close(fd)

def main():
    for name,digest in EXPECTED.items():need(bind(V/name)['sha256']==digest,'Exact final report version')
    result=read(V/'ROOT_RESULTS_KO.json');complete=read(V/'ROOT_READOUT_BUILD_COMPLETE.json')
    summary=read(V/'descriptive_source_tables_v1/SUMMARY.json');table_review=read(V/'source_table_review_b/REVIEW.json')
    nominal_review=read(V/'independent_nominal_review_b/actual_review_v1/REVIEW.json')
    contract=read(V/'source_table_review_b/REPORT_QA_CONTRACT_BEFORE_ROOT_REPORTS.json')
    need(result['all_source_population']==summary and table_review['PASS'],'All embedded source numbers exactly from independently checked summary')
    need(result['independent_nominal_max_difference']==nominal_review['maximum_numerical_differences'],'Independent numerical bound reported verbatim')
    for key in ('new_training_calls','new_initializer_forwards','PnP_or_pose_optimization_calls','new_source_semantic_labels','new_current223_or_publicheld45_or_syntheticVAL_body_reads'):need(result[key]==0,'No new action claim')
    need(result['full_goal_solved'] is False and result['selected_model'] is None and result['fit_admission'] is False,'Unsolved/no selection/admission')
    need(result['source_receipts']==complete['source_receipts'] and len(result['source_receipts'])==10,'All10 original source receipts bound consistently')
    for name,b in complete['outputs'].items():need(b==bind(V/name),'Exact builder report outputs')
    for binding in result['source_receipts'].values():assert_binding(binding)
    root=read(V/'ROOT_ACTUAL_FIXED_SOURCE_VISUAL_QA.json');a=read(V/'actual_visual_review_a/REVIEW.json');visual=read(V/'fixed_source_nominal_visual_v1/COMPLETE.json')
    need(len(root['root_actual_tool_views'])==9 and a['actual_view_image_calls']==8 and len(a['actual_viewed_images'])==8,'Root9/A8 actual receipts')
    need(CounterBasenames(root['root_actual_tool_views'])=={'ALL8_CONTACT.png':4,'OVERVIEW.png':2,'NATIVE_3_CARD.png':3},'Root precise visual scope')
    need(a['original_RGB_images_actually_viewed']==a['overview_images_actually_viewed']==4,'A precise visual scope')
    need(result['visual_observations_not_annotations']==root['observations_KO'],'Qualitative observations attributed to actual root receipt')
    need(visual['original_native_rows']==32 and visual['original_inputU']==3 and visual['all32_original_RGB96_valid96_sampler_and_cleanPNG_byteparity'] is True,'Fixed32/inputU3 preservation receipt')
    need(all(r['HTML_browser_rendered'] is False for r in (complete,root,a,visual)),'No browser rendering claim')
    html=(V/'ROOT_READOUT_KO.html').read_text();md=(V/'PORTABLE_REPORT_KO.md').read_text();parser=ReportHTML();parser.feed(html)
    need(len(parser.tables)==len(markdown_tables(md))==2,'Both reports have exact scene/native tables')
    scene=[[r['scene'],str(r['frames']),str(r['selected_obj1']),str(r['selected_other']),str(r['selected_UNKNOWN'])] for r in summary['per_scene']]
    native=[]
    for k in range(8):
        r=summary['per_native'][str(k)];native.append([str(k),str(r['total']),str(r['input_available']),str(r['inputU']),str(r['eligible_inside_original_crop_extent']),str(r['eligible_outside_original_crop_extent']),str(r['available_lookup_status'].get('UNKNOWN_NO_UNIQUE_OBJ1_WEAK_ASSOCIATION',0))])
    for table_set in (parser.tables,markdown_tables(md)):
        need(table_set[0][1:]==scene and table_set[1][1:]==native,'Every HTML/MD table cell agrees with audited scene/native counts')
    targets=parser.targets+[m.group(1) for m in re.finditer(r'!?\[[^\]]*\]\(([^)]+)\)',md)]
    for target in targets:local_link_exists(target)
    for text in (html,md):
        for phrase in ('185','1,480','194','924','650','274','362','137','40','8','UNKNOWN','WOOD/SQUARE','모델 성능 개선 결과가 아니다','identity 정확도가 아니다','실제 좌표 오류나 가림 정답으로 해석할 수 없다','기존 사람의 semantic D/E/S','HTML browser rendering은 검증하지 않았다'):
            need(phrase in text,'Required authority/denominator wording missing: '+phrase)
        need('Broad patch V' in text and 'native D' in text,'Broad support vs queried end distinction')
    need(len(result['preserved_failures'])==4,'Failure history remains visible')
    failure=V/'selected_instance_metadata_diagnostic_v1/FAILED_EXECUTION_PRESERVED.json';repair=V/'ROOT_NESTED_R_NOTATION_ONLY_REPAIR_SCOPE.json'
    need(read(failure)['completed'] is False and bool(read(repair)),'Linked V1 failure and notation-only repair preserved')
    old=V/'selected_instance_preparation_c/associate_frozen_source_instances.py'
    need(bind(old)['sha256']=='ac319f57518f3f362aaa76ed1c8ff12c7afe089c2cf35a1e14e4c033d9102f43','Original V1 code unchanged')
    # No linked image has been opened/hashed here. Only existing actual chart QA is cited.
    image_QA=V/'source_table_review_b/ACTUAL_FIGURE_VISUAL_QA.json';need(read(image_QA)['PASS'],'Earlier actual single-chart view bound')
    report=dict(completed=True,PASS=True,report_bindings={name:bind(V/name) for name in EXPECTED},build_complete=bind(V/'ROOT_READOUT_BUILD_COMPLETE.json'),
        prospective_QA_contract=bind(V/'source_table_review_b/REPORT_QA_CONTRACT_BEFORE_ROOT_REPORTS.json'),table_review=bind(V/'source_table_review_b/REVIEW.json'),
        actual_chart_view_QA=bind(image_QA),source_receipts_rehashed=complete['source_receipts'],HTML_source_checked=True,HTML_browser_rendered=False,
        scene_table_rows_checked=4,native_table_rows_checked=8,report_table_data_cells_checked=76,relative_href_src_checked=len(targets),unsafe_or_missing_links=0,
        source_all185_rows1480_U194_GT823_obj1_258_refs2064_exact=True,weak137_other40_unknown8_not_identity_accuracy=True,
        eligible924_inside650_outside274_not_physical_error=True,new_semantic_labels_UNKNOWN_old_annotations_unchanged=True,
        broad_support2_not_native_D_decoder=True,original_qL_native_KRT_unchanged=True,all10_receipts_exact=True,root_actual_views9_A8_scope_accurate=True,
        no_new_model_fit_PnP_labels_admission_or_goal_success_claim=True,preserved_failure_links_verified=[bind(failure),bind(repair)],preserved_V1_code=bind(old),
        findings=[],blocking_issues=0,new_source_raw_images_or_asset_bodies_opened=0,new_model_or_evaluation_body_reads=0,
        limits=['Text/source/relative-link QA only; HTML not browser-rendered.','Root9/A8 source-image claims are verified against their receipts; B newly viewed only the authorized scientific chart, not the raw source crops.','Source crop pixel replay remains the bound renderer proof; this review did not replay raw crops.','Root invocation failure statements are attributed to preserved root history; this review directly rechecked linked V1 failure/notation-repair metadata and unchanged V1 source.'])
    for name,digest in EXPECTED.items():need(bind(V/name)['sha256']==digest,'Report unchanged after review')
    with (H/'REVIEW.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,links=len(targets),table_data_cells=76,review=bind(H/'REVIEW.json'))))

def CounterBasenames(bindings):
    result={}
    for binding in bindings:
        name=Path(binding['path']).name;result[name]=result.get(name,0)+1
    return result

if __name__=='__main__':main()
