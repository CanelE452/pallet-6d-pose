"""Offline final report QA. Explicit --run-after-root-ready, never source NPZ.

Preparation exercises only invented strings/metadata; actual mode reads the
final reports, their explicit metadata receipts, and exact referenced images.
No browser, model, pixels-as-labels, fit, threshold or selection operation.
"""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import argparse,hashlib,json,os,re,stat

H=Path(__file__).absolute().parent;V=H.parent
ALLOWED={'.json','.jsonl','.csv','.py','.md','.html','.png','.jpg','.jpeg','.svg'}
PAYLOAD={'.npz','.npy','.pt','.pth','.safetensors','.bin'}

def safe(path):
    p=Path(path)
    assert p.is_absolute() and '..' not in p.parts and all('v112' not in x.casefold()for x in p.parts)
    assert p.is_relative_to(V),'Only final phase report references'
    return p

def contents(path):
    p=safe(path);assert p.suffix in ALLOWED and p.suffix not in PAYLOAD
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nf=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nf
        last=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    with os.fdopen(last,'rb')as f:
        assert stat.S_ISREG(os.fstat(f.fileno()).st_mode);return f.read()

def binding(path):return dict(path=str(path),sha256=hashlib.sha256(contents(path)).hexdigest())
def read(path):return json.loads(contents(path))
def checked(b):
    p=safe(b['path']);assert binding(p)==b,'Final binding drift '+str(p);return p
def write(path,obj):
    with path.open('x')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.words=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('href','src')and value:self.links.append(value)
    def handle_data(self,data):self.words.append(data)

def local_path(document,reference):
    u=urlsplit(reference)
    if u.scheme or u.netloc:return None,'EXTERNAL_OR_DATA_URL_NOT_FETCHED'
    if not u.path:return None,'ANCHOR'
    raw=unquote(u.path)
    assert not raw.startswith('/') and 'file:'not in raw
    assert all('v112'not in x.casefold()for x in raw.split('/')),'Forbidden name before filesystem access'
    parts=list(document.parent.parts)
    for part in raw.split('/'):
        if part in ('','.'):continue
        if part=='..':parts.pop()
        else:parts.append(part)
    p=safe(Path(*parts))
    if p.suffix.casefold()in PAYLOAD:return None,'RAW_PROVENANCE_LINK_NOT_OPENED_OR_HASHED'
    return p,'LOCAL_REPORT_OR_RENDERED_IMAGE'

def doc_links(path):
    text=contents(path).decode()
    if path.suffix=='.html':p=Links();p.feed(text);refs=p.links
    else:refs=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
    out=[]
    for ref in refs:
        target,status=local_path(path,ref)
        row=dict(document=str(path.relative_to(V)),reference=ref,status=status)
        if target is not None:row['target_binding']=binding(target)
        out.append(row)
    return out

def prepare():
    p=V/'a/b/report.md';checks={}
    checks['relative_link_normalizes_within_phase']=local_path(p,'../../ROOT_RESULTS_KO.json')[0]==V/'ROOT_RESULTS_KO.json'
    checks['external_URL_not_fetched']=local_path(p,'https://example.invalid/a')[0]is None
    checks['raw_NPZ_not_opened_or_hashed']=local_path(p,'inputs.npz')[1]=='RAW_PROVENANCE_LINK_NOT_OPENED_OR_HASHED'
    try:local_path(p,'../../forbidden_v112/file.json')
    except AssertionError:checks['forbidden_name_rejected_before_access']=True
    x=Links();x.feed('<p>known</p><img src="../plot.png"><a href="RESULT.json">data</a>')
    checks['HTML_href_src_extracted_without_execution']=x.links==['../plot.png','RESULT.json']and''.join(x.words)=='knowndata'
    assert all(checks.values())
    write(H/'CHECKER_PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,code=binding(Path(__file__).absolute()),invented_checks=checks,actual_final_report_reads=0,actual_source_or_payload_reads=0,browser_execution=False))

def run():
    result=read(V/'ROOT_RESULTS_KO.json');phase=read(V/'ROOT_PHASE_RESULT.json');index=read(V/'ARTIFACT_INDEX.json')
    own=read(V/'independent_descriptive_tables_review_c/REVIEW.json')
    checks={}
    def record(name,value):assert value,name;checks[name]=True
    record('root_phase_data_diagnostic_not_model_completion',result['completed']and result['PASS']and result['phase']=='V188'and result['result_type']=='SOURCE_DIRECT_END_CONTROL_DATA_DIAGNOSTIC'and not result['full_goal_solved']and result['selected_model']is None and not result['adopted'])
    record('new_fit_forward_evaluation_PnP_zero',all(result[k]==0 for k in ('model_fit_updates','model_forward_calls','current_or_held_or_VAL_data_reads','PnP'))and phase['new_fit_or_forward_or_evaluation']==0)
    record('old_labels_unchanged_new_native_quality_owner_unknown_noadmission',result['old_annotations_modified']is False and result['new_native_quality_owner_targets']==-1 and result['training_admission']is False)
    for k in ('synthetic','real','factor_cells','native8','all22_locality'):record('counts_exact_independent_'+k,result['summary'][k]==own[k])
    record('phase_unsolved_no_adoption_and_publication_pending',not phase['model_goal_solved']and not phase['new_model_adopted']and phase['publication_required'])
    for key,name in [('results','ROOT_RESULTS_KO.json'),('portable_report','PORTABLE_REPORT_KO.md'),('html_report','ROOT_READOUT_KO.html')]:record('phase_binding_'+key,phase[key]==binding(V/name))
    record('artifact_index_phase_and_builder_bindings',index['phase_result']==binding(V/'ROOT_PHASE_RESULT.json')and checked(index['root_report_builder']).name=='build_root_report.py')
    reviews={}
    for name,b in result['independent_actual_reviews'].items():
        d=read(checked(b));record('closed_independent_'+name,d['completed']and d['PASS']);reviews[name]=b
    for key in ('actual_synthetic_generation','actual_real_generation','source_RGB_observations','root_actual_control_visual_receipt','root_actual_synthetic_visual_receipt','separate_real_source_visual_QA'):
        p=checked(result[key]);d=read(p);record('closed_metadata_'+key,d['completed']is True)
    images=[checked(b)for b in result['figures']];galleries=[checked(b)for b in result['galleries']]
    record('coverageV2_and_two_portable_galleries_present',any(p.name=='SOURCE_CONTROL_COVERAGE_V2.png'for p in images)and len(galleries)==2 and all(p.name=='PORTABLE_GALLERY.md'for p in galleries))
    md=contents(V/'PORTABLE_REPORT_KO.md').decode();html=contents(V/'ROOT_READOUT_KO.html').decode();parser=Links();parser.feed(html);visible=' '.join(parser.words)
    record('portable_report_no_absolute_local_links','/home/'not in md and 'file://'not in md)
    phrases=['통과율은 모델 정확도가 아닙니다','15시도','5사례','Q0057','33시도','11사례','raw tap 공백','native6/7','두 프레임의 native0','작은 윤곽 변화','학습하기 쉬운 시각 신호나 실사 전이를 보증하지 않습니다','Q1624','Q1608','not-blind','별도']
    record('human_readout_cause_and_authority_caveats_preserved',all(p in md and p in visible for p in phrases))
    record('originalstep_formula_and_bilinear_authority_visible','min(0.03, float(min(part_side)) * 0.7)'in md and 'bilinear'in md and '준비 문구가 실제 입력 변형으로 실행된 적은 없습니다'in md)
    links=[]
    for p in [V/'PORTABLE_REPORT_KO.md',V/'ROOT_READOUT_KO.html']+galleries:links+=doc_links(p)
    record('all_local_report_and_gallery_links_hash_exist',bool(links)and all(r.get('target_binding')or r['status']in ('EXTERNAL_OR_DATA_URL_NOT_FETCHED','ANCHOR','RAW_PROVENANCE_LINK_NOT_OPENED_OR_HASHED')for r in links))
    # Exact saved report binding checks; image hashes do not constitute pixel QA.
    output=dict(completed=True,PASS=True,review_kind='INDEPENDENT_FINAL_SOURCE_REPORT_NUMBERS_AUTHORITY_AND_OFFLINE_LINKS',code=binding(Path(__file__).absolute()),root_results=binding(V/'ROOT_RESULTS_KO.json'),root_phase=binding(V/'ROOT_PHASE_RESULT.json'),readout=binding(V/'ROOT_READOUT_KO.html'),portable=binding(V/'PORTABLE_REPORT_KO.md'),artifact_index=binding(V/'ARTIFACT_INDEX.json'),checks=checks,number_checks=len(checks),figures=result['figures'],galleries=result['galleries'],independent_actual_reviews=reviews,link_records=links,limits=['Offline HTML source/link QA only; no browser rendering claim.','Referenced PNG bytes/hash checks only here; actual view scopes remain the separate root/A/C visual receipts.','No source rawRGB/NPZ/NPY/depth/mask/model/evaluation data reads or new scoring.','Conservative source controls and assistantweak observations are not model accuracy or physical/native gold.'],model_fit_forward_PnP_threshold_or_selection=0,payload_files_read_or_hashed=0,fully_solved=False,selected_model=None)
    write(H/'REVIEW.json',output)
    print(json.dumps(dict(completed=True,PASS=True,review=binding(H/'REVIEW.json'),checks=len(checks),local_links=len(links))))

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--run-after-root-ready',action='store_true');args=p.parse_args()
    prepare()if args.prepare else run()
