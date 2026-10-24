"""Factual report/CSV/link QA only; no models, input arrays or image decode."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import csv,hashlib,json,re,stat
H=Path(__file__).absolute().parent;V=H.parent;E=V/'saved_source_direct_end_evaluation_v2_PARTIAL_UNKNOWN_PRESERVED'

def safe(p,exists=True):
    p=Path(p);assert p.is_absolute()and'v112'not in str(p).casefold()and'..'not in p.parts
    cur=Path(p.anchor)
    for part in p.parts[1:]:
        cur/=part
        if not exists and not cur.exists():continue
        assert not stat.S_ISLNK(cur.lstat().st_mode)
    return p

def load(p):return json.loads(safe(p).read_bytes())
def binding(p):return dict(path=str(p),sha256=hashlib.sha256(safe(p).read_bytes()).hexdigest())
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in('href','src')and v:self.links.append(v)

def run():
    result=load(V/'ROOT_RESULTS.json');source=load(E/'SOURCE_DIRECT_WEAK_METRICS.json');done=load(E/'COMPLETE.json');md=safe(V/'PORTABLE_REPORT_KO.md').read_text();html=safe(V/'ROOT_READOUT_KO.html').read_text()
    checks={};checks['all_source_cohort_and_triplet_numbers_exact_saved_JSON']=result['cohorts']==source['cohorts']and result['triplets']==source['triplets']
    rows=list(csv.DictReader(safe(E/'SOURCE_DIRECT_WEAK_CONFUSIONS_AND_STRATA.csv').open()))
    for r in rows:
        cohort=source['cohorts'][r['cohort']];group=cohort['all']if r['stratum']=='ALL'else cohort['by_domain'if r['stratum'].startswith('DOMAIN:')else'by_native'][r['stratum'].split(':',1)[1]];m=group['models'][r['arm']]
        assert json.loads(r['CM_D_E_by_D_E_U'])==m['CM_D_E_by_D_E_U']
        for k,v in m.items():
            if k in('CM_D_E_by_D_E_U','UNKNOWN_target_prediction_counts'):continue
            assert (None if r[k]==''else float(r[k]))==v
    checks['all66_savedCSV_rows_agree_with_report_full_JSON']=len(rows)==66
    arms=result['arms'];h=result['cohorts']['HOLDOUT']['all'];a,b=[h['models'][x]for x in arms]
    checks['headline_E4to6_D28to26_correct32_net0']=a['CM_D_E_by_D_E_U']==[[28,2,0],[11,4,0]]and b['CM_D_E_by_D_E_U']==[[26,4,0],[9,6,0]]and h['paired_correctness']['recovered_by_XY']==h['paired_correctness']['damaged_by_XY']==3 and h['paired_correctness']['net_correct_change']==0
    checks['UNKNOWN_inputU_partial_source_scope_preserved']=result['target_UNKNOWN']==2772 and result['original_inputU']==112 and result['triplets']['FULL']['physical_all_native_attempt_groups']==1160 and result['triplets']['FULL']['partial_UNKNOWN_native_attempt_groups']==168 and result['triplets']['FULL']['physical_present_variant_rows']==3144 and result['HOLDOUT_known_synthetic_bases']==6 and result['HOLDOUT_real_known']==0
    checks['required_factual_limits_in_markdown']=all(token in md for token in ['382.95','4,000','이미 노출된','실사 개선','finite-end','UNKNOWN2,772','native4/6/7','activationtrace','두 팔 모두','목표는 미해결'])and'V199와는 recipe'in md
    checks['C_actual_fit_audit_exact_PASS_binding']=result['independent_actual_fit_audit']['status']=='PASS'and result['independent_actual_fit_audit']['binding']['sha256']=='7def791a9f153bc8491825a3af7406e1871718e97f9354a7411b2c536d3e168f'
    refs=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',md);parser=Links();parser.feed(html);refs+=parser.links;checked=[];future=[]
    for ref in refs:
        u=urlsplit(ref);assert not u.scheme and not u.netloc and not ref.startswith('/')and'v112'not in ref.casefold()
        target=V/unquote(u.path);assert '..'not in target.parts and target.is_relative_to(V)
        if target==H/'FACTUAL_QA.json'and not target.exists():future.append(target)
        else:assert safe(target).is_file()
        checked.append(ref)
    checks['portable_relative_MD_and_actualHTML_attribute_links_resolve']=True
    checks['no_external_dependency_or_filehome_link']='file://'not in md+html and '/home/'not in md+html and not any(urlsplit(x).scheme for x in checked)
    checks['not_adopted_not_solved_no_new_forward']=result['model_adopted']is False and result['fully_solved']is False and result['new_forward_fit_or_input_array_reads_for_report']==0
    assert all(checks.values()),checks
    qa=dict(completed=True,PASS=True,checks=checks,savedCSV_rows=len(rows),relative_link_occurrences=len(checked),deferred_self_output_links=[str(p)for p in future],actual_generated_chart_viewed_by_B=True,viewed_chart=binding(H/'SOURCE_DIRECT_CM_AND_PAIRED.png'),view_scope='One generated scientific scalar chart; no actual source RGB/crop or browser HTML render.',HTML_browser_rendered=False,source_input_models_weights_or_new_eval_reads=0,code=binding(__file__),report_results=binding(V/'ROOT_RESULTS.json'))
    with (H/'FACTUAL_QA.json').open('x')as f:json.dump(qa,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    for p in future:assert safe(p).is_file()
    ready=dict(completed=True,PASS=True,report_only=True,report_ROOT_artifacts={k:binding(V/k)for k in('PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html','ROOT_RESULTS.json')},figure={k:binding(H/k)for k in('SOURCE_DIRECT_CM_AND_PAIRED.png','SOURCE_DIRECT_CM_AND_PAIRED.svg')},builder=binding(H/'build_source_report.py'),build_complete=binding(H/'REPORT_BUILD_COMPLETE.json'),QA_code=binding(__file__),QA=binding(H/'FACTUAL_QA.json'),independent_actual_fit_audit=result['independent_actual_fit_audit']['binding'],source_evaluation_complete=binding(E/'COMPLETE.json'),scientific_phase_frozen=False,no_edits_after_READY_without_root_request=True)
    with (H/'FINAL_READY.json').open('x')as f:json.dump(ready,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(FINAL_READY=binding(H/'FINAL_READY.json'),**ready),ensure_ascii=False,indent=2))
if __name__=='__main__':run()
