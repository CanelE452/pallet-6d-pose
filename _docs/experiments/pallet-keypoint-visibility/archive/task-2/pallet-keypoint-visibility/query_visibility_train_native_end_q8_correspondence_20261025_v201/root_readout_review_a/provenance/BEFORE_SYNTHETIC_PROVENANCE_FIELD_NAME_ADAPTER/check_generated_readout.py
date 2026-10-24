"""V201 closed source-report QA only; no original score/pixel/tensor I/O."""
import hashlib, json, os, re, stat
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

V=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_train_native_end_q8_correspondence_20261025_v201')
H=V/'root_readout_review_a'

def safe(p):
    p=Path(p)
    if 'v112' in str(p).casefold():raise ValueError('Forbidden lexical path')
    p=Path(os.path.abspath(p))
    for q in [p,*p.parents]:
        if stat.S_ISLNK(os.lstat(q).st_mode):raise ValueError('Symlink refused')
    return p

def read(p):return json.loads(safe(p).read_text())
def bind(p):
    p=safe(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links += [v for k,v in attrs if k in ('href','src') and v]

def main():
    census=read(V/'q8_provenance_a/CENSUS_RESULT.json')
    target=read(V/'target_correspondence_b/SUMMARY.json')
    root=read(V/'ROOT_RESULTS_KO.json')
    assert root['completed'] is True and root['PASS'] is True
    assert root['TRAIN_rows']==64160 and root['TRAIN_inputU']==1145
    assert root['populations']==census['populations']
    assert root['direct_TRAIN_triplets']==target['total_triplets']==109
    assert root['known_weak_labels']==target['declared_known_weak_labels']==327
    assert root['target_domain_summary']==target['domain_summary']
    for flag in ('original_q8_is_actual_initializer_not_GT_projection','human_DE_authority_preserved','original_metadata_includes_HOLD_role_rows_but_TRAIN_only_aggregates'):
        assert root[flag] is True
    assert root['new_fit_forward_initializer_SAM_PnP']==0 and root['adopted'] is False and root['fully_solved'] is False
    assert root['new_model_architecture_or_threshold_selected'] is False
    for b in root['inputs'].values():assert bind(b['path'])['sha256']==b['sha256']
    md=safe(V/'ROOT_REPORT_KO.md').read_text();portable=safe(V/'PORTABLE_REPORT_KO.md').read_text()
    assert md==portable
    page=safe(V/'ROOT_READOUT_KO.html').read_text()
    for tag,m in root['status_metrics_reused'].items():
        assert (m['D_N'],m['E_N'],m['N'])==(1147,90,1237)
        cm=m['CM_D_E_by_D_E_U'];assert sum(cm[0])==1147 and sum(cm[1])==90
        assert abs(m['D_recall']-cm[0][0]/1147)<1e-12
        assert abs(m['E_recall']-cm[1][1]/90)<1e-12
        assert f'| {tag} | {m["D_recall"]*100:.2f}% | {m["E_recall"]*100:.2f}% | {m["E_precision"]*100:.2f}% | {m["U"]} / {m["U_rate"]*100:.2f}% |' in md
        assert '<tr><td>'+tag+'</td>'+''.join(f'<td>{m[x]*100:.2f}%</td>' for x in ('D_recall','E_recall','E_precision'))+'</tr>' in page
    for r in census['populations']:
        assert f'| {r["population"]} | {r["TRAIN_rows"]:,} | {r["TRAIN_inputU"]:,} | {r["context_groups_TRAIN"]} |' in md
    for key,label in [('SYNTH_FINITE_END','합성 finite-end'),('REAL_SEMANTIC_END_WEAK','공개 실사 weak end')]:
        d=target['domain_summary'][key]
        assert f'| {label} | {d["triplets"]} | {d["unique_source_groups"]} | {d["query_inside_any_candidate"]} | {d["region_distance_crop_px"]["median"]:.2f} px |' in md
    links=[]
    for filename in ('ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html'):
        text=safe(V/filename).read_text()
        assert '/home/' not in text and 'file://' not in text
        if filename.endswith('.html'):
            parser=Links();parser.feed(text);refs=parser.links
        else:refs=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
        for ref in refs:
            u=urlsplit(ref)
            if u.scheme or u.netloc:continue
            part=unquote(u.path)
            if not part:continue
            assert not part.startswith('/')
            p=safe(V/part);assert p.is_file() or p.is_dir()
            links.append(dict(page=filename,reference=ref,target=str(p)))
    output=dict(completed=True,PASS=True,checks=dict(source_population_rows_U_and_context_counts_exact=True,all109_source_domain_rows_and_spatial_table_cells_exact=True,old_status_MD_and_HTML_cells_match_reused_ROOT_JSON=True,reused_status_counts_and_recall_arithmetic_exact=True,all_ROOT_input_bindings_exact=True,MD_PORTABLE_byteexact=True,all_relative_links_exist_and_no_symlink=True),link_occurrences=len(links),links=links,root_documents=[bind(V/x) for x in ('ROOT_RESULTS_KO.json','ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html')],builder=bind(V/'build_root_readout.py'),code=bind(__file__),limits=['원 V200 score/ledger/metric body를 열지 않았다. 상태표는 이 ROOT JSON 재사용값과 문서 셀·분모·recall 산술의 일치를 검사했다.','이미지 디코드/새 view/browser render는 수행하지 않았다. 링크는 존재와 symlink 여부만 검사했다.','모든 과학적 해석의 참/거짓을 문자열 검사로 대체하지 않았다. 별도 사람용 정적검토 판단을 REVIEW.json에 남긴다.'])
    p=H/'GENERATED_DOCUMENT_CHECKS.json'
    with p.open('x') as f:json.dump(output,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(completed=True,PASS=True,link_occurrences=len(links),checks=bind(p))))

if __name__=='__main__':main()
