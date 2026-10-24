"""Final V220 text/link checks against completed independent scalar QA. No data replay."""
import csv,hashlib,io,json,math,os,re,stat
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit

V=Path(__file__).parent.parent
O=Path(__file__).parent
FILES={V/'ROOT_REPORT_KO.md':'981718fb279a39cfbbde0aae32a50e42be49ab1943f77bce5afa403ba945fc75',V/'PORTABLE_REPORT_KO.md':'981718fb279a39cfbbde0aae32a50e42be49ab1943f77bce5afa403ba945fc75',V/'ROOT_READOUT_KO.html':'3a752cf2806be59aa0b69be2e0e584f08443a10de8eac6150bfccb9fd2fb55e3',V/'ROOT_RESULT.json':'c720acca1eb23d3087c410c3065ec3689ea7165b2e732385ac53c9e37f06eeb5',O/'actual_review_v2/REVIEW.json':'6ba6f7519f13848b72070d19181bf395dc521de51588d8e838a013fcdf04326e',V/'saved_real_head_evaluation_v1/current1784/MANUAL_NEAR_D.csv':'859d449f427c003579c4c0628f782283c00435bdcafc48f7f3e32899584b83c7'}

def parentfd(p):
    assert p.is_absolute()and '..'not in p.parts and 'v112'not in str(p).lower()
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        return fd
    except BaseException:os.close(fd);raise
def read(p):
    assert p in FILES and p.suffix in('.json','.csv','.md','.html')
    fd=parentfd(p)
    try:leaf=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    with os.fdopen(leaf,'rb')as f:assert stat.S_ISREG(os.fstat(f.fileno()).st_mode);b=f.read()
    assert hashlib.sha256(b).hexdigest()==FILES[p];return b.decode()
def binding(p):return dict(path=str(p),sha256=FILES[p])
class Page(HTMLParser):
    def __init__(self):super().__init__();self.rows=[];self.row=None;self.cell=None;self.links=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='tr':self.row=[]
        elif tag in('td','th'):self.cell=''
        elif tag in('img','a'):self.links.append(a.get('src',a.get('href')))
    def handle_data(self,data):
        if self.cell is not None:self.cell+=data
    def handle_endtag(self,tag):
        if tag in('td','th')and self.cell is not None:self.row.append(self.cell.strip());self.cell=None
        elif tag=='tr':self.rows.append(self.row);self.row=None
def exists_link(base,ref):
    assert 'v112'not in ref.lower();u=urlsplit(ref);assert not u.scheme and not u.netloc
    s=unquote(u.path);assert s and not s.startswith('/')
    p=Path(os.path.normpath(str(base.parent/s)));assert p.is_relative_to(V.parent)
    fd=parentfd(p)
    try:z=os.stat(p.name,dir_fd=fd,follow_symlinks=False)
    finally:os.close(fd)
    assert stat.S_ISREG(z.st_mode),'missing or symlink link '+str(p)
    return str(p)
def check_primary(cells,m):
    assert json.loads(cells[0])==m['CM_D_E_by_D_E_U']
    assert cells[1]==' / '.join(format(m[k],'.2%')for k in('D_precision','D_recall'))
    assert cells[2]==' / '.join(format(m[k],'.2%')for k in('E_precision','E_recall'))
    assert cells[3]==format(m['U_rate'],'.2%')

def main():
    raw={p:read(p)for p in FILES};qa=json.loads(raw[O/'actual_review_v2/REVIEW.json']);root=json.loads(raw[V/'ROOT_RESULT.json']);md=raw[V/'ROOT_REPORT_KO.md']
    assert qa['completed']and qa['PASS']and root['completed']and root['PASS']and md==raw[V/'PORTABLE_REPORT_KO.md']
    assert root['primary_N']==1237 and root['primary_D']==1147 and root['primary_E']==90 and root['current_models16']==16 and root['public_models15']==15
    assert root['saved_evaluation_complete']==qa['evaluation_complete']and root['paired_prediction_seal']==qa['prediction_seal']and root['old14_metrics_exact']
    assert root['numeric_summary']==qa['root_numeric_summary']and root['source_training_updates_this_phase']==root['PnP_runs']==0
    assert root['current_input_U']==45 and root['public_input_U']==108 and root['source_model_parameters_each']==118359 and root['source_model_updates_each']==4000
    assert root['SELF_all136_paired21']and not root['trained_SELF_classifier']and root['repeated_exposed_not_fresh_blind']
    assert root['selected_model']is None and not root['adopted']and not root['fully_solved']and not root['browser_render_QA']
    assert root['gallery_complete']['sha256']=='1ebb6854732bf07f0bfb73f63ba1bf0fc1945e0452f2cc9fa714564f7f0c589d'
    assert root['root_visual_review']['sha256']=='0a4abde44e31744fe49c09fec2cf74ad6eeba32d8c07d9cf713a9b7484d4f0d7'
    for b in root['report_bindings'].values():assert FILES[Path(b['path'])]==b['sha256']
    for r in root['main_metrics']:
        m=qa['primary']['ALL'][r['model']];c=m['CM_D_E_by_D_E_U']
        assert [r[k]for k in('D_D','D_E','D_U','E_D','E_E','E_U')]==c[0]+c[1] and r['correct']==c[0][0]+c[1][1]
        for k in('D_precision','D_recall','E_precision','E_recall','U_rate'):assert math.isclose(r[k],m[k],rel_tol=0,abs_tol=1e-12)
    checked={'primary':0,'material':0,'SELF':0,'LE3':0}
    for line in md.splitlines():
        if not line.startswith('|'):continue
        cells=[x.strip()for x in line.split('|')[1:-1]]
        if len(cells)==5 and cells[0]in qa['models']:
            check_primary(cells[1:],qa['primary']['ALL'][cells[0]]);checked['primary']+=1
        elif len(cells)==6 and cells[0]in('WOOD_PRIMARY105','SQUARE118'):
            check_primary(cells[2:],qa['primary'][cells[0]][cells[1]]);checked['material']+=1
        elif len(cells)==3 and cells[0]in qa['models']:
            model=cells[0]
            if '%'in cells[-1]:
                m=qa['all_humanD_existing_reference_LE3_association'][model]
                assert cells[1]==f"{m['D']} / {m['binary_E']} / {m['U']}"and cells[2]==format(m['D']/m['N'],'.2%');checked['LE3']+=1
            else:
                a=qa['self_all136_primary_response']['ALL'][model];b=qa['self_response']['PAIRED_SELF21'][model]
                assert cells[1]==f"{a['D']} / {a['binary_E']} / {a['U']}"and cells[2]==f"{b['D']} / {b['binary_E']} / {b['U']}";checked['SELF']+=1
    assert checked==dict(primary=6,material=12,SELF=6,LE3=6)
    manual=list(csv.DictReader(io.StringIO(raw[V/'saved_real_head_evaluation_v1/current1784/MANUAL_NEAR_D.csv'])))
    aggregate={}
    for model in('C63','C65'):
        rr=[r for r in manual if r['model']==model];assert len(rr)==2 and sum(int(r['N'])for r in rr)==376
        aggregate[model]={k:sum(int(r[k])for r in rr)for k in('pred_visible','pred_binary_hidden','unresolved')}
    assert aggregate['C63']==dict(pred_visible=286,pred_binary_hidden=90,unresolved=0)and aggregate['C65']==dict(pred_visible=307,pred_binary_hidden=69,unresolved=0)
    wood={r['model']:int(r['pred_binary_hidden'])for r in manual if r['group']=='WOOD_PRIMARY105'};square={r['model']:int(r['pred_binary_hidden'])for r in manual if r['group']=='SQUARE118'}
    assert (wood['C63'],wood['C65'],square['C63'],square['C65'])==(18,17,72,52)
    for phrase in['문제는 아직 해결되지 않았습니다','이번 약한 손실 추가의 직접 비교는 C64↔C65','D 377개를 회복하고 58개','E는 2개를 회복하고 23개','단일 sequence의 AI 약한 center-owner/crop-presence','native 외부/자기 가림 gold가 아닙니다','known center-owner204/UNKNOWN156','입력 U108','legacy에는 과거 CAL 중복','학습 q8 자체도 GT 유래','가림 해결로 해석할 수 없습니다','새 모델 U=0은 유한한 출력을 이진으로 강제한 결과','전체 human SELF136','paired SELF subset은21','수동 click 근접 D376','오삭제90개→C65의69개','목재18→17개, 정사각형72→52개']:
        assert phrase in md,phrase
    page=Page();page.feed(raw[V/'ROOT_READOUT_KO.html']);assert len(page.rows)==7
    for row,model in zip(page.rows[1:],qa['models']):
        assert row[0]==model;m=qa['primary']['ALL'][model];assert json.loads(row[1])==m['CM_D_E_by_D_E_U']
        assert row[2:]==[format(m[k],'.2%')for k in('D_recall','E_recall','E_precision','U_rate')]
    link_counts={};targets=[]
    for name in('ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html'):
        base=V/name;links=page.links if name.endswith('.html')else re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',raw[base]);link_counts[name]=len(links)
        targets.extend(exists_link(base,ref)for ref in links)
    result=dict(completed=True,PASS=True,final_reports={n:binding(V/n)for n in('ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html','ROOT_RESULT.json')},actual_scalar_QA=binding(O/'actual_review_v2/REVIEW.json'),Markdown_table_rows_exact=checked,HTML_primary_rows_exact=6,manual_near376_saved_table=aggregate,manual_membership_not_independently_reconstructed=True,relative_link_counts=link_counts,relative_targets=targets,public_weak_scope='Only text/coverage204known156unknown/inputU108 matched the producer-declared protocol/COMPLETE authority; no public weak score-body numerical QA was performed by B.',visual_scope='Root actual12 gallery complete/visual receipt bindings checked against supplied hashes; B did not open gallery/PNG bodies or independently re-view illustrative cases.',L_scope='Truth CSV has no roi_side column; original L preservation relies on predictor RAW10/raw8 fingerprints and evaluator sameqL guard.',no_current_scalar_reduction_repeat=True,B_image_NPZ_PT_GT3d_or_model_access=0,browser_rendered=False,report_edits=0)
    p=O/'FINAL_REPORT_TEXT_AND_LINK_QA.json'
    with p.open('x')as f:json.dump(result,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
    print(json.dumps(dict(PASS=True,QA=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),tables=checked,links=link_counts)))

if __name__=='__main__':main()
