"""Closed V195 report/metadata/link QA only. No model, arrays, or source pixels."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import csv, hashlib, io, json, os, re, stat

HERE=Path(__file__).absolute().parent
V=HERE.parent
P=V.parent
MODELS=['v61','legacy','C52','C53','C54','C55','C56','C57']

def safe(path):
    raw=str(path)
    assert 'v112' not in raw.casefold(), 'Forbidden lexical path before access'
    p=Path(os.path.abspath(raw))
    assert P in p.parents and '..' not in p.parts
    return p

def exists(path):
    p=safe(path)
    cur=Path(p.anchor)
    for name in p.parts[1:]:
        cur=cur/name
        s=os.lstat(cur)
        assert not stat.S_ISLNK(s.st_mode), 'No symlink path component'
    assert stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)
    return p

def data(path):
    p=exists(path)
    assert p.suffix.lower() in {'.json','.md','.html','.csv','.py','.diff'}, 'Metadata/text only'
    with p.open('rb') as f:return f.read()

def load(path):return json.loads(data(path))
def text(path):return data(path).decode('utf-8')
def binding(path):
    p=exists(path)
    assert p.suffix.lower() not in {'.npz','.npy','.pth','.pt','.pkl'}, 'No array/weight access'
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return {'path':str(p),'sha256':h.hexdigest()}
def verify(b):
    assert binding(b['path'])==b, 'SHA binding mismatch '+b['path']
    return b
def write(name,d):
    with (HERE/name).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

class Page(HTMLParser):
    def __init__(self):
        super().__init__();self.refs=[];self.tables=[];self.table=None;self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        self.refs += [v for k,v in attrs if k in ('href','src') and v]
        if tag=='table':self.table=[]
        elif tag=='tr' and self.table is not None:self.row=[]
        elif tag in ('td','th') and self.row is not None:self.cell=''
    def handle_data(self,s):
        if self.cell is not None:self.cell+=s
    def handle_endtag(self,tag):
        if tag in ('td','th') and self.cell is not None:self.row.append(self.cell.strip());self.cell=None
        elif tag=='tr' and self.row is not None:self.table.append(self.row);self.row=None
        elif tag=='table':self.tables.append(self.table);self.table=None

def mdtables(s):
    tables=[];rows=[]
    for line in s.splitlines()+['']:
        if line.startswith('|') and line.endswith('|'):
            cells=[x.strip() for x in line.strip('|').split('|')]
            if not all(re.fullmatch(':?-+:?',x) for x in cells):rows.append(cells)
        elif rows:tables.append(rows);rows=[]
    return tables

def links(path,html=False):
    p=exists(path);s=text(p)
    assert '/home/' not in s and 'file://' not in s
    if html:q=Page();q.feed(s);refs=q.refs
    else:refs=re.findall(r'\]\(([^)]+)\)',s)
    checked=[]
    for r in refs:
        assert 'v112' not in r.casefold()
        u=urlsplit(r)
        if u.scheme or u.netloc:
            assert u.scheme in ('https','http','mailto');continue
        rel=unquote(u.path)
        assert not rel.startswith('/')
        if rel:exists(p.parent/rel)
        checked.append(r)
    return {'path':str(p),'local_references':len(checked),'all_exist_no_symlinks':True}

def pct(x):return '미정' if x is None else f'{100*x:.1f}%'
def metricrow(r,material=False):
    values=[r['model']]
    if material:values+=['WOOD' if r['group']=='WOOD_PRIMARY105' else 'SQUARE']
    return list(map(str,values+[r['N'],[r['D_D'],r['D_E'],r['D_U']],[r['E_D'],r['E_E'],r['E_U']],pct(r['D_precision']),pct(r['D_recall']),pct(r['E_precision']),pct(r['E_recall']),r['U'],pct(r['U']/r['N'])]))

def main():
    r=load(V/'ROOT_RESULTS_KO.json');phase=load(V/'ROOT_PHASE_RESULT.json')
    s=load(V/'fixed_direct_end_real_visual_v1/SUMMARY_AND_TABLES.json')
    a=load(V/'independent_saved_result_review_preparation_a/actual_review_v1/REVIEW.json')
    g=load(V/'fixed_direct_end_real_visual_v1/COMPLETE.json')
    ev=load(V/'saved_real_head_evaluation_v1/COMPLETE.json')
    proto=load(V/'report_preparation_b/PROTOCOL_BEFORE_FINAL_REPORT_METADATA_AND_WRITES.json')
    assert a['PASS'] and a['completed'] and a['primary1237_D1147_E90']
    assert a['all_existing_tables']['every_table_CSV_JSON_cell']
    assert r['models']==MODELS and r['table_all8_material24']==s['current_primary']['rows']
    assert r['table_primary8']==[x for x in s['current_primary']['rows'] if x['group']=='ALL']
    assert len(r['table_primary8'])==8 and len(r['table_all8_material24'])==24
    for x in r['table_all8_material24']:
        q=a['current']['primary'][x['group']][x['model']]
        matrix=[[x['D_D'],x['D_E'],x['D_U']],[x['E_D'],x['E_E'],x['E_U']]]
        assert matrix==q['CM_D_E_by_D_E_U'] and x['N']==q['N']
        assert x['D_support']==q['D_N'] and x['E_support']==q['E_N']
        assert all(x[k]==q[k] for k in ('D_precision','D_recall','E_precision','E_recall','U'))
    assert r['current_all1784']==s['current_all1784'] and a['current']['original_input_U']==45
    assert r['original_current_input_U_rate']==45/1784
    assert all(r['primary_U_rate_by_model'][x['model']]==x['U']/x['N'] for x in r['table_primary8'])
    for out,cohort,n in [('self_tables_paired21','paired_same_query_RGB',21),('self_tables_same_slot136','same_slot_pipeline',136)]:
        for x in r[out]:
            z=[v for v in s['self_separate'] if v['model']==x['모델'] and v['cohort']==cohort]
            assert [x[k] for k in ('별도 human S N','D 응답','binary hidden 응답','미결정')]==[sum(v[k] for v in z) for k in ('human_S_support','visible_response','binary_hidden_response','unresolved')]
            assert x['별도 human S N']==n
    assert 'D_vs_ES' in r['self_membership_authority'] and a['self_all136_paired21_separate']
    for x in r['reference_error_LE3_humanD461']:
        q=a['current']['all_registered_humanD_existing_reference_LE3_association'][x['모델']]
        assert (x['기존reference≤3 humanD N'],x['D'],x['E'],x['U'])==(q['N'],q['D'],q['E'],q['U'])
    assert r['manual_near_D_distinct']=={'N':376,'WOOD':153,'SQUARE':223} and a['reference461_manual376_separate']
    assert r['public_weak_center_only']==s['public_weak_owner_only'] and r['table_public_weak7']==s['public_weak_headline_table']
    cov=r['public_weak_center_only']['coverage']
    assert [cov[k] for k in ('known_center204','unknown_center156','original_input_U','center4_invalid','v61_own_U')]==[204,156,108,112,109]
    assert a['public']['weak_known']==204 and a['public']['weak_UNKNOWN']==156 and a['public_weak_only']
    assert len(r['native8_material_full_denominators'])==16 and sum(x['all_original_slots'] for x in r['native8_material_full_denominators'])==1784
    assert {x['native_slot'] for x in r['native8_material_full_denominators']}==set(range(8))
    for filename,expected in [('PRIMARY_ALL8_CM_PRECISION_RECALL_U.csv',r['table_all8_material24']),('ALL_ORIGINAL_NATIVE_MATERIAL_DENOMINATORS.csv',r['native8_material_full_denominators'])]:
        rows=list(csv.DictReader(io.StringIO(text(V/'fixed_direct_end_real_visual_v1'/filename))))
        assert len(rows)==len(expected)
        for row,x in zip(rows,expected):
            assert set(row)==set(x)
            for k,v in x.items():assert row[k]==('' if v is None else str(v))
    md=text(V/'ROOT_REPORT_KO.md');portable=text(V/'PORTABLE_REPORT_KO.md');h=text(V/'ROOT_READOUT_KO.html')
    assert md==portable
    expected=[
        [['모델','N','D행 [D,E,U]','E행 [D,E,U]','D 정밀도','D 재현율','E 정밀도','E 재현율','U','U율']]+[metricrow(x) for x in r['table_primary8']],
        [['모델','재질','N','D행 [D,E,U]','E행 [D,E,U]','D 정밀도','D 재현율','E 정밀도','E 재현율','U','U율']]+[metricrow(x,True) for x in r['table_new_C56_C57_material4']],
    ]
    for field in ('self_tables_paired21','self_tables_same_slot136','reference_error_LE3_humanD461'):
        rows=r[field];expected.append([list(rows[0])]+[[str(x[k]) for k in rows[0]] for x in rows])
    expected.append([['모델','T 정밀도','T 재현율','non-T 재현율','ownerF 정밀도','ownerF 재현율']]+[[x['model']]+[pct(x[k]) for k in ('T_precision_known','T_recall_including_U','nonT_recall_including_U','ownerF_precision_known','ownerF_recall_including_U')] for x in r['table_public_weak7']])
    htmlp=Page();htmlp.feed(h)
    assert mdtables(md)==expected==htmlp.tables, 'Every displayed MD/HTML cell'
    local=[links(V/f,html=f.endswith('.html')) for f in ['ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html','fixed_direct_end_real_visual_v1/PORTABLE_GALLERY_KO.md','fixed_direct_end_real_visual_v1/index.html']]
    # Hash only bounded closed metadata/code; never open evidence NPZ/weights or old raw data.
    for name,b in r['evidence'].items():
        if name!='source_fit_integrity':verify(b)
    assert phase['evidence']==r['evidence'] and phase['report_code']==r['report_code'] and phase['report_protocol']==r['report_protocol']
    for b in phase['reports'].values():verify(b)
    for b in (r['report_code'],r['report_protocol'],r['entry'],proto['evaluation_code'],proto['evaluation_protocol'],proto['visual_code'],proto['visual_protocol']):verify(b)
    assert ev['code']==proto['evaluation_code'] and ev['protocol']==proto['evaluation_protocol']
    assert g['code']==proto['visual_code'] and g['protocol']==proto['visual_protocol']
    assert a['complete']==r['evidence']['evaluation_complete'] and ev['prediction_seal']==r['evidence']['prediction_seal']
    fail=load(V/'ROOT_ACTUAL_SAVED_EVALUATOR_SELF_COHORT_GUARD_FAILURE_PRESERVED.json')
    repair=load(V/'SELF_COHORT_GUARD_REPAIR_RECEIPT.json')
    term=load(V/'ROOT_ACTUAL_CORRECTED_SAVED_REAL_EVALUATION_TERMINAL.json')
    for b in fail['archived_files'].values():verify(b)
    for key in ('prior_FINAL_READY','preserved_failure','diff'):verify(repair[key])
    assert fail['exit_code']==1 and term['exit_code']==0 and term['predictions_recomputed'] is False
    assert fail['prediction_seal_unchanged']==ev['prediction_seal']
    assert repair['all_other_scientific_functions_AST_unchanged'] and repair['predictor_code_and_protocol_bytes_unchanged']
    for d in (r,phase):assert d['selected_model'] is None and d['adopted'] is False and d['fully_solved'] is False
    assert r['source_training']['source_only_includes_public_real'] and r['source_training']['multiple_changed_conditions_not_causal_isolation'] and r['source_training']['real_end_holdout0']
    assert r['actual_visual_scope']=={'root_viewed_count':6,'all68_or1784_actually_viewed':False,'browser_rendered':False}
    for literal in ('synthetic-only 학습이라고 부르지 않는다','단일 인과 효과를 분리하지 않는다','legacy의 과거 realCAL','fresh 독립 검증이 아니다','차이115','manual-click near376','native 물리 gold','대표성/유병률을 주장하지 않는다'):
        assert literal in md and literal in h
    sel=load(HERE/'SELECTION_BEFORE_IMAGE_OR_REPORT_READS.json');vb=load(HERE/'FIXED6_ACTUAL_VIEW_BINDINGS.json')
    assert sel['selected_original_indices']==[1,3,4,5,7,8] and set(sel['selected_original_indices']).isdisjoint(sel['excluded_root_original_indices'])
    verify(sel['source_selection']);verify(vb['manifest'])
    for x in vb['selected']:verify(x['card'])
    visual={'completed':True,'PASS':True,'selection':binding(HERE/'SELECTION_BEFORE_IMAGE_OR_REPORT_READS.json'),'actual_viewed_card_count':6,'actual_view_image_calls':6,'actual_viewed_cards':vb['selected'],'all68_or1784_images_viewed':False,'browser_rendered':False,'observations_KO':[
        '6개 모두 목재 사례이며 표·footer·clean96 thumbnail·별도 nearest display의 cyan q·valid mask가 읽혔다. 겹침이나 잘림을 확인하지 못했다.',
        '#1/#3/#4/#5는 회색 배경이 crop의 상당 부분을 차지하고 나무 끝/경계가 q 가까이에 있다. #7은 회색과 목재 수직 경계, #8은 나무 블록/상판이 더 크게 보인다.',
        '#5의 기존 reference 오차는 약1.82px지만 humanD에 C56/C57 E 응답이 보인다. 작은 참조 수치를 정확한 물리 좌표 gold나 답의 증명으로 해석하지 않는다.'
    ],'limits_KO':['고정 원순서의 비대표 6개 확인이며 SQUARE/SELF/UNKNOWN/외부가림 전체 coverage를 주장하지 않는다.','local crop만으로 전체 selected instance, 실제 가림의 원인, 물리 좌표오류, 예측의 인과 원인을 확정하지 않는다.','PNG 원96/valid bytes parity는 gallery COMPLETE의 기존 검사를 인용하며 이 검토에서 원 배열을 재디코드하지 않았다.'],'no_new_model_label_threshold_query_training_or_data_selection':True,'selected_model':None,'adopted':False,'fully_solved':False}
    write('ACTUAL_VISUAL_REVIEW.json',visual)
    review={'completed':True,'PASS':True,'code':binding(__file__),'reports':{f:binding(V/f) for f in ['ROOT_RESULTS_KO.json','ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html','ROOT_PHASE_RESULT.json']},'independent_numeric':r['evidence']['independent_numeric'],'gallery_complete':r['evidence']['gallery_complete'],'all8_primary_and24_group_material_rows_exact':True,'every_displayed_MD_HTML_table_cell_exact':True,'displayed_data_rows':sum(len(t)-1 for t in expected),'displayed_data_cells':sum(sum(len(row) for row in t[1:]) for t in expected),'primary_U_rates_and_original45_of1784_distinct':True,'native8_full_denominators16_rows1784':True,'self21_ES_and_full136_separate':True,'reference461_manual376_distinct':True,'public204_UNKNOWN156_U108_and_v61U109_distinct':True,'portable_links':local,'preserved_failure_and_repaired_code_protocol_bindings_exact':True,'source_only_includes_public_real_no_causal_isolation_or_adoption_claim':True,'actual_visual':binding(HERE/'ACTUAL_VISUAL_REVIEW.json'),'limits':['Closed report/table/metadata comparison, not a repeated original numerical evaluation.','Only six fixed additional cards actually viewed; all other image references checked for existence only.','Source-fit outside V195 is an inherited exact metadata binding; no source cache, weights, arrays, original labels or new model read.','No browser execution or HTML screenshot verification.'],'blockers':[],'selected_model':None,'adopted':False,'fully_solved':False,'actual_forward_fit_original_array_or_label_reads':0}
    write('REVIEW.json',review)
    print(json.dumps({'completed':True,'PASS':True,'review':binding(HERE/'REVIEW.json'),'visual':binding(HERE/'ACTUAL_VISUAL_REVIEW.json'),'displayed_rows':review['displayed_data_rows'],'displayed_cells':review['displayed_data_cells'],'local_refs':sum(x['local_references'] for x in local),'actual_views':6}))

if __name__=='__main__':main()
