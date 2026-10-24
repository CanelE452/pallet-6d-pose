"""Final text/link consistency only. Does not repeat saved prediction reduction."""
import hashlib
import json
import math
import os
import re
import stat
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit

D=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_expected_part_weak_paired_source_training_20261025_v219')
O=Path(__file__).parent
FILES={
 D/'ROOT_REPORT_KO.md':'f243b4c451c90021642db45e27e325b1b9f64638c1068b85cc489178cf71b856',
 D/'PORTABLE_REPORT_KO.md':'f243b4c451c90021642db45e27e325b1b9f64638c1068b85cc489178cf71b856',
 D/'ROOT_READOUT_KO.html':'135b249d6982f8b0e869eb1a62e56721687f6840b729a407080288bae222eceb',
 D/'ROOT_RESULT.json':'2c600a8f31deb5cc531560adbb6d923e52a8b12aff3cf041f46cf368c4d17119',
 O/'ACTUAL_SOURCE_NUMERICAL_QA.json':'08988707f473ce9534d506b4e1d54659162dd1c7d605ccd5b798f9176a82741d',
 D/'saved_source_result_readout_v1/SOURCE_METRICS_AND_PAIRED.json':'beb496bc07d9c3ebb72ce40ab8c9d5d204d001949336372866829eb0cbe2ff47',
 D/'saved_source_result_readout_v1/ALL57_PAIRED_SOURCE_ROWS.json':'66e82a69bdd49b2e9da25924153112771251937d5901117922c7210e26de1918',
 D/'ROOT_ACTUAL_SOURCE_CROP_VISUAL_REVIEW.json':'8c07eb4362800044de82a18671438bf202806f9c4b604adfbf01b66b7b0ffd43',
 D/'independent_actual_pair_QA_a/actual_review_v1/REVIEW.json':'f1fc06899e57d01254e15e2cd12fbc58ebc98e3f6311f6da375f117fedebd6b5',
}

def parent_fd(p):
    assert p.is_absolute() and 'v112' not in str(p).lower() and '..' not in p.parts
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        return fd
    except BaseException:os.close(fd);raise

def read(p):
    p=Path(p);assert p in FILES and p.suffix in ('.md','.html','.json')
    fd=parent_fd(p)
    try:leaf=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    with os.fdopen(leaf,'rb') as f:assert stat.S_ISREG(os.fstat(f.fileno()).st_mode);b=f.read()
    assert hashlib.sha256(b).hexdigest()==FILES[p]
    return b.decode()

class Page(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.text=[];self.articles=[];self.current=None
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag in ('a','img'):self.links.append(a.get('href',a.get('src')))
        if tag=='article':self.current=[]
    def handle_data(self,data):
        self.text.append(data)
        if self.current is not None:self.current.append(data)
    def handle_endtag(self,tag):
        if tag=='article':self.articles.append(' '.join(self.current));self.current=None

def linked_file(base,ref):
    assert 'v112' not in ref.lower()
    parts=urlsplit(ref);assert not parts.scheme and not parts.netloc
    s=unquote(parts.path)
    assert s and not s.startswith('/')
    p=Path(os.path.normpath(str(base.parent/s)))
    assert p.is_relative_to(D.parent)
    fd=parent_fd(p)
    try:info=os.stat(p.name,dir_fd=fd,follow_symlinks=False)
    finally:os.close(fd)
    assert stat.S_ISREG(info.st_mode),'missing or symlink local link: '+str(p)
    return str(p)

def main():
    text={p:read(p) for p in FILES}
    root=json.loads(text[D/'ROOT_RESULT.json']);qa=json.loads(text[O/'ACTUAL_SOURCE_NUMERICAL_QA.json'])
    metrics=json.loads(text[D/'saved_source_result_readout_v1/SOURCE_METRICS_AND_PAIRED.json'])
    visual=json.loads(text[D/'ROOT_ACTUAL_SOURCE_CROP_VISUAL_REVIEW.json']);pairqa=json.loads(text[D/'independent_actual_pair_QA_a/actual_review_v1/REVIEW.json'])
    paired={r['row_id']:r for r in json.loads(text[D/'saved_source_result_readout_v1/ALL57_PAIRED_SOURCE_ROWS.json'])['rows57']}
    assert root['completed'] and root['PASS'] and root['source_only'] and root['phase']=='V219'
    assert root['source_training_updates_each']==pairqa['log_checks']['updates_each']==4000 and root['source_training_updates_total']==pairqa['log_checks']['log_rows']==8000
    assert root['source_probe_disposable_updates']==64 and root['source_model_parameters_each']==118359
    for key,value in [('weak_known57',57),('weak_FIT45',45),('weak_DEV12',12),('weak_DEV_components',3),('weak_DEV_nominal7',7)]:assert root[key]==value
    assert root['paired_fit_complete']['sha256']==qa['fit_complete']['sha256'] and root['saved_source_reduction']['sha256']==qa['reducer_complete']['sha256']
    assert root['weak_DEV']==metrics['weak']['SOURCE_DEV/ALL_STATES'] and root['weak_DEV_nominal']==metrics['weak']['SOURCE_DEV/NOMINAL_S0']
    for key in ('weak_DEV','weak_DEV_nominal'):
        q=qa['weak']['SOURCE_DEV/'+('ALL_STATES' if key=='weak_DEV' else 'NOMINAL_S0')]
        assert root[key]['rows']==q['rows'] and root[key]['unique_components']==q['unique_components'] and root[key]['paired']==q['paired']
        for arm in q['models']:assert root[key]['models'][arm]['metrics']==q['models'][arm]
    assert root['GT_derived_training_query'] is True and root['protected_evaluation_access']==0 and root['new_private_evaluation'] is False
    assert root['latest_completed_real_benchmark']=='V213' and root['selected_arm'] is None and root['adopted'] is False and root['fully_solved'] is False and root['browser_render_QA'] is False
    assert 'NOT_PHYSICAL_NATIVE_POINT_GOLD' in root['labels']
    for b in root['report_bindings'].values():assert FILES[Path(b['path'])]==b['sha256'] and len(text[Path(b['path'])].encode())==b['size_bytes']
    assert text[D/'ROOT_REPORT_KO.md']==text[D/'PORTABLE_REPORT_KO.md']
    md=text[D/'ROOT_REPORT_KO.md'];rows=[line.split('|')[1:-1] for line in md.splitlines() if line.startswith('| SOURCE_')]
    assert len(rows)==6
    for row in rows:
        cohort,N=re.fullmatch(r'(.+) \((\d+)\)',row[0].strip()).groups();arm='STRONG_ONLY_CONTROL' if row[1].strip()=='C64' else 'STRONG_PLUS_EXPECTED_PART_WEAK'
        m=qa['weak'][cohort]['models'][arm];assert int(N)==m['rows'] and json.loads(row[2])==m['CM_D_E_by_D_E_U']
        for cell,cls in zip(row[3:5],m['classes']):assert cell.strip()==' / '.join(f'{cls[k]:.2%}' for k in ('precision','recall'))
        assert row[5].strip()=='0%' and m['model_U_on_known_truth']==0
    for phrase in ['303행','화면 밖 150','미해결 72','UNKNOWN 81','GT에서 만든 별도 학습용 q8','물리 키포인트 gold가 아닙니다','학습 자료 적합도','새 blind 평가가 아닙니다','현재 실사 비교의 최신 완료 기록은 V213','제외 구간의 바이트 통과','E recall은 2/3','4행을 회복하고 1행','2행 회복·1행 악화']:
        assert phrase in md,phrase
    assert pairqa['completed'] and pairqa['PASS'] and pairqa['all8000_saved_log_rows_verified'] and pairqa['old192_target_class_distribution_RNG_not_independently_rebuilt'] and pairqa['strong_logits_not_replayed_without_SOURCE_RGB']
    assert pairqa['source_metadata_rows']=={'SYNTH_AUG65536':7040,'PUBLIC_TRAIN1480':1208,'CF3816':3816}
    for arm in ('STRONG_ONLY_CONTROL','STRONG_PLUS_EXPECTED_PART_WEAK'):
        r=pairqa['results'][arm];assert r['parameters']==118359 and r['optimizer_all_steps4000'] and r['weak57_all_saved_signs_exact']
    assert '3.624e-5/C65 3.052e-5' in md and math.isclose(pairqa['results']['STRONG_ONLY_CONTROL']['weak57_CPU_replay_batch32_maxabs'],3.62396240234375e-05) and math.isclose(pairqa['results']['STRONG_PLUS_EXPECTED_PART_WEAK']['weak57_CPU_replay_batch32_maxabs'],3.0517578125e-05)
    page=Page();page.feed(text[D/'ROOT_READOUT_KO.html']);assert len(page.articles)==visual['images_viewed']==8
    nominal_dev={r['row_id'] for r in paired.values() if r['source_role']=='SOURCE_DEV' and r['state_index']==0}
    assert len(nominal_dev)==7
    ids=[]
    for article,record in zip(page.articles,visual['source_images']):
        rid=record['row_id'];ids.append(rid);assert record['saved_prediction']==paired[rid] and rid in article
        p=paired[rid];assert 'weak '+('E' if p['target'] else 'D') in article
        assert 'C64 '+('E' if p['predictions']['STRONG_ONLY_CONTROL'] else 'D')+' → C65 '+('E' if p['predictions']['STRONG_PLUS_EXPECTED_PART_WEAK'] else 'D') in article
    assert set(ids)==nominal_dev|{'F05_C000000_S2_N1'}
    assert visual['root_actual_view_image_tool'] and visual['PASS'] and visual['browser_render_QA'] is False and visual['new_image_model_inference'] is False
    link_counts={};targets=[]
    for name in ('ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html'):
        base=D/name
        links=page.links if name.endswith('.html') else re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text[base])
        link_counts[name]=len(links)
        targets.extend(linked_file(base,ref) for ref in links)
    output=dict(completed=True,PASS=True,report_bindings={p.name:dict(path=str(p),sha256=s) for p,s in FILES.items() if p.parent==D and p.name in ('ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html','ROOT_RESULT.json')},numerical_QA=dict(path=str(O/'ACTUAL_SOURCE_NUMERICAL_QA.json'),sha256=FILES[O/'ACTUAL_SOURCE_NUMERICAL_QA.json']),report_JSON_full_DEV_strata_exact=True,Markdown_table_rows_exact=6,HTML_saved_case_cards_exact=8,link_counts=link_counts,relative_link_targets=targets,root_image_views_attested_by_receipt=8,B_actual_image_views=0,B_image_body_or_NPZ_PT_GT_protected_reads=0,browser_render_performed=False,source_output_data_reduction_repeated=False,limits='SOURCE-only expected-part weak labels; dependent preexposed DEV12 from3components/nominal7; not physical native-point gold or real benchmark improvement. A replay limits preserved.')
    out=O/'FINAL_REPORT_TEXT_AND_LINK_QA.json'
    with out.open('x') as f:json.dump(output,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n')
    print(json.dumps(dict(PASS=True,QA=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),links=link_counts)))

if __name__=='__main__':main()
