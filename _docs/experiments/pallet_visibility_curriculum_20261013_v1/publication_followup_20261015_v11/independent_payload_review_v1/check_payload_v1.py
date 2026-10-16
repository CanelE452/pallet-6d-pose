"""Read-only final V11 publication verification, not inference or packaging."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from html.parser import HTMLParser
from urllib.parse import unquote
import ast
import hashlib
import json
import re
import subprocess

ROOT=Path('/home/minjae/Documents/github/pallet-pose')
D=Path('_docs/experiments/pallet_visibility_curriculum_20261013_v1')
PUB=D/'publication_followup_20261015_v11'
B=D/'nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
OUT=ROOT/PUB/'independent_payload_review_v1'
WT=Path('/tmp/pallet-visibility-main-publication-20261015-v1')
M=PUB/'CLOSED_PAYLOAD_MANIFEST_V1.json'
V=PUB/'PAYLOAD_VALIDATION_V1.json'
EXPECTED_M='5107c0cfb4cde62a4983bcc385a7047cd237531cbff001ba157bc846352fac5b'
EXPECTED_V='144d7a357dd1e4e8439a9685abe2c32bb4768ec2b42473f73beac18e45c8dade'

def sha(path):
    p=Path(path);p=p if p.is_absolute() else ROOT/p
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads((ROOT/path).read_text())

class Links(HTMLParser):
    def __init__(self):super().__init__();self.targets=[]
    def handle_starttag(self,tag,attrs):
        self.targets.extend(v for k,v in attrs if k in ('src','href') and v)

def main():
    if (OUT/'REVIEW.json').exists():raise FileExistsError('Preserve independent review')
    checks=[]
    def check(name,value):checks.append(dict(name=name,passed=bool(value)))
    check('pinned_manifest',sha(M)==EXPECTED_M);check('pinned_validation',sha(V)==EXPECTED_V)
    m=read(M);v=read(V);rows=m['files'];names=[r['path'] for r in rows];selected=set(names)|{str(M),str(V)}
    listed=(ROOT/PUB/'CLOSED_FILES_TO_PUBLISH_V1.txt').read_text().splitlines()
    check('509_unique_publish_paths',len(listed)==len(set(listed))==509 and set(listed)==selected)
    check('507_manifest_rows',len(names)==len(set(names))==m['count']==507)
    check('payload_byte_total',sum(r['bytes'] for r in rows)==m['bytes']==91670780)
    check('owner_zero_failures',v['failure_count']==0 and not v['failures'] and v['manifest_sha256']==EXPECTED_M)
    for row in rows:
        p=ROOT/row['path']
        check('safe_path:'+row['path'],not Path(row['path']).is_absolute() and '..' not in Path(row['path']).parts
              and p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(ROOT))
        check('bytes:'+row['path'],p.stat().st_size==row['bytes'] and p.stat().st_size<100*1024*1024)
        check('sha:'+row['path'],sha(p)==row['sha256'])
    protected=m['protected_user_sha256']
    check('protected6_exact_population',len(protected)==6)
    for name,digest in protected.items():
        check('protected_excluded:'+name,name not in selected)
        check('protected_bytes_unchanged:'+name,sha(name)==digest)
    actual_images={name for name in names if Path(name).suffix.lower() in ('.png','.jpg','.jpeg')}
    actual_HTML={name for name in names if Path(name).suffix=='.html'}
    check('132_images_and12HTML',len(actual_images)==132 and len(actual_HTML)==12)
    for i in range(10):
        for branch in ('RAW_YOLO','N3_SUBPIX'):
            for suffix in ('01_initial','02_states','03_measurements','04_final'):
                check('80_BI_tiles:'+str(i)+branch+suffix,str(B/'new_gain10_paired_BI_process_v1'/f'case{i:02d}'/(branch+'_'+suffix+'.png')) in selected)
        check('old_gain10_process:'+str(i),str(B/'raw_yolo_N3_subpix_gain10_process_v1'/f'case{i:02d}'/'process.png') in selected)
    display=B/'largest_component_global_outline_process_v4_33_v1/display_v2'
    check('latest_V33_MD',str(display/'REPORT_KO.md') in selected)
    for i in range(3):
        for kind in ('observations','final_and_reference'):
            check('latest_V33_PNG:'+str(i)+kind,str(display/f'case{i}_{kind}_ROI.png') in selected)
    raw={str(B/'raw_ns_BI_gain10_input_v1/INPUTS.json'),str(B/'new_gain10_paired_BI_v2/PREDICTIONS.jsonl.gz'),str(B/'new_gain10_paired_BI_v2/OBSERVATIONS.jsonl.gz')}
    raw|={str(B/'new_gain10_paired_BI_v2/masks'/f'{i:02d}_{a}.npz') for i in range(10) for a in ('RAW_YOLO','N3_SUBPIX')}
    check('23_exact_authorized_raw_exceptions',len(raw)==23 and set(m['raw_exceptions'])==raw and raw<=selected)
    for name in names:
        parts=Path(name).parts
        check('no_spool_or_frames:'+name,not any(k in parts for k in ('frames','row_spool','prediction_spool','rows','rows_REST','__pycache__')))
        if Path(name).suffix in ('.npz','.npy') or Path(name).name in ('INPUTS.jsonl.gz','E_COMPARISON.jsonl.gz','PREDICTIONS.jsonl.gz','OBSERVATIONS.jsonl.gz','OUTLINE_OBSERVATIONS.jsonl.gz','CAL_OUTLINE_OBSERVATIONS.jsonl.gz','REST_OUTLINE_OBSERVATIONS.jsonl.gz'):
            check('raw_allowed_only_fixed10:'+name,name in raw)
        check('no_live_inventory:'+name,'repro_inventory_' not in name and Path(name).name not in ('LIVE_PROGRESS.json','CAL_LIVE_PROGRESS.json','REST_LIVE_PROGRESS.json'))
    tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=WT).decode().split('\0'))
    source=m['source_publication_status'];supplement=0;reused=0
    for row in source['rows']:
        name=row['path'];check('source_current:'+name,sha(name)==row['sha256'])
        if row['supplement_current_source']:
            supplement+=1;check('supplement_selected:'+name,name in selected)
        else:
            reused+=1
            check('HEAD_reuse_tracked_and_WT_exact:'+name,name in tracked and (WT/name).is_file() and sha(WT/name)==row['sha256'] and row['published_HEAD_sha256']==row['sha256'])
    check('333_source_310reuse23new',len(source['rows'])==333 and reused==310 and supplement==23)
    link_records=[];AST=0
    for name in names:
        p=ROOT/name
        if p.suffix=='.py':
            try:ast.parse(p.read_text());AST+=1
            except SyntaxError:check('AST:'+name,False)
        if p.suffix not in ('.md','.html'):continue
        text=p.read_text()
        if p.suffix=='.html':
            parser=Links();parser.feed(text);targets=parser.targets
        else:targets=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
        for target in targets:
            if target.startswith(('#','data:','mailto:')) or '://' in target:continue
            q=(p.parent/unquote(target.split('#')[0])).resolve();safe=q.is_relative_to(ROOT)
            relative=str(q.relative_to(ROOT)) if safe else str(q)
            covered=safe and (relative in selected or relative in tracked and q.is_file() and (WT/relative).is_file() and sha(q)==sha(WT/relative))
            check('local_link:'+name+':'+target,covered)
            link_records.append(dict(source=name,target=target,covered=covered))
    check('59_AST_and723links',AST==59 and len(link_records)==723)
    report=(ROOT/PUB/'REPORT_KO.md').read_text();gallery=(ROOT/PUB/'CASE_GALLERY_KO_V1.md').read_text()
    check('main_report_exact',sha(PUB/'REPORT_KO.md')=='ad4b27567d49d875f29be077bb31fea96bad63c13b67d97adb5cd925d387786c')
    check('MD_gallery_exact',sha(PUB/'CASE_GALLERY_KO_V1.md')=='d28cfca41803d69effea245ea466e3254fe885b891ae4d57ad5a054754aef34e')
    check('MD_gallery_all80_images',len(re.findall(r'!\[',gallery))==80)
    check('GitHub_static_usage_caveat',all(x in report for x in ('HTML 원문','Markdown·PNG','localhost 주소를 공개 URL로 사용하지 않는다')))
    check('posthoc10_and_no_full_population_claim',all(x in report for x in ('전체49/192의 새 성능이 아니다','사후 선택','raw 팔의 state calibration coverage','7/10·10/10','기본 e를 유지')))
    score=read(B/'new_gain10_paired_BI_v2/SCORE.json');summary=score['summary']
    check('closed_subset_0_10_7_10',[summary[a]['joint_successes'] for a in ('RAW_YOLO_F','N3_SUBPIX_F','RAW_YOLO_BI_ONLY_FRESH_PNP','N3_SUBPIX_BI_ONLY_FRESH_PNP')]==[0,10,7,10])
    check('before_after_report_values',all(x in report for x in ('6.4614','3.9925','3.4082','2.1831','5.3621','3.5690','RAW44점','N3 45점')))
    check('preserved_baseline_v10_report',str(D/'publication_followup_20261015_v10/REPORT_KO_FINAL_V1.md') in selected and sha(D/'publication_followup_20261015_v10/REPORT_KO_FINAL_V1.md')=='d97212bb5f0a070ef8230aef6046c2ada3a4b35637e425ca2f196964b9044b9b')
    check('final_immutable_manifest',sha(M)==EXPECTED_M and sha(V)==EXPECTED_V)
    failures=[x for x in checks if not x['passed']]
    result=dict(schema='INDEPENDENT_V11_CLOSED_PUBLICATION_PAYLOAD_REVIEW_V1',status='PASS' if not failures else 'FAIL',UTC=datetime.now(timezone.utc).isoformat(),checks=len(checks),failure_count=len(failures),failures=failures,
        manifest_sha256=EXPECTED_M,validation_sha256=EXPECTED_V,paths=509,manifest_rows=507,manifest_bytes=m['bytes'],image_files=len(actual_images),HTML_files=len(actual_HTML),source_AST_files=AST,local_links=len(link_records),maximum_file_bytes=max(r['bytes'] for r in rows),source_closure=dict(current=333,existing_HEAD_reuse=reused,supplemental=supplement),raw_exception_count=len(raw),protected6_excluded_and_unchanged=True,
        manual_report_review='Final V10 report/latest V33 display/old gain10/new paired BI report reviewed. Subset7/10vs10/10, one raw fallback, one N3 regression, actual movement44/45, constraint acceptance distinct from metric success, V33 default unchanged, HTML local-use and MD/PNG public paths all explicit.',
        limits=['Publication byte/link/scope review, not a new accuracy or independent physical-GT validation.','Images were verified as unchanged sealed assets; no new image or mask array decoding and no repeated visual audit. Prior owner/ROOT actual visual reviews remain separate.','Small raw gain10 input/journals/mask files are explicitly user-authorized exceptions; large full-population archives/duplicate frame dumps remain separate Release work, not claimed uploaded here.','No inference/PnP/SubPix/metric recomputation, packaging builder rerun, source change, Git index/commit/push or network action.'],
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},checked_input_sha256={str(M):EXPECTED_M,str(V):EXPECTED_V,str(PUB/'REPORT_KO.md'):sha(PUB/'REPORT_KO.md'),str(PUB/'CASE_GALLERY_KO_V1.md'):sha(PUB/'CASE_GALLERY_KO_V1.md')},checks_detail=checks)
    (OUT/'REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (OUT/'REVIEW_SEAL.json').write_text(json.dumps(dict(receipt_sha256=sha(OUT/'REVIEW.json'),source_sha256=result['source_sha256'],input_sha256=result['checked_input_sha256']),indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks','failure_count','paths','image_files','HTML_files','source_AST_files','local_links','maximum_file_bytes')},indent=2))
    print('REVIEW_SHA256',sha(OUT/'REVIEW.json'))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
