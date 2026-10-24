"""Root-entry-only closing recorder; stdlib, no Git/API/model/phase writes.

Verified payload X is the receipt authority. This fresh closing plan starts at X;
root later records closing commit Y locally, avoiding an own-commit hash cycle.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,os,re,stat

H=Path('/home/minjae/Documents/publication_closing_preparation_b_v200_20261024')
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
PREFIX='_docs/experiments/pallet-keypoint-visibility/'
DELTA='publication/deltas/V200_20261024_v1/'
PHASE='query_visibility_native_heads_frozen_real_benchmark_20261024_v200'
PROTOCOL=H/'PROTOCOL_BEFORE_CLOSING_PUBLICATION_INPUTS.json'
COPY_ROLES={'PUBLICATION_AUDITOR_CODE','PUBLICATION_AUDITOR_PROTOCOL','PUBLICATION_AUDITOR_CHECKS',
            'REMOTE_VERIFIER_CODE','REMOTE_VERIFIER_PROTOCOL','REMOTE_VERIFIER_CHECKS','REMOTE_PROOF',
            'POLICY_SNAPSHOT','HANDOFF_SNAPSHOT'}
REQUIRED_QA_ROLES=set(('BYTE_ROUNDTRIP', 'PORTABLE_LINKS_CREDENTIALS', 'SAVED_REAL_NUMERIC_QA', 'ACTUAL_FIXED_CARD_VISUAL_QA', 'FINAL_REAL_REPORT_QA', 'ROOT_PAYLOAD_PLAN_QA'))
PATTERNS={
    'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'AWS_key':re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'GitHub_token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
    'OpenAI_token':re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b'),
    'Slack_token':re.compile(r'\bxox[baprs]-[A-Za-z0-9-]{20,}\b'),
}


def need(v,msg):
    if not bool(v):raise AssertionError(msg)


def safe(value,absolute=True):
    p=Path(value);need(p.is_absolute()==absolute and p.parts and '..' not in p.parts and all('v112' not in x.casefold() for x in p.parts),'Safe name before filesystem access')
    return p


def read_bytes(value):
    p=safe(value);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=new
        last=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(last).st_mode),'Regular files only')
    with os.fdopen(last,'rb') as f:return f.read()


def sha(raw):return hashlib.sha256(raw).hexdigest()
def bind(path):return {'path':str(path),'sha256':sha(read_bytes(path))}
def checked(binding):
    need(re.fullmatch('[0-9a-f]{64}',binding['sha256']) is not None,'Exact SHA binding')
    raw=read_bytes(binding['path']);need(sha(raw)==binding['sha256'],'Bound bytes drift');return raw
def read_json(binding):return json.loads(checked(binding))
def encoded(obj):return (json.dumps(obj,ensure_ascii=False,indent=2)+'\n').encode()


def write_new(value,raw):
    p=safe(value);cur=Path(p.anchor)
    for part in p.parts[1:-1]:
        cur=cur/part
        try:mode=cur.lstat().st_mode
        except FileNotFoundError:cur.mkdir();mode=cur.lstat().st_mode
        need(stat.S_ISDIR(mode),'Destination ordinary directory only')
    with p.open('xb') as f:f.write(raw)


def field(obj,path):
    need(isinstance(path,list) and bool(path),'Typed nonempty field path')
    for key in path:obj=obj[key]
    return obj


def credential_counts(raw):
    text=raw.decode('utf-8')
    return {name:len(pattern.findall(text)) for name,pattern in PATTERNS.items() if pattern.search(text)}


def manifest_paths(manifest):
    need(isinstance(manifest.get('entries'),list),'Publication manifest object entries')
    result=set()
    for row in manifest['entries']:
        if row['status']=='EXCLUDED':continue
        result.add(str(safe(row['published_relative'],False)))
        if row.get('gzipchunks'):
            result.update(str(safe(p['path'],False)) for p in row['gzipchunks']['parts'])
            if row.get('gzip_descriptor_path'):result.add(str(safe(row['gzip_descriptor_path'],False)))
    return result


def collect_payload(plan,state,payload):
    need(plan['authorized_main_push'] is True and plan['repository']==str(REPO) and plan['branch']=='main' and plan['remote']=='https://github.com/CanelE452/pallet-6d-pose.git','Payload plan authority')
    need(state['completed_all'] is True and state['pending'] is None and state['expected_remote']==payload,'Payload push completion X')
    need(len(state['completed'])==len(plan['batches']) and bool(plan['batches']),'All planned batches completed')
    files=[]
    for batch,record in zip(plan['batches'],state['completed']):
        need(record['git_index_byte_identity'] is True and record['remote_verified'] is True,'Strict Git byte/remote proof')
        need(record['files']==len(batch['files']) and record['stored_bytes']==sum(f['size_bytes'] for f in batch['files']),'Pushstate file counts')
        files.extend(batch['files'])
    need(state['completed'][-1]['commit']==payload and len({f['path'] for f in files})==len(files),'Final payload commit and unique files')
    return {'physical_files':len(files),'physical_bytes':sum(f['size_bytes'] for f in files),'strict_Git_index_and_remote_verified':True}


def link_check(note,files,covered):
    """Portable new receipt links against sealed copied/published path names."""
    import posixpath
    text=note.decode('utf-8');need('/home/' not in text and 'file://' not in text,'Portable receipt must not expose local links')
    new_paths={f['path'][len(PREFIX):] for f in files}
    links=re.findall(r'\]\(([^\s)]+)\)',text)
    for link in links:
        if link.startswith('https://github.com/CanelE452/pallet-6d-pose/commit/'):
            continue
        need(not link.startswith(('/', 'http:', 'https:', 'file:')),'Only fixed commit URL or relative publication link')
        normalized=posixpath.normpath(posixpath.join(DELTA,link))
        need(not normalized.startswith('../') and normalized in covered|new_paths,'Portable link must target declared copied/published path')
    return len(links)


def run(entry_path):
    e=json.loads(read_bytes(entry_path))
    need(e['approved_verified_payload_closing'] is True and e['payload_push_completed'] is True,'Root verified-payload closing entry')
    need(e['code']==bind(Path(__file__).absolute()) and e['protocol']['path']==str(PROTOCOL),'Exact frozen recorder')
    p=read_json(e['protocol']);need(p['code']==e['code'],'Protocol/code closure')
    read_json(e['root_scope'])
    payload=e['payload_commit'];need(re.fullmatch('[0-9a-f]{40}',payload) is not None,'Actual root-provided payload X')
    need(e['repository']==str(REPO) and e['delta_relative']==DELTA,'Exact fresh V200 destination')
    plan=read_json(e['payload_push_plan']);state=read_json(e['payload_push_state'])
    need(state['plan_sha256']==e['payload_push_plan']['sha256'],'Payload state/plan seal')
    summary=collect_payload(plan,state,payload)
    remote=read_json(e['remote_review'])
    need(remote['completed'] is True and remote['PASS'] is True and remote['public'] is True and remote['default_branch']=='main','Independent public remote proof')
    need(remote['repository']=='CanelE452/pallet-6d-pose' and remote['selected_files_verified']==5,'Fixed5 independent blob metadata')
    need(remote['expected_payload_commit']==remote['remote_main_before']==remote['remote_main_after']==payload,'Remote proof exactly X')
    need(remote['sealed_push_plan']==e['payload_push_plan'] and remote['remote_content_raw_blob_contents_base64_or_pixel_reads']==0,'Remote metadata-only exact plan proof')
    qa={}
    for item in e['independent_QA']:
        role=item['role'];need(role not in qa,'Unique QA roles');report=read_json(item['binding'])
        need(report['completed'] is True and report['PASS'] is True,'Independent QA completed/PASS')
        need(item['assertions'],'Root exact typed QA field assertions required')
        for assertion in item['assertions']:need(field(report,assertion['field_path'])==assertion['equals'],'QA asserted field differs')
        qa[role]=item
    need(REQUIRED_QA_ROLES<=qa.keys(),'Mandatory byte/link, saved real numeric, fixed-card actual visual, final real report and rootpayload QA roles')
    manifests=[read_json(b) for b in e['immutable_manifests']];need(len(manifests)==15,'Exact BASE plusV187..V200 fifteen immutable publication manifests')
    covered=set().union(*(manifest_paths(d) for d in manifests))
    delta=read_json(e['payload_delta_manifest']);need(delta['phase']==PHASE,'Actual V200 payload delta')
    source=[r for r in delta['entries'] if r['status'].startswith('PUBLISHED_') and r.get('source_root_id')=='P']
    summary.update(new_original_source_files=len(source),new_original_source_bytes=sum(r['size_bytes'] for r in source))
    copies=e['copies'];roles=[c['role'] for c in copies]
    need(len(roles)==len(set(roles)) and COPY_ROLES<=set(roles),'Explicit auditor/verifier/proof/policy/handoff roles')
    need(next(c['binding'] for c in copies if c['role']=='REMOTE_PROOF')==e['remote_review'],'Copied remote proof exact')
    out=safe(e['output_directory']);need(out==H/'actual_closing_build_v1','Fresh outsidephase local output')
    out.mkdir(exist_ok=False);files=[];copied=[];redacted=[]
    try:
        # Pre-read all explicitly bound text before any publication write.
        staged=[]
        for c in copies:
            name=str(safe(c['closing_relative'],False));need(not name.startswith('.') and name not in ('PUBLICATION_RECEIPT_KO.md','PUBLICATION_RECEIPT_KO.json'),'Fresh nonreceipt copy name')
            relative=DELTA+'closing/'+name;need(relative not in covered,'Old manifest-covered path immutable')
            need(Path(c['binding']['path']).suffix.lower() in {'.py','.json','.jsonl','.md','.txt','.csv','.sh'},'Closing code/metadata/text only')
            raw=checked(c['binding']);counts=credential_counts(raw)
            redacted.append({'role':c['role'],'relative':relative,'credential_rules_and_counts':counts})
            need(not counts,'Credential pattern found; report only role/rule/count, no sensitive values')
            staged.append((relative,raw,c['binding'],c['role']))
        for name,path in [('record_v200_closing_publication.py',Path(__file__).absolute()),('CLOSING_PROTOCOL.json',PROTOCOL),('CLOSING_INVENTED_CHECKS.json',H/'GENERATED_INVENTED_CLOSING_CHECKS.json'),('ROOT_ACTUAL_CLOSING_ENTRY.json',Path(entry_path).absolute())]:
            raw=read_bytes(path);need(not credential_counts(raw),'Own closing metadata credential check')
            staged.append((DELTA+'closing/'+name,raw,bind(path),'CLOSING_CONTROL'))
        for role,item in qa.items():
            raw=checked(item['binding']);counts=credential_counts(raw)
            redacted.append({'role':role,'relative':DELTA+'closing/independent_QA/'+role+'.json','credential_rules_and_counts':counts})
            need(not counts,'Independent QA metadata credential pattern; role/rule/count only')
            staged.append((DELTA+'closing/independent_QA/'+role+'.json',raw,item['binding'],role))
        names=[name for name,*_ in staged];need(len(names)==len(set(names)),'No duplicate closing destination')
        need(all(n not in covered for n in names),'No logical/physical old path collision')
        receipt_paths=[DELTA+'PUBLICATION_RECEIPT_KO.json',DELTA+'PUBLICATION_RECEIPT_KO.md'];need(all(n not in covered for n in receipt_paths),'Fresh versioned receipts')
        for relative,raw,source_binding,role in staged:
            need(len(raw)<100*1024**2,'Each physical closing file below100MiB')
            write_new(REPO/(PREFIX+relative),raw)
            files.append({'path':PREFIX+relative,'sha256':sha(raw),'size_bytes':len(raw)})
            copied.append({'role':role,'source':source_binding,'published_relative':relative,'sha256':sha(raw),'size_bytes':len(raw)})
        receipt={'completed':True,'PASS':True,'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'phase':'V200','public':True,'repository':'https://github.com/CanelE452/pallet-6d-pose','branch':'main','verified_payload_commit':payload,'payload_summary':summary,'independent_QA':e['independent_QA'],'remote_metadata_review':e['remote_review'],'payload_delta_manifest':e['payload_delta_manifest'],'immutable_manifests':e['immutable_manifests'],'copies':copied,'only_new_closing_files':True,'old_payload_or_manifest_paths_overwritten':False,'global_README_mutation':False,'closing_operation_new_model_fit_labels_admission_or_goal_success':False,'completed_frozen_real_benchmark_phase':'V200은 source V199의 RAW nativezero8·onehot8 두 동결 상태를 원 query/crop 그대로 실사에서 비교한 단계입니다. Current humansemantic 10모델과 public assistant-weak 9모델의 저장 수치 및 같은 고정68카드를 구분해 기록합니다. 새 C58/C59가 기존 C57의 human D/E 개수보다 나은 개선을 입증하지 못했습니다. 기존 개발 노출이 반복된 비블라인드 평가이며 새로운 학습·source생성·SAM·PnP·좌표·문턱·모델/데이터/architecture 선택은 없습니다. Source 학습에 publicreal이 허용되지만 current/publicheld/COMMON/VAL을 fit하지 않습니다. SELF136/paired21, reference461/manual376 및 source direct-known native6/7 부재를 유지합니다. Humansemantic과 publicweak의 권한을 섞지 않으며 68카드 pixelparity와 실제12장 판독 범위도 구분합니다. 새 native physicalgold·실사 일반화·인과적 개선·채택·전체 목표 해결을 주장하지 않습니다.','receipt_cycle_rule':'This record binds payloadX. Root later records closingpushY locally after strict push; no owncommit or futureY in this receipt.','ongoing_authorization':'Completed attempts/methods/results/failures/figures/tables/verification are appended under root approved publication policy.'}
        raw=encoded(receipt);write_new(REPO/(PREFIX+receipt_paths[0]),raw);files.append({'path':PREFIX+receipt_paths[0],'sha256':sha(raw),'size_bytes':len(raw)})
        remote_copy=next(c['closing_relative'] for c in copies if c['role']=='REMOTE_PROOF')
        auditor_copy=next(c['closing_relative'] for c in copies if c['role']=='PUBLICATION_AUDITOR_CODE')
        note=('# V200 동결 실사 비교 main 게시 확인\n\n'+'V200은 source V199의 RAW nativezero8·onehot8 두 동결 상태를 원 query/crop 그대로 실사에서 비교한 단계입니다. Current humansemantic 10모델과 public assistant-weak 9모델의 저장 수치 및 같은 고정68카드를 구분해 기록합니다. 새 C58/C59가 기존 C57의 human D/E 개수보다 나은 개선을 입증하지 못했습니다. 기존 개발 노출이 반복된 비블라인드 평가이며 새로운 학습·source생성·SAM·PnP·좌표·문턱·모델/데이터/architecture 선택은 없습니다. Source 학습에 publicreal이 허용되지만 current/publicheld/COMMON/VAL을 fit하지 않습니다. SELF136/paired21, reference461/manual376 및 source direct-known native6/7 부재를 유지합니다. Humansemantic과 publicweak의 권한을 섞지 않으며 68카드 pixelparity와 실제12장 판독 범위도 구분합니다. 새 native physicalgold·실사 일반화·인과적 개선·채택·전체 목표 해결을 주장하지 않습니다.'+'\n\n'
              '- [V200까지의 전체 실험 인덱스](../../../EXPERIMENT_INDEX_THROUGH_V200_KO.md)\n'
              '- [V200 원 query 실사 비교·방법·한계](../../../archive/task-2/pallet-keypoint-visibility/'+PHASE+'/PORTABLE_REPORT_KO.md)\n'
              '- [독립 저장 실사 수치 검토](closing/independent_QA/SAVED_REAL_NUMERIC_QA.json)\n'
              '- [고정 카드 실제 시각 검토](closing/independent_QA/ACTUAL_FIXED_CARD_VISUAL_QA.json)\n'
              '- [독립 최종 실사 보고서 검토](closing/independent_QA/FINAL_REAL_REPORT_QA.json)\n'
              '- [게시 영수증 JSON](PUBLICATION_RECEIPT_KO.json)\n'
              '- [독립 원문 바이트 검산](closing/independent_QA/BYTE_ROUNDTRIP.json)\n'
              '- [독립 링크·credential 검산](closing/independent_QA/PORTABLE_LINKS_CREDENTIALS.json)\n\n'
              '- [독립 원격 public/main·5개 blob 검산](closing/'+remote_copy+')\n'
              '- [복사된 게시 원문 검산 코드](closing/'+auditor_copy+')\n\n'
              '검증된 payload는 '+str(summary['physical_files'])+'개 물리 파일, '+str(summary['physical_bytes'])+'바이트입니다. 신규 원문 기록 '+str(summary['new_original_source_files'])+'개와 지정5개 원격 blob metadata 검산을 영수증에 바인딩했습니다. 원격 콘텐츠·픽셀·모델·배열은 다운로드하거나 해석하지 않았습니다.\n\n'
              '검증된 payload commit X: ['+payload[:7]+'](https://github.com/CanelE452/pallet-6d-pose/commit/'+payload+'). 후속 closing commit Y는 push 뒤 로컬 영수증에만 기록하여 자기 commit 순환 참조를 피합니다.\n').encode()
        link_count=link_check(note,files,covered|set(receipt_paths));need(not credential_counts(note),'Portable receipt credential check')
        write_new(REPO/(PREFIX+receipt_paths[1]),note);files.append({'path':PREFIX+receipt_paths[1],'sha256':sha(note),'size_bytes':len(note)})
        need(len({f['path'] for f in files})==len(files) and all(f['size_bytes']<100*1024**2 for f in files) and sum(f['size_bytes'] for f in files)<480*1024**2,'Unique bounded closing-only plan')
        for f in files:need(bind(REPO/f['path'])['sha256']==f['sha256'],'Closing bytes unchanged')
        for b in e['immutable_manifests']:checked(b)
        closing_plan={'authorized_main_push':True,'repository':str(REPO),'remote':'https://github.com/CanelE452/pallet-6d-pose.git','branch':'main','initial_remote_main':payload,'manifest_sha256':e['payload_delta_manifest']['sha256'],'batch_max_bytes':480*1024**2,'batches':[{'files':files,'message':'Record V200 verified payload receipt, copied independent tools and remote proof'}],'scope':'Fresh V200 closing only; no payload/oldpaths/README changes; root strict push separately.'}
        write_new(out/'ROOT_V200_CLOSING_PUSH_PLAN.json',encoded(closing_plan));write_new(out/'PAYLOAD_X_PUBLICATION_RECEIPT_LOCAL.json',encoded(receipt))
        result={'completed':True,'PASS':True,'staged_only_not_pushed':True,'entry':bind(entry_path),'payload_X':payload,'files':len(files),'bytes':sum(f['size_bytes'] for f in files),'closing_push_plan':bind(out/'ROOT_V200_CLOSING_PUSH_PLAN.json'),'receipt_JSON':bind(REPO/(PREFIX+receipt_paths[0])),'receipt_MD':bind(REPO/(PREFIX+receipt_paths[1])),'credential_format_detections':0,'redacted_scan':redacted,'portable_relative_links_checked':link_count,'Git_or_remote_calls':0,'V200_scientific_phase_writes':0,'final_closing_Y':'PENDING_ROOT_STRICT_PUSH_THEN_LOCAL_ONLY_RECEIPT'}
        write_new(out/'COMPLETE.json',encoded(result));print(json.dumps(result),flush=True)
    except Exception:
        # Never include raw credential matches or copied contents in failure output.
        write_new(out/'FAILURE.json',encoded({'completed':False,'PASS':False,'failure':'Closing build failed; preserve partial files and command traceback, no automatic retry.','copied_files_before_failure':len(files),'redacted_scan':redacted,'automatic_retry':False}));raise


def selftest():
    need(sha(b'abc')=='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad','SHA fixture')
    need(field({'a':{'b':3}},['a','b'])==3,'Typed nested QA assertion')
    need(credential_counts(('ghp_'+'a'*40).encode())=={'GitHub_token':1},'Credential fixture redacted')
    need(not credential_counts(b'ordinary protocol and source path provenance'),'Benign text')
    try:safe('a/V112/b',False);raise RuntimeError('Forbidden check failed')
    except AssertionError:pass
    need(manifest_paths({'entries':[{'status':'PUBLISHED_GZIP_CHUNKS','published_relative':'old.jsonl','gzip_descriptor_path':'old.jsonl.parts.json','gzipchunks':{'parts':[{'path':'old.jsonl.parts/a.gz'}]}}]})=={'old.jsonl','old.jsonl.parts.json','old.jsonl.parts/a.gz'},'All physical/logical covered names')
    payload='a'*40;plan={'authorized_main_push':True,'repository':str(REPO),'branch':'main','remote':'https://github.com/CanelE452/pallet-6d-pose.git','batches':[{'files':[{'path':PREFIX+'new.json','size_bytes':7,'sha256':'b'*64}]}]}
    state={'completed_all':True,'pending':None,'expected_remote':payload,'completed':[{'git_index_byte_identity':True,'remote_verified':True,'files':1,'stored_bytes':7,'commit':payload}]}
    need(collect_payload(plan,state,payload)['physical_files']==1,'Typed completed push-state derivation')
    need(link_check(b'[receipt](PUBLICATION_RECEIPT_KO.json)',[],{DELTA+'PUBLICATION_RECEIPT_KO.json'})==1,'Portable relative link resolution')
    need(len(REQUIRED_QA_ROLES)==6 and {'SAVED_REAL_NUMERIC_QA','ACTUAL_FIXED_CARD_VISUAL_QA','FINAL_REAL_REPORT_QA','ROOT_PAYLOAD_PLAN_QA'}<=REQUIRED_QA_ROLES,'All6 saved-real numeric/fixed-card/report plus byte/link/root typed QA roles')
    need(not REQUIRED_QA_ROLES<={'BYTE_ROUNDTRIP','PORTABLE_LINKS_CREDENTIALS'},'Publication-only QA cannot replace source diagnostic/report/visual closure')
    return {'completed':True,'PASS':True,'invented_only':True,'checks':10,'actual_repository_root_input_or_remote_reads':0}


if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--selftest',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args()
    print(json.dumps(selftest())) if a.selftest else run(a.entry)
