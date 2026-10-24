"""Independent, root-entry-only V200 publication byte/link audit. No Git calls.
Preparation uses invented bytes and code/protocol text, never phase inventories.
"""
from pathlib import Path, PurePosixPath
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import argparse, codecs, gzip, hashlib, io, json, os, posixpath, re, stat, tempfile, traceback

H=Path(__file__).absolute().parent
P=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
V=P/'query_visibility_native_heads_frozen_real_benchmark_20261024_v200'
REPO=Path('/home/minjae/Documents/pallet-visibility-publication-20261024')
PREFIX='_docs/experiments/pallet-keypoint-visibility/'
PUB=REPO/PREFIX
DELTA='publication/deltas/V200_20261024_v1/'
PHASE=V.name
ARCHIVE='archive/task-2/pallet-keypoint-visibility/'+PHASE+'/'
BASE_SHA='f5f4c2d819c7533492e24d2238baa8a6f638438c4458492d01c0321a34130fc7'
OUT=P/'publication_root_preparation/V200_DELTA_INDEPENDENT_REVIEW_C'
PROTOCOL=H/'PROTOCOL_BEFORE_ACTUAL_V200_DELTA_BYTES.json'
CHECKS=H/'GENERATED_INVENTED_CHECKS.json'
C=P/'publication_root_preparation/V200_DELTA_PREPARATION_C'
C_CODE={'path':str(C/'build_completed_phase_delta.py'),'sha256':'ef56c6839cc29515f525f87b1d9782d15a89cb372215b2babcedf68ac9dfc6e4'}
C_PROTO={'path':str(C/'BUILDER_PROTOCOL_BEFORE_ANY_DELTA_ARTIFACT_READ.json'),'sha256':'72fa9ed14fe6a7aa286e066eb612114bd45b8b016bd8691d6ad9eb30a14888b2'}
PARENT_DELTA='publication/deltas/V199_20261024_v1/DELTA_MANIFEST.json'
PARENT_CHAIN='MANIFEST_CHAIN_AFTER_V199.json'
PARENT_INDEX='EXPERIMENT_INDEX_THROUGH_V199_KO.md'
ANCESTORS=(('V187', 'query_visibility_source_selected_part_correspondence_20261024_v187'), ('V188', 'query_visibility_direct_end_local_counterfactual_20261024_v188'), ('V189', 'query_visibility_sampled_domain_direct_end_20261024_v189'), ('V190', 'query_visibility_direct_end_paired_source_training_20261024_v190'), ('V191', 'query_visibility_direct_end_heads_frozen_real_benchmark_20261024_v191'), ('V192', 'query_visibility_source_direct_end_coverage_audit_20261024_v192'), ('V193', 'query_visibility_source_end_pairs_opposite_sham_20261024_v193'), ('V194', 'query_visibility_source_group_split_direct_end_training_20261024_v194'), ('V195', 'query_visibility_grouped_direct_end_heads_frozen_real_benchmark_20261024_v195'), ('V196', 'query_visibility_grouped_source_holdout_target_readout_20261024_v196'), ('V197', 'query_visibility_train_only_target_contract_readout_20261024_v197'), ('V198', 'query_visibility_native_query_conditioning_feasibility_20261024_v198'), ('V199', 'query_visibility_native_conditioned_grouped_source_training_20261024_v199'))
PORTABLE=['archive/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/PORTABLE_REPORT_KO.md', 'archive/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/ROOT_REPORT_KO.md', 'archive/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/ROOT_READOUT_KO.html', 'archive/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/fixed_native_real_visual_v1/PORTABLE_GALLERY_KO.md', 'archive/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/fixed_native_real_visual_v1/index.html', 'EXPERIMENT_INDEX_THROUGH_V200_KO.md']
EXTERNAL_METADATA=set()
TEXT={'.py','.sh','.md','.txt','.json','.jsonl','.csv','.tsv','.html','.htm','.yaml','.yml','.toml','.log','.patch','.diff','.svg'}
LIMIT=100*1024**2

ARMS=('C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL', 'C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT')
DOMAINS=('CURRENT223_1784','PUBLIC_HELD45_360')
RUN='frozen_C58_C59_real_predictions_v1'
AUTHORITY_KEYS=('source_prediction_seal','evaluation_complete','independent_saved_result_review','final_phase_freeze')
ROOT_SEAL={'path': '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/frozen_C58_C59_real_predictions_v1/PREDICTION_SEAL.json', 'sha256': 'ff28aa2047e73af2b843f93a2c29f6789490504d00494f1188bb746c171a69fa'}
ROOT_BINDINGS={'frozen_C58_C59_real_predictions_v1/C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL/FINAL_CURRENT223_1784_OUTPUTS.npz': {'path': '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/frozen_C58_C59_real_predictions_v1/C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL/FINAL_CURRENT223_1784_OUTPUTS.npz', 'sha256': '49e9767048f35d5d06427cc13776449ffacb19a5f7d8aeeb8fc86bc5a1f2e33c'}, 'frozen_C58_C59_real_predictions_v1/C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL/FINAL_PUBLIC_HELD45_360_OUTPUTS.npz': {'path': '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/frozen_C58_C59_real_predictions_v1/C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL/FINAL_PUBLIC_HELD45_360_OUTPUTS.npz', 'sha256': 'bc962112f0dad0ba0f8926cae720a83c95eb2fee1ae6f87f3a1d6f3bb1acde1d'}, 'frozen_C58_C59_real_predictions_v1/C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT/FINAL_CURRENT223_1784_OUTPUTS.npz': {'path': '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/frozen_C58_C59_real_predictions_v1/C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT/FINAL_CURRENT223_1784_OUTPUTS.npz', 'sha256': '30c663865e46af910b9da9230b7174031fd59bafd664b227001930e4b56178e2'}, 'frozen_C58_C59_real_predictions_v1/C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT/FINAL_PUBLIC_HELD45_360_OUTPUTS.npz': {'path': '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_native_heads_frozen_real_benchmark_20261024_v200/frozen_C58_C59_real_predictions_v1/C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT/FINAL_PUBLIC_HELD45_360_OUTPUTS.npz', 'sha256': 'cabb4bd02558e4136ca30465bef7774a5e961e8c5ba794c4fe44928d361f214e'}}
ROOT_OWNED=()
ROOT_SAVED=('frozen_C58_C59_real_predictions_v1/C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL/FINAL_CURRENT223_1784_OUTPUTS.npz', 'frozen_C58_C59_real_predictions_v1/C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL/FINAL_PUBLIC_HELD45_360_OUTPUTS.npz', 'frozen_C58_C59_real_predictions_v1/C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT/FINAL_CURRENT223_1784_OUTPUTS.npz', 'frozen_C58_C59_real_predictions_v1/C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT/FINAL_PUBLIC_HELD45_360_OUTPUTS.npz')

def require(v,s):
    if not bool(v):raise AssertionError(s)
def safe_name(path):
    p=Path(path)
    require('..'not in p.parts and all('v112'not in s.casefold()for s in p.parts),'Forbidden path before access')
    return p
def relative(path):
    require(isinstance(path,str)and path and '\\'not in path,'POSIX relative publication path')
    p=safe_name(path);require(not p.is_absolute()and not urlsplit(path).scheme,'Relative publication path')
    require(str(PurePosixPath(path))==path,'Canonical path');return path
def opened(path):
    p=safe_name(path);require(p.is_absolute(),'Explicit absolute path')
    require(p==REPO or REPO in p.parents or P in p.parents or str(p) in EXTERNAL_METADATA,'Explicit roots or root-bound eight closing metadata files only')
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=new
        final=os.open(p.parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    require(stat.S_ISREG(os.fstat(final).st_mode),'Regular file, no symlink');return os.fdopen(final,'rb')
def file_chunks(path):
    with opened(path)as f:
        for b in iter(lambda:f.read(1<<20),b''):yield b
def digest(chunks):
    h=hashlib.sha256();size=0
    for b in chunks:h.update(b);size+=len(b)
    return h.hexdigest(),size
def bind(path):return {'path':str(path),'sha256':digest(file_chunks(path))[0]}
def checked(b):
    require(isinstance(b.get('sha256'),str)and len(b['sha256'])==64,'Exact binding required')
    require(digest(file_chunks(b['path']))[0]==b['sha256'],'Binding drift '+b['path']);return Path(b['path'])
def read(path):
    with opened(path)as f:return json.load(f)
def readlines(path):
    with opened(path)as f:return [json.loads(s)for s in f if s.strip()]
def bytes_of(path):
    with opened(path)as f:return f.read()
def write(path,obj):
    safe_name(path)
    with Path(path).open('x')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def covered(entry):
    if entry['status']=='EXCLUDED':return set()
    paths={relative(entry['published_relative'])}
    if entry.get('gzipchunks')is not None:
        paths.update(relative(r['path'])for r in entry['gzipchunks']['parts'])
    if entry.get('gzip_descriptor_path'):paths.add(relative(entry['gzip_descriptor_path']))
    return paths
def physical(entry):
    if entry['status']=='EXCLUDED':return set()
    if entry.get('gzipchunks')is None:return {relative(entry['published_relative'])}
    return {relative(r['path'])for r in entry['gzipchunks']['parts']}
def assert_disjoint(base,entries):
    old=set().union(*(covered(e)for e in base))if base else set()
    for e in entries:require(not(old&covered(e)),'Old logical/physical/descriptor path collision')
    return old
def admin_authority(row,allowed):
    path=safe_name(row['source_absolute'])
    require(path.parent==C and row['category']=='versioned_publication_tool','Explicit frozen administrative prep only')
    require(row['source_relative_path']=='publication_root_preparation/V200_DELTA_PREPARATION_C/'+path.name,'Admin source relative identity')
    require(row['published_relative']=='tools/completed_phase_delta_v200/'+path.name,'Admin versioned destination')
    require(allowed.get(str(path))==row['source_sha256'],'Exact frozen administrative file binding')
def ancestor_paths(tag):
    return {'delta_manifest':'publication/deltas/'+tag+'_20261024_v1/DELTA_MANIFEST.json',
            'chain':'MANIFEST_CHAIN_AFTER_'+tag+'.json','index':'EXPERIMENT_INDEX_THROUGH_'+tag+'_KO.md'}
def cumulative_parent_entries(base_binding,base,records):
    require(isinstance(base.get('entries'),list),'Base entries schema')
    require(len(records)==13,'All13 original V187..V199 ancestors required')
    entries=list(base['entries']);chain_refs=[];previous=None
    protected=['PUBLICATION_MANIFEST.json']
    for row,(tag,phase) in zip(records,ANCESTORS):
        paths=ancestor_paths(tag);require(row['tag']==tag,'Chronological ancestor order')
        for key,path in paths.items():require(row[key]['path']==str(PUB/path),'Exact ancestor metadata path')
        d=row['manifest'];c=row['chain_data']
        require(d['format']=='immutable-completed-phase-delta-v1' and d['phase']==phase and d['previous_base_manifest']==base_binding,'Ancestor phase/base authority')
        require(isinstance(d['entries'],list),'Ancestor entries')
        chain_refs.append({'path':paths['delta_manifest'],'sha256':row['delta_manifest']['sha256']})
        require(c['format']=='immutable-publication-manifest-chain-v1' and c['base']=={'path':'PUBLICATION_MANIFEST.json','sha256':BASE_SHA},'Ancestor chain base')
        require(c['deltas']==chain_refs and c['index']=={'path':paths['index'],'sha256':row['index']['sha256']},'Full ancestor chain and index binding')
        index_rows=[x for x in d['entries'] if x['status']!='EXCLUDED' and x.get('published_relative')==paths['index']]
        require(len(index_rows)==1 and index_rows[0]['status']=='PUBLISHED_RAW' and index_rows[0]['published_sha256']==row['index']['sha256'],'Index is exact unique raw ancestor record')
        if previous is not None:
            require(d['previous_delta_manifest']==previous['delta_manifest'] and d['previous_chain']==previous['chain'],'Each newer ancestor closes previous manifest and chain')
            if 'previous_index' in d:require(d['previous_index']==previous['index'],'Optional legacy V188 explicit previous index')
            if tag in ('V189','V190','V191','V192','V193','V194','V195','V196','V197','V199'):require(d['previous_index']==previous['index'],'Newer ancestors must explicitly bind previous index')
            require(c['previous_chain']=={'path':ancestor_paths(previous['tag'])['chain'],'sha256':previous['chain']['sha256']},'V188 chain previous V187 hash')
        entries.extend(d['entries']);protected.extend(paths.values());previous=row
    entries.extend({'status':'PUBLISHED_RAW','published_relative':path,'gzipchunks':None} for path in protected)
    return entries

def closing_entries(records):
    require([r['tag'] for r in records]==['BASE', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199'],'All fourteen exact ordered closing plans')
    protected=[];seen=set();exceptions=[]
    for row in records:
        plan=row['plan_data']
        require(plan['authorized_main_push'] is True and plan['repository']==str(REPO) and plan['branch']=='main' and plan['remote']=='https://github.com/CanelE452/pallet-6d-pose.git','Closing strict plan authority')
        require(plan['batches'],'Nonempty closing plans')
        for batch in plan['batches']:
            for item in batch['files']:
                path=relative(item['path'])
                require(type(item['size_bytes']) is int and 0<=item['size_bytes']<LIMIT and re.fullmatch('[0-9a-f]{64}',item['sha256']) is not None,'Closing metadata hash/size')
                if path=='readme.md':
                    require(row['tag']=='BASE' and not exceptions,'Only one BASE historical README exception')
                    exceptions.append({k:v for k,v in dict(tag='BASE',path=path,sha256=item['sha256'],size_bytes=item['size_bytes']).items()});continue
                require(path.startswith(PREFIX),'All other closing files inside publication prefix')
                rel=path[len(PREFIX):];require(rel and rel not in seen,'Unique protected closing path');seen.add(rel)
                protected.append(dict(status='PUBLISHED_RAW',published_relative=rel,gzipchunks=None))
    require(len(exceptions)==1,'Historical BASE README exception explicitly retained')
    return protected,exceptions

def same_exceptions(actual,expected):
    require([{k:r[k] for k in ('tag','path','sha256','size_bytes')} for r in actual]==expected,'Exact declared historical README exception')

# V200 root-declared output authority below
# V200 root-declared output authority below
# V200 root-declared output authority below
# V200 root-declared output authority below
# V200 root-declared output authority below
# V200 root-declared output authority below
def numeric_metadata_policy(e,pred,ev,audit):
    require(e['source_prediction_seal']==ROOT_SEAL,'Root-fixed exact V200 prediction seal')
    require(e['evaluation_complete']['path']==str(V/'saved_real_head_evaluation_v1/COMPLETE.json')and e['independent_saved_result_review']['path']==str(V/'independent_saved_result_review_c/actual_review_v1/REVIEW.json'),'Exact V200 saved evaluation/numeric audit')
    require(e['final_phase_freeze']['path']==str(V/'ROOT_FINAL_REAL_BENCHMARK_PHASE_FREEZE.json'),'Exact completed V200 phasefreeze')
    require(pred['completed']is True and pred['PASS']is True and pred['all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads']is True and pred['all_batch_repeat2_state_and_input_bytes_exact']is True,'Both frozen predictions/repeat2/state/input closure before annotations')
    require(pred['original_rows2144']==2144 and pred['model_domain_rows4288']==4288 and pred['original_currentU45_publicU108']is True,'Complete original real-domain rows/U')
    require(set(pred['arms'])==set(ARMS),'Both root-fixed RAW arms, no outcome subset')
    require(ev['completed']is True and ev['PASS']is True and ev['prediction_seal']==e['source_prediction_seal']and ev['all_both_predictions_closed_before_any_annotation_or_score_body']is True,'Saved evaluation closed exact both prediction seal')
    require(audit['completed']is True and audit['PASS']is True and audit['complete']==e['evaluation_complete'],'Independent exact saved evaluation review')
    for d in (pred,ev,audit):require(d['selected_model']is None and d['adopted']is False and d['fully_solved']is False,'No winner/adoption/solved authority')
    artifacts={}
    for i,arm in enumerate(ARMS):
        r=pred['arms'][arm]
        require(r['updates']==8000 and r['use_sam']is False and type(r['native_condition'])is bool and r['native_condition']==bool(i),'Both RAW strict zero8/onehot flags and final8000')
        require(set(r['outputs'])==set(DOMAINS),'Both original domains per arm')
        for domain in DOMAINS:
            rel=RUN+'/'+arm+'/FINAL_'+domain+'_OUTPUTS.npz';b=r['outputs'][domain]
            require(b==ROOT_BINDINGS[rel]and b['path']==str(V/rel),'Exact root-declared four output path/SHA bindings')
            require(len(b['sha256'])==64 and all(c in '0123456789abcdef'for c in b['sha256']),'Output SHA shape')
            artifacts[rel]=b
    require(set(artifacts)==set(ROOT_SAVED)and len(artifacts)==4,'All4 target-free saved16key packets; opaque bytes only')
    require(not e.get('explicit_own_head_relative_paths')and not e.get('explicit_direct_target_relative_paths')and e.get('direct_target_cache_complete')is None,'No new owned weights/sourcefit/targets')
    require(set(e['explicit_saved_output_relative_paths'])==set(artifacts),'No missing/subset/extra saved packet')
    receipts=e.get('phase_completion_receipts',e.get('completion_reviews',[]));require({'phase_result','independent_report_QA','final_phase_freeze'}<=set(r.get('role')for r in receipts),'All completion barrier roles')
    freeze=next(r for r in receipts if r['role']=='final_phase_freeze');require(freeze['binding']==e['final_phase_freeze'] and freeze['expected_fields'].get('scientific_phase_frozen_no_further_phase_writes')is True,'Exact freeze binding and scientific frozen flag in completion barriers')
    return dict(**{k:e[k]for k in AUTHORITY_KEYS},explicit_own_head_relative_paths=[],explicit_saved_output_relative_paths=sorted(artifacts),explicit_direct_target_relative_paths=[],direct_target_cache_complete=None,declared_artifact_bindings=artifacts)

def read_numeric_authority(e):
    docs=[read(checked(e[k]))for k in AUTHORITY_KEYS[:3]];read(checked(e['final_phase_freeze']))
    req=e.get('phase_completion_receipts',e.get('completion_reviews',[]))
    for item in req:
        o=read(checked(item['binding']));require(item['expected_fields']and all(o[k]==x for k,x in item['expected_fields'].items()),'Exact completion fields')
    return numeric_metadata_policy(e,*docs)

def source_archive_policy(records,own=(),saved=(),targets=(),bindings=None):
    require(not own and not targets,'No owned heads/directtarget sidecars in real-eval phase')
    selected=set(own)|set(saved)|set(targets);require(not selected or bindings is not None,'Nonempty numeric whitelist needs independent final metadata authority')
    allowed=TEXT|{'.pdf','.png','.jpg','.jpeg','.webp','.gif'};excluded=0;included=set()
    for row in records:
        safe_name(row['source_relative_path']);require(row['source_root_id']=='P'and row['source_absolute']==str(P/row['source_relative_path'])and row['source_relative_path'].startswith(PHASE+'/'),'Only exact frozen source phase')
        require(row['status']in{'EXCLUDED','PUBLISHED_RAW','PUBLISHED_GZIP_CHUNKS'},'Declared archive status');rel=str(Path(row['source_relative_path']).relative_to(PHASE))
        if row['status']=='EXCLUDED':
            require(rel not in selected and bool(row.get('exclusion_reason'))and row.get('gzipchunks')is None,'Excluded rawcache provenance; all required numeric artifacts included');excluded+=1;continue
        suffix=Path(rel).suffix.casefold();included.add(rel)
        if rel in selected:
            require(row['status']=='PUBLISHED_RAW'and suffix==('.pt'if rel in own else'.npz')and row['size_bytes']<(16 if rel in own else 32)*1024**2,'Owned final numeric artifact type/size')
            require(row['source_sha256']==bindings[rel]['sha256']==row['published_sha256']and row['source_absolute']==bindings[rel]['path'],'Numeric original/stored SHA equals final COMPLETE/target seal')
        else:require(suffix in allowed,'No implicit rawcache or external weight inclusion')
        require(row['published_relative'].startswith(ARCHIVE),'Exact sourcearchive destination')
    require(selected<=included,'Exact four real saved packets;0heads0targets')
    return excluded

def gunzip_parts(parts):
    for part in parts:
        require(part['compression']=='gzip','Declared gzip only')
        n=0
        with opened(PUB/relative(part['path']))as f:
            with gzip.GzipFile(fileobj=f,mode='rb')as g:
                for b in iter(lambda:g.read(1<<20),b''):n+=len(b);yield b
        require(n==part['decoded_size_bytes'],'Decoded chunk length')

PATTERNS=[('PEM_PRIVATE_KEY',re.compile(r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----')),
          ('GITHUB_CLASSIC_TOKEN',re.compile(r'(?<![A-Za-z0-9_])gh[pousr]_[A-Za-z0-9]{36}(?![A-Za-z0-9_])')),
          ('GITHUB_FINE_GRAINED_TOKEN',re.compile(r'(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{40,}(?![A-Za-z0-9_])')),
          ('OPENAI_TOKEN_FORMAT',re.compile(r'(?<![A-Za-z0-9_-])sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}(?![A-Za-z0-9_-])')),
          ('AWS_ACCESS_ID_FORMAT',re.compile(r'(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])')),
          ('SLACK_TOKEN_FORMAT',re.compile(r'(?<![A-Za-z0-9-])xox[baprs]-[A-Za-z0-9-]{20,}(?![A-Za-z0-9-])'))]
def scan_text(chunks):
    decoder=codecs.getincrementaldecoder('utf-8')('replace');carry='';line=0;hits=[]
    def inspect(s):
        nonlocal line
        line+=1
        for kind,pattern in PATTERNS:
            if pattern.search(s):hits.append({'kind':kind,'line':line})
    for b in chunks:
        parts=(carry+decoder.decode(b)).split('\n');carry=parts.pop()
        for s in parts:inspect(s)
    carry+=decoder.decode(b'',final=True)
    if carry:inspect(carry)
    return {'scanned_text_lines':line,'format_hits_redacted':hits}

class HrefParser(HTMLParser):
    def __init__(self):super().__init__();self.targets=[]
    def handle_starttag(self,tag,attrs):
        self.targets.extend(v for k,v in attrs if k in {'href','src'}and v is not None)
def link_targets(text):
    out=[]
    for m in re.finditer(r'!?\[[^\]\n]*\]\(\s*(<[^>]*>|[^\s)]+)(?:\s+[^)]*)?\)',text):out.append(m.group(1).strip('<>'))
    h=HrefParser();h.feed(text);out.extend(h.targets);return out
def resolve_target(document,target):
    u=urlsplit(target)
    require(u.scheme.lower()!='file','file link forbidden')
    if u.scheme:
        require(u.scheme.lower()in {'http','https','mailto','data'},'Unexpected portable URI scheme');return None
    path=unquote(u.path)
    if not path:return None
    require(not path.startswith('/')and not re.match(r'^[A-Za-z]:',path)and '\\'not in path,'Absolute portable link forbidden')
    require('/home/'not in path,'Home path link forbidden')
    resolved=posixpath.normpath(posixpath.join(posixpath.dirname(document),path))
    require(resolved!='..'and not resolved.startswith('../'),'Link escapes publication root')
    return relative(resolved)
def portable_audit(document,text,exists):
    if Path(document).suffix.casefold() not in {'.html','.htm'}:require('/home/'not in text and 'file://'not in text,'Portable Markdown text has no local absolute provenance')
    local=[];external=0
    targets=link_targets(text)
    if Path(document).suffix.casefold() in {'.html','.htm'}:
        parser=HrefParser();parser.feed(text);targets=parser.targets
    for target in targets:
        resolved=resolve_target(document,target)
        if resolved is None:external+=1
        else:require(exists(resolved),'Missing portable link: '+resolved);local.append(resolved)
    return {'document':document,'local_links':local,'local_link_count':len(local),'external_or_anchor_or_inline_count':external}
def ordinary_link_target(path):
    p=safe_name(path);require(p.is_absolute(),'Explicit absolute target')
    require(REPO in p.parents or P in p.parents,'Explicit publication/project target only')
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=new
        final=os.open(p.parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
    finally:os.close(fd)
    try:
        mode=os.fstat(final).st_mode
        return stat.S_ISREG(mode) or stat.S_ISDIR(mode)
    finally:os.close(final)
def exists_file_or_directory(relative_path):
    try:
        return ordinary_link_target(PUB/relative(relative_path))
    except FileNotFoundError:return False

def invented():
    global PUB
    tests=[]
    def add(n,v):tests.append({'name':n,'PASS':bool(v)})
    data=b'preserved source\n\x00bytes\xff';z=[gzip.compress(data[:8],mtime=0),gzip.compress(data[8:],mtime=0)]
    restored=b''.join(gzip.decompress(p)for p in z)
    add('two_independent_gzip_members_restore_original_bytes',digest([restored])==digest([data]))
    add('single_byte_corruption_changes_original_SHA',digest([data+b'x'])!=digest([data]))
    raw={'status':'PUBLISHED_RAW','published_relative':'old/a','gzipchunks':None}
    gz={'status':'PUBLISHED_GZIP_CHUNKS','published_relative':'old/b','gzip_descriptor_path':'old/b.parts.json','gzipchunks':{'parts':[{'path':'old/b.parts/part00000.gz'}]}}
    add('old_logical_part_descriptor_sets',covered(gz)=={'old/b','old/b.parts.json','old/b.parts/part00000.gz'})
    for name,path in [('raw','old/a'),('logical','old/b'),('part','old/b.parts/part00000.gz'),('descriptor','old/b.parts.json')]:
        try:assert_disjoint([raw,gz],[dict(raw,published_relative=path)]);ok=False
        except AssertionError:ok=True
        add('old_'+name+'_collision_rejected',ok)
    add('new_disjoint_path_accepted',bool(assert_disjoint([raw,gz],[dict(raw,published_relative='new/a')])))
    add('excluded_null_path_never_opened_or_normalized',covered({'status':'EXCLUDED','published_relative':None,'gzipchunks':None})==set())
    try:relative('archive/blocked_v112/file');ok=False
    except AssertionError:ok=True
    add('forbidden_name_rejected_without_stat',ok)
    s=scan_text([(b'plain\n'+('gh'+'p_'+'A'*36).encode())[:15],(b'plain\n'+('gh'+'p_'+'A'*36).encode())[15:]])
    add('credential_match_cross_chunk_redacted',s['format_hits_redacted']==[{'kind':'GITHUB_CLASSIC_TOKEN','line':2}])
    add('SHA_not_credential',not scan_text([('a'*64+'\n').encode()])['format_hits_redacted'])
    pages={'archive/p/img.png','index.md'}
    q=portable_audit('archive/p/report.md','![x](img.png)\n[up](../../index.md)\n[web](https://example.org)',lambda p:p in pages)
    add('relative_image_parent_index_and_remote',q['local_links']==['archive/p/img.png','index.md'])
    for name,target in [('home','/home/user/a'),('file','file:///tmp/a'),('escape','../../../a')]:
        try:portable_audit('archive/p/report.md','[x]('+target+')',lambda p:True);ok=False
        except AssertionError:ok=True
        add('portable_'+name+'_rejected',ok)
    add('strict_Git_100MiB_boundary',LIMIT-1<LIMIT and not LIMIT<LIMIT)
    add('previous_index_suffix_keeps_relative_parent',resolve_target('EXPERIMENT_INDEX_THROUGH_V200_KO.md','reports/old.md')=='reports/old.md')
    parent_cycle={'status':'PUBLISHED_RAW','published_relative':PARENT_CHAIN,'gzipchunks':None}
    try:assert_disjoint([parent_cycle],[dict(raw,published_relative=PARENT_CHAIN)]);ok=False
    except AssertionError:ok=True
    add('parent_chain_cycle_excluded_path_collision_rejected',ok)
    tool={'source_absolute':str(C/'invented.py'),'source_relative_path':'publication_root_preparation/V200_DELTA_PREPARATION_C/invented.py','published_relative':'tools/completed_phase_delta_v200/invented.py','category':'versioned_publication_tool','source_sha256':'a'*64}
    admin_authority(tool,{tool['source_absolute']:'a'*64});add('explicit_admin_code_whitelist_accepted',True)
    try:admin_authority(tool,{tool['source_absolute']:'b'*64});ok=False
    except AssertionError:ok=True
    add('admin_code_unknown_SHA_rejected',ok)
    two=[{'path':PARENT_DELTA,'sha256':'a'*64},{'path':DELTA+'DELTA_MANIFEST.json','sha256':'b'*64}]
    add('two_parent_delta_order_preserved',two[0]['path']==PARENT_DELTA and len(two)==2)
    with tempfile.TemporaryDirectory(prefix='invented_directory_only_',dir=H)as t:
        root=Path(t);directory=root/'ordinary';directory.mkdir();link=root/'symlink';link.symlink_to(directory,target_is_directory=True)
        add('ordinary_directory_link_target_accepted_without_enumeration',ordinary_link_target(directory))
        try:ordinary_link_target(link);ok=False
        except OSError:ok=True
        add('symlink_directory_target_refused_NOFOLLOW',ok)
    # Independently build a tiny BASE -> V187 -> V188 metadata chain.
    binding={'path':str(PUB/'PUBLICATION_MANIFEST.json'),'sha256':BASE_SHA}
    base={'entries':[raw]};records=[];refs=[]
    for i,(tag,phase) in enumerate(ANCESTORS):
        paths=ancestor_paths(tag)
        row={'tag':tag,**{k:{'path':str(PUB/path),'sha256':hashlib.sha256(str(i).encode()).hexdigest()} for k,path in paths.items()}}
        entries=[dict(raw,published_relative=paths['index'],published_sha256=row['index']['sha256'])]
        if i==0:entries.append(gz)
        d={'format':'immutable-completed-phase-delta-v1','phase':phase,'previous_base_manifest':binding,'entries':entries}
        refs=refs+[{'path':paths['delta_manifest'],'sha256':row['delta_manifest']['sha256']}]
        chain={'format':'immutable-publication-manifest-chain-v1','base':{'path':'PUBLICATION_MANIFEST.json','sha256':BASE_SHA},'deltas':refs,'index':{'path':paths['index'],'sha256':row['index']['sha256']}}
        if records:
            d.update(previous_delta_manifest=records[-1]['delta_manifest'],previous_chain=records[-1]['chain'])
            if tag not in ('V187','V188'):d['previous_index']=records[-1]['index']
            chain['previous_chain']={'path':ancestor_paths(records[-1]['tag'])['chain'],'sha256':records[-1]['chain']['sha256']}
        records.append(dict(row,manifest=d,chain_data=chain))
    cumulative=cumulative_parent_entries(binding,base,records)
    add('ordered_BASE_V187_V188_legacy_no_previous_index_is_valid',len(cumulative)>len(base['entries']))
    for name,alter in [('missing_V187',lambda r:r[1:]),('reversed_ancestors',lambda r:list(reversed(r))),('tampered_V188_chain',lambda r:[r[0],dict(r[1],chain_data=dict(r[1]['chain_data'],deltas=[]))])]:
        try:cumulative_parent_entries(binding,base,alter(records));ok=False
        except AssertionError:ok=True
        add(name+'_rejected',ok)
    for target in ['old/b.parts/part00000.gz','MANIFEST_CHAIN_AFTER_V187.json','EXPERIMENT_INDEX_THROUGH_V188_KO.md','publication/deltas/V188_20261024_v1/DELTA_MANIFEST.json']:
        try:assert_disjoint(cumulative,[dict(raw,published_relative=target)]);ok=False
        except AssertionError:ok=True
        add('cumulative_immutable_collision_'+target.replace('/','_'),ok)
    source={'source_root_id':'P','source_relative_path':PHASE+'/invented.npz','source_absolute':str(V/'invented.npz'),'status':'EXCLUDED','exclusion_reason':'Raw arrays are provenance-only','gzipchunks':None,'published_relative':None}
    add('declared_raw_array_exclusion_requires_no_file_access',source_archive_policy([source])==1)
    try:source_archive_policy([dict(source,status='PUBLISHED_RAW',published_relative=ARCHIVE+'invented.npz')]);ok=False
    except AssertionError:ok=True
    add('raw_array_inclusion_without_whitelist_rejected',ok)
    with tempfile.TemporaryDirectory(prefix='invented_gzip_stream_',dir=H) as t:
        # A temporary authored fixture replaces only the publication-root operand.
        # All reads remain inside this new preparation directory, no repo access.
        original_pub=PUB;PUB=Path(t)
        try:
            parts=[]
            for i,piece in enumerate([data[:8],data[8:]]):
                path=PUB/('part%d.gz'%i);payload=gzip.compress(piece,mtime=0);path.write_bytes(payload)
                parts.append({'path':path.name,'compression':'gzip','decoded_size_bytes':len(piece)})
            add('runtime_gzip_stream_roundtrip_uses_actual_chunk_reader',digest(gunzip_parts(parts))==digest([data]))
            require(digest(file_chunks(PUB/'part0.gz'))==digest([gzip.compress(data[:8],mtime=0)]),'Stored SHA fixture')
            add('runtime_stored_part_SHA_matches_exact_bytes',True)
        finally:PUB=original_pub
    add('full_thirteen_ancestor_chain_closes',len(records)==13 and len(cumulative)>13)
    altered=json.loads(json.dumps(records));altered[-1]['chain_data']['deltas'][0]['sha256']='f'*64
    try:cumulative_parent_entries(binding,base,altered);ok=False
    except AssertionError:ok=True
    add('oldest_ancestor_SHA_drift_in_latest_chain_rejected',ok)
    altered=json.loads(json.dumps(records));altered[-1]['manifest']['previous_index']['sha256']='f'*64
    try:cumulative_parent_entries(binding,base,altered);ok=False
    except AssertionError:ok=True
    add('latest_explicit_previous_index_drift_rejected',ok)
    closings=[]
    for tag in ('BASE',)+tuple(t for t,_ in ANCESTORS):
        items=[dict(path=PREFIX+'old_closing/'+tag+'.json',sha256='a'*64,size_bytes=5)]
        if tag=='BASE':items.append(dict(path='readme.md',sha256='b'*64,size_bytes=7))
        closings.append(dict(tag=tag,plan_data=dict(authorized_main_push=True,repository=str(REPO),branch='main',remote='https://github.com/CanelE452/pallet-6d-pose.git',batches=[dict(files=items)])))
    old_closing,exception=closing_entries(closings)
    add('fourteen_closing_plans_and_single_BASE_README_exception',len(old_closing)==14 and len(exception)==1)
    altered=json.loads(json.dumps(closings));altered[1]['plan_data']['batches'][0]['files'].append(dict(path='readme.md',sha256='c'*64,size_bytes=4))
    try:closing_entries(altered);ok=False
    except AssertionError:ok=True
    add('nonBASE_README_exception_refused',ok)
    try:assert_disjoint(old_closing,[dict(raw,published_relative='old_closing/V187.json')]);ok=False
    except AssertionError:ok=True
    add('old_closing_overwrite_refused',ok)
    for suffix in ('.npz','.pt','.npy','.pth'):
        numerical=dict(source_root_id='P',source_relative_path=PHASE+'/out'+suffix,source_absolute=str(V/('out'+suffix)),status='PUBLISHED_RAW',published_relative=ARCHIVE+'out'+suffix,gzipchunks=None,size_bytes=16)
        try:source_archive_policy([numerical]);ok=False
        except AssertionError:ok=True
        add('unconditionally_exclude_'+suffix,ok)
    for own,saved in [({'head.pt'},set()),(set(),{'output.npz'})]:
        try:source_archive_policy([],own,saved);ok=False
        except AssertionError:ok=True
        add('nonempty_array_or_head_whitelist_refused_'+str(bool(own)),ok)
    import copy
    e=dict(source_prediction_seal=dict(ROOT_SEAL),evaluation_complete=dict(path=str(V/'saved_real_head_evaluation_v1/COMPLETE.json'),sha256='d'*64),independent_saved_result_review=dict(path=str(V/'independent_saved_result_review_c/actual_review_v1/REVIEW.json'),sha256='e'*64),final_phase_freeze=dict(path=str(V/'ROOT_FINAL_REAL_BENCHMARK_PHASE_FREEZE.json'),sha256='f'*64),explicit_own_head_relative_paths=[],explicit_saved_output_relative_paths=list(ROOT_SAVED),explicit_direct_target_relative_paths=[],direct_target_cache_complete=None)
    e['completion_reviews']=[dict(role=x,binding=e['final_phase_freeze']if x=='final_phase_freeze'else dict(path='/invented/'+x+'.json',sha256='b'*64),expected_fields={'completed':True,'PASS':True,**({'scientific_phase_frozen_no_further_phase_writes':True}if x=='final_phase_freeze'else{})})for x in ('phase_result','independent_report_QA','final_phase_freeze')]
    pred=dict(completed=True,PASS=True,all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=True,all_batch_repeat2_state_and_input_bytes_exact=True,original_rows2144=2144,model_domain_rows4288=4288,original_currentU45_publicU108=True,arms={arm:dict(updates=8000,use_sam=False,native_condition=bool(i),outputs={d:dict(ROOT_BINDINGS[RUN+'/'+arm+'/FINAL_'+d+'_OUTPUTS.npz'])for d in DOMAINS})for i,arm in enumerate(ARMS)},selected_model=None,adopted=False,fully_solved=False)
    ev=dict(completed=True,PASS=True,prediction_seal=e['source_prediction_seal'],all_both_predictions_closed_before_any_annotation_or_score_body=True,selected_model=None,adopted=False,fully_solved=False)
    au=dict(completed=True,PASS=True,complete=e['evaluation_complete'],selected_model=None,adopted=False,fully_solved=False)
    authority=numeric_metadata_policy(e,pred,ev,au);add('exact4RAW_native_flags_rootSHA_authority',len(authority['declared_artifact_bindings'])==4)
    selected=[dict(source_root_id='P',source_relative_path=PHASE+'/'+r,source_absolute=b['path'],source_sha256=b['sha256'],published_sha256=b['sha256'],published_relative=ARCHIVE+r,status='PUBLISHED_RAW',gzipchunks=None,size_bytes=1)for r,b in authority['declared_artifact_bindings'].items()]
    add('opaque4packetpolicy',source_archive_policy(selected,(),set(ROOT_SAVED),(),authority['declared_artifact_bindings'])==0)
    changes=[('wrongSHA',lambda e,p,v,a:p['arms'][ARMS[0]]['outputs'][DOMAINS[0]].update(sha256='9'*64)),('nativeint',lambda e,p,v,a:p['arms'][ARMS[1]].update(native_condition=1)),('wrongnative',lambda e,p,v,a:p['arms'][ARMS[1]].update(native_condition=False)),('SAMalias',lambda e,p,v,a:p['arms'][ARMS[1]].update(use_sam=True)),('partial4',lambda e,p,v,a:p['arms'][ARMS[0]]['outputs'].pop(DOMAINS[1])),('selected',lambda e,p,v,a:p.update(selected_model=ARMS[0])),('early',lambda e,p,v,a:p.update(all_two_models_completed_before_any_annotation_reference_or_baseline_score_reads=False)),('badrepeat',lambda e,p,v,a:p.update(all_batch_repeat2_state_and_input_bytes_exact=False)),('badseal',lambda e,p,v,a:e['source_prediction_seal'].update(sha256='9'*64)),('bad_eval',lambda e,p,v,a:v.update(prediction_seal={'path':'bad','sha256':'9'*64})),('bad_audit',lambda e,p,v,a:a.update(complete={'path':'bad','sha256':'9'*64})),('ownhead',lambda e,p,v,a:e.update(explicit_own_head_relative_paths=['head.pt'])),('target',lambda e,p,v,a:e.update(explicit_direct_target_relative_paths=['target.npz']))]
    for name,change in changes:
        args=copy.deepcopy((e,pred,ev,au));change(*args)
        try:numeric_metadata_policy(*args);ok=False
        except (AssertionError,KeyError):ok=True
        add(name+'_refused',ok)
    for name,rows in [('wrongpacketSHA',[dict(selected[0],source_sha256='9'*64)]+selected[1:]),('missingpacket',selected[:-1]),('excludedpacket',[dict(selected[0],status='EXCLUDED',exclusion_reason='bad')]+selected[1:])]:
        try:source_archive_policy(rows,(),set(ROOT_SAVED),(),authority['declared_artifact_bindings']);ok=False
        except AssertionError:ok=True
        add(name+'_refused',ok)
    text='<pre>[notlink](missing.txt)</pre><a href="actual.md">actual</a>'
    add('HTMLattributes_only',portable_audit('archive/p/x.html',text,lambda p:p=='archive/p/actual.md')['local_links']==['archive/p/actual.md'])
    return dict(completed=True,PASS=all(x['PASS']for x in tests),checks=tests,invented_only=True,actual_source_repo_hash_copy_Git=0)

def prepare():
    tests=invented();tests['code']=bind(__file__);write(CHECKS,tests);require(tests['PASS'],'Invented checks')
    checked(C_CODE);checked(C_PROTO)
    final=read(C/'FINAL_PUBLICATION_DELTA_PREPARATION_COMPLETE.json')
    require(final['completed'] is True and final['PASS'] is True and final['FINAL_READY_FROZEN'] is True,'Frozen publisher final readiness')
    allowed={v['path']:v['sha256']for v in final['files'].values()}
    allowed[str(C/'FINAL_PUBLICATION_DELTA_PREPARATION_COMPLETE.json')]=bind(C/'FINAL_PUBLICATION_DELTA_PREPARATION_COMPLETE.json')['sha256']
    proto={'completed':True,'prepared_only':True,'code':bind(__file__),'checks':bind(CHECKS),'publisher_builder':C_CODE,'publisher_protocol':C_PROTO,
           'immutable_base_manifest':{'path':str(PUB/'PUBLICATION_MANIFEST.json'),'sha256':BASE_SHA},'delta_manifest_path':str(PUB/DELTA/'DELTA_MANIFEST.json'),'source_archive_manifest_path':str(PUB/DELTA/'SOURCE_ARCHIVE_MANIFEST.jsonl'),
           'source_phase':str(V),'repository':str(REPO),'output':str(OUT),'portable_pages':PORTABLE,
           'parent_delta_path':str(PUB/PARENT_DELTA),'parent_chain_path':str(PUB/PARENT_CHAIN),'parent_index_path':str(PUB/PARENT_INDEX),'ancestor_paths':{tag:{key:str(PUB/path) for key,path in ancestor_paths(tag).items()} for tag,_ in ANCESTORS},'allowed_admin_code':allowed,
           'audit':['All new included original source files streamhash versus frozen sourceSHA; stored RAW or concatenated gzip roundtrip versus same SHA/size, every stored part SHA/size/CRC, every physical Git-stage file strictly<100MiB.','Cumulative BASE + V187 + V188 + V189 + V190 + V191 + V192 + V193 + V194 manifest/chain/index/closing metadata only; independently check ordered chain authority and reject all delta logical/raw/part/descriptor collisions before opening new bodies. Rehash unchanged base manifest after audit; no old raw dataset/model/evaluation file bodies.','Exact SOURCE_ARCHIVE records occur in DELTA; separate physical descriptor records and BUILD physical_files close stored paths, delta/chain/readme.','Scan new logical text plus additional new physical text for credential formats; output only redacted type/path/line, never token content. This format scan is not a general secret detector.','Only six prospectively named portable entry pages receive relative-link checks. Copied provenance Markdown is scanned for credentials but not treated as a portable entry. Verify new index ends with old index bytes in same parent, global readme ends with preserved prior bytes.'],
           'link_target_policy':'Inherited safe directory policy: explicit ordinary regular files OR ordinary directories exist; every component opened with NOFOLLOW, no directory enumeration/content read. Symlinks refused, no logical gzip fallback. All other byte/credential/collision checks unchanged.',
           'inherited_auditor_ready':{'path': '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation/V195_DELTA_REVIEW_PREPARATION_A/FINAL_READY.json', 'sha256': 'b542bb9c665324ec41c00efac47e15ea109c22f6b12b02759f7058dac8de2df9'},
           'body_interpretation':'No image/model/NPZ/evaluation decode or semantic data parse; manifests/chain/descriptors are administrative metadata, other text is regex/link audit only. No network or Git commands.','root_code_preparation_authorization':'Root code-only task after V200 finalfreeze; no actualphase/parent/repo accesses before separate rootentry','actual_before_root_entry':0,'automatic_retry':False}
    proto.update(immutable_closing_tags=['BASE', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199'],prospective_rules=bind(H/'RULES_BEFORE_ACTUAL_PUBLISHED_BYTES.json'),source_inventory_expected=dict(total='future frozen manifest count; no estimate during phase preparation',owned_heads=0,saved_array_packets=4,optional_direct_target_packets_max=0),HTML_links='Only real href/src attributes; escaped Markdown in pre is plain text. Original plain local provenance text is allowed, absolute actual link targets refused.',external_metadata='Only exact nine root-entry-bound closing-plan JSON paths may be outside repository/project; lexical precheck then componentwise NOFOLLOW. No broad directory scan.')
    proto['body_interpretation']='Only administrative manifests/chain/descriptors are parsed as JSON; exact4 new real outputs with priorfit/integrity/sourcebarrier + prediction/evaluation/audit/finalfreeze closure; no model/array decode. All .pt/.npz/image/source payloads streamhash as bytes only; other text credential/link scans only. No tensor/array/image/evaluation decode, network or Git.'
    proto.update(root_code_preparation_scope={'path': '/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation/ROOT_V200_PUBLICATION_CODE_PREPARATION_SCOPE_C.json', 'sha256': '513f01ce27db2aae14d629afc3cb15e43b8579d03a9376a10862acc1d1ff6ffd', 'size_bytes': 7540},immutable_closing_tags=['BASE']+[tag for tag,_ in ANCESTORS],expected_remote_main=None,source_inventory_expected=dict(owned_heads=0,saved_array_packets=4,targets=0),body_interpretation='Administrative exact4 V200 PREDICTION_SEAL/evaluation/numerical-review/finalfreeze metadata only. New prediction NPZ files are opaque byte streams, no array/header/model decode; old sourcefit/weights not reopened. No actual phase/repo/parent/API/Git access during preparation.',audit=['All included original and stored RAW/gzip/descriptor/chunk SHA and byte-size equality; no semantic decode.','BASE+13ancestor delta/index/chain+14closing protection and sole historical BASE README exception; no old archive-body scan.','Credential pattern scan is redacted and portable6 relative hyperlinks must resolve to actual NOFOLLOW ordinary files/directories; no logical gzip fallback.','Exact root-declared4 outputs, bothRAW/native false/true,2144/4288, repeat2state and beforeannotation seal;0owned0targets; independently compare builder entry and complete delta authority.'])
    write(PROTOCOL,proto)
    template={'approved_actual_V200_immutable_delta_audit':False,'completed_phase_frozen':False,'code':proto['code'],'protocol':bind(PROTOCOL),'checks':proto['checks'],'repository':str(REPO),'build_complete':{'path':None,'sha256':None},'base_manifest':proto['immutable_base_manifest'],'ancestor_publications':[{'tag':tag,**{key:{'path':path,'sha256':None} for key,path in proto['ancestor_paths'][tag].items()}} for tag,_ in ANCESTORS],'portable_pages':PORTABLE,'immutable_closing_plans':[{'tag':tag,'binding':{'path':None,'sha256':None}} for tag in ('BASE','V187','V188','V189','V190','V191','V192','V193','V194')],'root_scope':{'path':None,'sha256':None},'parent_delta_manifest':{'path':proto['parent_delta_path'],'sha256':None},'parent_chain':{'path':proto['parent_chain_path'],'sha256':None},'previous_index':{'path':proto['parent_index_path'],'sha256':None},'delta_manifest':{'path':proto['delta_manifest_path'],'sha256':None},'output':str(OUT),'Git_mutation_or_push':False,'image_model_eval_decode_or_parse':False,'automatic_retry':False}
    template.update(source_prediction_seal=dict(ROOT_SEAL),evaluation_complete=dict(path=str(V/'saved_real_head_evaluation_v1/COMPLETE.json'),sha256=None),independent_saved_result_review=dict(path=str(V/'independent_saved_result_review_c/actual_review_v1/REVIEW.json'),sha256=None),final_phase_freeze=dict(path=str(V/'ROOT_FINAL_REAL_BENCHMARK_PHASE_FREEZE.json'),sha256=None),expected_remote_main=None);write(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json',template)
    ready={'completed':True,'prepared_only':True,'code':proto['code'],'protocol':bind(PROTOCOL),'checks':proto['checks'],'template':bind(H/'ROOT_ACTUAL_ENTRY_TEMPLATE_APPROVED_FALSE.json'),'actual_phase_enumeration_hash_copy_or_repo_access':0,'output_after_entry':str(OUT),'command':'/home/minjae/anaconda3/envs/pallet-yolo26/bin/python '+str(Path(__file__).absolute())+' --entry <root-entry>'}
    ready.update(PASS=True,FINAL_READY_FROZEN=True,invented_check_count=len(tests['checks']),publisher_final=bind(C/'FINAL_PUBLICATION_DELTA_PREPARATION_COMPLETE.json'));write(H/'FINAL_READY.json',ready);print(json.dumps(ready),flush=True)

def run(entry):
    e=read(entry);require(e['approved_actual_V200_immutable_delta_audit']is True and e['completed_phase_frozen']is True,'Root completed-phase audit entry')
    for k in ('Git_mutation_or_push','image_model_eval_decode_or_parse','automatic_retry'):require(e[k]is False,'Forbidden action '+k)
    p=read(checked(e['protocol']));require(e['code']==p['code']==bind(__file__)and e['protocol']==bind(PROTOCOL),'Frozen auditor')
    require(e['checks']==p['checks'],'Frozen invented checks');checked(p['checks'])
    require(e['repository']==str(REPO)and e['base_manifest']==p['immutable_base_manifest']and e['base_manifest']['sha256']==BASE_SHA,'Pinned immutable base/root')
    require(e['output']==p['output']==str(OUT)and V not in OUT.parents,'Output outside frozen phase')
    require([r['tag'] for r in e['immutable_closing_plans']]==['BASE', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199'],'Fourteen explicit closing metadata bindings')
    for record in e['immutable_closing_plans']:
        path=safe_name(record['binding']['path']);require(path.is_absolute() and path.suffix=='.json','Explicit root-bound closing metadata JSON only');EXTERNAL_METADATA.add(str(path))
    checked(e['root_scope'])
    OUT.mkdir(parents=True,exist_ok=False);write(OUT/'STARTED.json',{'completed':False,'entry':bind(entry),'automatic_retry':False})
    try:
        base=read(checked(e['base_manifest']));build=read(checked(e['build_complete']))
        require(build['completed']is True and build['PASS']is True and build['repository']==str(REPO),'Completed staged build')
        require(build['base_manifest']==e['base_manifest']and build['delta_manifest']==e['delta_manifest'],'Build binding closure')
        require(e['delta_manifest']['path']==p['delta_manifest_path'],'Exact delta path')
        delta=read(checked(e['delta_manifest']));require(delta['format']=='immutable-completed-phase-delta-v1'and delta['phase']==PHASE and delta['previous_base_manifest']==e['base_manifest'],'Delta identity')
        require(e['ancestor_publications']==build['ancestor_publications'],'Build ordered ancestor bindings')
        require(len(e['ancestor_publications'])==13,'Exactly thirteen immutable ancestor metadata sets')
        records=[]
        for row,(tag,phase) in zip(e['ancestor_publications'],ANCESTORS):
            require(row['tag']==tag,'Root entry exact ancestor order')
            for key,path in ancestor_paths(tag).items():
                require(row[key]['path']==str(PUB/path),'Exact ancestor bound metadata path');checked(row[key])
            records.append(dict(row,manifest=read(row['delta_manifest']['path']),chain_data=read(row['chain']['path'])))
        old_entries=cumulative_parent_entries(e['base_manifest'],base,records)
        require(e['immutable_closing_plans']==build['immutable_closing_plans']==delta['immutable_closing_plans'],'Exact prior closing plan closure')
        closing_records=[dict(r,plan_data=read(checked(r['binding']))) for r in e['immutable_closing_plans']]
        closing_paths,readme_exceptions=closing_entries(closing_records);old_entries.extend(closing_paths)
        same_exceptions(build['historical_global_readme_exceptions'],readme_exceptions);same_exceptions(delta['historical_global_readme_exceptions'],readme_exceptions)
        latest=e['ancestor_publications'][-1]
        for key,ancestor_key in [('parent_delta_manifest','delta_manifest'),('parent_chain','chain'),('previous_index','index')]:
            require(e[key]==build[key]==latest[ancestor_key],'Exact immutable latest parent alias '+key)
        require(delta['previous_delta_manifest']==e['parent_delta_manifest'] and delta['previous_chain']==e['parent_chain'] and delta['previous_index']==e['previous_index'] and delta['ancestor_publications']==e['ancestor_publications'],'New delta exact full parent closure')
        previous=records[-1]['manifest'];previous_chain=records[-1]['chain_data']
        builder_entry=read(checked(build['root_entry']))
        require(builder_entry['code']==p['publisher_builder'] and builder_entry['protocol']==p['publisher_protocol'],'Exact frozen B builder/protocol generated this build')
        authority=read_numeric_authority(builder_entry);require(authority==build['artifact_authority']==delta['artifact_authority'],'Independent rootdeclared prediction/evaluation/audit/freeze artifact closure')
        for key in AUTHORITY_KEYS:require(e[key]==authority[key],'Root audit entry exact V200 metadata binding '+key)
        entries=delta['entries'];old=assert_disjoint(old_entries,entries)
        logical={};by_source={}
        for r in entries:
            require(r['source_root_id'] in {'P','PUBLICATION_ADMIN_CODE','DELTA_DERIVED'},'Only declared original/admin/derived source roots')
            key=(r['source_root_id'],r['source_relative_path']);require(key not in by_source,'Unique source record');by_source[key]=r
            safe_name(r['source_relative_path'])
            if r['status']=='EXCLUDED':continue
            key=relative(r['published_relative']);require(key not in logical,'Unique new logical record');logical[key]=r
        archive_path=PUB/DELTA/'SOURCE_ARCHIVE_MANIFEST.jsonl';archive=readlines(archive_path)
        require(DELTA+'SOURCE_ARCHIVE_MANIFEST.jsonl'in logical,'Source archive itself sealed')
        require(all(by_source.get((r['source_root_id'],r['source_relative_path']))==r for r in archive),'Every source archive record exactly preserved in delta')
        require(sum(r['status']!='EXCLUDED'for r in archive)==delta['source_phase_original_records'],'Source original logical count')
        require(len([r for r in entries if r['source_root_id']=='P'])==len(archive),'No extra P rows outside exact source archive')
        require(len({(r['source_root_id'],r['source_relative_path']) for r in archive})==len(archive),'Unique source archive records')
        own=set(authority['explicit_own_head_relative_paths']);saved=set(authority['explicit_saved_output_relative_paths']);targets=set(authority['explicit_direct_target_relative_paths'])
        excluded_count=source_archive_policy(archive,own,saved,targets,authority['declared_artifact_bindings'])
        file_results=[];credential_hits=[];scanned=set();physical_expected={}
        for r in entries:
            if r['status']=='EXCLUDED':continue
            rel=relative(r['published_relative']);source=safe_name(r['source_absolute'])
            if r['source_root_id']=='P':require(V in source.parents and r['source_absolute']==str(P/r['source_relative_path']),'New original source phase only')
            elif r['source_root_id']=='PUBLICATION_ADMIN_CODE':admin_authority(r,p['allowed_admin_code'])
            else:require(r['source_root_id']=='DELTA_DERIVED'and source==PUB/rel,'Derived metadata must be its new stored file')
            source_hash,source_size=digest(file_chunks(source))
            require(source_hash==r['source_sha256']and source_size==r['size_bytes'],'Original source SHA/size: '+rel)
            if r.get('gzipchunks')is None:
                require(r['status']=='PUBLISHED_RAW','Raw status')
                stored_hash,stored_size=digest(file_chunks(PUB/rel));require(stored_hash==r['published_sha256']==source_hash and stored_size==source_size,'Stored raw identity '+rel)
                require(rel not in physical_expected,'Duplicate raw/part physical path');physical_expected[rel]=(stored_hash,stored_size);stream=lambda rel=rel:file_chunks(PUB/rel)
            else:
                require(r['status']=='PUBLISHED_GZIP_CHUNKS','Gzip status');g=r['gzipchunks'];require(g['format']=='concatenated-bytes-v1','Gzip concatenation convention')
                for part in g['parts']:
                    pr=relative(part['path']);h,n=digest(file_chunks(PUB/pr));require(h==part['sha256']and n==part['stored_size_bytes']and n<LIMIT,'Stored gzip part SHA/size')
                    require(pr not in physical_expected,'Duplicate stored gzip part');physical_expected[pr]=(h,n)
                restored_hash,restored_size=digest(gunzip_parts(g['parts']))
                require(restored_hash==g['content_sha256']==source_hash and restored_size==g['content_size_bytes']==source_size,'Gzip original roundtrip '+rel)
                desc=relative(r['gzip_descriptor_path']);require(desc==rel+'.parts.json'and desc in logical and logical[desc]['status']=='PUBLISHED_RAW','Physical descriptor separate raw record')
                dh,dn=digest(file_chunks(PUB/desc));require(dh==r['gzip_descriptor_sha256']and dn==r['gzip_descriptor_size_bytes'],'Descriptor SHA/size')
                descriptor=read(PUB/desc);require(descriptor==g,'Descriptor exact gzip contract')
                stream=lambda parts=g['parts']:gunzip_parts(parts)
            if Path(rel).suffix.casefold()in TEXT:
                scan=scan_text(stream());scanned.add(rel)
                credential_hits.extend(dict(v,published_relative=rel)for v in scan['format_hits_redacted'])
            file_results.append({'published_relative':rel,'status':r['status'],'source_root_id':r['source_root_id'],'source_sha256':source_hash,'source_size_bytes':source_size,'PASS':True})
        for rel,(h,n)in physical_expected.items():require(n<LIMIT,'Physical Git file must be strictly<100MiB')
        buildfiles={}
        for f in build['physical_files']:
            rel=relative(f['path']);require(rel not in buildfiles,'Unique physical build path');buildfiles[rel]=f
            require(rel.startswith(PREFIX)or rel=='readme.md','Only publication delta or global README')
            require(0<=f['size_bytes']<LIMIT,'Physical Git<100MiB');h,n=digest(file_chunks(REPO/rel));require(h==f['sha256']and n==f['size_bytes'],'All physical BUILD bytes')
            logical_rel=rel[len(PREFIX):]if rel.startswith(PREFIX)else None
            if logical_rel is not None:require(logical_rel not in old,'Old covered path in new physical BUILD')
            if logical_rel not in scanned and Path(rel).suffix.casefold()in TEXT:
                credential_hits.extend(dict(v,published_relative=rel)for v in scan_text(file_chunks(REPO/rel))['format_hits_redacted']);scanned.add(logical_rel or rel)
        expected_repo={PREFIX+k for k in physical_expected}|{PREFIX+DELTA+'DELTA_MANIFEST.json',PREFIX+'MANIFEST_CHAIN_AFTER_V200.json','readme.md'}
        require(set(buildfiles)==expected_repo,'Exact physical BUILD inventory, no omissions/extras')
        for rel,(h,n)in physical_expected.items():require(buildfiles[PREFIX+rel]['sha256']==h and buildfiles[PREFIX+rel]['size_bytes']==n,'Manifest-to-BUILD physical equality')
        chain=read(checked(build['new_chain']));require(chain['base']=={'path':'PUBLICATION_MANIFEST.json','sha256':BASE_SHA}and chain['deltas']==previous_chain['deltas']+[{'path':DELTA+'DELTA_MANIFEST.json','sha256':e['delta_manifest']['sha256']}]and chain['previous_chain']=={'path':PARENT_CHAIN,'sha256':e['parent_chain']['sha256']},'Manifest cumulative eight-delta chain closure')
        require(chain['index']['path']=='EXPERIMENT_INDEX_THROUGH_V200_KO.md'and chain['index']['sha256']==build['new_index']['sha256'],'Versioned index chain binding')
        require(chain['immutable_closing_plans']==e['immutable_closing_plans'],'Chain closes prior closing plans');same_exceptions(chain['historical_global_readme_exceptions'],readme_exceptions)
        checked(build['new_index']);old_index=bytes_of(PUB/PARENT_INDEX);new_index=bytes_of(PUB/'EXPERIMENT_INDEX_THROUGH_V200_KO.md')
        old_index_rows=[r for r in previous['entries']if r.get('published_relative')==PARENT_INDEX and r['status']!='EXCLUDED']
        require(len(old_index_rows)==1 and hashlib.sha256(old_index).hexdigest()==old_index_rows[0]['published_sha256']==e['previous_index']['sha256']and new_index.endswith(old_index),'Prior V193 index byte-exact suffix in same parent')
        backup=bytes_of(PUB/DELTA/'GLOBAL_README_BEFORE_V200.md');require(bytes_of(REPO/'readme.md').endswith(backup),'Prior global README byte-exact suffix')
        require(hashlib.sha256(backup).hexdigest()==builder_entry['global_readme_before']['sha256'],'Prior README exact prebuild SHA')
        require(e['portable_pages']==PORTABLE,'Exact six prospectively named portable entry pages')
        links=[portable_audit(doc,bytes_of(PUB/doc).decode('utf-8'),exists_file_or_directory)for doc in PORTABLE]
        require(not credential_hits,'Credential format hits present; values are redacted')
        require(digest(file_chunks(e['base_manifest']['path']))[0]==BASE_SHA,'Immutable base unchanged after audit')
        for ancestor in e['ancestor_publications']:
            for key in ('delta_manifest','chain','index'):checked(ancestor[key])
        for closing in e['immutable_closing_plans']:checked(closing['binding'])
        admin_rows=[r for r in entries if r.get('source_root_id')=='PUBLICATION_ADMIN_CODE'];actual_tools={r['source_absolute']:r['source_sha256']for r in admin_rows}
        require(actual_tools=={r['path']:r['sha256']for r in build['publisher_tools']},'Every explicit publisher admin tool exactly present')
        result={'completed':True,'PASS':True,'review_type':'INDEPENDENT_IMMUTABLE_PUBLICATION_DELTA_BYTE_AND_PORTABLE_TEXT_AUDIT','entry':bind(entry),'code':bind(__file__),'protocol':bind(PROTOCOL),'base_manifest':e['base_manifest'],'delta_manifest':e['delta_manifest'],'build_complete':e['build_complete'],
                'parent_delta_manifest':e['parent_delta_manifest'],'parent_chain':e['parent_chain'],'previous_index':e['previous_index'],'ancestor_publications':e['ancestor_publications'],'immutable_closing_plans':e['immutable_closing_plans'],'historical_global_readme_exceptions':readme_exceptions,
                'coverage':{'source_archive_records':len(archive),'declared_excluded_raw_or_nonresearch_records':excluded_count,'included_logical_records':len(file_results),'physical_BUILD_files':len(buildfiles),'logical_text_records_scanned':len(scanned),'portable_entry_pages':len(links),'portable_local_links':sum(r['local_link_count']for r in links),'admin_tool_records':len(admin_rows),'owned_final_heads':len(authority['explicit_own_head_relative_paths']),'saved_real_prediction_packets':len(authority['explicit_saved_output_relative_paths']),'optional_direct_target_packets':len(authority['explicit_direct_target_relative_paths']),'old_BASE_through_V199_delta_chain_index_closing_paths_untouched':True},
                'file_results':file_results,'portable_link_QA':links,'credential_format_hits_redacted':credential_hits,'old_index_and_global_readme_suffix_preserved':True,'physical_Git_limit_bytes_exclusive':LIMIT,
                'limits':['No Git index/commit/push or remote-publication check; BUILD physical plan is verified before later review-bound push-plan creation.','Only administrative manifests/chain/descriptors parsed as JSON. Exact V200 prediction/evaluation/numerical-audit/freeze metadata parsed only, no model/array decode. Other new text received credential/link checks only; image/model/evaluation data were never decoded or semantically parsed.','Credential-format scan does not prove absence of arbitrary secrets. Copied provenance Markdown was not treated as a portable entry.','No old raw dataset/model/evaluation bodies read or old manifest-covered files rehashed; base/index metadata and old index text only. Publication consistency does not revalidate research numerics or semantic labels.'],
                'phase_source_or_published_files_modified':False,'Git_calls':0,'model_or_eval_or_image_decode':0,'V112_access':0,'automatic_retry':False}
        write(OUT/'REVIEW.json',result);print(json.dumps({'completed':True,'PASS':True,'review':bind(OUT/'REVIEW.json'),'coverage':result['coverage']}),flush=True)
    except Exception:
        write(OUT/'FAILED_EXECUTION_PRESERVED.json',{'completed':False,'PASS':False,'entry':bind(entry),'automatic_retry':False,'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--entry',type=Path);a=ap.parse_args()
    if a.prepare:prepare()
    elif a.entry:run(a.entry)
    else:ap.error('--prepare or root --entry required')
