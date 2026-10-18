"""Whitelist small publication derivatives; offline checks, never Git or scoring.

Original private bytes, root reports and rendered assets are preserved. Only
declared derivative targets and this module's two metadata files are writable.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gzip
import hashlib
from html.parser import HTMLParser
import io
import json
import math
from pathlib import Path
import re
import shutil
import time
from urllib.parse import unquote,urlsplit

from .registry import arms,EXPECTED_N

NAMESPACE='pallet_risk_information_v1'
META={'PUBLIC_PACK_MANIFEST.json','PUBLIC_PACK_AUDIT.json','PUBLICATION_RECEIPT.json','REMOTE_PUBLICATION_RECEIPT.json'}
CSV_NAMES=('COMPARISON_ALL_ARMS.csv','ROW_METRICS.csv.gz','PAIRED_TRANSITIONS.csv.gz',
    'ACTION_SOURCE_COUNTS.csv','GAIN_DAMAGE_ACTUAL_SOURCE.csv','ALL_FAILURE_RISK.csv',
    'ALL_FAILURE_RISK_ROWS.csv.gz','THREE_WAY_SUCCESS.csv','SELECTION_DIAGNOSIS.csv')
BASE_JSON=(('EXPERIMENT_PLAN.json','EXPERIMENT_PLAN.json',True),
    ('INPUT_MANIFEST.json','INPUT_MANIFEST.json',True),('ENVIRONMENT.json','ENVIRONMENT.json',True),
    ('EXTERNAL_SOURCE_READING.json','EXTERNAL_SOURCE_READING.json',True),
    ('FEATURE_SPEC.json','FEATURE_SPEC.json',True),('F1_FEATURE_EXECUTION_SUMMARY.json','F1_FEATURE_EXECUTION_SUMMARY.json',True),
    ('V_GATE.json','V_GATE.json',True),('PREDICTION_MANIFEST.json','PREDICTION_MANIFEST.json',True),
    ('COST_AND_EXECUTION.json','COST_AND_EXECUTION.json',True),('VERIFICATION.json','VERIFICATION.json',True),
    ('PIXEL_REVIEW.json','PIXEL_REVIEW.json',True),('RIGHTS_AND_PRIVACY.json','RIGHTS_AND_PRIVACY.json',True),
    ('SUPERVISION_DIAGNOSTICS.json','SUPERVISION_DIAGNOSTICS.json',False),
    ('V_TRAINING_EXECUTION_RECEIPT.json','V_TRAINING_EXECUTION_RECEIPT.json',False),
    ('R_F0_COST_AND_EXECUTION.json','R_F0_COST_AND_EXECUTION.json',False),
    ('R_F1_COST_AND_EXECUTION.json','R_F1_COST_AND_EXECUTION.json',True),
    ('ORIGINAL_PRESERVATION_VERIFICATION.json','ORIGINAL_PRESERVATION_INITIAL_FAIL.json',True),
    ('ORIGINAL_PRESERVATION_NORMAL_MODE_VERIFICATION.json','ORIGINAL_PRESERVATION_VERIFICATION.json',True),
    ('b_reference_metadata_v1/RECEIPT.json','B_RENDERER_METADATA_RECEIPT.json',False),
    ('B_SAME_OBJECTIVE_UNIQUE_FRAME_SUMMARY.json','B_SAME_OBJECTIVE_UNIQUE_FRAME_SUMMARY.json',False),
    ('B_REFERENCE_TECHNICAL_AMENDMENT.json','B_REFERENCE_TECHNICAL_AMENDMENT.json',False),
    ('F1_TECHNICAL_PACKAGING_AMENDMENT.json','F1_TECHNICAL_PACKAGING_AMENDMENT.json',False))
PRIVATE_PATH=re.compile(r'/(?:home/[^/\s\"\'<>`]+|Users/[^/\s\"\'<>`]+|tmp/[^/\s\"\'<>`]+|root|mnt|media|private|var/tmp|run)/[^\s\"\'<>`\]\[),;]*')
SECRETS=(re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'\bgh[pousr]_[A-Za-z0-9]{24,}\b'),re.compile(r'\bgithub_pat_[A-Za-z0-9_]{30,}\b'),
    re.compile(r'\bAKIA[0-9A-Z]{16}\b'),re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}\b'),
    re.compile(r'(?i)authorization\s*[:=]\s*[\"\']?Bearer\s+[A-Za-z0-9._-]{20,}'))
BULK_KEYS={'model_outputs','model_state_dict','optimizer_state_dict','state_dict','candidates','banks',
    'endpoint_journal','feasible_pose_samples','optimizer_endpoints','raw_RGB','reference_pose',
    'R_fixed','native_GT_xyz','projected_GT_native_xy','candidate_metrics','action_metrics',
    'R_physical','R_cf','centroid','native_t','actual_pose','predictions','actions'}
EXTENSIONS={'.md','.json','.csv','.csv.gz','.jsonl','.jsonl.gz','.png','.jpg','.jpeg','.html','.css','.js','.svg'}
MAX_DERIVED_BYTES=16*1024*1024


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb')as stream:
        for b in iter(lambda:stream.read(1048576),b''):h.update(b)
    return h.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def no_duplicates(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('Duplicate JSON key: '+str(k))
        out[k]=v
    return out
def read(path):return json.loads(Path(path).read_text(),object_pairs_hook=no_duplicates)
def rows(path):
    opener=gzip.open if str(path).endswith('.gz')else open
    with opener(path,'rt')as f:
        for line in f:
            if line.strip():yield json.loads(line,object_pairs_hook=no_duplicates)
def relative(value):
    p=Path(value)
    if p.is_absolute() or '..'in p.parts or not p.parts:raise ValueError('Explicit relative path required: '+str(value))
    return p
def under(path,root):return Path(path).resolve().is_relative_to(Path(root).resolve())
def suffix(path):return ''.join(Path(path).suffixes[-2:])if str(path).endswith(('.csv.gz','.jsonl.gz'))else Path(path).suffix.lower()


def redact(value,counts):
    def replace(match):
        raw=match.group();counts['private_path_replacements']+=1
        return '${PRIVATE_PATH}/'+hashlib.sha256(raw.encode()).hexdigest()[:16]+'/'+Path(raw).name
    return PRIVATE_PATH.sub(replace,value)


def sanitized(value,counts,*,hide_bulk=True):
    """Preserve all failure/delta scalars and lists; never truncate by length."""
    if isinstance(value,dict):
        out={}
        for key,item in value.items():
            name=redact(str(key),counts)
            if name in out:raise ValueError('Sanitized dictionary-key collision: '+name)
            if hide_bulk and key in BULK_KEYS and isinstance(item,(list,dict)):
                out[name]=dict(publication='PRIVATE_BULK_NOT_COPIED',items=len(item),
                    canonical_payload_sha256=hashlib.sha256(canonical(item)).hexdigest())
                counts['bulk_payload_summaries']+=1
            else:out[name]=sanitized(item,counts,hide_bulk=hide_bulk)
        return out
    if isinstance(value,list):return [sanitized(v,counts,hide_bulk=hide_bulk)for v in value]
    if isinstance(value,str):return redact(value,counts)
    if isinstance(value,float)and not math.isfinite(value):raise ValueError('Nonfinite JSON value')
    return value


def scan_text(text,label):
    if PRIVATE_PATH.search(text):raise ValueError('Private path remains: '+label)
    if any(p.search(text)for p in SECRETS):raise ValueError('Credential-like content: '+label)
    if re.search(r'(?i)data:image/|file://',text):raise ValueError('Embedded/private image URL: '+label)


def assert_no_bulk(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key in BULK_KEYS and isinstance(item,(dict,list)):
                if not(isinstance(item,dict)and set(item)=={'publication','items','canonical_payload_sha256'}and
                    item['publication']=='PRIVATE_BULK_NOT_COPIED'and isinstance(item['items'],int)and
                    re.fullmatch('[0-9a-f]{64}',str(item['canonical_payload_sha256']))):
                    raise ValueError('Raw bank/reference/weight payload in public JSON: '+key)
            assert_no_bulk(item)
    elif isinstance(value,list):
        for item in value:assert_no_bulk(item)


def public_jsonl_check(path,relative_name):
    if relative_name!='RUN_LOG.jsonl':raise ValueError('Only sanitized root RUN_LOG.jsonl is public')
    scan_text(Path(path).read_text(),relative_name)
    count=0
    for row in rows(path):
        if not isinstance(row,dict):raise ValueError('Execution JSONL row must be an object')
        assert_no_bulk(row)
        # Reject nonfinite scalars instead of normalizing them silently.
        canonical(row);count+=1
    return count


def metadata_exclusions(args):
    extra=set(getattr(args,'late_metadata',[])or[])
    for name in extra:
        if relative(name).name!=name or not name.endswith('.json')or name in {'PIXEL_REVIEW.json','VERIFICATION.json','RIGHTS_AND_PRIVACY.json'}:
            raise ValueError('Late metadata must be an explicitly named root-level external-receipt JSON')
    return META|extra


def csv_bytes(path,counts):
    raw=gzip.decompress(Path(path).read_bytes())if str(path).endswith('.gz')else Path(path).read_bytes()
    reader=csv.DictReader(io.StringIO(raw.decode(),newline=''))
    if not reader.fieldnames:raise ValueError('CSV header missing')
    if set(reader.fieldnames)&{'R_fixed','R_physical','native_GT_xyz','actual_pose','model_outputs','state_dict'}:
        raise ValueError('Pose/bank CSV is outside numeric result allowlist')
    output=io.StringIO(newline='');writer=csv.DictWriter(output,fieldnames=reader.fieldnames);writer.writeheader()
    for row in reader:
        if None in row:raise ValueError('CSV row width differs')
        writer.writerow({k:redact(v,counts)if v is not None else v for k,v in row.items()});counts['CSV_rows']+=1
    text=output.getvalue();scan_text(text,Path(path).name)
    return gzip.compress(text.encode(),mtime=0)if str(path).endswith('.gz')else text.encode()


def derivative_bytes(source,mode):
    counts=Counter()
    if mode=='csv':data=csv_bytes(source,counts)
    elif mode=='summary':data=encoded(sanitized(read(source),counts))
    else:raise ValueError('Unknown small derivative transform: '+mode)
    return data,dict(counts)


def selected_ledger(records,registered):
    result=[]
    for row in records:
        if set(row['predictions'])!=set(registered):raise ValueError('Prediction arm coverage changed')
        predictions={}
        for arm in registered:
            entry=row['predictions'][arm];pose=entry.get('actual_pose',{})
            if type(pose.get('available'))is not bool:raise ValueError('Explicit pose availability required')
            actual={k:pose[k]for k in ('available','R_physical','centroid')if k in pose}
            if actual['available']and not all(k in actual for k in ('R_physical','centroid')):raise ValueError('Selected available pose missing')
            kept={k:entry[k]for k in ('selected_candidate_id','selected_source_aliases','selected_candidate_aliases',
                'action_role','fallback','technical_failure','prediction_status','status','reason')if k in entry}
            # Status dictionaries sometimes contain old solver traces. Retain
            # scalar status fields only; exact original record SHA binds all.
            for key,value in list(kept.items()):
                if isinstance(value,dict):kept[key]={k:v for k,v in value.items()if not isinstance(v,(dict,list))}
                elif isinstance(value,list)and not all(isinstance(v,str)for v in value):raise ValueError('Selected alias list is not strings')
            kept.update(actual_pose=actual,original_selected_record_sha256=hashlib.sha256(canonical(entry)).hexdigest())
            predictions[arm]=kept
        result.append({**{k:row[k]for k in ('id','partition','session','cluster_id')},'predictions':predictions})
    return result


def validate_seals(run,evaluation):
    manifest=read(run/'PREDICTION_MANIFEST.json');seal=read(evaluation/'EVALUATION_SEAL.json')
    if manifest['status']!='ALL_REGISTERED_OUTPUTS_SEALED_BEFORE_REAL_SCORING' or manifest.get('real_reference_inputs')is not False:
        raise ValueError('All predictions must be sealed before real references')
    registered=list(arms(bool(manifest['visual_executed'])))
    if manifest['arms']!=registered or len(manifest['ordered_ids'])!=238 or len(set(manifest['ordered_ids']))!=238:
        raise ValueError('Exact registry/238 predictions required')
    if manifest['experiment_plan_sha256']!=sha(run/'EXPERIMENT_PLAN.json'):raise ValueError('Original plan changed')
    source=Path(__file__).parent
    for name,expected in manifest.get('source_sha256',{}).items():
        if sha(source/relative(name))!=expected:raise ValueError('Prediction source changed: '+name)
    if manifest['artifacts']['PREDICTIONS.jsonl.gz']!=sha(run/'PREDICTIONS.jsonl.gz'):raise ValueError('Original predictions changed')
    if seal['status']!='COMPLETE' or seal['prediction_manifest_sha256']!=sha(run/'PREDICTION_MANIFEST.json'):
        raise ValueError('Completed evaluation must bind sealed predictions')
    if seal.get('evaluation_source_sha256')!=sha(source/'evaluate.py')or seal.get('registry_source_sha256')!=sha(source/'registry.py'):
        raise ValueError('Evaluated method/protocol source changed')
    for name,expected in seal['artifacts'].items():
        path=evaluation/relative(name)
        if not path.is_file()or not under(path,evaluation)or sha(path)!=expected:raise ValueError('Original evaluation artifact changed: '+name)
    result=read(evaluation/'EVALUATION.json')
    if result['arms']!=registered:raise ValueError('Result registry mismatch')
    for pop,n in EXPECTED_N.items():
        methods=result['populations'][pop]
        if set(methods)!=set(registered)or any(methods[a]['denominator']!=n for a in registered):
            raise ValueError('Full population/arm denominators missing: '+pop)
    records=list(rows(run/'PREDICTIONS.jsonl.gz'))
    if [r['id']for r in records]!=manifest['ordered_ids']:raise ValueError('Original prediction row ordering changed')
    return manifest,seal,registered,records


def rgb_inventory(path):
    data=list(rows(path))if str(path).endswith('.jsonl.gz')else read(path)['frames']
    out={}
    for r in data:
        if r.get('partition')not in (None,'TEST','REMAINDER'):continue
        fid=r['id']
        if fid in out:raise ValueError('Duplicate raw RGB ID')
        image=Path(r.get('path',r.get('image','')));digest=r.get('sha256',r.get('image_sha256'))
        if not image.is_file()or sha(image)!=digest:raise ValueError('Raw RGB hash mismatch: '+fid)
        out[fid]=dict(path=image,sha256=digest,raw_hw=r['raw_hw'])
    if len(out)!=192:raise ValueError('Exact PRIMARY192 raw RGB inventory required')
    return out


def privacy_binding(public,privacy_path,rights_path,inventory):
    privacy=read(privacy_path);rights=read(rights_path);gm=read(public/'gallery/GALLERY_MANIFEST.json')
    if rights.get('primary192_derived_RGB_authorized')is not True or rights.get('measurement_changed')is not False or rights.get('raw_RGB_publish')is not False:
        raise ValueError('Explicit PRIMARY192 derived-only rights required')
    if rights.get('privacy_plan_sha256')!=sha(privacy_path)or gm.get('privacy_plan_sha256')!=sha(privacy_path):
        raise ValueError('Privacy source hash chain mismatch')
    if privacy.get('measurement_or_inference_changes')is not False or set(privacy['frames'])!=set(inventory):raise ValueError('Exact original display-only plan required')
    rr=rights.get('inventory',[]);rights_rows={r['id']:r for r in rr};transforms=gm.get('display_transforms',[]);display={r['id']:r for r in transforms}
    if len(rr)!=192 or len(rights_rows)!=192 or len(transforms)!=192 or len(display)!=192 or set(display)!=set(inventory)or set(rights_rows)!=set(inventory):raise ValueError('Rights/gallery full192 inventory mismatch')
    for fid,original in inventory.items():
        p=privacy['frames'][fid];r=rights_rows[fid];d=display[fid]
        if any(x['original_image_sha256']!=original['sha256']for x in (p,r,d)):raise ValueError('Original RGB source chain mismatch')
        if r['raw_hw']!=original['raw_hw']or d['raw_hw']!=original['raw_hw']:raise ValueError('Original raw canvas mapping mismatch')
        if r['redactions']!=p['redactions']or d['display_only_redaction_boxes']!=p['redactions']or d['measurement_changed']is not False:raise ValueError('Display redaction mapping changed')
        H,W=original['raw_hw']
        for redaction in p['redactions']:
            x0,y0,x1,y1=redaction['xyxy_raw_pixels']
            if not 0<=x0<x1<=W or not 0<=y0<y1<=H:raise ValueError('Privacy box outside raw canvas')
        for name,target in d['paths'].items():
            dest=public/'gallery'/relative(target)
            if not dest.is_file()or not under(dest,public/'gallery'):raise ValueError('Gallery display asset absent: '+name)
    if gm.get('primary_images')!=192 or gm.get('original_RGB_bulk_published')is not False:raise ValueError('Gallery scope differs from PRIMARY192 derived panels')
    return dict(status='BOUND',privacy_plan_sha256=sha(privacy_path),rights_receipt_sha256=sha(rights_path),rows=192,
        rights_scope='PRIMARY192 derived RGB panels only; rawRGB and synthetic RGB not authorized by this receipt',
        human_review='NOT_IMPLIED',physical_owner_truth='UNVERIFIED')


def _backup_write(path,data,run,*,replace=False,known=False):
    if path.exists():
        if path.read_bytes()==data:return
        if not replace or not known:raise FileExistsError('Preserve existing file: '+str(path))
        backup=run/'publication_prior'/sha(path)/path.name;backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists():shutil.copyfile(path,backup)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)


def prepare(args):
    if not args.publication_GO:raise PermissionError('Explicit root --publication-GO required after all sealed')
    run=Path(args.run_dir).resolve();repo=Path(args.repo).resolve();public=Path(args.public_root).resolve()
    if public!=repo/'_docs/experiments'/NAMESPACE:raise ValueError('Only the new public namespace is writable')
    evaluation=Path(args.evaluation).resolve();pm,ev,registered,records=validate_seals(run,evaluation)
    inv=rgb_inventory(args.rgb_inventory);privacy=privacy_binding(public,args.privacy_plan,args.rights_receipt,inv)
    if set(inv)!={r['id']for r in records if r['partition']!='CAL'}:raise ValueError('RGB inventory and evaluated primary IDs differ')
    figures=read(public/'figures/FIGURES.json');gallery=read(public/'gallery/GALLERY_MANIFEST.json')
    if (figures.get('evaluation_sha256')!=sha(evaluation/'EVALUATION_SEAL.json')or gallery.get('prediction_manifest_sha256')!=sha(run/'PREDICTION_MANIFEST.json')or
        gallery.get('reference_sha256')!=ev['reference_sha256']):
        raise ValueError('Figure/gallery sealed-result provenance mismatch')
    if len(figures.get('images',[]))<7 or len(figures.get('machine_readable_data',[]))<7:raise ValueError('Seven required graph families and machine CSVs required')
    for image in figures['images']:
        if sha(public/'figures'/relative(image['file']))!=image['sha256']:raise ValueError('Figure changed after generation')
    for name in figures['machine_readable_data']:
        if not(public/'figures'/relative(name)).is_file():raise ValueError('Figure CSV missing')
    prior=read(public/'PUBLIC_PACK_MANIFEST.json')if(public/'PUBLIC_PACK_MANIFEST.json').exists()else{}
    excluded=metadata_exclusions(args)
    known={r['target']for r in prior.get('derivatives',[])};items=[];missing=[]
    def add(source,target,mode,required=True):
        source=Path(source);target=relative(target)
        if not source.is_file():
            if required:raise FileNotFoundError('Required final publication input: '+source.name)
            missing.append(dict(source=source.name,target=str(target),status='NOT_AVAILABLE_NOT_CLAIMED'));return
        if source.stat().st_size>MAX_DERIVED_BYTES:raise ValueError('Private bulk is not a small derivative: '+source.name)
        if str(target)in excluded or str(target).startswith(('gallery/','figures/','b_diagnostics')):raise ValueError('Root/rendered assets are preserved')
        items.append((source,target,mode))
    for name in CSV_NAMES:add(evaluation/name,name,'csv')
    add(evaluation/'EVALUATION.json','EVALUATION.json','summary');add(evaluation/'EVALUATION_SEAL.json','EVALUATION_SEAL.json','summary')
    for src,dst,required in BASE_JSON:
        source=run/src
        if not source.is_file()and(public/dst).is_file():source=public/dst
        add(source,dst,'summary',required)
    # Every discovered registered fit attempt survives, including failures.
    for folder in sorted((run/'training').glob('*')):
        if not folder.is_dir():continue
        if not re.fullmatch(r'(?:R_CONT_F0|R_ORDER_F0|R_ORDER_F1_ZERO|R_ORDER_F1|V_ORDER_NULL|V_ORDER_RGB)_s202610(?:18|19|20)',folder.name):raise ValueError('Unregistered training folder')
        for name in ('RUN_RECEIPT.json','FAILED_TECHNICAL.json','HISTORY.json'):
            add(folder/name,'training/'+folder.name+'/'+name,'summary',False)
    for directory in sorted(run.glob('independent_*')):
        if directory.is_dir():
            for name in ('VERIFICATION.json','FAILED_ATTEMPT.json','ORIGINAL_STRICT_ROUNDTRIP_FAILURES.json','RENDERER_METADATA_AVAILABILITY.json'):
                add(directory/name,'verification/'+directory.name+'/'+name,'summary',False)
    for directory in sorted((run/'execution').glob('*')):
        if directory.is_dir():add(directory/'RECEIPT.json','execution/'+directory.name+'/RECEIPT.json','summary',False)
    add(run/'independent_p0_v1/INPUTS_PUBLIC_SUMMARY.json','INPUTS_PUBLIC_SUMMARY.json','summary',True)
    add(run/'b_reference_support_v2/B_REFERENCE_SUPPORT_AUDIT.json','B_REFERENCE_SUPPORT_AUDIT.json','summary',True)
    if args.extra_map:
        for item in read(args.extra_map):
            src=run/relative(item['source']);dst=relative(item['target']);mode=item['mode']
            if mode not in ('summary','csv')or suffix(src)not in ('.json','.csv','.csv.gz')or suffix(src)!=suffix(dst):raise ValueError('Extras permit explicit small JSON/CSV summaries only')
            if dst.name in {'README.md','REPORT_KO.md','METHOD_AND_SOURCES.md','REPRODUCE.md'}:raise ValueError('Root documents preserved')
            if not under(src,run):raise ValueError('Extra source escaped private run')
            add(src,dst,mode,item.get('required',True))
    targets=[str(dst)for _,dst,_ in items]+['SELECTED_POSES.jsonl.gz']
    if len(targets)!=len(set(targets)):raise ValueError('Duplicate publication targets, never overwrite by iteration order')
    derivatives=[]
    for src,dst,mode in items:
        data,counts=derivative_bytes(src,mode)
        scan_text((gzip.decompress(data)if str(dst).endswith('.gz')else data).decode(),str(dst))
        _backup_write(public/dst,data,run,replace=args.replace_derived,known=str(dst)in known)
        derivatives.append(dict(target=str(dst),original_sha256=sha(src),public_sha256=sha(public/dst),
            original_relative_source=str(src.relative_to(run))if under(src,run)else str(src.relative_to(public))if under(src,public)else 'evaluation/'+src.name,
            original_source_scope='run'if under(src,run)else'public_preserved'if under(src,public)else'evaluation',
            mode=mode,transform_counts=counts))
    ledger=selected_ledger(records,registered);counts=Counter()
    raw=''.join(json.dumps(sanitized(r,counts,hide_bulk=False),ensure_ascii=False,allow_nan=False)+'\n'for r in ledger).encode()
    _backup_write(public/'SELECTED_POSES.jsonl.gz',gzip.compress(raw,mtime=0),run,replace=args.replace_derived,known='SELECTED_POSES.jsonl.gz'in known)
    derivatives.append(dict(target='SELECTED_POSES.jsonl.gz',original_sha256=sha(run/'PREDICTIONS.jsonl.gz'),
        public_sha256=sha(public/'SELECTED_POSES.jsonl.gz'),original_relative_source='PREDICTIONS.jsonl.gz',
        mode='selected_pose_whitelist',rows=238,arms=len(registered),private_bank_logits_weights_references_copied=False))
    payload={}
    for path in sorted(public.rglob('*')):
        if not path.is_file()or str(path.relative_to(public))in excluded:continue
        if path.is_symlink()or not under(path,public):raise ValueError('Public symlink escape')
        if suffix(path)not in EXTENSIONS:raise ValueError('Unregistered public file type: '+path.name)
        if suffix(path)=='.jsonl.gz'and str(path.relative_to(public))!='SELECTED_POSES.jsonl.gz':raise ValueError('Raw JSONL/bank is not public')
        if suffix(path)=='.jsonl':public_jsonl_check(path,str(path.relative_to(public)))
        payload[str(path.relative_to(public))]=sha(path)
    sources={str(p.relative_to(repo)):sha(p)for p in sorted((repo/'scripts/research'/NAMESPACE).glob('*.py'))}
    value=dict(schema='RISK_PUBLIC_PACK_MANIFEST_V1',status='PREPARED_OFFLINE_CHECK_REQUIRED',
        source_sha256=sha(__file__),prediction_manifest_sha256=sha(run/'PREDICTION_MANIFEST.json'),
        evaluation_seal_sha256=sha(evaluation/'EVALUATION_SEAL.json'),original_evaluation_artifacts=ev['artifacts'],
        rgb_inventory_sha256=sha(args.rgb_inventory),privacy_binding=privacy,registered_arms=registered,
        expected_denominators=EXPECTED_N,derivatives=derivatives,missing_optional=missing,artifacts=payload,
        source_files=sources,instruction_hashes={p.name:sha(p)for p in sorted((run/'instruction_bundle').iterdir())if p.is_file()},
        private_execution_log_sha256=sha(run/'RUN_LOG.jsonl')if(run/'RUN_LOG.jsonl').is_file()else None,
        copied_raw_RGB=False,copied_private_bank=False,copied_logits=False,copied_weights=False,
        human_review='NOT_IMPLIED',independent_physical_truth='UNVERIFIED',remote_publication='NOT_CHECKED_BY_THIS_TOOL',
        late_metadata_excluded_from_payload_hashes=sorted(excluded),
        self_hash_policy='Manifest/audit/external publication receipts excluded from payload hashes but checked for safe content; resultcommit A verified externally, immutable A snapshot referenced by metadata B')
    _backup_write(public/'PUBLIC_PACK_MANIFEST.json',encoded(value),run,replace=args.replace_derived,known=True)


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key in ('href','src')and value:self.links.append((value,key=='src'or tag in('img','image')))


def local_link(source,target,repo,*,image=False):
    parts=urlsplit(target.strip())
    if parts.scheme:
        if parts.scheme not in ('https','http','mailto')or image:raise ValueError('Unsafe/external embedded asset URL')
        return
    if parts.netloc:raise ValueError('Protocol-relative URL outside offline bundle')
    if not parts.path:return
    path=unquote(parts.path)
    if path.startswith('/'):raise ValueError('Absolute local link outside bundle')
    path=re.sub(r':\d+$','',path)
    dest=(source.parent/path).resolve()
    if not under(dest,repo)or not dest.exists():raise ValueError('Broken or escaping relative link: '+target)


def text_links(path,repo):
    text=path.read_text();scan_text(text,str(path.name))
    if path.suffix in('.html','.svg'):
        parser=Links();parser.feed(text)
        for link,image in parser.links:local_link(path,link,repo,image=image)
    elif path.suffix=='.md':
        for match in re.finditer(r'(!?)\[[^\]]*\]\(\s*(<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)',text):
            target=match[2].strip('<>');local_link(path,target,repo,image=bool(match[1]))


def image_check(path,*,public=True):
    from PIL import Image
    # verify must be the first operation after a fresh open. Do not call
    # getexif/info/load first: PNG.verify relies on its original file pointer.
    with Image.open(path)as image:image.verify()
    with Image.open(path)as image:
        image.load();exif=image.getexif()
        if public and image.format not in('PNG','JPEG'):raise ValueError('Only PNG/JPEG rendered assets are accepted')
        if public and(exif or image.info.get('exif')):raise ValueError('EXIF is not permitted in public rendered assets')
        if image.width*image.height>100_000_000:raise ValueError('Unexpected oversized raster')
        for value in image.info.values():
            if public and isinstance(value,str):scan_text(value,path.name+' metadata')
        rgb=image.convert('RGB');pixels=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        return dict(width=image.width,height=image.height,pixel_sha256=pixels)


def public_raster(path,raw_byte_hashes,raw_pixel_hashes):
    if sha(path)in raw_byte_hashes:raise ValueError('Original raw RGB bytes cannot be published')
    info=image_check(path)
    if info['pixel_sha256']in raw_pixel_hashes:raise ValueError('Reencoded original raw RGB pixels cannot be published')
    return info


def initialize_audit_destination(public,run):
    """Make this checker's own linked output explicit before link validation.

    A crash leaves NOT_YET_VERIFIED, never a fabricated successful audit.
    Existing PASS/FAIL receipts are preserved and are not reset here.
    """
    path=public/'PUBLIC_PACK_AUDIT.json'
    if path.exists():return False
    _backup_write(path,encoded(dict(schema='RISK_PUBLIC_PACK_OFFLINE_AUDIT_V1',
        status='NOT_YET_VERIFIED',source_sha256=sha(__file__),human_review=False,
        independent_physical_truth='UNVERIFIED',remote_publication='NOT_CHECKED')),run)
    return True


def verify(args):
    if not args.publication_GO:raise PermissionError('Explicit root --publication-GO required')
    run=Path(args.run_dir).resolve();repo=Path(args.repo).resolve();public=Path(args.public_root).resolve()
    if public!=repo/'_docs/experiments'/NAMESPACE:raise ValueError('Only the registered namespace is checked')
    start=time.monotonic();problems=[];checked=Counter();manifest=read(public/'PUBLIC_PACK_MANIFEST.json')
    excluded=metadata_exclusions(args)
    def check(name,predicate,detail=None):
        checked[name]+=1
        if not predicate:problems.append(dict(check=name,detail=detail))
    pm,ev,registered,records=validate_seals(run,Path(args.evaluation));inv=rgb_inventory(args.rgb_inventory)
    privacy=privacy_binding(public,args.privacy_plan,args.rights_receipt,inv)
    created_pending_audit=initialize_audit_destination(public,run)
    check('original_binding',manifest['prediction_manifest_sha256']==sha(run/'PREDICTION_MANIFEST.json')and manifest['evaluation_seal_sha256']==sha(Path(args.evaluation)/'EVALUATION_SEAL.json'))
    check('privacy_and_inventory_binding',manifest['rgb_inventory_sha256']==sha(args.rgb_inventory)and manifest['privacy_binding']==privacy)
    check('late_metadata_policy',set(manifest['late_metadata_excluded_from_payload_hashes'])==excluded)
    check('exact_public_inventory',{str(p.relative_to(public))for p in public.rglob('*')if p.is_file()and str(p.relative_to(public))not in excluded}==set(manifest['artifacts']))
    for name,expected in manifest['source_files'].items():check('source_hash',sha(repo/relative(name))==expected,name)
    for d in manifest['derivatives']:
        src=run/relative(d['original_relative_source'])
        if d.get('original_source_scope')=='public_preserved':src=public/relative(d['original_relative_source'])
        elif d.get('original_source_scope')=='evaluation':
            src=Path(args.evaluation)/Path(d['original_relative_source']).name
        check('original_derivative_source',src.is_file()and sha(src)==d['original_sha256'],d['target'])
        check('public_derivative_hash',sha(public/relative(d['target']))==d['public_sha256'],d['target'])
        if d['mode'] in ('summary','csv'):
            expected_data,_=derivative_bytes(src,d['mode'])
            check('derivative_transform_exact',expected_data==(public/relative(d['target'])).read_bytes(),d['target'])
    # Rebuild selected-pose whitelist and compare every row, preserving all
    # actual returned poses while excluding private action scores/bank arrays.
    expected=sanitized(selected_ledger(records,registered),Counter(),hide_bulk=False)
    check('selected_poses_exact',list(rows(public/'SELECTED_POSES.jsonl.gz'))==expected)
    raw_bytes={r['sha256']for r in inv.values()};raw_pixels=set()
    for record in inv.values():
        info=image_check(record['path'],public=False)
        if [info['height'],info['width']]!=record['raw_hw']:raise ValueError('Original raw canvas differs')
        raw_pixels.add(info['pixel_sha256'])
    raster={}
    for name,digest in manifest['artifacts'].items():
        path=public/relative(name)
        try:
            if path.is_symlink()or not under(path,public):raise ValueError('Public file escaped namespace')
            check('artifact_hash',sha(path)==digest,name)
            check('no_raw_RGB_same_bytes',sha(path)not in raw_bytes,name)
            ext=suffix(path)
            if ext not in EXTENSIONS:raise ValueError('Forbidden extension')
            if ext=='.jsonl.gz'and name!='SELECTED_POSES.jsonl.gz':raise ValueError('Only root selected-pose ledger is public compressed JSONL')
            if ext in ('.png','.jpg','.jpeg'):
                info=public_raster(path,raw_bytes,raw_pixels);raster[name]=info
                check('no_raw_RGB_same_pixels',info['pixel_sha256']not in raw_pixels,name)
            else:
                data=gzip.decompress(path.read_bytes())if ext in ('.csv.gz','.jsonl.gz')else path.read_bytes()
                scan_text(data.decode(),name)
                if ext=='.json':assert_no_bulk(read(path))
                if ext=='.jsonl':checked['execution_JSONL_rows']+=public_jsonl_check(path,name)
                if ext in ('.md','.html','.svg'):text_links(path,repo)
            checked['file_decoded']+=1
        except Exception as exc:problems.append(dict(check='public_asset',file=name,error_type=type(exc).__name__,reason=str(exc)))
    check('required_final_docs',all((public/n).is_file()for n in ('README.md','REPORT_KO.md','REPRODUCE.md','METHOD_AND_SOURCES.md')))
    # Model pixel review is a separately sourced statement, never inferred
    # from PIL decode. Check hashes only for actually claimed reviewed files.
    review=read(public/'PIXEL_REVIEW.json')
    check('review_kind_explicit',review.get('reviewer_kind')=='model_visual_review_not_human')
    for item in review.get('reviewed',[]):check('pixel_review_hash',sha(public/relative(item['file']))==item['sha256'],item['file'])
    for name in excluded:
        if name in {'PUBLIC_PACK_AUDIT.json'}or not(public/name).is_file():continue
        value=read(public/name);scan_text(json.dumps(value,ensure_ascii=False),name);assert_no_bulk(value)
        checked['late_metadata_safe_content']+=1
    audit=dict(schema='RISK_PUBLIC_PACK_OFFLINE_AUDIT_V1',status='PASS'if not problems else'FAIL',
        failure_count=len(problems),problems=problems,checks=dict(checked),public_artifacts=len(manifest['artifacts']),
        raster_checks=raster,manifest_sha256=sha(public/'PUBLIC_PACK_MANIFEST.json'),
        source_sha256=sha(__file__),wall_seconds=time.monotonic()-start,
        actual_pixel_review_status=review.get('status','NOT_REPORTED'),
        pixel_review_performed_by_this_checker=False,human_review=False,independent_physical_truth='UNVERIFIED',
        remote_publication='NOT_CHECKED; offline artifact validation only',git_mutations=0)
    _backup_write(public/'PUBLIC_PACK_AUDIT.json',encoded(sanitized(audit,Counter())),run,
        replace=args.replace_derived or created_pending_audit,known=True)
    if problems:raise ValueError('Public offline check failed: '+str(len(problems)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','verify'))
    for name in ('run-dir','repo','public-root','evaluation','rgb-inventory','privacy-plan','rights-receipt'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--extra-map',type=Path);p.add_argument('--publication-GO',action='store_true')
    p.add_argument('--late-metadata',action='append',default=[],help='Root-level later external verification receipt; no payload self-hash')
    p.add_argument('--replace-derived',action='store_true',help='Back up previous generated derivatives/metadata privately; never overwrite root reports')
    a=p.parse_args();prepare(a)if a.command=='prepare'else verify(a)


if __name__=='__main__':main()
