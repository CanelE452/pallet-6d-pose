"""Explicit public-code/derived-result packaging; no scientific imports or work.

Build and verify a private exact-file allowlist now. Copy is a separate root-only
operation,creates a fresh private staging directory,and never commits/pushes.
"""
from __future__ import annotations
import argparse,ast,csv,hashlib,html,json,posixpath,re,shutil
from pathlib import Path,PurePosixPath
from urllib.parse import urlsplit,unquote

NS='pallet_bi_full_execution_v1'
CODE_PREFIX='scripts/research/'+NS+'/'
DOC_PREFIX='_docs/experiments/'+NS+'/'
GALLERIES={
 'S1_STAGE_GALLERY_V4':'S1_ORIGINAL_H1_vs_PF_UNION',
 'S3_S4_STAGE_GALLERY_V3':'S3_S4_ORIGINAL_H1_vs_UNIONS',
 'STAGE_GALLERY_S3_BI_VS_S3_COMMON_V2':'S3_COMMON_vs_BI',
 'STAGE_GALLERY_S4_CONTOUR_VS_S4_PL_V2':'S4_PL_vs_CONTOUR',
 'STAGE_GALLERY_S4_IMAGE_VS_S4_PL_V2':'S4_PL_vs_IMAGE',
 'STAGE_GALLERY_S4_IMAGE_VS_S4_CONTOUR_V2':'S4_CONTOUR_vs_IMAGE'}
DOCS=('PUBLIC_PACKAGE_PLAN.md','PUBLIC_STAGE_STATUS_DRAFT.csv','PUBLIC_REQUIREMENTS_STATUS_DRAFT.csv',
 'PUBLIC_CODE_ALLOWLIST_DRAFT.json','S1_D_EVIDENCE_DRAFT.md','S3_S4_EVIDENCE_DRAFT.md',
 'S3_S4_FULL_PRECISION_METRICS.csv','S3_S4_FULL_PRECISION_METRICS.json','METHOD_DIFFERENCE_LEDGER_DRAFT.csv',
 'METHOD_DIFFERENCE_LEDGER_DRAFT.json','REPRODUCE.md')
FORBIDDEN_KEYS={'R','t','R_fixed','R_physical','R_cf','t_gt','centroid','actual_pose','reference_pose','reference'}
NEURAL_PAYLOAD_KEYS={'state_dict','model_state_dict','optimizer_state_dict','training_normalizer',
 'normalizer','neural_weights','checkpoint_tensors','tensor_values','model_parameters'}
PUBLIC_ROTATION_TEXT={'deg','degrees','COMPLETE_NEW','COMPLETE_EXACT_REUSE','DIAGNOSTIC_ONLY',
 'BLOCKED_INPUT','BLOCKED_TECHNICAL','INCOMPLETE','NOT_APPLICABLE_WITH_REASON'}
FINAL_SCHEMA='EXPLICIT_FINAL_PUBLIC_PACKAGE_ALLOWLIST_V1'
PUBLICATION_STATUS='PREPARED_NOT_YET_COMMITTED_OR_PUSHED'
KINDS={'code','documentation','aggregate_metrics','gallery_html','gallery_derived_jpeg',
       'gallery_derived_metadata','scatter_html','scatter_derived_data','scatter_figure','orientation_derived_jpeg',
       'saved_metric_table','saved_metric_metadata'}
FORBIDDEN_SUFFIXES={'.pt','.pth','.ckpt','.onnx','.npy','.npz','.gz','.zip','.usd','.obj','.glb','.mtl','.pickle','.pkl','.bin'}

def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def write_new(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise ValueError('Preserve prior packaging record;choose a fresh path')
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def clean_relative(value):
    p=PurePosixPath(value)
    if not isinstance(value,str)or not value or '\\'in value or p.is_absolute()or any(v in ('..','.')for v in p.parts)or str(p)!=value:
        raise ValueError('Unsafe/noncanonical relative package destination')
    return value

def no_symlink(path):
    path=Path(path).absolute()
    if any(p.is_symlink()for p in (path,*path.parents)):raise ValueError('Symlink source/output is prohibited')
    if not path.is_file():raise ValueError('Regular allowlisted source file required')
    return path

def rotation_metadata(value):
    if isinstance(value,str):return value in PUBLIC_ROTATION_TEXT
    if isinstance(value,list):return value==['<=5','(5,45]','>45']
    if isinstance(value,dict):return value==dict(errors_sums_and_mean_contributions='deg',squared_deviations_and_variance_contributions='deg^2')
    return False

def guard_json(value):
    if isinstance(value,dict):
        if any(key in FORBIDDEN_KEYS and not(key=='R'and rotation_metadata(v))for key,v in value.items()):raise ValueError('Raw pose/reference field in public result JSON')
        if NEURAL_PAYLOAD_KEYS&set(value):raise ValueError('Neural checkpoint/tensor/training-normalizer payload in public JSON')
        for key,v in value.items():
            if key in ('raw_RGB_copied','reference_pose_or_GT_published','GT_pose_or_raw_RGB_included')and v is not False:raise ValueError('Public no-raw/GT assertion differs')
            guard_json(v)
    elif isinstance(value,list):
        for v in value:guard_json(v)

def checked_source(path,root):
    p=no_symlink(path);base=Path(root).resolve()
    if p.resolve().parent!=base:raise ValueError('Only an exact direct child of the declared source directory is allowed')
    return p

def build(repo,run,manifest):
    repo,run=Path(repo).resolve(),Path(run).resolve();docs=repo/DOC_PREFIX;catalog=read(docs/'PUBLIC_CODE_ALLOWLIST_DRAFT.json')
    if catalog['schema']!='EXPLICIT_PUBLIC_CODE_FILENAMES_V1':raise ValueError('Explicit code filename catalog required')
    visual=read(run/'DERIVED_VISUAL_REVIEW/VISUAL_REVIEW_SEAL.json');execution=read(run/'DERIVED_VISUAL_EXECUTION/FINAL_EXECUTION_SEAL.json')
    reviewpath=run/'DERIVED_VISUAL_REVIEW/VISUAL_REVIEW_SEAL.json'
    if not visual['complete']or not visual['all_public_pose_GT_guards_PASS']or not execution['complete']or execution['artifact_seals'].get(str(reviewpath))!=sha(reviewpath):raise ValueError('Completed authenticated visual review required')
    gallery_hashes={g['name']:g['manifest_sha256']for g in visual['galleries']}
    entries=[];bindings={str(reviewpath):sha(reviewpath),str(run/'DERIVED_VISUAL_EXECUTION/FINAL_EXECUTION_SEAL.json'):sha(run/'DERIVED_VISUAL_EXECUTION/FINAL_EXECUTION_SEAL.json')}
    def add(path,dest,kind,evidence=None):
        path=no_symlink(path);dest=clean_relative(dest)
        entries.append(dict(source=str(path),destination=dest,kind=kind,bytes=path.stat().st_size,sha256=sha(path),evidence=evidence))
    for name in catalog['filenames']:
        if '/'in name or Path(name).suffix not in ('.py','.md'):raise ValueError('Only explicit namespace code/text filenames allowed')
        p=checked_source(repo/CODE_PREFIX/name,repo/CODE_PREFIX)
        add(p,CODE_PREFIX+name,'code')
    for name in DOCS:
        p=checked_source(docs/name,docs);add(p,DOC_PREFIX+name,'aggregate_metrics'if name.startswith('S3_S4_FULL_PRECISION_METRICS')else'documentation')
    for source_name,final_name in GALLERIES.items():
        folder=run/source_name;mp=folder/'GALLERY_MANIFEST.json';m=read(mp)
        if sha(mp)!=gallery_hashes[source_name]or not m['complete']or m['CAL_RGB_opened']!=0 or m['raw_RGB_copied']or m['public_reference_pose_or_GT']:raise ValueError('Final PRIMARY gallery/privacy closure differs')
        bindings[str(mp)]=sha(mp)
        for name,digest in m['artifact_sha256'].items():
            p=checked_source(folder/name,folder)
            if sha(p)!=digest:raise ValueError('Final gallery bytes differ')
            if name=='index.html':kind='gallery_html'
            elif re.fullmatch(r'case_\d{4}(?:_context|_families)?\.jpg',name):kind='gallery_derived_jpeg'
            elif re.fullmatch(r'case_\d{4}\.json',name):
                v=read(p);guard_json(v)
                if v['partition']not in ('TEST','REMAINDER')or v['reference_pose_or_GT_published']or v['raw_RGB_copied']:raise ValueError('PRIMARY derived metadata only')
                kind='gallery_derived_metadata'
            else:raise ValueError('Unexpected artifact in final gallery')
            add(p,DOC_PREFIX+'galleries/'+final_name+'/'+name,kind,dict(seal_sha256=sha(mp),artifact_key=name))
    sf=run/'S1_S3_S4_COMPARISON_FIGURES';sp=sf/'FIGURE_MANIFEST.json';s=read(sp)
    if not s['complete']or not s['all_outliers_retained']or execution['artifact_seals'].get(str(sp))!=sha(sp):raise ValueError('Completed full-axis scatter closure differs')
    bindings[str(sp)]=sha(sp)
    for name,kind in [('index.html','scatter_html'),('SCATTER_DATA.json','scatter_derived_data'),('REPRESENTATIVE_FULL.png','scatter_figure'),('REPRESENTATIVE_FULL.svg','scatter_figure')]:
        p=checked_source(sf/name,sf)
        if sha(p)!=s['artifacts'][name]:raise ValueError('Completed figure differs')
        add(p,DOC_PREFIX+'figures/S1_S3_S4_T_horizontal_R_vertical/'+name,kind,dict(seal_sha256=sha(sp),artifact_key=name))
    of=run/'S1_ORIENTATION_SUPPLEMENT';op=of/'DISPLAY_SEAL.json';v=read(op);p=of/'ORIENTATION_NATIVE_AXES.jpg'
    if not v['complete']or v['GT_numeric_JSON_exported']or v['raw_RGB_copied']or sha(p)!=v['artifact_sha256'][p.name]or execution['artifact_seals'].get(str(op))!=sha(op):raise ValueError('Orientation display closure differs')
    bindings[str(op)]=sha(op);add(p,DOC_PREFIX+'figures/S1_orientation/ORIENTATION_NATIVE_AXES.jpg','orientation_derived_jpeg',dict(seal_sha256=sha(op),artifact_key=p.name))
    tables=run/'S1_S3_S4_PUBLIC_METRICS';tp=tables/'EXPORT_SEAL.json';t=read(tp)
    if not t['complete']or t['schema']!='PUBLIC_SAVED_METRICS_EXPORT_SEAL_V1'or t['identities']!=238 or t['metrics_recomputed']or t['raw_references_decoded']or t['solve_fit_forward_calls']:
        raise ValueError('Completed saved-scalar metric export required')
    bindings[str(tp)]=sha(tp)
    for name in ('ALL_POPULATION_METRICS.csv','ROW_METRICS.csv','METRICS_SCHEMA.json','INPUT_BINDINGS.json'):
        p=checked_source(tables/name,tables)
        if sha(p)!=t['artifacts'][name]:raise ValueError('Saved scalar metric export differs')
        add(p,DOC_PREFIX+'tables/S1_S3_S4_saved_metrics/'+name,'saved_metric_table'if p.suffix=='.csv'else'saved_metric_metadata',dict(seal_sha256=sha(tp),artifact_key=name))
    value=dict(schema='EXPLICIT_PUBLIC_PACKAGE_ALLOWLIST_DRAFT_V1',draft=True,whole_campaign_complete=False,
        publication_scope='S1/D/S3/S4 completed snapshot;S2/F pending. Public code,derived PRIMARY redacted galleries,error plots and small report tables/docs only.',
        root_git_commit_push_only=True,galleries_copied_during_preparation=False,
        package_base_execution_sha='3553202a3682faeb7be2e7df051b230683965382',
        source_roots=[str(repo/CODE_PREFIX),str(docs)]+[str(run/n)for n in GALLERIES]+[str(sf),str(of),str(tables)],
        maximum_file_bytes=4*1024*1024,maximum_total_bytes=160*1024*1024,
        excluded=['private24structure review and originalimages','CAL RGB','raw real/synthetic RGB','reference files or physical R/t','weights/checkpoints','CAD/texture/assets','banks/features/labels/normalizers','raw observation/FRAME/RESULT/extras archives','superseded galleries','stale REPORT_SCAFFOLD_KO.txt/GAPS_HANDOFF_DRAFT.txt/initial matrix','original worktree/paper/default files','Git history/index/credentials'],
        source_evidence_bindings=bindings,stage_status_source=DOC_PREFIX+'PUBLIC_STAGE_STATUS_DRAFT.csv',
        entries=sorted(entries,key=lambda e:e['destination']),declared_files=len(entries),declared_total_bytes=sum(e['bytes']for e in entries))
    write_new(manifest,value);return verify(manifest)

def links_in(text,suffix):
    if suffix=='.html':return re.findall(r'(?:src|href)=["\']([^"\']+)["\']',text)
    if suffix=='.md':return re.findall(r'\]\(([^)]+)\)',text)
    return []

def verify(manifest):
    m=read(manifest)
    if m.get('schema')=='EXPLICIT_FINAL_PUBLIC_PACKAGE_ALLOWLIST_V1':
        from . import public_final
        return public_final.verify(manifest)
    if m['schema']!='EXPLICIT_PUBLIC_PACKAGE_ALLOWLIST_DRAFT_V1'or m['whole_campaign_complete']is not False:raise ValueError('Explicit incomplete snapshot manifest required')
    roots={str(Path(p).resolve())for p in m['source_roots']};destinations=set();total=0;code_names=[];catalog_names=set()
    bound_seals={}
    for path,digest in m['source_evidence_bindings'].items():
        if sha(no_symlink(path))!=digest:raise ValueError('Private completed evidence seal changed')
        bound_seals.setdefault(digest,[]).append((Path(path),read(path)))
    for e in m['entries']:
        d=clean_relative(e['destination']);p=no_symlink(e['source'])
        if str(p.resolve().parent)not in roots or e['kind']not in KINDS:raise ValueError('Unapproved source root/kind')
        if d in destinations or Path(d).suffix.lower()in FORBIDDEN_SUFFIXES or p.suffix.lower()in FORBIDDEN_SUFFIXES:raise ValueError('Duplicate/prohibited source or output')
        if not d.startswith((CODE_PREFIX,DOC_PREFIX)):raise ValueError('Only new namespace/report destination allowed')
        if p.name!=PurePosixPath(d).name:raise ValueError('Source basename must be retained')
        if e['kind'] in ('documentation','aggregate_metrics') and d not in {DOC_PREFIX+n for n in DOCS}:raise ValueError('Unlisted report document/table')
        if e['kind'].startswith('gallery_'):
            basename=PurePosixPath(d).name
            if not d.startswith(DOC_PREFIX+'galleries/'):raise ValueError('Gallery destination outside final derived gallery tree')
            expected='gallery_html'if basename=='index.html'else'gallery_derived_jpeg'if re.fullmatch(r'case_\d{4}(?:_context|_families)?\.jpg',basename)else'gallery_derived_metadata'if re.fullmatch(r'case_\d{4}\.json',basename)else None
            if e['kind']!=expected:raise ValueError('Unexpected derived gallery filename/type')
        if e['kind'].startswith('saved_metric_'):
            if d not in {DOC_PREFIX+'tables/S1_S3_S4_saved_metrics/'+n for n in ('ALL_POPULATION_METRICS.csv','ROW_METRICS.csv','METRICS_SCHEMA.json','INPUT_BINDINGS.json')}:raise ValueError('Unexpected saved scalar table filename')
        if e['kind']not in ('code','documentation','aggregate_metrics'):
            proof=e.get('evidence')or{};valid=False
            for sealpath,seal in bound_seals.get(proof.get('seal_sha256'),[]):
                artifacts=seal.get('artifact_sha256',seal.get('artifacts',{}))
                if sealpath.parent.resolve()==p.resolve().parent and proof.get('artifact_key')==p.name and artifacts.get(p.name)==e['sha256']:valid=True
            if not valid:raise ValueError('Derived image/data entry lacks its exact completed producer-display seal')
        destinations.add(d);size=p.stat().st_size
        if size!=e['bytes']or size>m['maximum_file_bytes']or sha(p)!=e['sha256']:raise ValueError('Allowlisted file bytes/size differ')
        total+=size
        if e['kind']=='code':
            if not d.startswith(CODE_PREFIX):raise ValueError('Code outside new namespace')
            if p.suffix not in ('.py','.md')or '/'in d[len(CODE_PREFIX):]:raise ValueError('Only direct namespace .py/.md code filenames are allowed')
            code_names.append(p.name)
            if p.suffix=='.py':ast.parse(p.read_text())
        elif p.suffix=='.json':
            value=read(p);guard_json(value)
            if d==DOC_PREFIX+'PUBLIC_CODE_ALLOWLIST_DRAFT.json':
                if value.get('schema')!='EXPLICIT_PUBLIC_CODE_FILENAMES_V1':raise ValueError('Explicit public code catalog schema differs')
                catalog_names=set(value['filenames'])
            if e['kind']=='gallery_derived_metadata'and value.get('partition')not in ('TEST','REMAINDER'):raise ValueError('CAL/public gallery metadata is prohibited')
        elif p.suffix=='.csv':
            with p.open(newline='')as stream:
                if set(next(csv.reader(stream),[]))&FORBIDDEN_KEYS:raise ValueError('Raw pose/reference CSV column')
        elif p.suffix in ('.jpg','.png'):
            head=p.open('rb').read(16)
            if p.suffix=='.jpg'and not head.startswith(b'\xff\xd8\xff'):raise ValueError('Not a derived JPEG')
            if p.suffix=='.png'and not head.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('Not a derived PNG')
    if set(code_names)-catalog_names:raise ValueError('Code filename is absent from the exact public catalog')
    if total!=m['declared_total_bytes']or len(destinations)!=m['declared_files']or total>m['maximum_total_bytes']:raise ValueError('Package file/total byte limit differs')
    link_count=0
    for e in m['entries']:
        p=Path(e['source'])
        for link in links_in(p.read_text(),p.suffix)if p.suffix in ('.md','.html')else []:
            url=urlsplit(html.unescape(link))
            if url.scheme in ('http','https','mailto')or not url.path:continue
            if url.scheme or url.netloc or url.path.startswith('/'):raise ValueError('Nonportable/private file link')
            target=posixpath.normpath(posixpath.join(posixpath.dirname(e['destination']),unquote(url.path)))
            if target not in destinations:raise ValueError('Missing local public package link: '+target)
            link_count+=1
    return dict(PASS=True,manifest_sha256=sha(manifest),allowlisted_files=len(destinations),bytes=total,local_links_verified=link_count,
        heavy_scientific_imports_or_calls=0,raw_RGB_references_or_weights_copied=0,gallery_files_copied=0,whole_campaign_complete=False)

def contents(m,manifest_sha256):
    final=m['schema']==FINAL_SCHEMA
    result=dict(schema='PUBLIC_EXACT_CONTENTS_V1',draft=not final,whole_campaign_complete=False,base_execution_sha=m['package_base_execution_sha'],
        files=[{**{k:e[k]for k in ('destination','kind','bytes','sha256')},'evidence_seal_sha256':(e.get('evidence')or{}).get('seal_sha256')}for e in m['entries']],manifest_binding_sha256=manifest_sha256)
    if final:result.update(scientific_and_derived_complete=True,publication_ready=True,publication_status=PUBLICATION_STATUS)
    return result

def materialize(manifest,output,root_release_GO=False,expected_manifest_sha256=None):
    if root_release_GO is not True:raise PermissionError('Root final package release flag required;preparation must not copy galleries')
    if expected_manifest_sha256 is None or sha(manifest)!=expected_manifest_sha256:raise ValueError('Copy must bind the exact reviewed manifest SHA256')
    checked=verify(manifest);m=read(manifest);out=Path(output).absolute()
    forbidden=[Path('/home/minjae/Documents/github/pallet-pose').resolve(),Path('/tmp/pallet-bi-full-execution-20261019-v1').resolve()]
    if any(out.resolve()==p or p in out.resolve().parents for p in forbidden)or out.exists()or any(p.is_symlink()for p in (out,*out.parents)):raise ValueError('Use a fresh private staging output,never a source checkout')
    if any(out.resolve()==Path(p).resolve()or Path(p).resolve()in out.resolve().parents for p in m['source_roots']):raise ValueError('Staging output cannot be inside an input source directory')
    final=m['schema']==FINAL_SCHEMA
    public=contents(m,checked['manifest_sha256'])
    inventory_bytes=len((json.dumps(public,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    if final and (inventory_bytes>m['maximum_file_bytes']or m['declared_total_bytes']+inventory_bytes>m['maximum_total_bytes']):raise ValueError('Final staging including PUBLIC_CONTENTS exceeds reviewed caps;issue a fresh explicit size policy')
    out.mkdir(parents=True,exist_ok=False)
    for e in m['entries']:
        target=out/e['destination'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(e['source'],target)
        if sha(target)!=e['sha256']:raise ValueError('Copied output changed;preserve partial staging for review')
    write_new(out/'PUBLIC_CONTENTS.json',public);return dict(complete=True,staging=str(out),files=len(m['entries']),commit_or_push_performed=False,
        whole_campaign_complete=False,scientific_and_derived_complete=final,publication_ready=final,
        publication_status=PUBLICATION_STATUS if final else'DRAFT_NOT_READY',selected_file_bytes=m['declared_total_bytes'],PUBLIC_CONTENTS_bytes=inventory_bytes,
        actual_staged_total_bytes=m['declared_total_bytes']+inventory_bytes)

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--repo',required=True);b.add_argument('--run',required=True);b.add_argument('--manifest',required=True)
    f=sub.add_parser('final-build');f.add_argument('--repo',required=True);f.add_argument('--run',required=True);f.add_argument('--spec',required=True);f.add_argument('--manifest',required=True)
    t=sub.add_parser('final-template');t.add_argument('--repo',required=True);t.add_argument('--run',required=True);t.add_argument('--output',required=True)
    v=sub.add_parser('verify');v.add_argument('--manifest',required=True)
    c=sub.add_parser('copy');c.add_argument('--manifest',required=True);c.add_argument('--output',required=True);c.add_argument('--root-release-go',action='store_true');c.add_argument('--expected-manifest-sha256',required=True)
    a=p.parse_args()
    if a.command=='build':result=build(a.repo,a.run,a.manifest)
    elif a.command=='final-build':
        from . import public_final
        result=public_final.build(a.repo,a.run,a.spec,a.manifest)
    elif a.command=='final-template':
        from . import public_final
        result=public_final.template(a.repo,a.run,a.output)
    elif a.command=='verify':result=verify(a.manifest)
    else:result=materialize(a.manifest,a.output,a.root_release_go,a.expected_manifest_sha256)
    print(json.dumps(result,ensure_ascii=False))
