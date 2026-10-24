#!/usr/bin/env python3
"""Read report metadata only; never change original reports or follow symlinks."""
import argparse
import collections
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from urllib.parse import unquote, urlsplit

PROJECT = Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility')
VERSION = re.compile(r'(?:^|_)v(\d+)(?:_|$)', re.I)
EXCLUDED = re.compile(r'(?:^|_)v112(?:_|$)', re.I)
EXACT = {'ROOT_PHASE_RESULT.json', 'ROOT_RESULTS_KO.json', 'ROOT_READOUT_KO.html'}
ROOT_DOCS = {'README.md', 'START_HERE_KO.md', 'HANDOFF_KO.md',
             'PATH_INDEX.csv', 'ACTIVE_HANDOFF_KO.json', 'HANDOFF_VERIFICATION.json'}
MAX_REPORT_BYTES = 2 * 1024 * 1024
ARCHIVE_PREFIX = 'archive/task-2/pallet-keypoint-visibility'


def excluded_parts(path):
    return any(EXCLUDED.search(s) for s in Path(path).parts)


def inspect_path(root, target):
    """Name-prune first, then lstat each component; no resolve/readlink/open."""
    target = Path(os.path.abspath(target))
    try:
        rel = target.relative_to(root)
    except ValueError:
        return {'status': 'OUTSIDE_PROJECT', 'portable_project_target': None}
    if excluded_parts(rel):
        return {'status': 'EXCLUDED_V112', 'portable_project_target': None}
    cur = root
    info = None
    for part in rel.parts:
        cur = cur / part
        try:
            info = os.lstat(cur)
        except FileNotFoundError:
            return {'status': 'MISSING', 'portable_project_target': rel.as_posix()}
        if stat.S_ISLNK(info.st_mode):
            return {'status': 'SYMLINK', 'portable_project_target': None}
    return {'status': 'EXISTS', 'portable_project_target': rel.as_posix(),
            'target_size_bytes': info.st_size if info else 0,
            'target_is_directory': bool(info and stat.S_ISDIR(info.st_mode))}


def classify_reference(root, doc, raw):
    raw = str(raw).strip()
    if not raw:
        return {'status': 'EMPTY', 'portable_project_target': None}
    if raw.startswith('#'):
        return {'status': 'ANCHOR', 'portable_project_target': None}
    if re.match(r'^[A-Za-z]:[\\/]', raw):
        return {'status': 'OUTSIDE_PROJECT', 'portable_project_target': None}
    parsed = urlsplit(raw)
    if parsed.scheme and parsed.scheme != 'file':
        return {'status': 'EXTERNAL_URL', 'portable_project_target': None}
    if parsed.netloc and parsed.scheme != 'file':
        return {'status': 'EXTERNAL_URL', 'portable_project_target': None}
    path = unquote(parsed.path)
    if parsed.scheme == 'file' and parsed.netloc not in ('', 'localhost'):
        return {'status': 'OUTSIDE_PROJECT', 'portable_project_target': None}
    if not path:
        return {'status': 'ANCHOR', 'portable_project_target': None}
    target = Path(path) if os.path.isabs(path) else doc.parent / path
    result = inspect_path(root, target)
    result['fragment'] = parsed.fragment or None
    result['query'] = parsed.query or None
    result['original_absolute_reference'] = os.path.isabs(path) or parsed.scheme == 'file'
    return result


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ('href', 'src') and value is not None:
                self.links.append(('html_' + key, value))


def json_strings(value, trail='$'):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from json_strings(v, trail + '.' + str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from json_strings(v, trail + '[' + str(i) + ']')
    elif isinstance(value, str):
        yield trail, value


def references(doc, text, value):
    if doc.suffix.lower() == '.html':
        parser = Links()
        parser.feed(text)
        yield from parser.links
    elif doc.suffix.lower() == '.md':
        for m in re.finditer(r'!?\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+[^)]*)?\)', text):
            yield 'markdown', m.group(1).strip('<>')
    if value is not None:
        for trail, s in json_strings(value):
            if s.startswith(('/home/', 'file:', 'http://', 'https://')) or re.match(r'^[A-Za-z]:[\\/]', s):
                yield 'json_path:' + trail, s


def small_value(v):
    s = json.dumps(v, ensure_ascii=False)
    return v if len(s) <= 1600 else {'excerpt': s[:1600], 'truncated': True}


def selected_fields(value):
    if not isinstance(value, dict):
        return {}
    groups = {'goal_method': [], 'inputs_training_denominators': [],
              'conclusion_failures_limits': [], 'explicit_status': {}}
    for k in ('completed', 'PASS', 'phase_complete', 'source_experiment_complete',
              'source_phase_complete', 'selected_model', 'adopted', 'fully_solved',
              'whole_goal_solved', 'goal_status'):
        if k in value:
            groups['explicit_status'][k] = value[k]
    for key, v in value.items():
        if key in ('bindings', 'refs', 'reviews') or key in groups['explicit_status']:
            continue
        k = key.lower()
        if any(s in k for s in ('conclusion', 'outcome', 'decision', 'limit', 'failure', 'remaining', 'caveat', 'history', 'interpretation')):
            cat = 'conclusion_failures_limits'
        elif any(s in k for s in ('training', 'updates', 'inputu', 'cohort', 'denominator', 'coverage', 'rows', 'source_assets', 'primary', 'execution', 'population', 'independent_ray')):
            cat = 'inputs_training_denominators'
        elif any(s in k for s in ('goal', 'objective', 'method', 'schema', 'phase', 'decoder', 'definition', 'truth_and_metrics', 'publication_kind')):
            cat = 'goal_method'
        else:
            continue
        if len(groups[cat]) < 12:
            groups[cat].append({'json_path': '$.' + key, 'value': small_value(v)})
    return groups


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def md_cell(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    return value.replace('|', '\\|').replace('\n', '<br>')


def make_portable_pages(root, output, overview, values, crosswalk):
    """Derived Markdown only; relative links assume copying into version archive."""
    base = output/'portable_pages'
    base.mkdir()
    index = ['# 전체 실험 기록', '',
             '단계별 방법·결과·실패·검증·그림을 원문과 함께 보존한 목록입니다. 단계 PASS는 전체 목표 해결을 뜻하지 않습니다.', '',
             '최신 완료 실험은 V186의 고정4 source 프레임/5 GT instance 등록 진단과 독립400 ray 검산입니다. 모델 성공 결과가 아닙니다. 최신 실사 분류 비교는 V184이며 D recall 증가와 E recall 감소를 함께 보고했고 모델 채택이 없습니다. V185는 source 감독 준비도 검토입니다. 전체 WOOD/SQUARE·native8 목표는 미해결입니다.', '',
             'legacy 모델의 실사-CAL 이력, 새 모델의 평가-only 실사 사용, 역사적으로 반복 노출된 VAL/current/public held의 not-blind 한계를 각 원문대로 유지합니다. source point-owner와 regional end-support는 native 외부가림 정답과 동일하지 않습니다.', '',
             '이 게시본에는 허용된 버전 원문과 별도 Markdown 파생본이 공존합니다. 원문 로컬 절대 경로는 provenance이고 이 페이지 링크는 프로젝트 상대 경로입니다. 대용량/외부 의존성은 최종 publication manifest에 별도 기록합니다.', '',
             'V112는 명시 지시에 따라 디렉터리 이름 단계에서 제외했습니다. 그 내용을 읽거나 완료/실패로 평가하지 않았습니다.', '',
             '| 버전 | 개요 | 원본 디렉터리 |', '|---|---|---|']
    table_headers = {
        (184, 'table_primary'): ['모델','D precision','D recall','E precision','E recall','U rate','CM: truth D/E × pred D/E/U'],
        (184, 'table_material'): ['재질 그룹','모델','D/E 분모','D recall','E precision','E recall','U rate','E→D'],
        (184, 'table_manual_near376'): ['모델','N','D 응답','D 응답률','E 응답','U 응답'],
        (184, 'table_public_weak_center'): ['모델','weak truth T/nonT × pred T/nonT/U CM','T precision','T recall','전체 prediction U','known weak N','unknown weak N'],
    }
    for row in overview:
        dirname = row['directory']; ver = row['version']
        dest = base/dirname; dest.mkdir()
        target = Path(dirname)
        text = [f'# V{ver} — {dirname}', '',
                '기존 보고서에서 추출한 게시용 파생 개요입니다. 원본 파일을 수정하거나 실험·평가를 다시 실행하지 않았습니다. 원문 해시와 링크 대응은 publication manifest로 확인합니다.', '',
                '**상태:** ' + md_cell(row['phase_status'] or row['overview'].get('explicit_status', {}) or '명시된 완료 receipt 없음 — 완료 단정하지 않음'), '',
                '원본:', '']
        for doc in row['documents']:
            rel = os.path.relpath(doc['source_relative_path'], target.as_posix())
            text.append(f"- [{Path(rel).name}]({rel}) — SHA256 `{doc['sha256']}`")
        if not row['documents']:
            text += ['- 이 snapshot의 허용된 최상위 보고서 파일명에 일치하는 문서가 없습니다. 하위 기록·코드·실패 여부는 [원본 폴더](./)와 전체 파일 inventory를 보십시오. 이 상태를 실험 완료나 실패로 치환하지 않습니다.']
        labels = [('goal_method','방법·목표의 원문 필드'), ('inputs_training_denominators','입력·학습·평가 분모의 원문 필드'), ('conclusion_failures_limits','결론·실패·한계의 원문 필드')]
        for key, label in labels:
            fields = row['overview'].get(key, [])
            text += ['', '## '+label, '']
            if not fields:
                text += ['자동 추출 가능한 해당 최상위 필드가 없습니다. 원문 보고서와 phase receipt를 확인하십시오.']
            for field in fields:
                val = field['value']
                pretty = json.dumps(val, ensure_ascii=False, indent=2).replace(str(root)+'/', '') if not isinstance(val,str) else val.replace(str(root)+'/', '')
                text += [f"`{field['json_path']}`", '', '```text', pretty, '```', '']
        value = values.get(row['overview_source'], {})
        if isinstance(value, dict):
            if ver == 186 and isinstance(value.get('instance_table'),list):
                text += ['', '## 고정 source 등록 진단 5 instances', '',
                         '원문 수치를 그대로 표시합니다. Full-mask IoU는 target-alone 렌더와 제공 full mask의 정합이며, 부위 가림·모델 전이 성능이 아닙니다. 잔차는 원 정의의 model Z−sensor Z이고 결측은 별도입니다.', '',
                         '| frame | GT list index | full-mask IoU | visible mask pixels | visible sensor missing | measured intersection N | residual median mm |',
                         '|---|---|---|---|---|---|---|']
                for r in value['instance_table']:
                    keys=['frame_index','GT_instance_index','full_mask_IoU','provided_visible_mask_pixels','visible_mask_sensor_missing_pixels','measured_visible_render_intersection_N','signed_modelZ_minus_sensorZ_mm_median']
                    text.append('| '+' | '.join(md_cell(r.get(k)) for k in keys)+' |')
            for key, table in value.items():
                if not (key.startswith('table_') or key.endswith('_table')) or not isinstance(table,list) or not table or len(table)>200:
                    continue
                if not all(isinstance(r,list) and 1<=len(r)<=20 for r in table):
                    continue
                width = len(table[0])
                if any(len(r)!=width for r in table):
                    continue
                headers = value.get(key+'_columns')
                if not isinstance(headers,list) or len(headers)!=width:
                    headers = table_headers.get((ver,key))
                if headers is None:
                    headers = ['원문 열'+str(i+1) for i in range(width)]
                text += ['', '## 원문 표: '+key, '',
                         '행·값은 ROOT_RESULTS 원문 그대로입니다. 열 이름이 원문 JSON에 없으면 원문 열번호로 표시하며 정의는 원 보고서를 함께 확인합니다.', '',
                         '| '+' | '.join(map(md_cell,headers))+' |', '| '+' | '.join(['---']*width)+' |']
                text += ['| '+' | '.join(map(md_cell,r))+' |' for r in table]
        local = [r for r in crosswalk if r['document_relative_path'].startswith(dirname+'/') and r['status']=='EXISTS' and not r.get('target_is_directory')]
        links=[]
        seen=set()
        for r in local:
            path=r['portable_project_target']
            if path not in seen:
                seen.add(path); links.append(path)
        images=[x for x in links if Path(x).suffix.lower() in ('.png','.jpg','.jpeg','.svg')]
        documents=[x for x in links if Path(x).suffix.lower() in ('.py','.csv','.json','.jsonl','.md','.html')]
        text += ['', '## 코드·검증·전체 표·갤러리', '', '아래 목록은 원 보고서가 참조한 파일의 경로를 다시 연결한 것입니다. 이 개요 작성 중 이미지 내용이나 raw 평가 데이터를 새로 검사하지 않았습니다.', '']
        for path in documents[:24]:
            rel=os.path.relpath(path,target.as_posix());text.append(f'- [{path}]({rel})')
        if images:
            text += ['', '## 원 보고서의 그림', '', '원 보고서 참조 순서의 처음 최대4개를 표시합니다. 새 outcome 선택이나 전수 이미지 검증을 뜻하지 않습니다. 전체 갤러리는 위 원문 링크에서 확인합니다.', '']
            for path in images[:4]:
                rel=os.path.relpath(path,target.as_posix());text += [f'[{path}]({rel})', '', f'![원 보고서 진단 그림]({rel})', '']
        text += ['', '현재 사용자 목표 전체의 해결·새 모델 채택·실사 가림 원인 확정을 이 파생 개요가 추가로 주장하지 않습니다. UNKNOWN/U와 불리한 결과, 실패 실행 및 수정 provenance는 원문대로 보존해야 합니다.', '']
        (dest/'PORTABLE_REPORT_KO.md').write_text('\n'.join(text))
        index.append(f'| V{ver} | [방법·결과·한계]({ARCHIVE_PREFIX}/{dirname}/PORTABLE_REPORT_KO.md) | [원본]({ARCHIVE_PREFIX}/{dirname}/) |')
    (output/'EXPERIMENT_INDEX_KO.md').write_text('\n'.join(index)+'\n')


def collect(root, output):
    root = Path(os.path.abspath(root))
    if excluded_parts(output) or root not in output.parents:
        raise ValueError('Preparation output must be a nonexcluded project descendant')
    output.mkdir(parents=True, exist_ok=False)
    docs = []
    dirs = []
    pruned = []
    for e in os.scandir(root):
        # This check precedes is_symlink/is_dir/stat/read/hash for the entry.
        if EXCLUDED.search(e.name):
            pruned.append({'name': e.name, 'reason': 'V112 name-pruned before stat/read/hash/search'})
            continue
        if e.is_symlink():
            pruned.append({'name': e.name, 'reason': 'symlink not followed'})
            continue
        if e.is_dir(follow_symlinks=False):
            m = VERSION.search(e.name)
            if not m:
                continue
            directory = Path(e.path)
            dirs.append({'version': int(m.group(1)), 'directory': e.name})
            for c in os.scandir(directory):
                if EXCLUDED.search(c.name) or c.is_symlink():
                    continue
                name = c.name
                if name in EXACT or (name.lower().endswith('.md') and name.upper().startswith(('README', 'START_HERE', 'HANDOFF', 'PATH_INDEX'))):
                    docs.append(Path(c.path))
        elif e.name in ROOT_DOCS:
            docs.append(Path(e.path))
    reports, crosswalk, skipped, values = [], [], [], {}
    bydir = collections.defaultdict(list)
    for doc in sorted(set(docs)):
        rel = doc.relative_to(root).as_posix()
        inspected = inspect_path(root, doc)
        if inspected['status'] != 'EXISTS' or inspected['target_is_directory']:
            skipped.append({'path': rel, **inspected})
            continue
        if inspected['target_size_bytes'] > MAX_REPORT_BYTES:
            skipped.append({'path': rel, 'reason': 'report text over2MiB; preserved but not opened'})
            continue
        raw = doc.read_bytes()
        text = raw.decode('utf-8-sig')
        val = json.loads(text) if doc.suffix.lower() == '.json' else None
        if val is not None:
            values[rel] = val
        item = {'source_relative_path': rel, 'sha256': hashlib.sha256(raw).hexdigest(),
                'bytes': len(raw), 'extracted': selected_fields(val)}
        reports.append(item)
        bydir[doc.parent.name].append(item)
        for kind, original in references(doc, text, val):
            mapped = classify_reference(root, doc, original)
            project_target = mapped.get('portable_project_target')
            crosswalk.append({'document_relative_path': rel, 'reference_kind': kind,
                              'original_reference': original, **mapped,
                              'portable_published_target': ARCHIVE_PREFIX+'/'+project_target if project_target else None})
    overview = []
    for d in sorted(dirs, key=lambda d: (d['version'], d['directory'])):
        rs = bydir.get(d['directory'], [])
        best = next((r for r in rs if r['source_relative_path'].endswith('/ROOT_RESULTS_KO.json')), None)
        phase = next((r for r in rs if r['source_relative_path'].endswith('/ROOT_PHASE_RESULT.json')), None)
        overview.append({**d, 'documents': [{k: r[k] for k in ('source_relative_path', 'sha256', 'bytes')} for r in rs],
                         'overview_source': best['source_relative_path'] if best else None,
                         'overview': best['extracted'] if best else {},
                         'phase_status': phase['extracted']['explicit_status'] if phase else {},
                         'status_note': 'Original explicit flags only; phase PASS is not full-goal success.' if rs else 'No matching top-level report in this snapshot; not declared complete.'})
    write_json(output/'REPORT_SOURCE_MANIFEST.json', {'documents': reports, 'skipped': skipped})
    write_json(output/'EXPERIMENT_OVERVIEW.json', {'experiments': overview, 'excluded_names_only': pruned,
                                                'snapshot_limit': 'Top-level known report/README family only; no recursive raw/model/payload reads.'})
    with (output/'PORTABLE_TARGET_CROSSWALK.jsonl').open('w') as f:
        for r in crosswalk:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    counts = collections.Counter(r['status'] for r in crosswalk)
    summary = {'completed': True, 'metadata_preparation_only': True,
               'experiment_directories': len(overview), 'documents_read': len(reports),
               'directories_with_matching_reports': sum(bool(r['documents']) for r in overview),
               'crosswalk_reference_occurrences': len(crosswalk), 'link_status_counts': dict(counts),
               'excluded_names_only': pruned, 'original_documents_modified': 0,
               'referenced_artifact_body_or_hash_reads': 0, 'symlink_followed': 0,
               'V112_stat_read_hash_search': 0, 'new_model_or_raw_data_reads': 0,
               'external_urls_network_checked': False, 'Git_tracked_or_remote_accessibility_checked': False,
               'no_claim_all_experiments_completed': True,
               'portability_limit': 'Existing local target is not proof it will be staged/pushed; join C inventory and root final Git manifest.'}
    write_json(output/'LINK_AUDIT_SUMMARY.json', summary)
    make_portable_pages(root, output, overview, values, crosswalk)
    lines = ['# 실험 기록 공개 인덱스 — 준비용 미리보기', '',
             '원본 보고서·코드·표·그림으로 연결하는 목록입니다. 단계 완료와 전체 문제 해결을 구분합니다. 원본 문서는 수정하지 않았습니다.', '',
             'V112는 지시에 따라 디렉터리 이름 단계에서 제외했으며 내용을 읽지 않았습니다. 심볼릭 링크도 따라가지 않았습니다.', '',
             '현재 핵심 비교는 V184의 평가 전용 실사 비교입니다. V185는 source 감독 준비도, V186은 고정 source GT 등록 진단입니다. 서로 다른 성과이며 전체 목표는 미해결입니다.', '',
             'legacy 실사-CAL 이력, 현재/공개 held의 반복 노출로 인한 not-blind 한계, source TRAIN apparent fit과 자연 실사 전이를 각 원문대로 유지합니다.', '',
             '65536 augmented crops는 65536 독립 장면이 아닙니다. 기존 실제 이미지의 q/L/가림 주석과 source exact-point owner·regional end-support 정의를 합치지 않습니다.', '',
             '이 목록은 알려진 최상위 보고서만 추출했습니다. 하위 실패 trace·provenance·실행 로그·검산·전체 표·이미지는 C 파일 inventory 및 원 실험 디렉터리와 함께 보존해야 합니다.', '',
             '루트 README/START_HERE/HANDOFF에는 v60–v62 당시 인계 상태가 남아 있습니다. 그 역사적 원문을 최신 진행 상태로 해석하지 마십시오. 최신 active handoff와 각 phase receipt를 함께 읽어야 합니다.', '',
             '| 버전 | 원 실험 디렉터리 | 보고서 | 명시된 phase 상태 |', '|---|---|---|---|']
    for r in overview:
        d = r['directory']
        links = []
        for doc in r['documents']:
            name = Path(doc['source_relative_path']).name
            links.append('['+name+']('+os.path.relpath(root/doc['source_relative_path'], output)+')')
        explicit = r['phase_status'] or r['overview'].get('explicit_status', {})
        state = json.dumps(explicit, ensure_ascii=False) if explicit else '未확인 / 완료 단정 없음'
        lines.append(f"| V{r['version']} | [{d}]({os.path.relpath(root/d,output)}/) | {' · '.join(links) or '일치하는 최상위 보고서 없음'} | {state} |")
    lines += ['', '절대 로컬 경로는 원문 provenance에 남겨 두고 PORTABLE_TARGET_CROSSWALK.jsonl에 별도 project-relative 대응을 기록했습니다. 원 HTML을 실제 portable 배포본으로 제공하려면 href/src만 바꾼 파생본과 원문 해시를 함께 보존하고, 최종 Git에 포함된 대상인지 다시 검사해야 합니다. 외부 자산·심볼릭 링크·누락 경로·V112는 자동 복사하지 않습니다.', '']
    (output/'PORTABLE_INDEX_PREVIEW.md').write_text('\n'.join(lines))
    return summary


def audit_document_list(root, list_path, output_path):
    """Final-stage audit of an explicit list of portable text files, not a tree crawl."""
    root = Path(os.path.abspath(root))
    paths = json.loads(list_path.read_text())
    if not isinstance(paths, list) or not all(isinstance(x,str) for x in paths):
        raise ValueError('Audit list must be a JSON array of relative document paths')
    rows = []
    for rel in paths:
        doc = root/rel
        status = inspect_path(root, doc)
        if status['status'] != 'EXISTS' or status.get('target_is_directory'):
            rows.append({'document': rel, 'document_status': status['status']})
            continue
        if doc.suffix.lower() not in ('.md','.html','.json') or status['target_size_bytes']>MAX_REPORT_BYTES:
            rows.append({'document': rel, 'document_status': 'NOT_ALLOWED_TEXT_DOCUMENT'})
            continue
        text = doc.read_text(encoding='utf-8-sig')
        value = json.loads(text) if doc.suffix.lower()=='.json' else None
        for kind, raw in references(doc,text,value):
            rows.append({'document':rel,'reference_kind':kind,'original_reference':raw,
                         **classify_reference(root,doc,raw)})
    counts=collections.Counter(r.get('status',r.get('document_status')) for r in rows)
    result={'completed':True,'documents_requested':len(paths),'reference_occurrences':len(rows),
            'status_counts':dict(counts),'PASS':not any(counts[x] for x in ('MISSING','SYMLINK','EXCLUDED_V112','OUTSIDE_PROJECT','NOT_ALLOWED_TEXT_DOCUMENT')),
            'limits':'Existence only, not Git-tracked/remote/browser/network/hash equivalence; external/provenance references remain separate.',
            'V112_stat_read_hash_search':0,'symlink_followed':0,'artifact_payload_body_reads':0,'rows':rows}
    with output_path.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    return {k:v for k,v in result.items() if k!='rows'}


def self_test():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root/'report').mkdir(); (root/'images').mkdir()
        (root/'images'/'한 글.png').write_bytes(b'invented')
        doc = root/'report'/'readout.html'
        got = classify_reference(root, doc, '../images/%ED%95%9C%20%EA%B8%80.png?raw=1#x')
        assert got['status'] == 'EXISTS' and got['portable_project_target'] == 'images/한 글.png'
        assert got['query'] == 'raw=1' and got['fragment'] == 'x'
        assert classify_reference(root, doc, str(root/'images'/'한 글.png'))['status'] == 'EXISTS'
        assert classify_reference(root, doc, '#table')['status'] == 'ANCHOR'
        assert classify_reference(root, doc, 'https://example.org/x')['status'] == 'EXTERNAL_URL'
        assert classify_reference(root, doc, '../../../outside.bin')['status'] == 'OUTSIDE_PROJECT'
        assert classify_reference(root, doc, 'C:\\original\\weights.pt')['status'] == 'OUTSIDE_PROJECT'
        assert classify_reference(root, doc, '../query_visibility_any_v112/never-open')['status'] == 'EXCLUDED_V112'
        (root/'link').symlink_to(root/'images', target_is_directory=True)
        assert classify_reference(root, doc, '../link/한 글.png')['status'] == 'SYMLINK'
        assert classify_reference(root, doc, '../missing.png')['status'] == 'MISSING'
        parser = Links(); parser.feed('<a href="../x?a=1&amp;b=2">a</a><img src="/tmp/z">')
        assert parser.links == [('html_href','../x?a=1&b=2'),('html_src','/tmp/z')]
        old = os.lstat
        def forbidden(_):
            raise AssertionError('Excluded target must not reach lstat')
        os.lstat = forbidden
        try:
            assert inspect_path(root, root/'query_visibility_demo_v112'/'x')['status'] == 'EXCLUDED_V112'
        finally:
            os.lstat = old
        doc.write_text('<img src="../images/%ED%95%9C%20%EA%B8%80.png"><a href="#a">x</a>')
        manifest=root/'documents.json';manifest.write_text('["report/readout.html"]')
        audit=audit_document_list(root,manifest,root/'good.json')
        assert audit['PASS'] and audit['status_counts']=={'EXISTS':1,'ANCHOR':1}
        doc.write_text('<img src="../missing.png">')
        audit=audit_document_list(root,manifest,root/'bad.json')
        assert not audit['PASS'] and audit['status_counts']=={'MISSING':1}
    return {'PASS': True, 'invented_groups': 12, 'actual_payload_reads': 0,
            'checks': ['percent/unicode/query/fragment','local absolute mapping','anchors','external URL','escape rejection','Windows path','V112 pre-stat prune','symlink rejection','missing link','HTML entity parsing','instrumented forbidden-lstat exclusion','actual explicit-document audit PASS and missing-link FAIL']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=PROJECT)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--audit-list',type=Path,help='JSON array of relative portable doc paths')
    parser.add_argument('--audit-output',type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False))
    elif args.audit_list:
        if args.audit_output is None:
            parser.error('--audit-output is required with --audit-list')
        print(json.dumps(audit_document_list(args.project_root,args.audit_list,args.audit_output),ensure_ascii=False))
    else:
        if args.out is None:
            parser.error('--out is required (a fresh preparation directory)')
        print(json.dumps(collect(args.project_root, args.out), ensure_ascii=False))
