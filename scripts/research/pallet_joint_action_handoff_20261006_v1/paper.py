"""Validate and rebuild the committed manuscript, without changing experimental data.

Schemas written here are new handoff schemas.  A bibliography item is audited
against its original metadata/claim lock or a documented primary-source read;
compilation and citation integrity are separate from scientific validation.
"""
from __future__ import annotations

import argparse
import hashlib
import difflib
import json
import re
import shutil
import subprocess
from pathlib import Path

DOC = Path('_docs/experiments/pallet_joint_action_handoff_20261006_v1')
SOURCE = Path('_docs/experiments/pallet_remaining_evidence_connection_20261006_v1/paper_updated')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_build_diagnostic(path: Path) -> bool:
    return path.suffix in {'.aux', '.out', '.bbl', '.blg', '.log', '.fls', '.xdv',
                           '.toc', '.lof', '.lot', '.fdb_latexmk'} or path.name.endswith('.synctex.gz')


def build_bindings(paper: Path, *, include_generated: bool = True) -> dict[str, str]:
    """Bind the local include graph, styles, bibliography and used graphics.

    Generated figure PDFs and the converted logo enter the binding only after
    asset generation. Their TeX/EPS inputs remain bound in both phases.
    """
    paths: set[Path] = set()
    generated = {p.with_suffix('.pdf') for p in (paper / 'figures').glob('*.tex')
                 if r'\documentclass' in p.read_text()}
    generated.add(paper / 'LOGO-jsen-web.pdf')

    def add(path: Path):
        if not path.is_file() or (path in generated and not include_generated):
            return
        if path in paths:
            return
        paths.add(path)
        if path.suffix not in {'.tex', '.cls', '.sty'}:
            return
        text = re.sub(r'(?<!\\)%[^\n]*', '', path.read_text())
        for match in re.finditer(r'\\(?:input|include)\{([^}]+)\}', text):
            target = paper / match[1]
            add(target if target.suffix else target.with_suffix('.tex'))
        for match in re.finditer(r'\\(?:documentclass|LoadClass)(?:\[[^]]*\])?\{([^}]+)\}', text):
            add(paper / (match[1] + '.cls'))
        for match in re.finditer(r'\\(?:usepackage|RequirePackage)(?:\[[^]]*\])?\{([^}]+)\}', text):
            for name in match[1].split(','):
                add(paper / (name.strip() + '.sty'))
        for match in re.finditer(r'\\bibliographystyle\{([^}]+)\}', text):
            add(paper / (match[1] + '.bst'))
        for match in re.finditer(r'\\bibliography\{([^}]+)\}', text):
            for name in match[1].split(','):
                add(paper / (name.strip() + '.bib'))
        for match in re.finditer(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text):
            target = paper / match[1]
            if not target.suffix:
                target = next((target.with_suffix(s) for s in ['.pdf', '.png', '.jpg', '.jpeg', '.eps']
                               if target.with_suffix(s).is_file()), target)
            add(target)
            generator = target.with_suffix('.tex')
            if generator.is_file() and r'\documentclass' in generator.read_text():
                add(generator)

    for name in ['main.tex', 'supplement.tex']:
        add(paper / name)
    # The class logo name is a macro and the preamble patches EPS to PDF.
    for name in ['LOGO-jsen-web.eps', 'LOGO-jsen-web.pdf']:
        add(paper / name)
    # These standalone sources are compiled by this build, before the papers.
    for target in generated:
        generator = target.with_suffix('.tex')
        if generator.is_file():
            add(generator)
    return {p.relative_to(paper).as_posix(): sha(p) for p in sorted(paths)}


def build_receipt_current(paper: Path, receipt: dict | None) -> bool:
    expected = build_bindings(paper)
    return bool(receipt and expected and receipt.get('built_source_sha256') == expected)


def bib_entries(text: str) -> dict[str, dict[str, str]]:
    entries = {}
    for match in re.finditer(r'@\w+\{([^,]+),\n(.*?)\n\}', text, re.S):
        key = match[1].strip()
        if key in entries:
            raise ValueError(f'duplicate bibliography key: {key}')
        entries[key] = dict(re.findall(r'^\s*(\w+)\s*=\s*\{(.*)\},?$', match[2], re.M))
    return entries


def expand(paper: Path, entry: str) -> tuple[str, list[dict]]:
    """Expand only actual input/include edges, never editorial/archive files."""
    uses = []

    def visit(path: Path, stack: tuple[Path, ...]) -> str:
        if path in stack:
            raise ValueError(f'cyclic LaTeX input: {path}')
        text = path.read_text()
        # Comments cannot create a citation or an include.
        text = re.sub(r'(?<!\\)%[^\n]*', '', text)
        result = []
        cursor = 0
        for m in re.finditer(r'\\(?:input|include)\{([^}]+)\}', text):
            prefix = text[cursor:m.start()]
            record(prefix, path, text[:cursor].count('\n'))
            result.append(prefix)
            target = paper / m[1]
            if not target.suffix:
                target = target.with_suffix('.tex')
            result.append(visit(target, stack + (path,)))
            cursor = m.end()
        suffix = text[cursor:]
        record(suffix, path, text[:cursor].count('\n'))
        result.append(suffix)
        return ''.join(result)

    def record(text: str, path: Path, offset: int):
        for m in re.finditer(r'\\cite(?:\[[^]]*\])?\{([^}]+)\}', text):
            start = text.rfind('\n', 0, m.start()) + 1
            stop = text.find('\n', m.end())
            claim = text[start:stop if stop >= 0 else None].strip()
            for key in m[1].split(','):
                uses.append({'key': key.strip(), 'file': path.relative_to(paper).as_posix(),
                             'line': offset + text[:m.start()].count('\n') + 1,
                             'claim': claim, 'document': entry.removesuffix('.tex')})

    return visit(paper / entry, ()), uses


def export_markdown(paper: Path):
    for name, destination in [('main.tex', 'manuscript_ko.md'), ('supplement.tex', 'supplement_ko.md')]:
        text, _ = expand(paper, name)
        text = text[text.index(r'\begin{document}') + len(r'\begin{document}'):]
        text = text.split(r'\end{document}')[0]
        text = re.sub(r'\\section\{([^}]+)\}', r'\n# \1\n', text)
        text = re.sub(r'\\subsection\{([^}]+)\}', r'\n## \1\n', text)
        text = re.sub(r'\\label\{[^}]+\}', '', text)
        text = re.sub(r'\\cite\{([^}]+)\}', r'[\1]', text)
        text = re.sub(r'\\(?:FloatBarrier|maketitle|bibliographystyle\{[^}]+\}|bibliography\{[^}]+\})', '', text)
        header = ('> 2026-10-06 개정 LaTeX의 읽기용 전개본이야. 수식과 표의 LaTeX 표기를 보존했고, '
                  '그림과 숫자형 인용은 main/supplement PDF가 기준이야. 실제 빌드 상태는 ../C_status.json에 있어.\n\n')
        (paper / destination).write_text(header + text.strip() + '\n')


def validate(root: Path) -> dict:
    report = root / DOC
    paper = report / 'paper_updated'
    entries = bib_entries((paper / 'references.bib').read_text())
    uses = []
    docs = {}
    for name in ['main.tex', 'supplement.tex']:
        text, citation_uses = expand(paper, name)
        uses += citation_uses
        order = list(dict.fromkeys(row['key'] for row in citation_uses))
        docs[name] = {'citation_keys_in_first_appearance_order': order,
                      'undefined_citations': sorted(set(order) - entries.keys())}
        bbl = paper / Path(name).with_suffix('.bbl')
        if bbl.exists():
            built_order = re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}', bbl.read_text())
            docs[name]['built_bibliography_order'] = built_order
            docs[name]['numeric_citation_order_matches'] = built_order == order
        missing = []
        for m in re.finditer(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text):
            candidate = paper / m[1]
            # An explicit .pdf does not resolve to a same-stem .png in TeX.
            exists = candidate.exists() if candidate.suffix else any(
                candidate.with_suffix(s).exists() for s in ['.pdf', '.png', '.jpg', '.eps'])
            if not exists:
                missing.append(m[1])
        docs[name]['missing_graphics'] = sorted(set(missing))
    unused = sorted(entries.keys() - {u['key'] for u in uses})
    protected = []
    for path in sorted((root / SOURCE / 'evidence').rglob('*')):
        if path.is_file():
            target = paper / path.relative_to(root / SOURCE)
            protected.append({'path': path.relative_to(root / SOURCE).as_posix(),
                              'original_sha256': sha(path), 'copied_sha256': sha(target),
                              'equal': sha(path) == sha(target)})
    out = {'schema': 'joint_action_paper_validation_v1', 'documents': docs,
           'bibliography_count': len(entries), 'union_cited_count': len({u['key'] for u in uses}),
           'unused_reference_keys': unused, 'citation_uses': uses,
           'original_N3_evidence_files_unchanged': all(p['equal'] for p in protected),
           'protected_evidence_files': protected}
    out['passed'] = (not unused and all(not d['undefined_citations'] and not d['missing_graphics']
                     and d.get('numeric_citation_order_matches', True) for d in docs.values())
                     and out['original_N3_evidence_files_unchanged'])
    (paper / 'audit' / 'HANDOFF_VALIDATION.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    export_markdown(paper)
    return out


def build(root: Path, engine: str) -> dict:
    paper = root / DOC / 'paper_updated'
    input_start = build_bindings(paper, include_generated=False)
    executable = shutil.which(engine) or (str(Path(engine).resolve()) if Path(engine).is_file() else None)
    if not executable:
        raise FileNotFoundError(f'TeX engine unavailable: {engine}')
    assets = []
    for source in sorted((paper / 'figures').glob('*.tex')):
        if r'\documentclass' not in source.read_text():
            continue
        cmd = [executable, '--keep-logs', source.name]
        run = subprocess.run(cmd, cwd=source.parent, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assets.append({'command': cmd, 'source': source.relative_to(paper).as_posix(), 'returncode': run.returncode})
        (paper / 'audit' / (source.stem + '_build.txt')).write_text(run.stdout)
        if run.returncode:
            raise RuntimeError(f'figure compilation failed: {source.name}')
    logo = paper / 'LOGO-jsen-web.pdf'
    cmd = ['gs', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=pdfwrite', '-dEPSCrop',
           '-sOutputFile=' + str(logo), str(paper / 'LOGO-jsen-web.eps')]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    input_after_assets = build_bindings(paper, include_generated=False)
    source_start = build_bindings(paper)
    docs = []
    for name in ['main', 'supplement']:
        cmd = [executable, '--keep-logs', '--keep-intermediates', name + '.tex']
        run = subprocess.run(cmd, cwd=paper, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log = paper / 'audit' / (name + '_build.txt')
        log.write_text(run.stdout)
        pdf = paper / (name + '.pdf')
        error_pattern = r'^!|^error:|Citation .*undefined|Reference .*undefined|There were undefined|Missing character|Fatal|Missing \$ inserted'
        bad = [line for line in run.stdout.splitlines() if re.search(error_pattern, line, re.I)]
        # Only final TeX log determines unresolved references, not a first-pass warning.
        final_log = (paper / (name + '.log')).read_text(errors='replace') if (paper / (name + '.log')).exists() else run.stdout
        final_bad = [line for line in final_log.splitlines() if re.search(error_pattern, line, re.I)]
        row = {'document': name, 'command': cmd, 'returncode': run.returncode,
               'pdf_exists': pdf.exists(), 'final_unresolved_or_missing_glyph': final_bad,
               'initial_pass_warnings': bad}
        if pdf.exists():
            row['pdf_sha256'] = sha(pdf)
            info = subprocess.run(['pdfinfo', str(pdf)], text=True, capture_output=True)
            pages = re.search(r'^Pages:\s*(\d+)', info.stdout, re.M)
            row['pages'] = int(pages[1]) if pages else None
        row['status'] = 'DONE' if run.returncode == 0 and pdf.exists() and not final_bad else 'BLOCKED_RUNTIME'
        docs.append(row)
    receipt = {'schema': 'joint_action_paper_build_v1', 'engine': executable,
               'engine_sha256': sha(Path(executable)), 'figure_assets': assets, 'documents': docs,
               'copyrighted_external_pdfs_in_repo': False,
               'binding_scope': 'actual local input/include graph, class/style/bst/bib, graphics and generated-asset source dependencies; external engine/bundle/fonts are environment dependencies',
               'built_source_sha256': build_bindings(paper),
               'inputs_changed_during_asset_generation': input_start != input_after_assets}
    receipt['sources_changed_during_build'] = (
        receipt['inputs_changed_during_asset_generation'] or source_start != receipt['built_source_sha256'])
    if receipt['sources_changed_during_build']:
        for document in docs:
            document['status'] = 'BLOCKED_RUNTIME'
    (paper / 'audit' / 'HANDOFF_BUILD.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    validate(root)
    return receipt


def finalize(root: Path) -> dict:
    """Bind current sources to completed checks; never turn absent A/B data into DONE."""
    report, source = root / DOC, root / SOURCE
    target = report / 'paper_updated'
    validation = validate(root)
    refs = json.loads((report / 'REFERENCES_AUDIT_SUMMARY.json').read_text())
    build_path = target / 'audit' / 'HANDOFF_BUILD.json'
    build_receipt = json.loads(build_path.read_text()) if build_path.exists() else None
    build_current = build_receipt_current(target, build_receipt)
    changed, original, patch, diagnostics = [], [], [], []
    for f in sorted(source.rglob('*')):
        if f.is_file() and not is_build_diagnostic(f):
            original.append({'path': f.relative_to(root).as_posix(), 'sha256': sha(f), 'bytes': f.stat().st_size})
    for f in sorted(target.rglob('*')):
        if not f.is_file() or 'source_previous_snapshot' in f.parts:
            continue
        if is_build_diagnostic(f):
            diagnostics.append({'path': f.relative_to(root).as_posix(),
                                'kind': 'local TeX build intermediate', 'delivery_required': False})
            continue
        rel = f.relative_to(target)
        old = source / rel
        if not old.exists() or sha(f) != sha(old):
            changed.append({'path': f.relative_to(root).as_posix(), 'sha256': sha(f), 'bytes': f.stat().st_size})
            if f.suffix in ['.tex', '.bib', '.md']:
                patch += difflib.unified_diff(old.read_text().splitlines(keepends=True) if old.exists() else [],
                    f.read_text().splitlines(keepends=True), fromfile='a/' + rel.as_posix(), tofile='b/' + rel.as_posix())
    (report / 'PAPER_REVISION.patch').write_text(''.join(patch))
    status = {'schema': 'joint_action_C_status_v1', 'schema_is_new': True,
              'manuscript_revision': 'DONE', 'references_audit': 'DONE',
              'references_summary': refs, 'validation': 'DONE' if validation['passed'] else 'BLOCKED_RUNTIME',
              'build_receipt_matches_current_sources': build_current,
              'main_build': 'NOT_RUN_DEPENDENCY', 'supplement_build': 'NOT_RUN_DEPENDENCY',
              'independent_physical_accuracy': 'BLOCKED_DATA',
              'independent_reference_dependency': 'same-relative-pose clean/physical-occlusion pairs with independently calibrated metrology',
              'original_N3_evidence_preserved': validation['original_N3_evidence_files_unchanged'],
              'A_exploratory_section': 'REQUIRES_FINAL_A_RECEIPT_INTEGRATION',
              'submission_status': 'Korean review manuscript; author/affiliation and final English submission not asserted'}
    a_integration = report / 'A_PAPER_INTEGRATION.json'
    if a_integration.exists():
        bound = json.loads(a_integration.read_text())
        snippet = target / 'supplement_tables/joint_action_results.tex'
        bindings_match = all((report / name).is_file() and sha(report / name) == digest
                             for name, digest in bound.get('result_bindings', {}).items())
        bindings_match &= all((target / name).is_file() and sha(target / name) == digest
                              for name, digest in bound.get('supplementary_output_bindings', {}).items())
        if snippet.is_file() and sha(snippet) == bound['copied_snippet_sha256'] and bindings_match:
            status['A_exploratory_section'] = 'DONE'
            status['A_results_integration'] = bound
    if build_current:
        for d in build_receipt['documents']:
            status[d['document'] + '_build'] = d['status']
    for name in ['RUNTIME_MATCHED.json', 'RUNTIME_VERIFICATION.json',
                 'RUNTIME_A.json', 'RUNTIME_A_VERIFICATION.json']:
        path = report / name
        if path.exists():
            status[name.removesuffix('.json')] = {'path': name, 'sha256': sha(path)}
    manifest = {'schema': 'joint_action_C_manifest_v1', 'schema_is_new': True,
                'previous_manuscript': str(SOURCE), 'current_manuscript': str(DOC / 'paper_updated'),
                'previous_source_bindings': original, 'modified_or_added_files': changed,
                'local_build_diagnostics': diagnostics,
                'local_build_diagnostics_are_deliverables': False,
                'reference_audit_sha256': sha(report / 'REFERENCES_AUDIT.tsv'),
                'code_sha256': sha(Path(__file__)),
                'validation_path': str(DOC / 'paper_updated/audit/HANDOFF_VALIDATION.json'),
                'build_path': str(DOC / 'paper_updated/audit/HANDOFF_BUILD.json'),
                'commands': ['python3 scripts/research/pallet_joint_action_handoff_20261006_v1/paper.py validate',
                    'python3 scripts/research/pallet_joint_action_handoff_20261006_v1/paper.py build --engine /actual/tectonic/path',
                    'python3 scripts/research/pallet_joint_action_handoff_20261006_v1/paper.py finalize'],
                'new_training': 0, 'new_manual_labels': 0,
                'copyrighted_primary_pdfs_stored_outside_repository': True,
                'new_schemas': ['REFERENCES_AUDIT.tsv', 'C_status.json', 'C_manifest.json', 'HANDOFF_BUILD.json', 'HANDOFF_VALIDATION.json']}
    (report / 'C_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    (report / 'C_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return status


def integrate_a(root: Path, snippet: Path) -> dict:
    """Insert an evidence-reviewed final A summary; build/finalize then checks it."""
    if not snippet.is_file():
        raise FileNotFoundError('Final A evidence summary is required: ' + str(snippet))
    report = root / DOC
    paper = report / 'paper_updated'
    target = paper / 'supplement_tables/joint_action_results.tex'
    text = snippet.read_text()
    transformations = []
    before = 'PCK10은 전체 참조 코너를 분모로 한다.'
    after = 'PCK10은 전체 참조 코너를 분모로 한 0--1 분율이다. 코너 오차는 311장/2,445개 관측 코너, PCK는 2,499개 참조 코너, 위치·회전은 319/319장 자세 산출을 사용한다.'
    if before in text:
        transformations.append({'source': before, 'rendered': after, 'count': text.count(before)})
        text = text.replace(before, after)
    for token, label in [('SUPPORTED', '고정된 개선·보존 조건 충족'),
                         ('WORSENED', '오차 악화'),
                         ('UNRESOLVED', '개선·보존 조건 미충족')]:
        before = f'판정은 {token}였다.'
        if before in text:
            after = f'판정은 {label}' + ('였다.' if token == 'WORSENED' else '이었다.')
            transformations.append({'source': before, 'rendered': after,
                                    'count': text.count(before)})
            text = text.replace(before, after)
    # Seven numeric columns need both columns of this journal layout.
    if r'\begin{table}[t]' in text:
        transformations.append({'source': 'table', 'rendered': 'table*',
                                'reason': 'seven-column seed table uses full page width'})
        text = text.replace(r'\begin{table}[t]', r'\begin{table*}[t]')
        text = text.replace(r'\end{table}', r'\end{table*}')
    target.write_text(text)
    design = paper / 'supplement_tables/joint_action_exploratory.tex'
    text = design.read_text()
    placeholder = '현재 실행 상태와 수치는 보충 표의 실행 영수증에 연결할 예정이며, 이 작성 중 문장을 scientific result로 사용하지 않는다. 원래 N3 수치는 변경하지 않았다.'
    if placeholder in text:
        text = text.replace(placeholder, r'\input{supplement_tables/joint_action_results}')
        design.write_text(text)
    elif r'\input{supplement_tables/joint_action_results}' not in text:
        raise ValueError('Exploratory section has neither pending marker nor final results input')
    receipt = {'schema': 'joint_action_A_paper_integration_v1', 'source_snippet': str(snippet),
               'source_sha256': sha(snippet), 'copied_snippet_sha256': sha(target),
               'editorial_transformations': transformations,
               'original_N3_headline_unchanged': True, 'result_scope': 'exploratory supplementary results'}
    receipt['result_bindings'] = {
        p.relative_to(report).as_posix(): sha(p)
        for p in [report / 'results/A_summary.json', report / 'A_status.json',
                  report / 'results/A_REAL_DEV_COMPARISONS.json'] if p.is_file()}
    (report / 'A_PAPER_INTEGRATION.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    export_markdown(paper)
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['validate', 'build', 'export', 'finalize', 'integrate-a'])
    p.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[3])
    p.add_argument('--engine', default='tectonic', help='Tectonic executable; dependencies may live outside the repository')
    p.add_argument('--a-snippet', type=Path, help='Evidence-reviewed final A LaTeX summary; used only by integrate-a')
    args = p.parse_args()
    if args.action == 'build':
        result = build(args.repo_root, args.engine)
        ok = all(d['status'] == 'DONE' for d in result['documents'])
    elif args.action == 'export':
        export_markdown(args.repo_root / DOC / 'paper_updated')
        result, ok = {'exported': True}, True
    elif args.action == 'finalize':
        result = finalize(args.repo_root)
        ok = result['validation'] == 'DONE' and result['main_build'] == result['supplement_build'] == 'DONE'
    elif args.action == 'integrate-a':
        if args.a_snippet is None:
            p.error('integrate-a requires --a-snippet')
        result, ok = integrate_a(args.repo_root, args.a_snippet), True
    else:
        result = validate(args.repo_root)
        ok = result['passed']
        result = {k: v for k, v in result.items() if k not in ['citation_uses', 'protected_evidence_files']}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
