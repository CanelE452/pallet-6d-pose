"""Offline CPU-only draft compilation in this revision; original sources are never written."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD_RAW = ROOT / 'data/pallet/results/pallet_sensors_submission_v1'


def binding(path):
    return dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    compiler = OLD_RAW / 'tectonic'
    assert compiler.is_file()
    cache = HERE / '.tex_cache'
    if not cache.exists():
        # Copy once, never write the old draft's cache during this build.
        shutil.copytree(OLD_RAW/'tex_cache', cache)
    insertions = {}
    for filename in ('DOPE_RESULTS_BINDINGS.json', 'RESNET_RESULTS_BINDINGS.json'):
        value = json.loads((HERE/filename).read_text())
        for artifact in value['generated_tables'] + value.get('generated_assets', []):
            assert binding(ROOT/artifact['path']) == artifact, ('Stale generated manuscript artifact', artifact['path'])
        insertions[filename] = value
    assert 'ThirdEstimatorStatus' not in (HERE/'generated_tables/dope_status.tex').read_text()
    assert 'ThirdEstimatorStatus' in (HERE/'generated_tables/resnet_status.tex').read_text()
    inputs = [p for p in sorted(HERE.rglob('*')) if p.is_file() and
              '.tex_cache' not in p.parts and p.suffix in ('.tex', '.bib')]
    before = [binding(p) for p in inputs]
    results = []
    for name in ('manuscript', 'supplementary'):
        command = [str(compiler), '--only-cached', '--keep-logs', '--keep-intermediates', str(HERE/(name+'.tex'))]
        p = subprocess.run(command, cwd=HERE, env={**os.environ, 'XDG_CACHE_HOME':str(cache)},
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (HERE/(name.upper()+'_BUILD.txt')).write_text(p.stdout)
        print(p.stdout[-4000:])
        assert p.returncode == 0, (name, p.returncode)
        pdf = HERE/(name+'.pdf')
        info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
        pages = int(next(line.split(':',1)[1] for line in info.splitlines() if line.startswith('Pages:')))
        text = subprocess.check_output(['pdftotext','-layout',str(pdf),'-'], text=True)
        (HERE/(name.upper()+'_EXTRACTED.txt')).write_text(text)
        assert '??' not in text and 'Lorem ipsum' not in text
        rendered = []
        if args.render:
            target = HERE/'pdf_render'; target.mkdir(exist_ok=True)
            # This revision owns only these render files; remove stale trailing pages.
            for stale in target.glob(name+'-*.png'): stale.unlink()
            subprocess.run(['pdftoppm','-r','90','-png',str(pdf),str(target/name)], check=True)
            rendered = [binding(p) for p in sorted(target.glob(name+'-*.png'))]
        results.append(dict(document=name,pages=pages,pdf=binding(pdf),build_log=binding(HERE/(name.upper()+'_BUILD.txt')),
                            text=binding(HERE/(name.upper()+'_EXTRACTED.txt')),rendered_pages=rendered))
    assert before == [binding(p) for p in inputs], 'Compilation changed a source file'
    receipt = dict(schema='sensors_dope_draft_build_v1', complete=True, scope='TYPESETTING_ONLY',
        draft=True, not_submitted=True, results_insert_status=insertions['DOPE_RESULTS_BINDINGS.json']['status'],
        resnet_results_insert_status=insertions['RESNET_RESULTS_BINDINGS.json']['status'],
        insertion_receipts={name:binding(HERE/name) for name in insertions},
        author_approval=False, scientific_success_inferred=False, visual_review='PENDING',
        compiler=binding(compiler), code=binding(Path(__file__)), inputs=before, documents=results)
    (HERE/'DRAFT_BUILD.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(complete=True,pages={r['document']:r['pages'] for r in results},scope='TYPESETTING_ONLY')))


if __name__ == '__main__':
    main()
