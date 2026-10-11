"""Gate continuation publication; distinguish arithmetic pass from contract fail."""
import argparse
import ast
import hashlib
import re
import subprocess
import time
from pathlib import Path
from PIL import Image
from . import common as C

CONTINUATION_START_UNIX = 1791714116


def run(output_name='COMPLETION_PREPUBLICATION_AUDIT.json'):
    assert Path(output_name).name == output_name and output_name.endswith('.json')
    assert not (C.DOC/output_name).exists()
    C.lock_ready()
    required = ['COMPLETION_AUDIT.md', 'EXPERIMENT_STATUS_COMPLETION.csv',
        'RESULTS_KO.md', 'REPRODUCE.md', 'REQUESTED_PRIMARY_PAIRED.json',
        'COMPARISON_TRANSITIONS_COMPLETE.csv', 'C5_SUBGROUPS_SUPPLEMENT.json',
        'A3_RATIO_AXIS_CORRECTION.json', 'SUPPLEMENT_INDEPENDENT_VERIFICATION.json']
    assert all((C.DOC/name).is_file() for name in required)
    verified = C.read(C.DOC/'SUPPLEMENT_INDEPENDENT_VERIFICATION.json')
    assert verified['status'] == 'PASS', verified['status']
    start = C.read(C.PRIVATE/'START.json')
    def git(*args):
        return subprocess.check_output(['git', '-C', str(C.SOURCE), *args])
    status = git('status', '--porcelain=v1', '-z')
    diff = git('diff', '--binary', 'HEAD')
    assert status == (C.PRIVATE/'ORIGINAL_STATUS.bin').read_bytes()
    assert diff == (C.PRIVATE/'ORIGINAL_TRACKED.diff').read_bytes()
    assert hashlib.sha256(status).hexdigest() == start['status_sha256']
    assert hashlib.sha256(diff).hexdigest() == start['tracked_diff_sha256']
    receipts = [__import__('json').loads(line) for line in
                (C.PRIVATE/'PNP_RECEIPTS.jsonl').read_text().splitlines()]
    total_calls = sum(row['total'] for row in receipts)
    assert 49315 <= total_calls <= 50000
    previous = C.read(C.DOC/'COMPUTE_RECEIPT.json')
    assert previous['new_OpenCV_solve_refine_calls'] == 48277
    assert previous['new_classifier_fits'] == 8
    # The pause after the original publication contained no analysis/fitting.
    # Count the entire resumed audit phase conservatively, including editing.
    baseline = C.read(C.DOC/'PREPUBLICATION_AUDIT.json')['current_project_wall_minutes']
    continuation = (time.time()-CONTINUATION_START_UNIX)/60
    assert baseline + continuation < 60, ('Cumulative analysis wall budget', baseline, continuation)
    compute_path = C.DOC/'COMPLETION_COMPUTE_RECEIPT.json'
    compute_output = dict(
        status='ACTUAL_CUMULATIVE_WITH_PROTOCOL_REPAIR',
        original_receipt=C.binding(C.DOC/'COMPUTE_RECEIPT.json'),
        PnP_receipts=receipts, new_OpenCV_solve_refine_calls=total_calls,
        extra_PnP_calls=total_calls-48277, PnP_limit=50000,
        fixed_permutation_repair_calls=1038,
        transition_candidate_recovery_calls=total_calls-49315,
        unused_PnP_budget=50000-total_calls,
        classifier_fits=8, corrector_fits=2, total_corrector_updates=2000,
        extra_fits_or_updates=0, extra_CNN_or_GPU_inference=0,
        GPU_active_device_minutes=previous['GPU_active_device_minutes'],
        GPU_peak_allocated_bytes=previous['GPU_peak_allocated_bytes'],
        CPU_wall_accounting='Conservative original prepublication project wall plus full active continuation wall; idle between publication and continuation excluded. No retrospective CPU peak estimate.',
        original_conservative_wall_minutes=baseline,
        continuation_start_unix=CONTINUATION_START_UNIX,
        continuation_conservative_wall_minutes=continuation,
        cumulative_conservative_CPU_analysis_wall_minutes=baseline+continuation,
        CPU_limit_minutes=60, CPU_peak_memory='NOT_MEASURED',
        protected_FINAL_new_scoring=0, new_RGB_generation=0,
        new_depth_download_or_inference=0)
    if compute_path.exists():
        assert C.read(compute_path)['new_OpenCV_solve_refine_calls'] == total_calls
    else:
        C.write(compute_path, compute_output)
    links = []; figures = []; paths = []
    for folder in [C.DOC, C.ROOT/'scripts/research/pallet_wd_info_fast_20261011']:
        for f in sorted(folder.rglob('*')):
            if not f.is_file(): continue
            assert f.suffix not in ('.pt','.pth','.npz','.jpg','.jpeg','.pdf','.tex','.pyc')
            assert '__pycache__' not in f.parts and not f.name.endswith('.pending')
            assert f.stat().st_size < 16*1024*1024, f.name
            if f.suffix in ('.md','.json','.csv','.py','.mplstyle','.patch'):
                body = f.read_text()
                assert not any(token in body for token in
                    ['/'+'home'+'/', '/'+'dev'+'/'+'shm'+'/', '/'+'tmp'+'/']), f.name
                if f.suffix == '.py': ast.parse(body, filename=f.name)
                if f.suffix == '.md':
                    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', body):
                        if target.startswith(('https:','http:','mailto:')): continue
                        relative = target.split('#')[0].strip('<>')
                        if not relative: continue
                        path = (f.parent/relative).resolve()
                        exists = path.exists()
                        generated_self = path == C.DOC/output_name
                        if generated_self:
                            exists = True
                        if not exists and path.is_relative_to(C.ROOT):
                            exists = subprocess.run(['git','-C',str(C.ROOT),'cat-file','-e',
                                'HEAD:'+str(path.relative_to(C.ROOT))], capture_output=True).returncode == 0
                        assert exists, (f.name, target)
                        links.append(dict(document=f.name,target=target,
                            generated_by_this_audit=generated_self))
            if f.suffix == '.png':
                with Image.open(f) as im: im.verify()
                with Image.open(f) as im:
                    figures.append(dict(path=str(f.relative_to(C.ROOT)), size=im.size,
                        DPI=im.info.get('dpi'), sha256=C.sha(f)))
            paths.append(str(f.relative_to(C.ROOT)))
    assert len(figures) == 10
    C.write(C.DOC/output_name, dict(
        status='PASS_PUBLICATION_AND_ARITHMETIC',
        scientific_global_contract_status='FAIL_WITH_RELATED_COMPARISONS_BLOCKED',
        global_contract_failures=['C/D common synthetic split',
            'C1 signed-front physical pose restoration (2/16 exact cases)',
            'Requested Stage1 contrast was not the original pre-execution lock contrast'],
        original_status_entries=449, original_status_and_diff_bytes_identical=True,
        original_status_sha256=start['status_sha256'],
        original_tracked_diff_sha256=start['tracked_diff_sha256'],
        frozen_PLAN_core_hashes_unchanged=True,
        current_cumulative_conservative_CPU_analysis_wall_minutes=baseline+continuation,
        independent_supplement_verifier=C.binding(C.DOC/'SUPPLEMENT_INDEPENDENT_VERIFICATION.json'),
        cumulative_compute=C.binding(C.DOC/'COMPLETION_COMPUTE_RECEIPT.json'),
        local_markdown_links=links, figures=figures,
        public_file_bindings=[C.binding(C.ROOT/path) for path in paths],
        raw_RGB_reference_coordinates_private_checkpoints_or_paths_found=0,
        historical_publication_proofs='Earlier bindings describe their historical commit versions; continuation proof describes current reviewed files.'))
    assert (C.DOC/output_name).is_file()
    paths.append(str((C.DOC/output_name).relative_to(C.ROOT)))
    (C.PRIVATE/'COMPLETION_PUBLICATION_PATHS.txt').write_text('\n'.join(paths)+'\n')
    print('COMPLETION_PUBLICATION_PASS', len(paths), 'files;', len(links),
          'links;', len(figures), 'PNGs; cumulative conservative CPU minutes', baseline+continuation)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-name', default='COMPLETION_PREPUBLICATION_AUDIT.json')
    run(parser.parse_args().output_name)
