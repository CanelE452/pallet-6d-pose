"""Verify normal main publication against local bytes and remote HTTP bodies."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import subprocess
from urllib.request import Request, urlopen
from . import common as C


def run(sha, private):
    remote = subprocess.check_output(['git','-C',str(C.ROOT),'ls-remote',
        'origin','refs/heads/main'], text=True).split()[0]
    assert sha == remote
    filenames = ['RESULTS_KO.md','REPRODUCE.md','COMPLETION_AUDIT.md',
        'METRICS.json','REQUESTED_PRIMARY_PAIRED.json',
        'SUPPLEMENT_INDEPENDENT_VERIFICATION.json','COMPLETION_COMPUTE_RECEIPT.json',
        'EXPERIMENT_STATUS_COMPLETION.csv','COMPARISON_TRANSITIONS_COMPLETE.csv',
        'CHANGED_B1_CANDIDATES_COMPLETE.csv','OLD_P_CHANGED_CANDIDATES_COMPLETE.csv',
        'B1_PERMUTATION_REPAIR_RESULTS.json','A3_RATIO_AXIS_CORRECTION.json',
        'C5_SUBGROUPS_SUPPLEMENT.json','B_C_COMPLETION_SUPPLEMENT_KO.md']
    files = [C.DOC/name for name in filenames]
    files += sorted((C.DOC/'FIGURES').glob('*.png'))
    if private:
        files.append(C.DOC/'PUBLICATION_COMPLETION_VERIFICATION.json')
    assert len(list((C.DOC/'FIGURES').glob('*.png'))) == 10
    def fetch(path):
        relative = str(path.relative_to(C.ROOT))
        url = 'https://raw.githubusercontent.com/CanelE452/pallet-6d-pose/'+sha+'/'+relative
        with urlopen(Request(url,headers={'User-Agent':'pallet-experiment-verifier'}),timeout=40) as response:
            body = response.read(); status = response.status
        actual = hashlib.sha256(body).hexdigest()
        assert status == 200 and actual == C.sha(path), relative
        return dict(path=relative,url=url,HTTP_status=status,sha256=actual,bytes=len(body))
    with ThreadPoolExecutor(max_workers=2) as pool:
        bodies = list(pool.map(fetch, files))
    rendered = []
    for name in ['RESULTS_KO.md','REPRODUCE.md','COMPLETION_AUDIT.md',
                 'FIGURES/09_requested_stage1_paired_effects.png']:
        relative = str((C.DOC/name).relative_to(C.ROOT))
        url = 'https://github.com/CanelE452/pallet-6d-pose/blob/'+sha+'/'+relative
        with urlopen(Request(url,headers={'User-Agent':'pallet-experiment-verifier'}),timeout=40) as response:
            assert response.status == 200
            rendered.append(dict(url=url,HTTP_status=response.status))
    start = C.read(C.PRIVATE/'START.json')
    status = subprocess.check_output(['git','-C',str(C.SOURCE),'status','--porcelain=v1','-z'])
    diff = subprocess.check_output(['git','-C',str(C.SOURCE),'diff','--binary','HEAD'])
    assert hashlib.sha256(status).hexdigest() == start['status_sha256']
    assert hashlib.sha256(diff).hexdigest() == start['tracked_diff_sha256']
    # Recheck main after fetching every body, detecting intervening updates.
    final_remote = subprocess.check_output(['git','-C',str(C.ROOT),'ls-remote',
        'origin','refs/heads/main'], text=True).split()[0]
    assert final_remote == sha
    packet = dict(status='PASS_REMOTE_SHA_HTTP_AND_BYTE_HASH',
        checked_UTC=datetime.now(timezone.utc).isoformat(),main_SHA_at_verification=sha,
        normal_push=True,force_push=False,automatic_merge=False,
        original449_changes_preserved=True,original_status_and_diff_SHA_unchanged=True,
        raw_file_checks=bodies,GitHub_render_checks=rendered,PNG_count=10,
        note='Receipt records the verified science commit. A subsequent receipt-only commit cannot include its own SHA; the final main SHA is checked again privately and reported to the user.')
    if private:
        destination = C.PRIVATE/'FINAL_COMPLETION_REMOTE_VERIFICATION.json'
    else:
        destination = C.DOC/'PUBLICATION_COMPLETION_VERIFICATION.json'
    C.write(destination, packet)
    print('REMOTE_COMPLETION_PASS', sha, len(bodies),'exact raw bodies;',
          len(rendered),'render links;', 'original449 preserved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sha', required=True)
    parser.add_argument('--private', action='store_true')
    args = parser.parse_args()
    run(args.sha, args.private)
