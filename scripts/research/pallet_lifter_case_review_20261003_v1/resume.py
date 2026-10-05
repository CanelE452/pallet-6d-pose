"""One resume command: verified optional review import, frozen pass, separated metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from audit import HERE, OUT, dump
from run_pipeline import ASSETS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('review_export',nargs='?',type=Path,help='Optional exported human JSON; saved local review is the default')
    p.add_argument('--asset-root',type=Path,default=ASSETS)
    args = p.parse_args()
    # Human-reference absence/error never stops independent authorized frozen inference.
    process = subprocess.run([sys.executable,'-X','utf8',str(HERE/'run_pipeline.py'),'run',
                              '--asset-root',str(args.asset_root)],check=False)
    imported = None
    if args.review_export is not None:
        command = [sys.executable,'-X','utf8',str(HERE/'review/serve.py'),'import',
                   '--manifest',str(OUT/'review/MANIFEST.json'),'--plan',str(OUT/'LIFTER_EVALUATION_PLAN.json'),
                   '--contract',str(OUT/'review/CORNER_CONTRACT.json'),
                   '--store',str(OUT/'review/annotations_in_progress.json'),
                   '--input',str(args.review_export),'--merge']
        imported = subprocess.run(command,check=False).returncode
    sys.path.insert(0,str(HERE/'review'))
    from serve import Context
    context=Context(OUT/'review/MANIFEST.json',OUT/'LIFTER_EVALUATION_PLAN.json',
                    OUT/'review/CORNER_CONTRACT.json',OUT/'review/annotations_in_progress.json')
    submitted=context.export()
    reference_path=None
    if submitted.get('records') and submitted.get('source_kind')=='human_reviewed':
        # Submitted references are immutable snapshots; preserve earlier exports and edits.
        signature=hashlib.sha256(json.dumps({'bindings':submitted['bindings'],'records':submitted['records']},
                         sort_keys=True,ensure_ascii=False,allow_nan=False).encode('utf-8')).hexdigest()[:12]
        reference_path=OUT/'review'/('LIFTER_REFERENCE_REVIEWED_'+signature+'.json')
        if not reference_path.exists():
            dump(reference_path,submitted,frozen=True)
        dump(OUT/'review/REFERENCE_EXPORT_STATUS.json',{'source_kind':'machine_proposed',
              'submitted_reference_file':reference_path.name,'submitted_reference_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
              'submitted_records':len(submitted['records'])})
    command=[sys.executable,'-X','utf8',str(HERE/'metrics/evaluate.py'),
        '--manifest',str(OUT/'review/MANIFEST.json'),'--plan',str(OUT/'LIFTER_EVALUATION_PLAN.json'),
        '--corner-contract',str(OUT/'review/CORNER_CONTRACT.json'),
        '--predictions',str(OUT/'raw_predictions/ALL_STORED_FRAMES.jsonl'),
        '--output-dir',str(OUT)]
    if reference_path is not None:
        command+=['--reviewed',str(reference_path)]
    evaluated = subprocess.run(command,check=False).returncode
    record = {'schema_version':'lifter_resume_status_v1','inference_exit_code':process.returncode,
              'optional_reference_import_exit_code':imported,'metrics_exit_code':evaluated,
              'reference_source':'local_saved_store' if args.review_export is None else 'provided_human_export',
              'no_background_polling':True,'status':'VERIFIED_COMPLETE' if process.returncode==0 and evaluated==0 and imported in (None,0) else 'BLOCKED_CONTRACT'}
    dump(OUT/'RESUME_STATUS.json',record)
    return 0 if record['status']=='VERIFIED_COMPLETE' else 2


if __name__=='__main__':
    raise SystemExit(main())
