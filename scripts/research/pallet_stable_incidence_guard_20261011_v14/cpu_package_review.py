"""Run the unchanged CPU driver in its actual Python package and source env.

Both failed setup attempts remain separate inputs. No guard or test changes.
"""
import json
import os
from pathlib import Path
from . import cpu_checks as K

DOC=K.DOC;REPO=K.REPO
FILES=('PACKAGE_CPU_PROTOCOL.json','PACKAGE_CPU_STARTED.json','PACKAGE_CPU_CHECKS.json')
CONFIG=dict(PALLET_SOURCE_ROOT='/home/minjae/Documents/github/pallet-pose',
    PALLET_BASELINE_ROOT='/home/minjae/Documents/github/pallet-pose-handoff-20261006')
MARKER=dict(original_failed_CPU_attempt_preserved=True,failed_environment_review_preserved=True,
    original_tests_and_guard_changed=False,original_tests_run=7,original_errors=3,
    prior_environment_review_tests_run=0,Python_package_and_environment_setup_only=True,
    normal_original_package_import=True,configured_environment=CONFIG,
    configured_actual_groups=7,controlled_fixtures_are_not_natural_optimizer_results=True)

def require(v,s):
    if not v:raise ValueError(s)

def main():
    old=json.loads((DOC/'CPU_CHECKS.json').read_text())
    prior=json.loads((DOC/'REVIEW_CPU_CHECKS.json').read_text())
    require(old['passed'] is False and old['tests_run']==7 and old['failures']==[]
        and len(old['errors'])==3 and all("KeyError: 'PALLET_SOURCE_ROOT'\n" in e['traceback'] for e in old['errors'])
        and old['error'] is None and old['source_unchanged'] is True and old['opencv_threads_restored'] is True,
        'preserve original four-PASS/three-source-environment errors')
    require(prior['passed'] is False and prior['tests_run']==0 and prior['errors']==prior['failures']==[]
        and prior['error']['type']=='ImportError'
        and prior['error']['message']=='attempted relative import with no known parent package'
        and prior['source_unchanged'] is True,'preserve failed package setup review')
    original_inputs,original_write,original_files=K.inputs,K.write,K.FILES
    current=original_inputs();require(current==old['inputs'],'original CPU sources changed')
    for prefix,record in (('',old),('REVIEW_',prior)):
        protocol=DOC/(prefix+'CPU_PROTOCOL.json');started=DOC/(prefix+'CPU_STARTED.json')
        own=json.loads(protocol.read_text());start=json.loads(started.read_text())
        require(record['inputs']==own['inputs'] and record['own_protocol']==K.binding(protocol)
            and start['protocol']==K.binding(protocol),'preserved CPU attempt authority differs')
    for binding in prior['inputs'].values():
        require(K.binding(REPO/binding['path'])==binding,'prior environment-review dependency changed')
    def inputs():
        out=original_inputs()
        paths=[Path(__file__).resolve(),K.CODE/'cpu_environment_review.py']
        paths.extend(DOC/n for n in ('CPU_PROTOCOL.json','CPU_STARTED.json','CPU_CHECKS.json',
            'REVIEW_CPU_PROTOCOL.json','REVIEW_CPU_STARTED.json','REVIEW_CPU_CHECKS.json'))
        for path in paths:out[str(path.relative_to(REPO))]=K.binding(path)
        return out
    def write(path,value):
        value['package_review']=MARKER;original_write(path,value)
    previous={key:os.environ.get(key) for key in CONFIG}
    try:
        K.inputs,K.write,K.FILES=inputs,write,FILES
        for key,value in CONFIG.items():
            require(Path(value).is_dir(),'declared source environment directory missing')
            os.environ[key]=value
        K.main()
    finally:
        K.inputs,K.write,K.FILES=original_inputs,original_write,original_files
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value
        require(previous=={key:os.environ.get(key) for key in CONFIG},'CPU environment restoration failed')

if __name__=='__main__':main()
