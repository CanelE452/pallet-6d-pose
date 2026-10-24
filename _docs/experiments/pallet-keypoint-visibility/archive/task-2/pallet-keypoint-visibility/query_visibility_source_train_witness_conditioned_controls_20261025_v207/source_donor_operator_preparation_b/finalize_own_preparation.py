"""Seal existing own text outputs only; never rerun source metadata or assets."""
from pathlib import Path
import hashlib,json,os,stat

H=Path(__file__).parent
NAMES=('prepare_source_dependencies.py','DEPENDENCY_READINESS.json',
       'SOURCE_TRAIN59_DONOR_BINDINGS.jsonl','OPERATOR_READINESS.json',
       'SOURCE_SYNTH_NATIVE_CODE_POINTERS.json','INVENTED_GEOMETRY_CHECKS.json',
       'KO.md','PREPARATION_WRITER_FAILURE_PRESERVED.json')

def read_own(name):
    assert name in NAMES or name==Path(__file__).name
    p=H/name
    assert p.is_absolute() and '..' not in p.parts and 'v112' not in str(p).casefold()
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=nxt
        final=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    assert stat.S_ISREG(os.fstat(final).st_mode)
    with os.fdopen(final,'rb') as stream:return stream.read()

def binding(name):
    b=read_own(name)
    return dict(path=str(H/name),sha256=hashlib.sha256(b).hexdigest(),size_bytes=len(b))

def write_new(name,obj):
    assert name in {'PREPARATION_FINAL_BINDING_FAILURE_PRESERVED.json','FINAL_READY.json'}
    p=H/name
    assert 'v112' not in str(p).casefold()
    # H was opened with no-follow above; no existing file may be overwritten.
    with p.open('x',encoding='utf-8') as stream:
        json.dump(obj,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')

def main():
    bindings={n:binding(n) for n in NAMES}
    assert bindings['prepare_source_dependencies.py']['sha256']=='92c418a711eed0d72866fb773097e2cdf9dd343a7993ebe9e6ec3a3a2cbcf28c'
    readiness=json.loads(read_own('DEPENDENCY_READINESS.json'))
    checks=json.loads(read_own('INVENTED_GEOMETRY_CHECKS.json'))
    assert readiness['completed'] and readiness['PASS'] and readiness['donor_count']==59
    assert checks['completed'] and checks['PASS'] and checks['count']==8
    failure=dict(completed=True,preserved=True,terminal_chunk='8c092d',exit_code=1,
        stage='Only final own KO.md binding; readiness outputs already written',
        error='AssertionError: rb suffix whitelist excluded own .md',
        original_code=bindings['prepare_source_dependencies.py'],
        correction='Separate own-artifact-only finalizer explicitly whitelists own filenames; original code and artifacts unchanged',
        source_metadata_rerun=False,source_asset_reads=0)
    write_new('PREPARATION_FINAL_BINDING_FAILURE_PRESERVED.json',failure)
    ready=dict(completed=True,PASS=True,prepared_code_metadata_only=True,
        scope=readiness['scope'],code=bindings['prepare_source_dependencies.py'],
        finalizer=binding(Path(__file__).name),readiness=bindings['DEPENDENCY_READINESS.json'],
        donors=bindings['SOURCE_TRAIN59_DONOR_BINDINGS.jsonl'],operator=bindings['OPERATOR_READINESS.json'],
        source_synth=bindings['SOURCE_SYNTH_NATIVE_CODE_POINTERS.json'],
        checks=bindings['INVENTED_GEOMETRY_CHECKS.json'],KO=bindings['KO.md'],
        failures_preserved=['PREPARATION_WRITER_FAILURE_PRESERVED.json','PREPARATION_FINAL_BINDING_FAILURE_PRESERVED.json'],
        donor_count=59,source_scenes=7,source_objects=10,invented_geometry_checks=8,
        source_metadata_rerun=False,actual_source_asset_stat_hash_header_decode_or_new_ROI_reads=0,
        actual_intervention_E_fit_or_admission=False,
        limitations=['Recorded asset bindings only; fresh binary availability/integrity not checked',
          'SOURCE SYNTH field declarations are code evidence; actual corpus field presence remains UNKNOWN',
          'Point-reference, composite-region teacher and finite-end/support authority remain distinct',
          'No actual placement or same-footprint SHAM feasibility claim; infeasible cases must remain unresolved'])
    write_new('FINAL_READY.json',ready)
    print(json.dumps(dict(completed=True,PASS=True,ready_path=str(H/'FINAL_READY.json'),
        ready_sha256=hashlib.sha256((H/'FINAL_READY.json').read_bytes()).hexdigest(),
        original_metadata_not_rerun=True,source_assets_opened=0)))

if __name__=='__main__':main()
