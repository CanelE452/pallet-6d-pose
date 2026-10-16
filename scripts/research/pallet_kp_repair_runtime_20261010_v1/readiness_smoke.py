"""Zero-model readiness controls for the unchanged original runtime adapter.

No timing interval, model import, head forward, detector, ray, or pose solve is
executed. Schedule comparison reopens old *stored* rows and checks job IDs only.
"""
import argparse
import copy
import importlib.abc
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode=True
from . import adapter as A

DENIED=('torch','cv2','numpy','open3d','ultralytics',
        'scripts.research.pallet_observation_refiner_20261009_v1')


class NoHeavyImports(importlib.abc.MetaPathFinder):
    def __init__(self):self.attempted=[]
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==prefix or fullname.startswith(prefix+'.') for prefix in DENIED):
            self.attempted.append(fullname)
            raise AssertionError('Readiness smoke attempted heavy import: '+fullname)
        return None


def rejected(label,args,expected,call=A.output_guards):
    try:call(args)
    except RuntimeError as error:
        A.require(expected in str(error),'wrong rejection for '+label+': '+str(error))
        return dict(name=label,status='PASS',expected_rejection=expected,
                    actual_rejection=str(error))
    raise AssertionError('Guard accepted forbidden case: '+label)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default=str(A.DOC))
    cli=p.parse_args();output=Path(cli.output).resolve()
    A.require(output==A.DOC.resolve(),'readiness checks write only the new runtime result directory')
    A.check_frozen(_args())
    before=A.prior_snapshot(A.DOC/'PRIOR_PUBLICATION_BINDINGS.json')
    finder=NoHeavyImports();sys.meta_path.insert(0,finder)
    checks=[]
    try:
        warm,measured,proof=A.original_schedule()
        jobs=warm+measured
        original_rows=0
        for index,row in enumerate(A.rows(A.ORIGINAL/'RUNTIME_ROWS.jsonl.gz')):
            A.require(index<len(jobs),'old runtime has excess rows')
            A.require(all(row.get(key)==value for key,value in jobs[index].items()),
                      'original exact job ordering mismatch at '+str(index))
            original_rows+=1
        A.require(original_rows==600,'old runtime schedule reference population differs')
        schedule=dict(schema='original_runtime_schedule_readiness_check_v1',passed=True,
                      exact_original_AST_schedule_matches_stored600_job_rows=True,
                      original_runtime_rows=A.binding(A.ORIGINAL/'RUNTIME_ROWS.jsonl.gz'),
                      proof=proof,arms=list(A.ARMS),warmup=80,measured=520,total=600,
                      original_model_module_imported=False,
                      limits=['Stored job identity/order comparison only; no new timing or pipeline call.'])
        with tempfile.TemporaryDirectory(prefix='pallet-runtime-readiness-',dir='/dev/shm') as temporary:
            temp=Path(temporary)
            args=_args();args.output=str(temp/'output');args.scratch=str(temp/'scratch')
            args.completion=str(temp/'TRAINING_COMPLETION_ABSENT.json')
            # The real measure prefix runs frozen/outputs/prereqs; missing
            # completion must stop before any started claim or heavy import.
            checks.append(rejected('missing_completion_pre_measure',args,
                                   'corrected TRAINING_COMPLETION missing',A.measure))
            A.require(not Path(args.output).exists() and not Path(args.scratch).exists(),
                      'missing completion created output or scratch')
            for field in ['output','scratch']:
                for label,root in [('repository',A.REPO),('old_observation_doc',A.ORIGINAL),
                                   ('old_difficulty_doc',A.REPO/'_docs/experiments/pallet_kp_difficulty_20261010_v1'),
                                   ('old_gate_doc',A.REPO/'_docs/experiments/pallet_kp_supervision_gate_20261010_v1'),
                                   ('old_repair_doc',A.REPAIR),('old_private',A.PRIVATE_ROOTS[0])]:
                    candidate=copy.copy(args);setattr(candidate,field,str(root/'fresh_nonexistent_child'))
                    checks.append(rejected(field+'_protected_'+label,candidate,
                                           'output overlaps a protected experiment/source directory'))
                for label in ['source','baseline','fits']:
                    candidate=copy.copy(args);root=temp/(label+'_read_only')
                    setattr(candidate,{'source':'source_root','baseline':'baseline_root','fits':'fits'}[label],str(root))
                    setattr(candidate,field,str(root/'fresh_nonexistent_child'))
                    checks.append(rejected(field+'_protected_'+label+'_child',candidate,
                                           'output overlaps a protected experiment/source directory'))
            candidate=copy.copy(args);candidate.output=str(temp)
            checks.append(rejected('output_ancestor_of_scratch',candidate,'runtime output/scratch must be disjoint'))
            candidate=copy.copy(args);candidate.scratch=str(temp)
            checks.append(rejected('scratch_ancestor_of_output',candidate,'runtime output/scratch must be disjoint'))
            for name in ['RUNTIME.json','RUNTIME_ROWS.jsonl.gz','RUNTIME_ADAPTER_STARTED.json','RUNTIME_ADAPTER_RECEIPT.json']:
                location=temp/('exists_'+name.replace('.','_'));location.mkdir();(location/name).write_text('preserve')
                candidate=copy.copy(args);candidate.output=str(location)
                checks.append(rejected('existing_'+name,candidate,'preserve completed/interrupted '+name))
            location=temp/'interrupted_scratch';location.mkdir();(location/'RUNTIME_STARTED.json').write_text('preserve')
            candidate=copy.copy(args);candidate.scratch=str(location)
            checks.append(rejected('existing_scratch_runtime_journal',candidate,'preserve interrupted runtime journal'))
            location=temp/'nonempty_external';location.mkdir();(location/'unrelated.txt').write_text('preserve')
            candidate=copy.copy(args);candidate.output=str(location)
            checks.append(rejected('nonempty_external_output',candidate,'external runtime output must be new or empty'))
            candidate=copy.copy(args);candidate.scratch=str(location)
            checks.append(rejected('nonempty_external_scratch',candidate,'runtime scratch must be new or empty'))
            A.output_guards(args)
            candidate=copy.copy(args);candidate.output=str(A.DOC);A.output_guards(candidate)
            checks.append(dict(name='new_exact_doc_and_isolated_external_destinations',status='PASS'))
        A.require(not finder.attempted,'heavy import attempted')
        A.require(A.prior_snapshot(A.DOC/'PRIOR_PUBLICATION_BINDINGS.json')==before,'old311 inputs changed')
        A.write_new(output/'RUNTIME_SCHEDULE_CHECKS.json',schedule)
        A.write_new(output/'RUNTIME_GUARD_CHECKS.json',dict(
            schema='corrected_runtime_zero_execution_guard_check_v1',passed=True,
            checks=checks,checks_count=len(checks),protocol=A.binding(A.DOC/'RUNTIME_ADAPTER_PROTOCOL.json'),
            adapter=A.binding(Path(A.__file__)),test_code=A.binding(Path(__file__)),
            prior_files_preserved=311,heavy_import_attempts=finder.attempted,
            actual_model_imports=0,head_forwards=0,detector_forwards=0,rays=0,pose_solves=0,
            optimizer_updates=0,runtime_pipeline_calls=0,new_timing_intervals=0,
            limits=['Preflight/output-guard and exact stored schedule checks; not an actual runtime benchmark.',
                    'Corrected9000 TRAINING_COMPLETION remains missing; future positive completion/data guards are not exercised here.']))
        print('RUNTIME_READINESS_CHECKS_PASS',len(checks),'actual_pipeline_calls=0')
    finally:
        sys.meta_path.remove(finder)


def _args():
    args=A.parser().parse_args(['preflight'])
    args.completion=args.completion or str(Path(args.fits)/'TRAINING_COMPLETION.json')
    return args


if __name__=='__main__':main()
