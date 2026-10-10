"""Resume the frozen runtime after its zero-call legacy namespace failure.

The original protocol, runtime.py and failed attempt remain byte unchanged.
One outer legacy context retains the baseline package namespace throughout the
unchanged runtime.measure call. No coordinate, solver or timing policy changes.
"""
from __future__ import annotations

from pathlib import Path
import time

from . import common as C
from . import runtime as R


def failed_prefix(args):
    directory=Path(args.failed_output)
    run_path=directory/'RUNTIME.json'
    rows_path=directory/'RUNTIME_ROWS.jsonl.gz'
    journal_path=directory/'RUNTIME_STARTED.json'
    run=C.read(run_path)
    journal=C.read(journal_path)
    C.require(run['status']=='FAILED_RUNTIME' and not run['complete'] and
              not run['statistics_official'] and not run['model_initialization_complete'],
              'only the preserved zero-call import failure may be resumed')
    C.require(run['reason']==dict(type='ModuleNotFoundError',message="No module named 'scripts.research.pallet_joint_action_handoff_20261006_v1'"),
              'unrecognized original runtime failure; do not retry experiments')
    C.require(run['execution'].get('pipeline_calls_started',0)==0 and
              run['execution'].get('pipeline_calls_complete',0)==0 and not list(C.rows(rows_path)),
              'previous timed calls exist; fixed600 execution cannot be restarted')
    C.require(journal['status']=='FAILED_RUNTIME' and not journal['counts'] and
              run['configured_pipeline_calls']==600 and run['arms']==list(R.ARMS), 'failed journal/schedule differs')
    C.require(run['protocol']==C.binding(args.protocol) and run['runtime_code']==C.binding(R.__file__),
              'preserved failed attempt used different frozen runtime')
    C.bound(rows_path,run['raw_rows'],'preserved failed zero-row runtime')
    return dict(runtime=C.binding(run_path),raw_rows=C.binding(rows_path),journal=C.binding(journal_path),
        archived_public_names=dict(runtime='FAILED_RUNTIME_PREFIX.json',raw_rows='FAILED_RUNTIME_ROWS.jsonl.gz',
                                   journal='FAILED_RUNTIME_STARTED.json'),
        configured_timed_calls=600,actual_timed_calls=0,actual_timed_rows=0,
        successful_model_initializations=0,reason=run['reason'],
        correction='Retain one outer legacy context so cached baseline resolver imports remain reachable.')


def fixed(args):
    C.verify_protocol(args)
    previous=failed_prefix(args)
    return dict(protocol=C.binding(args.protocol),frozen_runtime=C.binding(R.__file__),
                frozen_pipeline=C.binding(Path(__file__).with_name('pipeline.py')),
                resume_code=C.binding(__file__),previous_attempt=previous,
                accuracy_seal=C.binding(Path(args.accuracy_output)/'GEOMETRY_SEAL.json'))


def prepare(args):
    R.prereqs(args)
    previous=C.prior_snapshot(args.prior_bindings)
    C.output_path(args,'RUNTIME_RESUME_STARTED.json')
    C.require(Path(args.output).resolve()!=Path(args.failed_output).resolve(), 'preserve failed runtime directory')
    path=Path(args.resume_protocol)
    C.require(path.resolve()==(C.DOC/'RUNTIME_RESUME_PROTOCOL.json').resolve(), 'resume protocol must be the new own DOC artifact')
    C.write_new(path,dict(schema='validated_boundary_runtime_namespace_resume_v2',
        status='PREPARED_NOT_MEASURED',fixed_inputs=fixed(args),fixed_timed_calls=600,
        arms=list(R.ARMS),model_decoder_solver_and_timed_body_unchanged=True,
        wrapper='with common.legacy_context(args) around unchanged runtime.measure(args)',
        no_accuracy_repeated=True,no_training_repeated=True,no_GT_tuning=True,
        previous_failure_preserved=True,protected_materialized_files=len(previous)))
    print('BOUNDARY_RUNTIME_RESUME_FROZEN',C.sha(path),flush=True)


def preflight(args):
    plan=C.read(args.resume_protocol)
    C.require(plan['fixed_inputs']==fixed(args) and plan['fixed_timed_calls']==600 and
              plan['arms']==list(R.ARMS) and plan['model_decoder_solver_and_timed_body_unchanged'] and
              plan['previous_failure_preserved'],'frozen runtime resume protocol differs')
    R.prereqs(args)
    C.output_path(args,'RUNTIME_RESUME_STARTED.json')
    for name in ('RUNTIME.json','RUNTIME_ROWS.jsonl.gz','RUNTIME_STARTED.json','RUNTIME_RESUME_RECEIPT.json'):
        C.output_path(args,name)
    C.require(Path(args.output).resolve()!=Path(args.failed_output).resolve(), 'preserve failed runtime directory')
    return plan


def measure(args):
    plan=preflight(args)
    C.write_new(C.output_path(args,'RUNTIME_RESUME_STARTED.json'),dict(
        resume_protocol=C.binding(args.resume_protocol),configured_new_timed_calls=600,
        previous_actual_timed_calls=0,previous_failed_attempt=plan['fixed_inputs']['previous_attempt'],
        claim='One resumed execution only; existing/interrupted output claims are never overwritten.'))
    started=time.monotonic()
    # Keep the baseline namespace installed across runtime's initial resource
    # probe context and its subsequent Pipeline context. The timer body is the
    # already frozen function; this outer wrapper lies outside every interval.
    with C.legacy_context(args):
        result=R.measure(args)
    C.write_new(C.output_path(args,'RUNTIME_RESUME_RECEIPT.json'),dict(
        schema='validated_boundary_runtime_namespace_resume_receipt_v2',complete=result['complete'],
        resume_protocol=C.binding(args.resume_protocol),previous_failed_attempt=plan['fixed_inputs']['previous_attempt'],
        result=C.binding(Path(args.output)/'RUNTIME.json'),rows=C.binding(Path(args.output)/'RUNTIME_ROWS.jsonl.gz'),
        resumed_actual_timed_calls=result['execution'].get('pipeline_calls_complete',0),
        frozen_runtime_body_unchanged=True,new_accuracy_frames=0,new_training_updates=0,
        outside_interval_wrapper_wall_seconds=time.monotonic()-started))


def main():
    parser=C.parser(__doc__,('freeze','preflight','measure'))
    parser.set_defaults(output=str(C.PRIVATE/'runtime_resume'))
    parser.add_argument('--accuracy-output',default=str(C.PRIVATE/'accuracy'))
    parser.add_argument('--failed-output',default=str(C.PRIVATE/'runtime'))
    parser.add_argument('--resume-protocol',default=str(C.DOC/'RUNTIME_RESUME_PROTOCOL.json'))
    parser.add_argument('--remaining-seconds',type=float,default=3600.)
    args=parser.parse_args()
    if args.stage=='freeze':prepare(args)
    elif args.stage=='preflight':
        preflight(args);print('BOUNDARY_RUNTIME_RESUME_PREFLIGHT_PASS',flush=True)
    else:measure(args)


if __name__=='__main__':main()
