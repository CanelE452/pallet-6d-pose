"""Additive source-closure guard around the immutable remaining143 scorer."""
from pathlib import Path
import argparse

from . import common as C
from . import score_remainder_derived_reference_v4_14 as SCORE

CLOSURE=SCORE.REMAINDER/'independent_pipeline_audit_v4_14'
EXPECTED_CLOSURE_SHA256='178b5d3833ac8cc5468e502eade32c3f0dc6101645f0eb44d274a8c6f5d4ed8d'


def verify_additional_source_closure():
    path=CLOSURE/'ADDITIONAL_EXECUTED_SOURCE_BINDINGS_V1.json'
    if C.sha(path)!=EXPECTED_CLOSURE_SHA256:
        raise ValueError('Approved pre-GT source closure receipt changed')
    receipt=C.read(path);source=receipt['audit_source']
    if C.sha(C.ROOT/source['path'])!=source['sha256']:
        raise ValueError('Source-closure helper differs from its approved binding')
    snapshot=CLOSURE/'audit_code_snapshot'/Path(source['path']).name
    if C.sha(snapshot)!=source['sha256']:
        raise ValueError('Source-closure helper snapshot changed')
    # Verify its own file before importing and invoking the helper.
    from . import audit_remainder_pipeline_v4_14 as AUDIT
    verified=AUDIT.verify_source_closure(CLOSURE)
    if verified!=receipt:raise ValueError('Closure receipt changed during verification')
    return dict(path=str(path.relative_to(C.ROOT)),sha256=EXPECTED_CLOSURE_SHA256,
        audit_source=source,conservative_source_count=len(receipt['conservatively_bound_source_union_sha256']),
        traced_transitive_count=len(receipt['traced_transitive_runtime_dependencies']))


def score(output=SCORE.OUTPUT):
    output=Path(output).resolve()
    if (output/'SCORE_SEAL.json').exists() or (output/'SOURCE_CLOSED_SCORE_RECEIPT_V1.json').exists():
        raise FileExistsError('Preserve every completed143 score and guard receipt')
    binding=verify_additional_source_closure()
    C.write(output/'SOURCE_CLOSED_SCORE_PREPARATION_V1.json',dict(source_closure=binding,
        wrapper_source_sha256=C.sha(__file__),frozen_scorer_sha256=C.sha(SCORE.__file__),
        annotation_numeric_decoded_by_wrapper=False,closure_verified_immediately_before_scorer=True))
    summary=SCORE.score(output)
    if verify_additional_source_closure()!=binding:raise ValueError('Source closure changed while scoring')
    C.write(output/'SOURCE_CLOSED_SCORE_RECEIPT_V1.json',dict(passed=True,source_closure=binding,
        wrapper_source_sha256=C.sha(__file__),frozen_scorer_sha256=C.sha(SCORE.__file__),
        score_seal_sha256=C.sha(output/'SCORE_SEAL.json'),
        score_sha256=C.sha(output/'POSE_REFERENCE_SCORE.json'),
        preparation_sha256=C.sha(output/'SOURCE_CLOSED_SCORE_PREPARATION_V1.json'),
        all143_prediction_seal_checked_by_frozen_scorer_before_first_annotation_decode=True,
        no_prediction_source_protocol_or_parameter_modified=True))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=SCORE.OUTPUT)
    args=parser.parse_args();print(score(args.output),flush=True)
