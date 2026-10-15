"""Score frozen v18c poses with a metadata-only legacy row adapter.

The original caller omitted tangent_control_audit, which the unchanged v15
scorer copies into its report without using it in any metric. Supply an
explicit missing-metadata marker in memory. Predictions, pose values,
references, metrics and cohort membership remain byte-for-byte unchanged.
"""
from pathlib import Path
import types

from . import refine_grazing_image_chain_mixtures_v4_18c as V


def main():
    C,E=V.C,V.E;output=V.OUTPUT
    if any((output/name).exists()for name in ('SCORE_SEAL.json','CAL_SCORE.json','TEST_SCORE.json','REMAINDER_SCORE.json')):
        raise FileExistsError('Never replace any existing score report')
    original_sha=C.sha(output/'PREDICTIONS.jsonl.gz');rows=list(C.rows(output/'PREDICTIONS.jsonl.gz'))
    input_pose_digest=E._digest([{k:r[k]for k in ('id','partition','arms','fixed_family')}for r in rows])
    def compatible_rows(path):
        if Path(path).resolve()!=(output/'PREDICTIONS.jsonl.gz').resolve():
            yield from C.rows(path);return
        for row in rows:
            if 'tangent_control_audit'in row:raise ValueError('Adapter is only for the omitted metadata field')
            yield {**row,'tangent_control_audit':dict(schema='SCORE_ONLY_MISSING_METADATA_MARKER_V18C',
                not_recorded_in_original_prediction_row=True,metric_input=False,
                no_prediction_pose_state_cost_or_reference_changed=True)}
    facade=types.SimpleNamespace(**{**C.__dict__,'rows':compatible_rows})
    scorer=V.BASE.ARCHIVE.score
    run=types.FunctionType(scorer.__code__,{**scorer.__globals__,'C':facade,'ARMS':V.ARMS},scorer.__name__)
    run(output)
    if C.sha(output/'PREDICTIONS.jsonl.gz')!=original_sha:raise ValueError('Frozen predictions changed')
    current=list(C.rows(output/'PREDICTIONS.jsonl.gz'))
    if E._digest([{k:r[k]for k in ('id','partition','arms','fixed_family')}for r in current])!=input_pose_digest:
        raise ValueError('Metric pose inputs changed')
    C.write(output/'SCORER_COMPATIBILITY_RECEIPT.json',dict(schema='METADATA_ONLY_SCORER_ADAPTER_RECEIPT_V18C',
        original_scorer_failure='KeyError: tangent_control_audit; no score reports written',
        predictions_unchanged_sha256=original_sha,metric_pose_inputs_digest=input_pose_digest,
        supplied_field='tangent_control_audit',field_used_in_numeric_metrics=False,
        numeric_metric_and_reference_code_unchanged=True,all238_rows_retained=True,
        adapter_source=E._binding(Path(__file__)),scorer_source=E._binding(Path(V.BASE.ARCHIVE.__file__)),
        prediction_seal=E._binding(output/'PREDICTION_SEAL.json'),score_seal=E._binding(output/'SCORE_SEAL.json')))
    C.write(output/'SCORER_COMPATIBILITY_SEAL.json',dict(receipt_sha256=C.sha(output/'SCORER_COMPATIBILITY_RECEIPT.json')))


if __name__=='__main__':main()
