"""Score the user's fixed posthoc gain10 only after all four arms are sealed."""
from pathlib import Path
import gzip
import json
import re

from . import common as C
from . import scoring as METRIC

ROOT=C.ROOT
B=ROOT/'_docs/experiments/pallet_visibility_curriculum_20261013_v1/nonsevere_v1/statewise_joint_ot_20261014_v2/nonsevere_v4_fix/boundary_recovery_v1'
INPUT=B/'raw_ns_BI_gain10_input_v1'
OUTPUT=B/'new_gain10_paired_BI_v1'
ARMS=('RAW_YOLO_F','N3_SUBPIX_F','RAW_YOLO_BI_ONLY_FRESH_PNP','N3_SUBPIX_BI_ONLY_FRESH_PNP')
INPUT_SEAL_SHA256='785a1238cfe4bbd95c78f1d9f3d06910b03101694e2a88abba26f61b8317e2ac'
INPUT_SHA256='da7888b529939ea64dd20a78dcd4fc0493943f64cea46ca9f0a77a5f92891212'
MANIFEST_SHA256='0fb8d1b31edde99cd9eff8709bcc0759a011064c1008e9b74c38c56ed798a2e9'
pose_metric=METRIC.pose_metric


def _verify(mapping):
    for name,digest in mapping.items():
        if C.sha(ROOT/name)!=digest:raise ValueError('Frozen source/input changed '+name)


def prediction_gate(folder,expected_seal_sha256):
    """Validate all predictions before opening the separate reference manifest."""
    if not isinstance(expected_seal_sha256,str)or not re.fullmatch('[0-9a-f]{64}',expected_seal_sha256):
        raise ValueError('Explicit complete prediction seal SHA required before IO')
    folder=Path(folder)
    if C.sha(folder/'PREDICTION_SEAL.json')!=expected_seal_sha256:raise ValueError('Prediction seal changed')
    seal=C.read(folder/'PREDICTION_SEAL.json');protocol=C.read(folder/'PROTOCOL.json')
    inputs=C.read(INPUT/'INPUT_SEAL.json');ids=inputs['ordered_ids']
    if (C.sha(INPUT/'INPUT_SEAL.json')!=INPUT_SEAL_SHA256 or C.sha(INPUT/'INPUTS.json')!=INPUT_SHA256
            or inputs['inputs_sha256']!=INPUT_SHA256):
        raise ValueError('Exact ten prediction-only input rows required')
    if len(ids)!=10 or len(set(ids))!=10:raise ValueError('All ten IDs required')
    if (seal['rows']!=10 or seal['ordered_ids']!=ids or seal['arms']!=list(ARMS)
            or seal['before_scoring']is not True or seal['GT_inputs']is not False
            or seal['protocol_sha256']!=C.sha(folder/'PROTOCOL.json')
            or seal['predictions_sha256']!=C.sha(folder/'PREDICTIONS.jsonl.gz')
            or protocol['input_seal_sha256']!=INPUT_SEAL_SHA256 or protocol['inputs_sha256']!=INPUT_SHA256):
        raise ValueError('Complete ordered ten x four arms prelabel prediction closure required')
    _verify(protocol.get('source_sha256',{}));_verify(protocol.get('input_sha256',{}))
    rows=[]
    with gzip.open(folder/'PREDICTIONS.jsonl.gz','rt')as stream:
        for line in stream:
            if not line.strip():raise ValueError('Unexpected empty prediction row')
            row=json.loads(line);index=len(rows)
            if index>=10 or row['id']!=ids[index]or set(row['predictions'])!=set(ARMS):
                raise ValueError('Missing or extra case/method; reference not opened')
            for arm in ARMS:
                if not isinstance(row['predictions'][arm]['actual_pose'],dict):raise ValueError('Explicit available/unavailable pose required')
            rows.append(row)
    if len(rows)!=10:raise ValueError('Every row required before reference IO')
    return seal,protocol,rows


def summary(records):
    """An unavailable pose stays a failure in the ten-case denominator."""
    result={}
    for arm in ARMS:
        values=[r['metrics'][arm]for r in records]
        finite=[x for x in values if x['translation_cm']is not None and x['rotation_deg']is not None]
        result[arm]=dict(rows=10,pose_available=len(finite),joint_successes=sum(x['success_5cm_5deg']for x in values),
            translation_lt5=sum(x['translation_cm']is not None and x['translation_cm']<5 for x in values),
            rotation_lt5=sum(x['rotation_deg']is not None and x['rotation_deg']<5 for x in values),
            mean_translation_cm=sum(x['translation_cm']for x in finite)/10 if len(finite)==10 else None,
            max_translation_cm=max(x['translation_cm']for x in finite)if len(finite)==10 else None,
            mean_rotation_deg=sum(x['rotation_deg']for x in finite)/10 if len(finite)==10 else None,
            max_rotation_deg=max(x['rotation_deg']for x in finite)if len(finite)==10 else None,
            missing_pose_count=10-len(finite),full_denominator_retained=True)
    return result


def score(*,root_scoring_GO=False,expected_prediction_seal_sha256=None,folder=OUTPUT):
    if root_scoring_GO is not True:raise PermissionError('Completed prediction scoring GO before IO')
    seal,protocol,predictions=prediction_gate(folder,expected_prediction_seal_sha256)
    folder=Path(folder)
    if (folder/'SCORE.json').exists()or(folder/'SCORE_SEAL.json').exists():raise FileExistsError('Preserve previous scores')
    # No numeric reference is opened until all40 predictions are validated.
    if C.sha(INPUT/'SCORING_REFERENCE_MANIFEST.json')!=MANIFEST_SHA256:raise ValueError('Separate scoring binding changed')
    manifest=C.read(INPUT/'SCORING_REFERENCE_MANIFEST.json')
    if manifest['input_seal_sha256']!=INPUT_SEAL_SHA256 or manifest['inference_INPUTS_sha256']!=INPUT_SHA256:
        raise ValueError('Scorer/inference input identity mismatch')
    reference=ROOT/manifest['reference_binding']['path']
    if C.sha(reference)!=manifest['reference_binding']['sha256']:raise ValueError('Fixed reference bytes changed')
    truth=C.read(reference)['rows'];rows=[]
    for row in predictions:
        fid=row['id'];values={};statuses={}
        for arm in ARMS:
            entry=row['predictions'][arm];pose=entry['actual_pose']
            values[arm]=pose_metric(pose,truth[fid]['target']['truth'])if pose.get('available',False)else dict(
                translation_cm=None,rotation_deg=None,ADDsym_cm=None,success_5cm_5deg=False)
            statuses[arm]={key:entry[key]for key in ('accepted','fallback')if key in entry}
        rows.append(dict(id=fid,metrics=values,prediction_status=statuses))
    paired={}
    for name,before,after in (('RAW_YOLO',ARMS[0],ARMS[2]),('N3_SUBPIX',ARMS[1],ARMS[3])):
        paired[name]=dict(gains=[r['id']for r in rows if not r['metrics'][before]['success_5cm_5deg']and r['metrics'][after]['success_5cm_5deg']],
            losses=[r['id']for r in rows if r['metrics'][before]['success_5cm_5deg']and not r['metrics'][after]['success_5cm_5deg']])
    scoring=dict(schema='POSTSEAL_FIXED_GAIN10_COMMON_BI_PIPELINE_SCORE_V1',rows=rows,summary=summary(rows),paired=paired,
        fixed_posthoc_gain10=True,full49_or192_performance_evidence=False,new_default_adoption=False,
        strict_thresholds='translation_cm <5 AND rotation_deg <5',all40_predictions_kept=True,
        reference_binding=manifest['reference_binding'],prediction_seal_sha256=expected_prediction_seal_sha256,
        reference_numeric_IO_began_after_prediction_gate=True,legacy_physical_symmetry_metric_unchanged=True)
    C.write(folder/'SCORE.json',scoring)
    sources={str(p.resolve().relative_to(ROOT)):C.sha(p)for p in (Path(__file__),Path(METRIC.__file__),Path(C.__file__))}
    C.write(folder/'SCORE_SEAL.json',dict(schema='SEALED_POSTHOC_GAIN10_COMMON_BI_SCORE_V1',rows=10,arm_rows=40,arms=list(ARMS),
        score_sha256=C.sha(folder/'SCORE.json'),prediction_seal_sha256=expected_prediction_seal_sha256,
        source_sha256=sources,scoring_manifest_sha256=MANIFEST_SHA256,reference_binding=manifest['reference_binding'],
        all_rows_kept=True,posthoc_selected_not_new_full_population_evidence=True))
    return C.sha(folder/'SCORE_SEAL.json')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root-scoring-GO',action='store_true')
    parser.add_argument('--prediction-seal-sha256',required=True);args=parser.parse_args()
    print(score(root_scoring_GO=args.root_scoring_GO,expected_prediction_seal_sha256=args.prediction_seal_sha256))
