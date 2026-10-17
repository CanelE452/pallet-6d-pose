"""Package complete small pilot outputs; retain raw SHA for aliased receipts."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def run(source_root,asset_root,private_root,score_private,state_private,output):
    started=time.monotonic()
    source,asset,private,score,state,out=map(Path,(source_root,asset_root,private_root,score_private,state_private,output))
    out.mkdir(parents=True,exist_ok=True)
    aliases=sorted([(str(p),a)for p,a in [(private,'PRIVATE_AUDIT'),(score,'PRIVATE_SCORE'),(state,'PRIVATE_STATE'),
        (source,'SOURCE_ROOT'),(asset,'ASSET_ROOT'),(Path.home()/'anaconda3/envs/pallet-pose','PYTHON_ENV'),
        (Path.home()/'anaconda3/envs/pallet-yolo26','ANALYSIS_ENV'),(Path.home()/'anaconda3','CONDA_ROOT')]],key=lambda x:len(x[0]),reverse=True)
    def sanitize(x):
        if isinstance(x,str):
            for p,a in aliases:x=x.replace(p,a)
            x=x.replace('/tmp/','LOCAL_TEMP/').replace(str(Path.home())+'/','LOCAL_ACCOUNT/')
            if '/home/'in x or '/Users/'in x:raise ValueError('Unbound private path')
            return x
        if isinstance(x,dict):return {sanitize(k):sanitize(v)for k,v in x.items()}
        if isinstance(x,list):return [sanitize(v)for v in x]
        return x
    derivations=[]
    def receipt(src,dst):
        f=private/src
        if not f.is_file():return
        if f.suffix=='.jsonl':
            values=[sanitize(json.loads(l))for l in f.read_text().splitlines()if l]
            p=out/dst;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_text(''.join(json.dumps(v,ensure_ascii=False,separators=(',',':'))+'\n'for v in values))
        else:dump(out/dst,sanitize(read(f)))
        derivations.append(dict(source='PRIVATE_AUDIT/'+src,original_sha256=sha(f),public=dst,
            public_sha256=sha(out/dst),derivation='local path alias replacement and JSON serialization; scientific values unchanged'))
    for name in ('PREDICTIONS.jsonl.gz','PREDICTION_SEAL.json','PARENT_SUPPORT.jsonl','CANDIDATE_COSTS.jsonl.gz'):
        shutil.copyfile(private/'actual_rerank_v2'/name,out/name)
    for name in ('METRICS.csv','TRANSITIONS.csv','SUMMARY.json'):shutil.copyfile(private/'eval_v1'/name,out/name)
    for f in (private/'diagnostic_v1').glob('*.csv'):shutil.copyfile(f,out/f.name)
    receipt('PROMPT_MANIFEST.jsonl','PROMPT_MANIFEST.jsonl')
    receipt('supplier_v3/NON_TARGET_INPUT_MANIFEST.jsonl','NON_TARGET_INPUT_MANIFEST.jsonl')
    frozen=[json.loads(l)for l in(private/'frozen_inputs_v1/INPUTS_FROZEN.jsonl').read_text().splitlines()]
    for row in frozen:
        if row['admissible_for_censoring']:
            f=Path(row['mask_path']);dst=Path('inputs/masks')/f.name
            (out/dst).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,out/dst)
            if sha(out/dst)!=row['mask_sha256']:raise ValueError('Frozen mask copy mismatch')
            row['mask_path']=str(dst)
    (out/'INPUTS_FROZEN.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n'for r in frozen))
    derivations.append(dict(source='PRIVATE_AUDIT/frozen_inputs_v1/INPUTS_FROZEN.jsonl',
        original_sha256=sha(private/'frozen_inputs_v1/INPUTS_FROZEN.jsonl'),public='INPUTS_FROZEN.jsonl',
        public_sha256=sha(out/'INPUTS_FROZEN.jsonl'),derivation='mask_path changed to exact public relative mask copy only'))
    for f in (private/'supplier_v3').glob('*__mask_*.png'):
        dst=out/'inputs/alternatives'/f.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dst)
    fixed_receipts=['START_BEFORE_FETCH.json','BASE_AND_INSTRUCTION.json','PROTOCOL_LOCK.json','MASK_REVIEW.json',
        'REQUIRED_DOCUMENT_READS.json','SUPPLIER_MODEL_CONFIG.json','SUPPLIER_ATTEMPT_HISTORY.json',
        'KERNEL_API_TEST_RECEIPT.json','KERNEL_API_TEST_RECEIPT_V2.json','FINAL_SOURCE_BINDING_RECHECK.json',
        'INDEPENDENT_CHECK_V1.json','INDEPENDENT_PUBLICATION_CHECK_V2.json','INDEPENDENT_PUBLICATION_CHECK_V1_FAILURE.json',
        'CONTROL_COMPARISON_RECEIPT.json','CONTROL_COMPARISON_ATTEMPT_V1_FAILURE.json',
        'PUBLICATION_ATTEMPT01.json','PUBLICATION_VALIDATION_ATTEMPT01.json','FINAL_VISUAL_REVIEW.json']
    for name in fixed_receipts:receipt(name,'evidence/'+name)
    for folder in ('supplier_v1','supplier_v2','supplier_v3'):
        for f in (private/folder).glob('*.json'):receipt(str(f.relative_to(private)),'evidence/'+folder+'/'+f.name)
    for folder in ('baseline_identity_v1','baseline_identity_v2','baseline_identity_v3','actual_rerank_v1','actual_rerank_v2'):
        receipt(folder+'/PREDICTION_SEAL.json','evidence/'+folder+'/PREDICTION_SEAL.json')
    for folder in ('gallery_attempt01_uint8_contract_preserved','gallery_attempt02_font_preserved'):
        for name in ('ATTEMPT_STATUS.json','GALLERY_RECEIPT.json'):
            receipt(folder+'/'+name,'evidence/'+folder+'/'+name)
    for a,b in [('neutral_v1/EXPORT_RECEIPT.json','EXPORT_RECEIPT.json'),
        ('neutral_v1/NEUTRAL_PACKAGE.json','NEUTRAL_PACKAGE.json'),
        ('frozen_inputs_v1/INPUT_SEAL.json','INPUT_SEAL.json'),
        ('eval_v1/EVALUATION_RECEIPT.json','EVALUATION_RECEIPT.json'),
        ('diagnostic_v1/DIAGNOSTIC_RECEIPT.json','DIAGNOSTIC_RECEIPT.json')]:receipt(a,'evidence/'+b)
    actual_source=private/'supplier_v3/SUPPLIER_SOURCE_AT_INFERENCE.py'
    if actual_source.exists():
        if '/home/'in actual_source.read_text()or '/tmp/'in actual_source.read_text():
            dump(out/'evidence/SUPPLIER_SOURCE_AT_INFERENCE_BINDING.json',dict(
                original_sha256=sha(actual_source),original_preserved='PRIVATE_AUDIT/supplier_v3/SUPPLIER_SOURCE_AT_INFERENCE.py',
                public_raw_copy=False,reason='Private original retained; final source and version differences documented in INPUT_SUPPLIER_AUDIT.md'))
        else:shutil.copyfile(actual_source,out/'evidence/SUPPLIER_SOURCE_AT_INFERENCE.py')
    # Every candidate, both directions and all three paths are included.
    with gzip.open(out/'CANDIDATE_COSTS.jsonl.gz','rt')as f:costs=[json.loads(l)for l in f]
    breakdown=[]
    for c in costs:
        for variant,score in c['scores'].items():
            for d in c['directions']:
                new=d['original_weighted_cost']if variant=='BASELINE_E'else d['support_weighted_cost']if variant=='SUPPORT_AWARE_RERANK'else d['mass_control_weighted_cost']
                breakdown.append(dict(id=c['id'],stage=c['stage'],family=c['family'],candidate_id=c['candidate_id'],variant=variant,
                    score=score,direction=d['direction'],old_direction_cost=d['original_weighted_cost'],new_direction_cost=new,
                    original_count=d['original_count'],retained_count=d['original_count']if variant=='BASELINE_E'else d['retained_count'],
                    old_mass=d['original_mass'],support_retained_mass=d['retained_mass'],support_retained_ratio=d['retained_mass_ratio'],
                    old_full_BI=c['original_weighted_components']['full_BI'],old_partial=c['original_weighted_components']['partial_normal'],
                    old_prior=c['original_weighted_components']['prior'],original_normalizer_preserved=True))
    with (out/'COST_BREAKDOWN.csv').open('w',newline='')as f:
        w=csv.DictWriter(f,fieldnames=list(breakdown[0]));w.writeheader();w.writerows(breakdown)
    # Alias existing supplier audit prose without touching numerical input bytes.
    audit=out/'INPUT_SUPPLIER_AUDIT.md'
    if audit.exists():audit.write_text(sanitize(audit.read_text()))
    binding=dict(schema='NON_TARGET_FUNCTIONAL_PILOT_INPUT_BINDINGS_V1',
        execution_base=read(private/'BASE_AND_INSTRUCTION.json'),
        original_score_result_commit='c56a4e92027a8e56f6704773f24d7c3da78082f1',
        original_state_result_commit='407de865445e4987614f50f893d2ccea78d507b6',
        export=read(private/'neutral_v1/EXPORT_RECEIPT.json'),
        neutral_package=read(private/'neutral_v1/NEUTRAL_PACKAGE.json'),
        protocol_lock=read(private/'PROTOCOL_LOCK.json'),
        fixed_input_seal=read(private/'frozen_inputs_v1/INPUT_SEAL.json'),
        source_sha256={str(f.relative_to(source)):sha(f)for f in sorted((source/'scripts/research/pallet_non_target_outline_pilot_v1').glob('*.py'))},
        target_mask_modified=False,weights_modified=False,reference_modified=False,defaults_or_papers_modified=False)
    dump(out/'INPUT_BINDINGS.json',sanitize(binding))
    measured=[]
    for folder in ('baseline_identity_v1','baseline_identity_v2','baseline_identity_v3','actual_rerank_v1','actual_rerank_v2'):
        o=read(private/folder/'PREDICTION_SEAL.json');measured.append(dict(stage=folder,wall_seconds=o['elapsed_seconds'],process_CPU_seconds=o['process_cpu_seconds']))
    for f in ('neutral_v1/EXPORT_RECEIPT.json','eval_v1/EVALUATION_RECEIPT.json','INDEPENDENT_CHECK_V1.json','INDEPENDENT_PUBLICATION_CHECK_V2.json'):
        o=read(private/f)
        measured.append(dict(stage=f,wall_seconds=o.get('elapsed_seconds'),process_CPU_seconds=o.get('process_cpu_seconds')))
    for f,label in ((out/'gallery/GALLERY_RECEIPT.json','final gallery'),
                    (private/'gallery_attempt02_font_preserved/GALLERY_RECEIPT.json','preserved gallery font attempt'),
                    (private/'CONTROL_COMPARISON_RECEIPT.json','same-six saved controls')):
        if f.exists():
            o=read(f);measured.append(dict(stage=label,wall_seconds=o.get('elapsed_seconds'),process_CPU_seconds=o.get('process_cpu_seconds')))
    for f in private.glob('*GALLERY*COST*.json'):
        receipt(f.name,'evidence/'+f.name)
    cost=dict(schema='NON_TARGET_PILOT_ACTUAL_COST_V1',measured_analysis_stages=measured,
        measured_analysis_wall_sum_seconds=sum(r['wall_seconds']or 0 for r in measured),
        measured_process_CPU_sum_seconds=sum(r['process_CPU_seconds']or 0 for r in measured),
        supplier=read(private/'supplier_v3/SUPPLIER_RUN.json'),
        supplier_attempt_history=read(private/'SUPPLIER_ATTEMPT_HISTORY.json'),
        new_image_encoder_calls=3,new_prompt_decode_calls=3,new_mask_decoder_forwards=3,new_prompts=3,
        supplier_recovery_new_NN=0,new_training=0,new_seed=0,new_candidates=0,new_PnP=0,new_pose_optimization=0,
        weights_threshold_or_prompt_search=0,AUTO_model_calls=0,AUTO238_executed=False,
        GPU_concurrency_max=1,CPU_heavy_workers_max=2,GPU_related_run_cap_seconds=1800,CPU_analysis_cap_seconds=7200,
        human_review_performed=False,human_review_seconds=0,model_visual_review_time_separately_measured=False,
        dataset_or_model_downloads=0,external_RGB_transmission=0,
        whole_RGB_to_pose_latency_measured=False,per_frame_encoder_decoder_seconds=None,
        per_frame_timing_missing_reason='Actual forward completed before packaging failed; no NN repeated to fill times.',
        final_gallery_and_publication_QA_cost_location='gallery/GALLERY_RECEIPT.json and VALIDATION.json',
        full_task_wall_seconds_at_packaging=time.time()-read(private/'START_BEFORE_FETCH.json')['start_epoch'],
        notes='Baseline v1/v2/v3 audit instrumentation and actual v1/v2 identical scientific outputs retained. Sum is measured stages only, not end-to-end FPS.')
    dump(out/'COST.json',sanitize(cost));dump(out/'PUBLIC_DERIVATIONS.json',dict(records=derivations,
        all_original_private_receipts_preserved=True,aliases_only_no_numerical_changes=True,
        copied_prediction_artifacts_match_original_seal=True,packaging_elapsed_seconds=time.monotonic()-started))
    return dict(candidate_cost_rows=len(costs),cost_breakdown_rows=len(breakdown),copied_receipts=len(derivations))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('source-root','asset-root','private-root','score-private','state-private','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();print(run(a.source_root,a.asset_root,a.private_root,a.score_private,a.state_private,a.output))
