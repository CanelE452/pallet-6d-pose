"""V3 ROOT-only exact GO sealer for the isolated PRIMARY372 filter diagnostics.

No RGB, annotation or reference-coordinate cache is opened. The one consumed
reference cache receives its frozen SHA from the prepared metadata authority;
the diagnostic score stage must check its bytes before parsing. Old proof
closure follows only the29 direct hash dependencies, never INITIAL_GATES'
nested score pointers. Checkpoints may be SHA-hashed, never deserialized.
Import performs no IO; this source must be reviewed before --seal is invoked.
Workflow ROOT GO uses dynamic runtime producer identity and exact static V3 source
SHA from OUTER_WORKFLOW_RESUME_AUTHORITY_V3.json; old failure records are preserved.
ROOT review/GO/failure sidecars use the _V3 suffix; core GO uses _V3 while tags/results stay canonical.
"""
from pathlib import Path
import argparse
import ast
import datetime
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parents[1];BASE=OUT.parent
WOOD=BASE/'dope-wood640-eval80-20261027-v1';TARGET=WOOD/'combined372'
MAIN=BASE/'dope-filter-ablation-per-type-20261027-v1'
B1=BASE/'dope-thesis-analysis-batch1-v1';FLIP=BASE/'dope-loo-flip-per-type-20261026-v1'
NOFLIP=BASE/'dope-loo-no-flip-per-type-20261026-v1'
CORE=WOOD/'code/merged_filter_diagnostics.py';TEST=WOOD/'code/test_merged_filter_diagnostics.py'
SOURCE=Path(__file__).resolve()
NORMALIZER=OUT/'code/prepare_ratio_evaluation_v2.py'
NORMALIZER_SHA='11278e1b623c251c98276305390de32cb3cd030cd740c25c52aeeb19da959322'
ADAPTER=OUT/'code/run_filter_diagnostics_with_metadata_v3.py'
ADAPTER_SHA='54f8cae3b31e8eea2e6060732f4d5bddf2d3803cbefe8105a5646bb2f1e15a61'
CORE_SHA='e83f23bfe3f1e1429654ed7e2036993abd68849498ff0cb0b30a1f9693f6832a'
TEST_SHA='a159c64b3ab0371fe85b6ffbc794fc03d71311350000016b6412823bbe18e2b0'
REFERENCE_SHA='7d1315bbfa15a117463962cc8a55ecca7da5570d9a76e9d1789dfd0a0ab19e0c'
REFERENCE_CACHES={TARGET/'MERGED_REFERENCES.json',FLIP/'SCORES.json',NOFLIP/'SCORES.json',WOOD/'cache/REFERENCE_POINTS_PRIVATE.json'}
OLD_MAPS=((B1/'preflight/T3_FILTER_EQUIVALENCE.json',('sources','gates','prediction_caches')),
    (MAIN/'preflight/FILTER_PARITY.json',('input_sha256',)),
    (MAIN/'cache/DIAGNOSTIC_SELECTIONS_R0_CANONICAL.json',('input_sha256',)),
    (MAIN/'cache/DIAGNOSTIC_SELECTIONS_R0_NO_CONFIDENCE.json',('input_sha256',)))


def require(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(path):
    p=Path(path);require(p not in REFERENCE_CACHES,'Sealer must not open reference-cache bytes')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


class Seal:
    def __init__(self,tag):
        self.tag=tag;self.used={};self.inputs={}
        self.root_review=WOOD/'preflight/FILTER_DIAGNOSTIC_ROOT_REVIEW_V3.json'
        self.review=self.read(self.root_review)
        self.source_review={str(CORE):CORE_SHA,str(TEST):TEST_SHA,str(SOURCE):sha(SOURCE),
            str(NORMALIZER):NORMALIZER_SHA,str(ADAPTER):ADAPTER_SHA}
        require(self.review['passed'] is True and self.review['tag']==tag and self.review['source_sha256']==self.source_review,
            'ROOT core/test/combined-workflow source review absent or stale')
        self.hash(CORE,CORE_SHA);self.hash(TEST,TEST_SHA)
        # Only simple, exact path constants and DEPS are evaluated from the
        # already SHA-frozen source. No imports, functions or runtime are run.
        names={'OUT','BASE','TARGET','MAIN','B1','FLIP','NOFLIP','REPO','GPU_LOCK','SOURCE','MODEL','ADAPTER',
            'F_SOURCE','D_SOURCE','T_SOURCE','A_SOURCE','DEPS'}
        nodes=[]
        for n in ast.parse(CORE.read_text()).body:
            if isinstance(n,ast.Assign) and all(isinstance(t,ast.Name) for t in n.targets) and all(t.id in names for t in n.targets):nodes.append(n)
        space={'Path':Path,'__file__':str(CORE)}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(CORE),'exec'),space)
        self.deps=space['DEPS'];require(len(self.deps)==12,'Frozen original dependency count differs')
        self.deps={**self.deps,str(NORMALIZER):NORMALIZER_SHA,str(ADAPTER):ADAPTER_SHA}
        require(len(self.deps)==14,'Exact metadata-only adapter/normalizer extension required')
        for p,digest in self.deps.items():self.hash(p,digest)

    def hash(self,path,expected=None):
        raw=Path(path);p=raw.resolve()
        require(str(p)==str(raw) and p.is_file() and not raw.is_symlink(),'Missing/noncanonical input: '+str(path))
        digest=sha(p)
        require(expected is None or digest==expected,'Expected metadata/source/checkpoint SHA differs: '+str(p))
        require(str(p) not in self.used or self.used[str(p)]==digest,'Input changed during sealing')
        self.used[str(p)]=digest;return digest

    def read(self,path):
        digest=self.hash(path);value=json.loads(Path(path).read_text());self.hash(path,digest);return value

    def add(self,path,category,expected=None):
        require(category in ('metadata','protocol','checkpoint_receipt','source_code','cached_predictions','dev_two_view_predictions','cached_masks','checkpoint'),
            'Only explicit nonraw diagnostic inputs may be hashed before GO')
        p=Path(path);digest=self.hash(p,expected)
        item=dict(category=category,sha256=digest,bytes=p.stat().st_size)
        require(str(p) not in self.inputs or self.inputs[str(p)]['sha256']==digest,'Conflicting exact input')
        self.inputs[str(p)]=item

    def unchanged(self):
        for p,digest in self.used.items():require(sha(p)==digest,'Sealing input/source changed: '+p)


def old_proof_closure(s):
    closure_path=WOOD/'preflight/FILTER_DIAGNOSTIC_OLD_PROOF_CLOSURE.json'
    closure=s.read(closure_path)
    require(s.hash(closure_path)==s.review['old_proof_closure_sha256'] and closure['unique_direct_paths']==29
        and closure['conflicting_SHA_declarations']==0 and closure['direct_reference_cache_intersection']==[],
        'ROOT must bind the inspected direct-only29 old-proof closure')
    actual={}
    for path,keys in OLD_MAPS:
        before=s.hash(path);raw=Path(path).read_text();s.hash(path,before)
        for key in keys:
            hits=list(re.finditer('"'+key+'"\\s*:\\s*(\\{)',raw))
            require(len(hits)==1,'Ambiguous old proof metadata map')
            mapping,_=json.JSONDecoder().raw_decode(raw,hits[0].start(1))
            require(isinstance(mapping,dict) and all(isinstance(p,str) and isinstance(h,str) and len(h)==64 for p,h in mapping.items()),'Not a direct path-SHA map')
            for p,digest in mapping.items():
                require(Path(p) not in REFERENCE_CACHES and (p not in actual or actual[p]==digest),'Reference pointer/conflicting old proof SHA')
                actual[p]=digest
        s.add(path,'cached_masks' if '/cache/' in str(path) else 'metadata')
    require(len(actual)==29 and set(actual)==set(closure['input_files']),'Old direct proof closure keyset changed')
    for p,digest in actual.items():
        item=closure['input_files'][p]
        require(item['sha256']==digest,'Old closure declaration changed')
        s.add(p,item['category'],digest)
    s.add(closure_path,'metadata')
    # This intentionally does not parse INITIAL_GATES.evidence or recursively
    # follow metadata's nested score references. It is not needed by verify_old.


def prerequisites(s):
    campaign=s.read(MAIN/'CAMPAIGN_COMPLETION.json');audit=s.read(MAIN/'preflight/FINAL_AUDIT.json')
    integrity=s.read(MAIN/'preflight/FINAL_INTEGRITY.json');complete=s.read(MAIN/'evaluation/final_v1/final/COMPLETE.json')
    require(campaign['finished'] is True and campaign['all_arms_complete'] is True and campaign['frozen_inputs_unchanged'] is True
        and audit['passed'] is True and audit['complete'] is True and audit['tag']=='final_v1'
        and integrity['passed'] is True and integrity['all_original_experiment_and_checkpoint_bytes_unchanged'] is True
        and integrity['all_allowed_input_bytes_unchanged'] is True and complete['complete'] is True and complete['main_groups']==51,
        'MAIN must complete and pass its immutable audit/integrity before GO')
    tests=s.read(WOOD/'preflight/FILTER_DIAGNOSTIC_PURE_TESTS.json')
    require(tests['passed'] is True and tests['returncode']==0 and tests['tests_expected']==10
        and tests['source_sha256']=={str(CORE):CORE_SHA,str(TEST):TEST_SHA}
        and tests['real_RGB_reads']==tests['real_annotation_reads']==tests['NN_forwards']==tests['training_updates']==0,
        'Actual ten pure fixtures absent or stale')
    s.hash(tests['stdout_log'],tests['stdout_sha256'])
    sem=s.read(WOOD/'preflight/SEMANTIC_CONTRACT_REVIEW.json')
    require(sem['passed'] is True and sem['same_index_0_7_compatible'] is True
        and sem['GT_optimized_permutation_used'] is False and sem['coordinates_or_indices_modified'] is False,
        'ROOT same-index2D review required')
    old_proof_closure(s)
    return audit


def population(s):
    merged=s.read(TARGET/'MERGED_MANIFEST.json');core=s.read(FLIP/'MANIFEST.json')['records']
    extra=s.read(WOOD/'MANIFEST.json')['records'];geometry=s.read(TARGET/'MERGED_GEOMETRY_INPUTS.json')
    preparation=s.read(TARGET/'PREPARATION_QA.json')
    require(s.hash(TARGET/'PREPARATION_QA.json')==s.review['preparation_qa_sha256'] and preparation['passed'] is True
        and preparation['manifest_sha256']==s.hash(TARGET/'MERGED_MANIFEST.json')
        and preparation['geometry_sha256']==s.hash(TARGET/'MERGED_GEOMETRY_INPUTS.json')
        and preparation['references_sha256']==REFERENCE_SHA,'Prepared merged metadata/reference authority changed')
    require(len(core)==292 and len(extra)==80 and len(merged['records'])==372
        and len({r['id'] for r in core+extra})==372 and {r['id'] for r in core+extra}=={r['id'] for r in merged['records']},'Exact disjoint292+80 required')
    require(geometry['manifest_sha256']==s.hash(TARGET/'MERGED_MANIFEST.json') and len(geometry['records'])==372,'Merged geometry binding differs')
    sem=s.read(WOOD/'preflight/SEMANTIC_CONTRACT_REVIEW.json')
    require(sem['manifest_sha256']==s.hash(WOOD/'MANIFEST.json'),'Same-index review is for another80')
    raw80=s.read(WOOD/'ALLOWLIST.json');before80=s.read(WOOD/'preflight/SNAPSHOT_BEFORE.json')
    require(raw80['authorization_sha256']==s.hash(WOOD/'AUTHORIZATION.json')
        and before80['allowlist_sha256']==s.hash(WOOD/'ALLOWLIST.json')
        and len(raw80['records'])==80 and len(raw80['input_files'])==len(before80['files'])==160
        and set(raw80['input_files'])==set(before80['files']),'Exact80 authorization/snapshot binding differs')
    # Both immutable sources were SHA-checked before this no-IO module import.
    from prepare_ratio_evaluation_v2 import normalized_authorized_rgb
    allow=raw80['input_files'];approved={}
    for frame in extra:
        p=frame['image_path'];item=normalized_authorized_rgb(allow[p],before80['files'][p],frame)
        require(item['category']=='authorized_WOOD80_RGB' and item['sha256']==frame['image_sha256']
            and item['frame_id']==frame['source_frame_id'] and (frame['width'],frame['height'])==(640,480)
            and frame['object_type']=='wood_small_80x59x14','Exact WOOD80 authorization mismatch')
        require(str(Path(p).resolve())==p and not Path(p).is_symlink(),'Noncanonical approved RGB path')
        s.inputs[p]=dict(item)  # Opaque declaration only: never open RGB bytes.
        approved[frame['id']]={k:frame[k] for k in ('image_path','image_sha256','width','height','object_type')}
    require(len(approved)==80,'Duplicate exact80 ID')
    reference=TARGET/'MERGED_REFERENCES.json'
    require(reference.is_file() and not reference.is_symlink(),'Missing prepared scoring-only reference cache')
    s.inputs[str(reference)]=dict(category='cached_scores',sha256=REFERENCE_SHA,bytes=reference.stat().st_size,
        permitted_stage='score_only',hash_bytes_deferred_until_score=True)
    for p in (TARGET/'MERGED_MANIFEST.json',TARGET/'MERGED_GEOMETRY_INPUTS.json',TARGET/'PREPARATION_QA.json',
        FLIP/'MANIFEST.json',WOOD/'MANIFEST.json',WOOD/'ALLOWLIST.json',WOOD/'AUTHORIZATION.json',
        WOOD/'preflight/SNAPSHOT_BEFORE.json',WOOD/'preflight/SEMANTIC_CONTRACT_REVIEW.json'):
        s.add(p,'metadata')
    for folder in (FLIP,NOFLIP):
        for kind in ('PLASTIC','WOOD','GREEN'):s.add(folder/kind/'PROTOCOL.json','protocol')
    return extra,approved


def first80(s,extra,audit):
    catalog=s.read(WOOD/'MODEL_CATALOG.json');done=s.read(WOOD/'inference/COMPLETE.json');scored=s.read(WOOD/'results/COMPLETE.json')
    require(catalog['complete'] is True and len(catalog['groups'])==17 and done['complete'] is True and scored['complete'] is True
        and scored['groups']==17 and done['groups']==sum(g['available'] for g in catalog['groups'])
        and done['catalog_sha256']==scored['catalog_sha256']==s.hash(WOOD/'MODEL_CATALOG.json'),'First80 must be fully inferred and scored before GO')
    for p,digest in catalog['source_code_sha256'].items():require(s.deps[p]==digest,'First80 canonical source differs')
    for model,condition,round_idx in (('R0_before_selftraining','R0',0),('WOOD_LOO_flip_R1','full',1),('WOOD_LOO_only_R1','no_flip',1)):
        groups=[g for g in catalog['groups'] if (g['condition'],g['round'])==(condition,round_idx)]
        require(len(groups)==1 and groups[0]['available'] is True,'Required original teacher unavailable')
        g=groups[0];cp=Path(g['checkpoint_path'])
        require(audit['input_sha256'].get(str(cp))==g['checkpoint_sha256'],'Teacher checkpoint not bound by immutable MAIN audit')
        s.add(cp,'checkpoint',g['checkpoint_sha256'])
        old_path=B1/'cache'/f'T3_PREDICTIONS_{model}.json';old=s.read(old_path)
        require(old['contract']['model']==model and old['contract']['checkpoint']['path']==str(cp)
            and old['contract']['checkpoint']['sha256']==g['checkpoint_sha256'],'Original T3 and first80 teacher mismatch')
        p=WOOD/'inference'/(g['group']+'.json');s.add(p,'cached_predictions',done['predictions_sha256'][str(p)])
        cached=s.read(p)
        require(cached['complete'] is True and cached['catalog_sha256']==s.hash(WOOD/'MODEL_CATALOG.json')
            and cached['runtime']==dict(batch=1,dtype='float32',decoder_threshold=.3,autocast=False,TF32=False)
            and [r['id'] for r in cached['records']]==[f['id'] for f in extra],'First80 teacher cache contract differs')
        for r,f in zip(cached['records'],extra):require(r['image_sha256']==f['image_sha256'] and r['checkpoint_sha256']==g['checkpoint_sha256']
            and len(r['points'])==len(r['valid'])==len(r['confidence'])==9,'First80 cache ID/image/CP/9slots differ')
        s.add(old_path,'dev_two_view_predictions')
    for p in (WOOD/'MODEL_CATALOG.json',WOOD/'inference/COMPLETE.json',WOOD/'results/COMPLETE.json'):s.add(p,'metadata')


def seal(tag):
    go=WOOD/'preflight/FILTER_DIAGNOSTIC_GO_V3.json';receipt=WOOD/'preflight/FILTER_DIAGNOSTIC_SEAL_COMPLETE_V3.json'
    require(not go.exists() and not receipt.exists(),'Preserve existing diagnostic GO/seal artifact')
    s=Seal(tag);audit=prerequisites(s);extra,approved=population(s);first80(s,extra,audit)
    for p,digest in s.deps.items():s.add(p,'source_code',digest)
    s.add(MAIN/'diagnostics/FILTER_DIAGNOSTICS_WITH_NO_CONFIDENCE.json','cached_masks')
    s.add(B1/'tables/T3_FILTER_STAGES.json','cached_masks')
    old=s.read(MAIN/'diagnostics/FILTER_DIAGNOSTICS_WITH_NO_CONFIDENCE.json');stages=s.read(B1/'tables/T3_FILTER_STAGES.json')
    require(old['complete'] is True and old['no_confidence_pending'] is False and len(old['rows'])==28
        and stages['complete'] is True and len(stages['rows'])==132,'Original28/132 reference aggregate catalogue differs')
    s.unchanged()
    value=dict(schema='merged_filter_diagnostics_go_v1',passed=True,utc=utc(),tag=tag,
        infer_authorized=True,select_authorized=True,score_authorized=True,source_sha256={str(CORE):CORE_SHA,**s.deps},
        input_files=s.inputs,approved_wood80_frames=approved,wood80_manifest_sha256=s.used[str(WOOD/'MANIFEST.json')],
        wood80_authorization_sha256=s.used[str(WOOD/'AUTHORIZATION.json')],ROOT_review_sha256=s.used[str(s.root_review)],
        sealer_sha256=sha(SOURCE),pure_test_receipt_sha256=s.used[str(WOOD/'preflight/FILTER_DIAGNOSTIC_PURE_TESTS.json')],
        initial_gate_recursion='direct29 only; nested old reference pointers not followed',
        scoring_only_reference_sha_authority='frozen PREPARATION_QA; actual reference bytes deferred to score stage',
        raw_RGB_bytes_opened=0,raw_annotation_bytes_opened=0,reference_cache_bytes_opened=0,
        neural_forwards=0,PnP_fits=0,checkpoint_deserializations=0)
    with go.open('x') as f:f.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    s.unchanged()
    with receipt.open('x') as f:f.write(json.dumps(dict(complete=True,passed=True,utc=utc(),GO_path=str(go),GO_sha256=sha(go),
        source_sha256=s.source_review,input_sha256=s.used,exact_input_paths=len(s.inputs),direct_old_dependency_paths=29,
        approved_RGB_paths=80,raw_RGB_or_annotation_bytes_opened=0,reference_cache_bytes_opened=0,NN_forwards=0,PnP_fits=0),ensure_ascii=False,indent=2)+'\n')
    print('FILTER_DIAGNOSTIC_GO_SEALED',tag,len(s.inputs),flush=True)


# Immutable ROOT authority records the old technical failures; it is not a waiver
# of scientific/core checks. Producer SHA placeholders live in ROOT metadata only.
PRIOR_RESUME_AUTHORITY=OUT/'preflight/OUTER_WORKFLOW_RESUME_AUTHORITY_V2.json'
PRIOR_RESUME_AUTHORITY_SHA='f0527f06976b17f761e6340f3ebd641eb4ae90b44d2c961fa0d4a18d409c6bc8'
RESUME_AUTHORITY=OUT/'preflight/OUTER_WORKFLOW_RESUME_AUTHORITY_V3.json'
OLD_FAILURES={OUT/'FINISH_EXPERIMENTS_FAILURE.json',OUT/'FILTER_DIAGNOSTIC_WORKFLOW_FAILURE.json',
    OUT/'preflight/FILTER_STATISTICS_QUEUE_FAILURE.json',OUT/'preflight/MAIN_MODEL_COMPARISONS_QUEUE_FAILURE.json',
    OUT/'FILTER_DIAGNOSTIC_WORKFLOW_FAILURE_V2.json',OUT/'preflight/FILTER_STATISTICS_QUEUE_FAILURE_V2.json'}


def resume_binding(go,producer_source,allowed_failures):
    before=sha(RESUME_AUTHORITY);authority=json.loads(RESUME_AUTHORITY.read_text())
    require(sha(RESUME_AUTHORITY)==before and go['resume_authority_sha256']==before,
        'ROOT V3 resume authority absent/stale')
    require(authority['schema']=='outer_workflow_resume_authority_v3' and authority['passed'] is True
        and authority['technical_repair_only'] is True and authority['scientific_core_or_target_tags_changed'] is False,
        'Technical-only ROOT resume authority required')
    require(sha(PRIOR_RESUME_AUTHORITY)==PRIOR_RESUME_AUTHORITY_SHA
        and authority['prior_resume_authority_sha256']==PRIOR_RESUME_AUTHORITY_SHA,
        'Preserved V2 ROOT resume authority changed or unbound')
    old=authority['old_failures'];require(set(old)=={str(p) for p in OLD_FAILURES},'Exact six preserved failure records required')
    for p in OLD_FAILURES:require(sha(p)==old[str(p)],'Preserved old failure SHA changed: '+str(p))
    item=authority['approved_rebindings'][str(SOURCE)];producer=go['producer']
    require(item['source_sha256']==sha(SOURCE)
        and set(item['allowed_old_failures'])=={str(p) for p in allowed_failures},'This V3 replacement is not authorized')
    require(set(producer)=={'pid','create_time','source_path','source_sha256'}
        and type(producer['pid']) is int and producer['pid']>0
        and isinstance(producer['create_time'],(int,float)) and 0<producer['create_time']<1e12
        and producer['source_path']==str(producer_source)
        and isinstance(producer['source_sha256'],str) and len(producer['source_sha256'])==64,
        'ROOT runtime producer identity incomplete')
    static={k:producer[k] for k in ('source_path','source_sha256')}
    require(item['producer']==static and sha(producer_source)==producer['source_sha256'],
        'Approved static V3 producer source differs')
    return before

def workflow_gate(tag):
    p=OUT/'preflight/FILTER_DIAGNOSTIC_WORKFLOW_GO_V3.json';go=json.loads(p.read_text())
    expected={str(CORE):CORE_SHA,str(TEST):TEST_SHA,str(SOURCE):sha(SOURCE),
        str(NORMALIZER):NORMALIZER_SHA,str(ADAPTER):ADAPTER_SHA}
    review=WOOD/'preflight/FILTER_DIAGNOSTIC_ROOT_REVIEW_V3.json'
    require(go['passed'] is True and go['workflow_authorized'] is True and go['tag']==tag
        and go['source_sha256']==expected and go['ROOT_review_sha256']==sha(review),'Explicit workflow ROOT GO absent/stale')
    parent=OUT/'code/finish_merged_experiments_v2.py'
    resume_binding(go,parent,{OUT/'FINISH_EXPERIMENTS_FAILURE.json',OUT/'FILTER_DIAGNOSTIC_WORKFLOW_FAILURE.json',
        OUT/'FILTER_DIAGNOSTIC_WORKFLOW_FAILURE_V2.json'})
    for path,digest in expected.items():require(sha(path)==digest,'Reviewed diagnostic workflow source changed')
    return go,sha(p)


def wait_for_all15(tag):
    complete=OUT/'FINISH_EXPERIMENTS_COMPLETE.json'
    def ready():
        workflow_gate(tag)  # Exact preserved-failure rebinding is checked on every readiness check.
        for name in ('FINISH_EXPERIMENTS_FAILURE_V2.json','MERGED_WORKFLOW_FAILURE.json','ABLATION_FINALIZE_FAILURE.json'):
            require(not (OUT/name).exists(),'Existing experiment producer failure requires ROOT repair')
        if not complete.is_file():return False
        try:value=json.loads(complete.read_text())
        except (json.JSONDecodeError,UnicodeError,OSError):return False
        require(value['complete'] is True and value['count_ratio_trials_completed']==15
            and value['primary_results_scored'] is True and value['future_R3_probe_completed'] is True,
            'All15 training/inference/teacher-probe completion required before extra GPU views')
        require(value['workflow_sha256']==workflow_gate(tag)[0]['producer']['source_sha256'],
            'All15 completion belongs to another producer source')
        return True
    while not ready():
        go,_=workflow_gate(tag);producer=go['producer']
        import psutil
        try:
            process=psutil.Process(producer['pid'])
            live=(abs(process.create_time()-producer['create_time'])<.01 and producer['source_path'] in process.cmdline()
                and process.status()!=psutil.STATUS_ZOMBIE)
            if not live:
                if ready():return
                raise RuntimeError('All15 producer identity/state changed before completion')
        except psutil.NoSuchProcess:
            if ready():return
            raise RuntimeError('All15 producer exited without completed receipt')
        time.sleep(30)


def run(tag):
    workflow_gate(tag);require(not (OUT/'FILTER_DIAGNOSTIC_WORKFLOW_COMPLETE.json').exists()
        and not (OUT/'FILTER_DIAGNOSTIC_WORKFLOW_FAILURE_V3.json').exists(),'Preserve completed workflow or V3 failure')
    with (OUT/'logs/FILTER_DIAGNOSTIC_WORKFLOW_V3.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            wait_for_all15(tag);workflow_gate(tag);seal(tag)
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
            stages=[];adapter_receipts={}
            for stage in ('infer','select','score'):
                workflow_gate(tag);log=OUT/'logs'/('filter372_'+tag+'_'+stage+'_V3.stdout.log')
                require(not log.exists(),'Preserve partial/existing stage log; no automatic retry')
                command=[sys.executable,'-B',str(ADAPTER),stage,'--tag',tag]
                begin=time.monotonic()
                with log.open('xb',buffering=0) as stream:result=subprocess.run(command,cwd=WOOD,env=env,stdout=stream,stderr=subprocess.STDOUT)
                stages.append(dict(stage=stage,command=command,returncode=result.returncode,elapsed_seconds=time.monotonic()-begin,stdout_log=str(log),stdout_sha256=sha(log)))
                require(result.returncode==0,'Diagnostic stage failed: '+stage)
                folder=TARGET/'filter_diagnostics'/tag
                proof_path=folder/('METADATA_ADAPTER_'+stage+'_COMPLETE.json');proof=json.loads(proof_path.read_text())
                original=folder/stage/('EQUIVALENCE.json' if stage=='select' else 'COMPLETE.json')
                preflight=folder/('METADATA_ADAPTER_'+stage+'.json')
                require(proof['complete'] is True and proof['passed'] is True and proof['stage']==stage and proof['tag']==tag
                    and proof['scientific_kernel_path']==str(CORE) and proof['scientific_kernel_sha256']==CORE_SHA
                    and proof['metadata_adapter_path']==str(ADAPTER) and proof['metadata_adapter_sha256']==ADAPTER_SHA
                    and proof['pure_normalizer_path']==str(NORMALIZER) and proof['pure_normalizer_sha256']==NORMALIZER_SHA
                    and proof['original_function_objects_unchanged'] is True and proof['original_SOURCE_unchanged'] is True
                    and proof['metadata_only_override'] is True and proof['bound_go_path_guard_only'] is True
                    and proof['original_files_modified'] is False
                    and proof['scientific_functions_replaced'] is False and proof['original_stage_receipt_path']==str(original)
                    and proof['original_stage_receipt_sha256']==sha(original) and proof['metadata_preflight_path']==str(preflight)
                    and proof['metadata_preflight_sha256']==sha(preflight)
                    and proof['source_sha256']==json.loads((WOOD/'preflight/FILTER_DIAGNOSTIC_GO_V3.json').read_text())['source_sha256'],
                    'Actual metadata-only adapter/kernel execution receipt differs')
                adapter_receipts[str(proof_path)]=sha(proof_path)
                workflow_gate(tag)
            p=TARGET/'filter_diagnostics'/tag/'score/COMPLETE.json';value=json.loads(p.read_text())
            require(value['complete'] is True and value['passed'] is True and value['component_rows']==35 and value['T3_stage_rows']==165
                and value['original292_component28_exact'] is True and value['original292_T3_stage132_exact'] is True,'Final diagnostic35/165 parity gate failed')
            dest=OUT/'FILTER_DIAGNOSTIC_WORKFLOW_COMPLETE.json'
            with dest.open('x') as f:f.write(json.dumps(dict(complete=True,passed=True,utc=utc(),tag=tag,score_complete_path=str(p),
                score_complete_sha256=sha(p),new_forwards=320,source_sha256=workflow_gate(tag)[0]['source_sha256'],
                stages=stages,resume_authority_sha256=sha(RESUME_AUTHORITY),metadata_adapter_receipt_sha256=adapter_receipts,
                outside_GLOBAL_lock_acquired=False,published=False,independent_numeric_QA_pending=True),indent=2)+'\n')
        except Exception as error:
            p=OUT/'FILTER_DIAGNOSTIC_WORKFLOW_FAILURE_V3.json'
            with p.open('x') as f:f.write(json.dumps(dict(utc=utc(),error=repr(error),automatic_retry=False,published=False),indent=2)+'\n')
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--seal',action='store_true');mode.add_argument('--run',action='store_true');parser.add_argument('--tag',required=True)
    args=parser.parse_args();require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',args.tag),'Unsafe diagnostic tag')
    workflow_gate(args.tag)
    if args.run:run(args.tag)
    else:wait_for_all15(args.tag);seal(args.tag)


if __name__=='__main__':main()
