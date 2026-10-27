"""V2: seal primary372 inputs, normalizing the existing WOOD80 SHA schema.

No model, RGB or raw annotation is opened. Original292 RGB hashes are copied
from its existing allowlist; exact authorized80 RGB hashes keep their original
authorization category. Reference coordinates are only in permitted caches.
The original sealer and failed log remain unchanged. This repair only adds
sha256/bytes from the existing byte-verified snapshot after matching its
declared manifest SHA, prepared MANIFEST SHA and unchanged authorization.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import re

OUT=Path(__file__).resolve().parents[1]
MAIN=OUT.parent/'dope-filter-ablation-per-type-20261027-v1'
RATIO=OUT.parent/'dope-r2-ratio-per-type-20261027-v1'
WOOD=OUT.parent/'dope-wood640-eval80-20261027-v1'
PRIOR=OUT.parent/'dope-loo-flip-per-type-20261026-v1'
NOFLIP=OUT.parent/'dope-loo-no-flip-per-type-20261026-v1'
COMBINED=WOOD/'combined372'
REPO=Path('/home/minjae/Documents/github/pallet-pose')
EVALUATOR={
    'ratio_eval_common.py':'3f16bcfe9dba1510c81984bb54113b18b2c43728158cf543465d106c03a495ad',
    'evaluate_ratios.py':'3ae3557ed1ec5dba298511e34de64a59a239a8018e5ae49ec731d6a44c8366ee',
    'plot_ratios.py':'9b92a83702ae943acceabe5b67c4ad9295deee7d7bfca93eb8f3ad0499569568',
    'future_teacher_diagnostics.py':'9d13a701f68320c05b74f99d382e34bd03eadec6f33b7ecc79c8a7fc98298b24',
    'test_ratio_evaluation.py':'9b9ab4cd7984443e1f586d8e84973e2098b2beb2ab7bdce652da95bd030894d3',
}
NEW_FROZEN_SOURCES={
    str(MAIN/'code/evaluate_models.py'):'a58463788e470470b0bf55326305393ddf1953cbfecd0c65274421755fead615',
    str(MAIN/'code/filter_ablation.py'):'d224fa16309ad094a0d9c2c58ffaeb2c8cb4efb0a4abffe28806f492b8a0ad46',
}


def require(ok,msg):
    if not ok:raise RuntimeError(msg)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_new(path,value):
    require(path.resolve().is_relative_to(RATIO),'Output outside new RATIO')
    with path.open('x') as stream:stream.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def normalized_authorized_rgb(item,baseline,frame):
    """Pure metadata normalization; never hash/open the RGB named by frame."""
    require(all(isinstance(x,dict) for x in (item,baseline,frame)),'Invalid exact80 metadata')
    digest=baseline.get('sha256');size=baseline.get('bytes')
    require(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest)
        and type(size) is int and size>=0,'Actual80 byte authority is missing')
    require(all(isinstance(frame.get(k),str) and frame[k] for k in ('source_frame_id','session')),
        'Exact80 prepared frame/session identity is missing')
    require(item.get('category')==baseline.get('category')=='authorized_WOOD80_RGB'
        and item.get('frame_id')==frame.get('source_frame_id')
        and item.get('session')==frame.get('session')
        and baseline.get('manifest_matches_actual') is True
        and item.get('manifest_sha256')==baseline.get('manifest_sha256')==digest==frame.get('image_sha256'),
        'Exact80 declared/actual/prepared RGB binding differs')
    require(('sha256' not in item or item['sha256']==digest)
        and ('bytes' not in item or item['bytes']==size),'Conflicting normalized80 authority')
    result=dict(item);result.update(sha256=digest,bytes=size)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seal',action='store_true',required=True);p.parse_args()
    targets=[RATIO/'preflight/EVAL_ALLOWLIST.json',RATIO/'preflight/EVALUATION_GO.json']
    require(all(not p.exists() for p in targets),'Preserve existing evaluation preparation')
    root_review=RATIO/'preflight/EVALUATOR_ROOT_REVIEW_V2.json'
    root=read(root_review)
    sources={str(RATIO/'code'/n):h for n,h in EVALUATOR.items()}
    require(root['passed'] is True and root['source_sha256']==sources
        and root['allowlist_preparation_source_sha256']==sha(__file__),'ROOT evaluator/allowlist source review absent or stale')
    for path,h in sources.items():require(sha(path)==h,'Frozen evaluator source changed')
    tests=read(RATIO/'preflight/EVALUATOR_PURE_TESTS.json')
    require(tests['passed'] is True and tests['tests_run']==12 and tests['failures']==tests['errors']==0
        and tests['neural_imports']==0 and tests['source_sha256']==sources,'Pure evaluator tests absent/stale')
    train=read(RATIO/'preflight/TRAINING_GO.json');require(train['passed'] is True and train['training_authorized'] is True,'Training GO absent')
    campaign=read(MAIN/'CAMPAIGN_COMPLETION.json');integrity=read(MAIN/'preflight/FINAL_INTEGRITY.json')
    audit=read(MAIN/'preflight/FINAL_AUDIT.json')
    require(campaign['finished'] is True and campaign['all_arms_complete'] is True
        and campaign['frozen_inputs_unchanged'] is True and audit['passed'] is True and audit['complete'] is True
        and integrity['passed'] is True and integrity['all_original_experiment_and_checkpoint_bytes_unchanged'] is True
        and integrity['all_allowed_input_bytes_unchanged'] is True,'Original MAIN completion/integrity failed')
    old_path=MAIN/'preflight/ALLOWLIST.json';old=read(old_path)
    require(sha(old_path)==read(RATIO/'preflight/ALLOWLIST.json')['previous_allowlist']['sha256'],'Prior allowlist changed')
    sem=read(WOOD/'preflight/SEMANTIC_CONTRACT_REVIEW.json')
    require(sem['passed'] is True and sem['same_index_0_7_compatible'] is True
        and sem['GT_optimized_permutation_used'] is False and sem['coordinates_or_indices_modified'] is False,'Same-index 2D review absent')
    done=read(WOOD/'inference/COMPLETE.json');require(done['complete'] is True,'WOOD80 inference incomplete')
    core=read(PRIOR/'MANIFEST.json')['records'];extra=read(WOOD/'MANIFEST.json')['records']
    require(len(core)==292 and len(extra)==80 and len({r['id'] for r in core+extra})==372,'Exact disjoint292+80 required')
    require(sem['manifest_sha256']==sha(WOOD/'MANIFEST.json'),'Semantic review/80 manifest differ')
    inputs={}

    def add(path,category=None,expected=None):
        path=Path(path);require(path.is_file() and not path.is_symlink(),'Missing/symlink external input: '+str(path))
        path=path.resolve();known=old['input_files'].get(str(path));digest=sha(path)
        if expected is not None:require(digest==expected,'Expected input SHA changed: '+str(path))
        if known is not None:
            require(digest==known['sha256'],'Original input changed: '+str(path))
            item=dict(known)
        else:
            require(category in ('metadata','cached_scores','cached_predictions') or
                category=='source_code' and expected is not None and NEW_FROZEN_SOURCES.get(str(path))==expected,
                'Only permitted new metadata/caches or two explicitly frozen MAIN sources')
            item=dict(category=category,sha256=digest,bytes=path.stat().st_size)
        require(item['category'] in ('metadata','protocol','checkpoint_receipt','source_code','cached_scores',
            'cached_predictions','dev_two_view_predictions'),'Raw/unused source cannot enter evaluation metadata')
        inputs[str(path)]=item

    for name in ('MANIFEST.json','SCORES.json','PREDICTIONS.json'):add(PRIOR/name)
    for source in (MAIN/'code/evaluate_models.py',MAIN/'code/filter_ablation.py',PRIOR/'geometry.py',
        NOFLIP/'geometry.py',PRIOR/'run_type.py',REPO/'Deep_Object_Pose/common/models.py',
        REPO/'scripts/research/pallet_dope_refiner_20261001_v1/dope_adapter.py'):
        add(source,'source_code',NEW_FROZEN_SOURCES.get(str(source)))
    for kind in ('PLASTIC','WOOD','GREEN'):
        add(PRIOR/kind/'PROTOCOL.json')
        for role in ('R0','R1','control_R2','oldfull_R2'):
            cache=COMBINED/'cache'/f'BASE_{kind}_{role}.json';item=read(cache)
            require(item['complete'] is True and item['kind']==kind and item['role']==role
                and item['manifest_sha256']==sha(COMBINED/'MERGED_MANIFEST.json'),'Merged baseline cache not complete/bound')
            for source,digest in item['source_sha256'].items():require(sha(source)==digest,'Merged baseline source changed')
            add(cache,'cached_predictions')
    for path in (WOOD/'MANIFEST.json',WOOD/'AUTHORIZATION.json',WOOD/'ALLOWLIST.json',
        WOOD/'preflight/SNAPSHOT_BEFORE.json',
        WOOD/'preflight/SEMANTIC_CONTRACT_REVIEW.json',WOOD/'inference/COMPLETE.json',
        COMBINED/'MERGED_MANIFEST.json',COMBINED/'MERGED_GEOMETRY_INPUTS.json',COMBINED/'PREPARATION_QA.json'):
        add(path,'metadata')
    add(COMBINED/'MERGED_REFERENCES.json','cached_scores')
    for frame in core:
        path=str(Path(frame['image_path']).resolve());item=old['input_files'][path]
        require(item['category']=='dev_rgb' and item['sha256']==frame['image_sha256'],'Original DEV RGB binding differs')
        inputs[path]=dict(item)  # Known opaque hash only; no RGB bytes opened.
    raw80=read(WOOD/'ALLOWLIST.json');before80=read(WOOD/'preflight/SNAPSHOT_BEFORE.json')
    require(raw80['authorization_sha256']==sha(WOOD/'AUTHORIZATION.json')
        and before80['allowlist_sha256']==sha(WOOD/'ALLOWLIST.json')
        and len(raw80['records'])==80 and len(raw80['input_files'])==len(before80['files'])==160
        and set(raw80['input_files'])==set(before80['files']),'Exact80 authorization/snapshot binding differs')
    exact80=raw80['input_files']
    for frame in extra:
        path=str(Path(frame['image_path']).resolve());item=exact80[path]
        inputs[path]=normalized_authorized_rgb(item,before80['files'][path],frame)
    allow=dict(schema='ratio_primary372_exact_evaluation_allowlist_v1',input_files=inputs,
        dev_frames={f['id']:f for f in core},approved_wood80_frames={f['id']:f for f in extra},
        protected_image_sha256=old.get('protected_image_sha256',[]),
        wood80_authorization=dict(manifest_path=str(WOOD/'MANIFEST.json'),manifest_sha256=sha(WOOD/'MANIFEST.json'),
            authorization_path=str(WOOD/'AUTHORIZATION.json'),authorization_sha256=sha(WOOD/'AUTHORIZATION.json')),
        raw_RGB_bytes_opened_during_preparation=0,raw_annotations_opened=0)
    write_new(targets[0],allow)
    write_new(targets[1],dict(passed=True,evaluation_authorized=True,canonical_FP32_authorized=True,
        future_probe_authorized=True,utc=utc(),eval_allowlist_sha256=sha(targets[0]),
        training_go_sha256=sha(RATIO/'preflight/TRAINING_GO.json'),fixed_plan_sha256=sha(RATIO/'FIXED_PLAN.json'),
        source_sha256=sources,allowlist_preparation_source_sha256=sha(__file__),
        root_review_sha256=sha(root_review),
        primary_population='372 frames/2891 reference corners; WOOD125/963',
        raw_reference_files_opened=0,neural_forwards=0))
    print('RATIO_PRIMARY372_EVALUATION_SEALED',len(inputs),flush=True)


if __name__=='__main__':main()
