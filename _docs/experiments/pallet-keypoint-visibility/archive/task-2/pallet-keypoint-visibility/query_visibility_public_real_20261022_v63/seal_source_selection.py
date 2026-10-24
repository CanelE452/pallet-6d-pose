"""Seal source-only candidate selection and authorize one frozen evaluation.

Reads source outputs, model artifacts and the private-input *binding template*.
Does not open any private input, image, annotation or score.
"""
from pathlib import Path
import datetime,hashlib,json

ROOT=Path(__file__).resolve().parent
MIXED=['MIXED_CONTEXT_V2','MIXED_CONTEXT_RGB_V3','MIXED_CONTEXT_RGB_SAM_V4']
CONTROLS=['SYNTH_CONTEXT_V2','SYNTH_CONTEXT_RGB_V3','SYNTH_CONTEXT_RGB_SAM_V4']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def bind(p):p=Path(p).resolve();return dict(path=str(p),sha256=sha(p))
def write(p,x):
    assert not Path(p).exists(),'Preserve original source selection/evaluation gate'
    Path(p).write_text(json.dumps(x,indent=2)+'\n')

reports=[]
for name in ['V61',*CONTROLS,*MIXED,'FROZEN_MOBILESAM']:
    p=ROOT/'source_comparison'/name/'SOURCE_INFERENCE.json';r=read(p)
    assert r['threshold']==.5 and {s['source']for s in r['source_metrics']}=={'synthetic','ycbv'}
    assert sum(s['N']for s in r['source_metrics'])==4942
    assert r.get('eval_images_accessed',False)is False and r.get('current_eval_accessed',False)is False
    reports.append(dict(name=name,**r,report_binding=bind(p),outputs_binding=bind(p.with_name('SOURCE_OUTPUTS.npz'))))
primary=min([r for r in reports if r['name']in MIXED],key=lambda r:r['macro_source_BCE'])
rules=dict(exact_pixel='Each head sigmoid>=.5, raw SAM logit>0. Raw probabilities retained.',
           self='Report raw learned output and saved-initial-pose cuboid self proxy separately; an optional fixed geometry-first decision is descriptive and never learned.',
           neighborhood='Raw SAM radii1/2 are prespecified finite-neighborhood support diagnostics; do not relabel exact-pixel source metrics or tune radius on current data.',
           ablations='Evaluate all six frozen heads, v61, legacy and fixed raw SAM. Private scores cannot replace the source-selected primary.',
           location='Existing reference-coordinate error is evaluation-only. <=3px and manual-only slices are descriptive, never an inference input.',
           abstention='Raw binary heads have no abstention policy. Missing SAM/input is unresolved. No new threshold calibration on current evaluation.')
seal=dict(schema='SOURCE_SELECTION_SEAL_V63',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          selected_primary=primary['name'],selection_rule='Strict minimum mean(SYNTH_VAL_BCE,YCB_VAL_BCE) among three predesignated mixed candidates',
          selected_macro_source_BCE=primary['macro_source_BCE'],threshold=.5,source_validation_queries=4942,
          current_eval_used_for_selection=False,current_eval_inference_started=False,new_PnP_calls=0,
          source_reports=reports,diagnostic_rules=rules,protocol=bind(ROOT/'PROTOCOL.json'),SAM_protocol=bind(ROOT/'model/SAM_RESIDUAL_V4_PROTOCOL.json'),
          source_SAM_seal=bind(ROOT/'sam_source/SEAL.json'),source_SAM_replay_audit=bind(ROOT/'sam_source/BASELINE_REPLAY_AUDIT.json'),
          original_v61_and_legacy_preserved=True,private_evaluation_role='Evaluation only for this experiment; historically reused development cohort, not a new independent holdout')
sealpath=ROOT/'SOURCE_SELECTION_SEAL.json';write(sealpath,seal)
template=read(ROOT/'inference/CURRENT_EVAL_GATE_TEMPLATE.json')
gate=dict(schema='CURRENT_EVAL_SOURCE_SELECTION_GATE_V1',approved=True,source_selection_complete=True,allow_current_evaluation_inference=True,current_eval_used_for_selection=False,
          source_selection_seal=bind(sealpath),selected_primary=primary['name'],inputs=template['inputs'],approved_candidates=[],diagnostic_rules=rules)
for name in [*CONTROLS,*MIXED]:
    run=ROOT/'runs'/name;f=read(run/'MODEL_FREEZE.json')
    assert f['completed']and f['eval_data_used']is False and f['PnP_used']is False
    c=dict(name=name,checkpoint=bind(run/'BEST_QUERY_HEAD.pt'),freeze=bind(run/'MODEL_FREEZE.json'),model_code=bind(run/'code/query_head.py'),
           geometry_helper=bind(ROOT/'contract/geometry_self_rule.py'),inference_code_sha256=sha(ROOT/'inference/infer_current_frozen.py'),output_directory=str(ROOT/'private_evaluation'/name))
    if f['head_config']['rgb_branch']:c['rgb_recipe']=bind(ROOT/'model/build_rgb_source_cache.py')
    if f['head_config'].get('sam_branch',False):
        loader=ROOT/'sam/standalone_mobile_sam.py'
        c['sam_assets']=dict(loader=bind(loader),checkpoint=bind(Path(read(ROOT/'sam/PREPARATION_LOCK.json')['checkpoint'])),runtime_audit=bind(ROOT/'contract/MOBILESAM_STANDALONE_AUDIT.json'),source_recipe=bind(ROOT/'sam_source/LOCK.json'))
        c['compute_sam_baseline']=True
    gate['approved_candidates'].append(c)
write(ROOT/'CURRENT_EVAL_GATE.json',gate)
print(json.dumps(dict(selected_primary=primary['name'],macro_source_BCE=primary['macro_source_BCE'],seal_sha256=sha(sealpath),gate_sha256=sha(ROOT/'CURRENT_EVAL_GATE.json'),private_files_opened=0)))
