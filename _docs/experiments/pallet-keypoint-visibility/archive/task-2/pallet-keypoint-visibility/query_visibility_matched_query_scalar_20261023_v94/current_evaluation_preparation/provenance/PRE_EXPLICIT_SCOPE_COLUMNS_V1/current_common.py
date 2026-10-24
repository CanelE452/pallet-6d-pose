"""Source-first authorization and inert imports. No current I/O on import."""
from pathlib import Path
import hashlib
import importlib.util
import json

HERE=Path(__file__).resolve().parent
V94=HERE.parent
P=V94.parent
PROTOCOL=V94/'CURRENT_PROTOCOL_BEFORE_CURRENT_ARTIFACTS.json'  # root creates after source gates
SOURCE_KEYS={'source_model_freeze','source_fit_complete','source_gate_results','source_postfit_audit',
             'source_fit_protocol','checkpoint','model_code'}
CURRENT_KEYS={'current_model_input','prior_input_rows','prior_preparation_seal','prior_inference_seal',
              'prior_input_gate','prepared_inputs','feature_metadata'}
EVAL_KEYS={'evaluation_manifest','manual_provenance','baseline_v91_scores','baseline_v91_inference_seal',
           'historical_v91_metrics','original_baseline_metrics'}
V91_CODE=P/'query_visibility_physical_cue_aux_20261023_v91/private_evaluation/infer_native_center_once.py'
V91_SHA='df126041c752c02ab5842765fcfcde294cf02ee666f5c604417b812b3991a5bb'
BASE_CODE=P/'query_visibility_cue_occlusion_20261023_v72/contract/saved_evaluate_current_scores.py'
BASE_SHA='d079650b65aef777675af552da61335a28b73f978580413d74d143433019bef6'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def bind(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text())


def require(ok,why):
    if not ok:raise PermissionError(why)


def same(a,b):return isinstance(a,dict)and a.get('path')==b.get('path')and a.get('sha256')==b.get('sha256')


def contains(tree,binding):
    if isinstance(tree,dict):return same(tree,binding)or any(contains(x,binding)for x in tree.values())
    if isinstance(tree,list):return any(contains(x,binding)for x in tree)
    return False


def check(d):
    require(isinstance(d,dict)and {'path','sha256'}<=set(d),'Immutable path/SHA required')
    require(sha(d['path'])==d['sha256'],'Changed artifact: '+d['path'])
    return Path(d['path'])


def import_code(path,expected,module_name):
    require(sha(path)==expected,'Pinned code changed')
    spec=importlib.util.spec_from_file_location(module_name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def prior_reader():return import_code(V91_CODE,V91_SHA,'v94_pinned_prior_input_reader')
def metric_base():return import_code(BASE_CODE,BASE_SHA,'v94_pinned_saved_metric_core')


def guard_source(protocol,b,freeze,completion,gates,audit,source_protocol):
    """Pure generated-testable guard; no current descriptor dereferencing."""
    require(same(protocol['selected_source_freeze'],b['source_model_freeze']),'Preselected source changed')
    require(freeze.get('completed')is True and int(freeze.get('selected_update',0))>0,'Incomplete source selection')
    require(freeze.get('source_only')is True and freeze.get('source_input_bindings',{}).get('current_allowed')is False,'Source-only scope missing')
    require(isinstance(freeze.get('entry'),dict)and {'path','sha256'}<=set(freeze['entry']),'Source entry binding missing')
    require(freeze.get('PnP',0)==0,'Source selection leakage')
    for key in ('checkpoint','model_code'):
        require(same(freeze.get(key),b[key]),'Source '+key+' mismatch')
    require(same(freeze.get('protocol'),b['source_fit_protocol']),'Source protocol mismatch')
    require(completion.get('completed')is True and contains(completion,b['source_model_freeze']),'Fit completion mismatch')
    require(gates.get('completed')is True and gates.get('all_gates_pass')is True,'Source gates not passed')
    names=protocol['source_gate_names'];checks=gates.get('gates',{})
    require(set(checks)==set(names)and len(names)==3,'Source gate coverage changed')
    require(all(v is True or(isinstance(v,dict)and v.get('PASS')is True)for v in checks.values()),'At least one source gate failed')
    require(audit.get('PASS')is True and contains(audit,b['source_model_freeze']),'Independent source audit missing')


def validate_entry(path,code_path,stage):
    """Source complete/PASS, then authorize current descriptors; bodies deferred."""
    e=read(path)
    require(e.get('schema')=='V94_ROOT_CURRENT_ENTRY_V1'and e.get('approved')is True,'Root approval required')
    require(e.get('stage')==stage,'Wrong execution stage')
    require(e.get('current_used_for_selection')is False and e.get('training_allowed')is False and e.get('PnP_allowed')is False,'Forbidden scope')
    for k in ('protocol','code','common_code','generated','independent_review','artifact_bindings'):check(e[k])
    require(same(e['code'],bind(code_path))and same(e['common_code'],bind(__file__)),'Unreviewed code')
    require(Path(e['protocol']['path']).resolve()==PROTOCOL.resolve(),'Root final protocol required')
    p=read(PROTOCOL)
    require(p.get('schema')=='V94_SAME_QUERY_CURRENT_EVALUATION_PROTOCOL_V1'and p.get('actual_current_allowed')is True,'No current execution under draft')
    require(p.get('raw_D_rule')=='z<0'and p.get('raw_positive_label')=='AUX_HIDDEN_E'and p.get('new_fit_allowed')is False,'Decoder/scope changed')
    proof=read(e['generated']['path']);review=read(e['independent_review']['path'])
    require(proof.get('PASS')is True and contains(proof,e['code']),'Generated proof/code mismatch')
    require(review.get('PASS')is True and not review.get('blockers',[])and contains(review,e['code']),'Review mismatch')
    packet=read(e['artifact_bindings']['path'])
    require(packet.get('completed')is True and packet.get('authorized_after_source_audit')is True,'Artifact authorization missing')
    b=packet['bindings'];require(set(b)==SOURCE_KEYS|CURRENT_KEYS|EVAL_KEYS,'Exact source/current binding schema required')
    for k in SOURCE_KEYS:check(b[k])
    freeze=read(b['source_model_freeze']['path']);check(freeze['entry'])
    guard_source(p,b,freeze,read(b['source_fit_complete']['path']),
                 read(b['source_gate_results']['path']),read(b['source_postfit_audit']['path']),read(b['source_fit_protocol']['path']))
    # No current artifact SHA/body was read above. Infer must not read EVAL_KEYS.
    return e,p,b


def write_new(path,obj):
    with Path(path).open('x')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
