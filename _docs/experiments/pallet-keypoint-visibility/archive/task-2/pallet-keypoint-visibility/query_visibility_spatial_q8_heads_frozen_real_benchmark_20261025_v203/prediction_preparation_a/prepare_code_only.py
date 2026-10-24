"""Run only authored arithmetic/CPU toy checks and freeze future false entries."""
from pathlib import Path
import ast,hashlib,importlib.util,json,os,stat,sys

H=Path(__file__).absolute().parent;V=H.parent;E=V/'evaluation_preparation_a'
PYTHON='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
SCOPE=V/'ROOT_PROSPECTIVE_FROZEN_REAL_BENCHMARK_SCOPE.json'
SCOPE_SHA='e49b15a8c298a6f2e5f4aaaf7787c4974b8b777abbc573752bb51ecdfcfd3bdf'

def safe(path):
    p=Path(path);assert p.is_absolute()and'..'not in p.parts and all('v112'not in x.casefold()for x in p.parts)
    cur=Path(p.anchor)
    for x in p.parts[1:]:cur/=x;assert not stat.S_ISLNK(os.lstat(cur).st_mode),cur
    return p

def bind(path):
    p=safe(path);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(path):return json.loads(safe(path).read_text())

def write(path,value):
    p=Path(path);safe(p.parent);text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with p.open('x')as f:f.write(text)

def module(path,name):
    p=safe(path);ast.parse(p.read_text());spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def run():
    assert Path(sys.executable)==Path(PYTHON)and bind(SCOPE)['sha256']==SCOPE_SHA
    p=module(H/'infer_both_frozen_spatial_q8_heads.py','v203_prediction_preparation_only')
    e=module(E/'evaluate_saved_spatial_q8_heads_real.py','v203_saved_evaluation_preparation_only')
    p.prepare();e.prepare()
    pred=read(H/'PREPARATION_COMPLETE.json');evaluation=read(E/'PREPARATION_COMPLETE.json')
    assert pred['PASS']and evaluation['PASS']
    schema=dict(completed=True,prepared_only=True,current_models=list(e.MODELS),new_models=list(e.TAGS),old_models=list(e.BASELINES+e.OLD_TAGS),direct_models=list(e.DIRECT_TAGS),
        prediction_keys16=list(p.META)+['raw_support_logits','raw_owner_logits','raw_direct_end_external_logit','support_prediction','owner_prediction','direct_end_prediction_D0_E1_Uminus1','regional_support_proxy_D0_E1_Uminus1','support_order','owner_order','direct_end_order'],
        current_ledger=dict(path='saved_real_head_evaluation_v1/current1784/ALL1784_CURRENT_NATIVE_LEDGER.jsonl',original_rows=1784,primary1237=dict(D=1147,E=90),inputU=45,
            core_fields=['global_index','id','native_corner','group','human_state','mapping_status','query_x','query_y','roi_side','query_available','human_reference_x','human_reference_y','paired_same_query_RGB_D_vs_E','same_slot_pipeline_D_vs_E','initial_reference_error_px','original_query_valid','legacy_original_state'],
            every_model_fields=['<model>_original_response','<model>_D_E_U'],
            support_owner_model_fields=['<tag>_owner_prediction','<tag>_support_prediction','<tag>_regional_support_proxy_D0_E1_Uminus1','<tag>_raw_support_logits','<tag>_raw_owner_logits','<tag>_primary_prediction','<tag>_primary_route'],
            direct_model_fields=['<tag>_raw_direct_end_external_logit','<tag>_direct_end_prediction_D0_E1_Uminus1'],
            self_guard='All136 humanSELF, paired21 uses original truth paired_same_query_RGB_D_vs_ES, not D_vs_E; ESflag is not exported in this current ledger.',
            reference_scope='Existing humanD reference error<=3px461 is separate from manualnear376(153+223), not physical coordinate gold.'),
        public=dict(path='saved_real_head_evaluation_v1/publicheld360/ALL360_PUBLIC_WEAK_LEDGER.jsonl',rows=360,inputU=108,models=['v61']+list(e.ALLTAGS),legacy='MISSING_NO_FORWARD',cohorts=['FULL360','KNOWN_CENTER_TRUTH','ALL11_MODEL_PAIRED_AVAILABLE','KNOWN_TRUTH_AND_ALL11_PAIRED_AVAILABLE'],authority='Existing weak center-owner/crop-presence only, not native visibility or physical external gold.'),
        summary_paths=['current1784/CURRENT_HUMAN_SEMANTIC_AND_SEPARATE_ASSOCIATIONS.json','publicheld360/PUBLIC_WEAK_OWNER_AND_SEPARATE_SUPPORT_ASSOCIATIONS.json'],
        old10_guard='Root-approved a33f5ae9 V200 COMPLETE binding -> current1784/METRICS.csv only after actual saved-evaluation entry; no old score/summary reads in preparation.',
        no_actual_input_annotation_score_weight_or_model_body=True)
    schema_path=E/'PREPARED_OUTPUT_SCHEMA.json';write(schema_path,schema)
    write(H/'FINAL_READY.json',dict(pred,prepare_code=bind(__file__),scope=bind(SCOPE),actual_fit_complete_or_QA_read=False,actual_prediction_or_evaluation=False))
    final=dict(completed=True,PASS=True,prepared_only=True,scope=bind(SCOPE),prediction=dict(code=pred['code'],protocol=pred['protocol'],checks=pred['checks'],entry_template=pred['entry_template'],preparation=bind(H/'PREPARATION_COMPLETE.json'),FINAL_READY=bind(H/'FINAL_READY.json')),
        evaluation=dict(code=evaluation['code'],protocol=evaluation['protocol'],checks=evaluation['checks'],entry_template=evaluation['entry_template'],preparation=bind(E/'PREPARATION_COMPLETE.json'),schema=bind(schema_path)),prepare_code=bind(__file__),actual_source_or_real_metadata_array_header_pixel_weight_score_or_result_reads=0,actual_model_or_fit_or_inference=0,invented_CPU_toy_only=True,
        context_contract='Same original input-only group8 availability/qL from original ROWS; V202 pure F64 clipped relative XY and self/U zero; C60 zeroXY, C61 spatialXY; no source-only loader invoked.',requires_separate_root_prediction_entry_after_source_finals_and_integrity_QA=True,requires_separate_saved_eval_entry_after_both_real_prediction_seal=True,automatic_retry=False,selected_model=None,fully_solved=False)
    write(E/'FINAL_READY.json',final)
    print(json.dumps(dict(completed=True,PASS=True,FINAL_READY=bind(E/'FINAL_READY.json'),prediction=final['prediction'],evaluation=final['evaluation'])))

if __name__=='__main__':
    try:run()
    except BaseException:
        import traceback
        failure=H/'PREPARATION_FAILURE_NO_AUTOMATIC_RETRY.json'
        write(failure,dict(completed=False,PASS=False,prepared_only=True,code=bind(__file__),traceback=traceback.format_exc(),actual_real_or_source_body_reads=0,automatic_retry=False))
        raise
