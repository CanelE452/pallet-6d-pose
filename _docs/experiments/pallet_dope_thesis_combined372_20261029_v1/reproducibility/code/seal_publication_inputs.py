"""Seal exact, already completed derived inputs in three delivery phases.

No training, inference, PnP, drawing, document authoring or Git operations.
ROOT supplies the reviewed seal plan; this never creates any execution GO.
Import performs no IO. Missing files fail before the final manifest is written.
"""
from pathlib import Path
import argparse
import json
import re
import publication_common as P

SOURCE=Path(__file__).resolve()
ROOT=P.OUT.parent
W80=ROOT/'dope-wood640-eval80-20261027-v1'
MAIN=ROOT/'dope-filter-ablation-per-type-20261027-v1'
RATIO=ROOT/'dope-r2-ratio-per-type-20261027-v1'
OLD=ROOT/'dope-graduation-thesis-20261026-v1'
PLAN=P.OUT/'preflight/ROOT_MANIFEST_SEAL_PLAN.json'
TARGETS={'numerical':P.NUMERICAL_INPUTS,'manuscript':P.MANUSCRIPT_INPUTS,'publication':P.INPUTS}
STAGES=('numerical','package','publish')
FIXED_TAGS={'combined':'final_v1','ratio':'ratios_v1','comparison':'main_v1','diagnostic':'filter372_v1'}
FILTER_QA=P.OUT/'preflight/MERGED_FILTER_STATISTICS_QA.json'
FILTER_VERIFIER_SHA='ad79ec5bf0f327a8464269ffe96df4427a299e67a4857ef99d269b06963a477e'


def safe_tag(value):
    P.require(isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}',value),'Unsafe ROOT tag')
    return value


def fixed_roles(tags):
    """Only named, scalar-derived outputs from the already reviewed producers."""
    a=W80/'combined372/evaluation'/safe_tag(tags['combined'])/'final'
    pose=a.parent/'supplemental-pose'
    ratio=RATIO/'evaluation'/safe_tag(tags['ratio'])/'all'
    teacher=RATIO/'diagnostics'/safe_tag(tags['ratio'])/'all'
    comparison=W80/'combined372/comparisons'/safe_tag(tags['comparison'])
    diag=W80/'combined372/filter_diagnostics'/safe_tag(tags['diagnostic'])
    result={}
    def add(role,path,destination=None,private=False):
        result[str(path)]={'role':role,'destination':destination,'private_input':private}
    for role,name in [('table_ablation_2d','MODEL_SUMMARY.csv'),('table_WOOD_manual','WOOD_MANUAL_SUMMARY.csv'),
        ('table_GREEN_manual','GREEN_MANUAL_SUMMARY.csv'),('ci_ablation','PAIRED_CI.csv'),
        ('ci_ablation_WOOD_manual','WOOD_MANUAL_PAIRED_CI.csv'),('ci_ablation_GREEN_manual','GREEN_MANUAL_PAIRED_CI.csv'),
        ('ci_ablation_WOOD_recording','WOOD_RECORDING_PAIRED_CI.csv')]:add(role,a/name,'results/ablation/'+name)
    add('private_ablation_errors',a/'SCORES.json',private=True)
    add('merged_complete',a/'COMPLETE.json','reproducibility/combined_COMPLETE.json')
    for role,name in [('table_supplemental_pose','MODEL_SUMMARY.csv'),('frames_supplemental_pose','FRAME_ERRORS.csv')]:
        add(role,pose/name,'results/supplemental_pose/'+name)
    for role,name in [('table_ratio','MODEL_SUMMARY.csv'),('table_ratio_GREEN_manual','GREEN_MANUAL_SUMMARY.csv'),
        ('ci_ratio','PAIRED_CI.csv'),('ci_ratio_GREEN_manual','GREEN_MANUAL_PAIRED_CI.csv')]:
        add(role,ratio/name,'results/ratios/'+name)
    add('private_ratio_errors',ratio/'SCORES.json',private=True)
    add('table_future_teacher',teacher/'FUTURE_R3_PSEUDO_QUALITY.csv','diagnostics/future_teacher/FUTURE_R3_PSEUDO_QUALITY.csv')
    for role,name in [('ci_main_comparisons','PAIRED_CI.csv'),('ci_main_comparisons_WOOD_manual','WOOD_MANUAL_PAIRED_CI.csv'),
        ('ci_main_comparisons_GREEN_manual','GREEN_MANUAL_PAIRED_CI.csv'),('ci_main_comparisons_WOOD_recording','WOOD_RECORDING_PAIRED_CI.csv')]:
        add(role,comparison/name,'results/main_comparisons/'+name)
    for role,name in [('table_R0_components372','COMPONENT_QUALITY.csv'),('table_T3_stages372','T3_FILTER_STAGES.csv')]:
        add(role,diag/'score/public'/name,'diagnostics/filters/'+name)
    add('latest_user_policy',P.POLICY_PATH,'reproducibility/CONTINUATION_POLICY.json')
    gates={}
    def gate(path,equals,role):
        gates[str(path)]={'phases':list(STAGES),'equals':equals}
        if str(path) not in result:add(role,path,'reproducibility/gates/'+role+'.json')
    gate(MAIN/'preflight/FINAL_AUDIT.json',{'passed':True,'complete':True},'original_MAIN_audit')
    gate(MAIN/'preflight/FINAL_INTEGRITY.json',{'passed':True},'original_MAIN_integrity')
    gate(P.OUT/'FINISH_EXPERIMENTS_COMPLETE.json',{'complete':True,'primary_frames':372},'all15_workflow_complete')
    gate(a/'COMPLETE.json',{'complete':True,'passed':True,'primary_frames':372,'primary_reference_points':2891,'main_groups':51},'merged_complete')
    gate(pose/'COMPLETE.json',{'complete':True,'passed':True,'groups':51,'WOOD_frames':125,
        'scope':'SUPPLEMENTAL_2D_RECONSTRUCTED_REFERENCE','independent_measured_6D':False},'supplemental_pose_complete')
    gate(ratio/'COMPLETE.json',{'complete':True,'phase':'all','main_groups':15,'manual_groups':5,
        'main_CI_rows':135,'GREEN_manual_CI_rows':45},'ratio_complete')
    gate(teacher/'COMPLETE.json',{'complete':True,'phase':'all','metric_rows':20,'future_R3_probe_only':True},'teacher_complete')
    gate(comparison/'COMPLETE.json',{'complete':True,'passed':True,'primary_frames':372,'primary_CI_rows':90,
        'WOOD_manual_CI_rows':30,'GREEN_manual_CI_rows':30,'WOOD_recording_CI_rows':60},'main_comparison_complete')
    gate(diag/'select/EQUIVALENCE.json',{'complete':True,'passed':True,'mask9_pass_inliers_exact':True,
        'original_final_decomposition_exact':True,'reference_coordinates_read':False},'filter_equivalence')
    gate(diag/'score/COMPLETE.json',{'complete':True,'passed':True,'population':'PRIMARY372','frames':372,'WOOD_frames':125,
        'references':{'PLASTIC':985,'WOOD':963,'GREEN':943},'manual_references':{'WOOD':330,'GREEN':600},'component_rows':35,'T3_stage_rows':165,
        'original292_component28_exact':True,'original292_T3_stage132_exact':True},'filter_complete')
    return result,gates


def capture(path,item,expected=None):
    p=Path(path)
    P.require(p.is_absolute() and p.resolve()==p and not p.is_symlink() and p.is_file(),'Missing/unsafe exact derived source: '+str(p))
    safe_roots=(P.OUT,W80/'combined372',RATIO/'evaluation',RATIO/'diagnostics',RATIO/'figures',MAIN/'preflight',
        OLD/'references',OLD/'runtime_compat',ROOT/'dope-thesis-analysis-batch1-v1/preflight',
        ROOT/'dope-thesis-analysis-batch2-v1/publication/pose_examples_revision_20261027_v2/payload')
    safe_exact=(P.POLICY_PATH,OLD/'draft_content.json',OLD/'render_docx_compat.py')
    docs=Path('/home/minjae/Documents/github/pallet-pose/_docs')
    P.require(p in safe_exact or any(p.is_relative_to(r) for r in safe_roots)
        or p.is_relative_to(docs) and p.suffix in ('.md','.txt'),'Only named derived artifacts/prose permitted; raw dataset paths blocked')
    P.require(p.suffix not in ('.pth','.pt','.ckpt','.npz','.npy','.safetensors','.pkl','.jpg','.jpeg'), 'Raw/model source prohibited')
    digest=P.sha(p);P.require(expected is None or digest==expected,'Reviewed source SHA differs: '+str(p))
    out=dict(item,sha256=digest)
    if out.get('destination') is not None:
        P.safe_relative(out['destination']);P.require(not out.get('private_input',False) and p.name not in P.PRIVATE_NAMES,'Private payload prohibited')
    # Private scalar error arrays are SHA-only here; numerical QA owns their parsing.
    if not out.get('private_input',False):
        if p.suffix=='.json':P.no_coordinates(json.loads(p.read_text()))
        elif p.suffix=='.csv':
            import csv
            with p.open(newline='') as stream:header=next(csv.reader(stream),[])
            P.require(not(P.FORBIDDEN_KEYS&set(header)),'Coordinate CSV prohibited')
    P.require(P.sha(p)==digest,'Source changed while sealing: '+str(p))
    return out


def validate_gate(path,rule,files):
    P.require(str(path) in files,'Every gate must be an exact source')
    obj=json.loads(Path(path).read_text());P.no_coordinates(obj)
    for key,value in rule['equals'].items():P.require(obj.get(key)==value,'Required gate failed: '+str(path)+'/'+key)


def producer_output_bindings(files,tags):
    base=W80/'combined372/evaluation'/tags['combined']
    folders=(base/'final',base/'supplemental-pose',W80/'combined372/comparisons'/tags['comparison'],
        W80/'combined372/filter_diagnostics'/tags['diagnostic']/'score')
    checked={}
    for folder in folders:
        receipt=folder/'COMPLETE.json';obj=json.loads(receipt.read_text());mapping=obj['output_sha256']
        for path,item in files.items():
            p=Path(path)
            if p!=receipt and p.is_relative_to(folder):
                P.require(mapping.get(path)==item['sha256'],'Producer COMPLETE does not bind exact output: '+path)
                checked[path]={'receipt':str(receipt),'receipt_sha256':files[str(receipt)]['sha256'],'output_sha256':item['sha256']}
    return {'verified_output_sha256':checked,'no_output_map_in_original_producer':[
        str(RATIO/'evaluation'/tags['ratio']/'all/COMPLETE.json'),str(RATIO/'diagnostics'/tags['ratio']/'all/COMPLETE.json')],
        'unmapped_ratio_teacher_outputs':'Exact existing bytes are sealed; scalar numerical QA is additionally required. No producer output-map key invented.'}


def seal(phase):
    P.require(not TARGETS[phase].exists(),'Preserve the existing frozen phase manifest')
    plan=json.loads(PLAN.read_text());plan_sha=P.sha(PLAN)
    P.require(plan.get('authorized') is True and plan.get('action')=='seal_primary372_delivery_inputs'
        and plan['sealer_sha256']==P.sha(SOURCE) and plan['helper_sha256']==P.sha(Path(P.__file__).resolve()),'Reviewed ROOT seal plan missing/different')
    P.require(plan['tags']==FIXED_TAGS,'Use the ROOT-confirmed completed result tags')
    files,gates=fixed_roles(plan['tags'])
    if phase!='numerical':
        previous=P.NUMERICAL_INPUTS if phase=='manuscript' else P.MANUSCRIPT_INPUTS
        inherited=json.loads(previous.read_text());P.require(inherited['schema']=='primary372_delivery_publication_inputs_v1','Unknown inherited contract')
        P.require(plan['inherited_manifest_sha256'][phase]==P.sha(previous),'ROOT inherited phase SHA differs')
        P.require(inherited['tags']==plan['tags'],'Keep the same completed result family across phases')
        for path,item in inherited['source_files'].items():files[path]=dict(item)
        for path,rule in inherited['required_gates'].items():
            P.require(path not in gates or rule==gates[path],'Inherited mandatory gate differs')
        gates.update(inherited['required_gates'])
    # Independent counter/moment audit is mandatory in addition to35/165 row counts.
    audit=plan['diagnostic_counter_qa'];ap=Path(audit['path'])
    required_audit={'passed':True,'complete':True,'population':'PRIMARY372','frames':372,'mismatch_count':0,
        'upstream_equivalence_all_exact':True,'scalar_counter_and_pooled_moment_verification_only':True,
        'error_SD_ddof':0,'verifier_source_sha256':FILTER_VERIFIER_SHA,'upstream_private_coordinate_files_opened_or_rehashed':0}
    P.require(ap==FILTER_QA and all(audit['equals'].get(k)==v for k,v in required_audit.items()),'Exact independent counter/moment QA required')
    audit_item={'role':'independent_filter_counter_QA','destination':'reproducibility/gates/independent_filter_counter_QA.json',
        'private_input':False,'sha256':audit['sha256']}
    audit_gate={'phases':list(STAGES),'equals':audit['equals']}
    P.require(str(ap) not in files or files[str(ap)]==audit_item,'Inherited diagnostic audit source changed')
    P.require(str(ap) not in gates or gates[str(ap)]==audit_gate,'Inherited diagnostic audit gate changed')
    files[str(ap)]=audit_item;gates[str(ap)]=audit_gate
    for entry in plan.get('additional_sources',{}).get(phase,[]):
        path=entry['path'];P.require(path not in files,'Duplicate phase source')
        files[path]={k:v for k,v in entry.items() if k!='path'}
    for path,rule in plan.get('additional_gates',{}).get(phase,{}).items():
        P.require(path in files and isinstance(rule['equals'],dict) and rule['equals'],'Gate requires an exact source and actual schema')
        P.require(set(rule['phases']).issubset(STAGES),'Unknown gate phase')
        P.require(path not in gates or gates[path]==rule,'Mandatory or inherited gate override prohibited');gates[path]=rule
    if phase!='numerical':
        nqa=P.OUT/'preflight/FINAL_NUMERICAL_QA.json'
        P.require(str(nqa) in files,'Carry actual final numerical QA into later phases')
        if str(nqa) in gates:
            P.require(all(gates[str(nqa)]['equals'].get(k) is True for k in ('passed','complete'))
                and set(STAGES).issubset(gates[str(nqa)]['phases']),'Keep mandatory numerical QA gate strength/phases')
        else:gates[str(nqa)]={'phases':list(STAGES),'equals':{'passed':True,'complete':True}}
    if phase=='manuscript':
        roles=[r['role'] for r in files.values()]
        for role in ('manuscript_seed','bundled_render_compat','evaluation_manifest_metadata','methods_origin_evidence','ablation_figure','ratio_figure'):
            P.require(role in roles,'Manuscript source role missing: '+role)
        qual=[r for r in files.values() if r['role']=='qualitative_figure']
        P.require(len(qual)==6 and all(sum(r['kind']==k for r in qual)==2 for k in P.EXPECTED),'Exactly two fixed examples per kind required')
    if phase=='publication':
        P.require(any(Path(p).suffix=='.docx' and r.get('destination') for p,r in files.items())
            and any(Path(p).suffix=='.pdf' and r.get('destination') for p,r in files.items()),'Actual rendered new manuscript DOCX/PDF required')
        for role in ('manuscript_visual_QA','figures_visual_QA'):
            paths=[p for p,r in files.items() if r['role']==role]
            P.require(len(paths)==1 and paths[0] in gates and gates[paths[0]]['equals'].get('passed') is True,'Actual all-page/image visual QA gate required: '+role)
    sealed={p:capture(p,item,item.get('sha256')) for p,item in files.items()}
    for path,rule in gates.items():validate_gate(path,rule,sealed)
    output_bindings=producer_output_bindings(sealed,plan['tags'])
    P.require(sealed[str(P.POLICY_PATH)]['sha256']==P.POLICY_SHA,'Authoritative continuation policy changed')
    for path,digest in plan['source_sha256'].items():P.require(Path(path).parent==P.OUT/'code' and P.sha(path)==digest,'Reviewed publication source changed')
    P.require({str(SOURCE),str(Path(P.__file__).resolve())}.issubset(plan['source_sha256']),'Sealer/helper source freeze required')
    spec={'schema':'primary372_delivery_publication_inputs_v1','primary_frames':372,'primary_reference_points':2891,
        'phase':phase,'tags':plan['tags'],'source_files':sealed,'required_gates':gates,'seal_plan_sha256':plan_sha,'sealer_sha256':P.sha(SOURCE),
        'reviewed_code_sha256':plan['source_sha256'],'producer_output_bindings':output_bindings,'execution_GO_created':False}
    P.require(P.sha(PLAN)==plan_sha and all(P.sha(p)==r['sha256'] for p,r in sealed.items()),'Seal inputs changed')
    P.write(TARGETS[phase],spec)
    print(json.dumps({'complete':True,'phase':phase,'path':str(TARGETS[phase]),'sha256':P.sha(TARGETS[phase]),
        'source_files':len(sealed),'execution_authorized':False},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=tuple(TARGETS));args=parser.parse_args();seal(args.phase)


if __name__=='__main__':main()
