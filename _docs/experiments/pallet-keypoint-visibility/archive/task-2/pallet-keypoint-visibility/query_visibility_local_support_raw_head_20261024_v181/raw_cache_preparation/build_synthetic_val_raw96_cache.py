"""Separate existing synthetic VALIDATION raw/SAM cache; evaluation-only root entry."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
BASE=H/'build_source_raw96_cache.py'
BASE_SHA='01b6be15ad7941384ef5a209b397ef1a0b240dec8b403564e5e54c40a702225f'
assert hashlib.sha256(BASE.read_bytes()).hexdigest()==BASE_SHA
spec=importlib.util.spec_from_file_location('v181_pinned_raw96_cache_pure_helpers',BASE);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
require,read,lines,write,writelines,sha,bind,checked,ash,same=base.require,base.read,base.lines,base.write,base.writelines,base.sha,base.bind,base.checked,base.ash,base.same
SCOPE=base.SCOPE;SOURCE=V/'synthetic_VAL2048_support_v1';S=V/'synthetic_val_preparation'
PLAN=H/'VAL2048_RAW_INPUT_PLAN_BEFORE_CACHE_BODIES.jsonl';PROTOCOL=H/'VAL_PROTOCOL_BEFORE_RAW_SAM_CACHE_BODIES.json'
CHECKS=H/'VAL_GENERATED_INVENTED_CHECKS.json';OUT=V/'synthetic_VAL2048_raw96_cache_v1'

def input_plan():
    rows=lines(S/'VAL2048_ORIGINAL_INPUTS_METADATA_ONLY.jsonl');out=[]
    require(len(rows)==2048 and [r['original_global_index'] for r in rows]==list(range(4096,6144)),'All original VAL2048')
    for i,r in enumerate(rows):
        require(r['split']=='VALIDATION' and r['native_slot']==i%8,'Original VAL/native order')
        out.append(dict(domain='SYNTH_VAL2048',domain_index=i,row_id=r['original_occurrence_id'],original_global_index=r['original_global_index'],native_slot=r['native_slot'],
            query_xy=r['original_query_xy'],L=r['original_L'],original_query_available=r['original_input_flags']['query_available'],split=r['split'],
            raw_packet=r['saved_crop_chunk'],raw_row=r['saved_crop_row'],rgb96_sha256=r['saved_rgb96_sha256_inherited'],valid96_sha256=r['saved_valid96_sha256_inherited'],
            sam_group_id='SYNTH_VAL_SCENE_%06d'%r['scene_index'],raw_hw=r['raw_hw'],original=r,evaluation_only=True,training_admission=False,
            new_native_visibility_target=-1,new_point_quality_target=-1))
    require(sum(not r['original_query_available'] for r in out)==8 and len(set(r['sam_group_id'] for r in out))==256,'VAL/U8/256 frames')
    return out

def prepare():
    require(sha(SCOPE)==base.SCOPE_SHA,'Frozen scope');rows=input_plan();complete=read(SOURCE/'COMPLETE.json')
    require(complete['completed'] and complete['PASS'] and complete['evaluation_only'],'Completed root VAL support dependency')
    checks={}
    toy=[dict(row_id='INVENTED',original_global_index=4096,native_slot=0,query_xy=None,L=None,original_query_available=False)]
    a=base.input_arrays(toy);t=base.target_arrays(toy)
    checks['same_INPUTS_schema_null_U_and_separate_targets']=set(a)=={'rgb96','valid96','sam_mask96','sam_available96','original_query_available','row_id','original_global_index','native_slot','query_xy','L'} and np.isnan(a['query_xy']).all() and not a['original_query_available'].any() and set(t)=={'support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass'}
    owner5=np.array([[.25,.125,.125,.25,.25],[np.nan]*5],np.float64);soft=owner5[:,:3].copy();amb=owner5[:,3].copy();invalid=owner5[:,4].copy()
    checks['five_owner_classes_preserved_without_renormalization']=np.array_equal(soft[0],[.25,.125,.125]) and soft[0].sum()==.5 and amb[0]==.25 and invalid[0]==.25 and np.isnan(soft[1]).all()
    checks['metadata_only_all2048_VAL_native8_U8']=len(rows)==2048 and all(r['split']=='VALIDATION' for r in rows) and all(sum(r['native_slot']==k for r in rows)==256 for k in range(8))
    require(all(checks.values()),'VAL invented checks');writelines(PLAN,rows)
    write(CHECKS,dict(completed=True,PASS=True,invented_only=True,checks={k:bool(v) for k,v in checks.items()},number_checks=len(checks),code=bind(__file__),base=bind(BASE),actual_cache_raw_SAM_or_target_payload_reads=0))
    p=dict(completed=True,prepared_only=True,code=bind(__file__),base=bind(BASE),scope=bind(SCOPE),checks=bind(CHECKS),plan=bind(PLAN),
        original_input_manifest=bind(S/'VAL2048_ORIGINAL_INPUTS_METADATA_ONLY.jsonl'),support_complete=bind(SOURCE/'COMPLETE.json'),support_sidecar=complete['sidecar'],
        independent_support_audit=bind(V/'independent_synthetic_val_support_review_a/REVIEW.json'),source_SAM_complete=base.bind(base.V180/'frozen_sam_allTRAIN4096_predictions_v2/COMPLETE.json'),
        domain='SYNTH_VAL2048',rows=2048,scenes=256,original_inputU8=True,original_split='VALIDATION',cache_contract_schema='V181_LOCAL96_PIXEL_CACHE_WITH_SEPARATE_TARGETS_v1',
        input_schema='Exactly same INPUTS schema as TRAIN A cache; original saved uint8RGB96/valid96 +samegrid frozenSAM>0 and availability. No GT, originalqL/native/inputflags immutable.',
        targets_schema='Evaluation-only exact5keys: support_targeti8[N,2],owner_softf64[N,3](T/F/BG rawmass),owner_targetall-1,owner_ambiguous_mass/owner_invalid_massf64[N]. Original saved VAL owner5 retained as3+1+1 without renormalization; no fit.',
        actual_dependency='Root entry must bind C completed VAL SAM seal with all256 originalscenes. No missing-mask substitution. Cache target bodies never used by trainer before both heads freeze.',
        evaluation_only=True,training_admission=False,output=str(OUT),actual_cache_or_model_forward_fit=0,selected_model=None,fully_solved=False)
    write(PROTOCOL,p);write(H/'ROOT_VAL_CACHE_ENTRY_TEMPLATE_APPROVED_FALSE.json',dict(approved_VAL_raw_cache=False,evaluation_only=True,scope=p['scope'],code=p['code'],base=p['base'],protocol=bind(PROTOCOL),VAL_SAM_complete=None,automatic_retry=False))
    write(H/'VAL_PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=p['code'],base=p['base'],protocol=bind(PROTOCOL),checks=p['checks'],plan=p['plan'],evaluation_only=True,pending_VAL_SAM_complete=True,actual_raw_SAM_or_label_cache_body_reads=0))

def run(entry):
    e=read(entry);require(e.get('approved_VAL_raw_cache') is True and e.get('evaluation_only') is True and e.get('automatic_retry') is False,'Separate approved VAL cache entry')
    for k,path in [('code',Path(__file__)),('base',BASE),('scope',SCOPE),('protocol',PROTOCOL)]:require(e[k]==bind(path),'VAL entry '+k)
    p=read(PROTOCOL);require(p['code']==bind(__file__) and p['base']==bind(BASE) and p['checks']==bind(CHECKS),'Frozen VAL prep')
    rows=lines(checked(p['plan']));require(len(rows)==2048 and all(r['split']=='VALIDATION' for r in rows),'VAL only')
    require(not OUT.exists(),'No overwrite or automatic retry');OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),protocol=bind(PROTOCOL),evaluation_only=True))
    start=time.monotonic();used={}
    try:
        seal=read(checked(e['VAL_SAM_complete']));source=read(checked(p['source_SAM_complete']))
        require(seal['completed'] and seal['PASS'] and seal['model_state_byte_unchanged'],'Completed frozen VAL SAM')
        require(seal['checkpoint']==source['checkpoint'] and seal['source_commit']==source['source_commit'],'Same frozen official SAM model')
        groups={s['group_id']:s for s in seal['scenes']};require(len(groups)==256 and set(groups)=={r['sam_group_id'] for r in rows},'All256 original VAL groups')
        x=base.input_arrays(rows);y=base.target_arrays(rows)
        with np.load(checked(p['support_sidecar']),allow_pickle=False) as z:
            require(z['evaluation_only'].item() and not z['training_admission'].item(),'VAL labels evaluation-only')
            for key,dst in [('original_global_index','original_global_index'),('original_occurrence_id','row_id'),('native_slot','native_slot'),('query_xy','query_xy'),('L','L'),('query_available','original_query_available')]:
                require(same(z[key],x[dst]),'VAL label original metadata '+key)
            require((z['split'].astype(str)=='VALIDATION').all(),'Original VAL split')
            for flag in rows[0]['original']['original_input_flags']:
                require(same(z[flag],np.array([r['original']['original_input_flags'][flag] for r in rows],bool)),'VAL original flag '+flag)
            owner5=z['owner_central4_soft5'].copy();require(owner5.dtype==np.float64 and owner5.shape==(2048,5),'Original VAL owner5')
            y.update(support_target=np.column_stack((z['visible_support_target'],z['foreign_front_support_target'])),owner_soft=owner5[:,:3].copy(),owner_ambiguous_mass=owner5[:,3].copy(),owner_invalid_mass=owner5[:,4].copy())
        require(y['support_target'].dtype==np.int8 and (y['support_target'][~x['original_query_available']]==-1).all(),'VAL U unchanged')
        raw_path=None;raw=None
        for startrow in range(0,2048,8):
            rr=rows[startrow:startrow+8];s=groups[rr[0]['sam_group_id']];logits,available=base.load_sam_group(s,rr,used)
            for j,r in enumerate(rr,startrow):
                path=Path(r['raw_packet']['path'])
                if path!=raw_path:
                    checked(r['raw_packet']);used[str(path)]=r['raw_packet']
                    with np.load(path,allow_pickle=False) as z:raw={k:z[k].copy() for k in ['rgb96','valid96','crop_constructed','global_index']}
                    base.validate_raw(raw);raw_path=path
                k=r['raw_row'];rgb=raw['rgb96'][k];valid=raw['valid96'][k]
                require(int(raw['global_index'][k])==r['original_global_index'] and bool(raw['crop_constructed'][k])==r['original']['original_input_flags']['crop_constructed'],'VAL original raw row')
                require(ash(rgb)==r['rgb96_sha256'] and ash(valid)==r['valid96_sha256'],'VAL original raw/valid hashes')
                x['rgb96'][j]=rgb;x['valid96'][j]=valid
                m,a=base.sample_sam(logits,r['query_xy'],r['L'],available and r['original_query_available'],r['raw_hw'])
                if available and r['original_query_available']:require(same(a,valid),'VAL original SAM sampledomain')
                x['sam_mask96'][j]=m;x['sam_available96'][j]=a;r['SAM_output']=s['outputs'];r['sam_prompt_available']=s['prompt_available'];r['sam_runtime_available']=s['runtime_available']
                r['actual_raw_RGB96_sha256']=ash(rgb);r['actual_raw_valid96_sha256']=ash(valid)
        ib=base.save_npz(OUT/'SYNTH_VAL2048_INPUTS.npz',x);tb=base.save_npz(OUT/'SYNTH_VAL2048_TARGETS_EVALUATION_ONLY.npz',y)
        mp=OUT/'SYNTH_VAL2048_ROWS.jsonl';writelines(mp,rows)
        for b in used.values():checked(b)
        checked(e['VAL_SAM_complete']);checked(p['support_sidecar']);require(bind(PLAN)==p['plan'],'VAL plan unchanged')
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,evaluation_only=True,training_admission=False,
            cache_contract_schema='V181_LOCAL96_PIXEL_CACHE_WITH_SEPARATE_TARGETS_v1',code=p['code'],base=p['base'],protocol=bind(PROTOCOL),entry=bind(entry),VAL_SAM_complete=e['VAL_SAM_complete'],
            support_complete=p['support_complete'],independent_support_audit=p['independent_support_audit'],domains={'SYNTH_VAL2048':dict(rows=2048,inputs=ib,targets=tb,metadata=bind(mp))},
            original_inputU8=True,all_original_qL_native_split_flags_preserved=True,all_used_source_SHA_unchanged=True,no_GT_modelinput=True,new_model_forward_fit_or_selection=0,
            selected_model=None,fully_solved=False,automatic_retry=False,seconds=time.monotonic()-start))
    except BaseException:
        write(OUT/'FAILED_EXECUTION_PRESERVED.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);args=ap.parse_args()
    prepare() if args.prepare else run(args.entry)
