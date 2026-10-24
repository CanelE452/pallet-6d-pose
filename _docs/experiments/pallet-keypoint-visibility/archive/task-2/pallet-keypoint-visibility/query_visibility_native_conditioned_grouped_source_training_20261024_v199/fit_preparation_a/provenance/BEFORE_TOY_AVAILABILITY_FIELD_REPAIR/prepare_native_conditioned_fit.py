"""Freeze code/protocol and CPU invented checks only; never access source bodies."""
from pathlib import Path
import hashlib, importlib.util, json, os, stat, sys
import numpy as np
import torch

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
PYTHON='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python'
SCOPE=V/'ROOT_PROSPECTIVE_NATIVE_CONDITIONING_CONTROLLED_CODE_PREPARATION_SCOPE.json'
SCOPE_SHA='7ccc2ed46c12e69c2f915adf0d6a41c5b75f4fb819bd6f7cb3be1c3bbbf6f92e'
OLD=P/'query_visibility_source_group_split_direct_end_training_20261024_v194/grouped_fit_preparation_b'
PROTOCOL=H/'PROTOCOL_BEFORE_NATIVE_CONDITIONED_SOURCE_CACHE_OR_FIT_BODIES.json'

def safe(path):
    p=Path(path);assert p.is_absolute() and '..' not in p.parts and all('v112'not in x.casefold()for x in p.parts)
    cur=Path(p.anchor)
    for x in p.parts[1:]:
        cur/=x;assert not stat.S_ISLNK(os.lstat(cur).st_mode),cur
    return p

def bind(path):
    p=safe(path);assert p.is_file();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def read(path):return json.loads(safe(path).read_text())

def write(path,value):
    p=Path(path);safe(p.parent)
    with p.open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def run():
    assert Path(sys.executable)==Path(PYTHON) and torch.__version__=='2.1.1+cu118'and np.__version__=='1.26.4'
    assert bind(SCOPE)['sha256']==SCOPE_SHA
    scope=read(SCOPE);old=read(OLD/'PROTOCOL_BEFORE_GROUPED_SOURCE_CACHE_ARRAY_OR_FIT_BODIES.json')
    trainer=bind(H/'train_native_conditioned_grouped_source.py');head=bind(H/'native_conditioned_head.py');test=bind(H/'check_invented_native_conditioning.py')
    spec=importlib.util.spec_from_file_location('native_conditioning_invented_checks',test['path']);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    checks=m.run_checks()
    assert all(checks.values())and not torch.cuda.is_initialized()
    cp=H/'GENERATED_INVENTED_NATIVE_CONDITIONING_CHECKS.json'
    write(cp,dict(completed=True,PASS=True,invented_CPU_only=True,trainer_code=trainer,native_head=head,test_code=test,checks=checks,count=len(checks),CUDA_initialized=False,actual_source_metadata_array_pixel_score_or_weight_reads=0,actual_model_forward_or_fit=0))
    source=dict(old['source_bindings']);source.update(V194_protocol=bind(OLD/'PROTOCOL_BEFORE_GROUPED_SOURCE_CACHE_ARRAY_OR_FIT_BODIES.json'),native_residual_head=head)
    p=dict(completed=True,prepared_only=True,trainer_code=trainer,native_head=head,prepare_code=bind(__file__),test_code=test,checks=bind(cp),prospective_scope=bind(SCOPE),
        inherited_V194_protocol=bind(OLD/'PROTOCOL_BEFORE_GROUPED_SOURCE_CACHE_ARRAY_OR_FIT_BODIES.json'),source_bindings=source,
        mask_adapter=old['mask_adapter'],split_complete=old['split_complete'],partition_QA=old['partition_QA'],training=old['training'],inherited_training=old['training'],trial=scope['trial'],
        old_cache_bindings=old['old_cache_bindings'],new_cache_code=old['new_cache_code'],new_cache_protocol=old['new_cache_protocol'],new_cache_schema=old['new_cache_schema'],direct_cache_expected=old['direct_cache_expected'],
        arms=scope['trial']['arms'],parameters_each=116407,base_parameters=115222,residual_parameters=1185,
        initialization='Original V181+V189 construction first under seed1810; append Linear72->16,SiLU,Linear16->1; final weight/bias exactzero. All original parameter/logit bytes and both arm initial state bytes verified on invented CPU inputs.',
        native_input='Explicit original stored native_slot0..7, losslessly int8->int64 at batch gather. Same five task draw order and direct32 indices. Native validated int64[N] at model call. zero8 vsonehot8 only; no q8,q/L,GT/enclosure/pose/reference/PnP input.',
        capacity_interpretation='Same stored structure and initial outputs, native information supplied or zeroed. Zero8 condition-column gradient is zero; onehot columns are trainable. No claim of equal effective learned capacity or improvement.',
        support_owner='Native residual feeds direct scalar only. Support/owner forward has no direct native injection; shared backbone may train differently via the direct loss.',
        mask_semantics=old['mask_semantics'],direct_join=old['direct_join'],output_schema=old['output_schema'],disclosure=old['disclosure'],
        train_native_prior_census='At actual fit input validation only: direct-known available component TRAIN counts by native and domain; raw E/(D+E) and expected balanced-D16/E16 P(E|domain,native). Expected .5 applies where complete same-native D/E/D triplets yield raw2:1, not finite8000draw realization or owner/public auxiliary priors. Native6/7 no known direct target remains disclosed. Does not change sampler/architecture.',
        train_RAW_identity_census='At actual validation only: all available known direct TRAIN rows (expected327), exact SHA256 of RGB96u8 HWC bytes concatenated valid96bool Cbytes, plus separate (samehash,native) groups. Report D/E conflicts, explicit U/UNKNOWN exclusions; XYconstant/SAMzero. Hash equality additionally checked as byteequality. No deletion, relabel, admission, architecture/data/threshold choice, or independent component/triplet claim.',
        independent_preflight_schema=dict(completed=True,PASS=True,code=trainer,head=head,protocol='binding to this prospective protocol'),
        output_directory=str(V/'runs/C58_C59_NATIVE_CONDITIONED_GROUPED_SOURCE_FIXED8000'),current_private_publicheld_COMMON_VAL_loader=False,
        all2_final8000_before_any_TRAIN_or_HOLDOUT_predictions=True,source_HOLDOUT_prediction_or_selection_before_barrier=False,
        legacy_native_quality_owner_or_target_changes=False,actual_array_header_checkpoint_source_pixel_or_fit_reads=0,actual_fit_requires_separate_root_entry=True,automatic_retry=False,selected_model=None,adopted=False,fully_solved=False)
    write(PROTOCOL,p)
    false=H/'ROOT_ACTUAL_NATIVE_CONDITIONED_FIT_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    write(false,dict(approved_native_conditioned_grouped_source_fit=False,source_only=True,automatic_retry=False,code=trainer,protocol=bind(PROTOCOL),checks=bind(cp),independent_preflight=None,root_scope=None,augmentation_complete=old['old_cache_bindings']['augmentation'],original_public_CF_complete=old['old_cache_bindings']['public_CF'],direct_cache_complete=None,split_complete=old['split_complete'],partition_QA=old['partition_QA'],output_directory=p['output_directory']))
    write(H/'FINAL_READY.json',dict(completed=True,PASS=True,prepared_only=True,trainer_code=trainer,native_head=head,protocol=bind(PROTOCOL),checks=bind(cp),test_code=test,prepare_code=bind(__file__),entry_template=bind(false),scope=bind(SCOPE),output_directory=p['output_directory'],parameters_each=116407,checks_count=len(checks),actual_source_metadata_array_pixel_score_or_weight_reads=0,actual_fit=0,actual_run_requires_separate_root_entry=True,command=[PYTHON,'-B',trainer['path'],'--entry','<ROOT_ACTUAL_ENTRY_ABSOLUTE_PATH>']))
    print(json.dumps(dict(completed=True,PASS=True,FINAL_READY=bind(H/'FINAL_READY.json'),trainer=trainer,head=head,protocol=bind(PROTOCOL),checks=bind(cp),checks_count=len(checks))))

if __name__=='__main__':run()
