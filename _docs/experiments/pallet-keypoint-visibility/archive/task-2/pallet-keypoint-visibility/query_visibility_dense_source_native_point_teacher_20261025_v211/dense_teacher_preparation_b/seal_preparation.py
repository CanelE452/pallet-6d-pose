"""Seal own prepared code/metadata. No original GT/config/binary filesystem access."""
from pathlib import Path
import ast,hashlib,json,os,stat
H=Path(__file__).parent;V=H.parent
NAMES={'lookup_dense_source_points.py','test_dense_lookup.py','seal_preparation.py','INVENTED_CHECKS.json','SELECTION_METADATA_ONLY.json',
       'GT110_WHITELIST_NOT_OPENED.json','ALL3520_ORIGINAL_OWN_SCENE_NATIVE_SELECTION.jsonl','ALL7040_C61_ELIGIBLE_IMAGE_SOURCE_MAPPING.jsonl',
       'PROTOCOL_BEFORE_DENSE_TRAIN_GT.json','ROOT_ACTUAL_ENTRY_APPROVED_FALSE_TEMPLATE.json','FINAL_READY.json'}
def rb(name):
    assert name in NAMES;p=H/name;assert 'v112'not in str(p).casefold()
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            n=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    assert stat.S_ISREG(os.fstat(n).st_mode)
    with os.fdopen(n,'rb')as f:return f.read()
def bind(name):return dict(path=str(H/name),sha256=hashlib.sha256(rb(name)).hexdigest())
def put(name,value):
    assert name in NAMES
    with(H/name).open('x',encoding='utf-8')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    for name in ('lookup_dense_source_points.py','test_dense_lookup.py','seal_preparation.py'):ast.parse(rb(name))
    selection=json.loads(rb('SELECTION_METADATA_ONLY.json'));whitelist=json.loads(rb('GT110_WHITELIST_NOT_OPENED.json'));checks=json.loads(rb('INVENTED_CHECKS.json'))
    assert checks['completed']and checks['PASS']and checks['invented_only']and checks['count']==17
    assert len(whitelist['GT_json_whitelist'])==110 and selection['counts']==dict(parents=110,scenes=440,own_native=3520,C61_eligible_inputs=7040)
    protocol=dict(completed=True,prospective_before_any_GT_config_or_binary_body=True,
        code=bind('lookup_dense_source_points.py'),scope=selection['metadata_sources']['scope'],selection=bind('SELECTION_METADATA_ONLY.json'),
        whitelist=bind('GT110_WHITELIST_NOT_OPENED.json'),original_native_rows=bind('ALL3520_ORIGINAL_OWN_SCENE_NATIVE_SELECTION.jsonl'),
        input_mapping=bind('ALL7040_C61_ELIGIBLE_IMAGE_SOURCE_MAPPING.jsonl'),checks=bind('INVENTED_CHECKS.json'),test_code=bind('test_dense_lookup.py'),
        counts=selection['counts'],metadata_counts=selection['metadata_counts'],
        selection_rule='All fixed110 component TRAIN bases/all4 original scenes/allnative0..7 regardless old region/direct/status/availability. C61 eligible7040=SELF_SCENE15 NOMINAL 3520 + BEFORE_ANCHOR4 3520; excluded jitter roles unchanged and not relabelled HOLDOUT.',
        GT_boundary='Root actual entry only: exactly110 SCENES_GT_ONLY path+recorded SHA from fixedV210 proposal. Selected110 BASE_COMPLETE receipts and text-source hashes may be verified. No configs/newprojection/rays/GTbestmatching, no NPZ/NPY/RGB/header/weights/model or held/protected reads.',
        parser='Explicit top rows -> unique exact own variant+scene_index+base -> records_CF only. Exactly8 integer CF/canonical bijection and scene perm agree with unchanged source declarations. Missing scene/records/perm or point/reference fields remain UNKNOWN; conflicting identities fail. Never nearest_pixel_diagnostic_CF, records_canonical or state8_CF fallback.',
        states='target_witness_states[0] and combined_witness_states[0] are saved point0 states; state and target_alone_region_state are separate contextual region teacher fields. Do not substitute region for point. Keep D/E/S/U/UNKNOWN separately and report E absence.',
        input_mapping_rule=selection['input_mapping_rule'],
        locality='Python float64 47.5+96*(saved_reference-originalq)/originalL; original frame[-.5,w-.5), sample-center[0,95], crop-canvas[-.5,95.5) separately. Original activeU remains U even if reference/q finite. Mapping locality uses actual input q/L anchor, while point/reference teacher comes from actual image ownscene. No recrop/qcorrection/physical coordinate-error or sampled RGB-visibility claim.',
        future_outputs='dense_source_point_teacher_v1/ALL3520_OWN_SCENE_NATIVE_POINT_TEACHER_METADATA.jsonl;ALL7040_INPUT_SOURCE_POINT_MAPPING.jsonl;SUMMARY.json;COMPLETE.json or preserved FAILURE.json. Metadata-only, no target arrays/admission.',
        actual_counts='Point target/combined/region by ownvariant/native/canonical/material/aspect/geometry/camera/factor/sourcegroup; originalU, mappingUNKNOWN and Eabsence explicitly retained. Do not hardcode observed point counts.',
        authority='Saved procedural SOURCE own-scene geometric point0 only, not physical real/human native gold or automatic training admission. No perfectgold/writer-path gate when exact existing scene identity can be checked.',
        entry_guard='Separate root actual approved_SOURCE_TRAIN_dense_saved_point_lookup true;new_training_admission false;root_actual_scope nonnull+exactSHA;exactcode/proto/selection/whitelist/originalrows/inputmapping;exact110 whitelist and output_directory;automatic_retry false before any GT access.',
        GT_config_arrays_RGB_weights_header_stat_hash_body_reads=0,actual_fit_forward_or_newlabel_admission=False)
    put('PROTOCOL_BEFORE_DENSE_TRAIN_GT.json',protocol)
    expected={name:bind(file)for name,file in [('code','lookup_dense_source_points.py'),('protocol','PROTOCOL_BEFORE_DENSE_TRAIN_GT.json'),('selection','SELECTION_METADATA_ONLY.json'),('whitelist','GT110_WHITELIST_NOT_OPENED.json'),('original_native_rows','ALL3520_ORIGINAL_OWN_SCENE_NATIVE_SELECTION.jsonl'),('input_mapping','ALL7040_C61_ELIGIBLE_IMAGE_SOURCE_MAPPING.jsonl')]}
    put('ROOT_ACTUAL_ENTRY_APPROVED_FALSE_TEMPLATE.json',dict(approved_SOURCE_TRAIN_dense_saved_point_lookup=False,new_training_admission=False,root_actual_scope=None,automatic_retry=False,
        **expected,GT_json_whitelist=whitelist['GT_json_whitelist'],output_directory=str(V/'dense_source_point_teacher_v1')))
    put('FINAL_READY.json',dict(completed=True,PASS=True,prepared_code_metadata_only=True,**expected,scope=protocol['scope'],
        checks=bind('INVENTED_CHECKS.json'),test_code=bind('test_dense_lookup.py'),false_entry=bind('ROOT_ACTUAL_ENTRY_APPROVED_FALSE_TEMPLATE.json'),
        counts=selection['counts'],metadata_counts=selection['metadata_counts'],GT_config_binary_header_reads=0,actual_fit_forward_or_newlabel_admission=False,
        command='/home/minjae/anaconda3/envs/pallet-yolo26/bin/python -B '+str(H/'lookup_dense_source_points.py')+' --entry <absolute root approved entry>'))
    print(json.dumps(dict(completed=True,PASS=True,ready=bind('FINAL_READY.json'),code=expected['code'],protocol=expected['protocol'],selection=expected['selection'])))

if __name__=='__main__':main()
