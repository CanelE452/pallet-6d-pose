"""All-row V82 metadata-only input audit. Does not open RGB/masks/cache/model."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import numpy as np
import independent_saved_grounding_math as m

R=Path(__file__).resolve().parent;V=R.parent;BASE=V.parent/'query_visibility_grounded_target_20261023_v81'
PACK=V/'input_preparation'
EXPECTED_SEAL='0bc3ed7a20e976c99792e09caf6ee64968437d7aab9352266c5df2ed6cb81de4'
READS=[]

def safe(path):
    p=Path(path);assert p.suffix in {'.json','.jsonl','.py'},'Metadata/code only'
    assert 'private_evaluation' not in p.parts and 'current223' not in str(p).lower()
    return p
def sha(p):p=safe(p);READS.append(str(p));return hashlib.sha256(p.read_bytes()).hexdigest()
def binding(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):assert sha(b['path'])==b['sha256'],b['path']
def read(p):p=safe(p);READS.append(str(p));return json.loads(p.read_text())
def lines(p):p=safe(p);READS.append(str(p));return [json.loads(x) for x in p.read_text().splitlines() if x]

def main():
    assert sha(PACK/'SEAL.json')==EXPECTED_SEAL
    seal=read(PACK/'SEAL.json');assert seal['completed'] is True and seal['PASS'] is True
    for b in seal['files']:check(b)
    preplock=read(R/'INDEPENDENT_PREPARATION_LOCK_BEFORE_OUTPUTS.json')
    for b in preplock['files']:check(b)
    proto=read(V/'EXPERIMENT_DECISIONS_BEFORE_INPUTS_AND_FORWARD.json');criteria=read(V/'SOURCE_COMPONENT_CRITERIA_BEFORE_FORWARD.json');trace=read(V/'ACTIVE_PROBABILITY_TRACE_ADDENDUM_BEFORE_FORWARD.json')
    assert sha(V/'EXPERIMENT_DECISIONS_BEFORE_INPUTS_AND_FORWARD.json')==criteria['base_protocol_SHA256']==trace['base_protocol_SHA256']
    policy=read(PACK/'INPUT_BUILD_POLICY_BEFORE_SOURCE_METADATA.json')
    for k in ['base_source_inputs','base_provenance','base_seal','class_metadata']:check(policy[k])
    original=lines(policy['base_source_inputs']['path']);provenance=lines(policy['base_provenance']['path'])
    requested=lines(PACK/'INPUTS_RGB_ONLY.jsonl');unique=lines(PACK/'UNIQUE_RGB_CATEGORY_INPUTS_ONLY.jsonl');links=lines(PACK/'TARGET_TO_REQUEST_INPUT_ONLY.jsonl');classlineage=lines(PACK/'CLASS_SOURCE_PROVENANCE_DO_NOT_FORWARD.jsonl')
    assert len(original)==len(provenance)==len(requested)==len(links)==len(classlineage)==1035
    canonical=proto['fixed_category_map_YCB_obj_ids_1_through_21'];assert len(canonical)==21
    # Only identity/nominal category keys are used from this quarantined metadata.
    # Mask path/area fields may exist in the file but are never opened or used.
    classmeta=lines(policy['class_metadata']['path'])
    expected_unique,expected_map=m.expected_unique(requested)
    assert unique==expected_unique and len(unique)==1035
    assert len(set(r['image_sha256'] for r in requested))==435
    assert len(set(r['input_id'] for r in requested))==1035
    categorycounts=Counter();sourcecounts=Counter();rgbidentity={};groups=Counter()
    wanted_lineage={'input_id','original_id_DO_NOT_FORWARD','original_row_index_DO_NOT_FORWARD','source_DO_NOT_FORWARD','partition','object_id_GT_METADATA_DO_NOT_FORWARD','requested_category','category_origin','class_metadata_evidence'}
    for i,(old,new,prov,link,cl) in enumerate(zip(original,requested,provenance,links,classlineage)):
        caption=m.validate_target(old,new,canonical+['pallet'])
        for key in ['initial_points','roi_side','query_box_xyxy']:
            if old[key] is not None:assert np.asarray(old[key],np.float64).tobytes()==np.asarray(new[key],np.float64).tobytes()
        assert set(cl)==wanted_lineage and cl['input_id']==new['input_id']==prov['input_id']
        assert cl['original_id_DO_NOT_FORWARD']==prov['original_id'] and cl['original_row_index_DO_NOT_FORWARD']==prov['original_row_index']
        assert cl['source_DO_NOT_FORWARD']==prov['source'] and cl['partition']==prov['normalized_partition']=='val'
        assert cl['requested_category']==new['requested_category']
        source=cl['source_DO_NOT_FORWARD'];sourcecounts[source]+=1
        if source=='SYNTHETIC':
            assert new['requested_category']=='pallet' and cl['object_id_GT_METADATA_DO_NOT_FORWARD'] is None
            assert cl['category_origin']=='GLOBAL_DEPLOYMENT_PALLET_CONSTANT' and cl['class_metadata_evidence'] is None
        else:
            assert source=='YCB_PUBLIC_REAL' and cl['category_origin']=='SOURCE_GT_METADATA_PROVIDED_CATEGORY_NOT_PREDICTED'
            obj=cl['object_id_GT_METADATA_DO_NOT_FORWARD'];assert type(obj) is int and 1<=obj<=21
            e=cl['class_metadata_evidence'];assert e['binding']==policy['class_metadata']
            raw=classmeta[e['line_index']];instance=raw['annotated_instances'][e['instance_index']]
            assert instance['input_id']==new['input_id'] and instance['original_id']==prov['original_id']
            assert instance['object_id_GT_diagnostic_only']==obj and new['requested_category']==canonical[obj-1]
            assert raw['image_sha256']==new['image_sha256']
        assert link==dict(input_id=new['input_id'],request_id=expected_map[i],image_sha256=new['image_sha256'],requested_category=new['requested_category'],caption=caption,request_key_sha256=m.request_key_sha256(new))
        key=m.request_tuple(new);groups[key]+=1;categorycounts[(source,new['requested_category'])]+=1
        identity=(new['image'],new['raw_hw'])
        if new['image_sha256'] in rgbidentity:assert rgbidentity[new['image_sha256']]==identity
        rgbidentity[new['image_sha256']]=identity
    assert sourcecounts=={'SYNTHETIC':235,'YCB_PUBLIC_REAL':800}
    duplicates=[(key,n) for key,n in groups.items() if n>1];assert duplicates==[]
    published=read(PACK/'INPUT_BUILD_RESULTS.json')
    cc=[dict(source=s,requested_category=c,N=n) for (s,c),n in sorted(categorycounts.items())]
    assert published['targets']==1035 and published['query_slots']==8280 and published['unique_RGB']==435 and published['unique_RGB_caption_requests']==1035
    assert published['source_target_counts']==dict(sourcecounts) and published['canonical_requested_category_counts']==cc
    assert published['same_class_same_RGB_duplicate_request_groups']==0 and published['same_class_duplicate_target_rows']==0 and published['duplicate_forward_requests_saved_by_dedupe']==0
    result=dict(completed=True,PASS=True,blockers=[],audit_code=binding(__file__),input_seal=binding(PACK/'SEAL.json'),preparation_lock=binding(R/'INDEPENDENT_PREPARATION_LOCK_BEFORE_OUTPUTS.json'),protocol=binding(V/'EXPERIMENT_DECISIONS_BEFORE_INPUTS_AND_FORWARD.json'),criteria=binding(V/'SOURCE_COMPONENT_CRITERIA_BEFORE_FORWARD.json'),active_probability_addendum=binding(V/'ACTIVE_PROBABILITY_TRACE_ADDENDUM_BEFORE_FORWARD.json'),
        counts=dict(targets=1035,query_slots=8280,unique_RGB=435,unique_RGB_caption_requests=1035,source_targets=dict(sourcecounts),category_counts=cc,same_category_same_RGB_duplicate_groups=0,saved_forward_requests_by_deduplication=0),
        invariants=dict(all_original7_fields_exact=True,float64_q8_L_querybox_serialized_bytes_exact=True,canonical_requestedcategory_separately_sourced=True,noGT_or_other_extra_forward_fields=True,first_occurrence_unique_requests_and_all_target_maps_exact=True,request_digest_rawSHA_newline_exact_caption=True,source_split_and_original_identity_preserved=True,all_rows_and_native_order_retained=True),
        category_evidence='Synthetic constant pallet; public canonical objID name rejoined by immutable separate metadata binding, row/instance index and target identity. Cached class metadata was used only for identity/nominal objID; no GT mask body, pose/depth or visibility was used.',
        changed_input_disclosure='YCB category is provided/oracle source metadata, not predicted. Stronger side-information benchmark and not same-input ablation. All1035 unique image/category requests mean same-category instance disambiguation remains untested.',
        limits='Raw image file hashes are copied/consistent with sealed V81 metadata; RGB bodies were not reopened by this audit. No actual detector output or model code readiness is certified here.',
        scope=dict(RGB_body_reads=0,GTmask_pixel_reads=0,pose_depth_or_visibility_used=0,weights_reads=0,model_construction_forward=0,actual_detector_outputs_read=0,current223_reads=0,fit=0,PnP=0,existing_artifacts_modified=0),read_metadata_paths=sorted(set(READS)))
    out=R/'INDEPENDENT_INPUT_PACKET_METADATA_AUDIT.json'
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(PASS=True,path=str(out),sha256=sha(out),counts=result['counts'])))

if __name__=='__main__':main()
