"""Read-only final detector review plus generated CPU arithmetic parity."""
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import json
import numpy as np
import torch
import test_generated_independent_grounding as independent

R=Path(__file__).resolve().parent;V=R.parent;CODE=V/'detector/run_requested_grounding.py'
EXPECTED_CODE='6dce5e9e48511760deb3311ec59c678762ea0410c0f86ace7b9e5971fc950fbb'

def binding(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def read(p):return json.loads(Path(p).read_text())

def main():
    assert binding(CODE)['sha256']==EXPECTED_CODE
    source=CODE.read_text();tree=ast.parse(source)
    adapter=independent.no_annotations(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='postprocess_with_trace'))
    adapter.body=[n for n in adapter.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    processor,ns=independent.official_pure_processor()
    exec(compile(ast.fix_missing_locations(ast.Module(body=[adapter],type_ignores=[])),str(CODE),'exec'),ns)
    torch.set_num_threads(1);generator=torch.Generator().manual_seed(20261023)
    cases=0
    for length in [7,77,256]:
        ids=torch.arange(length,dtype=torch.int64)[None,:].repeat(2,1)
        attention=torch.ones((2,length),dtype=torch.int64);attention[1,2]=0
        active=torch.zeros((2,256),dtype=torch.bool);active[:,:length]=attention.bool()
        logits=torch.full((2,12,256),-torch.inf,dtype=torch.float32)
        vals=torch.randn(logits.shape,generator=generator,dtype=torch.float32)
        logits[active[:,None,:].expand_as(logits)]=vals[active[:,None,:].expand_as(vals)]
        for q,p in enumerate([.3,.4,.5,.9]):logits[0,q,1]=torch.logit(torch.tensor(p,dtype=torch.float32))
        boxes=torch.rand((2,12,4),generator=generator,dtype=torch.float32)
        output=SimpleNamespace(logits=logits,pred_boxes=boxes)
        result,probs,maxscores,native=ns[adapter.name](processor,output,ids,[(30,40),(50,60)])
        official=processor.post_process_grounded_object_detection(output,ids,threshold=.4,text_threshold=.3,target_sizes=[(30,40),(50,60)])
        assert torch.equal(probs,logits.sigmoid())
        for a,b in zip(result,official):
            assert torch.equal(a['scores'],b['scores']) and torch.equal(a['boxes'],b['boxes']) and a['text_labels']==b['text_labels']
        for b in range(2):
            packed=probs[b][:,torch.nonzero(active[b]).flatten()]
            assert packed.dtype==torch.float32 and torch.equal(packed.max(-1).values,maxscores[b])
            assert not probs[b][:,~active[b]].any()
        cases+=1
    entry=read(V/'detector/DETECTOR_ENTRYLOCK.json');proof=read(V/'detector/GENERATED_REQUESTED_DETECTOR_TESTS.json');post=read(V/'detector/OFFICIAL_POSTPROCESS_TRACE_CONTRACT.json')
    assert entry['code']==proof['code']==binding(CODE) and proof['PASS'] is True
    for key in ['decisions','logging_addendum','old_helpers','postprocess_contract']:assert entry[key]==proof[key]
    assert entry['source_inputs']['seal']['sha256']=='0bc3ed7a20e976c99792e09caf6ee64968437d7aab9352266c5df2ed6cb81de4'
    assert entry['model_directory']==str(V.parent/'query_visibility_grounded_target_20261023_v81/model_assets')
    assert entry['output_directory']==str(V/'detector/outputs')
    assert entry['processor_use_fast'] is False and entry['parent_must_review_and_explicitly_invoke_run'] is True
    assert post['same_tensor_trace_key']=='raw_active_probabilities' and post['no_monkeypatch'] is True
    # Verify the reviewed order without executing run/freeze or touching assets.
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    body=ast.unparse(run)
    assert body.index("for k in ('generated'")<body.index('GroundingDinoProcessor.from_pretrained')
    assert body.index('old.validate_loading_info(loading)')<body.index('Image.open')
    assert body.index('validate_manifests(')<body.index('GroundingDinoForObjectDetection.from_pretrained')
    assert "set(inputs) <= {'pixel_values', 'pixel_mask', 'input_ids', 'attention_mask', 'token_type_ids'}" in body
    assert 'torch.equal(inputs[k], tokens[cap][k])' in body and 'output = model(**inputs)' in body
    assert "rawprob = probs[0][:, active].cpu().numpy()" in body and "rawlog = output.logits[0][:, active].cpu().numpy()" in body
    assert "old.validate_output_arrays(" in body and 'assert state_before == state_after' in body
    assert body.index('official = proc.post_process_grounded_object_detection')<body.index('np.savez_compressed')
    review=dict(completed=True,PASS=True,blockers=[],code=binding(CODE),entry=binding(V/'detector/DETECTOR_ENTRYLOCK.json'),producer_generated=binding(V/'detector/GENERATED_REQUESTED_DETECTOR_TESTS.json'),postprocess_contract=binding(V/'detector/OFFICIAL_POSTPROCESS_TRACE_CONTRACT.json'),independent_audit_code=binding(__file__),independent_preparation_lock=binding(R/'INDEPENDENT_PREPARATION_LOCK_BEFORE_OUTPUTS.json'),independent_input_audit=binding(R/'INDEPENDENT_INPUT_PACKET_METADATA_AUDIT.json'),criteria=binding(V/'SOURCE_COMPONENT_CRITERIA_BEFORE_FORWARD.json'),logging_addendum=binding(V/'ACTIVE_PROBABILITY_TRACE_ADDENDUM_BEFORE_FORWARD.json'),
        generated_checks=dict(official_AST_adapter_bitexact_batches=cases,active_probability_trace_is_same_sigmoid_tensor=True,allquery_active_maxscores_exact=True,inactive_probability_zero=True,CPU_only=True,model_constructors=0,forward_calls=0),
        code_review=dict(model_forward_only_RGB_and_exact_supplied_category_tokens=True,original_geometry_association_only=True,padding_and_loading4empty_before_source=True,explicit_slow_processor_and_fixed_model_precision=True,allrawquery_active_logit_probability_indices_and_specialtokens_saved=True,strict_dot4_dot3_noNMS_or_text_GT_selector=True,same_V81_inputIoU_dot1_ties_U_reasons=True,firstrepeat_official_postprocessor_and_state_checks=True,entry_code_asset_runtime_metadata_before_model_read=True,no_automatic_current_admission=True),
        entry_integration='Root must bind/check this independent review, actual input audit, and prospective criteria before explicitly invoking run. Frozen producer entry already binds code/proof/protocol/trace/assets/runtime/source packets. Criteria are a report-only screen, not a detector/mask choice.',
        producer_freeze_scope_note='freeze() hashes asset files, including weight file bytes, and explicitly checks the weight SHA. Therefore literal zero weight-file-byte reads is not claimed for producer preparation. Hashing is not a tensor load/model constructor/forward; this independent review did not read weight files or run freeze().',
        limits='This is code/generated preparation, not certification of future actual arrays. Omitted padding is certified by bound successful guard, while saved active probabilities permit independent exact threshold/text replay. Actual detector output audit remains required before SAM.',
        scope=dict(actual_detector_outputs_read=0,RGB_reads=0,GTmask_reads=0,weights_read_by_this_audit=0,model_constructor_or_forward=0,fit=0,current223_reads=0,PnP=0,existing_artifacts_modified=0))
    out=R/'INDEPENDENT_DETECTOR_CODE_REVIEW_BEFORE_FORWARD.json'
    with out.open('x') as f:json.dump(review,f,indent=2);f.write('\n')
    print(json.dumps(dict(PASS=True,path=str(out),sha256=binding(out)['sha256'])))

if __name__=='__main__':main()
