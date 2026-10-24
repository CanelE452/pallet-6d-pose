"""Generated metadata gate/output schema tests. Never opens actual current data."""
from pathlib import Path
from tempfile import TemporaryDirectory
import ast,copy,csv,hashlib,importlib.util,io,json,sys
import numpy as np

REAL_ROOT=Path(__file__).resolve().parents[1]
CODE=REAL_ROOT/'inference/infer_cached_current.py'
EVAL=REAL_ROOT/'contract/evaluate_sealed_current.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def main():
    checked=[];input_hashes=[]
    with TemporaryDirectory(prefix='cached_private_MOCK_',dir=REAL_ROOT/'inference')as tmp:
        root=Path(tmp);(root/'inference').mkdir();copied=root/'inference/infer_cached_current.py';copied.write_bytes(CODE.read_bytes())
        infer=module(copied,'v66_generated_gate_test')
        model=REAL_ROOT/'model/query_global_amodal_head.py';trainer=REAL_ROOT/'model/fit_partial_head.py'
        proto={'updates':2000,'validation_every':100,'seed':20261022,'validation_sources':['SYNTH','ORIGINAL_YCB','CONTROLLED_YCB','PALLOX'],
          'candidates':{'CONTROLLED_ONLY':{'training_sources':['SYNTH','ORIGINAL_YCB','CONTROLLED_YCB']},'CONTROLLED_PLUS_PALLOX':{'training_sources':['SYNTH','ORIGINAL_YCB','CONTROLLED_YCB','PALLOX']}},'bindings':[ref(model),ref(trainer)]}
        protocol=root/'PROTOCOL.json';write(protocol,proto)
        for name in ['TRAIN_SOURCE_MOCK','COMMON_VAL_SOURCE_MOCK','DENYLIST_MOCK']:(root/name).write_text('GENERATED_SOURCE_METADATA_ONLY')
        lock={'bindings':[ref(protocol),ref(model),ref(trainer)],'trainer_sha256':sha(trainer),'model_sha256':sha(model),'runs':{}}
        for arm in infer.ARMS:
            out=root/'runs'/arm;out.mkdir(parents=True);(out/'BEST_PARTIAL_HEAD.pt').write_text('MOCK_CHECKPOINT_NOT_A_MODEL:'+arm)
            lock['runs'][arm]={**{k:ref(root/name)for k,name in [('train','TRAIN_SOURCE_MOCK'),('val','COMMON_VAL_SOURCE_MOCK'),('denylist','DENYLIST_MOCK')]},'protocol':ref(protocol),'out':str(out)}
        lockpath=root/'FIT_INPUT_LOCK.json';write(lockpath,lock);candidates=[]
        for arm,nll in zip(infer.ARMS,[.8,.7]):
            out=root/'runs'/arm;cp=ref(out/'BEST_PARTIAL_HEAD.pt')
            freeze={'completed':True,'eval_data_used':False,'PnP_used':False,'optimizer_updates':2000,'threshold':.5,
              'checkpoint_sha256':cp['sha256'],'protocol_sha256':sha(protocol),'input_lock_sha256':sha(lockpath),
              'code':{'fit_partial_head.py':sha(trainer),'query_global_amodal_head.py':sha(model)},
              'train_sha256':lock['runs'][arm]['train']['sha256'],'val_sha256':lock['runs'][arm]['val']['sha256'],
              'validation_sources':sorted(proto['validation_sources']),'fitting_sources':sorted(proto['candidates'][arm]['training_sources']),
              'best_source_partial_NLL':nll}
            write(out/'MODEL_FREEZE.json',freeze);candidates.append(dict(candidate=arm,checkpoint=cp,freeze=ref(out/'MODEL_FREEZE.json')))
        audit=root/'SOURCE_AUDIT.json';write(audit,dict(PASS=True,generated_fixture=True))
        selection={'completed':True,'primary':infer.ARMS[1],'current_eval_used_for_selection':False,'thresholds':infer.THRESHOLDS,
          'fit_input_lock':ref(lockpath),'fit_protocol':ref(protocol),'candidates':candidates,'source_audits':{'generated_mock':ref(audit),'selection_code':ref(trainer)}}
        sealpath=root/'SOURCE_SELECTION_SEAL.json';write(sealpath,selection)
        cached=root/'MOCK_CACHED_INPUTS';cached.mkdir()
        for name in ['REAL_V65_INPUTS.npz','REAL_QUERY_METADATA.json','REAL_ROI_CACHE.npz']:(cached/name).write_text('MOCK_PRIVATE_FILE_MUST_NOT_BE_OPENED_BY_VALIDATE')
        private=dict(v65_inputs=ref(cached/'REAL_V65_INPUTS.npz'),query_metadata=ref(cached/'REAL_QUERY_METADATA.json'),roi_cache=ref(cached/'REAL_ROI_CACHE.npz'))
        prep={'completed':True,'candidate':'MIXED_GLOBAL_UNWEIGHTED','frames':223,'slots':1784,
          'private_ref_or_human_GT_read':False,'performance_tables_read':False,'threshold_or_model_selection':False,
          'optimizer_updates':0,'PnP_calls':0,'SAM_exact_native9_replay_mismatches':0,'original_query_coordinates_preserved':True,
          'geometry_self_overrides_learned_probability':False,'SAM_state_unchanged':True,
          'source_selection_seal':{'path':'MOCK_HISTORICAL_SOURCE_METADATA','sha256':infer.V65_SELECTION_SHA},
          'outputs':[private['v65_inputs'],private['query_metadata']],'input_artifacts':{'roi_cache':private['roi_cache']}}
        prepfile=root/'PREPARATION_SEAL.json';write(prepfile,prep)
        pure=REAL_ROOT.parent/'query_visibility_amodal_global_20261022_v65/inference/infer_current_frozen.py'
        out=root/'private_evaluation'/infer.ARMS[1]
        gate={'schema':'V66_CACHED_PRIVATE_EVAL_GATE_V1','approved':True,'source_selection_complete':True,'allow_current_evaluation_inference':True,'current_eval_used_for_selection':False,
          'source_selection_seal':ref(sealpath),'fit_input_lock':ref(lockpath),'fit_protocol':ref(protocol),'thresholds':infer.THRESHOLDS,'source_audits':selection['source_audits'],
          'approved_candidates':[{'name':infer.ARMS[1],'checkpoint':candidates[1]['checkpoint'],'freeze':candidates[1]['freeze'],'model_code':ref(model),'trainer_code':ref(trainer),
             'inference_code_sha256':sha(copied),'output_directory':str(out)}],
          'v65_forward_helper':ref(pure),'v65_preparation_seal':ref(prepfile),'private_inputs':private,
          'evaluation_code':{'wrapper':ref(EVAL)},'expected':{'frames':223,'slots':1784},
          'runtime':{'head_device':'cpu','head_torch':'2.5.1+cu121','head_executable':sys.executable}}
        gatepath=root/'GATE.json';write(gatepath,gate)
        actual_sha=infer.sha
        def logged_sha(p):input_hashes.append(str(Path(p).resolve()));return actual_sha(p)
        infer.sha=logged_sha
        old_np_load=np.load
        def forbid_array_load(*a,**kw):raise AssertionError('Mock gate attempted array read')
        np.load=forbid_array_load
        try:
            infer.validate_gate(gatepath);checked.append('approved-source-metadata-only')
            privatepaths={str(Path(b['path']).resolve())for b in private.values()}
            assert not privatepaths&set(input_hashes)
            def reject(name,mutate):
                bad=copy.deepcopy(gate);mutate(bad);write(gatepath,bad)
                try:infer.validate_gate(gatepath)
                except (PermissionError,ValueError,KeyError,FileNotFoundError):checked.append(name)
                else:raise AssertionError('Expected gate rejection '+name)
                assert not privatepaths&set(input_hashes),name
                write(gatepath,gate)
            reject('unapproved',lambda g:g.update(approved=False))
            reject('selection-current-leak',lambda g:g.update(current_eval_used_for_selection=True))
            reject('threshold-change',lambda g:g['thresholds'].update(binary_visible=.6))
            reject('source-seal-SHA',lambda g:g['source_selection_seal'].update(sha256='0'*64))
            reject('nonwinner-candidate',lambda g:g['approved_candidates'][0].update(name=infer.ARMS[0]))
            reject('checkpoint-change',lambda g:g['approved_candidates'][0]['checkpoint'].update(sha256='0'*64))
            reject('model-code-change',lambda g:g['approved_candidates'][0]['model_code'].update(sha256='0'*64))
            reject('trainer-code-change',lambda g:g['approved_candidates'][0]['trainer_code'].update(sha256='0'*64))
            reject('inference-code-change',lambda g:g['approved_candidates'][0].update(inference_code_sha256='0'*64))
            reject('missing-audits',lambda g:g.update(source_audits={}))
            reject('unsealed-cached-input',lambda g:g['private_inputs']['v65_inputs'].update(sha256='0'*64))
            reject('GPU-device',lambda g:g['runtime'].update(head_device='cuda'))
            reject('outside-new-V66-output',lambda g:g['approved_candidates'][0].update(output_directory=str(root/'WRONG')))
            badselection=copy.deepcopy(selection);badselection['primary']=infer.ARMS[0];write(sealpath,badselection)
            bad=copy.deepcopy(gate);bad['source_selection_seal']=ref(sealpath);write(gatepath,bad)
            try:infer.validate_gate(gatepath)
            except PermissionError:checked.append('source-primary-not-GPU-NLL-minimum')
            else:raise AssertionError('Nonminimum source primary accepted')
            write(sealpath,selection);write(gatepath,gate)
        finally:np.load=old_np_load
        # Only generated point/feature arrays; compile the same source functions.
        funcs=infer.pure_v65_functions(ref(pure));tree=ast.parse(pure.read_text())
        extra=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in {'output_rows','self_flags','query_validity'}]
        env=dict(np=np);exec(compile(ast.Module(body=extra,type_ignores=[]),str(pure),'exec'),env)
        geometry=type('UnusedGeometry',(),{})();records=[];rng=np.random.default_rng(20261022);schema=None
        for frame in range(223):
            f={'id':'GENERATED:'+str(frame),'initial_points':np.array([[10+5*k,20+3*k]for k in range(8)]).tolist(),
              'raw_hw':[480,640],'image_sha256':'a'*64,'initial_pose':{'available':False}}
            rows,_=env['output_rows'](f,{'side':32.},np.ones(8,bool),np.ones((8,65,9,9)),geometry)
            for k,row in enumerate(rows):
                if frame*8+k<45:row.update(valid=False,status='UNKNOWN',reason='GENERATED_INVALID')
            original=[{k:r[k]for k in ['query_x','query_y','native_corner','valid','geometry_self_hidden']}for r in rows]
            infer.fill_predictions(rows,rng.normal(size=(8,6)),funcs['probabilities'])
            for row,old in zip(rows,original):
                assert all(row[k]==v for k,v in old.items())
                if not row['valid']:assert row['p_visible']is None and row['logit']is None
                schema=schema or list(row);assert list(row)==schema
            records.extend(rows)
        tie=funcs['probabilities'](np.array([[0.,0.,-1000.,-1000.,-1000.,-1000.]]))
        assert tie['visible'][0,0]==.5 and np.array_equal(tie['confident'][0],[0,0])
        stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=schema);w.writeheader();w.writerows(records);stream.seek(0)
        sys.path.insert(0,str(REAL_ROOT/'contract'));evaluator=module(EVAL,'v66_saved_evaluator_generated_mock');score_audit=evaluator.validate_scores(list(csv.DictReader(stream)))
        assert score_audit==dict(rows=1784,valid=1739,invalid=45)
        # Frozen helper forward shape/plumbing with a generated callback, no
        # real architecture/weights load or trainable model instantiation.
        calls=[]
        def fake_forward(**kw):
            import torch
            assert kw['x'].shape==(8,65,9,9)and kw['global_input'].shape==(1,5,64,64)
            assert kw['global_index64'].dtype==torch.long and kw['global_query_valid'].dtype==torch.bool
            calls.append(1);return torch.zeros(8,6)
        features={k:np.zeros(shape,dtype)for k,shape,dtype in [('rgb_native',(8,75,9,9),np.uint8),('sam',(8,2,9,9),np.float16),('box_grid',(8,4,9,9),np.float32),('global_rgb',(1,3,64,64),np.uint8),('global_sam',(1,1,64,64),np.float16),('global_prompt',(1,1,64,64),bool),('global_index64',(8,9,9),np.int64),('global_index8',(8,9,9),np.int64),('global_query_valid',(8,9,9),bool)]}
        mocklogits=funcs['forward_frame'](fake_forward,np.zeros((8,65,9,9)),np.zeros(64),features,'cpu');assert mocklogits.shape==(8,6)and len(calls)==1
    report=dict(schema='V66_CACHED_PRIVATE_WRAPPER_GENERATED_MOCK_PROOF',PASS=True,
      inference_code_sha256=sha(CODE),evaluation_code_sha256=sha(EVAL),gate_tests=checked,
      source_gate_current_input_hash_or_NPZ_reads=0,generated_CSV_score_audit=score_audit,V65_output_row_fields_unchanged=True,
      invalid_main_predictions_blank=True,geometry_and_query_flags_preserved=True,confidence_exact_point5_tie_accepted=True,
      frozen_forward_generated_callback_shape_PASS=True,actual_source_or_private_input_RGB_GT_scores_reads=0,
      actual_checkpoint_loads_model_instantiations_optimizer_updates_SAM_P3_PnP_GPU_calls=0)
    out=REAL_ROOT/'inference/CACHED_PRIVATE_WRAPPER_MOCK_PROOF.json';out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
