"""Root-entry-only, TRAIN-only fixed-end supervision generation. No models/RGB IO."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import sys
import time
import traceback
sys.dont_write_bytecode=True
import numpy as np
import source_end_support as op

H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
DEFINITION=H/'PROTOCOL_BEFORE_NEW_SOURCE_END_PIXEL_BODIES.json'
PLAN=H/'TRAIN4096_END_SUPERVISION_INPUT_ONLY_PLAN.jsonl'
GEOMETRY=H/'TRAIN1024_FIXED_END_PATCH_GEOMETRY_METADATA.jsonl'
PROTOCOL=H/'RUNNER_PROTOCOL_BEFORE_SOURCE_END_PIXEL_BODIES.json'
TESTS=H/'GENERATED_RUNNER_CHECKS.json'
PREP=H/'RUNNER_PREPARATION_COMPLETE.json'
OUT=V/'source_end_supervision_train4096_v1'
SOURCE=P/'query_visibility_registered_native_diversity_20261023_v129/actual_corpus_v1'
FIELDS=('part_id','camera_depth','global_triangle_id','ambiguity','camera_K','camera_R','camera_t','native8_CF_xy','native8_canonical_xy','permCF_to_canonical','state8_CF')
FLAGS=('query_available','factual_query_available','feature_constructed','crop_constructed','context_available','Q_inside','RGB_central4_available','native_continuous_valid','native_nearest_valid')
def sha(p):
    p=Path(p)
    assert not any(s in str(p).lower() for s in ('_v112','_v177','_v178','_v169','/private_evaluation/'))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def checkbind(b):assert bind(b['path'])==b;return Path(b['path'])
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def metadata_arrays(rows):
    """One serializer for invented checks and actual4096 output; nulls stay explicit."""
    a={k:np.array([r[k] for r in rows],np.int32) for k in ('original_global_index','scene_index','base_index')}
    a.update({k:np.array([r[k] for r in rows],np.int8) for k in ('native_slot','declared_canonical_index')})
    a.update({k:np.array([r[k] for r in rows]) for k in ('original_occurrence_id','split','variant','original_semantic_state')})
    a.update(query_xy=np.asarray([[np.nan,np.nan] if r['query_xy'] is None else r['query_xy'] for r in rows],np.float64),query_xy_missing=np.array([r['query_xy'] is None for r in rows],bool),
             L=np.asarray([np.nan if r['L'] is None else r['L'] for r in rows],np.float64),L_missing=np.array([r['L'] is None for r in rows],bool),
             selected_index=np.array([-1 if r['selected_index'] is None else r['selected_index'] for r in rows],np.int32),selected_index_missing=np.array([r['selected_index'] is None for r in rows],bool),
             nominal_reference_xy=np.asarray([r['nominal_reference_xy'] for r in rows],np.float64),correspondence_boundary=np.array([r['correspondence_boundary'] for r in rows],bool),primary_part_id=np.array([r['primary_part_id'] for r in rows],np.int32))
    a.update({k:np.array([r['original_input_flags'][k] for r in rows],bool) for k in FLAGS})
    a.update({k:np.array([r['factors'][k] for r in rows]) for k in ('material','aspect','geometry_family','foreign_family','signed_quarter')})
    for key in ('original_query_metadata','original_input_flags','factors','incident_part_ids','active_witness_indices'):
        a[key+'_json']=np.array([json.dumps(r[key],sort_keys=True,separators=(',',':'),allow_nan=False) for r in rows])
    return a


def output_arrays(rows,outputs):
    assert len(rows)==len(outputs)
    a=metadata_arrays(rows)
    a['jointmass7']=np.asarray([[np.nan]*7 if r['full96_average_mass'] is None else r['full96_average_mass'] for r in outputs],np.float64)
    a['jointmass7_missing']=np.isnan(a['jointmass7']).all(1)
    for name in ('visible_support_target','foreign_front_support_target'):
        a[name]=np.array([r[name] for r in outputs],np.int8)
        assert np.isin(a[name],[-1,0,1]).all()
    a.update(source_end_class_order=np.array(op.CLASSES),new_native_visibility_target=np.full(len(rows),-1,np.int8),new_point_quality_target=np.full(len(rows),-1,np.int8),training_admission=np.array(False),reasons_json=np.array([json.dumps(r['reasons']) for r in outputs]))
    assert a['jointmass7'].shape==(len(rows),7)
    valid=~a['jointmass7_missing']
    assert np.isfinite(a['jointmass7'][valid]).all() and np.allclose(a['jointmass7'][valid].sum(1),1,rtol=0,atol=2e-13)
    return a


def load_scene(binding,records,geometry):
    path=Path(binding['path']).resolve()
    assert path.is_relative_to(SOURCE) and path.suffix=='.npz'
    assert sha(path)==binding['sha256']
    with np.load(path,allow_pickle=False) as z:
        a={k:z[k].copy() for k in FIELDS}
    expected={
        'part_id':(np.int32,(512,512)),'camera_depth':(np.float64,(512,512)),
        'global_triangle_id':(np.int32,(512,512)),'ambiguity':(bool,(512,512)),
        'camera_K':(np.float64,(3,3)),'camera_R':(np.float64,(3,3)),'camera_t':(np.float64,(3,)),
        'native8_CF_xy':(np.float64,(8,2)),'native8_canonical_xy':(np.float64,(8,2)),
        'permCF_to_canonical':(np.int32,(8,))}
    for k,(dtype,shape) in expected.items():assert a[k].dtype==dtype and a[k].shape==shape,k
    assert a['state8_CF'].dtype.kind=='U' and a['state8_CF'].dtype.itemsize==4 and a['state8_CF'].shape==(8,)
    assert [r['native_slot'] for r in records]==list(range(8)) and all(r['split']=='TRAIN' for r in records)
    assert a['state8_CF'].tolist()==[r['original_semantic_state'] for r in records]
    op.validate_buffer_metadata(a,geometry['camera'],geometry['native8_CF_xy'],geometry['permCF_to_canonical'])
    assert np.array_equal(a['native8_canonical_xy'][a['permCF_to_canonical']],a['native8_CF_xy'])
    return a


def summarize_rows(rows,outputs):
    result=[]
    selectors=[('ALL',list(range(len(rows))))]
    for key in ('native_slot','variant','original_semantic_state'):
        for value in sorted({r[key] for r in rows}):selectors.append((key+'::'+str(value),[i for i,r in enumerate(rows) if r[key]==value]))
    for mat,asp in sorted({(r['factors']['material'],r['factors']['aspect']) for r in rows}):selectors.append(('material_aspect::'+mat+'::'+asp,[i for i,r in enumerate(rows) if r['factors']['material']==mat and r['factors']['aspect']==asp]))
    for name,ii in selectors:
        known=[outputs[i]['full96_average_mass'] for i in ii if outputs[i]['full96_average_mass'] is not None]
        result.append(dict(group=name,N=len(ii),original_inputU=sum(not rows[i]['original_input_flags']['query_available'] for i in ii),missing_mass=len(ii)-len(known),sum_jointmass7=np.asarray(known,np.float64).sum(0).tolist() if known else [0.]*7,
            visible_counts={str(k):sum(outputs[i]['visible_support_target']==k for i in ii) for k in (-1,0,1)},foreign_counts={str(k):sum(outputs[i]['foreign_front_support_target']==k for i in ii) for k in (-1,0,1)},reasons=dict(Counter(reason for i in ii for reason in outputs[i]['reasons']))))
    return result


def invented():
    rows=[];outputs=[]
    for i in range(8):
        missing=i<2;rows.append(dict(original_global_index=i,scene_index=0,base_index=0,native_slot=i,declared_canonical_index=i,original_occurrence_id=f'INVENTED:native{i}',split='TRAIN',variant='BEFORE',original_semantic_state='U' if missing else 'D',query_xy=None if missing else [float(i),float(i+1)],L=None if missing else 20.,selected_index=None if missing else 0,nominal_reference_xy=[float(i),float(i)],correspondence_boundary=False,primary_part_id=0,original_input_flags={k:not missing for k in FLAGS},factors=dict(material='INVENTED',aspect='SQUARE',geometry_family='INVENTED',foreign_family='INVENTED',signed_quarter=0),original_query_metadata=dict(query_xy=None if missing else [float(i),float(i+1)]),incident_part_ids=[0],active_witness_indices=[0,1]))
        mass=None if missing else [0.,.5,.5,0,0,0,0]
        outputs.append(dict(full96_average_mass=mass,visible_support_target=-1 if missing else 1,foreign_front_support_target=-1 if missing else 1,reasons=['INVENTED']))
    arrays=output_arrays(rows,outputs)
    checks={
        'actual_serializer_null48_style_explicit_NaN_minus1_flags':arrays['query_xy'].shape==(8,2) and np.isnan(arrays['query_xy'][:2]).all() and arrays['selected_index'][:2].tolist()==[-1,-1] and arrays['query_xy_missing'].tolist()==[True,True,False,False,False,False,False,False],
        'actual_serializer_mass7_targets_and_native_quality_minus1':arrays['jointmass7'].shape==(8,7) and arrays['jointmass7_missing'].sum()==2 and arrays['visible_support_target'].tolist()==[-1,-1,1,1,1,1,1,1] and (arrays['new_native_visibility_target']==-1).all() and (arrays['new_point_quality_target']==-1).all() and not arrays['training_admission'].item(),
        'actual_serializer_original_native_semantic_and_all9_flags':arrays['original_semantic_state'].tolist()==['U','U','D','D','D','D','D','D'] and all(arrays[k].sum()==6 for k in FLAGS),
    }
    path=H/'INVENTED_ONLY_SERIALIZATION_CHECK.npz';np.savez_compressed(path,**arrays)
    with np.load(path,allow_pickle=False) as z:checks['actual_serializer_invented_NPZ_reload_bytes']=set(z.files)==set(arrays) and all(exact(z[k],a) for k,a in arrays.items())
    agg=summarize_rows(rows,outputs)[0]
    checks['actual_group_summary_conserves_all_rows_U_missing_and_mass']=agg['N']==8 and agg['original_inputU']==2 and agg['missing_mass']==2 and agg['visible_counts']=={'-1':2,'0':0,'1':6} and agg['sum_jointmass7']==[0.,3.,3.,0,0,0,0]
    assert all(checks.values())
    return dict(completed=True,PASS=True,invented_only=True,checks={k:bool(v) for k,v in checks.items()},number_checks=len(checks),code=bind(__file__),source_pixel_or_header_or_hash_reads=0)


def prepare():
    assert not PROTOCOL.exists()
    definition=read(DEFINITION);rows=lines(PLAN);geometries=lines(GEOMETRY)
    assert definition['code']==bind(H/'source_end_support.py') and len(rows)==4096 and len(geometries)==1024
    assert [r['original_global_index'] for r in rows]==list(range(4096)) and all(r['split']=='TRAIN' for r in rows)
    assert [r['scene_index'] for r in rows[::8]]==list(range(512)) and sorted({r['base_index'] for r in rows})==list(range(128))
    buffers={r['scene_buffer']['path']:r['scene_buffer'] for r in rows};assert len(buffers)==512
    write(TESTS,invented())
    p=dict(completed=True,prepared_only=True,code=bind(__file__),operator=bind(H/'source_end_support.py'),definition=bind(DEFINITION),operator_preparation=bind(H/'PREPARATION_COMPLETE.json'),scope=definition['scope'],plan=bind(PLAN),geometry=bind(GEOMETRY),checks=bind(TESTS),operator_checks=definition['checks'],runtime=dict(executable=sys.executable,numpy=np.__version__),output=str(OUT),
        counts=dict(TRAIN_rows=4096,scenes=512,bases=128,source_buffers512=512,expected_depth_caches1024=1024,all_native=list(range(8)),original_inputU80=80),
        frozen_calculation='Exact previously reviewed operator. Added optional expected-patch cache input only; invented cached/uncached equality. No SAM output, model score, patch/tolerance or target criterion changes.',
        loading='Verify each TRAIN source buffer SHA before loading. Load only part/depth/triangle/ambiguity and camera/native/CF/state metadata, never RGB_uint8. Require exact512 shapes/dtypes, camera/native/CF/source state8 equality. Recheck all512 hashes after generation. BEFORE reused perbase; all8 expected-patch maps cached perbase and reused unchanged across4variants.',
        output_schema='ALL4096_END_SUPPORT.jsonl preserves entire original plan metadata plus new measures. TRAIN4096_END_SUPPORT_ONLY.npz jointmass7f64[4096,7] or allNaN+missing; visible/foreign int8{-1,0,1}; full originalID/qL/nullflags/native/semantic/inputflags/factors; newnative/newquality all-1, training_admission=false. InputU/missing never dropped. Classorder is source-end7, distinct from V173 causal7.',
        no_actual_source_runs_or_source_binary_reads_during_prepare=True,no_model_RGB_SAM_forward_fit_or_selection=True,selected_model=None,fully_solved=False)
    write(PROTOCOL,p)
    entry=dict(approved_source_end_supervision=False,TRAIN_only=True,code=p['code'],operator=p['operator'],protocol=bind(PROTOCOL),definition=p['definition'],plan=p['plan'],geometry=p['geometry'],checks=p['checks'],scope=p['scope'],automatic_retry=False,no_models_RGB_fit_or_selection=True)
    write(H/'ROOT_RUNNER_ENTRY_TEMPLATE_APPROVED_FALSE.json',entry)
    write(PREP,dict(completed=True,PASS=True,prepared_only=True,code=p['code'],operator=p['operator'],protocol=bind(PROTOCOL),checks=p['checks'],operator_checks=p['operator_checks'],entry_template=bind(H/'ROOT_RUNNER_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_generation_runs=0,source_payload_body_header_hash_reads=0))
    print(json.dumps(dict(PASS=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],preparation=bind(PREP))),flush=True)


def run(entry):
    e=read(entry);assert e['approved_source_end_supervision'] is True and e['TRAIN_only'] is True and e['automatic_retry'] is False and e['no_models_RGB_fit_or_selection'] is True
    assert e['protocol']==bind(PROTOCOL);p=read(checkbind(e['protocol']))
    for k in ('code','operator','definition','plan','geometry','checks','scope'):assert e[k]==p[k];checkbind(p[k])
    assert p['code']==bind(__file__) and p['operator']==bind(H/'source_end_support.py')
    assert read(checkbind(p['checks']))['PASS'] and read(checkbind(p['operator_checks']))['PASS']
    assert sys.executable==p['runtime']['executable'] and np.__version__==p['runtime']['numpy'] and not OUT.exists()
    rows=lines(PLAN);gg=lines(GEOMETRY);gm={(g['base_index'],g['native_slot']):g for g in gg}
    assert len(rows)==4096 and len(gg)==1024 and [r['original_global_index'] for r in rows]==list(range(4096)) and all(r['split']=='TRAIN' for r in rows)
    OUT.mkdir();write(OUT/'STARTED.json',dict(entry=bind(entry),protocol=bind(PROTOCOL),TRAIN_only=True));used={};outputs=[];start=time.monotonic()
    try:
        with (OUT/'ALL4096_END_SUPPORT.jsonl').open('x') as jf:
            for base in range(128):
                rr=rows[base*32:(base+1)*32];assert len(rr)==32 and all(r['base_index']==base for r in rr)
                before_records=rr[:8];assert all(r['variant']=='BEFORE' for r in before_records)
                before_binding=before_records[0]['scene_buffer'];before=load_scene(before_binding,before_records,gm[base,0]);used[before_binding['path']]=before_binding
                cache={}
                for slot in range(8):
                    g=gm[base,slot];cache[slot]=dict(op.expected_patch_depth(g['quads_world'],g['camera'],g['raw_hw']),geometry_key=g['geometry_key'])
                for variant in range(4):
                    scene_records=rr[variant*8:(variant+1)*8];binding=scene_records[0]['scene_buffer']
                    assert scene_records[0]['scene_index']==base*4+variant and all(r['BEFORE_buffer']==before_binding for r in scene_records)
                    if variant==0:scene=before
                    else:scene=load_scene(binding,scene_records,gm[base,0]);used[binding['path']]=binding
                    for r in scene_records:
                        result=op.process_one(r,gm[base,r['native_slot']],before,scene,cache[r['native_slot']])
                        assert result['original_semantic_state']==r['original_semantic_state'] and result['original_input_flags']==r['original_input_flags'] and result['query_xy']==r['query_xy'] and result['L']==r['L']
                        output=dict(r);output.update(result);output['training_admission']=False
                        jf.write(json.dumps(output,ensure_ascii=False,allow_nan=False)+'\n');outputs.append(output)
                del cache,before,scene
                if (base+1)%8==0:print(json.dumps(dict(bases=base+1,rows=len(outputs),seconds=time.monotonic()-start)),flush=True)
        assert len(outputs)==4096 and len(used)==512
        arrays=output_arrays(rows,outputs);npz=OUT/'TRAIN4096_END_SUPPORT_ONLY.npz';np.savez_compressed(npz,**arrays)
        with np.load(npz,allow_pickle=False) as z:assert set(z.files)==set(arrays) and all(exact(z[k],a) for k,a in arrays.items())
        summary=OUT/'SUMMARY.json';write(summary,dict(completed=True,groups=summarize_rows(rows,outputs),training_admission=False,selected_model=None,fully_solved=False))
        for b in used.values():checkbind(b)
        for k in ('code','operator','definition','plan','geometry','checks','scope','operator_checks'):checkbind(p[k])
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=p['code'],operator=p['operator'],definition=p['definition'],protocol=bind(PROTOCOL),entry=bind(entry),plan=p['plan'],geometry=p['geometry'],checks=p['checks'],outputs=dict(rows=bind(OUT/'ALL4096_END_SUPPORT.jsonl'),sidecar=bind(npz),summary=bind(summary)),
            counts=dict(rows4096=len(outputs),source_buffers512=len(used),bases128=128,scenes512=512,expected_depth_maps1024=1024,original_inputU80=int((~arrays['query_available']).sum()),missing_mass=int(arrays['jointmass7_missing'].sum())),all_source_buffer_hashes_unchanged=True,original_metadata_q_L_native_and_old_semantics_unchanged=True,
            target_rule_unchanged_after_source_or_SAM_results=True,new_native_quality_minus1=True,training_admission=False,new_RGB_model_SAM_forward_fit_PnP_qL_or_threshold_selection=0,selected_model=None,fully_solved=False,seconds=time.monotonic()-start))
        print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'))),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION_PRESERVED.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();mode=ap.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare',action='store_true');mode.add_argument('--entry',type=Path);args=ap.parse_args()
    prepare() if args.prepare else run(args.entry)
