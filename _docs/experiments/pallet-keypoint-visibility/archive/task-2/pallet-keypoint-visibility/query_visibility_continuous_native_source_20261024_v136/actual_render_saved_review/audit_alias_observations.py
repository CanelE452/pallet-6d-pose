"""Saved RGB evidence diagnostics; never a replacement visibility teacher.

Every640base is retained. BEFORE-D/COVER-E and near-D flags describe existing
continuous37-ray labels only. Pixel changes do not relabel, filter or admit
data. Windows use a fixed one-native-pixel padding around active GT witnesses,
not an actual-query crop. Full saved-source audit must complete first.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys,time,traceback
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v136_saved_source_checks',H/'review_saved_source.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def pixel_delta(before,after,bbox):
    assert before.shape==after.shape==(512,512,3)and before.dtype==after.dtype==np.uint8
    box=np.asarray(bbox,np.float64);assert box.shape==(4,)and np.isfinite(box).all()and (box[2:]>=box[:2]).all()
    # Pixel-centre indices include floor(min)-1 .. ceil(max)+1, inclusive.
    lo=np.clip(np.floor(box[:2]).astype(int)-1,0,512)
    hi=np.clip(np.ceil(box[2:]).astype(int)+2,0,512)
    changed=np.any(before!=after,axis=-1);absolute=np.abs(after.astype(np.int16)-before.astype(np.int16))
    local=changed[lo[1]:hi[1],lo[0]:hi[0]];ld=absolute[lo[1]:hi[1],lo[0]:hi[0]]
    return dict(diagnostic_bbox_plus1_integer_xyxy=np.r_[lo,hi].tolist(),diagnostic_window_pixels=int(local.size),
        whole_RGB_exact_equal=bool(not changed.any()),wholeframe_changed_pixels=int(changed.sum()),
        active_bbox_plus1_changed_pixels=int(local.sum()),active_bbox_plus1_RGB_L1=int(ld.sum()),
        active_bbox_plus1_max_channel_difference=int(ld.max())if ld.size else 0,
        active_bbox_plus1_mean_abs_channel_difference=float(ld.mean())if ld.size else None,
        active_bbox_plus1_changed_fraction=float(local.mean())if local.size else None)

def describe(values):
    a=np.asarray(values,np.float64)
    return dict(N=len(a),min=float(a.min()),q25=float(np.quantile(a,.25)),median=float(np.median(a)),q75=float(np.quantile(a,.75)),max=float(a.max()))if len(a)else dict(N=0)

def generated():
    a=np.zeros((512,512,3),np.uint8);b=a.copy();b[10,10,0]=1;b[12,12,2]=3;b[0,0]=255
    r=pixel_delta(a,b,[10.1,10.1,10.9,10.9])
    assert r['diagnostic_bbox_plus1_integer_xyxy']==[9,9,13,13]and r['diagnostic_window_pixels']==16
    assert r['wholeframe_changed_pixels']==3 and r['active_bbox_plus1_changed_pixels']==2 and r['active_bbox_plus1_RGB_L1']==4
    assert r['active_bbox_plus1_max_channel_difference']==3 and not r['whole_RGB_exact_equal']
    assert pixel_delta(a,a,[-1,-1,0,0])['diagnostic_bbox_plus1_integer_xyxy']==[0,0,2,2]
    assert pixel_delta(a,a,[511.1,511.1,512,512])['diagnostic_bbox_plus1_integer_xyxy']==[510,510,512,512]
    assert pixel_delta(a,a,[600,600,601,601])['diagnostic_window_pixels']==0
    assert pixel_delta(a,a,[10,10,11,11])['whole_RGB_exact_equal']
    m.write(H/'GENERATED_ALIAS_PIXEL_METRIC_CHECKS.json',dict(completed=True,PASS=True,code=m.bind(__file__),invented_arrays_only=True,one_native_pixel_padding_exact=True,
        wholeframe_vs_local_change_counts_independent=True,signed_pixel_difference_no_uint8_wrap=True,bounds_clipped_empty_distinct=True,actual_RGB_GT_models_read=0))
    print(json.dumps(dict(generated=m.bind(H/'GENERATED_ALIAS_PIXEL_METRIC_CHECKS.json'),actual_source_reads=0)),flush=True)

def run():
    start=time.monotonic();saved=H/'SAVED_SOURCE_AUDIT_COMPLETE.json';audit=m.read(saved)
    assert audit['completed']and audit['PASS']and audit['slots']==20480 and audit['completed_render']==m.bind(m.COMPLETE)
    proof=m.read(H/'GENERATED_ALIAS_PIXEL_METRIC_CHECKS.json');assert proof['PASS']and proof['code']==m.bind(__file__)
    p,configs,core=m.source_metadata();rows=[]
    for b in range(640):
        cfg,checkpoint,inputs,gts=m.base_packet(b,configs);selected=cfg['selected_CF_index']
        before=m.arrays(m.check(gts[0]['raw_buffers']),('RGB_uint8',))['RGB_uint8']
        states=[gr['state8_CF'][selected]for gr in gts];bbox=gts[0]['records_CF'][selected]['projected_active_witness_bbox_xyxy']
        metrics={}
        for vi in(1,2,3):
            assert bbox==gts[vi]['records_CF'][selected]['projected_active_witness_bbox_xyxy']
            after=m.arrays(m.check(gts[vi]['raw_buffers']),('RGB_uint8',))['RGB_uint8']
            metrics[m.VARIANTS[vi]]=pixel_delta(before,after,bbox)
        rows.append(dict(base_index=b,base_id=cfg['base_id'],split=cfg['split'],factors=cfg['factors'],selected_CF_index=selected,
            selected_teacher_states_BEFORE_COVER_NEAR_FAR=states,verified_BEFORE_D_COVER_E=states[:2]==['D','E'],verified_NEAR_SHAM_D=states[0]==states[2]=='D',
            active_witness_bbox_xyxy=bbox,active_witness_span_wh_native_px=[bbox[2]-bbox[0],bbox[3]-bbox[1]],
            delta_vs_BEFORE=metrics,raw_RGB_inputs=[r['raw_RGB_PNG']for r in inputs],source_base_checkpoint=m.bind(m.SOURCE/'bases'/cfg['base_id']/'BASE_COMPLETE.json')))
        if(b+1)%128==0:print(json.dumps(dict(alias_bases_done=b+1,total640=True,seconds=time.monotonic()-start)),flush=True)
    assert len(rows)==640
    summaries={}
    for fold in('ALL','TRAIN','VALIDATION'):
        rr=[r for r in rows if fold=='ALL'or r['split']==fold];valid=[r for r in rr if r['verified_BEFORE_D_COVER_E']]
        near=[r for r in rr if r['verified_NEAR_SHAM_D']]
        coverage={}
        for variant,pairs in(('COVER',valid),('NEAR_SHAM',near)):
            changes=[r['delta_vs_BEFORE'][variant]['active_bbox_plus1_changed_pixels']for r in pairs]
            coverage[variant]=dict(existing_label_pair_count=len(pairs),whole_RGB_exact_equal_count=sum(r['delta_vs_BEFORE'][variant]['whole_RGB_exact_equal']for r in pairs),
                active_bbox_plus1_zero_changed_pixel_count=sum(x==0 for x in changes),active_bbox_plus1_one_to_four_changed_pixel_count=sum(1<=x<=4 for x in changes),
                active_bbox_plus1_changed_pixels=describe(changes),wholeframe_changed_pixels=describe([r['delta_vs_BEFORE'][variant]['wholeframe_changed_pixels']for r in pairs]),
                zero_local_change_base_indices=[r['base_index']for r in pairs if r['delta_vs_BEFORE'][variant]['active_bbox_plus1_changed_pixels']==0])
        summaries[fold]=dict(all_original_bases=len(rr),not_verified_beforeD_coverE=len(rr)-len(valid),active_witness_width_allbases=describe([r['active_witness_span_wh_native_px'][0]for r in rr]),
            active_witness_height_allbases=describe([r['active_witness_span_wh_native_px'][1]for r in rr]),
            active_witness_width_verifiedDE=describe([r['active_witness_span_wh_native_px'][0]for r in valid]),active_witness_height_verifiedDE=describe([r['active_witness_span_wh_native_px'][1]for r in valid]),pairs=coverage)
    result=dict(completed=True,PASS=True,code=m.bind(__file__),saved_source_audit=m.bind(saved),completed_render=m.bind(m.COMPLETE),generated=m.bind(H/'GENERATED_ALIAS_PIXEL_METRIC_CHECKS.json'),
        original_bases640_all_retained=True,original_scenes2560_all_retained=True,actual_initializer_query_access=0,summary=summaries,rows=rows,
        interpretation='Observational RGB evidence/alias audit only. Existing continuous37-ray teacher unchanged. Whole/local zero or low pixel change is explicitly retained, not used as a visibility label, quality filter or source admission decision. Padded GT witness window is not a future actual-query RGB96 crop.',
        no_GT_relabel_or_exclusion=True,no_quality_selection=True,new_teacher_calls=0,model_fit_private_V112_PnP_access=0,seconds=time.monotonic()-start)
    target=H/'SOURCE_NATIVE_WITNESS_RGB_ALIAS_OBSERVATIONS.json';m.write(target,result)
    print(json.dumps(dict(completed=True,PASS=True,result=m.bind(target),summary=summaries,seconds=time.monotonic()-start)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--generated-checks',action='store_true');ap.add_argument('--audit',action='store_true');a=ap.parse_args()
    try:
        if a.generated_checks and not a.audit:generated()
        elif a.audit and not a.generated_checks:run()
        else:raise SystemExit('Use --generated-checks or --audit')
    except BaseException:
        m.write(H/('FAILED_ALIAS_'+str(time.time_ns())+'.json'),dict(code=m.bind(__file__),traceback=traceback.format_exc(),source_files_mutated=False,automatic_retry=False));raise
