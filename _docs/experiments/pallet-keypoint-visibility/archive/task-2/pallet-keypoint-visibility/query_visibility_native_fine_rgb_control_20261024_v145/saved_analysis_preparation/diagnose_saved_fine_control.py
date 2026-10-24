"""Fixed source-only postfit pair and crop readout. No learned operator.

--prepare reads existing metadata only; --run requires terminal CONTROL_COMPLETE.
Original source q/L and all known D/E denominators are preserved. Actual q/L
can move between variants, so pair differences are descriptive, not isolated
pixel-cause estimates. New scores never select gallery examples.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;V=H.parent;P=V.parent
POLICY=V/'ROOT_FINE_NATIVE_CONTROL_POLICY_BEFORE_FIT.json'
PROTOCOL=H/'SAVED_ANALYSIS_PROTOCOL_BEFORE_NEW_SCORES.json'
INPUT=V/'source_inputs_v1/INPUT_AND_SPLIT_SEAL.json'
RUN=V/'runs/SHARP_BLURRED_NATIVE_FIXED2000';OUT=V/'saved_fine_control_diagnosis_v1'
REG=P/'query_visibility_broad_source_geometry_control_20261024_v139/saved_result_preparation/FIXED_CASES_BEFORE_NEW_SCORES.json'
ARMS=('SHARP_NATIVE_CENTRAL32','BLURRED_NATIVE_CENTRAL32');NO=6144;N=26624
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def read(path):return json.loads(Path(path).read_text())
def check(b):
    assert '_v112'not in b['path'].lower()and '/private_evaluation/'not in b['path'];assert sha(b['path'])==b['sha256'];return Path(b['path'])
def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def quant(a):
    a=np.asarray(a);a=a[np.isfinite(a)];return np.quantile(a,[0,.25,.5,.75,1]).tolist()if len(a)else None
def cm(state,pred,mask):
    matrix=[[int((mask&(state==s)&(pred==p)).sum())for p in(0,1,-1)]for s in'DE']
    d,e=matrix;dr=sum(d);er=sum(e)
    return dict(CM_D_E_rows_D_E_U_columns=matrix,known_DE=dr+er,teacher_S=int((mask&(state=='S')).sum()),teacher_U=int((mask&(state=='U')).sum()),
        D_recall=d[0]/dr if dr else None,E_recall=e[1]/er if er else None,
        D_precision=d[0]/(d[0]+e[0])if d[0]+e[0]else None,E_precision=e[1]/(e[1]+d[1])if e[1]+d[1]else None,
        known_input_U=d[2]+e[2],known_U_rate=(d[2]+e[2])/(dr+er)if dr+er else None,all_rows=int(mask.sum()),all_input_U=int((mask&(pred<0)).sum()))
def prepare():
    registry=read(REG);assert registry['prepared_before_new_outputs']and registry['no_new_score_or_RGB_reads']
    assert registry['fixed_base_slot_pairs']==[[519,0],[590,1]]and len(registry['cases'])==8
    p=dict(completed=True,prepared_before_new_scores=True,code=bind(__file__),policy=bind(POLICY),input_seal=bind(INPUT),fixed_registry=bind(REG),
        expected_control=str(RUN/'CONTROL_COMPLETE.json'),output=str(OUT),fixed_all8_indices=registry['fixed8_indices'],
        pair_definition='Every original base/native channel with CF-certified COVER E and NEAR_SHAM D; retain inputU; folds and qmotion are diagnostic only.',
        RGB='Unchanged inherited RGB96; central32 exact slice and prescribed fixed valid-normalized blur; nearest enlargement and uint8 display quantization only.',
        model_fit_encoder_private_newselection_qcorrection_calls=0)
    write(PROTOCOL,p);print(json.dumps(dict(completed=True,code=bind(__file__),protocol=bind(PROTOCOL))))
def run():
    p=read(PROTOCOL);assert p['code']==bind(__file__)and p['policy']==bind(POLICY)and p['input_seal']==bind(INPUT)and p['fixed_registry']==bind(REG)
    controlpath=RUN/'CONTROL_COMPLETE.json';control=read(controlpath)
    assert control['completed']and control['PASS']and control['updates_each']==2000 and control['selected_model']is None
    assert control['same_initial_weights_and_all2000_native_draws']and control['all26624_preserved']and not OUT.exists()
    seal=read(check(p['input_seal']));b=seal['source_bindings'];registry=read(check(p['fixed_registry']))
    with np.load(check(seal['input_only']),allow_pickle=False)as f:
        inp={k:f[k].copy()for k in('base_raw_logit','available','constructed','occurrence_id','global_index','corpus','query_xy','L','scene_index','base_index','native_slot','rgb32','valid32')}
    ledgers=[]
    for key,size in(('old_GT',NO),('new_GT',N-NO)):
        rr=[json.loads(x)for x in check(b[key]).read_text().splitlines()];rr=rr[::2]
        assert len(rr)==size and all(r['convention']=='CF_PRIMARY_DECLARED'for r in rr);ledgers.append(rr)
    rows=ledgers[0]+ledgers[1];state=np.array([r['teacher_state']for r in rows]);split=np.array([r['split']for r in rows]);bands=np.array([r['position_band']for r in rows])
    for i,r in enumerate(rows):assert r['query_id']==str(inp['occurrence_id'][i])and r['native_slot']==int(inp['native_slot'][i])
    raw={'C32':inp['base_raw_logit']};packets={};bindings={}
    for arm in ARMS:
        fitpath=RUN/arm/'FIT_COMPLETE.json';fit=read(fitpath);assert fit['completed']and fit['optimizer_updates']==2000
        assert control['arms'][arm]['fit']==bind(fitpath)
        with np.load(check(fit['scores']),allow_pickle=False)as f:a={k:f[k].copy()for k in f.files}
        for k in('available','constructed','global_index','source_occurrence_index','occurrence_id','corpus','scene_index','base_index','native_slot','query_xy','L'):
            if k in inp:assert eq(a[k],inp[k]),k
        assert eq(a['raw_frozen_C32_base_logit'],raw['C32'])and eq(a['raw_head_logit_before_U'],a['raw_frozen_C32_base_logit']+a['raw_residual_delta'])
        assert np.array_equal(np.isfinite(a['raw_head_logit_before_U']),inp['constructed'])
        raw[arm]=a['raw_head_logit_before_U'];packets[arm]=a;bindings[arm]=dict(fit=bind(fitpath),scores=fit['scores'],source_results=bind(RUN/arm/'SOURCE_RESULTS.json'))
    predictions={k:np.where(inp['available'],(a>=0).astype(np.int8),-1)for k,a in raw.items()}
    OUT.mkdir();start=time.monotonic();groups={}
    for corpus in('OLD_NATIVE','CONTINUOUS_NATIVE'):
        for fold in('ALL','TRAIN','VALIDATION'):
            mask=(inp['corpus']==corpus)&((split==fold)if fold!='ALL'else True)
            groups[corpus+'::'+fold]={model:cm(state,pred,mask)for model,pred in predictions.items()}
            for band in('LE3_PX','GT3_LE8_PX','GT8_LE20_PX','GT20_PX','NO_QUERY'):
                groups[corpus+'::'+fold+'::'+band]={model:cm(state,pred,mask&(bands==band))for model,pred in predictions.items()}
    pairrows=[]
    for offset,rr in((0,ledgers[0]),(NO,ledgers[1])):
        for base in range(len(rr)//32):
            for slot in range(8):
                e,d=offset+(base*4+1)*8+slot,offset+(base*4+2)*8+slot
                if state[e]!='E'or state[d]!='D':continue
                assert rows[e]['canonical_index']==rows[d]['canonical_index']and split[e]==split[d]
                motion=float(np.linalg.norm(inp['query_xy'][e]-inp['query_xy'][d]))if np.isfinite(inp['query_xy'][[e,d]]).all()else None
                deltaL=float(abs(inp['L'][e]-inp['L'][d]))if np.isfinite(inp['L'][[e,d]]).all()else None
                record=dict(corpus=str(inp['corpus'][e]),split=str(split[e]),base_index=base,native_slot=slot,canonical_index=rows[e]['canonical_index'],
                    indices_COVER_E_NEAR_D=[e,d],available_both=bool(inp['available'][e]and inp['available'][d]),qmotion_px=motion,L_change=deltaL,
                    qerrors=[rows[e]['error_px'],rows[d]['error_px']],both_LE3=bool(bands[e]==bands[d]=='LE3_PX'),factors=rows[e]['factors'],models={})
                for model,pr in predictions.items():record['models'][model]=dict(predictions=[int(pr[e]),int(pr[d])],both_correct=bool(pr[e]==1 and pr[d]==0),
                    scoregap_E_minus_D=float(raw[model][e]-raw[model][d])if np.isfinite(raw[model][[e,d]]).all()else None)
                pairrows.append(record)
    pairsummary={}
    for corpus in('OLD_NATIVE','CONTINUOUS_NATIVE'):
        for fold in('ALL','TRAIN','VALIDATION'):
            pp=[r for r in pairrows if r['corpus']==corpus and(fold=='ALL'or r['split']==fold)]
            pairsummary[corpus+'::'+fold]=dict(all_known_pair_N=len(pp),available_both=sum(r['available_both']for r in pp),qmotion_quantiles=quant([r['qmotion_px']for r in pp if r['qmotion_px']is not None]),
                models={model:dict(both_correct=sum(r['models'][model]['both_correct']for r in pp),gap_quantiles=quant([r['models'][model]['scoregap_E_minus_D']for r in pp if r['models'][model]['scoregap_E_minus_D']is not None]),
                    recovered_vs_C32=sum(r['models'][model]['both_correct']and not r['models']['C32']['both_correct']for r in pp),regressed_vs_C32=sum(not r['models'][model]['both_correct']and r['models']['C32']['both_correct']for r in pp))for model in raw})
    fixed=registry['cases'];strips=registry['strips'];used=sorted(set([r['global_index']for r in fixed]+[r['global_index']for s in strips for r in s['variants']]))
    def enrich(r):
        i=NO+r['global_index'];g=rows[i]
        assert r['teacher']==g['teacher_state']and r['q_error_px']==g['error_px']and r['L']==float(inp['L'][i])
        return dict(r,combined_global_index=i,models_native_fine={model:dict(prediction='U'if predictions[model][i]<0 else'DE'[int(predictions[model][i])],hidden_logit=float(raw[model][i]))for model in raw})
    fixedresult=dict(completed=True,PASS=True,registry=p['fixed_registry'],all8_preserved=True,no_new_case_selection=True,
        cases=[enrich(r)for r in fixed],strips=[dict(s,variants=[enrich(r)for r in s['variants']])for s in strips])
    write(OUT/'FIXED8_CASES_AND_TWO_STRIPS.json',fixedresult)
    write(OUT/'ALL_KNOWN_CF_PAIRS.json',dict(completed=True,pairs=pairrows,summary=pairsummary,full_inputU_pairs_retained=True,actual_qmotion_not_isolated_pixel_effect=True))
    results=dict(completed=True,PASS=True,all26624_preserved=True,groups=groups,pairs=pairsummary,
        source_native35={arm:read(check(bindings[arm]['source_results']))['metrics']['native35']for arm in ARMS},
        selected_model=None,private_or_encoder_or_modelcalls=0,GTerror_report_only=True,source_validation_development_exposed=True)
    write(OUT/'SOURCE_RESULTS.json',results)
    render(fixedresult,inp,used)
    write(OUT/'DIAGNOSIS_COMPLETE.json',dict(completed=True,PASS=True,control_complete=bind(controlpath),protocol=bind(PROTOCOL),code=bind(__file__),input_seal=p['input_seal'],fixed_registry=p['fixed_registry'],
        predictions=bindings,results=bind(OUT/'SOURCE_RESULTS.json'),pairs=bind(OUT/'ALL_KNOWN_CF_PAIRS.json'),fixed=bind(OUT/'FIXED8_CASES_AND_TWO_STRIPS.json'),
        visual=bind(OUT/'VISUAL_COMPLETE.json'),selected_model=None,model_fit_encoder_private_qcorrection_newselection_calls=0,seconds=time.monotonic()-start))
    print(json.dumps(dict(completed=True,seal=bind(OUT/'DIAGNOSIS_COMPLETE.json'),pair_summary=pairsummary)))
def render(fixed,inp,used):
    import torch
    from torch.nn import functional as F
    from PIL import Image,ImageDraw,ImageFont
    torch.set_num_threads(1);ix=NO+np.array(used);rgb=torch.from_numpy(inp['rgb32'][ix]).float()/255.;valid=torch.from_numpy(inp['valid32'][ix]).float()
    a=torch.arange(-2,3,dtype=torch.float32);g=torch.exp(-a.square()/2);g=g/g.sum();kernel=torch.outer(g,g)[None,None]
    den=F.conv2d(valid,kernel,padding=2);num=F.conv2d(rgb*valid,kernel.expand(3,1,5,5),padding=2,groups=3)
    blur=torch.where(den>0,num/den.clamp_min(1e-12),torch.zeros_like(num))*valid;sharp=rgb*valid
    assert not blur.masked_select(~valid.bool().expand(-1,3,-1,-1)).any()
    np.savez_compressed(OUT/'FIXED_PRESCRIBED_PIXEL_INPUTS.npz',new_source_global_index=np.array(used),sharp32=sharp.numpy(),blur32=blur.numpy(),valid32=valid.numpy())
    lookup={i:k for k,i in enumerate(used)};font=lambda n:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n)
    images=[]
    for strip in fixed['strips']:
        canvas=Image.new('RGB',(1440,810),'#f6f8fa');draw=ImageDraw.Draw(canvas)
        draw.text((12,8),f'V145 fixed base {strip["base_index"]}, native {strip["native_slot"]}: exact source RGB96 + central32 control',font=font(22),fill='black')
        for vi,row in enumerate(strip['variants']):
            x=vi*360;k=lookup[row['global_index']];original=np.asarray(Image.open(check(row['saved_RGB96'])).convert('RGB'))
            assert original.shape==(96,96,3)and hashlib.sha256(original.tobytes()).hexdigest()==row['RGB96_array_sha256']
            assert np.array_equal(original[32:64,32:64].transpose(2,0,1),inp['rgb32'][row['combined_global_index']])
            draw.text((x+8,43),f'{row["variant"]} teacher {row["teacher"]} qerr {row["q_error_px"]:.3f}px',font=font(16),fill='black')
            draw.text((x+8,67),f'Original query {row["global_index"]} L={row["L"]:.3f}',font=font(16),fill='black')
            canvas.paste(Image.fromarray(original).resize((288,288),Image.Resampling.NEAREST),(x+36,94))
            for array,xoffset,label in((sharp,8,'sharp 32'),(blur,184,'blur sigma1')):
                image=np.rint(array[k].numpy().transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
                canvas.paste(Image.fromarray(image).resize((168,168),Image.Resampling.NEAREST),(x+xoffset,414))
                draw.text((x+xoffset,390),label,font=font(17),fill='black')
            for mi,model in enumerate(('C32',)+ARMS):
                result=row['models_native_fine'][model];name={'SHARP_NATIVE_CENTRAL32':'sharp','BLURRED_NATIVE_CENTRAL32':'blur'}.get(model,model)
                draw.text((x+8,600+29*mi),f'{name}: {result["prediction"]} h={result["hidden_logit"]:+.6f}',font=font(18),fill='green'if result['prediction']==row['teacher']else'red')
        draw.text((12,710),'Top: original RGB96, 3x nearest; bottom: prescribed central32 sharp / valid-normalized blur, display quantized.',font=font(17),fill='black')
        draw.text((12,739),'q at center (15.5,15.5) of central32. Original q/L differ by variant; no GT recrop. D visible; E external cover.',font=font(17),fill='black')
        draw.text((12,768),'Fixed examples selected before these scores; source development VAL, no public/private transfer claim.',font=font(17),fill='black')
        path=OUT/f'BASE_{strip["base_index"]}_NATIVE_{strip["native_slot"]}_C32_SHARP_BLUR.png';canvas.save(path);images.append(bind(path))
    write(OUT/'VISUAL_COMPLETE.json',dict(completed=True,PASS=True,images=images,all_fixed8_and_strip_scores=bind(OUT/'FIXED8_CASES_AND_TWO_STRIPS.json'),
        fixed_input_pixels=bind(OUT/'FIXED_PRESCRIBED_PIXEL_INPUTS.npz'),original_RGB96_unmodified=True,central32_source_slice_exact=True,
        blur_fixed_formula_CPU_FP32_display_quantization=True,no_learned_model_or_GT_recrop=True,no_new_case_selection=True))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.prepare and not a.run:prepare()
    elif a.run and not a.prepare:run()
    else:raise SystemExit('Use --prepare or --run after terminal root completion')
