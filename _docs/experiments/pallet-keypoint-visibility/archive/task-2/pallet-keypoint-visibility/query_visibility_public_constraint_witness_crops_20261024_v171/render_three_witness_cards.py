"""Exactly the sealed V171 three-row witness selection; pixels only, no models."""
from pathlib import Path
import argparse,ast,hashlib,io,json,subprocess,textwrap,traceback
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont

V=Path(__file__).resolve().parent;P=V.parent
SEL=V/'WITNESS_SELECTION_BEFORE_RGB.json';SEL_SHA='03dba6a0579b724c4388283f8d61484963ca75ca203d2d0fbb2c964ba56e8c0e'
READY=V/'INPUT_PATH_READINESS.json';READY_SHA='116d9df7665a4944214e116ee27f8467e5960e56e3068b227041b6d4d5eefb79'
PROTOCOL=V/'RENDER_PROTOCOL_BEFORE_PIXEL_BODIES.json';TESTS=V/'GENERATED_PIXEL_CHECKS_BEFORE_ACTUAL.json';OUT=V/'forensic_cards_v1'
CROP=P/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py';CROP_SHA='6a6c2543e28715e3d47e15c543c25f75ff7d3efa3b881e523bd3451817fc004b'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(b):
    p=Path(b['path']).resolve();assert p.is_relative_to(P)
    assert not any(s in str(p).lower() for s in ('current223','_v169','_v112','/private_evaluation/'))
    assert sha(p)==b['sha256'];return p
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def crop_function():
    assert sha(CROP)==CROP_SHA;tree=ast.parse(CROP.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='crop'];assert len(nodes)==1
    ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(CROP)+'::PURE_RGB_CROP','exec'),ns);return ns['crop']
def central32(a):return a[32:64,32:64].copy()
def prepare():
    assert sha(SEL)==SEL_SHA and sha(READY)==READY_SHA
    crop=crop_function();checks={}
    a=np.arange(8*8*3,dtype=np.uint8).reshape(8,8,3);cr,valid=crop(a,np.array([3.,3.]),2.,2)
    manual=np.zeros((2,2,3),np.uint8)
    for iy,y in enumerate((2,3)):
        for ix,x in enumerate((2,3)):manual[iy,ix]=np.floor(a[y:y+2,x:x+2].astype(np.float64).mean((0,1))+.5).astype(np.uint8)
    checks['frozen_bilinear_rounding_vs_independent_four_pixel_average']=np.array_equal(cr.transpose(1,2,0),manual) and valid.all()
    outside,vo=crop(a,np.array([-20.,-20.]),2.,96);checks['out_of_frame_invalid_not_natural_background']=not vo.any() and not outside.any()
    b=np.arange(96*96*3,dtype=np.uint16).reshape(96,96,3).astype(np.uint8);c=central32(b)
    checks['central32_exact_saved96_slice']=c.shape==(32,32,3) and c.tobytes()==b[32:64,32:64].tobytes()
    original=b.copy();display=Image.fromarray(b.copy());ImageDraw.Draw(display).ellipse((20,20,40,40),outline='red',width=3)
    checks['overlay_uses_copy_only']=b.tobytes()==original.tobytes() and np.asarray(display).tobytes()!=b.tobytes()
    mem=io.BytesIO();Image.fromarray(b).save(mem,format='PNG');mem.seek(0)
    checks['PNG_output_roundtrip_raw_uint8_exact']=np.array_equal(np.asarray(Image.open(mem)),b)
    checks={k:bool(x) for k,x in checks.items()};assert all(checks.values())
    write(TESTS,dict(completed=True,PASS=True,invented_only=True,checks=checks,source_pixel_bodies_read=0,code=bind(__file__)))
    write(PROTOCOL,dict(completed=True,code=bind(__file__),selection=bind(SEL),readiness=bind(READY),crop_operator=bind(CROP),tests=bind(TESTS),actual_authorization='Root task message explicitly authorizes one render of the unchanged three selected rows; no extra entry required',raw96_and_central32_clean=True,overlay='Only display copies of fullRGB have original q red circle/L box and original nominal GT reference cyan cross; raw96/central32 untouched',no_new_selection=True,forbidden='No GT masks/new model/threshold fit/features/weights/current/V169/private/V112/PnP/qL or label changes',automatic_retry=False))
    print(json.dumps({'prepared':True,'protocol':bind(PROTOCOL),'tests':bind(TESTS)}),flush=True)
def fonts():
    p=subprocess.check_output(['fc-match','-f','%{file}','Noto Sans CJK KR'],text=True).strip()
    return {n:ImageFont.truetype(p,n) for n in (20,22,24,27,30,36)}
def render(row,rgb,crop96,valid,folder,font):
    canvas=Image.new('RGB',(1720,1180),'#f5f7fa');d=ImageDraw.Draw(canvas)
    def txt(x,y,s,n=24,fill='#172433'):d.text((x,y),s,font=font[n],fill=fill)
    gt=row['source_GT_metadata'];i=row['global_index']
    txt(28,20,f"V171  |  OLD6144 index {i}  |  {row['query_id']}",36)
    txt(28,78,'FIXED CONSTRAINT-BOUND WITNESS  /  public synthetic source  /  no label or threshold selection',24)
    scores='  |  '.join(arm.split('_')[0]+f' raw E-logit {score:.9f}' for arm,score in row['both_fixed_head_raw_E_logits'].items())
    txt(28,122,scores,27)
    txt(28,168,f"Original semantic role {gt['teacher_state']}  |  native {row['native_slot']} / canonical {gt['canonical_index']}  |  boundary {gt['correspondence_boundary']}  |  q-reference error {gt['error_px']:.6f} px",24)
    txt(28,206,f"{row['variant']}  |  {gt['factors']['material']}  |  {gt['factors']['geometry_family']}  |  SAME_TARGET_MATERIAL",22)
    scene=Image.fromarray(rgb.copy()).resize((640,640),Image.Resampling.NEAREST);sd=ImageDraw.Draw(scene)
    scale=640/rgb.shape[1];q=np.asarray(row['query_xy'])*scale;ref=np.asarray(gt['reference_xy'])*scale;L=row['L']*scale
    sd.rectangle((q[0]-L/2,q[1]-L/2,q[0]+L/2,q[1]+L/2),outline='#ff3845',width=2)
    sd.line((q[0],q[1],ref[0],ref[1]),fill='white',width=2)
    sd.ellipse((q[0]-5,q[1]-5,q[0]+5,q[1]+5),outline='#ff2739',width=3)
    sd.line((ref[0]-7,ref[1],ref[0]+7,ref[1]),fill='#00e5ff',width=3);sd.line((ref[0],ref[1]-7,ref[0],ref[1]+7),fill='#00e5ff',width=3)
    canvas.paste(scene,(28,280));txt(28,244,'Full RGB display copy: q red / nominal reference cyan',22)
    txt(724,244,'Actual saved RGB96 (clean)',24);txt(1210,244,'Central32 = [32:64] (clean)',24)
    canvas.paste(Image.fromarray(crop96).resize((384,384),Image.Resampling.NEAREST),(724,280))
    canvas.paste(Image.fromarray(central32(crop96)).resize((384,384),Image.Resampling.NEAREST),(1210,280))
    txt(724,694,'Actual valid96: white=valid',22)
    canvas.paste(Image.fromarray((valid.astype(np.uint8)*255)).convert('RGB').resize((192,192),Image.Resampling.NEAREST),(724,736))
    txt(952,710,f"q = ({row['query_xy'][0]:.9f}, {row['query_xy'][1]:.9f})",22)
    txt(952,752,f"L = {row['L']:.9f}  |  valid {int(valid.sum())}/9216",22)
    txt(952,794,'RGB96 reconstruction / source hashes: PASS',22)
    txt(952,836,'The semantic GT role is separate from literal q owner.',22)
    txt(28,948,'Bound memberships (order statistics; never selected operating thresholds):',24)
    y=994
    for m in row['bound_memberships']:
        txt(28,y,f"{m['arm'].split('_')[0]} {m['state']}: rank {m['rank']} / N {m['group_N']}  |  bound {m['bound_float64']:.12f}",22);y+=38
    txt(28,1090,'이 카드는 원본 소스 진단입니다. ref 오차/semantic 역할/q 중심 재료를 구분하며 라벨·좌표를 교정하지 않습니다.',22)
    txt(28,1130,'양 head·문턱·q/L·라벨 불변  |  새 학습·편입·성공 판정 없음  |  full goal 미해결',22)
    path=folder/f'OLD_{i:04d}_FORENSIC.png';canvas.save(path)
    return path
def run():
    assert sha(SEL)==SEL_SHA and sha(READY)==READY_SHA
    p=read(PROTOCOL);assert p['code']==bind(__file__) and p['selection']==bind(SEL) and p['readiness']==bind(READY)
    assert read(checked(p['tests']))['PASS'];crop=crop_function();selection=read(SEL)
    assert selection['deduplicated_rows']==3 and selection['total_memberships']==4 and not OUT.exists();OUT.mkdir()
    sources={};receipts=[];f=fonts()
    try:
        for row in selection['unique_witnesses']:
            rb=row['raw_RGB_PNG_binding_inherited'];cb=row['rgb96_crop_binding_inherited'];sources[rb['path']]=rb;sources[cb['path']]=cb
            image=cv2.imread(str(checked(rb)),cv2.IMREAD_COLOR);assert image is not None and image.dtype==np.uint8
            rgb=np.ascontiguousarray(image[:,:,::-1]);assert list(rgb.shape)==row['raw_hw']+[3] and ash(rgb)==row['decoded_RGB_sha256_inherited']
            with np.load(checked(cb),allow_pickle=False) as z:
                assert set(z.files)=={'rgb96','valid96','crop_constructed','global_index'};k=row['rgb96_row']
                assert z['global_index'][k]==row['global_index'] and z['crop_constructed'][k]
                stored=z['rgb96'][k].copy();valid=z['valid96'][k].copy();container_rows=len(z['global_index'])
            assert stored.shape==(96,96,3) and stored.dtype==np.uint8 and valid.shape==(96,96) and valid.dtype==bool
            assert ash(stored)==row['rgb96_array_sha256_inherited'] and ash(valid)==row['valid96_array_sha256_inherited']
            rebuilt,rv=crop(rgb,np.asarray(row['query_xy'],np.float64),row['L'],96)
            assert np.array_equal(rebuilt.transpose(1,2,0),stored) and np.array_equal(rv[0],valid)
            before=(ash(rgb),ash(stored),ash(valid));path=render(row,rgb,stored,valid,OUT,f)
            assert before==(ash(rgb),ash(stored),ash(valid))
            receipts.append(dict(global_index=row['global_index'],query_id=row['query_id'],card=bind(path),raw_RGB=rb,crop_chunk=cb,
                decoded_RGB_array_sha256=ash(rgb),rgb96_array_sha256=ash(stored),valid96_array_sha256=ash(valid),central32_array_sha256=ash(central32(stored)),
                frozen_crop_reconstruction_byte_exact=True,clean_arrays_unmodified_after_overlay=True,central32_exact_saved96_slice=True,valid_pixels=int(valid.sum()),crop_container_rows=container_rows,
                crop_container_loading_disclosure='NPZ container carries512 public synthetic rows; only selected saved row materialized/inspected per witness, no features or mixed labels loaded.'))
        for b in sources.values():checked(b)
        complete=dict(completed=True,PASS=True,phase='V171',selection=bind(SEL),readiness=bind(READY),code=bind(__file__),protocol=bind(PROTOCOL),tests=bind(TESTS),cards=receipts,unique_cards=3,all4_bound_memberships_preserved=True,
            actual_source_PNG_files=3,actual_crop_containers=len({r['crop_chunk']['path'] for r in receipts}),source_RGB_and_crop_encoded_SHA_reverified=True,decoded_and_crop_pixel_SHA_reverified=True,
            exact_qL_crop_reconstruction_all_PASS=True,no_label_qL_or_threshold_changes=True,models_features_weights_current_V169_private_V112_access=0,new_fit_or_forward=0,training_admission=False,fully_solved=False,
            actual_visual_inspection='Pending separate tool view of each rendered card; no browser render claimed')
        write(OUT/'COMPLETE.json',complete);print(json.dumps({'complete':bind(OUT/'COMPLETE.json'),'cards':[r['card']['path'] for r in receipts]}),flush=True)
    except BaseException:
        write(OUT/'FAILED_EXECUTION.json',dict(completed=False,traceback=traceback.format_exc(),automatic_retry=False));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.prepare and not a.run:prepare()
    elif a.run and not a.prepare:run()
    else:raise SystemExit('Use --prepare then --run exactly once')
