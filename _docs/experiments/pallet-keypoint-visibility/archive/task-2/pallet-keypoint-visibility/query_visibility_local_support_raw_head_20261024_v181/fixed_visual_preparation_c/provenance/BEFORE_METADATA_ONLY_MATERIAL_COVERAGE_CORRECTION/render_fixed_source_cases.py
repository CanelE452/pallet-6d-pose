#!/usr/bin/env python3
"""Outcome-blind fixed source evidence. Preparation reads text metadata only.

Actual saved cache/output/GT reads require a separately bound root entry after
both models freeze. No inference, fitting, RGB reconstruction or input changes.
"""
from pathlib import Path
import argparse, hashlib, html, io, json, runpy, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
PREP = Path(__file__).resolve().parent
BASES = (128,132,144,148,160,164,176,180)
ARMS = ('C50_RAW96_DUAL_SUPPORT_OWNER_AUX','C51_RAW96_PLUS_FROZEN_SAM_DUAL_SUPPORT_OWNER_AUX')
VAL_INPUT = ROOT/'synthetic_val_preparation/VAL2048_ORIGINAL_INPUTS_METADATA_ONLY.jsonl'
PUBLIC_INPUT = ROOT/'sam_input_preparation_c/PUBLIC1480_ORIGINAL_INPUTS_METADATA_ONLY.jsonl'
PLAN = ROOT/'synthetic_val_preparation/VAL2048_SUPPORT_INPUT_PLAN.jsonl'
GEOMETRY = ROOT/'synthetic_val_preparation/VAL512_FIXED_PATCH_METADATA.jsonl'
OP = ROOT/'synthetic_val_preparation/source_end_support_eval.py'
SCOPE = ROOT/'ROOT_SAVED_EVALUATION_AND_FIXED_VISUAL_SCOPE.json'
CASES = PREP/'FIXED288_CASES_BEFORE_OUTPUT_OR_PIXEL_BODIES.jsonl'
PROTOCOL = PREP/'PROTOCOL_BEFORE_SAVED_CACHE_OUTPUT_OR_GT_DISPLAY_BODIES.json'
COLORS = np.array([[130,136,145],[38,200,94],[239,62,68],[49,108,233],
                   [170,77,212],[247,139,41],[252,218,48]],np.uint8)
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def require(ok, msg):
    if not ok: raise AssertionError(msg)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def ash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def binding(p): return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def readj(p): return json.loads(Path(p).read_text())
def readrows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def json_safe(x):
    if isinstance(x,dict): return {k:json_safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [json_safe(v) for v in x]
    if isinstance(x,(float,np.floating)) and not np.isfinite(x): return None
    if isinstance(x,np.generic): return x.item()
    return x
def writej(p, x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def writerows(p, rows): Path(p).write_text(''.join(json.dumps(json_safe(r),ensure_ascii=False,allow_nan=False)+'\n' for r in rows))

def verified(b):
    p=Path(b['path'])
    require('v112' not in str(p).lower(), 'Forbidden lineage path')
    require(sha(p)==b['sha256'], 'Binding drift: '+str(p))
    return p

def original_available(r):
    return bool(r.get('original_query_available',r['original_input_flags']['query_available']))

def choose_cases(val, public):
    require(len(val)==2048 and len(public)==1480,'Original input census')
    chosen=[('SYNTH_VAL2048',r) for r in val if r['base_index'] in BASES]
    first={}
    for r in public:
        require(r['split']=='SOURCE_DEVELOPMENT','No held public rows')
        seq=r['source_id'].split(':')[0]
        first[seq]=min(first.get(seq,r['original_frame_index']),r['original_frame_index'])
    require(len(first)==4,'Original public four sequences')
    chosen += [('PUBLIC_TRAIN1480',r) for r in public if r['original_frame_index']==first[r['source_id'].split(':')[0]]]
    out=[]
    for domain,r in chosen:
        synth=domain=='SYNTH_VAL2048'
        group=('VAL_SCENE_%06d'%r['scene_index']) if synth else ('PUBLIC_FRAME_%04d'%r['original_frame_index'])
        out.append(dict(domain=domain,group_id=group,case_id=group+'_N%d'%r['native_slot'],
                        original_global_index=int(r['original_global_index']),
                        original_occurrence_id=r['original_occurrence_id'],native_slot=int(r['native_slot']),
                        query_xy=r['original_query_xy'],L=r['original_L'],split=r['split'],
                        original_query_available=original_available(r),
                        original_input_flags=r['original_input_flags'],
                        scene_index=r.get('scene_index'),base_index=r.get('base_index'),
                        original_frame_index=r.get('original_frame_index'),variant=r.get('variant'),
                        source_id=r.get('source_id'),factors=r.get('factors',{})))
    out.sort(key=lambda r:(r['domain']=='PUBLIC_TRAIN1480',r['group_id'],r['native_slot']))
    require(len(out)==288 and len({r['case_id'] for r in out})==288,'Fixed288')
    require(sum(r['domain']=='SYNTH_VAL2048' for r in out)==256,'Fixed VAL256')
    require(len({r['group_id'] for r in out})==36,'32 scenes + 4 frames')
    for g in {r['group_id'] for r in out}:
        require([r['native_slot'] for r in out if r['group_id']==g]==list(range(8)),'Native8 unchanged')
    return out

def index_unique(values, name):
    vals=[str(x) for x in values]
    require(len(set(vals))==len(vals), 'Duplicate '+name)
    return {k:i for i,k in enumerate(vals)}

def equal_qL(a,b):
    return np.array_equal(np.asarray(a['query_xy'],np.float64),np.asarray(b['query_xy'],np.float64),equal_nan=True) and np.array_equal(np.asarray(a['L'],np.float64),np.asarray(b['L'],np.float64),equal_nan=True)

def decode(raw_s,raw_o,available):
    s=np.full((len(available),2),-1,np.int8)
    o=np.full(len(available),-1,np.int8)
    proxy=np.full(len(available),-1,np.int8)
    require(np.isfinite(raw_s[available]).all() and np.isfinite(raw_o[available]).all(),'Available logits finite')
    require(np.isnan(raw_s[~available]).all() and np.isnan(raw_o[~available]).all(),'Unavailable logits NaN')
    s[available]=(raw_s[available]>0).astype(np.int8)
    o[available]=np.argmax(raw_o[available],axis=1).astype(np.int8)
    proxy[available & (s[:,1]==1)]=1
    proxy[available & (s[:,1]==0) & (s[:,0]==1)]=0
    return s,o,proxy

def loadnpz(b, keys=None):
    p=verified(b)
    with np.load(p,allow_pickle=False) as z:
        if keys is None: keys=z.files
        return {k:z[k] for k in keys}

def validate_domain(inputs, targets, rows, outputs, expected):
    keys={'rgb96','valid96','sam_mask96','sam_available96','original_query_available','row_id',
          'original_global_index','native_slot','query_xy','L'}
    require(set(inputs)==keys,'Exact input whitelist')
    require(set(targets)=={'support_target','owner_soft','owner_target','owner_ambiguous_mass','owner_invalid_mass'},'Exact separate target whitelist')
    n=expected
    require(len(rows)==n and inputs['rgb96'].shape==(n,96,96,3) and inputs['rgb96'].dtype==np.uint8,'Raw RGB schema')
    for k in ('valid96','sam_mask96','sam_available96'):
        require(inputs[k].shape==(n,96,96) and inputs[k].dtype==bool,k+' schema')
    require(inputs['query_xy'].shape==(n,2) and inputs['query_xy'].dtype==np.float64 and inputs['L'].dtype==np.float64,'Original qL schema')
    require(inputs['native_slot'].dtype==np.int8 and inputs['original_global_index'].dtype==np.int64,'Original IDs schema')
    require(inputs['original_query_available'].shape==(n,) and inputs['original_query_available'].dtype==bool,'Availability schema')
    require(targets['support_target'].shape==(n,2) and targets['owner_soft'].shape==(n,3),'Target shape')
    rowmap=index_unique(inputs['original_global_index'],'original global index')
    index_unique(inputs['row_id'],'row_id')
    for i,r in enumerate(rows):
        require(str(r['row_id'])==str(inputs['row_id'][i]),'Metadata roworder/rowID')
        require(int(r['original_global_index'])==int(inputs['original_global_index'][i]) and int(r['native_slot'])==int(inputs['native_slot'][i]),'Metadata original ID')
        require(equal_qL(r,dict(query_xy=inputs['query_xy'][i],L=inputs['L'][i])),'Metadata qL')
        require(ash(inputs['rgb96'][i])==r['actual_raw_RGB96_sha256'] and ash(inputs['valid96'][i])==r['actual_raw_valid96_sha256'],'Cache raw pixel hash')
    for arm,z in outputs.items():
        for k in ('row_id','original_global_index','native_slot','query_xy','L','original_query_available'):
            require(z[k].dtype==inputs[k].dtype and np.array_equal(z[k],inputs[k],equal_nan=True) if z[k].dtype.kind not in 'US' else np.array_equal(z[k],inputs[k]),'Output identity '+arm+'/'+k)
        require(z['raw_support_logits'].shape==(n,2) and z['raw_support_logits'].dtype==np.float32,'Support logits schema')
        require(z['raw_owner_logits'].shape==(n,3) and z['raw_owner_logits'].dtype==np.float32,'Owner logits schema')
        s,o,p=decode(z['raw_support_logits'],z['raw_owner_logits'],inputs['original_query_available'])
        require(np.array_equal(s,z['support_prediction']) and np.array_equal(o,z['owner_prediction']) and np.array_equal(p,z['regional_support_proxy_D0_E1_Uminus1']),'Saved fixed decoder')
    return rowmap

def font(n): return ImageFont.truetype(FONT,n)
def rgbmask(a): return np.repeat((np.asarray(a,dtype=np.uint8)*255)[...,None],3,axis=2)
def png_raw(path,a):
    Image.fromarray(a).save(path)
    back=np.asarray(Image.open(path).convert('RGB'))
    require(np.array_equal(back,a),'Clean PNG byte roundtrip')
    return dict(**binding(path),decoded_array_sha256=ash(back))

def txt(draw,xy,text,size=17,fill='#e3e9f2'):
    draw.text(xy,str(text),font=font(size),fill=fill)

def fmt(a):
    return '['+', '.join('U' if not np.isfinite(x) else '%.3f'%x for x in np.asarray(a).ravel())+']'

def draw_tile(canvas,xy,case,pixels,truth,models,gt=None,v61=None):
    x,y=xy;d=ImageDraw.Draw(canvas)
    d.rounded_rectangle((x,y,x+1128,y+472),radius=12,fill='#182334',outline='#465266',width=2)
    txt(d,(x+14,y+8),case['case_id']+'  global='+str(case['original_global_index']),19)
    q=case['query_xy'];qtext='missing' if q is None else ','.join('%.2f'%v for v in q)
    txt(d,(x+14,y+36),'original q=('+qtext+') L='+str(case['L'])+'; input '+('AVAILABLE' if case['original_query_available'] else 'U'),15)
    panels=[('CLEAN RGB96 x2',pixels['rgb96'],192),('valid96 x1',rgbmask(pixels['valid96']),96),
            ('SAM >0 x2',rgbmask(pixels['sam_mask96']),192),('SAM valid x1',rgbmask(pixels['sam_available96']),96)]
    if gt is not None: panels.append(('GT region blend x2',gt,192))
    px=x+14
    for title,a,side in panels:
        txt(d,(px,y+63),title,14)
        canvas.paste(Image.fromarray(a).resize((side,side),Image.Resampling.NEAREST),(px,y+85))
        px+=side+20
    if gt is None:
        txt(d,(x+680,y+120),'No certified regional GT',17)
        txt(d,(x+680,y+149),'for public real inputs',17)
    txt(d,(x+14,y+287),'Support truth V/F='+str(truth['support_target'])+'; owner T/F/BG mass='+fmt(truth['owner_soft']),16)
    txt(d,(x+14,y+310),'Owner ambiguity=%.4g invalid=%.4g; public TRAIN = apparent fit only'%(truth['owner_ambiguous_mass'],truth['owner_invalid_mass']) if case['domain']=='PUBLIC_TRAIN1480' else 'Owner ambiguity=%.4g invalid=%.4g; synthetic VAL historically reused'%(truth['owner_ambiguous_mass'],truth['owner_invalid_mass']),15)
    for k,arm in enumerate(ARMS):
        m=models[arm]
        txt(d,(x+14,y+337+k*30),arm[:3]+': logits V/F '+fmt(m['raw_support_logits'])+' -> '+str(m['support_prediction'])+'; owner '+str(m['owner_prediction'])+'; proxy '+str(m['proxy']),16)
    txt(d,(x+14,y+402),truth.get('reference_note','Public: native/ref unavailable; no new lookup or assignment.'),15)
    txt(d,(x+14,y+427),'v61 historical pixel-presence: '+(v61 or 'not available for public frame; no emulated substitute'),15)
    txt(d,(x+14,y+450),'GT/display only; T/F/BG at q != native visibility; U kept; no image used as model input.',13,fill='#b9c6d7')

def geometry_loader():
    # This module contains numpy-only pure geometry/sampling functions, no I/O.
    return runpy.run_path(str(OP))

def get_gt96(case, plan, geom, op, buffers, patches):
    r=plan[case['original_global_index']];g=geom[r['geometry_key']]
    require(r['native_slot']==case['native_slot'] and equal_qL(r,case),'GT plan original identity')
    key=g['geometry_key']
    if key not in patches: patches[key]=op['expected_patch_depth'](g['quads_world'],g['camera'],g['raw_hw'])
    for b in (r['BEFORE_buffer'],r['scene_buffer']):
        if b['path'] not in buffers:
            buffers[b['path']]=loadnpz(b,('part_id','camera_depth','global_triangle_id','ambiguity','camera_K','camera_R','camera_t','native8_CF_xy','permCF_to_canonical'))
        op['validate_buffer_metadata'](buffers[b['path']],g['camera'],g['native8_CF_xy'],g['permCF_to_canonical'])
    q=case['query_xy'];L=case['L']
    if q is None or L is None or not np.isfinite(q).all() or not np.isfinite(L) or L<=0:
        return np.full((96,96,3),COLORS[6],np.uint8),r,[-1,-1]
    classes=op['classify_end_pixels'](buffers[r['BEFORE_buffer']['path']],buffers[r['scene_buffer']['path']],patches[key],g['primary_part_id'])
    weights=op['sample_classes'](classes,op['original_crop_positions'](q,L))
    # Fractional class blend displays the exact bilinear onehot mixture; no hard
    # argmax GT or native label is created. Clean original RGB is a separate panel.
    summary=op['summarize'](weights,patches[key],case['original_query_available'])
    targets=[summary['visible_support_target'],summary['foreign_front_support_target']]
    return np.rint(weights@COLORS.astype(np.float64)).clip(0,255).astype(np.uint8).reshape(96,96,3),r,targets

def prepare():
    PREP.mkdir(exist_ok=True)
    cases=choose_cases(readrows(VAL_INPUT),readrows(PUBLIC_INPUT));writerows(CASES,cases)
    checks=[]
    def check(name,fn): fn();checks.append({'name':name,'PASS':True})
    check('fixed288_allnative8_36groups_inputonly',lambda:require(len(cases)==288,'cases'))
    def duplicate():
        try:index_unique([1,1],'toy')
        except AssertionError:return
        raise AssertionError('Duplicate accepted')
    check('duplicate_identity_failclosed',duplicate)
    def decoder():
        s=np.array([[0,0],[1,1],[-1,1],[np.nan,np.nan]],np.float32)
        o=np.array([[0,0,0],[1,2,2],[3,1,0],[np.nan]*3],np.float32)
        a=np.array([1,1,1,0],bool);sp,ow,p=decode(s,o,a)
        require(sp.tolist()==[[0,0],[1,1],[0,1],[-1,-1]] and ow.tolist()==[0,1,0,-1] and p.tolist()==[-1,1,1,-1],'Strict0/tie/U/proxy')
    check('actual_decoder_strict0_firsttie_Ufirst_foreignpriority',decoder)
    def schema_drift():
        require(not equal_qL(dict(query_xy=[1,2],L=3),dict(query_xy=[2,1],L=3)),'Swapped q accepted')
    check('q_identity_order_drift_rejected',schema_drift)
    def invented_render():
        a=np.arange(96*96*3,dtype=np.uint8).reshape(96,96,3);orig=a.copy()
        pix=dict(rgb96=a,valid96=np.ones((96,96),bool),sam_mask96=np.eye(96,dtype=bool),sam_available96=np.zeros((96,96),bool))
        truth=dict(support_target=[-1,1],owner_soft=[.1,.2,.3],owner_ambiguous_mass=.3,owner_invalid_mass=.1)
        mods={k:dict(raw_support_logits=[0,1],support_prediction=[0,1],owner_prediction=0,proxy=1) for k in ARMS}
        card=Image.new('RGB',(1160,500),'#0e1724');draw_tile(card,(12,12),cases[0],pix,truth,mods,gt=np.full((96,96,3),COLORS[4],np.uint8),v61='D (separate rule)')
        require(np.array_equal(a,orig),'Display mutated raw RGB')
        with tempfile.TemporaryDirectory(prefix='V181_invented_render_') as tmp:
            rec=png_raw(Path(tmp)/'CLEAN96.png',a)
            require(rec['decoded_array_sha256']==ash(orig),'Invented clean digest')
            card.save(Path(tmp)/'TOY_CARD.png')
    check('actual_PIL_card_cleanPNG_roundtrip_raw_unchanged',invented_render)
    def classblend():
        z=np.eye(7)[[1,2,3,4,5,6,0]];c=np.rint(z@COLORS).astype(np.uint8)
        require(np.array_equal(c,COLORS[[1,2,3,4,5,6,0]]),'Classblend')
    check('GT_bilinear_classblend_distinguishes_unknown_invalid',classblend)
    def unknown_json():
        value=json_safe(dict(soft=[np.nan,.1],scalar=np.int64(2)))
        require(json.loads(json.dumps(value,allow_nan=False))==dict(soft=[None,.1],scalar=2),'Unknown null preservation')
    check('unknown_mass_and_U_logits_serialize_null_without_imputation',unknown_json)
    code=binding(__file__)
    receipt=dict(completed=True,PASS=True,invented_only=True,checks=checks,source_pixel_GT_output_or_weight_bodies_read=0)
    writej(PREP/'GENERATED_MEANINGFUL_CHECKS.json',receipt)
    protocol=dict(completed=True,prepared_only=True,code=code,scope=binding(SCOPE),
        fixed_cases=binding(CASES),input_metadata=[binding(VAL_INPUT),binding(PUBLIC_INPUT)],
        GT_display_metadata=[binding(PLAN),binding(GEOMETRY)],pure_GT_display_operator=binding(OP),
        fit_protocol='Dynamic rootentry fit_complete binding, including final immutable trainer protocol; no old RUN or mutable preparation dependency',
        checks=binding(PREP/'GENERATED_MEANINGFUL_CHECKS.json'),
        selection='ALL fixed VAL bases128,132,144,148,160,164,176,180 x4variants xnative8=256 plus firstframe perpublic4sequences xnative8=32; allU retained',
        output='New root-entry outputdir only; 288 clean96 PNGs,36 native8 cards,18 two-card contact sheets, all288 manifest and localHTML',
        panels='Original RGB96 separately unchanged x2nearest; valid/SAM96/SAMavailability; VAL fixed-region7class float64bilinear blend; originalqL/flags, bothsupport2/owner/proxy, separate oldnative/ref metadata and historicalv61',
        no_private_held_or_new_inference=True,raw_source_RGB_reread=False,
        actual_entry='approved_actual_saved_fixed_render true; exactcode/protocol/fitcomplete/traincache/VALcache bindings; all2freeze beforeoutputs/VALrequired',
        GT_limits='Fixed constructive end patch support only; nonexclusive YES/NO/U; GTdisplay never modelinput; sourceoldD/E/S and <=3referenceproxy separate; publicnative/ref absent',
        no_outcome_selection_no_new_targets_no_adoption=True,browser_render_claim=False,actual_PNG_QA='Pending separate actual view_image of produced evidence')
    writej(PROTOCOL,protocol)
    entry=dict(approved_actual_saved_fixed_render=False,source_only=True,code=code,protocol=binding(PROTOCOL),
               fit_complete=None,training_cache_complete=None,validation_cache_complete=None,v61_seal=None,v61_predictions=None,
               output_dir=str(ROOT/'fixed_source_visual288_v1'))
    writej(PREP/'ROOT_ACTUAL_FIXED_RENDER_ENTRY_TEMPLATE_APPROVED_FALSE.json',entry)
    writej(PREP/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=code,protocol=binding(PROTOCOL),checks=protocol['checks'],fixed_cases=protocol['fixed_cases'],entry_template=binding(PREP/'ROOT_ACTUAL_FIXED_RENDER_ENTRY_TEMPLATE_APPROVED_FALSE.json'),actual_saved_body_reads=0))
    print(json.dumps(dict(completed=True,prepared_only=True,PASS=True,cases=288,groups=36,code=code,protocol=binding(PROTOCOL))))

def run(entry_path):
    entry=readj(entry_path)
    require(entry.get('approved_actual_saved_fixed_render') is True and entry.get('source_only') is True,'Root actual entry required')
    require(verified(entry['code'])==Path(__file__).resolve() and verified(entry['protocol'])==PROTOCOL,'Entry code/protocol binding')
    protocol=readj(PROTOCOL)
    for b in protocol['input_metadata']+protocol['GT_display_metadata']+[protocol['fixed_cases'],protocol['pure_GT_display_operator'],protocol['scope']]: verified(b)
    cases=readrows(CASES)
    require(cases==choose_cases(readrows(VAL_INPUT),readrows(PUBLIC_INPUT)),'Fixed selection immutable')
    fit=readj(verified(entry['fit_complete']))
    require(fit['completed'] is True and fit['PASS'] is True and fit['source_only'] is True and fit['all2_frozen_before_outputs_or_VAL_access'] is True,'Both models frozen first')
    freeze=readj(verified(fit['all2_freeze']))
    require(freeze['completed'] is True and freeze['saved_outputs_emitted'] is False and freeze['validation_cache_or_targets_read'] is False,'Freeze barrier')
    out=Path(entry['output_dir']);require(out.parent.resolve()==ROOT.resolve() and not out.exists(),'Fresh phase outputdir required')
    out.mkdir();(out/'clean96').mkdir();(out/'cards').mkdir();(out/'contacts').mkdir()
    caches={'SYNTH_VAL2048':readj(verified(entry['validation_cache_complete'])),'PUBLIC_TRAIN1480':readj(verified(entry['training_cache_complete']))}
    domains={};v61={}
    for domain,complete in caches.items():
        require(complete['completed'] is True and complete['PASS'] is True,'Cache complete')
        b=complete['domains'][domain]
        inp=loadnpz(b['inputs']);targets=loadnpz(b['targets']);rows=readrows(verified(b['metadata']))
        outputs={}
        for arm in ARMS:
            require(fit['arms'][arm]['updates']==8000,'Fixed final8000')
            outputs[arm]=loadnpz(fit['arms'][arm]['outputs'][domain],('raw_support_logits','raw_owner_logits','support_prediction','owner_prediction','regional_support_proxy_D0_E1_Uminus1','row_id','original_global_index','native_slot','query_xy','L','original_query_available'))
        mp=validate_domain(inp,targets,rows,outputs,2048 if domain=='SYNTH_VAL2048' else 1480)
        domains[domain]=(inp,targets,rows,outputs,mp)
    if entry.get('v61_seal'):
        seal=readj(verified(entry['v61_seal']))
        # Exact metadata-selected binding must be provided explicitly, never a
        # guessed/emulated source packet or a new v61 forward.
        b=entry['v61_predictions'];z=loadnpz(b,('raw_visible_logit','prediction_D0_E1_Uminus1','query_available','global_index','native_slot','query_xy','L'))
        m=index_unique(z['global_index'],'v61 global index')
        for c in cases:
            if c['domain']!='SYNTH_VAL2048':continue
            i=m[str(c['original_global_index'])]
            require(int(z['native_slot'][i])==c['native_slot'] and equal_qL(c,dict(query_xy=z['query_xy'][i],L=z['L'][i])),'v61 original qL identity')
            pred=-1 if not z['query_available'][i] else 0 if z['raw_visible_logit'][i]>=0 else 1
            require(pred==int(z['prediction_D0_E1_Uminus1'][i]),'v61 fixed>=0 rule')
            v61[c['case_id']]='%s (>=0 D; historical pixel presence only)'%({-1:'U',0:'D',1:'E'}[pred])
        # Explicit binding must also occur within inherited seal text.
        require(b['sha256'] in json.dumps(seal),'v61 packet not in inherited seal')
    plan={r['original_global_index']:r for r in readrows(PLAN)}
    geom={r['geometry_key']:r for r in readrows(GEOMETRY)};op=geometry_loader();buffers={};patches={}
    ledger=[];cards=[]
    for group in sorted({c['group_id'] for c in cases}):
        groupcases=[c for c in cases if c['group_id']==group]
        card=Image.new('RGB',(2320,2160),'#0b1320');d=ImageDraw.Draw(card)
        txt(d,(20,12),group+' | all native0..7, outcome-blind fixed evidence',29)
        txt(d,(20,54),'Raw96 x2 nearest; valid x1; SAM>0 x2; SAM availability x1. GT is display/supervision only.',20)
        txt(d,(20,84),'Support V/F nonexclusive; proxy foreign>0 E else visible>0 D else U. Owner T/F/BG at q differs from native visibility.',18)
        for k,c in enumerate(groupcases):
            inp,targ,rows,mods,mp=domains[c['domain']];i=mp[str(c['original_global_index'])]
            require(int(inp['native_slot'][i])==c['native_slot'] and equal_qL(c,dict(query_xy=inp['query_xy'][i],L=inp['L'][i])),'Selected identity')
            require(bool(inp['original_query_available'][i])==c['original_query_available'],'Original input U unchanged')
            pix={n:inp[n][i] for n in ('rgb96','valid96','sam_mask96','sam_available96')}
            clean=png_raw(out/'clean96'/(c['case_id']+'.png'),pix['rgb96'])
            truth={n:(targ[n][i].tolist() if np.asarray(targ[n][i]).ndim else float(targ[n][i])) for n in targ}
            gt=None;reference=None
            if c['domain']=='SYNTH_VAL2048':
                gt,reference,derived_support=get_gt96(c,plan,geom,op,buffers,patches)
                require(derived_support==truth['support_target'],'GT display operator agrees with bound support targets')
                truth['reference_note']='Old declared native='+reference['original_semantic_state']+'; q-ref=%.2f px (%s), source proxy only.'%(reference['q_reference_error_px'],reference['position_band']) if reference['q_reference_error_px'] is not None else 'Old native/ref diagnostic missing; retained U.'
            modelrows={a:dict(raw_support_logits=z['raw_support_logits'][i].tolist(),raw_owner_logits=z['raw_owner_logits'][i].tolist(),support_prediction=z['support_prediction'][i].tolist(),owner_prediction=int(z['owner_prediction'][i]),proxy=int(z['regional_support_proxy_D0_E1_Uminus1'][i])) for a,z in mods.items()}
            draw_tile(card,(20+(k%2)*1140,125+(k//2)*480),c,pix,truth,modelrows,gt,v61.get(c['case_id']))
            rec=dict(c);rec.update(cache_index=i,row_id=str(inp['row_id'][i]),clean96=clean,rgb96_array_sha256=ash(pix['rgb96']),valid96_array_sha256=ash(pix['valid96']),sam96_array_sha256=ash(pix['sam_mask96']),sam_available96_array_sha256=ash(pix['sam_available96']),support_truth=truth,models=modelrows,reference_diagnostic=reference,GT_display_present=gt is not None)
            ledger.append(rec)
        yy=2060
        for j,name in enumerate(op['CLASSES']):
            xx=20+(j%4)*575;ty=yy+(j//4)*34
            d.rectangle((xx,ty,xx+22,ty+22),fill=tuple(COLORS[j]));txt(d,(xx+30,ty),name,15)
        p=out/'cards'/(group+'_CARD.png');card.save(p);cards.append(dict(group_id=group,case_ids=[c['case_id'] for c in groupcases],image=binding(p)))
    contacts=[]
    for i in range(0,len(cards),2):
        sheet=Image.new('RGB',(2320,4320),'#0b1320')
        for j,rec in enumerate(cards[i:i+2]):
            with Image.open(rec['image']['path']) as im:sheet.paste(im,(0,j*2160))
        p=out/'contacts'/('CONTACT_%02d.png'%(i//2));sheet.save(p);contacts.append(binding(p))
    writerows(out/'ALL288_FIXED_VISUAL_CASES.jsonl',ledger);writej(out/'CARDS_AND_CONTACTS.json',dict(cards=cards,contacts=contacts))
    page='<!doctype html><meta charset="utf-8"><title>Fixed source 288 evidence</title><style>body{font:17px sans-serif;background:#101827;color:#e5edf8;margin:24px}img{max-width:100%;height:auto}a{color:#9bcbff}section{margin:30px 0}details{margin:24px 0}</style><h1>고정 source 288개 입력·출력</h1><p>합성 VALIDATION 256개(32 scene)와 공개 TRAIN 32개(4 frame)를 결과와 무관하게 모두 보존했습니다. RGB96은 원본 픽셀 그대로 별도 패널에 표시합니다. support와 q 소속·원 native 의미는 서로 다른 진단입니다. 공개 TRAIN 출력은 apparent fit이며 synthetic VAL도 반복 사용된 자료입니다. 모델 선택·새 정답·성공 판정은 없습니다.</p><p>PNG nearest 배율은 카드에 명시. GT 색 혼합은 원96 bilinear onehot이며 RGB나 모델 입력이 아닙니다. HTML 브라우저 렌더 검증 주장은 없습니다. 실제 PNG view QA는 별도 영수증으로 기록합니다.</p><p><a href="ALL288_FIXED_VISUAL_CASES.jsonl">all288 manifest</a> | <a href="CARDS_AND_CONTACTS.json">cards/contact bindings</a></p>'
    for rec in cards:
        rel='cards/'+Path(rec['image']['path']).name
        page+='<section><h2>'+html.escape(rec['group_id'])+'</h2><a href="'+rel+'"><img loading="lazy" src="'+rel+'"></a></section>'
    page+='<details><summary>전체 18 contact sheets</summary>'+''.join('<p><a href="contacts/'+Path(b['path']).name+'">'+Path(b['path']).name+'</a></p>' for b in contacts)+'</details>'
    (out/'index.html').write_text(page)
    require(len(ledger)==288 and len(cards)==36 and len(contacts)==18,'Full fixed evidence completeness')
    writej(out/'COMPLETE.json',dict(completed=True,PASS=True,source_only=True,actual_entry=binding(entry_path),code=binding(__file__),protocol=binding(PROTOCOL),fit_complete=entry['fit_complete'],cases=288,synthetic_VAL=256,public_TRAIN=32,cards=36,two_card_contact_sheets=18,clean96_PNGs=288,clean_png_decoded_array_parity=True,all_unknown_and_input_U_preserved=True,manifest=binding(out/'ALL288_FIXED_VISUAL_CASES.jsonl'),card_contact_inventory=binding(out/'CARDS_AND_CONTACTS.json'),index=binding(out/'index.html'),actual_PNG_view_QA='PENDING separate tool view_image receipt',browser_render_verified=False,no_model_run_or_new_labels=True,selected_model=None,fully_solved=False))
    print(json.dumps(dict(completed=True,PASS=True,cases=288,cards=36,contacts=18,complete=binding(out/'COMPLETE.json'))))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--root-entry',type=Path);args=ap.parse_args()
    if args.prepare: prepare()
    else:
        require(args.root_entry is not None,'Use --prepare or separately approved --root-entry')
        run(args.root_entry)
