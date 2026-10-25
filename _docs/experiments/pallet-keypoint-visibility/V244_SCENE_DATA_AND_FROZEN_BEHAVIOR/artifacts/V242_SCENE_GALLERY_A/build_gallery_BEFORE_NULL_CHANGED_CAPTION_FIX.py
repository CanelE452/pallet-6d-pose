"""V242 pre-fit SOURCE scene-CF gallery. No model or new teacher is computed.

Separate exact Root entry; fixed case IDs precede any source/cache pixel access.
"""
import argparse,ast,csv,hashlib,html,io,json,os,traceback,types,zipfile
from pathlib import Path
import numpy as np
from PIL import Image

H=Path(__file__).absolute().parent
R=H.parent
ROOT_PROTOCOL_SHA='03177adde9dca4e91e04663bd0a7a1976d176a876c114fb67844acc6f27ecc2c'
ROLES=('BEFORE','COVER','ADJACENT_SHAM')

def lexical(value):
    value=str(value)
    if any(s in value.casefold() for s in ('v112','__pycache__','.pyc','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation')):
        raise ValueError('Forbidden path before filesystem access')
    p=Path(value)
    if not p.is_absolute() or '..' in p.parts:raise ValueError('Exact absolute path without traversal required')
    return p

def safe(value):
    p=lexical(value)
    for q in reversed((p,*p.parents)):
        if q.is_symlink():raise ValueError('Symlink path forbidden')
    return p

def identity(b):return str(lexical(b['path'])),b['sha256']

def read(path):
    with os.fdopen(os.open(safe(path),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()

def checked(b,white):
    if identity(b) not in {identity(x) for x in white}:raise ValueError('Exact whitelist required before FS')
    p=safe(b['path']);digest=hashlib.sha256();size=0
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block);size+=len(block)
    if digest.hexdigest()!=b['sha256'] or (b.get('size_bytes') is not None and b['size_bytes']!=size):raise ValueError('Changed bound bytes')
    return p

def binding(path):
    raw=read(path);return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}

def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [plain(v) for v in value]
    if isinstance(value,(np.integer,np.bool_)):return value.item()
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value

def write_json(path,value):
    with os.fdopen(os.open(safe(path),os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),'w') as f:
        json.dump(plain(value),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return binding(path)

def select_cases(rows):
    """No success-only filter, no score/ranking/substitution; all four buckets retained."""
    seen=set()
    for i,r in enumerate(rows):
        if int(r['pair_index'])!=i or int(r['parent_sample_index']) in seen or type(r['proof_pass']) is not bool:raise ValueError('Unique ordered original parent pairs / bool proof')
        seen.add(int(r['parent_sample_index']))
        if r['original_input']['fit'] is not True or int(r['original_input']['target'])!=0:raise ValueError('Original FIT D only')
    out=[]
    for family in (0,1):
        for proof_pass in (True,False):
            candidates=[r for r in rows if int(r['original_input']['family_code'])==family and bool(r['proof_pass'])==proof_pass]
            chosen=min(candidates,key=lambda r:int(r['parent_sample_index'])) if candidates else None
            out.append({'case_number':len(out)+1,'family':family,'proof_pass':proof_pass,'eligible_population':len(candidates),
                        'pair_index':None if chosen is None else int(chosen['pair_index']),
                        'parent_sample_index':None if chosen is None else int(chosen['parent_sample_index']),'missing':chosen is None})
    return out

def load_code(b,white,names=None):
    raw=read(checked(b,white));tree=ast.parse(raw)
    if names is not None:
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes}!=set(names):raise ValueError('Exact named pure CODE functions')
        tree=ast.Module(body=nodes,type_ignores=[])
    ns={'__name__':'approved_gallery_pure_helpers','np':np};exec(compile(tree,b['path'],'exec'),ns)
    return types.SimpleNamespace(**ns)

def decode_scene(image,white,raw_hw):
    """Only selected public RGB; direct requested SYN member, no other NPZ arrays."""
    p=checked(image,white)
    if image['format']=='npz':
        if image.get('member')!='RGB_uint8':raise ValueError('Only original RGB_uint8 member')
        with zipfile.ZipFile(p) as z:
            with z.open('RGB_uint8.npy') as f:scene=np.load(io.BytesIO(f.read()),allow_pickle=False)
    elif image['format']=='png':
        with Image.open(p) as im:scene=np.asarray(im.convert('RGB')).copy()
    else:raise ValueError('No alternate RGB route')
    if scene.dtype!=np.uint8 or scene.shape!=(*map(int,raw_hw),3):raise ValueError('Exact original HWC scene dimensions')
    return scene

def save_rgb(path,chw):
    if chw.dtype!=np.uint8 or chw.shape!=(3,96,96):raise ValueError('Original uint8 CHW96 required')
    Image.fromarray(chw.transpose(1,2,0),'RGB').save(safe(path))
    with Image.open(path) as im:restored=np.ascontiguousarray(np.asarray(im.convert('RGB')).transpose(2,0,1))
    if array_sha(restored)!=array_sha(chw):raise ValueError('RAW96 PNG CHW roundtrip')
    return {'PNG':binding(path),'original_CHW_sha256':array_sha(chw),'decoded_CHW_sha256':array_sha(restored)}

def display_context(chw):
    if chw.dtype!=np.float32 or chw.shape!=(3,224,224) or not np.isfinite(chw).all():raise ValueError('Context FP32 CHW224')
    return np.clip(np.floor(chw.transpose(1,2,0).astype(np.float64)+.5),0,255).astype(np.uint8)

def marker(ax,xy,color,label,shape):
    xy=np.asarray(xy,np.float64)
    if xy.shape!=(2,) or not np.isfinite(xy).all():return
    shown=np.clip(xy,[0,0],[shape[1]-1,shape[0]-1])
    ax.scatter([shown[0]],[shown[1]],marker='+',s=65,color=color)
    ax.annotate(label,shown,xytext=(3,4),textcoords='offset points',color=color,fontsize=7,
                bbox={'facecolor':'black','alpha':.4,'edgecolor':'none','pad':.3})

def render_card(path,case,row,group,packets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    active=int(row['active_slot']);q8=np.asarray(group['q8'],np.float64);L8=np.asarray(group['L8'],np.float64)
    avail=np.asarray(group['original_available8'],bool);box=group['box_xyxy'];ref=row['teacher_reference_xy']
    fig=plt.figure(figsize=(20,15),dpi=120);outer=fig.add_gridspec(1,3,wspace=.18)
    for ri,role in enumerate(ROLES):
        sub=outer[ri].subgridspec(5,4,height_ratios=[1.4,1.1,.8,.8,.7],hspace=.52,wspace=.15)
        packet=packets.get(role)
        if packet is None:
            ax=fig.add_subplot(sub[:,:]);ax.axis('off');ax.text(.5,.5,role+'\nUNCONSTRUCTED / UNAVAILABLE\nTarget UNKNOWN; original D retained',ha='center',va='center',fontsize=15);continue
        nodes=packet['NODE_RGB96'];valid=packet['NODE_VALID96'];context=packet['CONTEXT_RGB224_F32']
        ax=fig.add_subplot(sub[0,:]);ax.imshow(display_context(context),interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
        if box is not None:
            b=np.asarray(box,np.float64)
            if b.shape==(4,) and np.isfinite(b).all() and (b[2:]>b[:2]).all():
                for j,q in enumerate(q8):
                    if np.isfinite(q).all() and not np.array_equal(q,[-1,-1]):marker(ax,(q-b[:2])/(b[2:]-b[:2])*224-.5,'yellow' if j==active else 'cyan',f'q{j}'+(' U' if not avail[j] else ''),(224,224))
                if ref is not None:marker(ax,(np.asarray(ref)-b[:2])/(b[2:]-b[:2])*224-.5,'red','r TEACHER DIAGNOSTIC',(224,224))
        ax.set_title(role+' | same-scene context224\nDisplay-only quantization; clipped markers',fontsize=11)
        ax=fig.add_subplot(sub[1,:2]);ax.imshow(nodes[active].transpose(1,2,0),interpolation='nearest');marker(ax,[47.5,47.5],'yellow','original q',(96,96))
        if ref is not None:marker(ax,(np.asarray(ref)-q8[active])*96/L8[active]+47.5,'red','r diagnostic',(96,96))
        ax.set_xticks([]);ax.set_yticks([]);ax.set_title('Active original q/L RAW96',fontsize=10)
        ax=fig.add_subplot(sub[1,2:]);ax.imshow(valid[active],vmin=0,vmax=1,cmap='gray');ax.set_xticks([]);ax.set_yticks([]);ax.set_title('Input validity, NOT visibility',fontsize=10)
        support=avail&valid[:,47:49,47:49].all((1,2))&valid.any((1,2))
        for j in range(8):
            ax=fig.add_subplot(sub[2+j//4,j%4]);ax.imshow(nodes[j].transpose(1,2,0),interpolation='nearest');ax.scatter([47.5],[47.5],marker='+',color='yellow' if j==active else 'cyan',s=25)
            ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f'slot{j} av={int(avail[j])}\nsupport={int(support[j])} L={L8[j]:.1f}',fontsize=8)
        target=0 if role=='BEFORE' else row['targets_D0_E1_Uminus1'][ri-1]
        ax=fig.add_subplot(sub[4,:]);ax.axis('off')
        text=f"target {'D' if target==0 else 'E' if target==1 else 'UNKNOWN'} at SOURCE reference; NOT q-centre\nproof_pass={row['proof_pass']} constructed={row['constructed']}\n{row['status']}\nchanged={row.get('changed_pixels',{}).get(role,{})}"
        ax.text(0,1,text,va='top',fontsize=8.5,wrap=True)
    family='SYN witness0 MAIN' if case['family']==0 else 'YCB associated-r AUX / native UNKNOWN'
    fig.suptitle(f"V242 pre-fit | case{case['case_number']} parent{case['parent_sample_index']} | {family}\nproof bucket={case['proof_pass']} population={case['eligible_population']} | original q/L/native/U preserved",fontsize=14,y=.99)
    fig.text(.5,.014,'SOURCE FIT selected before any model output | red r is teacher-only display, never model input/recenter | opacity/support is not whole-part/native gold | no new forward',ha='center',fontsize=9)
    fig.subplots_adjust(top=.92,bottom=.06,left=.025,right=.98);fig.savefig(safe(path));plt.close(fig)

def build(entry_path):
    e=json.loads(read(entry_path))
    if e.get('approved_V242_SOURCE_gallery') is not True or e.get('source_only') is not True or e.get('automatic_retry') is not False:raise ValueError('Separate exact Root gallery approval required')
    meta,codes,payload=e['metadata_whitelist'],e['code_whitelist'],e['payload_whitelist']
    checked(e['code'],codes);checked(e['implementation_protocol'],codes);checked(e['root_scope'],meta)
    if identity(e['code'])!=(str(safe(__file__)),binding(__file__)['sha256']) or e['root_protocol']['sha256']!=ROOT_PROTOCOL_SHA:raise ValueError('Exact frozen gallery/prospective policy')
    checked(e['root_protocol'],meta)
    complete=json.loads(read(checked(e['dataset_complete'],meta)));qa=json.loads(read(checked(e['data_QA'],meta)))
    if not all(d.get('completed') is True and d.get('PASS') is True for d in (complete,qa)):raise ValueError('Actual data COMPLETE and independent QA required')
    if identity(qa['dataset_complete'])!=identity(e['dataset_complete']):raise ValueError('Independent QA must bind this actual dataset')
    if complete['role_order']!=['COVER','ADJACENT_SHAM'] or complete['training_executed'] is not False:raise ValueError('Data only / exact variant order')
    if identity(e['groups_metadata'])!=identity(complete['metadata']['groups']):raise ValueError('Exact closed original group metadata')
    for key,dep in [('V90_crop_CODE','crop90'),('V107_context_CODE','context107'),('V237_lift_CODE','scene_math'),('V241_math_CODE','scene_cf_math')]:
        if identity(e[key])!=identity(complete['code_dependencies'][dep]):raise ValueError('Same frozen dataset pure math code')
    if identity(e['pair_metadata'])!=identity(complete['outputs']['PAIR_METADATA.jsonl']):raise ValueError('Exact saved pair metadata')
    rows=[json.loads(s) for s in read(checked(e['pair_metadata'],meta)).splitlines()]
    if len(rows)!=5406:raise ValueError('All original FIT D5406 retained')
    cases=select_cases(rows);selected=[c['pair_index'] for c in cases if not c['missing']]
    out=safe(e['output_directory'])
    if not out.is_relative_to(H) or out.exists():raise ValueError('Fresh own gallery output required')
    out.mkdir();selection=write_json(out/'CASE_IDS_BEFORE_PIXELS.json',{'completed':True,'PASS':True,'dataset_complete':e['dataset_complete'],'data_QA':e['data_QA'],'cases':cases,'selection':'minimum parent_sample_index per family x proof_pass true/false; no substitute','sealed_before_source_or_cache_pixel_hash_header_decode':True})
    try:
        # All subsequent reads remain restricted to selected public scenes and selected stored pair/group rows.
        groups={int(g['group_index']):g for g in (json.loads(s) for s in read(checked(e['groups_metadata'],meta)).splitlines())}
        crop90=load_code(e['V90_crop_CODE'],codes,{'crop'}).crop;context107=load_code(e['V107_context_CODE'],codes)
        scene_math=load_code(e['V237_lift_CODE'],codes);pure=load_code(e['V241_math_CODE'],codes)
        layout={'CF_NODE_RGB96.npy':(np.uint8,(5406,2,8,3,96,96)),
                'CF_CONTEXT_RGB224_F32.npy':(np.float32,(5406,2,3,224,224)),
                'GROUP_NODE_VALID96.npy':(np.bool_,(1198,8,96,96)),
                'GROUP_CONTEXT_VALID224.npy':(np.bool_,(1198,224,224)),
                'PAIR_LAYER_RGB96.npy':(np.uint8,(5406,2,3,96,96)),
                'PAIR_ALPHA_PACKBITS.npy':(np.uint8,(5406,2,1152))}
        arrays={}
        for name,(dtype,shape) in layout.items():
            b=complete['outputs'][name];a=np.load(checked(b,payload),mmap_mode='r',allow_pickle=False)
            if a.dtype!=dtype or a.shape!=shape:raise ValueError('Exact stored array header')
            arrays[name]=a
        outputs=[selection];data=[];native_count=0
        for case in cases:
            item=dict(case)
            if case['missing']:data.append(item);continue
            row=rows[case['pair_index']];group=groups[int(row['group_index'])];active=int(row['active_slot']);j=int(row['pair_index']);gc=int(row['group_cache_index'])
            image=group['image'];base=decode_scene(image,e['source_image_whitelist'],group['raw_hw'])
            before=pure.resample_all(base,group,crop90,context107);packets={'BEFORE':before};scenes={'BEFORE':base}
            if array_sha(base)!=row['native_scene_sha256']['BEFORE']:raise ValueError('Decoded BEFORE scene bytes differ')
            for key,value in [('RGB96_CHW',before['NODE_RGB96'][active]),('valid96_bool',before['NODE_VALID96'][active])]:
                if array_sha(value)!=row['expected_BEFORE_'+key+'_sha256'] or array_sha(value)!=row['actual_BEFORE_'+key+'_sha256']:raise ValueError('Exact stored original active input replay')
            if row['constructed']:
                for k,role in enumerate(ROLES[1:]):
                    layer=np.array(arrays['PAIR_LAYER_RGB96.npy'][j,k],copy=True)
                    alpha=np.unpackbits(np.array(arrays['PAIR_ALPHA_PACKBITS.npy'][j,k],copy=True),bitorder='big').reshape(96,96).astype(bool)
                    if array_sha(layer)!=row['carrier_RGB96_CHW_sha256'][k] or array_sha(alpha)!=row['alpha_bool_sha256'][k]:raise ValueError('Exact saved carrier/alpha')
                    drgb,na=scene_math.lift_raw96_layer(layer.transpose(1,2,0),alpha,np.asarray(group['q8'][active],np.float64),np.float64(group['L8'][active]),base.shape[:2])
                    scene=base.copy();scene[na]=drgb[na]
                    if array_sha(scene)!=row['native_scene_sha256'][role]:raise ValueError('Exact stored native scene reconstruction')
                    scenes[role]=scene
                    packet=pure.resample_all(scene,group,crop90,context107)
                    stored_node=np.array(arrays['CF_NODE_RGB96.npy'][j,k],copy=True);stored_ctx=np.array(arrays['CF_CONTEXT_RGB224_F32.npy'][j,k],copy=True)
                    if not np.array_equal(packet['NODE_RGB96'],stored_node) or not np.array_equal(packet['CONTEXT_RGB224_F32'],stored_ctx):raise ValueError('Stored same-scene input replay differs')
                    packets[role]=packet
            if not np.array_equal(before['NODE_VALID96'],arrays['GROUP_NODE_VALID96.npy'][gc]) or not np.array_equal(before['CONTEXT_VALID224'],arrays['GROUP_CONTEXT_VALID224.npy'][gc]):raise ValueError('Original stored support replay differs')
            folder=out/f"CASE_{case['case_number']:02d}";folder.mkdir();raw=[]
            for role,packet in packets.items():
                proof=save_rgb(folder/(role+'_ACTIVE_RAW96.png'),packet['NODE_RGB96'][active]);outputs.append(proof['PNG']);raw.append({'role':role,**proof})
                if e.get('save_selected_native_PNGs',False):
                    p=folder/(role+'_NATIVE_SCENE.png');Image.fromarray(scenes[role],'RGB').save(p)
                    with Image.open(p) as im:roundtrip=np.asarray(im.convert('RGB')).copy()
                    if array_sha(roundtrip)!=array_sha(scenes[role]):raise ValueError('Selected native PNG roundtrip')
                    outputs.append(binding(p));native_count+=1
            card=folder/'CARD.png';render_card(card,item,row,group,packets);outputs.append(binding(card))
            item.update(row_id=row['original_input']['row_id'],group_index=row['group_index'],active_slot=active,status=row['status'],constructed=row['constructed'],targets=row['targets_D0_E1_Uminus1'],teacher_reference_xy=row['teacher_reference_xy'],teacher_authority=row['teacher_authority'],proof=row['proof'],changed_pixels=row.get('changed_pixels'),original_q8=group['q8'],original_L8=group['L8'],original_available8=group['original_available8'],source_image_binding=image,active_PNG_proofs=raw,card=binding(card),native_scene_sha256={r:array_sha(a) for r,a in scenes.items()},replayed_selected_inputs_exact=True)
            data.append(item)
        if native_count>12:raise AssertionError('At most4x3 selected native images')
        outputs.append(write_json(out/'CASE_DATA.json',{'completed':True,'PASS':True,'cases':data,'teacher_display_only':True,'native_gold':False,'training_or_model_forward':0}))
        cols=['case_number','family','proof_pass','eligible_population','pair_index','parent_sample_index','missing','row_id','group_index','active_slot','status','constructed']
        p=out/'CASES.csv'
        with p.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows({k:r.get(k) for k in cols} for r in data)
        outputs.append(binding(p))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(14,16),dpi=110)
        for ax,item in zip(axes.flat,data):
            ax.axis('off')
            if item['missing']:ax.text(.5,.5,f"family{item['family']} proof={item['proof_pass']}\nEMPTY bucket / no substitute",ha='center',va='center')
            else:
                with Image.open(item['card']['path']) as im:ax.imshow(np.asarray(im).copy())
        fig.tight_layout();contact=out/'FIXED4_SCENE_CF.png';fig.savefig(contact);plt.close(fig);outputs.append(binding(contact))
        body=['<!doctype html><meta charset="utf-8"><title>V242 SOURCE scene CF pre-fit</title><style>body{font:17px sans-serif;max-width:1800px;margin:25px auto}img{max-width:100%}section{border-top:1px solid #aaa;margin-top:25px}</style><h1>V242 SOURCE scene CF: fixed pre-fit examples</h1><p>Family x renewed proof PASS/UNKNOWN minimum original parent index. Empty buckets preserved. No model outputs or case substitution. Original q/L/native/U unchanged; red r is teacher-only diagnostic overlay, never a model input or recrop. SOURCE synthetic teacher footprint is not literal q-centre presence, whole-part or native gold.</p><p><a href="CASE_IDS_BEFORE_PIXELS.json">IDs sealed before pixels</a> · <a href="CASES.csv">cases CSV</a> · <a href="CASE_DATA.json">proof/identity details</a></p><img src="FIXED4_SCENE_CF.png" alt="fixed four buckets">']
        for item in data:
            body.append('<section><h2>'+html.escape(f"family{item['family']} proof={item['proof_pass']} N={item['eligible_population']}")+'</h2>')
            if item['missing']:body.append('<p>Empty; no substitute.</p>')
            else:body.append('<img src="'+str(Path(item['card']['path']).relative_to(out))+'" alt="original cover sham scene/context/crops">')
            body.append('</section>')
        p=out/'PORTABLE_GALLERY_KO.html';p.write_text('\n'.join(body));outputs.append(binding(p))
        outputs.append(write_json(out/'IMAGE_MANIFEST.json',{'completed':True,'PASS':True,'selection':selection,'images':[b for b in outputs if b['path'].endswith('.png')],'source_teacher_overlay_only':True,'actual_model_inputs_modified':False,'new_forward_or_training':0}))
        complete_out={'completed':True,'PASS':True,'entry':binding(entry_path),'code':e['code'],'dataset_complete':e['dataset_complete'],'data_QA':e['data_QA'],'selection':selection,'cases':4,'nonempty_cases':len(selected),'missing_buckets':4-len(selected),'selected_parent_sample_indices':[c['parent_sample_index'] for c in cases if not c['missing']],'selected_native_images_saved':native_count,'outputs':outputs,'source_RAW_only_selected_scenes':True,'new_GT_model_features_or_protected_access':0,'new_teacher_labels_or_fit':0,'automatic_retry':False}
        write_json(out/'COMPLETE.json',complete_out);print(json.dumps({'completed':True,'PASS':True,'COMPLETE':binding(out/'COMPLETE.json')}),flush=True)
    except BaseException:
        write_json(out/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()});raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--entry',required=True);build(parser.parse_args().entry)
