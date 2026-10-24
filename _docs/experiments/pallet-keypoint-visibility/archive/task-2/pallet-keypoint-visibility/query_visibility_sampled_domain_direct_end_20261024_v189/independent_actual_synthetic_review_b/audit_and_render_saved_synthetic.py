"""V189 independent sampled-domain saved-array audit + complete display. No producer import."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,html,json,math,os,stat,sys,time,traceback
import numpy as np
from PIL import Image,ImageDraw,ImageFont

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
C=V/'direct_end_training_preparation_c';ROOT=V/'synthetic_sampled_end_controls_v1'
PROTO=H/'PROTOCOL_BEFORE_ACTUAL_SYNTHETIC_AUDIT.json';OUT=H/'actual_review_v1'
VIS=V/'synthetic_end_review_visuals_b'
VARIANTS=('BEFORE','COVER','SHAM')
KEYS={'full_RGB','rgb96','valid96','query_xy','L','native_slot','original_global_index','original_occurrence_id','original_query_available','direct_end_external_target','new_native_visibility_target','new_quality_target','new_owner_target','broad_support2_diagnostic','foreign_alpha','end_related96_mask','variant'}
SIGNS=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],np.int8)


def need(v,msg):
    if not bool(v):raise AssertionError(msg)


def opened(value):
    p=Path(value);need(p.is_absolute() and '..' not in p.parts and all('v112' not in x.casefold() for x in p.parts),'Lexical safe path before stat')
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        last=os.open(p.parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(last).st_mode),'Regular only')
    return os.fdopen(last,'rb')


def data(path):
    with opened(path) as f:return f.read()


def bind(path):return {'path':str(path),'sha256':hashlib.sha256(data(path)).hexdigest()}
def checked(b):
    raw=data(b['path']);need(hashlib.sha256(raw).hexdigest()==b['sha256'],'SHA drift '+b['path']);return raw
def jd(b):return json.loads(checked(b))
def jl(b):return [json.loads(s) for s in checked(b).splitlines()]
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def write(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def arrays(b,keys):
    with opened(b['path']) as f:
        h=hashlib.sha256()
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
        need(h.hexdigest()==b['sha256'],'Array file SHA drift');f.seek(0)
        with np.load(f,allow_pickle=False) as z:
            need(set(keys)<=set(z.files),'Required arrays')
            return {k:z[k].copy() for k in keys},set(z.files)


def crop(rgb,q,L):
    h,w=rgb.shape[:2]
    if q is None or L is None:return np.zeros((96,96,3),np.uint8),np.zeros((96,96),bool)
    off=((np.arange(96,dtype=np.float64)+.5)/96-.5)*L
    x,y=np.meshgrid(q[0]+off,q[1]+off);ix=np.floor(x).astype(np.int64);iy=np.floor(y).astype(np.int64);fx=x-ix;fy=y-iy
    value=np.zeros((96,96,3),np.float64)
    for dx,dy,weight in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)):
        xx,yy=ix+dx,iy+dy;inside=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
        value[inside]+=rgb[yy[inside],xx[inside]].astype(np.float64)*weight[inside,None]
    valid=(x>=-.5)&(x<w-.5)&(y>=-.5)&(y<h-.5)
    result=np.floor(np.clip(value,0,255)+.5).astype(np.uint8);result[~valid]=0
    return result,valid


def footprint(q,L,hw):
    h,w=hw;result=set()
    for j in range(96):
        y=q[1]+((j+.5)/96-.5)*L
        for i in range(96):
            x=q[0]+((i+.5)/96-.5)*L
            if not(-.5<=x<w-.5 and -.5<=y<h-.5):continue
            ix,iy=math.floor(x),math.floor(y);fx,fy=x-ix,y-iy
            for dx,dy,v in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)):
                if v>0 and 0<=ix+dx<w and 0<=iy+dy<h:result.add((ix+dx,iy+dy))
    return result


def sampled_mask(q,L,hw,ep):
    h,w=hw;out=np.zeros((96,96),bool)
    for j in range(96):
        y=q[1]+((j+.5)/96-.5)*L
        for i in range(96):
            x=q[0]+((i+.5)/96-.5)*L
            if not(-.5<=x<w-.5 and -.5<=y<h-.5):continue
            ix,iy=math.floor(x),math.floor(y);fx,fy=x-ix,y-iy
            out[j,i]=any(weight>0 and 0<=ix+ax<w and 0<=iy+ay<h and (ix+ax,iy+ay) in ep
                         for ax,ay,weight in ((0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)))
    return out


def sampled_extent_contains(q,L,box):
    lo=[z+(.5/96-.5)*L for z in q];hi=[z+(95.5/96-.5)*L for z in q]
    return lo[0]<=box[0]<=box[2]<=hi[0] and lo[1]<=box[1]<=box[3]<=hi[1]


def declared_quads(cfg,k):
    c=int(cfg['permCF_to_canonical'][k]);sg=SIGNS[c];dims=np.array([1.2,.144,1.0]);anchor=sg*dims/2
    n=5 if cfg['template_index']==0 else 7;top=.022 if n==5 else .018;bottom=.020 if n==5 else .018;bw=.140 if n==5 else .110
    if sg[1]<0:
        zl,zh=(-.5,-.5+bw) if sg[2]<0 else (.5-bw,.5)
        low=np.array([-.6,-.072,zl]);high=np.array([.6,-.072+top,zh]);pid=0 if sg[2]<0 else n-1
    else:
        xl,xh=(-.6,-.6+bw) if sg[0]<0 else (.6-bw,.6)
        low=np.array([xl,.072-bottom,-.5]);high=np.array([xh,.072,.5]);pid=n+9+(0 if sg[0]<0 else 2)
    step=min(.030,float(np.min(high-low))*.7);broad=[]
    for axis in range(3):
        a,b=[j for j in range(3) if j!=axis];u=np.zeros(3);v=np.zeros(3);u[a]=-sg[a]*step;v[b]=-sg[b]*step
        broad.append(np.array([anchor,anchor+u,anchor+u+v,anchor+v]))
    scale=np.asarray(cfg['positive_scale_xyz'],np.float64);broad=np.asarray(broad)*scale;anchor=anchor*scale
    need(np.array_equal(anchor,np.asarray(cfg['native8_canonical_xyz'][c],np.float64)),'Independent original canonical anchor')
    return anchor+.2*(broad-anchor),broad,pid


def projection(quads,cfg):
    cam=cfg['camera'];R=np.asarray(cam['R']);t=np.asarray(cam['t']);K=np.asarray(cam['K'])
    pc=quads@R.T+t;hp=pc@K.T;return hp[...,:2]/hp[...,2,None],pc


def interior_patch_depth(quads,cfg,hw):
    """Independent bounded scalar-ray/Gram intersections; boundary separately kept."""
    xy,pc=projection(quads,cfg);K=np.asarray(cfg['camera']['K']);inv=np.linalg.inv(K);h,w=hw
    depths={};boundary=0
    for face,uv in zip(pc,xy):
        a=face[1]-face[0];b=face[3]-face[0];normal=np.cross(a,b)
        aa=float(a@a);ab=float(a@b);bb=float(b@b);det=aa*bb-ab*ab
        x0=max(0,math.floor(float(uv[:,0].min())));x1=min(w-1,math.ceil(float(uv[:,0].max())))
        y0=max(0,math.floor(float(uv[:,1].min())));y1=min(h-1,math.ceil(float(uv[:,1].max())))
        for y in range(y0,y1+1):
            for x in range(x0,x1+1):
                ray=inv@np.array([x,y,1.]);den=float(ray@normal)
                if den==0:continue
                hit=ray*(float(normal@face[0])/den);d=hit-face[0];ad=float(d@a);bd=float(d@b)
                u=(ad*bb-bd*ab)/det;v=(bd*aa-ad*ab)/det
                if hit[2]<=.001 or not(-1e-10<=u<=1+1e-10 and -1e-10<=v<=1+1e-10):continue
                if min(u,v,1-u,1-v)<=1e-10:boundary+=1;continue
                if (x,y) not in depths or hit[2]<depths[x,y]:depths[x,y]=float(hit[2])
    return depths,boundary


def expected_alpha(before,box,z):
    h,w=before['part_id'].shape;x0,y0,x1,y1=box;need(0<=x0<=x1<w and 0<=y0<=y1<h,'Plane bbox in frame')
    mask=np.zeros((h,w),bool);mask[y0:y1+1,x0:x1+1]=True;dep=before['camera_depth'];finite=np.isfinite(dep)
    tol=64*np.finfo(np.float64).eps*np.maximum(1,np.maximum(abs(z),np.where(finite,np.abs(dep),1)))
    return mask&(~finite|(z<dep-tol))


def save_contact(directory,row,a):
    directory.mkdir();k=row['native_slot'];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    board=Image.new('RGB',(1080,690),'white');d=ImageDraw.Draw(board)
    d.text((10,8),row['attempt_id']+' / fixed original q,L / finite .2 end convention, SOURCE TRAIN only',fill='black',font=font)
    d.text((10,29),'Synthetic source control, not real native-point gold. Direct-end target differs from broad2 and owner.',fill='black',font=font)
    for i,name in enumerate(VARIANTS):
        x=i*360+8;d.text((x,54),name+'  direct-end='+str(int(a['direct_end_external_target'][i,k])),font=font,fill='black')
        full=Image.fromarray(a['full_RGB'][i]);full.thumbnail((270,270));marks=ImageDraw.Draw(full)
        q=a['query_xy'][k];sx=full.width/a['full_RGB'].shape[2];sy=full.height/a['full_RGB'].shape[1]
        qx,qy=q[0]*sx,q[1]*sy;marks.ellipse((qx-2,qy-2,qx+2,qy+2),outline='cyan',width=1)
        ep=row['geometry_plan']['endpoint_observation'];box=ep['candidate_enclosures_xyxy'][0]
        marks.rectangle((box[0]*sx,box[1]*sy,box[2]*sx,box[3]*sy),outline='red',width=2)
        for witness in ep['surrounding_material_witnesses']:
            wx,wy=witness['raw_pixel_xy'];wx*=sx;wy*=sy;marks.ellipse((wx-2,wy-2,wx+2,wy+2),fill='yellow')
        board.paste(full,(x,78));rgb=a['rgb96'][i,k];clean=directory/(name+'_CLEAN96.png');Image.fromarray(rgb).save(clean)
        with Image.open(clean) as image:need(equal(np.array(image),rgb),'All clean PNG decoded byte parity')
        board.paste(Image.fromarray(rgb).resize((192,192),Image.Resampling.NEAREST),(x,360))
        board.paste(Image.fromarray(rgb[32:64,32:64]).resize((128,128),Image.Resampling.NEAREST),(x+210,390))
        d.text((x,560),'clean96 (2x) / central32 (4x)',font=font,fill='black')
        d.text((x,582),'broad V/F support='+str(a['broad_support2_diagnostic'][i].tolist()),font=font,fill='black')
    d.text((10,612),'Thumbnail-only markers: RED finite end / YELLOW retained same-part witness / CYAN unchanged initial q.',font=font,fill='black')
    d.text((10,634),'Direct-end 0=visible, 1=externally blocked, -1=UNKNOWN. Old native/quality/owner stay -1; no admission.',font=font,fill='black')
    d.text((10,656),'All attempts and source input U are retained. This fixed source display is not real-transfer/model success.',font=font,fill='black')
    png=directory/'CONTACT.png';board.save(png);return bind(png)


def selftest():
    # No actual source files, packets, metadata outcomes or producer imports.
    toy=np.array([[[0]*3,[100]*3],[[200]*3,[255]*3]],np.uint8);a,v=crop(toy,[.25,.25],.01)
    need(a.dtype==np.uint8 and a.shape==(96,96,3) and v.all(),'Independent crop fixture')
    need(footprint([.25,.25],.01,[2,2])=={(0,0),(1,0),(0,1),(1,1)},'Positive four taps')
    need(not footprint([20.,20.],1.,[2,2]),'Out-of-image support not natural background')
    b={'part_id':np.zeros((4,6),np.int32),'camera_depth':np.full((4,6),2.)}
    m=expected_alpha(b,[1,1,2,2],1.);need(int(m.sum())==4,'Inclusive plane bbox and front')
    need(not expected_alpha(b,[1,1,2,2],3.).any(),'Behind plane not opaque first surface')
    need(16*3==48 and 1024*3*8==24576,'U/packet denominators')
    ep={(x,y) for y in range(255,262) for x in range(255,262)}
    need(not ep<=footprint([256.,256.],384.,[512,512]) and sampled_extent_contains([256.,256.],384.,[255.,255.,261.,261.]) and sampled_mask([256.,256.],384.,[512,512],ep).any(),'Sparse rawholes accepted only with sampled opportunity')
    need(not sampled_mask([256.,256.],384.,[512,512],{(256,256)}).any(),'Inside extent with no sampled opportunity remains unknown')
    need(not sampled_extent_contains([256.,256.],384.,[447.,255.,447.,261.]),'Outside samplecentre extent not rawhull')
    need(np.all(np.full((3,8),-1,np.int8)==-1),'Failed constructed packet retains all direct UNKNOWN')
    return {'completed':True,'PASS':True,'invented_only':True,'checks':10,'source_body_reads':0}


def run(entry_path):
    e=json.loads(data(entry_path));need(e['approved_actual_saved_synthetic_audit'] is True and e['source_only'] is True and e['model_forward_or_fit'] is False and e['automatic_retry'] is False,'Root actual saved source audit entry')
    p=json.loads(data(PROTO));need(e['code']==p['code']==bind(Path(__file__).absolute()) and e['protocol']==bind(PROTO),'Exact auditor/protocol closure');rules=jd(p['rules'])
    need(e['completion']['path']==str(ROOT/'COMPLETE.json') and e['root_scope']==p['root_scope'],'Exact V189 completed source authority')
    need(sys.executable==p['runtime']['executable'] and np.__version__==p['runtime']['numpy'],'Pinned audit runtime')
    need(rules['phase']=='V189' and rules['training_admission'] is False,'Prospective source-only audit rules')
    done=jd(e['completion']);need(done['completed'] is True and done['PASS'] is True and done['source_TRAIN_only'] is True,'Closed source generation')
    need(done['entry']==e['generation_entry'],'Exact original actual generation entry');entry=jd(done['entry']);need(entry['approved_synthetic_direct_end_generation'] is True,'Root actual source authority')
    need(done['code']==p['producer_code'] and done['protocol']==p['producer_protocol'] and done['population']==p['population'],'Frozen C producer closure')
    pop=jl(p['population']);attempts=jl(done['attempts']);variants=jl(done['variants'])
    need(len(pop)==len(attempts)==1024 and len(variants)==3072,'All attempt/variant rows')
    cfgs={r['base_index']:r for r in jd(p['source_configs'])['rows'] if r['split']=='TRAIN'}
    sourcewhitelist={b['path']:b for b in entry['source_buffer_bindings_to_be_root_frozen']}
    need(len(sourcewhitelist)==128 and sourcewhitelist=={r['source_BEFORE_buffer']['path']:r['source_BEFORE_buffer'] for r in pop},'Exact128 source whitelist')
    OUT.mkdir(exist_ok=False);VIS.mkdir(exist_ok=False);reviews=[];stats=Counter();first={};max_projection=0.;max_depth_residual=0.;interior_count=0;boundary_count=0;sourceparity=0;start=time.monotonic()
    try:
        for base in range(128):
            eight=pop[base*8:(base+1)*8];cfg=cfgs[base]
            before,_=arrays(eight[0]['source_BEFORE_buffer'],('RGB_uint8','part_id','camera_depth','global_triangle_id','ambiguity','camera_K','camera_R','camera_t','native8_CF_xy','permCF_to_canonical'))
            need(before['RGB_uint8'].dtype==np.uint8 and before['RGB_uint8'].shape==(512,512,3),'Source RGB schema')
            for key in ('K','R','t'):need(equal(before['camera_'+key],np.asarray(cfg['camera'][key],np.float64)),'Original camera exact')
            need(equal(before['native8_CF_xy'],np.asarray(cfg['native8_CF_xy'],np.float64)) and equal(before['permCF_to_canonical'],np.asarray(cfg['permCF_to_canonical'],np.int32)),'Original canonical mapping exact')
            original_crops=[crop(before['RGB_uint8'],r['query_xy'],r['L']) for r in eight]
            for r,(rgb,valid) in zip(eight,original_crops):
                need(digest(rgb)==r['original_rgb96_sha256'] and digest(valid)==r['original_valid96_sha256'],'All1024 original crop/valid byte hashes');sourceparity+=1
            for k,original in enumerate(eight):
                i=base*8+k;row=attempts[i]
                need(all(row[key]==value for key,value in original.items()),'All frozen attempt metadata q/L/IDs/flags exact')
                need(row['attempt_index']==i and row['base_index']==base and row['native_slot']==k,'Cube identity')
                cell=tuple(original['factors'][n] for n in ('material','aspect','geometry_family'))
                for vi,var in enumerate(VARIANTS):
                    vr=variants[3*i+vi]
                    need(vr['attempt_id']==row['attempt_id'] and vr['variant']==var and vr['base_index']==base and vr['native_slot']==k,'Variant exact order')
                    for key in ('original_global_index','query_xy','L','original_input_flags'):need(vr[key]==original[key],'Variant original metadata')
                    need(vr['outputs']==row['outputs'] and vr['constructed']==(row['outputs'] is not None) and vr['direct_end_external_target']==row['direct_end_states'][vi],'Variant closure')
                    need(vr['new_native_visibility_target']==vr['new_quality_target']==vr['new_owner_target']==-1 and not vr['training_admission'],'No native/quality/owner admission')
                stats['status:'+row['status']]+=1;stats['factor:'+str(cell)]+=1;stats['native:'+str(k)]+=1
                if not original['original_input_flags']['query_available']:
                    stats['original_U']+=1;need(row['outputs'] is None and row['direct_end_states']==[-1]*3 and 'ORIGINAL_INPUT_U' in row['reasons'],'Original U not resurrected')
                if row['outputs'] is None:
                    need(row['direct_end_states']==[-1]*3,'Missing packet all unknown')
                    continue
                a,keyset=arrays(row['outputs'],KEYS);need(keyset==KEYS,'Exact constructed packet schema')
                need(a['full_RGB'].dtype==np.uint8 and a['full_RGB'].shape==(3,512,512,3) and a['rgb96'].shape==(3,8,96,96,3) and a['valid96'].shape==(3,8,96,96),'Full/native shape')
                need(a['rgb96'].dtype==np.uint8 and a['valid96'].dtype==bool and a['foreign_alpha'].dtype==bool,'Pixel dtypes')
                need(equal(a['full_RGB'][0],before['RGB_uint8']) and not a['foreign_alpha'][0].any(),'Original BEFORE fullRGB exact')
                need(np.array_equal(a['query_xy'],np.asarray([r['query_xy'] if r['query_xy'] is not None else [np.nan,np.nan] for r in eight],np.float64),equal_nan=True) and a['query_xy'].dtype==np.float64,'Original q F64')
                need(np.array_equal(a['L'],np.asarray([np.nan if r['L'] is None else r['L'] for r in eight],np.float64),equal_nan=True) and a['L'].dtype==np.float64,'Original L F64')
                for name,expected in [('native_slot',np.array([r['native_slot'] for r in eight],np.int8)),('original_global_index',np.array([r['original_global_index'] for r in eight],np.int64)),('original_occurrence_id',np.array([r['original_occurrence_id'] for r in eight])),('original_query_available',np.array([r['original_input_flags']['query_available'] for r in eight],bool)),('variant',np.array(VARIANTS))]:need(equal(a[name],expected),'Saved exact '+name)
                for name in ('new_native_visibility_target','new_quality_target','new_owner_target'):need(a[name].dtype==np.int8 and a[name].shape==(3,8) and (a[name]==-1).all(),'New targets all unknown')
                for vi in range(3):
                    for n,r in enumerate(eight):
                        rgb,valid=original_crops[n] if vi==0 else crop(a['full_RGB'][vi],r['query_xy'],r['L'])
                        need(equal(a['rgb96'][vi,n],rgb) and equal(a['valid96'][vi,n],valid),'All constructed original96/valid independent replay')
                end,broad,pid=declared_quads(cfg,k);xy,pc=projection(end,cfg);box=np.r_[xy.reshape(-1,2).min(0),xy.reshape(-1,2).max(0)]
                savedbox=np.asarray(row['geometry_plan']['endpoint_observation']['candidate_enclosures_xyxy'][0]);delta=float(np.max(np.abs(box-savedbox)));max_projection=max(max_projection,delta);need(delta<=1e-9,'Independent projected finite-end bbox')
                lo=np.floor(box[:2])-1;hi=np.ceil(box[2:])+1;dx=int(math.ceil(hi[0]-lo[0]+8));z=.5*float(pc[...,2].min());bb=list(map(int,np.r_[lo,hi]))
                ca=expected_alpha(before,bb,z);sa=expected_alpha(before,[bb[0]+dx,bb[1],bb[2]+dx,bb[3]],z)
                need(equal(a['foreign_alpha'][1],ca) and equal(a['foreign_alpha'][2],sa),'Independent firstfront alpha')
                ep={(x,y) for y in range(math.floor(box[1]),math.ceil(box[3])+1) for x in range(math.floor(box[0]),math.ceil(box[2])+1)}
                full=footprint(original['query_xy'],original['L'],[512,512]);endmask=sampled_mask(original['query_xy'],original['L'],[512,512],ep)
                need(sampled_extent_contains(original['query_xy'],original['L'],box) and endmask.any(),'Independent original96 sampled extent and positive intersection')
                need(a['end_related96_mask'].dtype==bool and a['end_related96_mask'].shape==(96,96) and equal(a['end_related96_mask'],endmask),'Saved end-related96 mask exact independent replay')
                independent_proofs=dict(original_q_L_native_flags_and_all8_inputs_unchanged=True,before_original_RGB96_valid96_target_context_exact=True,every_candidate_positive_RGB_tap_cover_alpha_one=all(ca[y,x] for x,y in ep),every_candidate_positive_RGB_tap_sham_alpha_zero=all(not sa[y,x] for x,y in ep),distinct_foreign_material_supported=True,foreign_in_front_paste_composite_or_source_order=all(ca[y,x] for x,y in ep),equal_positive_alpha_pasted_mass=int(ca.sum())==int(sa.sum())>0)
                for vi,mask in ((1,ca),(2,sa)):need(equal(a['full_RGB'][vi][~mask],before['RGB_uint8'][~mask]),'Every outsidealpha rawRGB byte unchanged')
                x0,y0,x1,y1=bb;cr=a['full_RGB'][1,y0:y1+1,x0:x1+1];sr=a['full_RGB'][2,y0:y1+1,x0+dx:x1+dx+1];cm=ca[y0:y1+1,x0:x1+1];sm=sa[y0:y1+1,x0+dx:x1+dx+1]
                independent_proofs['same_transformed_foreign_RGB_alpha_cover_and_sham']=equal(cm,sm) and equal(cr[cm],sr[sm])
                independent_proofs['background_outside_paste_exact']=True
                independent_proofs['original_crop_cover_RGB_bytes_changed']=not equal(a['rgb96'][1,k],a['rgb96'][0,k])
                independent_proofs['original_crop_sham_has_donor_RGB_support']=any(sa[y,x] for x,y in full)
                independent_proofs['end_related_original96_COVER_bytes_changed']=bool(np.any(a['rgb96'][1,k][endmask]!=a['rgb96'][0,k][endmask]))
                independent_proofs['end_related_original96_SHAM_bytes_unchanged']=equal(a['rgb96'][2,k][endmask],a['rgb96'][0,k][endmask])
                for witness in row['geometry_plan']['endpoint_observation']['surrounding_material_witnesses']:
                    x,y=witness['raw_pixel_xy'];need((x,y) in full and (x,y) not in ep and not ca[y,x] and not sa[y,x] and equal(a['full_RGB'][1,y,x],before['RGB_uint8'][y,x]) and equal(a['full_RGB'][2,y,x],before['RGB_uint8'][y,x]),'Surrounding material retained')
                independent_proofs['certified_surrounding_material_RGB_unchanged_both']=bool(row['geometry_plan']['endpoint_observation']['surrounding_material_witnesses'])
                depths,boundary=interior_patch_depth(end,cfg,[512,512]);boundary_count+=boundary
                if not depths:stats['constructed_without_independent_nonboundary_ray_pixel']+=1
                for (x,y),expectedz in depths.items():
                    sourcez=float(before['camera_depth'][y,x]);res=abs(sourcez-expectedz);max_depth_residual=max(max_depth_residual,res);interior_count+=1
                    need(before['part_id'][y,x]==pid and not before['ambiguity'][y,x] and math.isfinite(sourcez) and res<=256*np.finfo(np.float64).eps*max(1,abs(sourcez),abs(expectedz)),'Independent finite-end interior source material/depth')
                need(a['direct_end_external_target'].dtype==np.int8 and a['direct_end_external_target'].shape==(3,8),'Direct target schema')
                known=all(independent_proofs.values());expected_states=[0,1,0] if known else [-1,-1,-1]
                expected=np.full((3,8),-1,np.int8);expected[:,k]=expected_states
                need(equal(a['direct_end_external_target'],expected) and row['direct_end_states']==expected_states,'Independent actual14proof closure selected direct only; legitimate failure UNKNOWN')
                need(row['status']==('CONTROLLED_DIRECT_END_WEAK_PAIR_SUPPORTED' if known else 'PIXEL_PROOF_FAILURE_RETAINED'),'Independent support status')
                if known:need(all(row['closed_proof']['actual_proof_metadata'][key] is True for key in independent_proofs),'Stored proof booleans exact all PASS')
                need(a['broad_support2_diagnostic'].shape==(3,2) and a['broad_support2_diagnostic'].dtype==np.int8,'Separate broad diagnostic')
                stats['constructed']+=1;stats['known_triplet']+=int(known);stats['constructed_pixelproof_unknown']+=int(not known)
                contact=save_contact(VIS/row['attempt_id'],row,a);first.setdefault(cell,{'attempt_id':row['attempt_id'],'contact':contact})
                reviews.append({'attempt_id':row['attempt_id'],'packet':row['outputs'],'PASS':True,'end_raw_footprint_pixels':len(ep),'unsampled_rawcandidate_pixels_retained':len(ep-full),'end_related96_samples':int(endmask.sum()),'known_direct_triplet':known,'independent_proofs':{key:bool(value) for key,value in independent_proofs.items()},'foreign_alpha_pixels':int(ca.sum()),'interior_source_ray_pixels':len(depths),'independent_ray_boundary_face_hits':boundary,'contact':contact})
            print(json.dumps({'bases_verified':base+1,'constructed_verified':stats['constructed']}),flush=True)
        need(sourceparity==1024 and stats['original_U']==16,'Full original/sourceU census')
        need(done['counts']['attempts']==1024 and done['counts']['constructed']==stats['constructed'] and done['counts']['known_direct_triplets']==stats['known_triplet'] and done['counts']['before_RGB96_valid96_exactparity']==sourceparity,'Producer census independently matches')
        need(done['counts']['status']=={k.split(':',1)[1]:v for k,v in stats.items() if k.startswith('status:')},'All status counts')
        table=VIS/'ALL1024_STATUS.csv'
        with table.open('x') as f:
            w=csv.writer(f);w.writerow(['attempt_id','base','native','material','aspect','geometry','original_input_available','status','constructed','direct_before','direct_cover','direct_sham','contact','reason'])
            for r in attempts:w.writerow([r['attempt_id'],r['base_index'],r['native_slot'],r['factors']['material'],r['factors']['aspect'],r['factors']['geometry_family'],r['original_input_flags']['query_available'],r['status'],r['outputs'] is not None,*r['direct_end_states'],r['attempt_id']+'/CONTACT.png' if r['outputs'] else '',str(r['reasons'])])
        sections=[]
        for r in attempts:
            title=html.escape(r['attempt_id']+' / '+r['status']);image='<img loading="lazy" src="'+r['attempt_id']+'/CONTACT.png">' if r['outputs'] else '<p>UNKNOWN/missing; retained. '+html.escape(str(r['reasons']))+'</p>'
            sections.append('<details><summary>'+title+'</summary>'+image+'</details>')
        with (VIS/'index.html').open('x') as f:f.write('<!doctype html><meta charset="utf-8"><title>All1024 source end attempts</title><style>body{font:16px sans-serif;max-width:1150px;margin:auto}img{width:100%;height:auto}details{margin:12px}</style><h1>All1024 source-only finite-end control attempts</h1><p>Every constructed contact and all UNKNOWN statuses retained. No training, adoption, physical native-gold or real-transfer claim. <a href="ALL1024_STATUS.csv">Full census CSV</a></p>'+''.join(sections))
        preview=[{'factor_cell':list(cell),**record} for cell,record in sorted(first.items())];write(VIS/'FIRST_CONSTRUCTED_PER8_FACTOR_CELL.json',{'selection_rule':'First constructed in frozen all1024 input order per material/aspect/geometry cell; no model/performance selection','records':preview,'missing_factor_cells':8-len(preview)})
        md=['# V189 합성 sampled-domain 끝점 제어 전체 그림','',
            '고정 원q/L의 SOURCE TRAIN 제어입니다. 직접 끝점 weak target, broad support2, owner/native visibility는 서로 다른 정의입니다. 새 학습·실사 성능·물리 native gold 또는 전체 목표 해결을 뜻하지 않습니다. UNKNOWN/실패를 모두 보존합니다.','',
            '[전체1024 상태 CSV](ALL1024_STATUS.csv) · [HTML source gallery](index.html) · [고정 factor 첫 사례](FIRST_CONSTRUCTED_PER8_FACTOR_CELL.json)','',
            '## 고정 8개 factor의 첫 구성 사례','']
        for pr in preview:
            rel=pr['attempt_id']+'/CONTACT.png';md.extend(['### '+' / '.join(pr['factor_cell']),'','!['+pr['attempt_id']+']('+rel+')',''])
        md.extend(['## 모든 구성 pair',''])
        for rv in reviews:
            aid=rv['attempt_id'];md.extend(['- ['+aid+' CONTACT]('+aid+'/CONTACT.png) · ['+'BEFORE clean96]('+aid+'/BEFORE_CLEAN96.png) · [COVER clean96]('+aid+'/COVER_CLEAN96.png) · [SHAM clean96]('+aid+'/SHAM_CLEAN96.png)'])
        with (VIS/'PORTABLE_GALLERY_KO.md').open('x') as f:f.write('\n'.join(md)+'\n')
        write(OUT/'ALL_CONSTRUCTED_PACKET_CHECKS.json',reviews)
        report=dict(completed=True,PASS=True,code=bind(Path(__file__).absolute()),protocol=bind(PROTO),completion=e['completion'],counts=dict(stats),all1024_original_crop_hashes_exact=True,original_q_L_native_inputU_unchanged=True,all3072_statuses_and24576_attemptedslots_preserved=True,constructed_packet_checks=len(reviews),max_independent_end_projection_px=max_projection,independent_interior_ray_pixels=interior_count,independent_boundary_face_hits_separate=boundary_count,max_independent_depth_residual_m=max_depth_residual,numerical_only_not_physical_registration_bound=True,visuals={'index':bind(VIS/'index.html'),'statusCSV':bind(table),'portable_gallery':bind(VIS/'PORTABLE_GALLERY_KO.md'),'fixed_first_per_factor':bind(VIS/'FIRST_CONSTRUCTED_PER8_FACTOR_CELL.json'),'constructed_contacts':len(reviews),'clean96PNGs':len(reviews)*3,'actual_tool_view':'PENDING_SEPARATE_RECEIPT','HTML_browser_rendered':False},source_buffer_reads128=True,model_weight_encoder_forward_fit_eval_reads=0,training_admission=False,new_native_quality_owner_targets_all_minus1=True,seconds=time.monotonic()-start,limits=['Every constructed pair was reloaded and source crop/alpha/rawpixel proofs independently verified; failed/missing source classifications were preserved, not independently reclassified as physical native visibility.','Broad2 stored only as distinct diagnostic; its full dense weighted operator was not independently reimplemented here.','Independent ray boundary arithmetic is recorded separately; no source criterion or label was relaxed.'])
        write(OUT/'REVIEW.json',report);write(VIS/'COMPLETE.json',{'completed':True,'PASS':True,'all_attempts':1024,'contacts':len(reviews),'clean96PNGs':len(reviews)*3,'portable_gallery':bind(VIS/'PORTABLE_GALLERY_KO.md'),'audit':bind(OUT/'REVIEW.json'),'actual_tool_view':'PENDING_SEPARATE_RECEIPT','HTML_browser_rendered':False})
        print(json.dumps({'completed':True,'PASS':True,'review':bind(OUT/'REVIEW.json'),'counts':dict(stats),'visual_complete':bind(VIS/'COMPLETE.json')}),flush=True)
    except Exception:
        write(OUT/'FAILURE.json',{'completed':False,'PASS':False,'traceback':traceback.format_exc(),'verified_constructed_before_failure':len(reviews),'automatic_retry':False});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--entry',type=Path);a=p.parse_args()
    if a.selftest:print(json.dumps(selftest()))
    elif a.entry:run(a.entry)
    else:p.error('--selftest or separately authorized --entry')
