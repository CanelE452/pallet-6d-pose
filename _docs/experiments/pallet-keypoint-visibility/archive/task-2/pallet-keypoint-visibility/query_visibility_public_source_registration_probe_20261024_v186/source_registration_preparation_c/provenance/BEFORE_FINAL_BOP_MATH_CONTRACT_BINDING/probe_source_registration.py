"""V186 fixed four SOURCE TRAIN frames/all five obj1 instances, CPU only.
--prepare reads scope/code and invented arrays only; no source asset/header/hash.
--entry needs separate root authorization. No pose fitting, labels or model calls.
"""
from pathlib import Path
from io import BytesIO
import argparse,hashlib,importlib.util,json,math,shutil,struct,sys,tempfile,traceback
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parent;V=H.parent
SCOPE=V/'ROOT_SCOPE_BEFORE_SOURCE_RGB_DEPTH_MASK_POSE_MESH_BODIES.json'
PROTOCOL=H/'PROTOCOL_BEFORE_SOURCE_REGISTRATION_ASSET_BODIES.json';CHECKS=H/'GENERATED_RASTERIZER_CHECKS.json'
OUT=V/'fixed_source_registration_probe_v1';NEAR_MM=1e-6;EDGE_ULPS=64
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
SCALARS={'char':'b','int8':'b','uchar':'B','uint8':'B','short':'h','int16':'h','ushort':'H','uint16':'H','int':'i','int32':'i','uint':'I','uint32':'I','float':'f','float32':'f','double':'d','float64':'d'}

def require(ok,msg):
    if not bool(ok):raise AssertionError(msg)
def safe(path):
    path=Path(path);s=str(path).lower();require('_v112' not in s and '/private_evaluation/' not in s and '/val/000001/' not in s,'Forbidden path before access');return path
def sha(path):
    h=hashlib.sha256()
    with safe(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(path):return dict(path=str(Path(path).absolute()),sha256=sha(path))
def checked(binding):require(bind(binding['path'])=={k:binding[k] for k in ['path','sha256']},'Exact source binding '+binding['path']);return safe(binding['path'])
def read(path):return json.loads(safe(path).read_text())
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    return x
def write(path,x):
    with Path(path).open('x') as f:json.dump(clean(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def ash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def parse_ply_bytes(payload):
    """Exact PLY vertices; no process/repair/center/unit conversion/weld.
    Nontriangular faces are fan primitives, with original face ID retained.
    """
    stream=BytesIO(payload);require(stream.readline().strip()==b'ply','PLY signature');elements=[];fmt=None;header_bytes=4
    while True:
        line=stream.readline();header_bytes+=len(line);require(line and header_bytes<2**21,'Finite PLY header');fields=line.decode('ascii').strip().split()
        if not fields:continue
        if fields[0]=='format':require(len(fields)==3 and fields[2]=='1.0','PLY format version');fmt=fields[1]
        elif fields[0]=='element':elements.append(dict(name=fields[1],count=int(fields[2]),properties=[]));require(0<=elements[-1]['count']<=20000000,'PLY resource count')
        elif fields[0]=='property':
            require(elements,'PLY property element')
            if fields[1]=='list':require(fields[2] in SCALARS and fields[3] in SCALARS,'PLY list types');elements[-1]['properties'].append(('list',fields[2],fields[3],fields[4]))
            else:require(fields[1] in SCALARS,'PLY scalar type');elements[-1]['properties'].append(('scalar',fields[1],fields[2]))
        elif fields[0]=='end_header':break
        else:require(fields[0] in ['comment','obj_info'],'Unrecognized PLY header item')
    require(fmt in ['ascii','binary_little_endian','binary_big_endian'],'Supported exact PLY format');endian='<' if fmt=='binary_little_endian' else '>'
    vertices=[];faces=[]
    def number(value,kind):return float(value) if SCALARS[kind] in ['f','d'] else int(value)
    def binary(kind):
        code=endian+SCALARS[kind];data=stream.read(struct.calcsize(code));require(len(data)==struct.calcsize(code),'PLY finite record');return struct.unpack(code,data)[0]
    for element in elements:
        for _ in range(element['count']):
            record={};tokens=stream.readline().decode('ascii').split() if fmt=='ascii' else None;pos=0
            for prop in element['properties']:
                if prop[0]=='scalar':
                    if fmt=='ascii':require(pos<len(tokens),'PLY ASCII scalar');value=number(tokens[pos],prop[1]);pos+=1
                    else:value=binary(prop[1])
                    record[prop[2]]=value
                else:
                    if fmt=='ascii':require(pos<len(tokens),'PLY ASCII list count');count=number(tokens[pos],prop[1]);pos+=1
                    else:count=binary(prop[1])
                    require(isinstance(count,int) and 0<=count<=1000000,'PLY list resource count')
                    if fmt=='ascii':require(pos+count<=len(tokens),'PLY ASCII list');value=[number(t,prop[2]) for t in tokens[pos:pos+count]];pos+=count
                    else:value=[binary(prop[2]) for _ in range(count)]
                    record[prop[3]]=value
            if fmt=='ascii':require(pos==len(tokens),'PLY no ignored trailing data')
            if element['name']=='vertex':require(all(k in record for k in ['x','y','z']),'PLY xyz fields');vertices.append([record['x'],record['y'],record['z']])
            elif element['name']=='face':
                keys=[k for k in ['vertex_indices','vertex_index'] if k in record];require(len(keys)==1,'PLY exact face index property');faces.append(record[keys[0]])
    vertices=np.asarray(vertices,np.float64);require(vertices.ndim==2 and vertices.shape[1]==3 and len(vertices)>0 and np.isfinite(vertices).all(),'Finite unmodified mesh vertices')
    triangles=[];face_ids=[]
    for face_id,face in enumerate(faces):
        require(len(face)>=3 and all(isinstance(k,(int,np.integer)) and 0<=k<len(vertices) for k in face),'PLY valid original face indices')
        for j in range(1,len(face)-1):triangles.append([face[0],face[j],face[j+1]]);face_ids.append(face_id)
    triangles=np.asarray(triangles,np.int64);require(triangles.ndim==2 and triangles.shape[1]==3 and len(triangles)>0,'Nonempty triangle primitives')
    return vertices,triangles,np.asarray(face_ids,np.int64),dict(format=fmt,original_vertices=len(vertices),original_faces=len(faces),raster_triangles=len(triangles),nontriangular_faces=sum(len(f)!=3 for f in faces),vertex_array_sha256=ash(vertices),triangle_index_array_sha256=ash(triangles),face_index_array_sha256=ash(np.asarray(face_ids,np.int64)),repairs_or_mesh_unit_transform=0)

def clip_near(triangle,near=NEAR_MM):
    out=[]
    for i in range(3):
        a=triangle[i-1];b=triangle[i];ina=a[2]>=near;inb=b[2]>=near
        if ina!=inb:
            t=(near-a[2])/(b[2]-a[2]);point=a+t*(b-a);point[2]=near;out.append(point)
        if inb:out.append(b.copy())
    return out

def rasterize(vertices,triangles,face_ids,K,R,t,hw):
    """Integer pixel-center rays, both faces, reciprocal-Z barycentric first hit."""
    h,w=map(int,hw);K=np.asarray(K,np.float64);R=np.asarray(R,np.float64);t=np.asarray(t,np.float64).reshape(3)
    require(h>0 and w>0 and K.shape==R.shape==(3,3) and np.isfinite(K).all() and np.isfinite(R).all() and np.isfinite(t).all(),'Camera schema')
    require(np.array_equal(K[2],np.array([0.,0.,1.])) and K[0,0]>0 and K[1,1]>0 and np.linalg.det(K)!=0,'BOP pinhole K')
    require(vertices.dtype==np.float64 and np.isfinite(vertices).all() and triangles.dtype==np.int64 and face_ids.shape==(len(triangles),),'Exact geometry schema')
    cam=vertices@R.T+t;depth=np.full((h,w),np.inf,np.float64);triid=np.full((h,w),-1,np.int32);faceid=np.full((h,w),-1,np.int32);stats=dict(source_triangles=len(triangles),behind_or_empty=0,near_clipped_triangles=0,degenerate_projected_primitives=0,raster_primitives=0)
    eps=EDGE_ULPS*np.finfo(np.float64).eps
    for tid,index in enumerate(triangles):
        original=cam[index];polygon=clip_near(original)
        if len(polygon)<3:stats['behind_or_empty']+=1;continue
        if (original[:,2]<NEAR_MM).any():stats['near_clipped_triangles']+=1
        for fan in range(1,len(polygon)-1):
            vv=np.asarray([polygon[0],polygon[fan],polygon[fan+1]],np.float64);p=vv@K.T;uv=p[:,:2]/p[:,2,None]
            require(np.isfinite(uv).all(),'Finite projected triangle')
            den=(uv[1,1]-uv[2,1])*(uv[0,0]-uv[2,0])+(uv[2,0]-uv[1,0])*(uv[0,1]-uv[2,1])
            if den==0:stats['degenerate_projected_primitives']+=1;continue
            xmin=max(0,math.ceil(float(uv[:,0].min())-eps));xmax=min(w-1,math.floor(float(uv[:,0].max())+eps));ymin=max(0,math.ceil(float(uv[:,1].min())-eps));ymax=min(h-1,math.floor(float(uv[:,1].max())+eps))
            if xmin>xmax or ymin>ymax:continue
            xx,yy=np.meshgrid(np.arange(xmin,xmax+1,dtype=np.float64),np.arange(ymin,ymax+1,dtype=np.float64))
            a=((uv[1,1]-uv[2,1])*(xx-uv[2,0])+(uv[2,0]-uv[1,0])*(yy-uv[2,1]))/den
            b=((uv[2,1]-uv[0,1])*(xx-uv[2,0])+(uv[0,0]-uv[2,0])*(yy-uv[2,1]))/den;c=1-a-b
            inside=(a>=-eps)&(b>=-eps)&(c>=-eps);inverse=a/vv[0,2]+b/vv[1,2]+c/vv[2,2]
            z=np.divide(1.,inverse,out=np.full_like(inverse,np.inf),where=inverse>0);region=depth[ymin:ymax+1,xmin:xmax+1];update=inside&(z>=NEAR_MM)&(z<region)
            region[update]=z[update];triid[ymin:ymax+1,xmin:xmax+1][update]=tid;faceid[ymin:ymax+1,xmin:xmax+1][update]=int(face_ids[tid]);stats['raster_primitives']+=1
    silhouette=np.isfinite(depth);depth[~silhouette]=np.nan;require(np.array_equal(triid>=0,silhouette) and np.array_equal(faceid>=0,silhouette),'Firstsurface depth/identity coavailability')
    return dict(model_Z_mm=depth,triangle_id=triid,face_id=faceid,silhouette=silhouette),stats

def quantiles(values):
    v=np.asarray(values,np.float64);v=v[np.isfinite(v)]
    return dict(N=int(len(v)),quantile_percent=[0,1,5,25,50,75,95,99,100],quantile_mm=None if not len(v) else np.percentile(v,[0,1,5,25,50,75,95,99,100]).tolist(),mean_mm=None if not len(v) else float(v.mean()),std_mm=None if not len(v) else float(v.std()),negative=int((v<0).sum()),exact_zero=int((v==0).sum()),positive=int((v>0).sum()))
def compare(raster,sensor,mask):
    rendered=raster['silhouette'];valid=np.isfinite(sensor)&(sensor>0);intersection=rendered&mask;union=rendered|mask;measured=intersection&valid;delta=raster['model_Z_mm']-sensor
    return dict(mask_pixels=int(mask.sum()),rendered_pixels=int(rendered.sum()),intersection=int(intersection.sum()),union=int(union.sum()),render_only=int((rendered&~mask).sum()),mask_only=int((mask&~rendered).sum()),IoU=None if not union.any() else float(intersection.sum()/union.sum()),mask_sensor_known=int((mask&valid).sum()),mask_sensor_missing=int((mask&~valid).sum()),mask_rendered_sensor_known=int(measured.sum()),mask_rendered_sensor_missing=int((mask&rendered&~valid).sum()),mask_unrendered_sensor_known=int((mask&~rendered&valid).sum()),mask_unrendered_sensor_missing=int((mask&~rendered&~valid).sum()),signed_modelZ_minus_sensorZ=quantiles(delta[measured]))
def boundary(mask):
    erode=mask.copy();erode[1:]&=mask[:-1];erode[:-1]&=mask[1:];erode[:,1:]&=mask[:,:-1];erode[:,:-1]&=mask[:,1:];erode[[0,-1],:]=False;erode[:,[0,-1]]=False;return mask&~erode
def palette(values,valid,signed=False):
    z=np.where(valid,values,0.)
    if signed:
        a=np.clip(z/100.,-1,1);out=np.stack((np.where(a>=0,255,255*(1+a)),255*(1-np.abs(a)),np.where(a<=0,255,255*(1-a))),axis=-1)
    else:
        a=np.clip(z/6000.,0,1);out=np.stack((255*np.clip(2*a,0,1),255*np.clip(2*(1-np.abs(a-.5)),0,1),255*np.clip(2*(1-a),0,1)),axis=-1)
    out=np.rint(out).astype(np.uint8);out[~valid]=0;return out
def diagnostics(rgb,raster,sensor,full,visible,label,directory):
    rendered=raster['silhouette'];overlay=rgb.copy();overlay[rendered]=np.rint(.72*rgb[rendered].astype(np.float64)+.28*np.array([220,40,200])).astype(np.uint8)
    overlay[boundary(full)]=[0,220,255];overlay[boundary(visible)]=[40,255,70];overlay[boundary(rendered)]=[255,100,240]
    overlay_path=directory/'RGB_NOMINAL_MESH_OVERLAY.png';Image.fromarray(overlay).save(overlay_path)
    valid=np.isfinite(sensor)&(sensor>0);measured=rendered&valid;delta=raster['model_Z_mm']-sensor;overlap=np.zeros_like(rgb);overlap[full]=[20,130,220];overlap[visible]=[20,220,90];overlap[rendered&full]=[240,220,30];overlap[rendered&~full]=[230,70,210]
    panels=[('Model first Z mm: fixed display 0..6000',palette(raster['model_Z_mm'],rendered)),('Sensor Z mm: raw * depth_scale; zero missing',palette(sensor,valid)),('Model Z - sensor Z: fixed display -100..100mm',palette(delta,measured,True)),('Masks: full blue / visible green / model+full yellow / model-only magenta',overlap)]
    h,w=rgb.shape[:2];board=Image.new('RGB',(2*w+60,2*h+200),'#f4f7fa');draw=ImageDraw.Draw(board);ft=ImageFont.truetype(FONT,24);draw.text((20,12),label+' | unmodified K/R/t/mesh; target-alone; no labels',font=ft,fill='#20364c')
    for j,(title,image) in enumerate(panels):
        x=20+(j%2)*(w+20);y=65+(j//2)*(h+65);draw.text((x,y-32),title,font=ImageFont.truetype(FONT,18),fill='#20364c');board.paste(Image.fromarray(image),(x,y))
    draw.text((20,2*h+160),'Black = no rendered/valid depth. Colors are display only, not depth thresholds or external/self truth. Overlay cyan=full, green=visible, pink=nominal model.',font=ImageFont.truetype(FONT,18),fill='#20364c');diag=directory/'DEPTH_DIAGNOSTIC.png';board.save(diag)
    return dict(overlay=bind(overlay_path),depth_diagnostic=bind(diag),display_ranges=dict(depth_mm=[0,6000],signed_mm=[-100,100]),display_clipped_sensor_over6000=int((valid&(sensor>6000)).sum()),display_clipped_model_over6000=int((rendered&(raster['model_Z_mm']>6000)).sum()),display_clipped_signed_abs_over100=int((measured&(np.abs(delta)>100)).sum()),actual_tool_view='pending',HTML_browser_rendered=False)

def oracle_depth(vertices,K,hw):
    """Invented-fixture independent ray/triangle intersection, no screen barycentric."""
    out=np.full(hw,np.nan,np.float64);a,b,c=vertices;u=b-a;v=c-a;normal=np.cross(u,v);inv=np.linalg.inv(K)
    for y in range(hw[0]):
        for x in range(hw[1]):
            ray=inv@np.array([x,y,1.]);den=normal@ray
            if den==0:continue
            depth=(normal@a)/den
            if depth<NEAR_MM:continue
            p=ray*depth;uv=np.linalg.lstsq(np.column_stack((u,v)),p-a,rcond=None)[0]
            if uv[0]>=-1e-12 and uv[1]>=-1e-12 and uv.sum()<=1+1e-12:out[y,x]=p[2]
    return out
def invented():
    checks=[]
    def ok(name,value):require(value,name);checks.append(dict(name=name,PASS=True))
    verts=np.array([[0,0,2],[8,0,2],[8,8,2],[0,8,2]],np.float64);tri=np.array([[0,1,2],[0,2,3]],np.int64);face=np.array([0,1],np.int64);K=np.eye(3);R=np.eye(3);t=np.zeros(3)
    a,_=rasterize(verts,tri,face,K,R,t,[6,6]);ok('Frontplane integer-pixel full25 silhouette and exact Z2',a['silhouette'].sum()==25 and (a['model_Z_mm'][a['silhouette']]==2).all())
    backward,_=rasterize(verts,tri[:,::-1].copy(),face,K,R,t,[6,6]);ok('Bothfaces no backface culling',np.array_equal(a['silhouette'],backward['silhouette']) and np.allclose(a['model_Z_mm'],backward['model_Z_mm'],rtol=0,atol=1e-14,equal_nan=True))
    vv=np.vstack((verts[:3]*2,verts[:3]));tt=np.array([[0,1,2],[3,4,5]],np.int64);z,_=rasterize(vv,tt,face,K,R,t,[6,6]);ok('NearestZ wins independent of sourcefarfirst order',z['model_Z_mm'][1,2]==2 and z['triangle_id'][1,2]==1)
    z,_=rasterize(np.vstack((verts[:3],verts[:3])),tt,face,K,R,t,[6,6]);ok('Exactdepth tie keeps first originaltriangle ID',z['triangle_id'][1,2]==0)
    tilted=np.array([[0,0,2],[16,0,4],[0,32,8]],np.float64);tr=np.array([[0,1,2]],np.int64);fa=np.array([7],np.int64);z,_=rasterize(tilted,tr,fa,K,R,t,[6,6]);oracle=oracle_depth(tilted,K,[6,6]);ok('Perspective correct reciprocalZ equals independentray oracle not linearZ',abs(z['model_Z_mm'][1,1]-32/11)<1e-14 and np.array_equal(z['silhouette'],np.isfinite(oracle)) and np.allclose(z['model_Z_mm'],oracle,rtol=0,atol=2e-13,equal_nan=True))
    mixed=np.array([[-1.,-1.,-1.],[4.,-2.,3.],[-2.,4.,3.]],np.float64);kk=np.array([[2.,0.,3.],[0.,2.,3.],[0.,0.,1.]]);z,st=rasterize(mixed,tr,fa,kk,R,t,[8,8]);oracle=oracle_depth(mixed,kk,[8,8]);ok('Nearplane clipping retains firstsurface match to independentray',st['near_clipped_triangles']==1 and np.array_equal(z['silhouette'],np.isfinite(oracle)) and np.allclose(z['model_Z_mm'],oracle,rtol=0,atol=2e-12,equal_nan=True))
    z,_=rasterize(-verts,tri,face,K,R,t,[6,6]);ok('Behindcamera never becomes foreground or zero validdepth',not z['silhouette'].any() and np.isnan(z['model_Z_mm']).all() and (z['triangle_id']==-1).all())
    sensor=np.full((6,6),2.,np.float64);sensor[1,1]=0.;mask=np.ones((6,6),bool);c=compare(a,sensor,mask);ok('Missing sensordepth preserved in raw partitions and signedstats',c['mask_pixels']==36 and c['mask_sensor_missing']==1 and c['mask_rendered_sensor_known']==24 and c['signed_modelZ_minus_sensorZ']['N']==24 and c['signed_modelZ_minus_sensorZ']['exact_zero']==24)
    raw=np.array([[0,100]],np.uint16);sensor=raw.astype(np.float64)*.1;ok('BOP depthscale in mm unchanged; zero remainsmissing',sensor.tolist()==[[0.,10.]] and quantiles([])['quantile_mm'] is None)
    header='ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n';body='0 0 2\n8 0 2\n8 8 2\n0 8 2\n4 0 1 2 3\n';v,tt,ff,meta=parse_ply_bytes((header+body).encode());ok('PLY ASCII xyz unchanged and polygonfan keeps originalface ID',np.array_equal(v,verts) and np.array_equal(tt,tri) and ff.tolist()==[0,0] and meta['repairs_or_mesh_unit_transform']==0)
    for order,name in [('<','binary_little_endian'),('>','binary_big_endian')]:
        h=header.replace('ascii',name).encode();payload=h+b''.join(struct.pack(order+'fff',*point) for point in verts)+struct.pack(order+'Biiii',4,0,1,2,3);v,tt,ff,_=parse_ply_bytes(payload);ok('PLY '+name+' exact original geometry',np.array_equal(v,verts) and np.array_equal(tt,tri) and ff.tolist()==[0,0])
    try:parse_ply_bytes((header+body.replace('4 0 1 2 3','4 0 1 2 99')).encode())
    except AssertionError:rejected=True
    else:rejected=False
    ok('Invalid sourceface indices failclosed without meshrepair',rejected);return checks

def validate_scope(scope):
    frames=scope['fixed_frames_before_body_reads'];require([f['frame_index'] for f in frames]==[0,118,166,199] and [f['scene'] for f in frames]==['000000','000002','000005','000006'],'Fixed SOURCE TRAIN fourframes')
    require(scope['allowed_body_bindings']['frame_assets']==frames and sum(len(f['all_obj1_GT_instance_indices']) for f in frames)==5,'All original five obj1instances, no selection')
    for f in frames:require(f['internal_split']=='SOURCE_DEVELOPMENT' and f['obj_id']==1 and f['frame_key']=='0' and set(f['all_obj1_GT_instance_indices'])=={r['gt_instance_index'] for r in f['full_masks']}=={r['gt_instance_index'] for r in f['visible_masks']},'Originalsplit and perinstance masks')
    return frames
def prepare():
    scope=read(SCOPE);frames=validate_scope(scope);tests=invented();write(CHECKS,dict(completed=True,PASS=True,code=bind(__file__),checks=tests,actual_source_RGB_depth_mask_pose_camera_mesh_header_or_body_hash_reads=0))
    protocol=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),checks=bind(CHECKS),fixed_frame_indices=[f['frame_index'] for f in frames],original_instance_indices={str(f['frame_index']):f['all_obj1_GT_instance_indices'] for f in frames},input_bindings_inherited_not_read=scope['allowed_body_bindings'],runtime=dict(executable=sys.executable,numpy=np.__version__,Pillow_import_spec=importlib.util.find_spec('PIL').origin,CPU_only=True,no_torch_model_or_encoder_import=True),
        geometry='BOP unmodified model xyz and model-to-camera Xc=R@Xm+t, mesh/t/depth in mm; K fromcam_K. Parse ASCII/little/bigendian PLY without processing, welding, repair, normalization, inferredscale orbestpose. Sourcepolygonfan primitives preserve originalface IDs.',
        raster='Integerpixelcenters u=x,v=y using suppliedK (BOP project_pts/depth image convention). Bothfaces, closed barycentric edges with fixed64Float64ULP numerical tolerance; perspective invZ barycentric; strict minpositiveZ; firstsource triangle exacttie. Nearclip fixed1e-6mm only numericalcamera-plane clipping, nofarplane. Targetalone eachinstance independently. No claim of officialVispy maskbyteidentity: raster/fill/pixel-boundary differences retained without tuningK/offset.',
        sensor='Integer depthPNG multiplied by exact frame depth_scale inmm. Rawzero/invalid nonpositive are missing; no interpolation/filling/alignment. Compare modelZ-depthZ only where bothvalid.',
        measurements='All5instances: full/visible mask rawoverlap with nominal silhouette, masksensor/render-known/missing partitions; rawsigned modelZ−sensorZ quantiles[0,1,5,25,50,75,95,99,100], mean/std/signcounts, originalfull/visible cross-overlap. No residual threshold selection or external/self/native labels.',
        outputs='OriginalencodedRGB bytecopy and originalcanonicalRGB fullsizePNG, separate nominalmeshoverlay and depthdiagnostic PNG, firstsurface float64 Z/triangleid/originalfaceid NPZ with sensor/masks/camera arrays, all5JSON/COMPLETE, offlineHTML links only. No crop/qL input used/changed.',
        display='Fixed color ranges model/sensor0..6000mm and signed−100..100mm, invalidblack; saturationcounts explicitlydisplayonly, no quantitative clipping. Fullimagepixels preserved inoriginalPNG; overlays separate copies.',
        authority='Registrationprobe/nominalgeometry evidence only; masks/pose/mesh correspondence to realphysical native8/endparts remainsunassumed. No label generation/trainingadmission/posefit/PnP/classifier/SAM/initializer/sourceVAL/private/held use.',output=str(OUT),new_training_admission=False,selected_model=None,adopted=False,fully_solved=False,automatic_retry=False)
    write(PROTOCOL,protocol);template=H/'ROOT_ACTUAL_SOURCE_REGISTRATION_ENTRY_TEMPLATE_APPROVED_FALSE.json';write(template,dict(approved_source_registration_probe=False,source_TRAIN_only=True,code=bind(__file__),protocol=bind(PROTOCOL),scope=bind(SCOPE),automatic_retry=False));write(H/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=bind(__file__),protocol=bind(PROTOCOL),checks=bind(CHECKS),entry_template=bind(template),actual_source_asset_header_body_or_hash_reads=0));print(json.dumps(dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),preparation=bind(H/'PREPARATION_COMPLETE.json'))))

def run(entry):
    e=read(entry);require(e.get('approved_source_registration_probe') is True and e.get('source_TRAIN_only') is True and e.get('automatic_retry') is False,'Root separateactual sourceprobe entry')
    require(e['code']==bind(__file__) and e['protocol']==bind(PROTOCOL) and e['scope']==bind(SCOPE),'Frozen code/protocol/scope');p=read(PROTOCOL);require(p['code']==bind(__file__) and p['checks']==bind(CHECKS),'Frozen checked code');scope=read(SCOPE);require(p['scope']==bind(SCOPE),'Exact sourcebindings');frames=validate_scope(scope);assets=scope['allowed_body_bindings']
    require(not OUT.exists(),'Single sourceprobe attempt; no overwrite orretry');OUT.mkdir();write(OUT/'STARTED_BEFORE_SOURCE_ASSET_BODIES.json',dict(entry=bind(entry),code=bind(__file__),protocol=bind(PROTOCOL),source_TRAIN_only=True));used=[];records=[];frame_records=[]
    try:
        mesh_path=checked(assets['mesh']);used.append(assets['mesh']);vertices,triangles,faceids,meshmeta=parse_ply_bytes(mesh_path.read_bytes());model_info=read(checked(assets['models_info']));used.append(assets['models_info']);meshmeta.update(bounds_min_mm=vertices.min(0),bounds_max_mm=vertices.max(0),declared_models_info_obj1=model_info['1'],unchanged_model_binding=assets['mesh'])
        for frame in frames:
            directory=OUT/('FRAME_%03d'%frame['frame_index']);directory.mkdir();bindings=assets['GT_camera'][frame['scene']];gtall=read(checked(bindings['GT']));camall=read(checked(bindings['camera']));used.extend([bindings['GT'],bindings['camera']]);gt=gtall[frame['frame_key']];cam=camall[frame['frame_key']]
            require([i for i,g in enumerate(gt) if g['obj_id']==1]==frame['all_obj1_GT_instance_indices'],'All sourceGTobj1instances exactno bestmatching')
            K=np.asarray(cam['cam_K'],np.float64).reshape(3,3);scale=float(cam['depth_scale']);require(np.isfinite(scale) and scale>0,'Exact positive BOP depth_scale')
            canonical=np.load(checked(frame['canonical_RGB_NPY']),allow_pickle=False);encoded=checked(frame['RGB']);rgb=np.asarray(Image.open(encoded).convert('RGB'));require(canonical.dtype==rgb.dtype==np.uint8 and canonical.shape==rgb.shape and canonical.ndim==3 and canonical.shape[2]==3,'Original RGB/canonical HWC shape');h,w=canonical.shape[:2]
            rawdepth=np.asarray(Image.open(checked(frame['depth'])));require(rawdepth.ndim==2 and rawdepth.shape==(h,w) and rawdepth.dtype.kind in 'ui','Original integer sensor depthPNG');sensor=rawdepth.astype(np.float64)*scale;sensor[(rawdepth<=0)|~np.isfinite(sensor)]=np.nan
            original=directory/'ORIGINAL_CANONICAL_RGB.png';Image.fromarray(canonical).save(original);shutil.copy2(encoded,directory/('ORIGINAL_ENCODED_RGB'+encoded.suffix));require(ash(np.asarray(Image.open(original).convert('RGB')))==ash(canonical),'OriginalcanonicalPNGdecoded unchanged')
            used.extend([frame['canonical_RGB_NPY'],frame['RGB'],frame['depth']]);frame_records.append(dict(frame_index=frame['frame_index'],source_id=frame['source_id'],split=frame['internal_split'],raw_hw=[h,w],original_RGB=bind(original),original_encoded_RGB=bind(directory/('ORIGINAL_ENCODED_RGB'+encoded.suffix)),canonical_array_sha256=ash(canonical),Pillow_original_decoded_sha256=ash(rgb),Pillow_vs_canonical_different_channel_samples=int((rgb!=canonical).sum()),Pillow_vs_canonical_max_channel_difference=int(np.abs(rgb.astype(np.int16)-canonical.astype(np.int16)).max()),depth_raw_dtype=str(rawdepth.dtype),depth_scale_mm=scale,sensor_missing_depth_all=int((~np.isfinite(sensor)).sum()),sensor_depth_quantiles=quantiles(sensor),original_GT_indices=frame['all_obj1_GT_instance_indices']))
            fullmaps={};vismaps={}
            for index in frame['all_obj1_GT_instance_indices']:
                db=next(r['binding'] for r in frame['full_masks'] if r['gt_instance_index']==index);vb=next(r['binding'] for r in frame['visible_masks'] if r['gt_instance_index']==index);fullraw=np.asarray(Image.open(checked(db)));visraw=np.asarray(Image.open(checked(vb)));used.extend([db,vb]);require(fullraw.ndim==visraw.ndim==2 and fullraw.shape==visraw.shape==(h,w) and fullraw.dtype.kind in 'buif' and visraw.dtype.kind in 'buif','Provided instance mask raster schema');full=fullraw!=0;visible=visraw!=0;fullmaps[index]=full;vismaps[index]=visible
                pose=gt[index];R=np.asarray(pose['cam_R_m2c'],np.float64).reshape(3,3);t=np.asarray(pose['cam_t_m2c'],np.float64).reshape(3);instance_dir=directory/('INSTANCE_%03d'%index);instance_dir.mkdir();raster,rastermeta=rasterize(vertices,triangles,faceids,K,R,t,[h,w]);delta=raster['model_Z_mm']-sensor;delta[~raster['silhouette']|~np.isfinite(sensor)]=np.nan
                path=instance_dir/'FIRST_SURFACE_AND_RAW_DIAGNOSTICS.npz';np.savez_compressed(path,**raster,sensor_depth_raw=rawdepth,sensor_Z_mm=sensor,sensor_depth_valid=np.isfinite(sensor),signed_modelZ_minus_sensorZ_mm=delta,provided_full_mask_raw=fullraw,provided_visible_mask_raw=visraw,provided_full_mask=full,provided_visible_mask=visible,K=K,R_m2c=R,t_m2c_mm=t)
                data=dict(frame_index=frame['frame_index'],id=frame['id'],source_id=frame['source_id'],split=frame['internal_split'],obj_id=1,gt_instance_index=index,all_obj1_indices=frame['all_obj1_GT_instance_indices'],camera_K=K,R_m2c=R,t_m2c_mm=t,depth_scale_mm=scale,R_determinant=float(np.linalg.det(R)),R_orthogonality_max_abs=float(np.abs(R.T@R-np.eye(3)).max()),render=rastermeta,provided_full_mask_values=np.unique(fullraw),provided_visible_mask_values=np.unique(visraw),visible_outside_full_pixels=int((visible&~full).sum()),full_mask=compare(raster,sensor,full),visible_mask=compare(raster,sensor,visible),rendered_silhouette=compare(raster,sensor,raster['silhouette']),mask_union=compare(raster,sensor,full|visible),all_image=compare(raster,sensor,np.ones((h,w),bool)),arrays=bind(path),images=diagnostics(canonical,raster,sensor,full,visible,f'frame{frame["frame_index"]} GTinstance{index}',instance_dir),original_qL_native_pose_K_or_mesh_changes=0,new_labels_or_training_admission=False)
                write(instance_dir/'REGISTRATION_DIAGNOSTICS.json',data);records.append(data);print(json.dumps(dict(frame_index=frame['frame_index'],gt_instance_index=index,completed_instances=len(records),all5=5)),flush=True)
            frame_records[-1]['provided_full_mask_pair_overlaps']=[dict(instance_a=a,instance_b=b,pixels=int((fullmaps[a]&fullmaps[b]).sum())) for a in fullmaps for b in fullmaps if a<b];frame_records[-1]['provided_visible_mask_pair_overlaps']=[dict(instance_a=a,instance_b=b,pixels=int((vismaps[a]&vismaps[b]).sum())) for a in vismaps for b in vismaps if a<b]
        require(len(records)==5 and len(frame_records)==4,'All fixed fourframes/all5instances retained');write(OUT/'ALL5_REGISTRATION_DIAGNOSTICS.json',dict(completed=True,mesh=meshmeta,frames=frame_records,instances=records));body=['<!doctype html><html lang="ko"><meta charset="utf-8"><title>V186 source registration probe</title><style>body{font:18px system-ui;max-width:1500px;margin:auto;padding:24px}img{width:100%}</style><h1>공개 TRAIN 4프레임 · nominal 등록 probe</h1><p>원 mesh/K/R/t 그대로, 모든 obj1 instance5개를 독립 target-alone 렌더링했습니다. 실사 native/부위/외부·자기가림 정답이나 학습 편입이 아닙니다. pixel-center/fill 규약 차이와 미측정 깊이를 보존하며 IoU에 맞춘 포즈·K 수정은 없습니다. HTML browser 검증은 주장하지 않습니다.</p>']
        for f in frame_records:body.append('<h2>'+f['source_id']+'</h2><p>원 canonical RGB · 별도 overlay만 표시용</p><img src="'+str(Path(f['original_RGB']['path']).relative_to(OUT))+'">')
        for r in records:
            body.append(f'<h2>frame{r["frame_index"]} GTinstance{r["gt_instance_index"]}</h2>'+''.join('<p>'+key+'</p><img src="'+str(Path(r['images'][key]['path']).relative_to(OUT))+'">' for key in ['overlay','depth_diagnostic']))
        index=OUT/'index.html';index.write_text('\n'.join(body)+'</html>');[checked(b) for b in used];checked(p['scope']);require(p['code']==bind(__file__),'Sourcecode unchanged')
        write(OUT/'COMPLETE.json',dict(completed=True,PASS=True,code=bind(__file__),protocol=bind(PROTOCOL),scope=bind(SCOPE),entry=bind(entry),all_original_source_bindings_unchanged=used,fixed_source_frames=[f['frame_index'] for f in frames],all_obj1_instances=5,summary=bind(OUT/'ALL5_REGISTRATION_DIAGNOSTICS.json'),index=bind(index),original_images=[f['original_RGB'] for f in frame_records],overlay_and_depth_images=[r['images'] for r in records],geometry_registration_evidence_only=True,source_TRAIN_only=True,new_native_part_owner_quality_visibility_labels=0,new_training_admission=False,new_fit_model_SAM_initializer_PnP_qL_pose_K_or_mesh_change=0,actual_visual_inspection='pending separate tools.view_image receipt',HTML_browser_rendered=False,selected_model=None,adopted=False,fully_solved=False,automatic_retry=False));print(json.dumps(dict(completed=True,PASS=True,complete=bind(OUT/'COMPLETE.json'))))
    except BaseException:
        write(OUT/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),automatic_retry=False));raise

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare() if a.prepare else run(a.entry)
