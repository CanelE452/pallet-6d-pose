"""V89 deterministic input geometry; PNG header only; no GT arrays or RGB decode."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,struct,datetime,math,argparse
ROOT=Path(__file__).resolve().parent

def binding(path):
 p=Path(path);x=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(x).hexdigest(),'bytes':len(x)}
def read(path):return json.loads(Path(path).read_text())
def save(path,obj):
 with Path(path).open('x')as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def centres(dim,side):
 assert type(dim)is int and dim>0 and side in (128,256)
 return sorted(set(range(0,dim,side//2))|{dim-1})
def geometry(W,H,x,y,L):
 assert 0<=x<W and 0<=y<H
 lo_x=x-L/2.;lo_y=y-L/2.;hi_x=x+L/2.;hi_y=y+L/2.
 nx=max(0,min(W,math.ceil(hi_x))-max(0,math.ceil(lo_x)))
 ny=max(0,min(H,math.ceil(hi_y))-max(0,math.ceil(lo_y)))
 return {'window_xyxy_half_open':[lo_x,lo_y,hi_x,hi_y],'native_valid_pixels':nx*ny,'padding_pixels':L*L-nx*ny,'native_valid_fraction':nx*ny/(L*L)}
def png_header(path):
 with Path(path).open('rb')as f:h=f.read(24)
 assert len(h)==24 and h[:8]==b'\x89PNG\r\n\x1a\n' and h[12:16]==b'IHDR'
 W,H=struct.unpack('>II',h[16:24]);assert W>0 and H>0
 return H,W,h.hex()
def fixture():
 results=[]
 for W,H in [(1,1),(2,3),(64,64),(127,129),(128,128),(129,257),(640,480),(641,481)]:
  for L in (128,256):
   xs,ys=centres(W,L),centres(H,L)
   assert xs[0]==ys[0]==0 and xs[-1]==W-1 and ys[-1]==H-1
   assert len(xs)==len(set(xs))and len(ys)==len(set(ys))
   for x,y in [(xs[0],ys[0]),(xs[-1],ys[-1]),(xs[len(xs)//2],ys[len(ys)//2])]:
    g=geometry(W,H,x,y,L);a,b,c,d=g['window_xyxy_half_open']
    # Independent enumeration of integer native pixel centres, not formula reuse.
    n=sum(a<=i<c for i in range(W))*sum(b<=j<d for j in range(H))
    assert n==g['native_valid_pixels'] and n+g['padding_pixels']==L*L
   results.append({'raw_hw':[H,W],'L':L,'x':xs,'y':ys,'tiles':len(xs)*len(ys)})
 assert len(centres(640,128))*len(centres(480,128))==99
 assert len(centres(640,256))*len(centres(480,256))==30
 return {'PASS':True,'cases':results,'checks':'lastcentre de-duplicated; half-open validity independently enumerated incl1px/edge and asymmetricnativeendpoint; noimagesread'}
def build():
 protocol=read(ROOT/'BEFORE_GRID_PROTOCOL.json');cohort_b=protocol['source_cohort'];assert binding(cohort_b['path'])==cohort_b
 entry=read(ROOT/'INPUT_GRID_ENTRYLOCK_BEFORE_HEADERS.json');assert entry['builder']==binding(__file__)
 cohort=read(cohort_b['path']);sources=cohort['rows']
 assert len(sources)==50 and len({r['pair_id']for r in sources})==50 and sum(r['seed_V84_reuse']for r in sources)==1
 target=ROOT/'inputs_metadata';target.mkdir(exist_ok=False)
 frames=[];counts=Counter();scale_count=Counter();headers=[];per_frame_counts=[]
 gridpath=target/'INPUTS_GRID_ONLY.jsonl'
 with gridpath.open('x')as f:
  for source in sources:
   for view in ('before','after'):
    image=source[view+'_RGB'];H,W,hhex=png_header(image['path']);headers.append({'image':image['path'],'header24_hex':hhex})
    fid='PALLOX50:'+source['pair_id']+':'+view
    frame={'frame_input_id':fid,'image':image['path'],'image_sha256':image['sha256'],'raw_hw':[H,W],'pair_id':source['pair_id'],'view':view,'split':source['split'],'source_family':'ONE_PALLOX_PHYSICAL_BIN','session':source['session'],'seed_V84_reuse':source['seed_V84_reuse']}
    n=0
    for L in (128,256):
     for y in centres(H,L):
      for x in centres(W,L):
       tile={'tile_id':fid+f':L{L}:x{x}:y{y}',**frame,'query_xy':[float(x),float(y)],'roi_side':L,**geometry(W,H,x,y,L)}
       f.write(json.dumps(tile,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');n+=1
       counts[source['split']]+=1;scale_count[L]+=1
    frames.append(frame);per_frame_counts.append(n)
 assert len(frames)==100 and len({r['frame_input_id']for r in frames})==100
 assert len({r['image']for r in frames})==100 and len({r['image_sha256']for r in frames})==100
 with (target/'FRAMES_INPUTS_ONLY.jsonl').open('x')as f:
  for r in frames:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
 save(target/'PNG_HEADER24_ONLY_RECEIPT.json',{'completed':True,'files':headers,'bytes_read_per_PNG':24,'RGBdecoded':0,'full_file_SHA_recomputed':False,'source_SHA':'Copied immutable cohort SHAmetadata, not newrawpixelread'})
 summary={'completed':True,'pairs':50,'frames':100,'tiles':sum(per_frame_counts),'frames_by_split':dict(Counter(r['split']for r in frames)),'tiles_by_split':dict(counts),'tiles_by_L':dict(scale_count),'raw_HW_counts':dict(Counter(str(r['raw_hw'])for r in frames)),'per_frame_tile_counts':dict(Counter(per_frame_counts)),'seed_pairs':1,'seed_frames':sum(r['seed_V84_reuse']for r in frames),'native8_slots_or_context42_created':False,'query_selection_GT_free':True,'selection_data':'Only original cohort image lineage, source PNGheadernativeHW and fixedL/stridefunction','actual_crop_RGB_P3_arrays_created':0,'GTjoin_performed':False,'training_admission':False,'prior_GT_exposure_disclosed':True,'protocol':binding(ROOT/'BEFORE_GRID_PROTOCOL.json'),'builder':binding(__file__),'entrylock':binding(ROOT/'INPUT_GRID_ENTRYLOCK_BEFORE_HEADERS.json')}
 save(target/'INPUT_GRID_SUMMARY_KO.json',summary)
 files=sorted(p for p in target.iterdir()if p.is_file())
 save(target/'INPUT_GRID_SEAL_BEFORE_GT_JOIN.json',{'completed':True,'PASS':True,'stage':'INPUT_MANIFEST_SEALED_BEFORE_GT_JOIN','files':[binding(p)for p in files],'summary':binding(target/'INPUT_GRID_SUMMARY_KO.json'),'protocol':binding(ROOT/'BEFORE_GRID_PROTOCOL.json'),'native8_gold_or_training_admission':False,'RGBdecode_crop_feature_forward_fit_current_PnP':0})
 print(json.dumps({'summary':summary,'input_manifest':binding(gridpath),'input_seal':binding(target/'INPUT_GRID_SEAL_BEFORE_GT_JOIN.json')},ensure_ascii=False,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--fixtures-only',action='store_true');a=ap.parse_args()
 if a.fixtures_only:
  r=fixture();save(ROOT/'GENERATED_GRID_EDGE_FIXTURES.json',r);print(json.dumps({'PASS':r['PASS'],'cases':len(r['cases'])}))
 else:build()
