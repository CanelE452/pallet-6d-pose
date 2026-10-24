"""Small invented numerical cases only; no source pixels or models."""
from pathlib import Path
import importlib.util,json
import numpy as np
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('generated_y109',H/'build_surface_source.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def main():
 ones=np.ones(9,bool);zeros=np.zeros(9,bool);count=np.ones(9,np.uint8);raw=np.full(9,100,np.uint16);ownbits=np.ones(9,np.uint64);foreign=np.full(9,2,np.uint64)
 def label(g='SELF_VISIBLE',q=True,r=True,dep=raw,owners=count,own=ones,bits=ownbits,full=ones,pixelZ=100.):return m.classify(g,q,r,ones,dep,owners,own,bits,full,0,1.,100.,pixelZ)
 assert label()[0]=='D'
 assert label(dep=np.full(9,116,np.uint16))[0]=='U'
 eraw=np.full(9,50,np.uint16);assert label(dep=eraw,own=zeros,bits=foreign)[0]=='E'
 center0=eraw.copy();center0[4]=0;assert label(dep=center0,own=zeros,bits=foreign)[0]=='U'
 six=eraw.copy();six[[0,1,2]]=0;assert label(dep=six,own=zeros,bits=foreign)[0]=='U'
 overlap=count.copy();overlap[0]=2;assert label(dep=eraw,owners=overlap,own=zeros,bits=foreign)[0]=='U'
 mixed=foreign.copy();mixed[0]=4;assert label(dep=eraw,own=zeros,bits=mixed)[0]=='U'
 assert label(dep=eraw,own=zeros,bits=foreign,full=zeros)[0]=='U'
 assert label(g='SELF')[0]=='S'and label(g='SELF',q=False)[0]=='U'
 assert m.geometry_state(np.array([10.,10.]),12.)[0]=='SELF_VISIBLE'
 assert m.geometry_state(np.array([10.,10.]),12.01)[0]=='SELF'
 assert m.geometry_state(np.array([10.,10.002]),12.)[0]=='UNKNOWN'
 assert m.geometry_state(np.array([np.nan,10.]),12.)[0]=='UNKNOWN'
 kernel=m.module(m.KERNEL,'generated_pilot_ray');xyz=np.array([[0.,0.,10.],[2.,0.,10.],[0.,2.,10.],[0.,0.,20.],[2.,0.,20.],[0.,2.,20.]]);faces=np.array([[0,1,2],[3,4,5]])
 a=kernel.ray_hits(xyz,faces,np.array([0.,0.,20.]),20.,{1});assert all(x['first_hit_range_mm']==10. for x in a['sensitivity'])
 r=np.array([[x,y]for x in (0.,40.)for y in (0.,40.,80.,120.)]);q=m.jitter_q(r,np.random.Generator(np.random.PCG64(207)));assert np.array_equal(q,m.jitter_q(r,np.random.Generator(np.random.PCG64(207))))and np.linalg.norm(q-r,axis=1).max()<=4
 _,_,L=m.native_grid_function()(q,[480,640]);assert L==float(np.clip(.25*np.max(np.ptp(q,axis=0)),16,640))
 proof=dict(completed=True,PASS=True,code=m.bind(H/'build_surface_source.py'),test_code=m.bind(Path(__file__)),checks=['D strict depth/owner','E positive nearer centre','E at least7 positive','sameforeign9/overlap/amodal','S separate and offimage priority','two-bary stability and fixed2mm','target-alone two-plane ray','PCG64disk4 and exact V60L'],source_RGB_depth_mask_pixels_read=0,model_fit_private_PnP=0)
 m.write(H/'GENERATED_POLICY_TESTS.json',proof);print(json.dumps(dict(PASS=True,proof=m.bind(H/'GENERATED_POLICY_TESTS.json'))))
if __name__=='__main__':main()
