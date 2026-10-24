"""Small invented arrays only; no source pixel, label or model evaluation."""
from pathlib import Path
import importlib.util,json,sys
sys.dont_write_bytecode=True
import numpy as np
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('v113_generated',H/'build_material_teacher.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def main():
 yes=np.ones(9,bool);no=np.zeros(9,bool);count=np.ones(9,np.uint8)
 raw=np.full(9,100,np.uint16);bits=np.ones(9,np.uint64);foreign=np.full(9,2,np.uint64)
 def label(g='SELF_VISIBLE',q=True,r=True,dep=raw,owners=count,own=yes,owner_bits=bits,full=yes,z=100.):
  return m.classify(g,q,r,yes,dep,owners,own,owner_bits,full,0,1.,100.,z)
 assert m.classify is m.old.classify and m.geometry_state is m.old.geometry_state
 assert label()[0]=='D'
 assert label(dep=np.full(9,116,np.uint16))[0]=='U'
 e=np.full(9,50,np.uint16);assert label(dep=e,own=no,owner_bits=foreign)[0]=='E'
 missing=e.copy();missing[4]=0;assert label(dep=missing,own=no,owner_bits=foreign)[0]=='U'
 assert label(g='SELF',dep=np.zeros(9,np.uint16))[0]=='S'
 assert label(g='SELF',q=False)[0]=='U'and label(g='SELF',r=False)[0]=='U'
 overlap=count.copy();overlap[0]=2;assert label(dep=e,owners=overlap,own=no,owner_bits=foreign)[0]=='U'
 assert label(dep=e,own=no,owner_bits=foreign,full=no)[0]=='U'
 assert m.geometry_state(np.array([10.,10.]),12.)[0]=='SELF_VISIBLE'
 assert m.geometry_state(np.array([10.,10.]),12.01)[0]=='SELF'
 assert m.geometry_state(np.array([10.,10.002]),12.)[0]=='UNKNOWN'
 r=np.array([[x,y]for x in(0.,40.)for y in(0.,40.,80.,120.)]);q=m.jitter_q(r,np.random.Generator(np.random.PCG64(213)))
 d=np.random.Generator(np.random.PCG64(213)).random((8,2));expected=r+np.stack((4*np.sqrt(d[:,1])*np.cos(2*np.pi*d[:,0]),4*np.sqrt(d[:,1])*np.sin(2*np.pi*d[:,0])),1)
 assert np.array_equal(q,expected)and np.linalg.norm(q-r,axis=1).max()<=4
 _,_,L=m.native_grid_function()(q,[480,640]);assert L==float(np.clip(.25*np.max(np.ptp(q,axis=0)),16,640))
 kernel=m.module(m.KERNEL,'v113_generated_ray');xyz=np.array([[0.,0.,10.],[2.,0.,10.],[0.,2.,10.],[0.,0.,20.],[2.,0.,20.],[0.,2.,20.]])
 a=kernel.ray_hits(xyz,np.array([[0,1,2],[3,4,5]]),np.array([0.,0.,20.]),20.,{1})
 assert all(x['first_hit_range_mm']==10. for x in a['sensitivity'])
 proof=dict(completed=True,PASS=True,code=m.bind(H/'build_material_teacher.py'),test_code=m.bind(__file__),exact_functions=m.bind(m.OLD),
            checks=['V109 exact function identity','D/E centre-depth rules','S priority and offimage U','ownership overlap/amodal support','two-ray fixed geometry boundary','PCG64seed213 disk4 exactdraw','exactV60 L','unchanged target-alone ray'],
            source_pixels_read=0,teacher_labels_generated=False,learned_model_fit_private_PnP=0)
 m.write(H/'GENERATED_POLICY_TESTS.json',proof);print(json.dumps(dict(PASS=True,proof=m.bind(H/'GENERATED_POLICY_TESTS.json'))))
if __name__=='__main__':main()
