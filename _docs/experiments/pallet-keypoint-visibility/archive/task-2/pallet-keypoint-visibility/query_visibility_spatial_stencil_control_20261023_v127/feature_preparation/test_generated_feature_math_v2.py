"""Generic NumPy/static fixtures only; never imports Torch or source payloads."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,sys,tempfile
import numpy as np
H=Path(__file__).resolve().parent;V=H.parent;CODE=H/'prepare_stencil_features_v2.py'
s=importlib.util.spec_from_file_location('v127_feature_generated',CODE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);c=m.c

def main(out):
 checks={}
 yy,xx=np.meshgrid(np.arange(16),np.arange(16),indexing='ij');grid=np.broadcast_to((100*yy+xx)[None,:,:,None],(2,16,16,384)).astype(np.float32).copy();grid+=np.arange(384,dtype=np.float32)[None,None,None,:]
 raw,pool,ordered=c.stencil_numpy(grid)
 expected=np.array([100*(y+.5)+(x+.5)for y in c.AXIS for x in c.AXIS],np.float32)
 checks['nine_bilinear_half_node_phase']=np.array_equal(raw[0,:,0],expected)
 checks['rowmajor_outer8_order']=c.OUTER==(0,1,2,3,5,6,7,8)and np.array_equal(ordered.reshape(2,8,384),np.stack([c.normalize_np(raw[:,k])for k in c.OUTER],1))
 checks['centre_reduction_literal_C11_mean']=np.array_equal(raw[:,4],grid[:,7:9,7:9].mean((1,2)))
 checks['quarter_L_mapping']=np.array_equal((np.array([3.5,7.5,11.5])+.5)/16-.5,[-.25,0,.25])
 checks['pool_RAW9_then_L2_repeat8']=np.array_equal(pool.reshape(2,8,384),np.repeat(c.normalize_np(raw.mean(1))[:,None,:],8,1))
 fixture=np.array([[3.,0.]]*5+[[0.,1.]]*4,np.float32)
 checks['raw_pool_not_normalize_then_pool']=not np.allclose(c.normalize_np(fixture.mean(0)),c.normalize_np(c.normalize_np(fixture).mean(0)))
 checks['shapes_float32_finite']=raw.shape==(2,9,384)and pool.shape==ordered.shape==(2,3072)and all(x.dtype==np.float32 and np.isfinite(x).all()for x in(raw,pool,ordered))
 checks['zero_nodes_safe']=not c.stencil_numpy(np.zeros_like(grid))[1].any()
 checks['capacity_arithmetic']=3464*64+64+3072*64+64+1==418433
 source=CODE.read_text();tree=ast.parse(source);run=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='run');rt=ast.unparse(run)
 checks['original3n_and_full_row_positions']="torch.cat((local[:n], local), 0)"in rt and "out['x_norm_clstoken'][n:2 * n]"in rt and "out['x_norm_patchtokens'][n:2 * n]"in rt
 checks['strict_F_P_and_support_masks']='torch.equal(full, expectedF)'in rt and 'torch.equal(pv, expectedP)'in rt and 'frac[:, 0] > 0'in rt and 'vv[:, 0, 47:49, 47:49]'in rt and 'allclose'not in rt and 'isclose'not in rt
 checks['no_Torch_or_model_import_in_generated_process']='torch'not in sys.modules
 checks['no_GT_or_label_binding_in_feature_protocol_prepare']='GT_occurrences'not in ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='prepare'))
 with tempfile.TemporaryDirectory()as td:
  p=Path(td)/'noentry.json';p.write_text(json.dumps({'approved_source_stencil_feature_extraction':False}))
  try:m.authorize(p)
  except AssertionError:checks['unapproved_entry_stops_before_source_IO']=True
  else:raise AssertionError('Entry approval missing')
 assert all(checks.values()),checks
 result=dict(completed=True,PASS=True,feature_code=c.bind(CODE),common_code=c.bind(V/'stencil_common.py'),checks={k:bool(v)for k,v in checks.items()},runtime=dict(Python=sys.version.split()[0],NumPy=np.__version__,Torch_imported=False),actual_source_input_RGB_feature_weight_model_forward_fit_current_V112=0)
 with Path(out).open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);main(ap.parse_args().output)
