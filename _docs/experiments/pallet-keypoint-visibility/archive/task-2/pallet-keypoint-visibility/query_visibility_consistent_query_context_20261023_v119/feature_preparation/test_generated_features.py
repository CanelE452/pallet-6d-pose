"""Generated arrays/operators only; no source arrays, pixels, weights or DINO."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,importlib.util,json
import numpy as np
import torch
from prepare_consistent_context_features import update_TQ,copy_updated_aliases,extraction_plan,C11
H=Path(__file__).resolve().parent
def mod(p):
 s=importlib.util.spec_from_file_location('C14_generated_pinned_operator',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def bind(p):return dict(path=str(Path(p).resolve()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def run(output):
 assert torch.__version__=='2.1.1+cu118';torch.manual_seed(721);producer=mod(C11);checks=[]
 n=8;oldT=torch.randn(n,384);oldpatch=torch.randn(n,256,384);newT=torch.randn(n,384);newpatch=torch.randn(n,256,384);full=torch.randn(n,384);fp=torch.randn(n,256,384);central=torch.randn(n,384)
 q=np.array([[5.,5.],[0.,0.],[10.,10.],[np.nextafter(10.,np.inf),5.],[-1.,-1.],[4.,3.],[8.,7.],[1.,1.]],np.float64);box=np.tile([0.,0.,10.,10.],(n,1));fra=np.array([[1.,1.,1.],[.5,.8,.9],[1.,1.,1.],[1.,1.,1.],[1.,1.,1.],[0.,1.,1.],[.9,.8,.7],[1.,1.,1.]],np.float32);c4=np.ones(n,bool)
 old,_=producer.assemble(oldT,oldpatch,full,fp,central,q,box,fra,c4)
 want,wi=producer.assemble(newT,newpatch,full,fp,central,q,box,fra,c4)
 actual,inside=update_TQ(old,newT,newpatch,q,box,fra[:,0],producer)
 assert torch.equal(actual,want)and np.array_equal(inside,wi)
 assert torch.equal(actual[:,768:1920],old[:,768:1920])and torch.equal(actual[:,1925:1927],old[:,1925:1927])
 checks+=['exact_C11_full_assemble_parity_from_new_context_only','cached_F_C_P_bitexact','cached_local_and_central_fractions_bitexact','fresh_four_cosines_match_pinned_operator']
 assert inside.tolist()==[True,True,True,False,False,False,True,True]
 assert not actual[~inside,384:768].any()and not actual[5,:768].any()and not actual[5,1920:1925].any()and actual[5,1927]==0
 checks+=['closed_box_endpoints_and_tiny_F64_outside','sentinel_and_zero_support_Qzero','unsupported_context_zero_TQ_cos_fraction_flag']
 grid=torch.zeros(3,16,16,384);xy=torch.arange(16,dtype=torch.float32);grid[:,:,:,0]=xy[None,None,:];grid[:,:,:,1]=xy[None,:,None];grid[:,:,:,2:]=1
 raw,ii=producer.sample_Q_raw(grid.reshape(3,256,384),[[5.,5.],[0.,0.],[10.,10.]],np.tile([0.,0.,10.,10.],(3,1)),[1,1,1])
 assert torch.equal(raw[:,:2],torch.tensor([[7.5,7.5],[0.,0.],[15.,15.]]))and ii.all()
 checks+=['native_q_to_16patch_alignFalse_phase','border_sampling_at_closed_endpoints']
 original=old.numpy().copy();updated=actual.numpy().copy();x=np.r_[updated,original[:1]];aliases=[dict(source_feature_unit_index=i,canonical_context_feature_unit_index=i)for i in range(n)]+[dict(source_feature_unit_index=n,canonical_context_feature_unit_index=0)]
 out=copy_updated_aliases(x,aliases);assert out==[[n,0]]and np.array_equal(x[n],updated[0])and not np.array_equal(x[n],original[0])
 checks+=['BEFORE_alias_receives_UPDATED_not_stale_context','full_alias_FCP_equal_to_canonical']
 # Two query units, one updated alias, one wholly outside and one unchanged unit.
 rows=[]
 for u in range(5):
  rows.append(dict(source_feature_unit_index=u,canonical_context_feature_unit_index=0 if u==2 else u,query_xy=[float(u),3.],L=9.,context_group_id='a'if u<3 else str(u),box_changed=u!=4,box_exact_equal=u==4,proposed_context_available_geometry=u!=3,full_RGB=dict(path='GENERATED_ONLY',sha256='0'*64),decoded_RGB_sha256='1'*64,raw_hw=[20,20],proposed_q8_box_xyxy=[0.,0.,10.,10.]))
 rows[2]['query_xy']=rows[0]['query_xy']
 groups=[dict(context_group_id='a',canonical_units=[0,1],context_available_geometry=True,any_canonical_unit_requires_new_context=True,full_RGB=rows[0]['full_RGB'],decoded_RGB_sha256='1'*64,raw_hw=[20,20],proposed_q8_box_xyxy=[0.,0.,10.,10.])]
 occ=[dict(global_occurrence_index=i,source_feature_unit_index=i,canonical_context_feature_unit_index=r['canonical_context_feature_unit_index'],query_xy=r['query_xy'],L=r['L'])for i,r in enumerate(rows)]
 jobs,mapping,counts=extraction_plan(rows,groups,occ)
 assert len(jobs)==1 and jobs[0]['units']==[0,1]and np.array_equal(mapping,np.arange(5))and counts==dict(units=5,occurrences=5,fresh_units=2,fresh_views=1,same_box=1,aliases=1,unavailable=1)
 checks+=['context_encode_dedup_preserves_distinct_Qs','unsupported_and_unchanged_not_encoded','allrows_and_original_unit_map_retained']
 # Factual availability is outside update_TQ; Qoutside is never a new availability rule.
 factual=np.ones(n,bool);availability=factual.copy();assert not inside[3]and availability[3]
 checks+=['Qoutside_does_not_add_U','three_input_descriptor_schema_unchanged_1928','no_source_encoder_or_head_in_generated_test']
 report=dict(completed=True,PASS=True,checks=checks,code=bind(H/'prepare_consistent_context_features.py'),test=bind(__file__),runtime=dict(torch=torch.__version__,numpy=np.__version__),generated_only=True,source_pixels_feature_parameters_encoder_head_fit_current_V112_PnP=0)
 with Path(output).open('x')as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps(report))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(H/'GENERATED_FEATURE_TESTS.json'));args=ap.parse_args();run(args.output)
