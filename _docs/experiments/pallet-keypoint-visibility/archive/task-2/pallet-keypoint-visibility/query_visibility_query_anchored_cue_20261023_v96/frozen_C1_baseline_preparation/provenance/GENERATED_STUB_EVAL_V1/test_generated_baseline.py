"""Generated metadata/three-tensor stub only; no C1 checkpoint or source reads."""
from pathlib import Path
import argparse,importlib.util,json
import numpy as np
import run_frozen_C1_baseline as b
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('c3_generated_metadata_only',H.parent/'source_fit_preparation/test_generated_source_trainer.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
def run(output):
 t=b.trainer();line,gt=g.metadata();meta=t.join_metadata(line,gt);z=np.where(meta['labels']==0,-2.,2.).astype(np.float32);av=np.ones(31680,bool);av[34]=False;z[34]=0
 r=b.diagnose(z,av,meta,t);assert r['diagnostic_only']and not r['model_selection_used']and not r['current_authorized']and r['all31680_preserved']
 assert meta['original_split'][33*160]=='TRAIN_LAB'and meta['split'][33*160]=='TRAIN'
 assert r['counts']['TRAIN']['before_E']==r['counts']['VALIDATION']['before_E']==2
 assert r['metrics']['VALIDATION']['paired']['all3_correct']==5 and r['metrics']['VALIDATION']['paired']['all3_correct_rate']==5/6
 assert np.sum(meta['labels'][::33]==1)==4
 try:b.guard({'root_approved':False})
 except AssertionError:pass
 else:raise AssertionError('Unapproved baseline accepted')
 class ThreeOnly:
  def __call__(self,*args):
   assert len(args)==3
   import torch
   return {'query_logit':torch.zeros(len(args[0])),'query_available':torch.ones(len(args[0]),dtype=torch.bool)}
 d={'p3_roi':np.zeros((2,65,9,9),np.float16),'rgb96':np.zeros((2,3,96,96),np.uint8),'crop_valid96':np.ones((2,1,96,96),bool),'GT_FORBIDDEN':np.array([0,1])}
 out=t.forward(ThreeOnly(),d,[0,1],'cpu');assert set(out)=={'query_logit','query_available'}
 z,a=t.predict(ThreeOnly(),d,np.array([0,1,0,0,1]),'cpu');assert len(z)==len(a)==5 and np.all(z==0)and a.all()
 b.write(output,dict(completed=True,PASS=True,code=b.bind(b.__file__),generated_code=b.bind(__file__),shared_trainer=b.bind(b.T),diagnostic_not_selector=True,all31680_generated_occurrences_preserved=True,natural_E4_before_only=True,rank_aware_triples_with_U_denominator=True,raw_source_split_mapping_fixed=True,three_only_forward_excludes_GT=True,unique_units_prediction_expands_to_all_occurrences=True,unapproved_entry_before_files_rejected=True,source_arrays_GT_checkpoint_parameter_private_reads=0,model_backbone_fit_PnP=0))
 print(json.dumps(b.bind(output)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=H/'GENERATED_BASELINE_TESTS.json');run(p.parse_args().output)
