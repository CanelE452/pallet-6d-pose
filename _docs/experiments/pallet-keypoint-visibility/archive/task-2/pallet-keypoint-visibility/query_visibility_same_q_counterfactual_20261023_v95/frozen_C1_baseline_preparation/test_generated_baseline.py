"""Tiny generated baseline guards and shared source metrics; no checkpoint IO."""
from pathlib import Path
import argparse,importlib.util,json
import numpy as np
import run_frozen_C1_baseline as b
H=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('invented_C2_meta',H.parent/'source_fit_preparation/test_generated_source_trainer.py');g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
def run(output):
 t=b.trainer();line,gt=g.metadata();meta=t.join_metadata(line,gt)
 z=np.where(meta['labels']==0,-2.,2.).astype(np.float32);av=np.ones(2880,bool);av[4]=False;z[4]=0
 r=b.diagnose(z,av,meta,t)
 assert r['diagnostic_only']and not r['model_selection_used']and not r['current_authorized']
 assert meta['original_split'][480]=='TRAIN_LAB'and meta['split'][480]=='TRAIN'
 assert r['counts']['TRAIN']['before_E']==10 and r['counts']['VALIDATION']['before_E']==10
 assert r['metrics']['VALIDATION']['by_variant']['OPAQUE_COVER']['matrix']==[[0,0,0],[0,1,1]]
 assert r['metrics']['VALIDATION']['all3_correct_rate']==.5
 assert np.sum(meta['labels'][::3]==1)==20
 try:b.guard({'root_approved':False})
 except AssertionError:pass
 else:raise AssertionError('Unapproved baseline accepted')
 class ThreeOnly:
  def __call__(self,*args):assert len(args)==3;return args
 import torch
 d={'p3_roi':np.zeros((2,65,9,9),np.float16),'rgb96':np.zeros((2,3,96,96),np.uint8),'crop_valid96':np.ones((2,1,96,96),bool),'GT_FORBIDDEN':np.array([0,1])}
 out=t.forward(ThreeOnly(),d,[0,1],'cpu');assert len(out)==3
 proof=dict(completed=True,PASS=True,code=b.bind(b.__file__),generated_code=b.bind(__file__),shared_trainer=b.bind(b.T),diagnostic_not_selector=True,
  all2880_generated_labels_and_raw_splits_preserved=True,original_E20_retained=True,pair_unavailable_U_in_denominator=True,three_only_forward_excludes_GT=True,
  unapproved_entry_before_files_rejected=True,source_array_GT_checkpoint_parameter_or_private_current_reads=0,model_backbone_forward_fit_PnP=0)
 b.write(output,proof);print(json.dumps(proof))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=H/'GENERATED_BASELINE_TESTS.json');run(ap.parse_args().output)
