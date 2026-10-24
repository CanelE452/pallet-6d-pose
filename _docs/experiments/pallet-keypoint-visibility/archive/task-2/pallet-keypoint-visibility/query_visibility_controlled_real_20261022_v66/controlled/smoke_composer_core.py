"""Generated-array causal paste/partial-truth checks; no source image reads."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import composer_core as core
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'contract/controlled_truth.py';spec=importlib.util.spec_from_file_location('controlled_truth_smoke',p)
t=importlib.util.module_from_spec(spec);sys.modules[spec.name]=t;spec.loader.exec_module(t)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
 cv2.setNumThreads(1)
 yy,xx=np.mgrid[:80,:90];rgb=np.stack((xx*2,yy*2,xx+yy),-1).astype(np.uint8)
 donor=np.full((22,26,3),[240,20,80],np.uint8);dm=np.zeros((22,26),bool);dm[1:21,3:23]=True
 vis=np.zeros((80,90),bool);vis[20:60,20:60]=True;full=np.zeros_like(vis);full[17:64,17:64]=True
 q=np.array([[30.2,30.3],[52.1,51.9],[3.,3.],[72.,70.],[22.,52.],[52.,22.],[70.,5.],[5.,70.]])
 cut=core.resize_visible_cutout(donor,dm,20);assert cut['resized_hw']==[20,20]
 # Padded square has four equal deepest center candidates; row-major first wins.
 assert cut['donor_anchor_xy']==[9,9]
 paste,canvas,transform=core.paste_support(vis.shape,cut,[30,30]);ok,reason,facts=core.candidate_checks(vis,full,paste,q,0)
 assert ok and facts['anchor_strict5x5_new_E']and facts['visible_control_queries']and facts['outside_control_queries']
 composite=core.compose_rgb(rgb,canvas,paste)
 assert np.array_equal(composite[~paste],rgb[~paste])and np.array_equal(composite[paste],canvas[paste])
 cert=t.construct_certainty(vis,full,paste);labels=t.label_queries(cert,q)
 assert labels[0]['exact_state']==t.E and labels[0]['radius2_strict_new_controlled_E_core']
 assert labels[1]['exact_state']==t.V and labels[2]['exact_state']==t.O
 assert(cert['exact_state'][full&~vis]==t.U).all()
 assert not core.strict_square(np.ones((5,5),bool),0,0)
 clipped,cc,_=core.paste_support(vis.shape,cut,[0,0]);assert clipped[0,0]and not core.strict_square(clipped,0,0)
 invalid=q.copy();invalid[0]=[89.8,1.];_,valid=core.query_pixels(invalid,[80,90]);assert not valid[0]
 assert core.rank_key('target','donor','a')==core.rank_key('target','donor','a')
 result={'PASS':True,'RGB_outside_actual_paste_exact':True,'RGB_inside_transformed_cutout_exact':True,'resize_padded_distance_tie_rowmajor':[9,9],
  'strict25_pixel_anchor_and_V_O_controls':True,'unknown_original_hidden_not_relabelled':True,'clipped_border_not_strict25':True,
  'nearest_rounding_invalid_query_rejected':True,'source_RGB_or_mask_reads':0,'current_data_reads':0,'GPU_model_training_PnP_calls':0,
  'code_sha256':{str(p.relative_to(ROOT)):sha(p)for p in [Path(core.__file__),Path(__file__),ROOT/'contract/controlled_truth.py']}}
 (ROOT/'controlled/COMPOSER_CORE_SMOKE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
