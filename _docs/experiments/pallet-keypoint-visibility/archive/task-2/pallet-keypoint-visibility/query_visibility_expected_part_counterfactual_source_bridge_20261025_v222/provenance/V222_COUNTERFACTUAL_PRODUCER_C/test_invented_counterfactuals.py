"""원 SOURCE PNG를 열지 않는 CPU pixel/geometry fixture."""
import json,numpy as np
from pathlib import Path
import build_expected_part_counterfactuals as m
def run():
 before=np.arange(96*96*3,dtype=np.uint8).reshape(96,96,3);valid=np.ones((96,96),bool);target=np.zeros_like(valid);target[38:48,38:48]=True;target[42:55,44:54]=True
 z=m.make_pair(before,valid,target,'invented_parent','SOURCE_FIT');assert z['constructed']
 p=z['proof'];assert p['COVER_covers_full_exact_union']and p['SHAM_disjoint_exact_union']and p['same_actual_RGB_patch']and p['equal_alpha_mass']and p['SHAM_target_RGB_byte_unchanged']and p['native_point_gold_certified']is False
 # OR와 enclosing rectangle의 차이를 보존: OR는valid여도 rectangle invalid이면실패.
 bad=valid.copy();t=np.zeros_like(valid);t[10:13,10:13]=True;t[17:20,17:20]=True;bad[15,15]=False
 assert m.make_pair(before,bad,t,'invented_parent','SOURCE_FIT')['status']=='COVER_RECTANGLE_NOT_ALL_STRICT_VALID'
 assert m.make_pair(before,valid,valid.copy(),'full_parent','SOURCE_FIT')['status']=='NO_ALLVALID_TARGET_DISJOINT_SHAM'
 a,meta=m.patch_for('parent','SOURCE_FIT',11,7);b,mm=m.patch_for('parent','SOURCE_FIT',11,7);c,cm=m.patch_for('parent','SOURCE_DEV',11,7)
 assert np.array_equal(a,b)and meta==mm and meta['seed']!=cm['seed']and not meta['pattern_chosen_using_source_RGB']
 for key in('COVER','SHAM'):assert np.array_equal(m.decode_RGB(m.png_bytes(z[key])),z[key])
 return dict(completed=True,PASS=True,invented_only=True,groups=['actual opaque rectangle/translated sameRGB and alpha/fullunion/outside/SHAM-preserve proof','exactOR versus invalid enclosingrect failure','no legal SHAM retained failure','deterministic source-role-separated lowfrequency pattern','unmarked trainingPNG encoded/decoded equality'],actual_source_PNG_or_model_or_array_or_evaluation_access=0)
if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False))
