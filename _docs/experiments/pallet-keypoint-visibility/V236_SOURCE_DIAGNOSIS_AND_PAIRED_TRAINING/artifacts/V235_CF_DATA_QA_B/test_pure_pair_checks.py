import copy,json
from pathlib import Path
import numpy as np
from pure_pair_checks import *
H=Path(__file__).parent
fail=lambda fn: rejects(fn)
def rejects(fn):
 try:fn()
 except AssertionError:return True
 raise AssertionError('expected rejection')
taps,inside=required_four_taps([100.,100.],96.,[100.,100.]);assert inside and len(taps)==16
integer_taps,_=required_four_taps([100.5,100.5],96.,[100.,100.]);assert (46,46) in integer_taps and (49,49) in integer_taps
outside,inside=required_four_taps([100.,100.],10.,[150.,150.]);assert not inside
base=np.zeros((96,96,3),np.uint8);valid=np.ones((96,96),bool)
a=np.zeros((96,96),bool);a[44:56,44:56]=True
cover=base.copy();yy,xx=np.indices((96,96));layer=np.stack([xx+30,yy+20,xx+yy+10],axis=2).astype(np.uint8);cover[a]=layer[a]
b,saved=shifted_mask_and_pixels(a,cover,(-35,0));sham=base.copy();sham[b]=saved[b]
parent={'parent_row_id':'invented','family':'YCB_AUX','query_xy':[100.,100.],'L':96.,'native_slot':None,'condition32':[0.]*32,'valid_sha256':'invented','original_query_available':True,'internal_source_role':'INTERNAL_SOURCE_FIT','target':0,'object_id':7}
donor={'internal_source_role':'INTERNAL_SOURCE_FIT','original_associated_r_state':'D','object_id':2}
meta={n:{**parent,'target':t,'input_available':True} for n,t in [('BEFORE',0),('COVER',1),('SHAM',0)]}
r=validate_constructed_pair(base,valid,cover,sham,a,b,taps,(-35,0),parent,donor,meta)
corrupt=cover.copy();corrupt[0,0]=255;assert rejects(lambda:validate_constructed_pair(base,valid,corrupt,sham,a,b,taps,(-35,0),parent,donor,meta))
wrong=sham.copy();wrong[b]=255;assert rejects(lambda:validate_constructed_pair(base,valid,cover,wrong,a,b,taps,(-35,0),parent,donor,meta))
assert rejects(lambda:validate_constructed_pair(base,valid,cover,sham,a,b,taps,(-35,0),parent,{**donor,'object_id':7},meta))
changed=copy.deepcopy(meta);changed['COVER']['condition32'][0]=1;assert rejects(lambda:validate_constructed_pair(base,valid,cover,sham,a,b,taps,(-35,0),parent,donor,changed))
failed={'BEFORE':{'target':0,'input_available':True},'COVER':{'constructed':False,'target':None,'input_available':False,'RGB':None},'SHAM':{'constructed':False,'target':None,'input_available':False,'RGB':None}}
assert validate_failed_construction(parent,failed);bad=copy.deepcopy(failed);bad['COVER']['target']=1;assert rejects(lambda:validate_failed_construction(parent,bad))
result={'completed':True,'PASS':True,'invented_only':True,'groups':['source-nearest3x3 to all four raw96 neighbours including integer/OOF','matched alpha/opaque RGB translation and outside-alpha exactness','FIT/D/different-object and immutable input metadata rejection','failed C/S never semantic E or available new input'],'positive_result':r,'actual_source_or_payload_reads':0}
(H/'INVENTED_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
