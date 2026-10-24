"""Generated display-format/selection/immutability fixtures only."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import numpy as np
from PIL import Image
import visual_math as m

R=Path(__file__).resolve().parent
V90=R.parents[1]/'query_visibility_public_cue_inputs_20261023_v90/producer/spatial_input_math.py'


def main(output):
    tests=[]
    def ok(name,condition=True):
        assert condition,name;tests.append(name)
    a=[dict(attempt_index=i,pair_ready=i%5==0)for i in range(448)]
    original=json.dumps(a,sort_keys=True);s=m.select_attempts(a)
    ok('first3_each_canonical_order',s['accepted']==[0,5,10] and s['failed']==[1,2,3] and s['selected_attempt_indices']==[0,1,2,3,5,10])
    ok('selection_no_metadata_mutation',json.dumps(a,sort_keys=True)==original)
    allfail=m.select_attempts([dict(attempt_index=i,pair_ready=False)for i in range(448)])
    ok('missing_accepted_no_replacement',allfail['accepted']==[] and allfail['failed']==[0,1,2] and allfail['missing_groups']==[dict(group='accepted',requested=3,present=0,missing=3)])
    allpass=m.select_attempts([dict(attempt_index=i,pair_ready=True)for i in range(448)])
    ok('missing_failed_no_replacement',allpass['failed']==[] and allpass['accepted']==[0,1,2])
    try:m.select_attempts(a[::-1])
    except ValueError:ok('reordered_attempts_rejected')
    else:raise AssertionError('Reorder accepted')
    def crop_ast(path):
        tree=ast.parse(path.read_text());return ast.dump(next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='crop'),include_attributes=False)
    ok('V90_crop_AST_exact',crop_ast(Path(m.__file__))==crop_ast(V90))
    rgb=np.arange(30*40*3,dtype=np.uint8).reshape(30,40,3);saved=rgb.copy()
    support=np.zeros((30,40),bool);support[10:15,8:13]=True;ss=support.copy()
    crop,valid=m.crop(rgb,[10.25,12.5],24.,96);ok('crop_shape_dtype',crop.shape==(3,96,96)and crop.dtype==np.uint8 and valid.shape==(1,96,96)and valid.dtype==bool)
    out,pv=m.crop(rgb,[0.,0.],24.,96);ok('OOB_black_padding',np.all(out[:,~pv[0]]==0)and not pv.all())
    local_support=m.protected_crop(support,[10.25,12.5],24.)
    im=m.marked(rgb,support,[10.25,12.5],[0,0,24,24]);markedcrop=m.marked(crop.transpose(1,2,0),local_support,[47.5,47.5])
    donor,box=m.donor_display(rgb,support)
    ok('native_alpha_display_extent',box==[8,10,13,15]and donor.size==(200,200))
    raw=Image.fromarray(crop.transpose(1,2,0));ctx=Image.fromarray(rgb)
    card=m.card('GENERATED ONLY','Invented pixels; no real source', [ctx]*3,[im]*3,[raw]*3,[markedcrop]*3,donor,'Canonical native exclusive-alpha display only','Rows: raw full / marked full / raw96 / marked96. AUX only.')
    ok('card_format_RGB_1440x1450',card.mode=='RGB'and card.size==(1440,1450))
    contact=m.contact([card,card],[]);ok('contact_format',contact.size==(1440,795))
    empty=m.contact([],['accepted','failed']);ok('empty_contact_missing_groups_visible',empty.size==(900,180))
    ok('all_render_functions_no_input_pixel_mutation',np.array_equal(rgb,saved)and np.array_equal(support,ss))
    proof=dict(completed=True,PASS=True,test_count=len(tests),tests=tests,
        bindings=dict(helper=dict(path=str(Path(m.__file__).resolve()),sha256=hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()),
                      tests=dict(path=str(Path(__file__).resolve()),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
                      pinned_crop_code=dict(path=str(V90),sha256=hashlib.sha256(V90.read_bytes()).hexdigest())),
        source_payload_reads=0,source_metadata_outcomes_read=0,model_imports_or_calls=0,current_V112_reads=0,fit=0,PnP=0)
    with Path(output).open('x')as f:json.dump(proof,f,indent=2);f.write('\n')
    print(json.dumps(dict(PASS=True,tests=len(tests))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);main(p.parse_args().output)
