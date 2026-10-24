"""Display root-fixed V99 cases from saved RGB only; no actual rerender."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
H=Path(__file__).resolve().parent;P=H.parents[1]
V99=P/'query_visibility_diverse_expected_cue_source_20261023_v99';O=V99/'render_preparation/actual_corpus'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def b(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x)for x in Path(p).read_text().splitlines()if x.strip()]
def check(x):assert sha(x['path'])==x['sha256'];return Path(x['path'])
def write(p,x):
    with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def main():
    selection=read(H/'FIXED_SELECTION_BEFORE_SAVED_CONTACTS.json');assert selection['base_indices_in_requested_order']==[16,2,3]
    seal=read(O/'RENDER_SEAL.json');assert seal['completed']and seal['PASS_execution']
    manifest=read(check(seal['manifest']));images=rows(check(manifest['full_RGB']));qrows=rows(check(manifest['query_inputs']))
    protocol=read(check(seal['protocol']));jobs=rows(check(protocol['jobs']));gt=rows(check(protocol['GT_only_targets']))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);receipts=[]
    for i in selection['base_indices_in_requested_order']:
        j=jobs[i];target=j['selected_aux_index'];assert target is not None
        with np.load(check(manifest['chunks'][i//8]['file']),allow_pickle=False)as z:
            crops=z['rgb96'][i%8];valid=z['crop_valid96'][i%8];q=z['query8_xy'][i%8];L=z['L'][i%8]
        assert np.array_equal(q,j['query8_xy'])and np.array_equal(L,j['L'])and qrows[i]['query8_xy']==j['query8_xy']
        out=Image.new('RGB',(1440,1030),'#eef2f6');d=ImageDraw.Draw(out)
        d.text((12,8),f'{j["base_id"]} | {j["split"]} | aspect{j["aspect_index"]} | {j["appearance_family"]}',font=font,fill='#14283c')
        records=[]
        for si,stage in enumerate(('BEFORE','COVER','SHAM')):
            r=images[3*i+si];raw=np.array(Image.open(check(r['image'])).convert('RGB'),copy=True)
            assert array_sha(raw)==r['decoded_RGB_sha256']and r['variant']==stage
            state=gt[i]['stages'][stage][target]['state'];all8=''.join(x['state']for x in gt[i]['stages'][stage])
            d.text((480*si+12,39),f'{stage}: AUX{target} {state}; all8={all8}',font=font,fill='#14283c')
            out.paste(Image.fromarray(raw).resize((456,456),Image.Resampling.LANCZOS),(480*si+12,70))
            d.text((480*si+12,540),'Selected unmarked RGB96 | 4x nearest display',font=small,fill='#14283c')
            out.paste(Image.fromarray(crops[si,target].transpose(1,2,0)).resize((384,384),Image.Resampling.NEAREST),(480*si+12,568))
            d.text((480*si+12,963),f'q={np.round(q[target],3).tolist()}  L={L[target]:.3f}',font=small,fill='#14283c')
            records.append(dict(stage=stage,image=r['image'],decoded_RGB_sha256=r['decoded_RGB_sha256'],selected_aux=target,crop_RGB96_sha256=array_sha(crops[si,target]),valid96_sha256=array_sha(valid[si,target]),q=q[target].tolist(),L=float(L[target]),caption_GT_state=state,all8_GT_states=all8))
        d.text((12,1000),'Saved-only source illustration; AUX/native8 UNKNOWN. No rerender, model, features, fitting, current223 or PnP.',font=small,fill='#14283c')
        file=H/f'V99_FIXED_BASE_{i:03d}_SAVED_CONTACT.png';out.save(file)
        receipts.append(dict(base_index=i,base_id=j['base_id'],split=j['split'],aspect_index=j['aspect_index'],appearance_family=j['appearance_family'],image=b(file),records=records))
    write(H/'CONTACTS_AND_PROVENANCE.json',dict(completed=True,selection=b(H/'FIXED_SELECTION_BEFORE_SAVED_CONTACTS.json'),builder=b(__file__),V99_render_seal=b(O/'RENDER_SEAL.json'),manifest=seal['manifest'],contacts=receipts,new_renders_features_models_fit_private_current_PnP=0,native8_mapping='UNKNOWN',human_gold=False,training_admission=False))
    print(json.dumps(b(H/'CONTACTS_AND_PROVENANCE.json')))
if __name__=='__main__':main()
