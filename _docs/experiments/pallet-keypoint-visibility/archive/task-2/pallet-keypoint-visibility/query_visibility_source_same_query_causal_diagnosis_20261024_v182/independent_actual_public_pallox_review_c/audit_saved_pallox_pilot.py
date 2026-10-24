"""Independent saved-public-pilot byte/display audit; no network/model imports."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re
import cv2,numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;P=ROOT.parent
PILOT=ROOT/'public_pallox_rgb_mask_pilot_v1';COMPLETE=PILOT/'COMPLETE.json'
OFFICIAL=P/'query_visibility_public_source_target_definition_audit_20261024_v179/OFFICIAL_DATAVERSE_DATASET_METADATA.json'
USED={}
NOTES={
'0.55m_hall:0000':'Low front view; outer wooden bin silhouette is marked. Dark fork apertures are largely white/red as well, while a narrow plank gap is partly black. This is not verified literal material ownership.',
'0.55m_lab:0000':'Low front view; white mask is mostly a filled bin silhouette, including the dark lower fork apertures and most plank gaps. Small edge irregularities remain. No per-pixel physical label correction is made.',
'1.23m_hall:0000':'Open bin and contents are visible. The central contents region is mostly excluded; the rim/front/base are included. Some fine interior/edge speckles are ambiguous, and fork spaces remain largely filled.',
'1.23m_lab:0000':'Open bin contents are largely excluded by a coarse central hole. Rim/front boards are included; narrow board gaps and inner-wall/content borders are not a certified material mask.',
'1.92m_hall:0000':'High view shows a bin silhouette surrounding an excluded contents region. A loose wooden crosspiece among the contents is not treated simply as all wood. Bottom truncation limits physical/native interpretation.',
'1.92m_lab:0000':'High view; interior contents are mostly excluded, bin rim/front included. Same-material loose contents and wall boundaries remain an instance-definition ambiguity, not a native-end witness.',
'0.55m_hall:0001':'Foreground cardboard on the left is visibly excluded by the mask, while exposed front boards remain white/red. Lower dark fork apertures are still largely filled, so visible occluder exclusion does not prove universal literal ownership.',
'0.55m_hall:0002':'Foreground cardboard is excluded along the visible occlusion boundary. In this changed viewpoint, lower fork openings are mostly black and the center support white, contrasting with filled gaps in other pilot cards. This cautions against a universal material/visibility convention.'}

def require(ok,msg):
    if not bool(ok):raise AssertionError(msg)
def sha(path):
    path=Path(path).resolve();require(path.is_relative_to(P) and '_v112' not in str(path).lower(),'Forbidden path')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def check(b):
    p=Path(b['path']);require(sha(p)==b['sha256'],'SHA '+str(p));USED[str(p.resolve())]=b['sha256'];return p
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s]
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def main():
    complete=read(COMPLETE);USED[str(COMPLETE)]=sha(COMPLETE)
    require(complete['completed'] and complete['PASS'] and complete['attempts']==complete['intake_PASS']==8 and complete['intake_failed']==0,'Actual all8 intake')
    fixed=lines(check(complete['fixed8']));attempts=lines(check(complete['all8_attempts']));census=lines(check(complete['all322_source_census']))
    require(len(census)==322 and sum(r['mask_missing'] for r in census)==32 and all(r['split']=='SOURCE_TRAIN_CANDIDATE' and not r['training_admission'] for r in census),'All322/missing32 candidate census')
    official=read(OFFICIAL)['data']['latestVersion']['files'];USED[str(OFFICIAL)]=sha(OFFICIAL)
    registry={f['dataFile']['id']:f for f in official};proof=[];cards=[];payload_count=0
    require([r['source_id'] for r in fixed]==[r['source_id'] for r in attempts]==list(NOTES),'Frozen pilot order')
    for f,r in zip(fixed,attempts):
        require(r['intake_PASS'] and not r['training_admission'] and r['new_owner_native_or_quality_labels'] is False,'No admission/newlabels')
        decoded={};file_proofs={}
        for mode in ('rgb','masks'):
            expected=f['files'][mode];record=r['files'][mode];off=registry[expected['official_file_id']]['dataFile']
            require(off['filesize']==expected['declared_bytes'] and off['checksum']==expected['declared_checksum'],'Official size/checksum registration')
            require(record['original_file_record']==expected,'Intake original file binding')
            p=check(record['saved']);raw=p.read_bytes();md5=hashlib.md5(raw).hexdigest()
            require(len(raw)==expected['declared_bytes']==record['bytes'] and md5==expected['declared_checksum']['value']==record['MD5'],'Original payload size/MD5 exact')
            a=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_UNCHANGED);require(a is not None,'Decode saved original')
            if mode=='rgb':
                require(a.dtype==np.uint8 and a.shape==(480,640,3),'Original declared RGB schema')
                a=a[:,:,::-1].copy()  # Independent BGR->RGB channel permutation.
                require(np.array_equal(a,np.asarray(Image.open(p).convert('RGB'))),'PIL independent PNG RGB agrees with cv2 BGR permutation')
            decoded[mode]=a;file_proofs[mode]=dict(file_id=expected['official_file_id'],path=str(p),bytes=len(raw),MD5=md5,SHA256=sha(p),dtype=str(a.dtype),shape=list(a.shape));payload_count+=1
        rgb=decoded['rgb'];mask=decoded['masks']
        canonical=np.load(check(r['canonical_RGB']),allow_pickle=False);saved_mask=np.load(check(r['mask']),allow_pickle=False)
        require(canonical.dtype==np.uint8 and np.array_equal(canonical,rgb),'Canonical RGB NPY exact original decoded RGB')
        require(mask.dtype==saved_mask.dtype==np.uint8 and mask.shape==saved_mask.shape==(480,640) and np.array_equal(mask,saved_mask),'Original mask dtype/values exact NPY')
        require(np.array_equal(np.unique(mask),np.array([0,255],np.uint8)),'Mask original 0/255')
        require(ash(rgb)==r['canonical_RGB']['decoded_RGB_sha256'] and ash(mask)==r['mask']['decoded_mask_sha256'],'Decoded pixel hashes')
        clean=np.asarray(Image.open(check(r['clean_RGB_PNG'])).convert('RGB'));require(np.array_equal(clean,rgb),'Clean PNG decoded exact original RGB')
        card_path=check(r['card']);card=np.asarray(Image.open(card_path).convert('RGB'));require(card.shape==(700,2000,3),'Card geometry')
        require(np.array_equal(card[132:612,20:660],rgb),'Clean card RGB panel exact source pixels1x')
        require(np.array_equal(card[132:612,680:1320],np.repeat(mask[:,:,None],3,axis=2)),'Mask card panel exact original mask1x')
        overlay=rgb.copy();on=mask==255;overlay[on]=np.rint(.65*rgb[on].astype(np.float64)+.35*np.array([239,62,68])).astype(np.uint8)
        require(np.array_equal(card[132:612,1340:1980],overlay),'Display overlay arithmetic exact; clean source panel separate')
        require(read(card_path.parent/'INTAKE_PROOF.json')==r,'Per-case proof and all8 ledger exact')
        cards.append(card)
        proof.append(dict(source_id=r['source_id'],original_files=file_proofs,mask_white_pixels=int(np.count_nonzero(on)),mask_black_pixels=int(np.count_nonzero(~on)),same_size_binary_mask=True,
            canonical_RGB_byte_exact=True,original_mask_byte_exact=True,clean_card_panels_byte_exact=True,display_overlay_copy_exact=True,
            actual_view=dict(tool='tools.view_image',detail='original',image=bind(card_path),actual_card_dimensions=[2000,700],calls=1),visual_observation=NOTES[r['source_id']],
            observation_authority='Qualitative card review only; no per-pixel owner/native annotations or masks corrected',training_admission=False))
    require(payload_count==16,'All16 original payloads audited')
    contact=np.asarray(Image.open(check(complete['contact'])).convert('RGB'));require(contact.shape==(1400,2000,3),'2x4 contact dimensions')
    for i,card in enumerate(cards):
        expected=np.asarray(Image.fromarray(card).resize((1000,350),Image.Resampling.NEAREST));x=(i%2)*1000;y=(i//2)*350
        require(np.array_equal(contact[y:y+350,x:x+1000],expected),'All8 contact tile assembly')
    page=check(complete['index']).read_text();sources=re.findall(r'<img[^>]+src="([^"]+)"',page)
    require(len(sources)==8 and all((PILOT/rel).is_file() for rel in sources),'Offline HTML eight image links')
    for path,h in USED.items():require(sha(path)==h,'Audited files changed')
    report=dict(completed=True,PASS=True,UTC=datetime.now(timezone.utc).isoformat(),code=bind(Path(__file__)),pilot_complete=bind(COMPLETE),
      checks=[dict(name=n,PASS=True) for n in ['Official16file IDs/bytes/MD5/SHA beforeafter unchanged','Canonical cv2 BGR->RGB and independent PIL RGB agree exactly','Original8 mask uint8/480x640/0,255 arrays unchanged','Clean source/card RGB and mask panels exact original pixels','Display overlays are separate copies with declared alpha arithmetic','All8 actual fullcards viewed once each via tool; no claim allpixels annotated','All8/2x4 contact assembly and offlineHTMLlinks','All322candidate/missing32/selection preserved; trainingadmissionfalse']],
      originals_verified=16,pairs8=True,cards_actually_viewed=8,actual_view_tool_calls=8,full_card_view_scope='All eight frozen2000x700 cards, containing640x480 source RGB/mask/overlay at1x. This is visual inspection, not exhaustive manual per-pixel QA or 322-frame review.',
      case_proofs=proof,contact_assembly_verified_not_additional_tool_view=True,browser_render_verified=False,
      semantic_findings=['The two left-foreground-cardboard cases visibly exclude cardboard while retaining exposed bin wood, supporting visible-region-like behavior at those boundaries only.',
        'Several low-view cards fill dark fork gaps with white/red, while another view leaves these apertures mostly black. The author mask is not certified literal material ownership.',
        'Higher views largely exclude interior contents but same-material loose wood, bin-wall edges and coarse holes are an instance-definition ambiguity. Metadata/this pilot do not resolve universal visible-versus-amodal or human-QA authority.',
        'The target is a whole Pallox bin; no flat-pallet/native8/end-part/self/coordinate gold or foreign-versus-background truth is certified.'],
      next_source_readiness=dict(status='Ready for definition and original-query input audit only, not training admission',recommendation='Prospectively define whether target means bin instance envelope or visible bin material, document fork-hole/interior-content treatment, then separately authorize the unchanged original fullRGB initializer q8/L and inspect its exact raw96/valid crops against these author masks. Preserve U/OOF/noDetection and unresolved pixels. Only after this source-only authority check consider separately frozen opaque-donor/adjacent-sham tests requiring BEFORE target evidence at the SAME q/crop; no GTcrop centers or GT-bestnative. No generation/production queries/weak labels are created here.',basis='This fixed public8 pilot, official README and unchanged original goal only; no model/VAL/private/held outcomes.'),
      no_new_download_initializer_SAM_head_weight_label_or_fit=True,production_queries_generated=0,training_admission=False,selected_model=None,fully_solved=False)
    (HERE/'REVIEW.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(completed=True,PASS=True,originals16=True,canonical_pairs8=True,actual_card_views8=True,review=bind(HERE/'REVIEW.json'))))

if __name__=='__main__':main()
