from pathlib import Path
from collections import Counter
import json,hashlib,datetime
import numpy as np
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent
P=O.parents[1]
MR=P/'query_visibility_mr6d_full_public_source_20261024_v153'
def read(p):return json.loads(Path(p).read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def put(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def chk(b):assert bind(b['path'])==b;return Path(b['path'])
def ah(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
ids=read(O/'ID_SEAL_BEFORE_RGB_READS.json');meta=read(O/'PERFRAME_SAVED_COUNTS_BEFORE_VISUAL_NOTES.json');recs=read(MR/'original_query_evidence_full230/ALL230_SAVED_CROP_EVIDENCE.json')['records'];facts=[json.loads(s) for s in (MR/'full230_native_DINO_v1/FEATURE_INPUT_FACTS1840_INPUT_ONLY.jsonl').read_text().splitlines()]
obs={
0:'Low viewpoint of two similar stacked empty pallets. Top deck and EPAL side blocks recur across crops; rear bench/KLT bins remain in context.',
1:'Same stacked empty pallets at a slightly higher view. n4..n6 contain topwood plus bench/bin background; bottom-left query crops have padding.',
2:'Stacked pallet view; n1/n2 side-block crops include dark floor and rear bench. One bottom query is unavailable, but saved crop retained.',
3:'Upper pallet deck dominates n4..n7 while n0/n1/n2 mix sidewood and background; lower pallet is a similar visual distractor.',
4:'Both stacked pallets remain visible. Rear wooden edge and bench overlap in local crops; no loaded occluder on top deck is apparent at board scale.',
5:'Camera reveals more of lower pallet and gap between pallets. Local sidewood crops can belong visually to either repeated stacked structure.',
6:'Tilted view with two original query centers outside image. Remaining crops show exposed slatted top deck, bench legs, and dark floor.',
7:'Stacked pallets under oblique view. Bottom query crops include lower-pallet EPAL blocks; top query crops include upper deck and blue bin background.',
8:'Stacked pallets now centrally visible. Some nominal bottom role queries lie on lower-pallet-looking sidewood; original U weak roles remain U.',
9:'Both pallets are clear at board scale. n0/n1/n2/n3 all show similar sidewood or floor while top query crops show slats and bench/KLT context.',
10:'Same empty stacked structure. Lower-pallet side blocks and upper-pallet topwood recur in separate local crops; one unavailable query retained.',
11:'Oblique stacked pallets. n2 crop is mainly dark floor despite nearby sidewood in context; weak role U remains U and is not assigned D/E.',
73:'Original q8 form a small box on left background chair/bench structure, away from the loaded pallet. All roles U; known negative frame entity is preserved.',
74:'Loaded pallet at image bottom: woven basket, pale slatted wooden crate, and tall white slab. E-role n4 crop shows basket/white slab; n5 shows load wood/background.',
75:'Same loaded scene with bottom four centers outside image. Local n0..n3 are mostly padding; n4..n7 show basket/crate and deck together.',
76:'Loaded pallet is visible in scene but original no-detection packet has all black unconstructed crops. All eight query rows and U availability retained.',
77:'Loaded pallet low view. n0/n1 show exposed front corners, E-role n5 shows pale crate slats and dark background; basket dominates left rear local crops.',
78:'Front EPAL blocks and top deck are visible below basket/crate. Rear local crops mix loaded crate wood with pallet deck wood and background.',
79:'Whole front face of loaded pallet is visible. n0..n3 show side blocks, n4/n7 woven load, n5/n6 pale slatted crate; all8 input available.',
80:'Similar loaded pallet, slightly tilted. Lower-left query outside input; right-side crop has heavy padding while rear crops are load-heavy.',
81:'Loaded pallet mostly below image lower edge. Four bottom query centers outside image, retaining small visible wood strips and black padding.',
82:'Loaded scene remains visible but original no-detection rows yield all black crops. These U rows are availability evidence, not external-occlusion positives.',
83:'Tilted loaded scene with one available front-corner query. Left E-role crop includes basket/white panel, right E-role crop slatted crate/backdrop.',
84:'Loaded pallet remains apparent at board scale while frame-shared auxiliary entity predicts rejection in C37/C38. Crop n0 is floor/white-slab/deck mix; uncertainty retained.',
85:'Camera tilt reveals deck gaps and front EPAL blocks. Basket/white slab and pale crate dominate rear crops; original query geometry unchanged.',
86:'Higher oblique loaded view; front-side crops include floor and pallet blocks. E-role local regions contain load wood and basket rather than isolated pallet corner.',
87:'Oblique scene with basket and crate openings visible. Rear n5/n6 contain slatted crate wall and rear bench, while front n0/n1 contain sidewood.',
88:'Top surfaces of both loads visible; bottom query n3 outside image. Repeated pale crate boards resemble pallet slats locally.',
89:'Higher view exposes deck between loads. One E role remains at pale crate; basket/white slab role is U, preserved without visual relabel.',
90:'Loaded pallet at high oblique view, one side partly out of image. n5/n6 contain crate wall/background; shared entity rejects frame despite known weak positive.',
91:'Deck gaps and basket/crate openings visible. n4 D crop shows white slab/edge context, n5 E crop shows crate/backdrop; n7 S kept separate from E.',
92:'Top-down loaded view with one unavailable bottom query. n4/n7 basket/white slab and n5/n6 crate wood recur; shared entity rejection continues.',
93:'Three bottom query centers unavailable. Available n0 has floor/pallet edge with padding; rear local crops contain load walls and background.',
94:'Loaded pallet fills lower scene; bottom query crops are heavily padded. Rear slat/basket crops remain visually similar to neighboring loaded views.',
95:'Higher view of woven basket and pale crate. Available n0 contains floor/deck, n4 white slab/basket, n5 pale load wood; no visual label reassignment.',
96:'Original q context is narrow relative to pallet extent. Four bottom centers unavailable; n4 crop mainly white shelf/background and basket edge, though weak role is D.',
97:'Four bottom saved crops entirely black from unavailable centers. Available top crops show white shelf, basket, and pale wooden crate.',
98:'Loaded view with four unavailable bottom centers. n7 S crop shows basket wall/rim; its predictions are recorded but it is never treated as E truth.',
99:'Loaded pallet from above. Only one bottom center available; rear crop n4 includes white panel/basket edge while crate slats dominate n5/n6.',
100:'Loaded pallet with visible front left corner and two unavailable right-side centers. n4 weak E crop now shows basket rim/white panel/backdrop.',
101:'Similar high view; local n4 contains basket rim and white panel, n5/n6 pale crate and floor cables. Shared entity continues rejection.',
102:'Pallet deck remains visible around basket/crate. Two side-center crops nearly black; top crops mix load walls, white shelf, and floor cables.',
103:'Loaded pallet viewed from higher left. Rear local crop appearance is basket/white shelf or slatted crate/cables while exposed front corner remains visible.',
104:'Scene includes the same loads and rear metal bin. n4 shows basket/white panel, n5/n6 include bin/crate/cables; known role E not relabeled from local appearance.',
105:'Deck gaps and front corner clear at board scale. One bottom crop completely black; right rear crops increasingly contain metal bin and floor cables.',
106:'Loaded pallet and metal bin near right rear edge. n4 basket crop, n5 bin/cables, n6 crate/bin; shared entity rejects known positive frame.',
107:'Higher view shows pallet deck around loads. n5/n6 include metal bin and cable background; original D role n5 remains D despite mixed local content.',
108:'More floor/bench context visible, pallet remains in lower scene. Query n2 unavailable; n5/n6 mostly bin/cables plus pallet edge.',
109:'Oblique loaded pallet with a small hand/body edge at lower left. C37/C38 entity recovers acceptance; n4 still basket/white panel and n5/n6 bin/cables.',
110:'Loaded scene at greater distance. Front side queries include pallet corner and floor; rear right crops mostly bin/cable and a thin deck edge.',
111:'Pallet smaller in context with basket/crate/metal bin. One side-center crop fully black; rear crops show mixed load and background.',
112:'Pallet and loads still visible; original role labels all U although frame entity is weak positive. Crops mix basket/crate/bin and deck; no roles inferred from scene.',
113:'Motion-softened oblique loaded scene. n1/n2 now crop crate/bin regions while n4/n7 show deck/floor; original native order remains fixed.',
114:'Whole loaded pallet remains visible but original no-detection packet has all black crops and U availability. No inferred query or replacement crop is added.',
115:'Higher view with pale crate closest at right. n1/n2 show crate wall and deck, n5/n6 crate/basket; shared entity rejects weak positive frame.',
116:'Loaded scene with metal bin near front-right corner. D-role n1 crop contains bin base and deck; C35/C36/C37 predict E while C38 predicts D.',
117:'High view with basket, crate and bin. All roles U; n4/n7 show basket wall, n5/n6 deck/bin edge; shared auxiliary rejects weak positive frame.',
44:'Prescribed OOF categories: bare pallet under rotated viewpoint, n4/n7 crops still show wood/background even while original role state is OOF.',
59:'Prescribed U category: original pallet mostly below scene edge, n6 crop a wood corner and dark floor; all D/E scores stay descriptive for U.',
60:'Original q8 tracks background bench/KLT structure while pallet is at bottom edge. Selected U row n2 crop is blue-gray KLT; no weak role converted to E.',
125:'Different TRAIN loaded scene: blue plastic crate and gray bin on pallet with wood behind. Selected weak E n5 exact crop shows gray bin wall.',
130:'TRAIN loaded scene with blue crate, gray bin and rear wood. Selected weak D n4 exact crop mostly blue crate plus background; original weak D preserved.',
151:'TRAIN loaded scene under oblique view. Selected OOF n4/n7 exact crops show blue crate texture; OOF remains unavailable truth for D/E scoring.'}
assert set(obs)==set(r['frame_index'] for r in meta)
verify=[]
for n in meta:
 f=n['frame_index'];r=recs[f];board=Image.open(chk(r['PNG'])).convert('RGB')
 with np.load(chk(r['saved_crop_packet']),allow_pickle=False) as z:rgb=z['rgb96'].copy();valid=z['valid96'].copy();gidx=z['global_index'].copy()
 assert np.array_equal(gidx,np.arange(f*8,f*8+8))
 for k in range(8):
  g=f*8+k;expected=Image.fromarray(rgb[k]).resize((144,144),Image.Resampling.NEAREST);x=10+(k%4)*198;y=443+(k//4)*180
  assert np.array_equal(np.asarray(board.crop((x,y,x+144,y+144))),np.asarray(expected))
  assert ah(rgb[k])==facts[g]['RGB96_HWC_sha256'] and ah(valid[k])==facts[g]['valid96_sha256']
 n['visual_observation']=obs[f]
 n['visual_review_scope']='full original board + eight saved local crops' if n['selection']!='FIRST_CATEGORY_EXTRA' else 'full original board added in EXTRA_CATEGORY_FRAMES contact sheet; deterministic category query crop also enlarged'
 n['interpretation_limit']='Descriptive public-source context and exact saved local96 only. Assistant weak labels unchanged; cm-uncertain source GT pose/native is not physical or pixel gold. No evidence here certifies exact point visibility.'
 verify.append(dict(frame_index=f,source_board=r['PNG'],crop_packet=r['saved_crop_packet'],all8_saved_RGB96_valid_hashes_and_display_pixels_match=True))
# Extra complete boards selected by first-category rule, not visual interest.
outs=[];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
extra=ids['category_extra_frame_indices']
for start in range(0,len(extra),4):
 chunk=extra[start:start+4];canvas=Image.new('RGB',(1640,1800),'#f0f3f7');d=ImageDraw.Draw(canvas)
 d.text((20,10),'PUBLIC SOURCE: extra frames from prespecified first-category rows',font=font,fill='#16334c')
 for j,f in enumerate(chunk):
  x=20+(j%2)*810;y=60+(j//2)*865
  d.text((x,y),f'#{f:03d} '+recs[f]['source_id'],font=font,fill='#16334c');canvas.paste(Image.open(chk(recs[f]['PNG'])).convert('RGB'),(x,y+30))
 path=O/'contact_sheets'/f'EXTRA_CATEGORY_FRAMES_{start//4+1:02d}.png';canvas.save(path);outs.append(dict(kind='EXTRA_CATEGORY_FRAMES',frame_indices=chunk,PNG=bind(path)))
put(O/'PERFRAME_VISUAL_NOTES_AND_SAVED_COUNTS.json',dict(completed=True,source_only=True,reviewed_frames=len(meta),held45=45,first12TRAIN=12,category_extra_frames=6,perframe=meta))
entityreject=[n['frame_index'] for n in meta if n['selection']=='ALL45_HELD' and n['frame_entity_target']==1 and n['saved_counts']['C37']['entity_prediction_when_frame_shared']==0]
held=[n for n in meta if n['selection']=='ALL45_HELD'];train=[n for n in meta if n['selection']=='FIRST12_TRAIN']
aggregate={group:{a:{key:sum(n['saved_counts'][a][key] for n in rows) for key in ('predicted_D','predicted_E','input_U','known_D_as_E','known_E_as_D','known_DE_input_U')} for a in ('C35','C36','C37','C38')} for group,rows in [('ALL45_HELD',held),('FIRST12TRAIN',train)]}
findings=dict(completed=True,source_only=True,selection_seal=bind(O/'ID_SEAL_BEFORE_RGB_READS.json'),scope='All45 held + first12TRAIN boards, extra6 prescribed first-category frames and32 unique prescribed query crops. All images are existing public MR input evidence. No V112/private/current bodies, models, new fits/forward or coordinate correction.',inventory=dict(existing_public_MR_original_query_boards=230,existing_board_size_WH=[800,820],existing_local_crop_size_WH=[96,96],existing_board_display_local_WH=[144,144],existing_display_nearest_enlargement=1.5,reviewed_full_boards=63,all_local_saved_crop_cells_verified=504,visual_contact_PNGs=21),exact_input_status='Local RGB96 matches producer-saved pre-preprocessing model input bytes, and each displayed144x144 local cell matches nearest x1.5. These are not all17 final encoder tensors: meanfill/full and central preprocessing occurs later. Top context panel is original RGB with q/minmax overlay, not exact224x224 target input; producer stores only target hash. No target224 recreation attempted.',patterns=[dict(pattern='Repeated similar pallet structures in initial TRAIN context',evidence_frames=list(range(12)),observation='First12TRAIN show two similar empty stacked pallets and bench/KLT background, whereas held sequence usually shows one loaded pallet. This selected subset describes a context difference; it does not estimate all TRAIN prevalence.'),dict(pattern='Loaded wood versus pallet wood ambiguity',evidence_global_rows=[596,621,668,1005,1044],observation='Weak E examples show woven basket/white panel, pale slatted crate or gray bin in exact saved local crops; selected D example1044 shows blue crate. Local wood or load texture alone does not establish named pallet native role. Exact role-to-query displacement and physical point truth remain uncertain.'),dict(pattern='Context and boundary availability',evidence_frames=[74,75,81,96,97,98,113,116],observation='Large crop padding, unavailable original centers, load walls, floor cables and background coexist in one board. Input U/OOF is separate from semantic external E.'),dict(pattern='Frame-wide auxiliary rejection persists on many apparent loaded-pallet contexts',C37_C38_known_positive_rejected_frame_indices=entityreject,count=len(entityreject),observation='Shared auxiliary entity decisions reject84,90..108,115..117 despite existing weak positive frame labels. Pallet structure is still visible at contact-board scale in these frames. This is descriptive evidence of broad frame rejection, not a new association label or proof of why the head rejects.'),dict(pattern='Original labels deliberately not changed',evidence_frames=[73,76,82,112,114,117],observation='Background-associated negative frame73, noquery76/82/114, and all-role-U frames112/117 remain as saved. S/OOF/U predictions are displayed but never counted as E truth.')],saved_primary_aggregates=aggregate,caveats=['All source assistant labels remain weak; no human or exact real point gold.','Provided source GT pose/native has cm-scale uncertainty; it is not physical/native/pixel gold.','Visual subset is sealed and descriptive; no score-magnitude selection or replacement, architecture/threshold/checkpoint selection, or causal performance inference.','No held45/fixednative35 relabel or reassignment.'],model_calls=0,held_label_changes=0)
put(O/'SOURCE_CROP_REVIEW_FINDINGS.json',findings)
manifest=read(O/'CONTACT_SHEET_MANIFEST.json');manifest['outputs']+=outs;put(O/'CONTACT_SHEET_MANIFEST.json',manifest)
put(O/'SOURCE_CROP_PIXEL_PARITY_QA.json',dict(completed=True,PASS=True,frames=len(verify),rows=504,all_selected_existing_board_cells_bitexact_nearest_saved96=True,perframe=verify,code=bind(__file__),review_renderer_code=bind(O/'review_saved_public_crops.py'),idseal=bind(O/'ID_SEAL_BEFORE_RGB_READS.json'),bindings=bind(O/'SOURCE_BINDINGS.json'),notes=bind(O/'PERFRAME_VISUAL_NOTES_AND_SAVED_COUNTS.json'),findings=bind(O/'SOURCE_CROP_REVIEW_FINDINGS.json'),contact_manifest=bind(O/'CONTACT_SHEET_MANIFEST.json'),no_relabel_or_coordinate_correction=True))
print(json.dumps(dict(completed=True,PASS=True,frames=63,localcells=504,entity_rejected_positive_C37_C38=entityreject,aggregate=aggregate,extraPNGs=[x['PNG']['path'] for x in outs])))
