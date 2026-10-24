"""Independent saved V197 text/PNG audit; never source packets, arrays or predictions."""
from pathlib import Path
from collections import Counter
import argparse,ast,csv,hashlib,io,json,math,os,re,stat
from PIL import Image
H=Path(__file__).absolute().parent;V=H.parent;G=V/'fixed_train_target_contract_viewer_v1'
SALT='V197_TRAIN_CONTRACT_FIXED12_V1|';VARS=('BEFORE','COVER','SHAM')
PROTO=H/'PROTOCOL_BEFORE_GENERATED_TRAIN_VIEWER_READS.json'
def need(v,s):
 if not bool(v):raise AssertionError(s)
def safe(p):
 p=Path(p);need(p.is_absolute()and '..'not in p.parts and all('v112'not in x.casefold()for x in p.parts),'Lexical guard before access');need(H in p.parents or G in p.parents,'Only generated viewer or own audit files');return p
def opened(p):
 p=safe(p);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  for n in p.parts[1:-1]:
   nxt=os.open(n,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
  end=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
 finally:os.close(fd)
 need(stat.S_ISREG(os.fstat(end).st_mode),'Ordinary generated file');return os.fdopen(end,'rb')
def raw(p):
 with opened(p)as f:return f.read()
def sha(p):return hashlib.sha256(raw(p)).hexdigest()
def bind(p):return dict(path=str(p),sha256=sha(p))
def checked(b):need(bind(b['path'])=={k:b[k]for k in('path','sha256')},'Binding drift');return Path(b['path'])
def jd(p):return json.loads(raw(p))
def bound(b):return jd(checked(b))
def write(p,d):
 safe(p)
 with Path(p).open('x')as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def key(t):return(t['domain'],t['attempt_id'],int(t['selected_native_slot']))
def ranking(t):return(hashlib.sha256((SALT+t['attempt_id']).encode()).hexdigest(),t['attempt_id'],t['domain'],int(t['selected_native_slot']))
def fixed_selection(allmeta):
 ordered=sorted(allmeta,key=ranking);chosen={key(t)for t in ordered[:12]};per={}
 for n in range(8):
  first=next((t for t in ordered if int(t['selected_native_slot'])==n and t['selected_query_available']is True),None)
  per[str(n)]=None if first is None else dict(domain=first['domain'],attempt_id=first['attempt_id'],selected_native_slot=n,rank_sha256=ranking(first)[0])
  if first:chosen.add(key(first))
 return[t for t in ordered if key(t)in chosen],per

def png(b,mode=None,size=None):
 path=checked(b)
 with opened(path)as f:
  with Image.open(f)as im:
   need(im.format=='PNG','Only generated PNG');im.load()
   if mode is not None:need(im.mode==mode,'Original PNG channel mode')
   if size is not None:need(im.size==size,'Original PNG size')
   return im.copy()
def decoded_valid(im):
 b=im.tobytes();need(set(b)<={0,255},'Binary valid PNG');return bytes(v//255 for v in b)
def textvalue(v):return ''if v is None else str(v)
def invented():
 rr=[dict(domain='SYNTH_FINITE_END',attempt_id='TOY%02d'%i,selected_native_slot=i%6,selected_query_available=i!=0)for i in range(18)]
 selected,per=fixed_selection(rr);again,_=fixed_selection(list(reversed(rr)));tests={
 'independent_literal_salt_rank_first12_in_union':{key(t)for t in sorted(rr,key=ranking)[:12]}<={key(t)for t in selected},
 'order_invariant_selection':list(map(key,selected))==list(map(key,again)),
 'max20_and_missing_native_no_replacement':len(selected)<=20 and per['6']is None and per['7']is None,
 'CSV_None_bool_scalar_encoding':list(map(textvalue,[None,False,True,3]))==['','False','True','3']}
 im=Image.new('L',(96,96),255);tests['validPNG_bytes_to_original_bool_bytes']=decoded_valid(im)==b'\x01'*(96*96)
 im=Image.new('RGB',(96,96),(7,23,155));b=io.BytesIO();im.save(b,format='PNG');b.seek(0)
 with Image.open(b)as decoded:tests['authored_RGB_PNG_roundtrip']=decoded.tobytes()==im.tobytes()
 need(all(tests.values()),'Invented checks');return dict(completed=True,PASS=True,invented_only=True,tests=tests,actual_generated_or_original_source_reads=0)
def run(entry):
 e=jd(entry);need(e['approved_generated_saved_TRAIN_viewer_review']is True and e['root_terminal_PASS_recorded']is True and e['original_source_NPZ_or_predictions_or_models_allowed']is False,'Root terminal authorization')
 p=bound(e['protocol']);need(e['code']==p['code']==bind(__file__)and e['protocol']==bind(PROTO),'Frozen own code/protocol')
 c=bound(e['viewer_complete']);need(c['completed']and c['PASS']and c['code']==p['producer_code']and c['protocol']==p['producer_protocol'],'Exact completed V2 viewer')
 need(c['score_bodies_or_model_forward_fit_SAM_initializer_recrop_PnP']==0 and c['source_HOLDOUT_pixel_or_score_access']==0 and c['training_admission']is False,'No model/eval/admission')
 checks=[]
 def ok(n,v):need(v,n);checks.append(dict(check=n,PASS=True))
 selection=bound(c['selection']);allmeta=bound(c['associations'])['triplets109'];ok('All109 TRAIN triplet metadata before selectedpixels',len(allmeta)==109 and c['all109_TRAIN_known_triplets_metadata_first']is True and c['all327_TRAIN_known_labels_preserved']is True)
 ok('Exact109 unique source triplets/ranks',len({key(t)for t in allmeta})==109 and all(t['rank_sha256']==ranking(t)[0]and len(t['cache_indices24'])==len(set(t['cache_indices24']))==24 for t in allmeta))
 selected,per=fixed_selection(allmeta);expected=[dict(domain=t['domain'],attempt_id=t['attempt_id'],native=int(t['selected_native_slot']),rank_sha256=ranking(t)[0])for t in selected]
 ok('Independent fixedhash12/nativefirstavailable union equals frozen selection',selection['selected']==expected and selection['per_native_available_first']==per and len(selected)<=20)
 ok('Selection closure and body-before counters',selection['all109_metadata']==c['associations']and selection['original_q_L_native_all8_unchanged']is True and selection['prediction_or_pixel_or_target_header_reads_so_far']==0)
 rule=selection['rule'];ok('Exact prospective fixedsalt/no score choice',rule==dict(salt=SALT,first12=True,add_first_hash_available_per_native=list(range(8)),max_union20=True,no_missing_native_replacement=True))
 expectedpackets={b['path']:b for t in selected for b in t['source_packet_bindings']};ok('Exact selected packet metadata whitelist; no packet opened',expectedpackets=={b['path']:b for b in selection['source_packet_whitelist']})
 mapping=bound(c['mapping'])if str(c['mapping']['path']).endswith('.json')else list(csv.DictReader(raw(checked(c['mapping'])).decode().splitlines()))
 savedjson=jd(G/'SELECTED_TRAIN_ALL8_TARGET_MAPPING.json')['rows'];ok('CSV/JSON full mapping allfields exact',len(mapping)==len(savedjson)and all({k:textvalue(v)for k,v in j.items()}==r for r,j in zip(mapping,savedjson)))
 rows=savedjson;N=len(selected);ok('Full selected all3/all8 rows unique indices/IDs preserved',len(rows)==c['selected_all8_rows']==N*24 and c['selected_triplets']==N and len({r['cache_index']for r in rows})==len(rows)and len({r['row_id']for r in rows})==len(rows))
 manifest=bound(c['manifest']);ims={r['cache_index']:r for r in manifest['images']};ok('Image manifest oneperselectedrow',len(ims)==len(rows))
 groups={};rawimages={};validimages={};unknown=0;inputU=0
 for t in selected:
  rr=[r for r in rows if(r['domain'],r['attempt_id'])==(t['domain'],t['attempt_id'])];groups[(t['domain'],t['attempt_id'])]=rr;slot=int(t['selected_native_slot']);assoc=t['association']
  ok('All24 rows/no promotion '+t['attempt_id'],len(rr)==24 and {(r['variant'],r['native_slot'])for r in rr}=={(v,n)for v in VARS for n in range(8)}and{r['cache_index']for r in rr}==set(t['cache_indices24'])and all(r['role']=='TRAIN'and r['training_admission']is False for r in rr))
  need(assoc['association_only_not_nativephysicalgold']is True and assoc['training_admission']is False,'Weak association only')
  for n in range(8):
   nr=[next(r for r in rr if r['variant']==v and r['native_slot']==n)for v in VARS]
   need([r['direct_weak_target']for r in nr]==([0,1,0]if n==slot else[-1]*3),'Selectedweak DED/otherUNKNOWN')
   for field in ['original_global_index','original_query_xy','original_L','original_query_available']:
    need(len({json.dumps(r[field],sort_keys=True)for r in nr})==1,'Original metadata same3 variants '+field)
   if n==slot:need(json.loads(nr[0]['original_query_xy'])==assoc['original_selected_query_xy']and nr[0]['original_L']==assoc['original_selected_L'],'Selectedweak query association exact')
  for r in rr:
   unknown+=r['direct_weak_target']==-1;inputU+=not r['original_query_available'];item=ims[r['cache_index']]
   need(str(G/r['clean_path'])==item['clean']['path']and str(G/r['valid_path'])==item['valid']['path'],'Image path mapping')
   im=png(item['clean'],'RGB',(96,96));va=png(item['valid'],'L',(96,96));need(hashlib.sha256(im.tobytes()).hexdigest()==item['source_raw96_sha256']and hashlib.sha256(decoded_valid(va)).hexdigest()==item['source_valid96_sha256'],'Generated PNG decodedsource hashes')
   rawimages[r['cache_index']]=im;validimages[r['cache_index']]=va
 ok('All decoded clean/valid PNGs equal source bytehash manifest',True)
 ok('Known/UNKNOWN denominator and U retention',Counter(r['direct_weak_target']for r in rows)=={-1:21*N,0:2*N,1:N}and unknown==21*N and all(r['original_query_available'] or r['direct_weak_target']==-1 for r in rows))
 contacts=manifest['contacts'];ok('All selected contacts complete exact',contacts==c['contacts']and len(contacts)==N and [t['attempt_id']for t in selected]==[x['attempt_id']for x in contacts])
 contexts=manifest['fullcontexts'];ok('Exactly3context statuses per selectedtriplet',len(contexts)==N*3)
 assigned=[]
 for i,t in enumerate(selected):
  rr=groups[(t['domain'],t['attempt_id'])];card=png(contacts[i]['contact'],'RGB',(1488,1040));all8=png(contacts[i]['all8'],'RGB',(1230,1310))
  for vi,v in enumerate(VARS):
   r=next(r for r in rr if r['variant']==v and r['native_slot']==t['selected_native_slot']);x=18+486*vi
   need(card.crop((x,158,x+384,542)).tobytes()==rawimages[r['cache_index']].resize((384,384),Image.Resampling.NEAREST).tobytes(),'Contact cleanROI original')
   for n in range(8):
    r=next(r for r in rr if r['variant']==v and r['native_slot']==n);x=12+410*vi;y=70+155*n;need(all8.crop((x,y,x+96,y+96)).tobytes()==rawimages[r['cache_index']].tobytes(),'All8 contact cleanROI original')
  ctxs=[x for x in contexts if x['attempt_id']==t['attempt_id']];need({x['variant']for x in ctxs}==set(VARS),'Contextvariants exact')
  for ctx in ctxs:
   need(expectedpackets.get(ctx['source_packet']['path'])==ctx['source_packet'],'Context onlysame selected exactpacket')
   if ctx['available']:
    original=png(ctx['original'],'RGB');display=png(ctx['display'],'RGB');need(original.size==display.size and ctx['exact_member']=='full_RGB'and ctx['display_only']is True and hashlib.sha256(original.tobytes()).hexdigest()==ctx['original_RGB_array_sha256'],'Fullcontext originalbytes/declaredmember nofallback')
   else:need(ctx['reason']=='full_RGB_MEMBER_NOT_DECLARED_OR_MISSING_NO_FALLBACK','Missingcontext explicit')
  if i>=8:assigned.append(dict(index=i,attempt_id=t['attempt_id'],contact=contacts[i]['contact'],BEFORE_fullcontext=next((x['display']for x in ctxs if x['variant']=='BEFORE'and x['available']),None)))
 ok('Contacts cleanROI/all8 originalbytes and allfullcontexts PNGhash/member exact',True)
 ok('Fullcontext availability census',sum(x['available']for x in contexts)==c['fullcontext_available']and sum(not x['available']for x in contexts)==c['fullcontext_missing'])
 result=dict(completed=True,PASS=True,entry=bind(entry),code=bind(__file__),protocol=e['protocol'],viewer_complete=e['viewer_complete'],checks=checks,all109_triplets=109,selected_triplets=N,selected_domain_counts=dict(Counter(t['domain']for t in selected)),selected_all8_rows=len(rows),known_D=2*N,known_E=N,UNKNOWN=unknown,inputU=inputU,fullcontext_available=c['fullcontext_available'],fullcontext_missing=c['fullcontext_missing'],PNGsourcebytehash_checks=2*len(rows),assigned_B_actual_views=assigned,actual_image_tool_views_by_this_script=0,HTML_browser_rendered=False,original_NPZ_header_weights_privateeval_scores_or_newmodels_read=0,training_admission=False,selected_model=None,fully_solved=False,limits=['Generated data/PNG consistency only, not a rerun of original NPZ/source proofs.','Root00..07/B08..last actualtools views recorded separately; script PNG decode is not manual view.','Weak finite-end/sourcecanonical association is not human/nativephysicalgold or new label.'])
 write(H/'NUMERICAL_AND_PNG_BINDING_REVIEW.json',result);print(json.dumps(dict(completed=True,PASS=True,review=bind(H/'NUMERICAL_AND_PNG_BINDING_REVIEW.json'),selected=N,rows=len(rows),inputU=inputU,assigned=assigned)))
if __name__=='__main__':
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--invented',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();print(json.dumps(invented()))if a.invented else run(a.entry)
