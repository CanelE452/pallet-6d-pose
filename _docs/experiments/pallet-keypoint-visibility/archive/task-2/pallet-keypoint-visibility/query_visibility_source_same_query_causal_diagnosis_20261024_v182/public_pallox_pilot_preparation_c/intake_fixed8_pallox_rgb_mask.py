"""Frozen metadata-first public RGB/mask pilot. --prepare never uses network.

Actual root entry permits only sixteen declared RGB/mask files. No pose/depth/
mesh, initializer, SAM, features, weights, model inference, training or labels.
"""
from pathlib import Path
from collections import Counter
from urllib.parse import urlparse
import argparse,hashlib,io,json,traceback
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;P=ROOT.parent
SCOPE=ROOT/'ROOT_SCOPE_BEFORE_NEW_COMMON_QUERY_OR_PUBLIC_PIXELS.json'
SCOPE_SHA='142ba2fd62dfa16c79bda690bf90814d0a9cdd330898722b8f896edf76b396a1'
OLD=P/'query_visibility_public_source_target_definition_audit_20261024_v179'
SOURCE=OLD/'pallox_source_intake_readiness_c/ALL322_REGISTERED_FRAME_BINDINGS_METADATA_ONLY.jsonl'
SOURCE_SHA='b12aca1cdb6ef17fa49f87a412a8c71ab47f633ed50c2822a8b4a1a04a1e84b5'
OFFICIAL=OLD/'OFFICIAL_DATAVERSE_DATASET_METADATA.json'
ALL=HERE/'ALL322_SOURCE_TRAIN_CANDIDATE_METADATA_ONLY.jsonl'
PILOT=HERE/'FIXED8_RGB_MASK_PAIRS_BEFORE_PIXELS.jsonl'
PROTOCOL=HERE/'PROTOCOL_BEFORE_PALLOX_RGB_MASK_DOWNLOAD_OR_BODIES.json'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def require(ok,msg):
    if not bool(ok):raise AssertionError(msg)
def sha(path):
    path=Path(path).resolve();require(path.is_relative_to(P) and '_v112' not in str(path).lower(),'Forbidden path')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def binding(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def verified(b):require(binding(b['path'])==b,'Binding drift '+b['path']);return Path(b['path'])
def readj(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x]
def writej(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def writerows(p,rr):Path(p).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rr))
def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def select8(rr):
    ordered=sorted(rr,key=lambda r:(r['acquisition_group'],r['frame_id']))
    require(len(ordered)==322 and len({r['source_id'] for r in ordered})==322,'Original322 census')
    available=[r for r in ordered if r['files']['masks'] is not None]
    require(len(available)==290,'Original mask290 and missing32')
    groups=sorted({r['acquisition_group'] for r in ordered})
    require(len(groups)==6,'Original six acquisition groups')
    chosen=[next(r for r in available if r['acquisition_group']==g) for g in groups]
    chosen += [r for r in available if r['source_id'] not in {x['source_id'] for x in chosen}][:2]
    require(len(chosen)==8 and len({r['source_id'] for r in chosen})==8,'Fixed8 distinct cases')
    return chosen

def official_records():
    d=readj(OFFICIAL)['data'];out={}
    for f in d['latestVersion']['files']:
        z=f['dataFile'];out[z['id']]=dict(relative_path=(f.get('directoryLabel','')+'/'+z['filename']).lstrip('/'),
            declared_bytes=z['filesize'],declared_checksum=z['checksum'],restricted=f['restricted'])
    return out

def validate_file_record(r,mode,official):
    f=r['files'][mode];require(f is not None,'Selected modality registered')
    o=official[f['official_file_id']]
    for k in ('relative_path','declared_bytes','declared_checksum','restricted'):require(f[k]==o[k],'Exact official metadata '+k)
    require(not f['restricted'] and f['declared_checksum']['type']=='MD5','Unrestricted declared MD5')
    expected='https://rdr.kuleuven.be/api/access/datafile/'+str(f['official_file_id'])
    require(f['download_API_URL']==expected,'Exact official API ID URL')
    require('/'+('rgb/' if mode=='rgb' else 'masks/') in f['relative_path'],'RGB/mask-only file class')

def canonical_decode_rgb(payload):
    a=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_UNCHANGED)
    require(a is not None and a.dtype==np.uint8 and a.ndim==3 and a.shape[2]==3,'Original RGB must decode uint8 BGR3')
    return cv2.cvtColor(a,cv2.COLOR_BGR2RGB)

def decode_mask(payload):
    a=cv2.imdecode(np.frombuffer(payload,np.uint8),cv2.IMREAD_UNCHANGED)
    require(a is not None,'Original mask decode failed')
    return a

def mask_facts(mask,hw):
    values=np.unique(mask)
    binary=bool(mask.dtype==np.uint8 and mask.ndim==2 and tuple(mask.shape)==tuple(hw) and np.isin(values,[0,255]).all())
    return dict(dtype=str(mask.dtype),shape=list(mask.shape),unique_value_count=len(values),
        unique_values=values.tolist() if len(values)<=64 else None,unique_values_sha256=array_sha(values),
        min_value=int(values.min()),max_value=int(values.max()),pixel_size_agrees_with_RGB=bool(mask.ndim==2 and tuple(mask.shape)==tuple(hw)),
        declared_binary255_schema_matches=binary,physical_alignment_or_human_review_certified=False)

def display_overlay(rgb,mask):
    require(mask_facts(mask,rgb.shape[:2])['declared_binary255_schema_matches'],'Overlay only declared binary aligned schema')
    view=rgb.copy();on=mask==255
    view[on]=np.rint(.65*rgb[on].astype(np.float64)+.35*np.array([239,62,68])).astype(np.uint8)
    return view

def verify_payload(payload,record):
    require(len(payload)==record['declared_bytes'],'Declared byte size mismatch')
    require(hashlib.md5(payload).hexdigest()==record['declared_checksum']['value'],'Declared MD5 mismatch')

def fetch_one(record,destination):
    # Only called after exact root-entry/code/manifest checks in run(). No retry.
    from urllib.request import urlopen,Request
    expected=record['declared_bytes'];url=record['download_API_URL']
    req=Request(url,headers={'User-Agent':'source-metadata-bound-pilot/1.0'})
    with urlopen(req,timeout=30) as response:
        final=urlparse(response.url)
        require(final.scheme=='https' and (final.hostname=='kuleuven.be' or final.hostname.endswith('.kuleuven.be')),'Unexpected official redirect host')
        payload=response.read(expected+1)
        response_meta={'HTTP':response.status,'final_URL_without_query':final._replace(query='',fragment='').geturl()}
    destination.write_bytes(payload)  # Retain exact returned bytes even if verification fails.
    verify_payload(payload,record)
    return payload,dict(original_file_record=record,bytes=len(payload),MD5=hashlib.md5(payload).hexdigest(),saved=binding(destination),**response_meta)

def font(size):return ImageFont.truetype(FONT,size)
def text(draw,xy,s,size=20):draw.text(xy,str(s),fill='#e2eaf5',font=font(size))
def save_clean(path,a):
    Image.fromarray(a).save(path)
    back=np.asarray(Image.open(path).convert('RGB'))
    require(np.array_equal(back,a),'Clean canonical RGB PNG array parity')
    return dict(**binding(path),decoded_array_sha256=array_sha(back))

def make_card(row,rgb,mask,facts):
    im=Image.new('RGB',(2000,700),'#111b2a');d=ImageDraw.Draw(im)
    text(d,(20,14),row['source_id']+' | SOURCE_TRAIN_CANDIDATE | mask-defined whole Pallox bin only',28)
    text(d,(20,57),'Original RGB (canonical BGR -> RGB); original mask; separate display overlay. No q/GT crop/model inputs created.',20)
    panels=[]
    if rgb is not None:panels.append(('CLEAN original RGB 1x',Image.fromarray(rgb)))
    if mask is not None and facts is not None and facts['declared_binary255_schema_matches']:
        panels.append(('Author mask 0/255 1x',Image.fromarray(mask).convert('RGB')))
        panels.append(('DISPLAY COPY: red author-mask',Image.fromarray(display_overlay(rgb,mask))))
    for j,(title,p) in enumerate(panels):
        x=20+660*j;text(d,(x,96),title,21)
        if p.size==(640,480):im.paste(p,(x,132))
        else:
            # Preserve source array separately. Unexpected shapes are fitted only
            # in the display copy, with explicit actual shape/scale caption.
            p.thumbnail((640,480),Image.Resampling.NEAREST);im.paste(p,(x,132))
            text(d,(x,610),'Unexpected source shape: display fitted nearest',14)
    if len(panels)<3:text(d,(680,200),'UNRESOLVED or failed mask/RGB schema; raw payload and facts preserved.',19)
    text(d,(20,625),'Whole-bin raster != flat-pallet/native-end visibility. Black non-target does not distinguish FOREIGN from BACKGROUND.',19)
    text(d,(20,658),'SAM1 filename does not prove human QA or visible/amodal authority. No initializer/model/labels/training admission.',18)
    return im

def make_contact(cards):
    require(len(cards)==8,'All fixed8 card images retained')
    contact=Image.new('RGB',(2000,1400),'#111b2a')
    for i,card in enumerate(cards):
        contact.paste(card.resize((1000,350),Image.Resampling.NEAREST),((i%2)*1000,(i//2)*350))
    return contact

def prepare():
    require(sha(SCOPE)==SCOPE_SHA and sha(SOURCE)==SOURCE_SHA,'Frozen source/scope')
    original=rows(SOURCE);official=official_records();selected=select8(original)
    census=[]
    for r in original:
        out=dict(r);out.update(split='SOURCE_TRAIN_CANDIDATE',source_split_authority='Prospective source-only candidate census, not official ML split',mask_missing=r['files']['masks'] is None,training_admission=False)
        census.append(out)
    pilot=[]
    for rank,r in enumerate(selected):
        for mode in ('rgb','masks'):validate_file_record(r,mode,official)
        out=dict(r);out.update(selection_rank=rank,split='SOURCE_TRAIN_CANDIDATE',training_admission=False,original_metadata_source=binding(SOURCE))
        pilot.append(out)
    writerows(ALL,census);writerows(PILOT,pilot)
    tests=[]
    def passed(name,ok):require(ok,name);tests.append({'name':name,'PASS':True})
    passed('all322/missing32 and no official MLsplit relabel claim',len(census)==322 and sum(r['mask_missing'] for r in census)==32 and all(not r['training_admission'] for r in census))
    passed('first per6groups then first2remaining sorted8 outcome-blind',[r['source_id'] for r in selected]==['0.55m_hall:0000','0.55m_lab:0000','1.23m_hall:0000','1.23m_lab:0000','1.92m_hall:0000','1.92m_lab:0000','0.55m_hall:0001','0.55m_hall:0002'])
    rgb=np.array([[[3,7,11],[19,23,29]],[[31,37,41],[43,47,53]]],np.uint8);copy=rgb.copy()
    ok,encoded=cv2.imencode('.png',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR));require(ok,'Invented PNG encode')
    passed('actual cv2 unchanged decode BGR->RGB exactbytes',np.array_equal(canonical_decode_rgb(encoded.tobytes()),rgb))
    payload=encoded.tobytes();record=dict(declared_bytes=len(payload),declared_checksum=dict(type='MD5',value=hashlib.md5(payload).hexdigest()))
    verify_payload(payload,record);passed('actual payload bytes and MD5 verifier accepts exact inventedPNG',True)
    rejected=0
    for bad in (dict(record,declared_bytes=len(payload)+1),dict(record,declared_checksum=dict(type='MD5',value='0'*32))):
        try:verify_payload(payload,bad)
        except AssertionError:rejected+=1
    passed('actual payload verifier rejects size or checksum drift without decoding',rejected==2)
    m=np.array([[0,255],[255,0]],np.uint8);ok,encoded=cv2.imencode('.png',m);require(ok,'Invented mask encode')
    back=decode_mask(encoded.tobytes());passed('actual mask unchanged decode dtype/values',np.array_equal(back,m) and back.dtype==m.dtype)
    facts=mask_facts(m,(2,2));passed('mask binary schema/size inspection explicit',facts['declared_binary255_schema_matches'] and not facts['physical_alignment_or_human_review_certified'])
    passed('unexpected uint16 or nonbinary mask stays unresolved',not mask_facts(m.astype(np.uint16),(2,2))['declared_binary255_schema_matches'] and not mask_facts(np.ones((2,2),np.uint8),(2,2))['declared_binary255_schema_matches'])
    overlay=display_overlay(rgb,m);passed('overlaycopy never mutates clean RGB/mask',np.array_equal(rgb,copy) and np.array_equal(m,[[0,255],[255,0]]) and not np.array_equal(overlay,rgb))
    # Exercise complete card geometry only with invented640x480 arrays.
    toy=np.tile(rgb,(240,320,1));toy_mask=np.tile(m,(240,320));card=make_card(pilot[0],toy,toy_mask,mask_facts(toy_mask,(480,640)))
    passed('actual PIL 3panel card layout invented only',card.size==(2000,700) and np.array_equal(np.asarray(card)[132:612,20:660],toy))
    contact=make_contact([Image.new('RGB',(2000,700),(i,0,0)) for i in range(8)])
    passed('actual contact all8 cards 2columns4rows without gaps or clipping',contact.size==(2000,1400) and all(contact.getpixel(((i%2)*1000+500,(i//2)*350+175))==(i,0,0) for i in range(8)))
    checks=HERE/'GENERATED_INVENTED_CHECKS.json';writej(checks,dict(completed=True,PASS=True,tests=tests,network_or_source_payload_reads=0))
    protocol=dict(completed=True,prepared_only=True,scope=binding(SCOPE),code=binding(__file__),source_metadata=binding(SOURCE),official_metadata=binding(OFFICIAL),original_README=binding(OLD/'OFFICIAL_README_SOURCE.txt'),prior_readiness=binding(OLD/'pallox_source_intake_readiness_c/REVIEW.json'),all322_census=binding(ALL),fixed8=binding(PILOT),checks=binding(checks),
        selection='First maskregistered frame per6sorted acquisitiongroups then first2remaining sorted(group,frame); no output/pixel selection',
        only_network_assets='Exactly RGB and mask registered fileIDs for these8pairs; standard official Dataverse URLs; bytes+MD5 before decode and SHA preserved. No automatic retries/replacements.',
        canonical_decode='cv2 IMREAD_UNCHANGED uint8BGR3 -> cvtColor BGR2RGB. Mask IMREAD_UNCHANGED, preserve original dtype/values; unsupported schema retained as unresolved, never normalized into labels.',
        output='Fresh rootentry outputdir; original16payload attempts retained, canonical RGB/mask NPY, rawpixel SHA, clean PNG and separate display overlay/cards/contact/index, all8 attempt rows',
        no_initializer_model_or_query_generation=True,no_pose_depth_mesh_read_or_download=True,no_new_labels_or_training_admission=True,
        authority='WholePalloxbin mask-defined auxiliary only; visible/amodal/humanQA and selected-end/physicalnative authority unresolved. NonT is not FOREIGN-v-BG.',
        source_split='SOURCE_TRAIN_CANDIDATE all322; not an independent holdout or official ML split',browser_render_verified=False,actual_visual_QA='Pending actual source pilot/root entry and separate view_image receipt')
    writej(PROTOCOL,protocol)
    template=HERE/'ROOT_ACTUAL_PALLOX_PILOT_ENTRY_TEMPLATE_APPROVED_FALSE.json'
    writej(template,dict(approved_actual_pallox_RGB_mask_pilot=False,source_only=True,scope=binding(SCOPE),code=binding(__file__),protocol=binding(PROTOCOL),fixed8=binding(PILOT),output_dir=str(ROOT/'public_pallox_rgb_mask_pilot_v1')))
    writej(HERE/'PREPARATION_COMPLETE.json',dict(completed=True,PASS=True,prepared_only=True,code=protocol['code'],protocol=binding(PROTOCOL),checks=binding(checks),all322_census=binding(ALL),fixed8=binding(PILOT),entry_template=binding(template),actual_network_download_or_source_pixel_reads=0))
    print(json.dumps(dict(completed=True,PASS=True,prepared_only=True,census322=True,pilot8=True,code=binding(__file__),protocol=binding(PROTOCOL))))

def run(entry_path):
    e=readj(entry_path);require(e.get('approved_actual_pallox_RGB_mask_pilot') is True and e.get('source_only') is True,'Separate root actual entry required')
    for key,path in (('scope',SCOPE),('code',Path(__file__)),('protocol',PROTOCOL),('fixed8',PILOT)):require(e[key]==binding(path),'Entry exact '+key)
    require(sha(SCOPE)==SCOPE_SHA,'Frozen scope')
    p=readj(PROTOCOL)
    for key in ('source_metadata','official_metadata','original_README','prior_readiness','all322_census','fixed8','checks'):verified(p[key])
    fixed=rows(PILOT);require([r['source_id'] for r in fixed]==[r['source_id'] for r in select8(rows(SOURCE))],'Same frozen8')
    official=official_records()
    out=Path(e['output_dir']);require(out.parent.resolve()==ROOT.resolve() and not out.exists(),'Fresh phase output only; no overwrite/retry')
    out.mkdir();(out/'cases').mkdir();results=[];cards=[]
    writej(out/'STARTED.json',dict(entry=binding(entry_path),code=binding(__file__),protocol=binding(PROTOCOL),fixed8=binding(PILOT),automatic_retry=False))
    for row in fixed:
        cid=row['source_id'].replace(':','_');folder=out/'cases'/cid;folder.mkdir()
        rec=dict(source_id=row['source_id'],split='SOURCE_TRAIN_CANDIDATE',selection_rank=row['selection_rank'],files={},failures=[],training_admission=False,new_owner_native_or_quality_labels=False)
        payload={};rgb=None;mask=None;facts=None
        for mode,name in (('rgb','ORIGINAL_RGB.png'),('masks','ORIGINAL_MASK.png')):
            try:
                validate_file_record(row,mode,official)
                payload[mode],receipt=fetch_one(row['files'][mode],folder/name);rec['files'][mode]=receipt
            except BaseException as error:
                rec['failures'].append(dict(modality=mode,error=type(error).__name__+': '+str(error)))
                dest=folder/name
                rec['files'][mode]=dict(declared=row['files'][mode],saved_returned_bytes=binding(dest) if dest.exists() else None,PASS=False)
        try:
            if 'rgb' in payload:
                rgb=canonical_decode_rgb(payload['rgb']);np.save(folder/'CANONICAL_RGB.npy',rgb,allow_pickle=False)
                rec['canonical_RGB']=dict(**binding(folder/'CANONICAL_RGB.npy'),shape=list(rgb.shape),dtype=str(rgb.dtype),decoded_RGB_sha256=array_sha(rgb),README_dimensions_match=rgb.shape[:2]==(480,640))
                rec['clean_RGB_PNG']=save_clean(folder/'CLEAN_RGB_CANONICAL.png',rgb)
            if 'masks' in payload:
                mask=decode_mask(payload['masks']);np.save(folder/'MASK_ORIGINAL_DTYPE.npy',mask,allow_pickle=False)
                facts=mask_facts(mask,rgb.shape[:2] if rgb is not None else (-1,-1));rec['mask']=dict(**binding(folder/'MASK_ORIGINAL_DTYPE.npy'),decoded_mask_sha256=array_sha(mask),**facts)
            rec['intake_PASS']=len(rec['failures'])==0 and rgb is not None and mask is not None
        except BaseException as error:
            rec['failures'].append(dict(stage='decode_or_save',error=type(error).__name__+': '+str(error)));rec['intake_PASS']=False
        card=make_card(row,rgb,mask,facts);card_path=folder/'CARD.png';card.save(card_path);rec['card']=binding(card_path);cards.append(card_path)
        writej(folder/'INTAKE_PROOF.json',rec);results.append(rec)
        print(json.dumps(dict(case=cid,intake_completed=True,failed=bool(rec['failures']))),flush=True)
    decoded_cards=[]
    for card_path in cards:
        with Image.open(card_path) as im:decoded_cards.append(im.copy())
    contact=make_contact(decoded_cards)
    contact_path=out/'ALL8_RGB_MASK_CONTACT.png';contact.save(contact_path)
    writerows(out/'ALL8_INTAKE_ATTEMPTS.jsonl',results)
    page='<!doctype html><meta charset="utf-8"><title>Fixed Pallox8 RGB-mask intake</title><style>body{font:18px sans-serif;background:#101927;color:#e2eaf5;margin:24px}img{width:100%;height:auto}a{color:#9fcdff}</style><h1>Pallox 공개 RGB-mask 고정 8쌍</h1><p>전체322 SOURCE_TRAIN_CANDIDATE census의 metadata 선택이며 학습·initializer·모델 추론·새 정답은 없습니다. 원 RGB/마스크 payload 및 dtype/값을 보존했습니다. 마스크는 whole bin이며 native/end/external/self 정답이 아닙니다. SAM1 humanQA/visible-amodal 권한은 미확정입니다. HTML 브라우저 렌더 검증 주장은 없습니다.</p>'
    for r in results:
        rel='cases/'+r['source_id'].replace(':','_')+'/CARD.png'
        page+='<h2>'+r['source_id']+'</h2><a href="'+rel+'"><img src="'+rel+'"></a>'
    (out/'index.html').write_text(page)
    passed=sum(r['intake_PASS'] for r in results);binary=sum(bool(r.get('mask',{}).get('declared_binary255_schema_matches')) for r in results)
    writej(out/'COMPLETE.json',dict(completed=True,PASS=passed==8,scope=binding(SCOPE),entry=binding(entry_path),code=binding(__file__),protocol=binding(PROTOCOL),all322_source_census=binding(ALL),fixed8=binding(PILOT),attempts=8,intake_PASS=passed,intake_failed=8-passed,declared_binary_same_size_mask_count=binary,unresolved_mask_schema_count=8-binary,all8_attempts=binding(out/'ALL8_INTAKE_ATTEMPTS.jsonl'),contact=binding(contact_path),index=binding(out/'index.html'),all8_cards=cards and [binding(x) for x in cards],actual_visual_QA='PENDING separate view_image receipt',browser_render_verified=False,pose_depth_mesh_initializer_model_forward_fit_new_labels=0,training_admission=False,selected_model=None,fully_solved=False))
    print(json.dumps(dict(completed=True,PASS=passed==8,attempts=8,complete=binding(out/'COMPLETE.json'))))
    if passed!=8:raise SystemExit('Intake failures preserved; no automatic retry')

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--root-entry',type=Path);args=ap.parse_args()
    if args.prepare:prepare()
    else:run(args.root_entry)
