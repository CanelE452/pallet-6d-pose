"""GT-independent source9x9 proposals, actual RGB96 inputs, then GT-only labels."""
from pathlib import Path
import argparse
from collections import Counter,defaultdict
import hashlib,json,math,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CUE=HERE.parent/'ycb_cue_review'
SOURCE=CUE/'INPUTS_RGB_ONLY.jsonl'
GT_PROPOSAL=CUE/'FROZEN_AI_CUE_PROPOSALS_GT_ONLY.json'
GT_SANITY=CUE/'FROZEN_CUE_OWNERSHIP_SANITY/SANITY_REPORT.json'
ROOT_REVIEW=ROOT/'source_audit/ROOT_YCB_CUE_CONDITIONAL_TEACHER_REVIEW.json'
NPZ_KEYS={'rgb96','crop_valid96','query_valid','query_xy','box_xyxy','L'}
INPUT_META_KEYS={'id','source','split','image','image_sha256','raw_hw','query8','side','box'}
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def need(x,message):
    if not x:raise ValueError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def checked(b):need(sha(b['path'])==b['sha256'],'Binding changed '+b['path']);return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def write(p,d):need(not p.exists(),'Preserve '+str(p));p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def lines(p,d):need(not p.exists(),'Preserve '+str(p));p.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n'for r in d))


def queries_from_box(box):
    b=np.asarray(box,np.float64);need(b.shape==(4,)and np.isfinite(b).all()and b[2]>b[0]and b[3]>b[1],'Valid input box')
    yy,xx=np.mgrid[:9,:9]
    return np.stack([b[0]+(xx.ravel()+.5)*(b[2]-b[0])/9,b[1]+(yy.ravel()+.5)*(b[3]-b[1])/9],axis=1)


def center_valid(q,hw):
    h,w=hw;q=np.asarray(q);xy=np.floor(q+.5).astype(int)
    return bool(np.isfinite(q).all()and 0<=q[0]<w and 0<=q[1]<h and 0<=xy[0]<w and 0<=xy[1]<h)


def crop96(rgb,q,L):
    """Native-area pixel-center bilinear sampling; constant0 outside, no L change."""
    h,w=rgb.shape[:2];u=(np.arange(96,dtype=np.float64)+.5)/96-.5
    sx=q[0]+u*L;sy=q[1]+u*L;xx,yy=np.meshgrid(sx,sy)
    x0=np.floor(xx).astype(int);y0=np.floor(yy).astype(int);dx=xx-x0;dy=yy-y0
    acc=np.zeros((96,96,3),np.float64)
    for xo,yo,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
        x=x0+xo;y=y0+yo;inside=(x>=0)&(x<w)&(y>=0)&(y<h)
        acc[inside]+=rgb[y[inside],x[inside]].astype(np.float64)*weight[inside,None]
    valid=(xx>=-.5)&(xx<w-.5)&(yy>=-.5)&(yy<h-.5)
    result=np.floor(np.clip(acc,0,255)+.5).astype(np.uint8);result[~valid]=0
    return result.transpose(2,0,1).copy(),valid[None].copy()


def availability(q,L,hw,bb):
    if not center_valid(q,hw):return 'INVALID'
    h,w=hw;b=[max(-.5,q[0]-L/2),max(-.5,q[1]-L/2),min(w-.5,q[0]+L/2),min(h-.5,q[1]+L/2)]
    if b[0]<=bb[0]and b[1]<=bb[1]and b[2]>=bb[2]and b[3]>=bb[3]:return 'FULL'
    if min(b[2],bb[2])>max(b[0],bb[0])and min(b[3],bb[3])>max(b[1],bb[1]):return 'PARTIAL'
    return 'MISS'


def strict_conflicts(groups,labels):
    result=[]
    for key,ix in groups.items():
        states={labels[i]['training_state']for i in ix if labels[i]['training_state']!='U'}
        if {'D','E'}<=states:result.append(dict(key=key,indices=ix))
    return result


def freeze():
    need(not(HERE/'PROTOCOL.json').exists(),'Existing protocol')
    source=rows(SOURCE);need(len(source)==24 and all(set(r)==INPUT_META_KEYS for r in source),'Whitelisted24view source metadata')
    protocol=dict(schema='V71_YCB_GT_INDEPENDENT_SOURCE_PROPOSAL_GRID_V1',source_only=True,
      scope='New emulated source proposal distribution. NOT current/native8 initializer results, canonical8 semantics, humanGT, model success, or current-data inference.',
      query_recipe='Row-major9x9 cell centers: qx=oldbox.x0+(col+.5)*oldbox.width/9; qy=oldbox.y0+(row+.5)*oldbox.height/9. No cue/GT/mask/state/score participates.',
      L='Exact preexisting old source L, unchanged for all81 proposals in both original and controlled views.',
      counts=dict(targets=12,views=24,queries_per_view=81,all_input_crops=1944,TRAIN_targets=8,VALIDATION_targets=4),
      crop_recipe='RGB96 output resolution, oldL native span: samplexy=q+((j+.5)/96-.5)*L; float64 bilinear at native pixel centers, outside neighbor0, uint8 floor(value+.5). Samples outside raw pixel-cell footprint[-.5,W-.5)x[-.5,H-.5) are RGB0 and cropvalidFalse.',
      query_valid='finite q, continuous0<=qx<W/0<=qy<H AND floor(q+.5) native integer bounds. Invalid queries retained with teacherU.',
      model_input_NPZ_whitelist=sorted(NPZ_KEYS),
      GT_forbidden_in_NPZ=['reference_xy','reference_uncertainty','GT_state','mask','alpha','source','semanticID','native8_ID','offset','targetID','sampleID'],
      metadata='IDs, source/split/family/group, rawRGB/path/hash and grid row/col are external index lineage only, not NPZ model tensors.',
      GT_stage='Only after input grid+actualRGB/inputNPZ seal, join fixed12 refs and root-approved conditionalD7/U5 in separate GT-only files. FULL plus conditionalD=>trainingD; PARTIAL/MISS/INVALID or sourceU=>trainingU. Existing cutout uncertainties clear12, so neutralD only/E0; do not manufacture E.',
      conflict_audit='Exact full decodedRGB/q/L/box tuple and exact effectiveRGBcrop/valid/queryvalid/q/L/box tuple; strict supervisedD-versusE conflicts reject. Preserve D/U mixed knowledge groups as unresolved diagnostics.',
      preserve_all_proposals=True,GT_center_recenter=False,L_expansion=False,new_jitters_or_composites=False,
      code=ref(__file__),source_input_manifest=ref(SOURCE),
      later_GT_bindings_only=[ref(GT_PROPOSAL),ref(GT_SANITY),ref(ROOT_REVIEW)],
      teacher_GT_content_parsed_before_grid=False,current_model_forward_training_PnP=0)
    write(HERE/'PROTOCOL.json',protocol)
    grid=[]
    for view_index,r in enumerate(source):
        qgrid=queries_from_box(r['box'])
        for k,q in enumerate(qgrid):
            grid.append(dict(input_index=len(grid),view_index=view_index,view_id=r['id'],original_target_id=r['id'] if r['source']=='YCB_ORIGINAL'else ':'.join(r['id'].split(':')[1:5]),source_lineage_only=r['source'],split=r['split'],image=r['image'],image_sha256=r['image_sha256'],raw_hw=r['raw_hw'],query_xy=q.tolist(),grid_row=k//9,grid_col=k%9,box_xyxy=r['box'],L=r['side'],native_query_valid_from_metadata=center_valid(q,r['raw_hw'])))
    need(len(grid)==1944,'All grid sites retained')
    lines(HERE/'FROZEN_QUERY_GRID_INPUT_METADATA.jsonl',grid)
    write(HERE/'GRID_INPUT_LOCK.json',dict(completed=True,source_only=True,code=ref(__file__),protocol=ref(HERE/'PROTOCOL.json'),source=ref(SOURCE),grid=ref(HERE/'FROZEN_QUERY_GRID_INPUT_METADATA.jsonl'),query_derived_from_box_only=True,cue_GT_mask_state_not_parsed=True,actual_RGB_reads_before_grid_lock=0,current_model_forward_training_PnP=0))
    print(json.dumps(ref(HERE/'GRID_INPUT_LOCK.json')))


def extract():
    lock=read(HERE/'GRID_INPUT_LOCK.json');need(checked(lock['code'])==Path(__file__).resolve(),'Input extractor frozen code');checked(lock['grid']);checked(lock['protocol']);checked(lock['source'])
    grid=rows(HERE/'FROZEN_QUERY_GRID_INPUT_METADATA.jsonl');N=len(grid);need(N==1944,'Complete fixed grid')
    arrays=dict(rgb96=np.empty((N,3,96,96),np.uint8),crop_valid96=np.empty((N,1,96,96),bool),query_valid=np.empty(N,bool),query_xy=np.empty((N,2),np.float64),box_xyxy=np.empty((N,4),np.float64),L=np.empty(N,np.float64))
    meta=[];cache={}
    for i,r in enumerate(grid):
        if r['image']not in cache:
            need(sha(r['image'])==r['image_sha256'],'Raw RGB hash');rgb=np.asarray(Image.open(r['image']).convert('RGB')).copy();need(list(rgb.shape[:2])==r['raw_hw'],'Actual raw native shape')
            cache[r['image']]=(rgb,hashlib.sha256(rgb.tobytes()).hexdigest())
        rgb,fullsha=cache[r['image']];q=r['query_xy'];L=r['L'];c,v=crop96(rgb,q,L)
        arrays['rgb96'][i]=c;arrays['crop_valid96'][i]=v;arrays['query_valid'][i]=center_valid(q,rgb.shape[:2]);arrays['query_xy'][i]=q;arrays['box_xyxy'][i]=r['box_xyxy'];arrays['L'][i]=L
        need(bool(arrays['query_valid'][i])==r['native_query_valid_from_metadata'],'Native metadata validity replay')
        meta.append(dict(**r,decoded_fullRGB_sha256=fullsha))
    need(set(arrays)==NPZ_KEYS,'Strict tensor whitelist')
    dest=HERE/'INPUT_ONLY.npz';need(not dest.exists(),'No input overwrite');np.savez_compressed(dest,**arrays)
    lines(HERE/'INPUT_INDEX_LINEAGE_ONLY.jsonl',meta)
    mosaics=[]
    for split in ['TRAIN','VALIDATION']:
        target=next(r['original_target_id']for r in meta if r['split']==split);ix=[r['input_index']for r in meta if r['original_target_id']==target]
        need(len(ix)==162,'Both complete81grid views')
        im=Image.new('RGB',(1740,950),'#f5f7fa');d=ImageDraw.Draw(im);d.text((12,6),split+' first fixed target '+target+' | original81 / controlled81 | oldL native span ->RGB96',font=ImageFont.truetype(FONT,21),fill='#172536')
        for j,i in enumerate(ix):
            col=j%9;row=(j%81)//9;view=j//81;im.paste(Image.fromarray(arrays['rgb96'][i].transpose(1,2,0)),(6+view*870+col*96,70+row*96))
        f=HERE/(split+'_ALL81_INPUT_CROPS_FIRST_TARGET.png');im.save(f);mosaics.append(ref(f))
    write(HERE/'INPUT_EXTRACTION_SEAL.json',dict(completed=True,source_only=True,input_lock=ref(HERE/'GRID_INPUT_LOCK.json'),code=ref(__file__),NPZ=ref(dest),index=ref(HERE/'INPUT_INDEX_LINEAGE_ONLY.jsonl'),mosaics=mosaics,N=1944,query_valid=int(arrays['query_valid'].sum()),NPZ_keys=sorted(arrays),GT_arrays_read_before_input_seal=0,source_GT_semantic_fields_in_NPZ=0,runtime=dict(executable=sys.executable,numpy=np.__version__),new_model_forward_training_current_PnP=0))
    print(json.dumps(ref(HERE/'INPUT_EXTRACTION_SEAL.json')))


def label_and_audit():
    seal=read(HERE/'INPUT_EXTRACTION_SEAL.json');checked(seal['NPZ']);checked(seal['index']);checked(seal['input_lock'])
    protocol=read(HERE/'PROTOCOL.json')
    for b in protocol['later_GT_bindings_only']:checked(b)
    proposal=read(GT_PROPOSAL);sanity=read(GT_SANITY);review=read(ROOT_REVIEW)
    need(sanity['reference_geometry_unchanged']and len(proposal['proposals'])==12,'Fixed reference proof')
    fixed={r['target_id']:r for r in proposal['proposals']};states={r['target_id']:r for r in sanity['cases']}
    need(sum(r['original_cue_teacher_candidate'].startswith('D_')for r in states.values())==7,'Frozen conditionalD7/U5')
    need(all(r['existing_controlled_paste_covered_uncertainty_pixels']==0 for r in states.values()),'Existing controlled RGB neutral at entire cue uncertainty')
    need(review.get('completed')is True,'Root source teacher review')
    index=rows(HERE/'INPUT_INDEX_LINEAGE_ONLY.jsonl');labels=[];fullgroups=defaultdict(list);cropgroups=defaultdict(list)
    with np.load(seal['NPZ']['path'],allow_pickle=False)as z:
        need(set(z.files)==NPZ_KEYS,'No GT tensor keys');d={k:z[k]for k in z.files}
    for i,r in enumerate(index):
        f=fixed[r['original_target_id']];s=states[r['original_target_id']];base='D'if s['original_cue_teacher_candidate'].startswith('D_')else'U';a=availability(r['query_xy'],r['L'],r['raw_hw'],f['uncertainty_bbox_xyxy']);teacher='D'if base=='D'and a=='FULL'else'U'
        labels.append(dict(input_index=i,view_id=r['view_id'],original_target_id=r['original_target_id'],split=r['split'],source_lineage_only=r['source_lineage_only'],grid_row=r['grid_row'],grid_col=r['grid_col'],cue_refID=f['cue_refID'],reference_xy=f['reference_xy'],uncertainty_bbox_xyxy=f['uncertainty_bbox_xyxy'],source_conditional_state=base,crop_availability=a,training_state=teacher,why='conditional sourceD plus FULL'if teacher=='D'else'sourceU or nonFULL/INVALID; not semanticE/O/S',native8_semantic_ID='UNKNOWN',model_input_contains_this_GT=False))
        params=d['query_xy'][i].tobytes()+d['box_xyxy'][i].tobytes()+np.asarray(d['L'][i],np.float64).tobytes()
        fullkey=hashlib.sha256(bytes.fromhex(r['decoded_fullRGB_sha256'])+params).hexdigest()
        cropkey=hashlib.sha256(d['rgb96'][i].tobytes()+d['crop_valid96'][i].tobytes()+np.asarray(d['query_valid'][i],bool).tobytes()+params).hexdigest()
        fullgroups[fullkey].append(i);cropgroups[cropkey].append(i)
    need(len(labels)==1944 and not any(r['training_state']=='E'for r in labels),'All proposals, no inventedE')
    conflict={}
    for name,groups in [('same_fullRGB_query_L_box',fullgroups),('same_effectivecrop_valid_query_L_box',cropgroups)]:
        bad=strict_conflicts(groups,labels);need(not bad,'Strict supervisedD/E conflict')
        mixed=[dict(key=k,indices=ix)for k,ix in groups.items()if len({labels[i]['training_state']for i in ix})>1]
        conflict[name]=dict(groups=len(groups),duplicate_groups=sum(len(ix)>1 for ix in groups.values()),duplicate_rows=sum(len(ix)for ix in groups.values()if len(ix)>1),strict_D_E_conflicts=bad,D_U_mixed_knowledge_groups=mixed)
    lines(HERE/'GT_ONLY_GRID_TEACHERS.jsonl',labels)
    write(HERE/'INPUT_IDENTITY_CONFLICT_AUDIT.json',dict(completed=True,N=1944,audits=conflict,strict_supervised_D_E_conflicts=0,GT_ambiguous_U_not_relabelled_E=True,model_input_hash_ignores_GT_reference_state_source_ID=True))
    def counts(rr):return dict(N=len(rr),source_conditional_states=dict(Counter(r['source_conditional_state']for r in rr)),availability=dict(Counter(r['crop_availability']for r in rr)),training=dict(Counter(r['training_state']for r in rr)))
    result=dict(schema='V71_YCB_EMULATED_SOURCE_PROPOSAL_GRID_RESULT_V1',completed=True,status='ACTUAL_INPUT_AND_GT_ONLY_COVERAGE_AUDIT_COMPLETE_NO_MODEL_FIT',source_only=True,
      counts=counts(labels),by_split={s:counts([r for r in labels if r['split']==s])for s in ['TRAIN','VALIDATION']},by_view_source={s:counts([r for r in labels if r['source_lineage_only']==s])for s in ['YCB_ORIGINAL','YCB_CONTROLLED']},
      denominator_all_proposals=1944,all_grid_proposals_preserved=True,conditional_reference_targets=dict(D=7,U=5),old_q8_training_D=dict(TRAIN=0,VALIDATION=2),
      actual_input_keys=sorted(d),input_GT_ref_state_mask_alpha_source_ID_offset_fields=0,GT_only_whitelist=sorted(labels[0]),
      scope='New9x9 source-emulated proposal query distribution, not actual native8 initializer success, source-human semantic gold, training result, or current223 result.',
      E_count=0,existing_cutout_role='Neutral sourceD comparison only; every fixed reference uncertainty has pastecoverage0.',
      reference_or_query_GT_recenter=False,L_expansion=False,source_frame_and_donor_split_preserved=True,native8_semantic_dictionary='UNKNOWN',semantic_training_admission_pending=True,
      identity_audits=conflict,bindings=dict(protocol=ref(HERE/'PROTOCOL.json'),grid_input_lock=ref(HERE/'GRID_INPUT_LOCK.json'),input_extraction_seal=ref(HERE/'INPUT_EXTRACTION_SEAL.json'),input_NPZ=seal['NPZ'],lineage_index=seal['index'],GT_only_teachers=ref(HERE/'GT_ONLY_GRID_TEACHERS.jsonl'),fixed_reference_proposals=ref(GT_PROPOSAL),source_conditional_sanity=ref(GT_SANITY),root_conditional_teacher_review=ref(ROOT_REVIEW),conflict_audit=ref(HERE/'INPUT_IDENTITY_CONFLICT_AUDIT.json'),code=ref(__file__)),
      mosaics=seal['mosaics'],direct_QA='Pending visual inspection; separate additive DIRECT_QA.json will bind this result.',new_model_forward_calls=0,training_calls=0,current_data_reads=0,PnP_calls=0)
    write(HERE/'GRID_RESULT.json',result);print(json.dumps(dict(report=ref(HERE/'GRID_RESULT.json'),counts=result['counts'],by_split=result['by_split']),ensure_ascii=False))


def mock(path):
    q=queries_from_box([0.,0.,9.,9.]);need(np.array_equal(q[0],[.5,.5])and np.array_equal(q[-1],[8.5,8.5]),'Cell centers')
    yy,xx=np.mgrid[:19,:23];rgb=np.stack([xx,yy,xx+yy],axis=2).astype(np.uint8);c,v=crop96(rgb,[11.,9.],12.)
    u=(np.arange(96)+.5)/96-.5;expected=np.floor(11+u*12+.5).astype(np.uint8)
    need(np.array_equal(c[0],np.tile(expected,(96,1)))and v.all(),'Actual bilinear affine/ramp phase')
    need(availability([10.,10.],10.,[20,20],[8,8,12,12])=='FULL','FULL')
    need(availability([10.,10.],2.,[20,20],[8,8,12,12])=='PARTIAL','PARTIAL')
    need(availability([2.,2.],2.,[20,20],[8,8,12,12])=='MISS','MISS')
    need(availability([19.9,10.],2.,[20,20],[8,8,12,12])=='INVALID','INVALID rounded')
    need(len(strict_conflicts({'k':[0,1]},[dict(training_state='D'),dict(training_state='E')]))==1,'Reject D/E conflict')
    need(not strict_conflicts({'k':[0,1]},[dict(training_state='D'),dict(training_state='U')]),'U is not knownE')
    write(Path(path),dict(completed=True,status='PASS',generated_only=True,code=ref(__file__),GT_data_RGB_source_reads=0,affine_bilinear_phase=True,all_coverage_states=True,strict_D_E_conflict_and_D_U_distinction=True,new_model_training_current_PnP=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['mock','freeze','extract','label']);p.add_argument('--out',type=Path);a=p.parse_args()
    if a.mode=='mock':mock(a.out)
    elif a.mode=='freeze':freeze()
    elif a.mode=='extract':extract()
    else:label_and_audit()
