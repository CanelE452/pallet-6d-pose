"""Saved V207 output-only technical audit. No producer import or original payload follow."""
from pathlib import Path
import ast, collections, csv, hashlib, io, json, math, os, stat, traceback
import numpy as np
from PIL import Image
H=Path(__file__).absolute().parent;V=H.parent;OUT=V/'raw96_weak_witness_control_pilot_v1'
COMP={'path':str(OUT/'COMPLETE.json'),'sha256':'cd496ea6af6995e6e922994958ea90f41c96b50c0e8cea704faf60d1bb571c38'}
N=collections.Counter()

def check(x,m):
    assert x,m
    N[m]+=1
def rb(p):
    p=Path(p);check(p.is_absolute()and V in p.parents and'..'not in p.parts and all('v112'not in x.casefold()for x in p.parts),'lexical ownphase/no V112')
    check(p.suffix in {'.py','.json','.jsonl','.csv','.png','.md','.html'},'no original array/GT/weight/header access')
    if p.suffix=='.png':check(OUT in p.parents,'only generated output PNG, never source26')
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for x in p.parts[1:-1]:n=os.open(x,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    with os.fdopen(n,'rb')as f:check(stat.S_ISREG(os.fstat(f.fileno()).st_mode),'regular no symlink');return f.read()
def checked(b,parse=True):
    z=rb(b['path']);check(hashlib.sha256(z).hexdigest()==b['sha256'],'exact encoded binding')
    if 'size_bytes'in b:check(len(z)==b['size_bytes'],'exact stored size')
    return json.loads(z)if parse else z
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(rb(p)).hexdigest())
def put(p,x):
    with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
def rect(box):
    check(len(box)==4 and all(type(x)is int for x in box),'inclusive integer ROI')
    x0,y0,x1,y1=box;check(0<=x0<=x1<=95 and 0<=y0<=y1<=95,'ROI boundary')
    return {(x,y)for x in range(x0,x1+1)for y in range(y0,y1+1)}
def decode(b,kind):
    raw=checked(b,False)
    with Image.open(io.BytesIO(raw))as im:
        check(im.format=='PNG'and im.size==(96,96),'PNG original96 shape');arr=np.asarray(im).copy()
    check(arr.dtype==np.uint8,'PNG u8')
    if 'decoded_sha256'in b:check(hashlib.sha256(arr.tobytes()).hexdigest()==b['decoded_sha256'],'decoded bytes SHA')
    if kind=='RGB':check(arr.shape==(96,96,3),'RGB96 HWC')
    else:check(arr.shape==(96,96)and set(np.unique(arr))<={0,255},'binary PNG alpha/valid96')
    return arr if kind=='RGB'else arr>0
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def compare_pair(before,cover,sham,ca,sa,valid,case,delta):
    A=rect(case['ROI_A']);C=rect(case['ROI_C']);feature=A|C
    cp={(int(x),int(y))for y,x in np.argwhere(ca)};sp={(int(x),int(y))for y,x in np.argwhere(sa)}
    dx,dy=delta
    check(feature<=cp and feature.isdisjoint(sp),'full exact rectangle-union COVER1/SHAM0')
    check({(x+dx,y+dy)for x,y in cp}==sp,'same translated positive alpha exact, no clipping')
    check(len(cp)==len(sp)>0,'equal positive alpha mass')
    check(not np.any(ca&~valid)and not np.any(sa&~valid),'all positive alpha valid')
    check(np.array_equal(cover[~ca],before[~ca])and np.array_equal(sham[~sa],before[~sa]),'outside-alpha RGB exact')
    check(all(np.array_equal(cover[y,x],sham[y+dy,x+dx])for x,y in cp),'same translated opaque RGB pattern exact')
    check(all(np.array_equal(sham[y,x],before[y,x])for x,y in feature),'SHAM entire weak feature RGB unchanged')
    changed=np.any(cover!=before,axis=2);s_changed=np.any(sham!=before,axis=2)
    fractions={k:dict(changed_pixels=sum(bool(changed[y,x])for x,y in pts),area=len(pts),changed_fraction=sum(bool(changed[y,x])for x,y in pts)/len(pts))for k,pts in [('A',A),('C',C)]}
    check(all(f['changed_pixels']>0 for f in fractions.values()),'COVER actual RGB change in each saved weak ROI')
    return fractions,int(changed.sum()),int(s_changed.sum()),len(cp)

def main():
    # One independent union-vs-bbox fixture, without touching any source data.
    a=rect([0,0,1,1]);c=rect([3,3,4,4]);check(len(a|c)==8 and(2,2)not in a|c,'invented exact union retains bbox holes')
    comp=checked(COMP);selection=checked(comp['selection']);entry=checked(comp['entry']);scope=checked(comp['scope'])
    check(comp['completed']and comp['PASS']and entry['approved']and entry['actual_payload_access_authorized'],'actual source pilot closed')
    check(entry['selection']==comp['selection']and comp['scope']==selection['scope'],'fixed prospective selection binding')
    check(len(selection['cases'])==8 and len(selection['donors'])==10 and len(selection['assignments'])==comp['attempts']==80,'full8x10 denominator')
    for k in ['code','geometry']:ast.parse(checked(comp[k],False).decode())
    checked(comp['protocol']);cases=selection['cases'];donors={d['donor_id']:d for d in selection['donors']}
    check(len({d['object_id']for d in donors.values()})==10 and all(d['role']=='YCB_V_TRAIN_FOREIGN_DONOR_NOT_PALLET_TARGET'for d in donors.values()),'fixed TRAIN donor identities, not native labels')
    attempts=[json.loads(x)for x in checked(comp['outputs']['ALL80_ATTEMPTS.jsonl'],False).splitlines()if x.strip()]
    csvrows=list(csv.DictReader(io.StringIO(checked(comp['outputs']['ALL80_STATUS.csv'],False).decode())))
    preview=checked(comp['outputs']['FIXED8_PREVIEW_MANIFEST.json']);check(len(attempts)==len(csvrows)==80 and len(preview['cards'])==8,'all80 retained and8 previews')
    check([a['attempt_id']for a in attempts]==[a['attempt_id']for a in selection['assignments']],'original assignment order/IDs retained')
    before={};valid={}
    for i,(case,card)in enumerate(zip(cases,preview['cards'])):
        check(case['position']==card['position']==i and case['row_id']==card['row_id'],'fixed original8 order')
        # Original bindings are strings only. The source16 files are never opened.
        check(card['BEFORE']['sha256']==case['CLEAN96']['sha256']and card['VALID']['sha256']==case['VALID96']['sha256'],'output BEFORE/VALID encoded SHA matches declared original16')
        before[i]=decode(card['BEFORE'],'RGB');valid[i]=decode(card['VALID'],'mask')
        check(not np.any(before[i][~valid[i]]),'original invalid RGB zero')
    successes=[];directions=collections.Counter();percase=collections.Counter();objectcounts=collections.Counter();allproofs=[]
    for a,decl,cr in zip(attempts,selection['assignments'],csvrows):
        case=cases[a['case_position']];meta={k:case[k]for k in ['row_id','frame_key','native_slot','original_query_xy','original_L','original_query_available']}
        check(all(a[k]==decl[k]for k in decl)and a['original_meta']==meta and a['ROI_A']==case['ROI_A']and a['ROI_C']==case['ROI_C'],'original case qL/native/U/ROI metadata exact')
        check(a['training_admission']is False and a['new_native_E_label']is False,'no new E/fit admission all80')
        check(all(cr[k]==str(a[k])for k in cr),'status CSV all cells exact')
        check(a['assignment_hash']==sha('V207_RAW96_ASSIGN_V1|'+a['row_id']+'|'+a['donor_id']),'metadata-only assignment hash exact')
        order=sorted([(n,r)for n in [32,40,48,56,64,72,80]for r in [0,90,180,270]],key=lambda z:sha(f'V207_RAW96_TRANSFORM_V1|{a["row_id"]}|{a["donor_id"]}|{z[0]}|{z[1]}'))
        tried=a['transforms_tested'];check([(t['longside'],t['rotation'])for t in tried]==order[:len(tried)],'fixed transform hash-order prefix')
        good=a['status']=='CONSTRUCTED_RAW96_CONTROL_NO_LABEL_ADMISSION'
        if not good:
            check(a['status']=='GEOMETRY_EXHAUSTED'and a['outputs']is None and len(tried)==28 and all(t['COVER_positions']==0 or t['SHAM_positions']==0 for t in tried),'exhausted retained/no fabricated output')
            continue
        check(tried[-1]['COVER_positions']>0 and tried[-1]['SHAM_positions']>0 and all(t['COVER_positions']==0 or t['SHAM_positions']==0 for t in tried[:-1]),'first geometry-feasible candidate retained')
        check(a['transform']=={k:tried[-1][k]for k in ['longside','rotation']},'chosen transform exact')
        outputs=a['outputs'];cover=decode(outputs['COVER_CLEAN96'],'RGB');sham=decode(outputs['SHAM_CLEAN96'],'RGB');ca=decode(outputs['COVER_ALPHA'],'mask');sa=decode(outputs['SHAM_ALPHA'],'mask')
        cp=a['COVER_origin'];sp=a['SHAM_origin'];delta=(sp[0]-cp[0],sp[1]-cp[1])
        shape=donors[a['donor_id']]['recorded_array_proof']['alpha']['shape'];scale=a['transform']['longside']/max(shape)
        hh=max(1,math.floor(shape[0]*scale+.5));ww=max(1,math.floor(shape[1]*scale+.5))
        if a['transform']['rotation']in [90,270]:hh,ww=ww,hh
        for origin,alpha in [(cp,ca),(sp,sa)]:
            x,y=origin;check(0<=x and 0<=y and x+ww<=96 and y+hh<=96,'entire same transformed rectangle inside96')
            coords=np.argwhere(alpha);check(bool(len(coords))and all(y<=yy<y+hh and x<=xx<x+ww for yy,xx in coords),'alpha support inside declared origin rectangle')
        values=compare_pair(before[a['case_position']],cover,sham,ca,sa,valid[a['case_position']],case,delta)
        p=a['proof'];check(p['feature_changes']==values[0]and p['cover_changed_RGB_pixels']==values[1]and p['sham_changed_RGB_pixels']==values[2]and p['pasted_alpha_pixels']==values[3],'all saved actual pixel proof scalars independently exact')
        check(all(p[k]is True for k in ['actual_change_PASS','obligations_opaque','SHAM_feature_RGB_unchanged','outside_alpha_exact','matched_positive_alpha_mass'])and p['training_admission']is p['new_native_E_label']is False,'all proof flags no admission')
        successes.append(a);percase[a['case_position']]+=1;objectcounts[a['object_id']]+=1;directions[tuple((d>0)-(d<0)for d in delta)]+=1
        allproofs.append({'attempt_id':a['attempt_id'],'feature_changes':values[0],'cover_changed_RGB_pixels':values[1],'sham_changed_RGB_pixels':values[2],'alpha_mass':values[3],'translation_xy':list(delta)})
    check(collections.Counter(a['status']for a in attempts)==comp['statuses']=={'CONSTRUCTED_RAW96_CONTROL_NO_LABEL_ADMISSION':18,'GEOMETRY_EXHAUSTED':62},'independent actual18/62 census')
    for i,card in enumerate(preview['cards']):
        pool=[a for a in successes if a['case_position']==i];chosen=min(pool,key=lambda a:a['assignment_hash'])if pool else None
        check(card['chosen_attempt_id']==(chosen['attempt_id']if chosen else None)and card['pair_outputs']==(chosen['outputs']if chosen else None),'fixed8 minimum assignment hash among constructed or factual failure')
        check(card['all10_attempts_retained']is True,'all ten donor attempts remain')
    for b in preview['pngs']:checked(b,False)
    check(len(preview['pngs'])==96 and len({b['path']for b in preview['pngs']})==96,'all96 output PNG encoded bindings')
    check(sum(c['chosen_attempt_id']is not None for c in preview['cards'])==6,'fixed8 sixpairs twofailure')
    rootvis={'path':str(V/'ROOT_ACTUAL_FIXED8_RAW96_CONTROL_VISUAL_QA.json'),'sha256':'c12cc6ce48dec3953eacca2b7767a7dd004be49bf3bfcd9f63887811281b6ca4'};checked(rootvis)
    for k in ['PORTABLE_GALLERY_KO.md','index.html']:checked(comp['outputs'][k],False)
    report=dict(completed=True,PASS=True,code=bind(__file__),actual_source_complete=COMP,selection=comp['selection'],actual_entry=comp['entry'],root_actual_fixed8_visual_QA=rootvis,
        attempts=80,constructed=18,geometry_exhausted=62,original_cases=8,source_frames=len({c['frame_key']for c in cases}),fixed_preview_constructed=6,fixed_preview_factual_failures=2,
        constructed_by_case={str(i):percase[i]for i in range(8)},constructed_by_object_id=dict(objectcounts),translation_sign_patterns={str(k):v for k,v in directions.items()},
        exact_original_meta_ROI_qL_native_U=True,declared_original16_encoded_SHA_matched_without_follow=True,all18_pixel_proofs_independent=True,
        original26_payloads_reopened_or_hashed_or_statted=0,output_PNGs_decoded=88,output_PNGs_encoded_SHA_checked=96,new_tool_visual_views=0,producer_import_or_rerun=0,
        no_new_E_or_fit_admission=True,new_model_eval_GT_or_protected_source_array_reads=0,assertions=sum(N.values()),checks=dict(N),proof_rows=allproofs,
        limits=['Independent audit compares saved outputs and source declared SHA metadata only; original26 payloads/donor textures are not reopened. It verifies positive opaque RGB pattern agreement between COVER/SHAM, not fresh donor provenance replay.',
          '62 exhausted results are checked against their recorded all28 candidate counts/order; exhaustive infeasibility is not recalculated from the unopened original donor transforms.',
          'This is post-sampler RAW96 2D weak-feature control evidence. No physical scene/native E gold, new labels, semantic adjudication, training or improvement claim. Root separately toolviewed fixed8.'])
    put(H/'REVIEW.json',report)
    with (H/'KO.md').open('x')as f:f.write('''# V207 저장 RAW96 control 독립 기술 검산\n\n80시도 전체, 18구성/62 geometry exhausted, 고정8 표시의6구성+2factual failure를 보존했다. 원 case q/L/native/U와 두 ROI, metadata assignment hash·transform순서·고정표시 선택을 대조했다. 결과 BEFORE/VALID16 PNG의 encoded SHA는 선언된 원본 binding과 같으며 원본26 payload는 재열람하지 않았다.\n\n18쌍의 저장 RGB/alpha를 실제 decode해 두 inclusive ROI의 정확한 픽셀합집합 전체 COVER opacity·SHAM alpha0, 각 ROI 실제 COVER RGB변화, SHAM feature RGB불변, 밖의 RGB불변, alpha 전체 validity·동일이동·동일질량 및 opaque RGB pattern 동일이동을 독립 검사했다. 18쌍 모두 저장 proof scalar와 일치했다. 88개 출력96 PNG decode/96개 출력 PNG encoded SHA검사를 했고 tool visual view는 새로 하지 않았다.\n\n구성18은 object ID1/7/21/9에 한정되며 세 원 프레임 자료다. 실패62는 원기록의28후보 exhaustion을 확인한 것이고, 미열람 donor 원형으로 전체 infeasibility를 다시 계산했다고 주장하지 않는다. 원본 donor provenance/변환 replay도 이번 검사의 범위 밖이다. Root의 별도 fixed8 실제 시각 영수증을 바인딩했다. 새 E·학습 허가·물리 native gold·성능 개선·모델 원인 판단은 없다.\n''')
    print(json.dumps(bind(H/'REVIEW.json')))

if __name__=='__main__':
    try:main()
    except BaseException as e:
        put(H/'FAILED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),automatic_retry=False));raise
