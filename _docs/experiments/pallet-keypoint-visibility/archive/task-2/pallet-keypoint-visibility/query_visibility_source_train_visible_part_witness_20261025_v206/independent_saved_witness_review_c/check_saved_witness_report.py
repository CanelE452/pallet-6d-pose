"""Independent saved-rectangle/metadata/report checks. Stdlib only; no producer import or PNG decode."""
from pathlib import Path
import ast, collections, csv, hashlib, html, io, json, os, re, stat, statistics
from html.parser import HTMLParser

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
PRIOR=P/'query_visibility_public_train_native_visibility_annotation_pilot_20261025_v205'
COMP={'path':str(V/'joint_witness_display_v1/COMPLETE.json'),'sha256':'01afd28f4ffa32dbb74f968b8045eabdc597acd9f8c6c6b131df536f5709a54c'}
VIS={'path':str(V/'ROOT_ACTUAL_FIXED8_WITNESS_VISUAL_QA.json'),'sha256':'259e61bbea9379cee7485216f02610548973a97bb2c7bf50fbf732e606e2426a'}
BUILD={'path':str(V/'ROOT_REPORT_BUILD_COMPLETE.json'),'sha256':'deb0b0505fe4b6816c3d343c5d65a280ab3df182de0703cd2b03e9288293f8c8'}
EDIT={'visible_intended_part_identified_YES_UNKNOWN','witness_ROI_raw96_xyxy_or_null','visible_feature_description','positive_local_visual_evidence','ambiguity_note','actually_viewed_image_bindings'}
STATE='visible_intended_part_identified_YES_UNKNOWN';ROI='witness_ROI_raw96_xyxy_or_null'
COUNTS=collections.Counter()

def need(x,why):
    assert x,why
    COUNTS[why]+=1
def cleanpath(p):
    p=Path(os.path.normpath(p))
    need(p.is_absolute() and all('v112'not in x.casefold()for x in p.parts),'pre-filesystem V112/name guard')
    need(V in p.parents or PRIOR in p.parents,'bounded phase/prior context path')
    need(not {'.npz','.npy','.pt','.pth','.ckpt','.safetensors','.bin'}.intersection(s.lower()for s in p.suffixes),'no source array/weight/header access')
    return p
def opened(p):
    p=cleanpath(p);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for s in p.parts[1:-1]:
            n=os.open(s,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=n
        n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(n).st_mode),'regular file no symlink')
    return os.fdopen(n,'rb')
def rb(p):
    with opened(p)as f:return f.read()
def bind(p):return {'path':str(p),'sha256':hashlib.sha256(rb(p)).hexdigest()}
def checked(b):
    z=rb(b['path']);need(hashlib.sha256(z).hexdigest()==b['sha256'],'exact bound SHA')
    return json.loads(z)
def put(p,x):
    with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')

def pixels(box):
    if box is None:return None
    need(isinstance(box,list)and len(box)==4 and all(type(v)is int for v in box),'inclusive integer rectangle')
    l,t,r,b=box;need(0<=l<=r<=95 and 0<=t<=b<=95,'raw96 rectangle boundary')
    return {(x,y)for x in range(l,r+1)for y in range(t,b+1)}
def geometry(a,c):
    aa=pixels(a);cc=pixels(c)
    if aa is None or cc is None:
        return dict(category='BOTH_UNKNOWN'if aa is cc is None else'ONE_UNKNOWN',area_A=None if aa is None else len(aa),area_C=None if cc is None else len(cc),intersection_area=None,union_area=None,IoU=None,enclosing_union_bbox=None)
    inter=aa&cc;union=aa|cc
    return dict(category='BOTH_YES_OVERLAP'if inter else'BOTH_YES_DISJOINT',area_A=len(aa),area_C=len(cc),intersection_area=len(inter),union_area=len(union),IoU=len(inter)/len(union),enclosing_union_bbox=[min(x for x,y in union),min(y for x,y in union),max(x for x,y in union),max(y for x,y in union)])

class Parser(HTMLParser):
    def __init__(self):super().__init__();self.refs=[];self.h2=[];self.tag=None;self.buf=[];self.paragraphs=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in {'href','src'}:self.refs.append(v)
        if tag in {'h2','p'}:self.tag=tag;self.buf=[]
    def handle_data(self,data):
        if self.tag:self.buf.append(data)
    def handle_endtag(self,tag):
        if tag==self.tag:
            (self.h2 if tag=='h2'else self.paragraphs).append(''.join(self.buf));self.tag=None

def main():
    need(geometry([0,0,0,0],[0,0,0,0])['IoU']==1,'independent one-pixel inclusive fixture')
    need(geometry(None,[1,1,2,2])['IoU']is None and geometry(None,None)['category']=='BOTH_UNKNOWN','UNKNOWN no imputation fixture')
    need(geometry([0,0,0,0],[95,95,95,95])['category']=='BOTH_YES_DISJOINT','disjoint fixture no threshold')
    comp=checked(COMP);vis=checked(VIS);build=checked(BUILD)
    need(all(x['completed']and x['PASS']for x in [comp,vis,build]),'all closed artifacts PASS')
    selection=checked(comp['inputs']['selection']);cases=selection['source_cases8'];rule=checked(comp['inputs']['rule'])
    need(rule['all8_fixed_source_cases_retained']and rule['no_automatic_externalE_or_training_admission'],'prospective fixed8 no admission')
    entry=checked(comp['entry']);need(entry['observers']==comp['observer_seals'],'exact actual seal bindings')
    docs={};views={};templates={};seals={}
    # Both completed seals and independence-zero receipts checked before A/C new values.
    for who in ['A','C']:
        s=comp['observer_seals'][who];seals[who]=checked(s['complete']);validation=checked(s['validation']);ind=checked(s['independence'])
        need(seals[who]['completed']and seals[who]['PASS']and validation['completed']and validation['PASS'],'both seals+validators closed')
        need(ind[s['independence_zero_field']]==0,'independent new other ROI reads0 at own seal')
        need(seals[who]['observations']==s['observations'],'observations bound through own COMPLETE')
    for who in ['A','C']:
        docs[who]=checked(comp['observer_seals'][who]['observations']);templates[who]=checked(comp['inputs']['observer_'+who+'_template'])['rows']
        v=checked(seals[who]['actual_viewed_images'if who=='A'else'actual_views'])
        views[who]=[{k:r[k]for k in ['path','sha256']}for r in v['actual_views']]if who=='A'else[r['binding']for r in v['views']]
    cguard=checked(seals['C']['identity_view_precondition'])
    need(cguard['completed']and cguard['PASS']and cguard['selection']==comp['inputs']['selection']and cguard['template']==comp['inputs']['observer_C_template'],'C separate identity/view guard binding')
    need(all(cguard[k]is True for k in ['rows8_original_order_ID_qL_native_U_exact','all_prior_flags_unchanged','training_admission_false_exact','eachcase_sameCLEAN96_plus_sameframe_fullcontext_actually_toolviewed','all_viewbindings_in_root_whitelist']),'C separate identity/view guard flags')
    joined=checked(comp['joined']);display=checked(comp['display_manifest']);csv_rows=list(csv.DictReader(io.StringIO(rb(comp['CSV']['path']).decode())))
    need(len(cases)==len(joined['rows'])==len(csv_rows)==len(display['cards'])==8,'all fixed8 denominator')
    ids=[c['original_slot']['row_id']for c in cases]
    need(len(set(ids))==8 and len({c['original_slot']['frame_key']for c in cases})==3,'fixed8/3sourceframes')
    for who in ['A','C']:need([r['row_id']for r in docs[who]['rows']]==ids,'observer original order exact')
    calculated=[];imagechecks=0
    for i,(case,j,cr,card,vr)in enumerate(zip(cases,joined['rows'],csv_rows,display['cards'],vis['rows'])):
        s=case['original_slot'];f=case['original_frame_inputs'];allowed=[s['clean96'],s['valid96'],f['fulloriginal'],f['overlay'],f['all8contact']]
        need(j['row_id']==cr['row_id']==card['row_id']==vr['row_id']==ids[i],'joined/CSV/display/root row join')
        for who in ['A','C']:
            r=docs[who]['rows'][i];t=templates[who][i]
            need(set(r)==set(t)and all(r[k]==t[k]for k in t if k not in EDIT),'all original noneditable metadata exact')
            orig=r['original_input']if who=='A'else dict(frame_key=r['frame_key'],original_frame_index=r['original_frame_index'],native_slot=r['native_slot'],query_xy=r['original_query_xy'],L=r['original_L'],original_query_available=r['original_query_available'])
            need(all(orig[k]==s[k]for k in orig),'original qL/native/U/meta exact')
            need(r['training_admission']is False,'observer fitfalse unchanged')
            actual=r['actually_viewed_image_bindings']
            need(s['clean96']in actual and any(f[k]in actual for k in ['fulloriginal','overlay'])and all(b in allowed and b in views[who]for b in actual),'samecase CLEAN96+sameframe actually viewed exact')
            need(r[STATE]in {'YES','UNKNOWN'}and (r[ROI]is None if r[STATE]=='UNKNOWN'else bool(r['positive_local_visual_evidence'].strip())),'actual state/UNKNOWN null/positive evidence')
            need(j['state_'+who]==r[STATE]and j['ROI_'+who]==r[ROI],'both original ROI values unchanged')
            need(j['feature_'+who]==r['visible_feature_description']and j['positive_evidence_'+who]==r['positive_local_visual_evidence']and j['ambiguity_'+who]==r['ambiguity_note'],'original visual evidence unchanged')
        need(j['original_query_xy']==s['query_xy']and j['original_L']==s['L']and j['native_slot']==s['native_slot']and j['original_query_available']==s['original_query_available'],'joined original qL/native/U exact')
        g=geometry(j['ROI_A'],j['ROI_C']);need(all(j[k]==v for k,v in g.items()),'independent enumerated pixelset geometry exact');calculated.append(g)
        need(all(cr[k]==str(v)for k,v in j.items()),'every CSV cell exact JSON value')
        for who in ['A','C']:need(vr['ROI_valid_pixel_fractions'][who]==1.0,'root saved ROI valid fraction1.0')
        need(vr['observer_extents_preserved_not_adjusted']and vr['weak_corresponding_local_visible_feature_complex_supported']and vr['new_E_or_training_admission']is False and vr['exact_physical_native_vertex_or_complete_part_segmentation_certified']is False,'root qualitative scope retains extents/weak authority')
        need(vr['actually_viewed_card']==card['images']['card'],'root exact fixed card actualviewbinding')
        for kind in ['clean96','valid96']:
            a=rb(s[kind]['path']);b=rb(card['images'][kind]['path']);need(a==b and hashlib.sha256(a).hexdigest()==s[kind]['sha256']==card['images'][kind]['sha256'],'original/copy encoded PNG bytes exact');imagechecks+=1
        need(bind(card['images']['card']['path'])==card['images']['card'],'generated card hash exact')
        need(j['training_admission']is False and j['corrected_keypoint_or_new_E_label']is False,'no corrected point/new E/fit')
    need(vis['actual_toolview_fixed_cards']==8 and vis['actual_unique_toolview_images']==11 and len(vis['actual_toolview_original_q8_contexts'])==3,'root actual8+3=11 claimed views')
    need(vis['actual_toolview_original_q8_contexts']==[next(c['original_frame_inputs']['overlay']for c in cases if c['original_slot']['frame_key']==fk)for fk in dict.fromkeys(c['original_slot']['frame_key']for c in cases)],'root exact3 original q8 contexts')
    for b in vis['actual_toolview_original_q8_contexts']:need(bind(b['path'])==b,'root q8context encoded SHA exact')
    need(seals['A']['actual_unique_generated_PNG_views']==len(views['A'])==11 and seals['C']['new_actual_generated_PNG_views']==len(views['C'])==25,'A11/C25 actualview receipts exact')
    # Static source inspection only. No import or execution of join/report producers.
    need(bind(comp['code']['path'])==comp['code']and bind(comp['protocol']['path'])==comp['protocol'],'frozen join code/protocol exact')
    need(bind(build['code']['path'])==build['code'],'root report producer code exact');ast.parse(rb(comp['code']['path']).decode());ast.parse(rb(build['code']['path']).decode())
    for b in build['outputs']:need(bind(b['path'])==b,'root report output SHA exact')
    result=json.loads(rb(V/'ROOT_RESULTS_KO.json'));ious=[g['IoU']for g in calculated]
    numbers={'IoU_min':min(ious),'IoU_median':statistics.median(ious),'IoU_max':max(ious)}
    need(all(result['metrics'][k]==v for k,v in numbers.items()),'root exact IoU min median max')
    need(result['fixed_source_cases']==8 and result['original_source_frames']==3 and result['metrics']['categories']=={'BOTH_YES_OVERLAP':8}and result['metrics']['same_local_feature_complex_weak_support']==8 and result['metrics']['all_A_C_ROI_pixels_valid']is True,'root complete fixed source counters')
    need(result['new_external_E_model_fit_forward_admission_or_query_change']is False and result['physical_native_gold_or_model_accuracy_claim']is False and result['goal_solved']is False,'root no new label/fit/success claim')
    md=rb(V/'PORTABLE_REPORT_KO.md').decode();need(rb(V/'ROOT_REPORT_KO.md')==md.encode(),'ROOT/PORTABLE byteexact')
    table=[r for r in md.splitlines()if r.startswith('| V153_')]
    need(len(table)==8,'rendered root table8 rows')
    for line,j,vr in zip(table,joined['rows'],vis['rows']):
        cols=[x.strip()for x in line.split('|')[1:-1]]
        need(cols==[j['row_id'],str(j['ROI_A']),str(j['ROI_C']),f"{j['IoU']:.6f}",vr['semantic_evidence']],'rendered root row ROI/IoU/root evidence exact')
    links=[];pages=[V/'ROOT_REPORT_KO.md',V/'PORTABLE_REPORT_KO.md',V/'ROOT_READOUT_KO.html',Path(comp['gallery']['path']),Path(comp['HTML']['path'])]
    for page in pages:
        text=rb(page).decode()
        if page.suffix=='.html':parser=Parser();parser.feed(text);refs=parser.refs
        else:refs=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
        for ref in refs:
            ref=html.unescape(ref).strip('<>');path=ref.split('#',1)[0].split('?',1)[0]
            if not path or re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:',path):continue
            target=cleanpath(page.parent/path)
            with opened(target):pass
            links.append({'page':str(page.relative_to(V)),'reference':ref,'target':str(target)})
    need(len(links)>0,'all explicit relative report/gallery links resolved')
    # Factual language was read directly: weak SOURCE evidence, geometry descriptive, no q correction/new E/fit/performance, no blind-human claim.
    limits=['C independently recalculates rectangle arithmetic and original metadata joins; C authored its own original observations and is not an independent semantic third rater.',
            'ROI valid fractions1.0 and root8card+3context actual visual semantics are checked through exact saved root receipt; no fresh PNG decode or C joint-card toolview in this saved QA.',
            'Original/copy PNG16 pairs checked as encoded bytes only. Root actual observations and extents remain unchanged; IoU is descriptive, not physical gold/accuracy or admission.']
    put(H/'REVIEW.json',{'completed':True,'PASS':True,'code':bind(__file__),'joint_complete':COMP,'root_actual_visual_QA':VIS,'root_report_build_complete':BUILD,
        'checks':dict(COUNTS),'check_assertions':sum(COUNTS.values()),'fixed_rows':8,'source_frames':3,'categories':dict(collections.Counter(g['category']for g in calculated)),
        'independent_geometry':numbers,'original_copy_encoded_PNG_pairs':imagechecks,'root_actual_images_bound':11,'A_actual_images_bound':11,'C_prior_actual_images_bound':25,
        'new_QA_image_decodes_or_toolviews':0,'relative_pages':len(pages),'relative_links_checked':len(links),'relative_links':links,
        'all_original_ROI_META_priorflags_and_fitfalse_unchanged':True,'separate_C_identity_view_guard_verified':True,'UNKNOWN_no_imputation':True,
        'new_label_or_fit_or_q_change_or_model_eval_or_sourcearray_access':False,'semantic_ROI_adjustment':False,'limits':limits})
    with (H/'KO.md').open('x')as f:f.write('''# V206 저장 결합·보고서 독립 검산\n\n고정 8행/3프레임의 양측 원 ROI, 원 q/L/native/가용성·prior flags·학습 불허와 실제 view 바인딩을 대조했다. 두 원 봉인과 독립0 영수증, C의 별도 identity/view guard를 확인했다. 원 ROI를 다시 정하거나 의미 라벨을 수정하지 않았다.\n\n직사각형 픽셀 좌표 집합을 독립 열거하여 inclusive 면적·교집합·합집합·IoU·enclosing bbox를 검산했다. 8쌍 모두 겹치며 IoU 최소/중앙/최대는 0.2644135188866799 / 0.5459574468085107 / 0.6147783251231527이다. UNKNOWN이면 수치나 union을 보충하지 않는 규칙도 순수 가상 입력으로 확인했다. 원 PNG16쌍은 encoded byte/SHA가 같고, CSV 전 셀 및 보고서의8행·숫자·상대 링크가 맞았다.\n\nRoot가 새로 본 8카드+3 q8 맥락의 정확 바인딩과 A11/C25 실제 관찰 영수증을 확인했다. 이번 저장 QA에서는 PNG를 새로 decode/view하지 않았다. ROI validity1.0 및 국소 의미 검토는 원 root receipt에 근거하며 C가 픽셀에서 재계산한 결과라고 주장하지 않는다.\n\n영역 겹침은 coarse한 visible-feature 약근거이며 물리 native gold·정확한 부품 mask·키포인트 교정·새 E·학습 허가·성능 개선이 아니다. q의 완벽한 선행 정합을 요구하지 않는 설명과 미해결 범위가 일치한다. C 자신이 원 C 관찰을 작성했으므로 독립 세 번째 의미 관찰자는 아니며, 여기서 독립인 것은 저장 수학·메타데이터·보고서 대조 경로다.\n''')
    print(json.dumps(bind(H/'REVIEW.json')))

if __name__=='__main__':main()
