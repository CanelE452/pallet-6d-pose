"""Independent saved weak-observation join arithmetic; no producer/helper import.

Only explicit V216 text bindings and original declared V215 VALID96 PNGs are
read. Rectangle unions are independently expanded to Python coordinate sets.
No source RGB, overlay view, arrays, GT, model, score or semantic relabeling.
"""
from pathlib import Path
from collections import Counter
from html.parser import HTMLParser
import csv, datetime, hashlib, io, json, os, re, stat, traceback

V=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/query_visibility_public_train_part_query_support_contract_20261025_v216')
OLD=V.parent/'query_visibility_public_train_visible_part_coverage_expansion_20261025_v215'
J=V/'joined_part_query_support_v2'
O=V/'independent_saved_join_review_c'
PROTO=O/'PROTOCOL_BEFORE_SAVED_JOIN_RECOMPUTATION.json'
COMPLETE_SHA='5389f830f72c92f8269285bdf7d8076bd09d9bcb450bd475c27e04430eac5bb4'
AXES=('instance','part','witness','addressability','query_support','visibility')
READS=[];VALID_READS=[]

def need(v,m):
    if not v: raise AssertionError(m)

def safe(p,valid=False,link=False):
    s=str(p);need('v112' not in s.casefold(),'Forbidden lexical V112 before filesystem')
    p=Path(s);need(p.is_absolute() and '..' not in p.parts,'Canonical absolute owned path')
    need(V in p.parents or (OLD in p.parents and (valid or link)),'Bound own V216 or original valid96 only')
    if not link:
        if valid:need(OLD in p.parents and re.fullmatch(r'NATIVE_[0-7]_VALID96\.png',p.name),'No RGB/overlay PNG body')
        else:need(p.suffix in ('.json','.csv','.md','.html','.py'),'Text only, no arrays/weights/image body')
    return p

def parentfd(p,**kw):
    p=safe(p,**kw);fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        return p,fd
    except BaseException:os.close(fd);raise

def rb(p,valid=False):
    p,fd=parentfd(p,valid=valid)
    try:n=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    try:
        need(stat.S_ISREG(os.fstat(n).st_mode),'Regular file')
        with os.fdopen(n,'rb') as f:data=f.read()
    except BaseException:
        try:os.close(n)
        except OSError:pass
        raise
    (VALID_READS if valid else READS).append(str(p));return data

def sha(b):return hashlib.sha256(b).hexdigest()
def ident(b):
    need(isinstance(b,dict) and set(b) in ({'path','sha256'},{'path','sha256','size_bytes'}),'Binding keys')
    need(isinstance(b['path'],str) and re.fullmatch('[0-9a-f]{64}',b['sha256']),'Binding types')
    if 'size_bytes' in b:need(type(b['size_bytes']) is int and b['size_bytes']>=0,'Optional exact integer size')
    need('v112' not in b['path'].casefold(),'Binding forbidden before filesystem')
    return b['path'],b['sha256']
def same(a,b):return ident(a)==ident(b)
def bound(b,valid=False):
    ident(b);data=rb(b['path'],valid=valid);need(sha(data)==b['sha256'],'Bound SHA '+b['path'])
    if 'size_bytes' in b:need(len(data)==b['size_bytes'],'Optional bound byte size')
    return data
def js(b):return json.loads(bound(b))
def binding(p):b=rb(p);return dict(path=str(p),sha256=sha(b),size_bytes=len(b))
def put(name,x):
    p=O/name
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')
    return binding(p)
def rect(r):
    if r is None:return set()
    need(type(r) is list and len(r)==4 and all(type(x) is int and 0<=x<96 for x in r),'Inclusive integer ROI')
    x0,y0,x1,y1=r;need(x0<=x1 and y0<=y1,'Ordered ROI')
    return {(x,y) for y in range(y0,y1+1) for x in range(x0,x1+1)}
def box(rs):
    rr=[r for r in rs if r is not None]
    return None if not rr else [min(r[0] for r in rr),min(r[1] for r in rr),max(r[2] for r in rr),max(r[3] for r in rr)]
def maskbytes(s):return bytes(int((x,y) in s) for y in range(96) for x in range(96))
def pos_text(v):return isinstance(v,str) and bool(v.strip())

def validate_row(r,t,rid,viewed):
    need(set(r)==set(t) and r['schema_version']==t['schema_version'],'Exact observation schema, no hidden target fields')
    need(r['reader_id']==rid and r['observation_complete'] is True and r['training_admission'] is False,'Actual row identity/no admission')
    need(r['input']==t['input'] and r['display_bindings']==t['display_bindings'],'Original inputs/display exact')
    m=r['input'];need(m['role']=='TRAIN' and m['internal_source_role'] in ('SOURCE_FIT','SOURCE_DEV'),'Separate original/internal roles')
    own={ident(b) for b in r['actually_viewed_image_bindings']};d=t['display_bindings']
    allowed={ident(d['contact']),ident(d['clean96'])}|{ident(b) for b in d['full_context']}
    need(own and own<=allowed and all((m['position'],b) in viewed for b in own),'Actual same-frame view receipt')
    need(ident(d['contact']) in own or (ident(d['clean96']) in own and bool(own & {ident(b) for b in d['full_context']})),'Crop plus full context')
    if not m['original_query_available']:
        need(all(r[a]['state' if a in ('query_support','visibility') else 'status']=='UNKNOWN' for a in AXES),'Original U all UNKNOWN')
        need(r['witness']['roi'] is None and r['visibility']['evidence_roi'] is None,'Original U no ROI');return
    need(r['instance']['status'] in ('YES','UNKNOWN') and r['part']['status'] in ('YES','UNKNOWN'),'Instance/part enum')
    w=r['witness'];a=r['addressability'];v=r['visibility'];q=r['query_support']
    need(w['status'] in ('VISIBLE','UNKNOWN') and a['status'] in ('QUERY_NATIVE_SUPPORTED','EXPLICIT_ROI_ONLY','UNKNOWN'),'Witness/address enum')
    need(v['state'] in ('D','E','SELF','UNKNOWN') and q['state'] in ('ON','NEAR','OFF','UNKNOWN'),'Visibility/query enum')
    if r['instance']['status']=='YES':need(pos_text(r['instance']['evidence']),'Instance evidence')
    if r['part']['status']=='YES':need(r['instance']['status']=='YES' and pos_text(r['part']['description']) and pos_text(r['part']['evidence']),'Part evidence')
    if w['status']=='VISIBLE':need(bool(rect(w['roi'])) and w['kind']!='UNKNOWN' and pos_text(w['description']) and pos_text(w['evidence']),'Visible feature evidence')
    else:need(w['roi'] is None and w['kind']=='UNKNOWN','UNKNOWN null witness')
    if a['status']=='QUERY_NATIVE_SUPPORTED':need(r['instance']['status']==r['part']['status']=='YES' and pos_text(a['evidence']),'Positive addressability')
    if a['status']=='EXPLICIT_ROI_ONLY':need(w['status']=='VISIBLE' and pos_text(a['evidence']),'ROI-only feature')
    if v['state']!='UNKNOWN':
        need(a['status']=='QUERY_NATIVE_SUPPORTED' and pos_text(v['evidence']),'Visibility addressability')
        if v['state']=='D':need(w['status']=='VISIBLE','D witness')
        else:need(bool(rect(v['evidence_roi'])) and pos_text(v['occlusion_evidence']),'Positive E/SELF evidence')

def independent_candidate(a,c,t):
    m=t['input'];wa,wc=a['witness'],c['witness'];va,vc=a['visibility'],c['visibility']
    witness=bool(m['original_query_available'] and wa['status']==wc['status']=='VISIBLE' and wa['kind']==wc['kind'] and rect(wa['roi'])&rect(wc['roi']))
    addressed=a['addressability']['status']==c['addressability']['status']=='QUERY_NATIVE_SUPPORTED'
    state='UNKNOWN'
    if addressed and va['state']==vc['state']!='UNKNOWN':
        if va['state']=='D' and witness:state='D'
        elif va['state'] in ('E','SELF') and rect(va['evidence_roi'])&rect(vc['evidence_roi']):state=va['state']
    reasons=[]
    if not m['original_query_available']:reasons.append('ORIGINAL_INPUT_U')
    if not witness:reasons.append('NO_TWO_READER_SPATIAL_WITNESS_AGREEMENT')
    if not addressed:reasons.append('NATIVE_QUERY_ADDRESSABILITY_NOT_BOTH_SUPPORTED')
    if va['state']!=vc['state']:reasons.append('VISIBILITY_DISAGREEMENT')
    return dict(row_id=m['row_id'],component_id=m['component_id'],source_group_key=m['source_group_key'],original_role=m['role'],internal_source_role=m['internal_source_role'],original_inputU=not m['original_query_available'],query_support_agreement=a['query_support']['state'] if a['query_support']['state']==c['query_support']['state'] else 'UNKNOWN',spatial_witness_agreement_candidate=witness,witness_ROI_union=box([wa['roi'],wc['roi']]) if witness else None,witness_addressability='QUERY_NATIVE_SUPPORTED' if addressed else 'EXPLICIT_ROI_ONLY' if witness else 'UNKNOWN',native_weak_visibility_agreement_candidate=state,native_weak_DE_target_proposal={'D':0,'E':1}.get(state,-1),local_witness_visible_target_proposal=0 if witness else -1,candidate_may_enter_future_FIT_review=m['internal_source_role']=='SOURCE_FIT' and (witness or state in ('D','E')),root_semantic_evidence_check_still_required=True,reasons=reasons,native_authority='TWO_READER_SOURCE_INTENDED_PART_VISIBILITY_WEAK_NOT_NATIVE_GOLD',witness_authority='TWO_READER_SOURCE_LOCAL_PART_WITNESS_ROI_WEAK',primary_unconditioned_native_loss_from_witness_prohibited=True,training_admission=False)

def differences(a,c):
    out={}
    for axis in AXES:
        x,y=a[axis],c[axis]
        out[axis]=dict(categorical_difference=any(x[k]!=y[k] for k in ('status','state','kind') if k in x),any_field_difference=x!=y,different_fields=[k for k in x if x[k]!=y[k]])
    return out

class HTML(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.rows=[];self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ('href','src') and v)
        if tag=='tr':self.row=[]
        if tag in ('td','th'):self.cell=[]
    def handle_data(self,data):
        if self.cell is not None:self.cell.append(data)
    def handle_endtag(self,tag):
        if tag in ('td','th') and self.cell is not None:self.row.append(''.join(self.cell));self.cell=None
        if tag=='tr' and self.row is not None:self.rows.append(self.row);self.row=None

def main():
    protocol=json.loads(rb(PROTO));need(same(protocol['code'],binding(__file__)),'Pinned own code before saved recompute')
    cb=dict(path=str(J/'COMPLETE.json'),sha256=COMPLETE_SHA);complete=js(cb)
    need(complete['completed'] is True and complete['PASS'] is True and complete['root_semantic_review']=='PENDING' and complete['training_admission'] is False,'Completed candidate-only result')
    entry=js(complete['entry']);need(entry['approved_actual_two_reader_join_and_overlay'] is True and entry['root_verified_both_sealed_readers_before_join'] is True and entry['training_admission'] is False,'Actual root authority')
    scope=js(entry['root_actual_scope']);need(scope['approved_actual_two_reader_join_and_overlay'] is True,'Separate root actual scope')
    tmpl=js(complete['inputs']['template'])['rows'];split=js(complete['inputs']['split'])['rows96'];decl=js(complete['inputs']['declared_PNGs'])['images']
    declared={ident(x['binding']):x for x in decl};allowed={ident(x) for x in entry['png_input_whitelist']}
    need(len(tmpl)==len(split)==96 and len({t['input']['row_id'] for t in tmpl})==96,'Full96 original denominator')
    seals={}
    for rid in ('A','C'):
        b=entry['readers'][rid];c=js(b['complete']);qa=js(b['validator']);vr=js(b['view_receipt']);ind=js(b['independence']);bound(b['entry'])
        need(all(x['completed'] is True and x['PASS'] is True for x in (c,qa,vr,ind)),'All metadata seals PASS')
        need(c['reader_id']==vr['reader_id']==ind['reader_id']==rid and c['training_admission'] is False,'Named reader')
        for k,other in [('observations','observations'),('view_receipt','view_receipt'),('schema_QA','validator'),('entry','entry')]:need(same(c[k],b[other]),'Exact seal graph '+k)
        need(same(ind['own_observations_binding'],b['observations']) and ind['other_reader_new_observation_files_read']==0 and ind['other_reader_new_values_received']==0,'Original independence seal')
        views=set()
        for x in vr['views']:
            bid=ident(x['binding']);need(x['actually_viewed'] is True and x['tool'] and bid in declared and declared[bid]['position']==x['position'],'Exact actual view metadata chain')
            views.add((x['position'],bid))
        seals[rid]=(b,views)
    # Both metadata seals have passed before either observation rowset is opened.
    obs={}
    for rid in ('A','C'):
        b=seals[rid][0];doc=js(b['observations'])
        need(doc['completed'] is True and doc['PASS'] is True and doc['reader_id']==rid and doc['training_admission'] is False and same(doc['entry'],b['entry']) and same(doc['template'],complete['inputs']['template']) and same(doc['contract'],complete['inputs']['contract']),'Sealed original observation document')
        obs[rid]=doc['rows96']
    need(len(obs['A'])==len(obs['C'])==96,'Both full96 observations')
    artifacts={k:bound(v) for k,v in complete['artifacts'].items()}
    joined=json.loads(artifacts['ALL96_JOINED_CANDIDATES.json']);side=json.loads(artifacts['ALL96_EXACT_WITNESS_MASK_SIDECAR.json']);summary=json.loads(artifacts['SUMMARY.json']);root=json.loads(artifacts['ROOT_RESULTS_KO.json']);qa=js(complete['QA'])
    need(len(joined['rows96'])==len(side['rows96'])==96 and joined['training_admission'] is False and side['training_admission'] is False,'Full candidate/sidecar population')
    recalculated=[];validity=Counter();invalid=[]
    for i,(t,s,j,z,a,c) in enumerate(zip(tmpl,split,joined['rows96'],side['rows96'],obs['A'],obs['C'])):
        m=t['input'];need(all(m[k]==s[k] for k in m),'Original source/split metadata unchanged')
        for rid,r in [('A',a),('C',c)]:validate_row(r,t,rid,seals[rid][1])
        raw=independent_candidate(a,c,t);diff=differences(a,c)
        need(j['input']==m and j['display_bindings']==t['display_bindings'] and j['raw_helper_candidate']==raw and j['all_axis_differences']==diff,'All raw candidates/axis differences independently match')
        for rid,r in [('A',a),('C',c)]:need(j['reader_'+rid+'_axes']=={k:r[k] for k in AXES+('ambiguity_note',)},'All source axes/notes preserved')
        need(j['training_admission'] is False and j['root_semantic_same_feature_and_addressability_review']=='PENDING','No semantic admission')
        ar,cr=a['witness']['roi'],c['witness']['roi'];u=rect(ar)|rect(cr);mask=maskbytes(u)
        need(z['input']==m and z['row_id']==m['row_id'] and z['original_A_witness_ROI']==ar and z['original_C_witness_ROI']==cr,'Original ROI/input preserved')
        need(z['original_A_visibility_evidence_ROI']==a['visibility']['evidence_roi'] and z['original_C_visibility_evidence_ROI']==c['visibility']['evidence_roi'],'Visibility evidence ROI preserved')
        for rid in ('A','C'):need(same(z['whole_'+rid+'_observations'],seals[rid][0]['observations']) and same(complete['sealed_observations'][rid],seals[rid][0]['observations']),'Sealed whole observations binding')
        need(z['exact_OR_area']==len(u) and z['exact_OR_mask_sha256']==sha(mask) and z['legacy_helper_witness_union_bbox_DISPLAY_ONLY']==raw['witness_ROI_union'] and z['all_available_ROIs_bbox_DISPLAY_ONLY']==box([ar,cr]),'Independent set-based OR SHA/area and display bbox')
        need(z['both_original_witness_ROIs_present']==(ar is not None and cr is not None) and z['bbox_is_never_alpha_or_proof_mask'] is True and z['training_admission'] is False and z['root_semantic_review_pending'] is True,'Sidecar authority')
        need(same(z['original_valid96'],s['valid96']) and same(z['original_clean96'],t['display_bindings']['clean96']),'Original image metadata bindings')
        if not u:
            validity['noROI']+=1;need(z['candidate_union_all_valid96'] is None and z['valid96_check_status']=='NO_WITNESS_ROI_TO_CHECK' and z['pixel_proof_failures']==[],'NoROI UNKNOWN, no fabricated proof')
        else:
            b=s['valid96'];bid=ident(b);need(bid in declared and bid in allowed and declared[bid]['kind']=='valid96' and declared[bid]['position']==m['position'] and declared[bid]['native_slot']==m['native_slot'],'Exact permitted original valid PNG membership before read')
            data=bound(b,valid=True)
            from PIL import Image
            with Image.open(io.BytesIO(data)) as im:
                im.load();need(im.mode=='L' and im.size==(96,96),'Original valid96 only');pixels=im.tobytes()
            need(set(pixels)<={0,255},'Binary valid mask')
            bad={(x,y) for x,y in u if pixels[y*96+x]!=255};ok=not bad
            validity['valid' if ok else 'invalid']+=1
            need(z['candidate_union_all_valid96'] is ok and z['valid96_check_status']=='CHECKED_EXACT_OR_NO_ROI_REPAIR' and z['pixel_proof_failures']==([] if ok else ['EXACT_OR_TOUCHES_ORIGINAL_INVALID96']),'Exact union/validity proof status')
            if bad:invalid.append(dict(row_id=m['row_id'],native_slot=m['native_slot'],internal_source_role=m['internal_source_role'],exact_OR_area=len(u),invalid_union_pixels=len(bad),original_ROIs_preserved=True))
        recalculated.append(dict(m=m,a=a,c=c,raw=raw,diff=diff))
    need(validity=={'valid':79,'invalid':4,'noROI':13},'79 valid /4 invalid /13 noROI unchanged')
    independently={};csv_expected=[]
    for role in ('ALL','SOURCE_FIT','SOURCE_DEV'):
        rr=[x for x in recalculated if role=='ALL' or x['m']['internal_source_role']==role]
        item=dict(N=len(rr),frames=len({x['m']['position'] for x in rr}),components=len({x['m']['component_id'] for x in rr}),original_inputU=sum(not x['m']['original_query_available'] for x in rr),readers={},join={},native8={},local_witness_candidates=sum(x['raw']['spatial_witness_agreement_candidate'] for x in rr))
        for who,key in [('A','a'),('C','c')]:item['readers'][who]={ax:dict(Counter(x[key][ax]['state'] for x in rr)) for ax in ('visibility','query_support')}
        item['join']=dict(visibility_candidate=dict(Counter(x['raw']['native_weak_visibility_agreement_candidate'] for x in rr)),query_support_agreement=dict(Counter(x['raw']['query_support_agreement'] for x in rr)),categorical_axis_disagreements={ax:sum(x['diff'][ax]['categorical_difference'] for x in rr) for ax in AXES})
        def emit(native,reader,axis,state,n):csv_expected.append(dict(internal_source_role=role,native_slot=str(native),reader=reader,axis=axis,state=state,N=str(n)))
        for who in ('A','C'):
            for ax,states in item['readers'][who].items():
                for state,n in states.items():emit('ALL',who,ax,state,n)
        for ax in ('visibility_candidate','query_support_agreement'):
            for state,n in item['join'][ax].items():emit('ALL','JOIN_RAW_CANDIDATE',ax,state,n)
        for k in range(8):
            nn=[x for x in rr if x['m']['native_slot']==k]
            ni=dict(N=len(nn),original_inputU=sum(not x['m']['original_query_available'] for x in nn),local_witness_candidates=sum(x['raw']['spatial_witness_agreement_candidate'] for x in nn),join_visibility_candidate=dict(Counter(x['raw']['native_weak_visibility_agreement_candidate'] for x in nn)),join_query_support_agreement=dict(Counter(x['raw']['query_support_agreement'] for x in nn)))
            item['native8'][str(k)]=ni
            emit(k,'INPUT_METADATA','population','NATIVE_ROWS',ni['N']);emit(k,'INPUT_METADATA','population','ORIGINAL_INPUT_U',ni['original_inputU']);emit(k,'JOIN_RAW_CANDIDATE','local_witness','SPATIAL_CANDIDATE',ni['local_witness_candidates'])
            for ax,states in [('visibility_candidate',ni['join_visibility_candidate']),('query_support_agreement',ni['join_query_support_agreement'])]:
                for state,n in states.items():emit(k,'JOIN_RAW_CANDIDATE',ax,state,n)
            for who,key in [('A','a'),('C','c')]:
                for ax in ('visibility','query_support'):
                    for state,n in Counter(x[key][ax]['state'] for x in nn).items():emit(k,who,ax,state,n)
        independently[role]=item
    need(summary['counts']==independently and root['counts']==independently,'Independent all role/native/readers summary')
    actual_csv=list(csv.DictReader(io.StringIO(artifacts['SOURCE_COUNTS_BY_ROLE_NATIVE_READER.csv'].decode())))
    canonical=lambda row:tuple(sorted(row.items()))
    need(Counter(map(canonical,actual_csv))==Counter(map(canonical,csv_expected)),'Entire role/native/readers CSV')
    dcsv=list(csv.DictReader(io.StringIO(artifacts['ALL96_AXIS_DIFFERENCES.csv'].decode())));need(len(dcsv)==96,'Differences CSV96')
    for d,x in zip(dcsv,recalculated):
        need(d['row_id']==x['m']['row_id'] and d['internal_source_role']==x['m']['internal_source_role'] and int(d['native_slot'])==x['m']['native_slot'] and all(json.loads(d[a])==x['diff'][a] for a in AXES),'All axis differences CSV')
    need(independently['ALL']['join']['visibility_candidate']=={'D':11,'UNKNOWN':85} and independently['SOURCE_FIT']['join']['visibility_candidate'].get('D',0)==9 and independently['SOURCE_DEV']['join']['visibility_candidate'].get('D',0)==2,'D11=FIT9+DEV2')
    need([independently[k]['local_witness_candidates'] for k in ('ALL','SOURCE_FIT','SOURCE_DEV')]==[62,40,22],'Local witness62=40+22')
    need(independently['ALL']['join']['query_support_agreement']=={'ON':24,'NEAR':17,'OFF':11,'UNKNOWN':44},'Query independent 24/17/11/44')
    need(root['training_admission'] is False and root['root_semantic_samepart_and_addressability_review']=='PENDING' and root['DEV_all_future_loss_pool_normalization_exclusion'] is True and complete['DEV_future_allbranch_fit_exclusion'] is True and root['raw_candidates_not_semantic_gold'] is True and root['fully_solved'] is False,'No semantic/DEV admission')
    md=artifacts['ROOT_REPORT_KO.md'].decode();need(artifacts['ROOT_REPORT_KO.md']==artifacts['PORTABLE_REPORT_KO.md'],'MD byte equality')
    hp=HTML();hp.feed(artifacts['ROOT_READOUT_KO.html'].decode())
    for role,s in independently.items():
        for who in ('A','C','JOIN_RAW_CANDIDATE'):
            vv=s['join']['visibility_candidate'] if who=='JOIN_RAW_CANDIDATE' else s['readers'][who]['visibility'];qq=s['join']['query_support_agreement'] if who=='JOIN_RAW_CANDIDATE' else s['readers'][who]['query_support']
            row=[role,who,' / '.join(str(vv.get(k,0)) for k in ('D','E','SELF','UNKNOWN')),' / '.join(str(qq.get(k,0)) for k in ('ON','NEAR','OFF','UNKNOWN'))]
            need('|'+'|'.join(row)+'|' in md and row in hp.rows,'Report MD/HTML categorical table values')
        need('|'+role+'|'+str(s['N'])+'|'+str(s['original_inputU'])+'|'+str(s['local_witness_candidates'])+'|' in md,'Report population table')
    linkcounts={}
    for name in ('ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','PORTABLE_GALLERY_KO.md','ROOT_READOUT_KO.html'):
        text=artifacts[name].decode()
        if name.endswith('.html'):parser=HTML();parser.feed(text);links=parser.links
        else:links=re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text)
        for link in links:
            need(not link.startswith(('/','file:','http:','https:')) and '://' not in link,'Relative local link')
            target=Path(os.path.normpath(str(J/link)));need(target.suffix not in ('.npz','.npy','.pt','.pth','.bin','.jpg','.jpeg'),'No original binary link follow')
            target,fd=parentfd(target,link=True)
            try:st=os.stat(target.name,dir_fd=fd,follow_symlinks=False);need(stat.S_ISREG(st.st_mode),'Relative regular target, no symlink')
            finally:os.close(fd)
        linkcounts[name]=len(links)
    need(qa['completed'] is True and qa['PASS'] is True and qa['training_admission'] is False and qa['portable_link_counts']==linkcounts,'Saved QA link counts')
    counts=dict(rows96=96,original_inputU=11,SOURCE_FIT64_DEV32=True,D_agreement=11,FIT_D9_DEV_D2=True,local_witness_agreement62=True,query_agreement=independently['ALL']['join']['query_support_agreement'],validity=dict(validity),valid96_PNGs_read=len(VALID_READS),source_RGB_or_overlay_views=0)
    calc=put('INDEPENDENT_RECOMPUTATION.json',dict(completed=True,PASS=True,counts=counts,all_role_native_reader_counts=independently,invalid_union_rows=invalid,link_counts=linkcounts,source_observation_bindings=complete['sealed_observations'],training_admission=False))
    receipt=put('ACTUAL_CHECKER_RECEIPT.json',dict(completed=True,PASS=True,code=binding(__file__),protocol=binding(PROTO),join_complete=cb,join_entry=complete['entry'],recomputation=calc,text_reads=READS,exact_valid96_PNG_reads=VALID_READS,source_RGB_overlay_GT_array_model_score_or_Git_reads=0,producer_import_or_rerun=0,source_file_modifications=0))
    review=put('REVIEW.json',dict(completed=True,PASS=True,join_complete=cb,evaluation_complete=cb,code=binding(__file__),protocol=binding(PROTO),actual_checker_receipt=receipt,independent_recomputation=calc,counts=counts,links=linkcounts,checks=dict(both_reader_seals_before_rowsets=True,original96_order_qL_native_U_flags_roles=True,all_raw_candidates_axis_disagreements_and_whole_bindings=True,both_original_ROIs_exact_set_OR_SHA_area=True,all_CSV_JSON_MD_HTML_table_values=True,invalid4_preserved_without_ROI_change=True,root_semantic_review_PENDING_and_all_training_false=True,DEV_future_allbranch_exclusion=True),limits_KO=['실제 저장된 관찰/metadata와 valid96 마스크의 수학·바이트 검산이다. 두 관찰자의 semantic 동일 부위 또는 3D/native gold를 새로 인증하지 않았다.','4 invalid ROI union은 실패 상태 그대로 보존했다. 79 valid는 유효 crop 영역 안에 있다는 뜻이며 cover/sham alpha 또는 같은 semantic 부위 proof가 아니다.','본 QA에서 RGB/새 overlay toolview와 HTML browser render는 0이다. 기존 C 독립 관찰과 root의 별도 시각 QA 권한을 구분한다.','SOURCE_DEV는 미래 모든 fit branch에서 제외할 정책이며 새 blind/HOLD 증거가 아니다. 현재 training admission은 전부 false다.'],own_review_frozen_no_further_writes=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(completed=True,PASS=True,REVIEW=review,receipt=receipt,recomputation=calc,counts=counts),ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except BaseException:
        put('FAILURE_PRESERVED_NO_AUTOMATIC_RETRY.json',dict(completed=False,PASS=False,traceback=traceback.format_exc(),source_file_modifications=0,automatic_retry=False));raise
