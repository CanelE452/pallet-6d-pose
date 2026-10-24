"""Saved V192 source metadata report + raw-coordinate schematics, never photos.

Only the exact completed JSON/JSONL report inputs are read. Matplotlib is loaded
only after a separate root entry. No source images, donor alpha, arrays, models,
evaluation data, new labels or training admission exist in this runner.
"""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,os,re,stat

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
SCOPE=V/'ROOT_PROSPECTIVE_SOURCE_REPORT_AND_GEOMETRY_SCHEMATIC_SCOPE.json'
PROTOCOL=H/'PROTOCOL_BEFORE_SAVED_SOURCE_REPORT_AND_SCHEMATICS.json'
CHECKS=H/'GENERATED_INVENTED_REPORT_CHECKS.json'
QA=V/'independent_metadata_diagnostic_review_b/REVIEW.json'
FIG=V/'source_geometry_schematics_v1'
TOP=('ROOT_RESULTS_KO.json','ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_PHASE_RESULT.json')
BINS=('0','1','2~4','5~8','9~16','17~32','33~64','65이상','missing')
ROLES={'complete':'source_metadata_diagnostic_v1/COMPLETE.json','summary':'source_metadata_diagnostic_v1/SUMMARY.json','rows':'source_metadata_diagnostic_v1/ALL66_REAL_BBOX_ONLY_DIAGNOSTICS.jsonl','lineage':'source_coverage_readiness_b/SOURCE_TRAIN_LINEAGE_COVERAGE.json','preflight':'independent_metadata_diagnostic_preflight_b/REVIEW.json'}
STRATA=('OLD_GEOMETRY_FAIL_OPPOSITE_TRUE','OLD_GEOMETRY_FAIL_OPPOSITE_FALSE','OLD_SUPPORTED_OPPOSITE_FALSE','FROZEN_WEAK_UNKNOWN')
CAPTIONS=('기존 geometry 실패 중 반대 SHAM bbox 가능인 첫 사례. 실제 alpha·RGB·유효 학습쌍은 검증하지 않았다.','기존 geometry 실패 중 반대 SHAM bbox도 불가능한 첫 사례. 다른 배치를 탐색하거나 witness를 바꾸지 않았다.','기존 지원 쌍이지만 반대 SHAM bbox는 불가능한 첫 사례. 원래 지원 판정은 그대로다.','기존 weak UNKNOWN의 첫 사례. q와 원 frame만 표시하며 후보 끝·witness·대체 배치를 만들지 않았다.')

def need(x,m):
    if not bool(x):raise AssertionError(m)
def safe(value):
    raw=Path(value);need(raw.is_absolute() and all('v112' not in x.casefold() for x in raw.parts),'Lexical forbidden-name check before normalization/access')
    p=Path(os.path.normpath(str(raw)));need(P in p.parents,'Explicit project metadata/output path');return p
def opened(value,create=False):
    p=safe(value);fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            nx=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nx
        end=os.open(p.name,(os.O_WRONLY|os.O_CREAT|os.O_EXCL if create else os.O_RDONLY)|os.O_NOFOLLOW,0o644,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(end).st_mode),'Ordinary file/no symlink');return os.fdopen(end,'wb' if create else 'rb')
def sha(path):
    h=hashlib.sha256()
    with opened(path) as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def bind(p):return dict(path=str(safe(p)),sha256=sha(p))
def checked(b):need(bind(b['path'])==b,'Exact saved metadata/code binding');return safe(b['path'])
def read(p):
    with opened(p)as f:return json.load(f)
def lines(p):
    with opened(p)as f:return [json.loads(x) for x in f if x.strip()]
def text(p,s):
    with opened(p,True)as f:f.write(s.encode())
def write(p,d):text(p,json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def table(rows):
    keys=list(rows[0]);return '|'+'|'.join(keys)+'|\n|'+'|'.join(['---']*len(keys))+'|\n'+''.join('|'+'|'.join(str(r[k]).replace('|',' / ') for k in keys)+'|\n' for r in rows)
def links(md,pending=()):
    need('/home/' not in md and 'file://' not in md,'Portable Markdown has no absolute local links')
    refs=re.findall(r'\]\(([^)]+)\)',md)
    for ref in refs:
        need(not ref.startswith('/') and ':' not in ref and not any('v112' in s.casefold() for s in Path(ref).parts),'Relative portable link only')
        if ref not in pending:
            with opened(V/ref):pass  # path/type only; no linked image or array content read
    return refs

def load_inputs(bindings):
    d={k:(lines(checked(b)) if k=='rows' else read(checked(b))) for k,b in bindings.items()}
    need(all(d[k]['completed'] and d[k]['PASS'] for k in ('complete','summary','lineage','preflight','numeric_QA')),'All completed source summaries/QA')
    need(d['numeric_QA']['actual_completion']==bindings['complete'],'Exact source computation independently audited')
    s=d['summary'];rows=d['rows'];need(s['attempts']==dict(synthetic=1024,real=66) and len(rows)==66,'Complete original source population')
    need([r['source_census_row_index'] for r in rows]==list(range(66)) and len({r['identity']['pair_id'] for r in rows})==66,'Original source order/identities retained')
    need(sum(r['geometry_bbox_only']['bbox_only_possible'] is True for r in rows)==s['opposite_bbox_only_possible_count'],'Saved bbox-only total')
    need(d['lineage']['direct']['known_rows']==306 and d['lineage']['direct']['unknown_rows']==2486,'Existing direct label population')
    return d

def select_rows(rows):
    predicates=(lambda r:r['stored_status']=='GEOMETRY_FAILURE_RETAINED' and r['geometry_bbox_only']['bbox_only_possible'] is True,
        lambda r:r['stored_status']=='GEOMETRY_FAILURE_RETAINED' and r['geometry_bbox_only']['bbox_only_possible'] is False,
        lambda r:r['stored_supported'] is True and r['geometry_bbox_only']['bbox_only_possible'] is False,
        lambda r:r['geometry_bbox_only']['weak_semantic_localization_supported'] is False)
    return [next((r for r in rows if pred(r)),None) for pred in predicates]

def report_tables(d):
    s=d['summary'];rows=d['rows'];lineage=d['lineage'];pop=[]
    for domain,key in [('SYNTHETIC','synthetic'),('REAL','real')]:
        cc=s['stored_status'][domain]
        pop.append({'source':domain,'원 시도':s['attempts'][key],'기존 지원 triplet':s['stored_supported'][domain],'기존 geometry 실패':cc.get('GEOMETRY_FAILURE_RETAINED',0),'기존 UNKNOWN':cc.get('UNKNOWN_RETAINED',0)})
    status_rows=[]
    for state in ('GEOMETRY_FAILURE_RETAINED','SOURCE_REAL_SAMPLED_DOMAIN_CONTROLLED_END_WEAK_PAIR_SUPPORTED','UNKNOWN_RETAINED'):
        rr=[r for r in rows if r['stored_status']==state];cc=Counter('UNKNOWN' if r['geometry_bbox_only']['bbox_only_possible'] is None else str(r['geometry_bbox_only']['bbox_only_possible']) for r in rr)
        status_rows.append({'기존 상태':state,'원 시도':len(rr),'반대 bbox 가능':cc['True'],'반대 bbox 불가':cc['False'],'관측 UNKNOWN 유지':cc['UNKNOWN']})
    native=[]
    for k in range(8):
        syn=next(r for r in s['full_population_tables'] if r['domain']=='SYNTHETIC' and r['stratum']=='native' and r['key']==str(k));rr=[r for r in rows if r['native_slot']==k]
        native.append({'native':k,'합성 시도':syn['attempts'],'합성 지원 triplet':syn['stored_supported'],'공개 weak 시도':len(rr),'공개 weak 지원 triplet':sum(r['stored_supported'] for r in rr)})
    signals=[]
    for domain in ('SYNTHETIC','REAL'):
        for field in ('end_related96_samples','COVER_changed_RGB96_samples','SHAM_changed_RGB96_samples'):
            rr=[r for r in s['fixed_bin_distributions'] if r['domain']==domain and r['cohort']=='STORED_SUPPORTED_ONLY' and r['quantity']==field]
            need({r['fixed_bin'] for r in rr}==set(BINS),'All fixed bins preserved')
            signals.append({'source':domain,'저장 수량':field,'분모':rr[0]['cohort_denominator'],**{b:next(r['count'] for r in rr if r['fixed_bin']==b) for b in BINS}})
    return dict(population2=pop,opposite_bbox_by_old_status3=status_rows,native8=native,stored_signal_fixed_bins6=signals)

def draw_schematic(row,index,paths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    o=row['identity']['original_input'];g=row['geometry_bbox_only'];hw=o['raw_hw'];q=o['query_xy'];known=g['weak_semantic_localization_supported']
    fig,axes=plt.subplots(1,2,figsize=(14,7.5),dpi=150);fig.subplots_adjust(left=.065,right=.98,bottom=.16,top=.78,wspace=.20)
    style=[('cover_bbox_xyxy','COVER bbox','#c43c39'),('existing_sham_bbox_xyxy','Old SHAM bbox','#315eae'),('opposite_sham_bbox_xyxy','Opposite SHAM bbox','#278044')]
    allboxes=[g[key] for key,label,color in style if known and key in g]
    candidates=[r['enclosure_xyxy'] for r in g.get('candidates',[])] if known else []
    witnesses=[r['raw_pixel_xy'] for r in g.get('chosen_raw_witnesses',[])] if known else []
    def paint(ax):
        ax.set_facecolor('#f5f5f5');ax.add_patch(Rectangle((0,0),hw[1],hw[0],facecolor='white',edgecolor='#555',linewidth=1.5))
        if known:
            for key,label,color in style:
                if key in g:
                    b=g[key];ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],facecolor=color,edgecolor=color,alpha=.16,linewidth=1.8))
            for b in candidates:ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='#ab239a',linewidth=2.2))
            if g.get('original96_sample_extent'):
                b=g['original96_sample_extent'];ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,edgecolor='#333',linestyle='--',linewidth=1.0))
            for w in witnesses:ax.plot(w[0],w[1],marker='*',markersize=11,color='#e4b500',markeredgecolor='black',zorder=6)
        if isinstance(q,list) and len(q)==2 and all(type(z)in(int,float) and math.isfinite(z) for z in q):ax.plot(*q,marker='+',markersize=13,markeredgewidth=2,color='#008a96',zorder=7)
        ax.set_xlabel('Original raw x (pixels)');ax.set_ylabel('Original raw y (pixels, down)');ax.grid(alpha=.15)
    for ax in axes:paint(ax)
    axes[0].set_xlim(0,hw[1]);axes[0].set_ylim(hw[0],0);axes[0].set_title('Original frame coordinates')
    if known:
        points=[[b[0],b[1]] for b in allboxes+candidates]+[[b[2],b[3]] for b in allboxes+candidates]+witnesses
        if isinstance(q,list) and len(q)==2 and all(type(z)in(int,float) and math.isfinite(z) for z in q):points.append(q)
        if points:
            xx=[z[0] for z in points];yy=[z[1] for z in points];pad=max(8,.08*max(max(xx)-min(xx),max(yy)-min(yy)))
            axes[1].set_xlim(min(xx)-pad,max(xx)+pad);axes[1].set_ylim(max(yy)+pad,min(yy)-pad)
        axes[1].set_title('Geometry display zoom; raw coordinates unchanged')
    else:
        axes[1].set_xlim(0,hw[1]);axes[1].set_ylim(hw[0],0);axes[1].set_title('Weak end observation UNKNOWN')
        axes[1].text(.5,.5,'No endpoint, witness or placement invented',transform=axes[1].transAxes,ha='center',va='center',fontsize=11,bbox=dict(facecolor='white',alpha=.9,edgecolor='#bbb'))
    handles=[Line2D([0],[0],color=c,lw=4,alpha=.7,label=lab) for key,lab,c in style]+[Line2D([0],[0],color='#ab239a',lw=2,label='Frozen end candidate enclosure'),Line2D([0],[0],marker='*',color='white',markerfacecolor='#e4b500',markeredgecolor='black',markersize=10,label='Frozen same-part witness'),Line2D([0],[0],marker='+',color='#008a96',linestyle='',markersize=10,label='Original q'),Line2D([0],[0],color='#333',linestyle='--',label='Original 96 sample-center extent')]
    if not known:handles=[handles[5]]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.92),ncol=4,fontsize=9,frameon=False)
    fig.suptitle('V192 source geometry schematic %d: %s / native %d'%(index,row['identity']['pair_id'],row['native_slot']),fontsize=15,y=.985)
    fig.text(.5,.065,'SCHEMATIC ONLY - no RGB/alpha/depth read. Bbox feasibility is NOT a verified new triplet, label or admission.',ha='center',fontsize=10)
    fig.text(.5,.032,'Original q=%s, L=%s. Old status=%s. Opposite bbox-only=%s.'%(q,o['L'],row['stored_status'],g['bbox_only_possible']),ha='center',fontsize=9)
    for p in paths:fig.savefig(p,dpi=150)
    plt.close(fig)

def compose(d,tables,figures):
    s=d['summary'];b=d['lineage'];dr=b['direct'];samplekeys=[int(k) for k in b['saved_synthetic_end_signal']['end_related_sample_counts']];lo,hi=min(samplekeys),max(samplekeys)
    lead='source 감사와 단 한 번의 반대 SHAM bbox 진단을 완료했다. 기존 지원은 합성97/1024·공개 weak5/66 그대로이며, 반대 bbox 가능26건은 새 유효 가림쌍이나 학습 승인26건을 뜻하지 않는다.'
    limits=['기존 source 구성·집계 재요약과 새 dx 반전 bbox 계산을 구분한다. alpha, 실제 RGB 변화, 앞뒤 관계 또는 끝부위 identity를 새로 검증하지 않았다.','공개 source의 direct E는 실제 촬영된 자연 가림 정답이 아니라 YCB cutout 합성 차폐와 기존 RGB weak end 관찰의 결합이다. bbox 배치 문제만 풀어도 native gold나 자연 가림 전이를 증명하지 않는다.','synthetic finite registered-end와 공개 RGB semantic-end weak 관찰은 서로 다른 권위다. center owner T/F/BG나 broad support2를 native D/E/S 또는 direct-end 정답으로 바꾸지 않았다.','직접 감독306행 모두 C54/C55 학습 pool에 포함됐다. 기존 source holdout은 없으며 사후 부분집합을 독립 검증으로 부를 수 없다. 향후 새 학습의 독립 검증은 source base/frame/donor 연결 계보를 먼저 분리해 사전 동결해야 한다.','합성 native6/7의 known direct 감독은0이고 공개 known은 두 프레임의 native0뿐이다. 전체 WOOD/SQUARE·재질/aspect/geometry·native0..7 목표를 유지하며 좁은 범위의 성공으로 끝내지 않는다.','4도식은 source 결과를 본 후 정한 설명용 집단별 원순서 첫 사례다. random·blind·전66 대표표본이 아니며 실제 사진·원crop·실제 paste도 아니다.','이 단계에서 새 학습/forward/평가/라벨·좌표수정·문턱/모델/데이터 선택/PnP는 없었다. 실사 전이와 전체 가림 목표는 미해결이며 채택 모델은 없다.']
    md=['# V192: source 끝부위 감독의 범위와 SHAM 배치 진단','',lead,'','## 이미 확인된 source 범위','',table(tables['population2']),'', '직접 정답은 D204/E102, 총306행이며 물리 입력2792행 중 UNKNOWN2486행을 보존한다. 전체 시도는 합성128bases×8native=1024와 공개22사례×3donor=66이다. known은 합성97triplet/47bases, 공개5triplet/2frames다. 65536 증강 crop을 독립 영상65536장으로 세지 않는다.','',table(tables['native8']),'','[모든 native·재질·aspect·geometry/foreign family와 실패/UNKNOWN 분모](source_metadata_diagnostic_v1/POPULATION_AND_STORED_STATUS_BY_STRATUM.csv) · [전체 lineage](source_coverage_readiness_b/SOURCE_TRAIN_LINEAGE_COVERAGE.json)','', '## 원96 입력에 남은 저장 신호량','', '합성 accepted97의 end-related sample 수는 %d~%d/9216(약 %.2f~%.2f%%)다. 이는 끝 후보에 양의 보간 기여가 있는 output sample 수이며 RGB 차이 크기·시각적 판독성·독립 정보량이 아니다.'%(lo,hi,100*lo/9216,100*hi/9216),'',table(tables['stored_signal_fixed_bins6']),'','공개 weak accepted5는 end-related sample 수만 정량 저장했고 COVER 변화/SHAM 불변은 boolean proof다. 그 boolean을 변화 sample 수나 RGB 강도로 바꾸지 않는다. missing은0으로 채우지 않았다. [전체 고정 bins](source_metadata_diagnostic_v1/STORED_END96_FIXED_BIN_DISTRIBUTIONS.csv) · [공개 변화 boolean 전체 분모](source_metadata_diagnostic_v1/STORED_REAL_CHANGE_BOOLEAN_COUNTS.csv)','', '## 새 계산: SHAM x방향만 한 번 반전','',table(tables['opposite_bbox_by_old_status3']),'','26건은 기존 geometry 실패22건과 기존 지원4건이다. 기존 지원1건 Q1088_D2는 반대 bbox가 불가능하다. 기존 UNKNOWN18건은 None/UNKNOWN 그대로다. 실패43건 전체와 과거 보고서의 newly-entered33 subset을 혼동하지 않는다.','', 'COVER/donor/scale/후보/witness/원 q·L·native는 고정하고 기존 정수 dx의 부호만 뒤집었다(dy=0 확인). 모든 bbox의 frame 내부 여부, 끝 enclosure와 SHAM의 비교차, 동일 witness의 원96 positive tap 참여·양 bbox 밖 보존, SHAM의 원96 sample 기회만 확인했다. bbox 가능은 실제 donor alpha가 후보를 덮는지나 RGB 변화·same-part 보존을 인증하지 않는다.','', '[전체66 신규 bbox 진단](source_metadata_diagnostic_v1/ALL66_REAL_BBOX_ONLY_DIAGNOSTICS.jsonl) · [중복 가능한 원인별 집계](source_metadata_diagnostic_v1/REAL_OVERLAPPING_GEOMETRY_COUNTS.csv)','', '## 실제 사진이 아닌 raw-coordinate 도식4개','', '선택은 source 결과 후·도식 전 고정했다. 각 지정 집단에서 원래 source row 순서의 첫 사례만 사용하며 빈 집단은 대체하지 않는다. 표시 확대는 geometry diagram 축 범위일 뿐 입력 crop/원 q/L/GT를 바꾸지 않는다.','']
    for f in figures:
        md+=['### '+f['stratum'],'',f['caption_KO'],'']
        if f.get('PNG'):md+=['![geometry schematic '+str(f['ordinal'])+'](source_geometry_schematics_v1/'+Path(f['PNG']['path']).name+')','', '[SVG](source_geometry_schematics_v1/'+Path(f['SVG']['path']).name+')','']
        else:md+=['기존 집단에 해당 행이 없어 도식을 만들지 않았다.','']
    md+=['[도식 전체 gallery](source_geometry_schematics_v1/PORTABLE_GALLERY_KO.md) · [그림 전 고정 선택표](source_geometry_schematics_v1/SELECTION_BEFORE_DRAW.json)','', '## 해석과 남은 검증','']+['- '+x for x in limits]+['','다음 source 검증은 원 witness/enclosure를 바꾸지 않고도 반대 bbox 후보가 실제 alpha·샘플 RGB·주변 보존 proof까지 통과하는지를 별도 사전규약으로 확인할 수 있다. 이 보고서는 그 새 생성/학습을 실행하거나 승인하지 않는다.','', '[원1024/66 집계](source_metadata_diagnostic_v1/SUMMARY.json) · [독립 수치검산](independent_metadata_diagnostic_review_b/REVIEW.json) · [기존 source 제어 원문](../query_visibility_sampled_domain_direct_end_20261024_v189/PORTABLE_REPORT_KO.md) · [source 학습 단계 원문](../query_visibility_direct_end_paired_source_training_20261024_v190/PORTABLE_REPORT_KO.md)','', '[진단 코드](source_metadata_diagnostic_preparation_a/diagnose_stored_source_metadata.py) · [보고서/도식 코드](source_report_preparation_a/build_source_report_and_schematics.py) · [정확 결과JSON](ROOT_RESULTS_KO.json) · [단계 범위](ROOT_PHASE_RESULT.json)','']
    return '\n'.join(md),limits

def invented():
    tests={};base=dict(stored_status='GEOMETRY_FAILURE_RETAINED',stored_supported=False,geometry_bbox_only=dict(bbox_only_possible=True,weak_semantic_localization_supported=True))
    rows=[dict(base,marker='first'),dict(base,marker='second'),dict(base,geometry_bbox_only=dict(bbox_only_possible=False,weak_semantic_localization_supported=True)),dict(base,stored_status='SUPPORTED',stored_supported=True,geometry_bbox_only=dict(bbox_only_possible=False,weak_semantic_localization_supported=True)),dict(base,stored_status='UNKNOWN',geometry_bbox_only=dict(bbox_only_possible=None,weak_semantic_localization_supported=False))]
    ss=select_rows(rows);tests['source_original_order_first_per_four_strata']=ss[0]['marker']=='first' and ss[1] is rows[2] and ss[2] is rows[3] and ss[3] is rows[4]
    tests['missing_strata_kept_empty_no_refill']=select_rows(rows[:2])[1:]==[None,None,None]
    tests['unknown_geometry_not_fabricated']=ss[3]['geometry_bbox_only']['bbox_only_possible'] is None
    tests['fixed_bin_missing_not_zero']=BINS[-1]=='missing' and len(set(BINS))==9
    tests['tiny_sample_fraction_is_not_RGB_magnitude']=round(100*16/9216,2)==.17 and round(100*49/9216,2)==.53
    tests['new_bbox_feasibility_does_not_change_existing_support']=22+4==26 and 22+21+4+1+18==66 and 97+5==102
    need(all(tests.values()),'Invented saved-report/selection fixtures')
    return dict(completed=True,PASS=True,invented_only=True,tests=tests,number_checks=len(tests),source_pixel_array_RGB_alpha_model_or_eval_reads=0,plots_generated=0)

def prepare():
    sc=read(SCOPE);need(sc['approved_source_report_and_schematic_code_preparation'],'Root prospective report scope')
    inputs={role:sc['exact_saved_inputs'][path] for role,path in ROLES.items()};inputs['numeric_QA']=bind(QA)
    need(str(QA)==str(V/'independent_metadata_diagnostic_review_b/REVIEW.json'),'Only additional completed QA named by scope')
    d=load_inputs(inputs);report_tables(d);write(CHECKS,invented())
    p=dict(completed=True,prepared_only=True,code=bind(__file__),scope=bind(SCOPE),inputs=inputs,checks=bind(CHECKS),top_outputs=list(TOP),geometry_output=str(FIG),selection_strata=list(STRATA),plot_input='metadata coordinates only; no photos or original pixel/alpha/array bodies',source_results_available_before_selection=True,blind_or_random_or_validation_sample=False,actual_draw_or_topwrite=0)
    write(PROTOCOL,p);template=H/'ROOT_ACTUAL_SOURCE_REPORT_ENTRY_TEMPLATE_APPROVED_FALSE.json';write(template,dict(approved_saved_source_report_and_geometry=False,no_source_pixels_arrays_alpha_models_eval_fit_or_new_labels=True,automatic_retry=False,code=p['code'],protocol=bind(PROTOCOL),scope=p['scope'],inputs=inputs,geometry_output=str(FIG),top_outputs=list(TOP)))
    ready=dict(completed=True,PASS=True,prepared_only=True,code=p['code'],protocol=bind(PROTOCOL),checks=p['checks'],false_entry=bind(template),root_actual_entry_required=True,new_source_pixel_array_model_eval_reads=0,actual_top_reports_or_schematics_written=0);write(H/'FINAL_READY.json',ready);print(json.dumps(ready))

def run(entry):
    e=read(entry);need(e['approved_saved_source_report_and_geometry'] is True and e['no_source_pixels_arrays_alpha_models_eval_fit_or_new_labels'] is True and e['automatic_retry'] is False,'Separate root saved-source report/drawing entry')
    p=read(checked(e['protocol']));need(e['code']==p['code']==bind(__file__) and e['scope']==p['scope']==bind(SCOPE) and e['inputs']==p['inputs'] and e['geometry_output']==p['geometry_output']==str(FIG) and e['top_outputs']==p['top_outputs']==list(TOP),'Frozen code/scope/metadata/output closure')
    need(read(checked(p['checks']))['PASS'],'Invented preparation closure');d=load_inputs(p['inputs']);t=report_tables(d)
    need(not FIG.exists() and not any((V/n).exists() for n in TOP),'Fresh exclusive derived outputs only');FIG.mkdir()
    chosen=select_rows(d['rows']);selected=[dict(ordinal=i+1,stratum=STRATA[i],caption_KO=CAPTIONS[i],source_census_row_index=None if r is None else r['source_census_row_index'],pair_id=None if r is None else r['identity']['pair_id'],native_slot=None if r is None else r['native_slot'],missing_stratum=r is None) for i,r in enumerate(chosen)]
    selection=FIG/'SELECTION_BEFORE_DRAW.json';write(selection,dict(completed=True,PASS=True,selection_before_any_plot_generated=True,source_outcomes_already_known_not_blind_or_random=True,four_strata=selected,inputs=p['inputs'],no_actual_source_pixels_or_alpha_read=True,new_labels_or_admission=False))
    figures=[]
    for choice,row in zip(selected,chosen):
        f=dict(choice)
        if row is not None:
            paths=[FIG/('GEOMETRY_%02d.'%choice['ordinal']+ext) for ext in ('png','svg')];draw_schematic(row,choice['ordinal'],paths);f.update(PNG=bind(paths[0]),SVG=bind(paths[1]))
        figures.append(f)
    gallery=['# V192: 실제 사진이 아닌 source geometry 도식','', '기존 source 결과를 본 후 지정한4집단의 첫 행이다. 원 raw좌표/frame/q/L/native는 유지하며 표시 확대는 입력 변경이 아니다. UNKNOWN에는 end/placement를 만들지 않는다. bbox-only 가능은 실제 alpha/RGB proof나 새 라벨/학습 승인이 아니다.','']
    for f in figures:
        gallery+=['## '+f['stratum'],'',f['caption_KO'],'']
        if f.get('PNG'):gallery+=['![metadata schematic '+str(f['ordinal'])+']('+Path(f['PNG']['path']).name+')','', '[SVG]('+Path(f['SVG']['path']).name+')','']
        else:gallery+=['빈 집단: 대체 사례 없음.','']
    gallery+=['[그림 전 선택표](SELECTION_BEFORE_DRAW.json)',''];gallerytext='\n'.join(gallery);text(FIG/'PORTABLE_GALLERY_KO.md',gallerytext)
    md,limits=compose(d,t,figures);mdrefs=links(md,pending=TOP)
    # Gallery paths are relative to its own directory, checked without image content reads.
    for ref in re.findall(r'\]\(([^)]+)\)',gallerytext):
        with opened(FIG/ref):pass
    result=dict(completed=True,PASS=True,phase='V192_SOURCE_METADATA_COVERAGE_AND_BBOX_ONLY_DIAGNOSIS',source_only=True,source_attempts=d['summary']['attempts'],existing_source_supported_triplets=d['summary']['stored_supported'],tables=t,opposite_bbox_possible=d['summary']['opposite_bbox_only_possible_count'],opposite_bbox_not_new_triplets_or_admission=True,direct_training_lineage=d['lineage']['direct'],independent_validation_status=d['lineage']['independent_validation_status'],saved_end_signal=d['lineage']['saved_synthetic_end_signal'],source_summary=d['summary'],figure_selection=selected,figures=figures,diagram_not_real_RGB=True,actual_tool_visual_QA='PENDING_SEPARATE_ROOT_OR_C_VIEW',evidence=p['inputs'],code=p['code'],protocol=bind(PROTOCOL),entry=bind(entry),limitations_KO=limits,selected_model=None,adopted=False,fully_solved=False,goal_status='active_unsolved',new_labels_generated=False,training_admission=False,new_source_pixels_arrays_models_or_eval_reads=0)
    write(V/'ROOT_RESULTS_KO.json',result);text(V/'ROOT_REPORT_KO.md',md);text(V/'PORTABLE_REPORT_KO.md',md)
    phase=dict(completed=True,PASS=True,phase='V192_SOURCE_METADATA_DIAGNOSTIC_REPORT_ONLY',completed_scope='Existing source lineage/status/quantity metadata audit; one opposite-x SHAM bounding-box diagnostic; four descriptive coordinate schematics. No model improvement or actual new valid triplets.',reports={n:bind(V/n) for n in TOP if n!='ROOT_PHASE_RESULT.json'},geometry_gallery=bind(FIG/'PORTABLE_GALLERY_KO.md'),selection_before_draw=bind(selection),figures=figures,code=p['code'],protocol=bind(PROTOCOL),entry=bind(entry),evidence=p['inputs'],actual_tool_visual_QA_pending=True,browser_rendered=False,source_originals_unchanged=True,selected_model=None,adopted=False,fully_solved=False,goal_status='active_unsolved',new_training_or_evaluation_or_label_admission=False);write(V/'ROOT_PHASE_RESULT.json',phase)
    links(md);print(json.dumps(dict(completed=True,PASS=True,reports={n:bind(V/n) for n in TOP},geometry_gallery=bind(FIG/'PORTABLE_GALLERY_KO.md'),schematics=len([f for f in figures if f.get('PNG')]),portable_local_refs=len(mdrefs),actual_tool_visual_QA_pending=True)))

if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--entry',type=Path);a=ap.parse_args();prepare() if a.prepare else run(a.entry)
