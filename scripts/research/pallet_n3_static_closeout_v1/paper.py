"""One numeric source for Korean v3 Markdown copy, LaTeX fragments and cell map."""
import csv,re,json
from pathlib import Path
import numpy as np
from .compute import C,R,M,ROOT,DOC,RAW,SOURCE,write,bind

GEN=DOC/'generated_tables'
cells=[];tables={}
def fmt(v):
    if v is None:return 'x'
    if isinstance(v,str):return v
    if isinstance(v,(int,np.integer)):return str(v)
    return f'{v:.6f}' if abs(v)<.1 and v!=0 else f'{v:.3f}'
def md(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(fmt(v) for v in row)+' |\n' for row in rows)
def esc(s):return str(s).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&').replace('#',r'\#')
def table(label,headers,rows,caption,source,population,denominator,seeds='1,2,3 statistics mean',original_x=True,insert=True):
    GEN.mkdir(exist_ok=True)
    slug=label.replace(':','_');markdown=md(headers,rows)+'\n'+caption+'\n'
    (GEN/(slug+'.md')).write_text(markdown)
    tex='\\begin{table*}[t]\n\\centering\n\\caption{'+esc(caption)+'}\n\\label{'+label+'}\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{'+'l'*len(headers)+'}\n\\hline\n'+' & '.join(esc(x) for x in headers)+' \\\\\n\\hline\n'
    tex+=''.join(' & '.join(esc(fmt(x)) for x in r)+' \\\\\n' for r in rows)+'\\hline\n\\end{tabular}}\n\\end{table*}\n'
    (GEN/(slug+'.tex')).write_text(tex)
    with (GEN/(slug+'.csv')).open('w',newline='') as f:w=csv.writer(f);w.writerow(headers);w.writerows(rows)
    source=Path(source);binding=bind(source)
    for ri,row in enumerate(rows):
        for ci,value in enumerate(row):
            if ci==0:continue
            if isinstance(value,str) and value not in ('x','NA','OUT_OF_SCOPE_USER'):continue
            missing=value is None or value=='x';na=value=='NA'
            cells.append({'table_label':label,'row_index':ri,'row_identity':' / '.join(str(x) for x in row[:2] if isinstance(x,str)),
                'metric_and_unit':headers[ci],'population':population,'mode':denominator,'seed':seeds,
                'original_x_or_new_column':original_x,'value':value,'display':fmt(value),'status':'NEED_REVIEW_OR_REFERENCE' if missing else 'NOT_APPLICABLE' if na else 'INSERTED' if insert else 'READY_TO_INSERT',
                'markdown_inserted':insert,'latex_inserted':False,'latex_status':'READY_TO_INSERT_FRAGMENT; matching Korean v3 .tex source absent',
                'source_path':binding['path'],'source_sha256':binding['sha256'],'new_table_path':str(GEN/(slug+'.md')),
                'aggregation':'per-seed statistic then arithmetic mean; no prediction ensemble','raw_chain':'SOURCE_BINDINGS.json and table-specific source JSON','reason':caption})
    tables[label]=(markdown,insert)
def vals(h):return [h['matched_pooled_corner8_median_px'],h['matched_pooled_corner8_P90_px'],100*h['full_PCK10_fraction'],h.get('pose_translation_cm_median'),h.get('pose_rotation_deg_median')]
def main():
    core=C.read(DOC/'CORE_RESULTS.json')['methods'];reuse=C.read(C.DOC/'REUSE_RESULTS.json')['contracts'];square=C.read(DOC/'SQUARE_AUDIT.json');labels=C.read(DOC/'STATIC_LABEL_AUDIT.json');paired=C.read(DOC/'PAIRED_POSE_ANALYSIS.json');tail=C.read(DOC/'POSE_TRACE_AND_TAIL.json')['tail']
    names={'R0':'R0','OLD_P':'P','N0_BASE_REPLAY':'N0 replay P','N1_SYM_ONLY':'N1 symmetry','N2_DIM_ONLY':'N2 dimensions','N3_DIM_SYM':'N3 dimensions + symmetry'}
    composition=[]
    for material in ['plastic','wood']:
        counts=[labels['cross_counts'].get(material+'::'+g,0) for g in ['clean','moderate','severe','unclassified']]
        composition.append(['Rectangular '+material,*counts,sum(counts)])
    composition.append(['DEV total',29,20,79,191,319]);composition.append(['Square GREEN0918',0,0,0,119,119])
    table('tab:composition',['Group','Labeled clean','Labeled moderate','Labeled severe','Unknown','Total'],composition,
        'Counts of assigned labels, not inferred occlusion prevalence. Square119 has no linked approved occlusion labels. Zero assigned labels does not imply no actual clean/severe frames.',DOC/'STATIC_LABEL_AUDIT.json','DEV319 and GREEN0918 separately','frames per row')
    heads={k:v['mean']['headline'] for k,v in core.items()};common=['R0','OLD_P','N2_DIM_ONLY','N3_DIM_SYM']
    source=DOC/'CORE_RESULTS.json'
    table('tab:corners',['Method','Median px','P90 px','PCK10 %','E_sym'],[[names[k],*vals(heads[k])[:3],heads[k]['E_sym']] for k in common],
        'DEV319; conditional 2445 observed corners, full PCK 2499 reference corners. Mean of three seed statistics; R0 single frozen model.',source,'DEV319','319 frames / 2499 full / 2445 observed',original_x=False)
    rows=[]
    for k,h in heads.items():rows.append([names[k],h['pose_translation_cm_median'],h['pose_translation_cm_P90'],h['pose_rotation_deg_median'],h['pose_rotation_deg_P90'],h['pose_yaw_deg_median'],h['pose_yaw_deg_P90'],h['pose_IoU3D_median'],h['pose_ADDsym_AUC_full'],h.get('pose_available_frames',round(319*h['pose_coverage'])),100*h['pose_coverage']])
    table('tab:pose',['Method','T median cm','T P90 cm','R median deg','R P90 deg','Yaw median deg','Yaw P90 deg','IoU3D median','ADDsym AUC full','Pose frames','Pose %'],rows,
        'Prediction-only unchanged PnP on DEV319. Reconstructed image/geometry reference, not independent physical 6D truth. Failures retained in AUC/coverage.',source,'DEV319','319 pose attempts',original_x='N0/N1 and P90 columns newly supplied')
    table('tab:ablation_results',['Method','Median px','P90 px','PCK10 %','E_sym','T median cm','R median deg'],[[names[k],*vals(heads[k])[:3],heads[k]['E_sym'],*vals(heads[k])[3:]] for k in R.DCP_ARMS],
        'Controlled N0/N1/N2/N3, three seeds each. OLD_P remains separate. Full2499/observed2445 corners; no seed selection.',source,'DEV319','2499 full / 2445 observed',original_x='N0/N1 original x; new 6D columns')
    contrasts=[]
    for label,b,a in [('N2-N0','N0_BASE_REPLAY','N2_DIM_ONLY'),('N1-N0','N0_BASE_REPLAY','N1_SYM_ONLY'),('N3-N2','N2_DIM_ONLY','N3_DIM_SYM'),('N3-N1','N1_SYM_ONLY','N3_DIM_SYM')]:
        contrasts.append([label,*[heads[a][k]-heads[b][k] for k in ['matched_pooled_corner8_median_px','matched_pooled_corner8_P90_px']],100*(heads[a]['full_PCK10_fraction']-heads[b]['full_PCK10_fraction']),*[heads[a][k]-heads[b][k] for k in ['pose_translation_cm_median','pose_translation_cm_P90','pose_rotation_deg_median','pose_rotation_deg_P90','pose_yaw_deg_median','pose_yaw_deg_P90']]])
    table('tab:ablation_deltas',['Contrast','Delta median px','Delta P90 px','Delta PCK10 pp','Delta T med cm','Delta T P90 cm','Delta R med deg','Delta R P90 deg','Delta yaw med deg','Delta yaw P90 deg'],contrasts,
        'After minus before. Negative error deltas favor after; positive PCK deltas favor after. Session paired intervals in supplementary uncertainty table.',source,'DEV319','319 frames; 3 seed means',insert=True)
    rows=[]
    for material in ['plastic','wood']:
        for k in common:
            h=core[k]['mean']['material'][material];rows.append(['Rectangular '+material,names[k],h['frames'],h['full_supervised_corners'],h['observed_corners'],*vals(h)])
    table('tab:type_results',['Group','Method','Frames','Full corners','Observed corners','Median px','P90 px','PCK10 %','T med cm','R med deg'],rows,
        'Rectangular DEV material strata. Square119 is reported separately in two fixed denominator panels.',source,'DEV319 material strata','full vs observed denominators per row')
    for mode in ['manual_declared','manual_in_frame']:
        rows=[]
        for family,d in square['results']['yolo'][mode].items():
            h=d.get('single',d.get('mean_of_seed_summaries'));rows.append(['YOLO',names[family],119,square[mode],h['observed_corners'],h['matched_pooled_corner8_median_px'],h['matched_pooled_corner8_P90_px'],100*h['PCK']['10'],None,None])
        for b in ['dope','resnet18']:
            methods=square['results'][b][mode];base=methods['base'];mean=R._mean_dict([methods[f'n3_seed{s}'] for s in (1,2,3)])
            for name,h in [('Base',base),('N3',mean)]:rows.append([b,name,119,square[mode],h['observed_corners'],*vals(h)[:3],None,None])
        table('tab:square_'+mode,['Backbone','Method','Frames','Full corners','Observed corners','Median px','P90 px','PCK10 %','T cm','R deg'],rows,
            'GREEN0918_119, '+mode+', one capture session, WDH=1.10/1.10/0.15 m. Same mode across methods. T/R x: no independent canonical pose reference. Generalization CI NA.',DOC/'SQUARE_AUDIT.json','GREEN0918_119',mode)
    sqcontrast=[]
    for mode in ['manual_declared','manual_in_frame']:
        f=square['results']['yolo'][mode];n3=f['N3_DIM_SYM']['mean_of_seed_summaries']
        for label,b in [('N3-R0',f['R0']['single']),('N3-N2',f['N2_DIM_ONLY']['mean_of_seed_summaries'])]:sqcontrast.append([mode,label,n3['matched_pooled_corner8_median_px']-b['matched_pooled_corner8_median_px'],n3['matched_pooled_corner8_P90_px']-b['matched_pooled_corner8_P90_px'],100*(n3['PCK']['10']-b['PCK']['10'])])
    table('tab:square_contrasts',['Mode','Contrast','Delta median px','Delta P90 px','Delta PCK10 pp'],sqcontrast,'Whole-package transfer N3-R0 is distinct from added symmetry N3-N2. One session and no C4 training rows.',DOC/'SQUARE_AUDIT.json','GREEN0918_119','mode-specific 600/602')
    rows=[]
    for group in ['all','clean','moderate','severe','unclassified']:
        for k in common:
            h=heads[k] if group=='all' else core[k]['mean']['occlusion'][group]
            rows.append([group,names[k],h['frames'],h['full_supervised_corners'],h['observed_corners'],*vals(h)[:3],h['pose_translation_cm_median'],h['pose_translation_cm_P90'],h['pose_rotation_deg_median'],h['pose_rotation_deg_P90'],100*h['pose_coverage']])
    table('tab:occlusion_results',['Group','Method','Frames','Full corners','Observed corners','Median px','P90 px','PCK10 %','T med cm','T P90 cm','R med deg','R P90 deg','Pose %'],rows,
        'Direct labels: clean29/moderate20/severe79; unknown191 retained. Overall P90 computed from all original scores, not subgroup averages. Labels are descriptive, not an occlusion causal intervention.',source,'DEV319','current direct-review version; counts per row')
    vr=labels['visibility_results'];rows=[]
    for group in ['DIRECT_VISIBLE','EXTERNAL_OCCLUDED','UNKNOWN']:
        for k in common:
            h=vr['R0'][group] if k=='R0' else R._mean_dict([vr[f'{k}_seed{s}'][group] for s in (1,2,3)])
            rows.append([group,names[k],h['corners'],h['observed_corners'],h['median_px'],h['P90_px'],100*h['PCK10_fraction']])
    table('tab:visibility',['Status','Method','Full corners','Observed corners','Median px','P90 px','PCK10 %'],rows,
        'Existing FINAL_V2 reviewed native statuses only: visible66, external-occluded5, unknown2428. Frozen DEV coordinates retained; updated manual clicks differ. Whole-object correspondence selected once, canonical identities propagated. Prior prediction exposure unknown.',DOC/'STATIC_LABEL_AUDIT.json','DEV319 approved visibility strata','66 / 5 / 2428 full reference corners')
    rows=[]
    for k in ['OLD_P','N3_DIM_SYM']:
        for seed in (1,2,3):
            d=tail[k][str(seed)];f=d['fixed_branch'];bins=d['inside_outside'];j=d['joint']
            rows.append([names[k],seed,f['initial_error_inside_cap_corners'],f['initial_error_outside_cap_corners'],*[bins[b].get(x,0) for b in ['inside','outside'] for x in ['improved','unchanged','worsened']],f['good5_to_bad10_corners'],f['bad20_to_good10_corners'],j.get('2d_improved_both_TR_improved',0),j.get('2d_improved_either_TR_worsened',0)])
    table('tab:tail',['Method','Seed','Inside count','Outside count','Inside better','Inside same','Inside worse','Outside better','Outside same','Outside worse','Good5-bad10','Bad20-good10','2D+T+R better frames','2D better T/R worse frames'],rows,
        'Fixed base-selected correspondence; 1% cap; 2445 comparable corners and311 frames. Seed integer counts shown separately, not rounded seed averages.',DOC/'POSE_TRACE_AND_TAIL.json','DEV319','311 matched frames /2445 corners',seeds='separate seed1/2/3')
    runtime={b:C.read(C.DOC/f'RUNTIME_{b.upper()}_SEED1.json') for b in ['dope','resnet18']};runtime['yolo']=C.read(DOC/'RUNTIME_YOLO_SEED1.json');rows=[]
    for b,d in runtime.items():
        for name,path in [('Base','base_e2e'),('N3','n3_seed1_e2e')]:
            h=d['summary'][path];wall=h.get('synchronized_wall_ms',h.get('wall_ms'));head=d['summary']['n3_seed1_only']['cuda_event_ms']['median_ms'] if name=='N3' else 'NA'
            rows.append([b,'pallet-yolo26' if b=='yolo' else 'pallet-pose',name,0 if name=='Base' else d['parameters']['n3_total'],h['cuda_event_ms']['median_ms'],h['cuda_event_ms']['p90_ms'],wall['median_ms'],head,h['peak_allocated_bytes']/1024**2])
    write(DOC/'RUNTIME_PANEL.json',runtime)
    table('tab:cost',['Backbone','Environment','Path','Added params','CUDA med ms','CUDA P90 ms','Wall med ms','N3 only med ms','Peak allocated MiB'],rows,
        'Same physical RTX3080, fixed26 frames, seed1, warmup20/path,5 repeats (130/path), batch1, FP32, threads4/OpenCV1. RAM RGB through PnP; feature-resident correction separately. YOLO environment and feature-capture base overhead differ: no cross-environment absolute speed ranking. DOPE/ResNet measurements reused.',DOC/'RUNTIME_PANEL.json','fixed26 static DEV images','130 measured calls/path',seeds='seed1')
    rows=[]
    for k in ['R0','OLD_P']:rows.append([names[k],*vals(heads[k])])
    for k,p in reuse['H']['methods'].items():rows.append([k,*vals(p['mean']['headline'])])
    rows.append(['N3',*vals(heads['N3_DIM_SYM'])])
    table('tab:comparators',['Method','Median px','P90 px','PCK10 %','T med cm','R med deg'],rows,
        'Same DEV319/8-corner contract. D/L/PoseFix are full-method comparisons with different architectures/losses/input budgets. PoseFix synthetic-only PRIOR last6000, raw first pass; not later real Replay. No same-boundary comparator runtime claim.',C.DOC/'REUSE_RESULTS.json','DEV319','2499 full corners; per-method observed support')
    rows=[]
    for b in ['yolo','dope','resnet18']:
        if b=='yolo':panels=[('Base',heads['R0']),('N3',heads['N3_DIM_SYM'])]
        else:
            d=C.read(C.RAW/f'evaluation/{b}.json');panels=[('Base',d['methods']['base']['headline']),('N3',d['seed_aggregate']['mean_of_seed_metrics'])]
        for name,h in panels:rows.append([b if b!='resnet18' else 'ResNet18 RGB 10ep CONSTANT-fold',name,*vals(h),h['matched_frames'],h['observed_corners'],h.get('pose_available_frames',round(319*h['pose_coverage'])),100*h['pose_coverage']])
    table('tab:backbones',['Backbone','Path','Median px','P90 px','PCK10 %','T med cm','R med deg','Matched frames','Observed corners','Pose frames','Pose %'],rows,
        'Three separately trained N3 heads per backbone, identical additional6000-update budget. Bases RGB only; N3 image features plus WDH and symmetry supervision. All319 and2499 full-reference denominator retained. Different base training budgets and conditional supports prevent causal backbone ranking.',DOC/'CROSS_BACKBONE_SUBGROUPS.json','DEV319','319 frames /2499 full corners; observed/pose support per row')
    safe=reuse['I']['safe_common_cohort'];rows=[]
    for k,p in safe['methods'].items():h=p['mean']['headline'];rows.append([k,h['frames'],h['full_supervised_corners'],h['observed_corners'],*vals(h)])
    table('tab:update_alternative',['Method','Frames','Full corners','Observed corners','Median px','P90 px','PCK10 %','T med cm','R med deg'],rows,
        'Separate common non-exposed plastic128 panel,7 recording groups, reused DEV. R0 and N3 also scored on128. Original319 student panel BLOCKED_CONTRACT. Corrected teacher uses9 manually supervised images/38 corners; not N3. Historical LR5 selection used plastic194 DEV.',C.DOC/'REUSE_RESULTS.json','HELDOUT128 reused DEV','128 frames; common cohort, no319 values')
    rows=[]
    for b,d in paired['backbones'].items():
        for metric,h in d['mean_of_seed_statistics'].items():rows.append([b,metric,h['mean_seed_delta'],*h['CI95'],h['seed_sd_ddof1'],h['seed_min'],h['seed_max']])
    table('tab:pose_uncertainty',['Backbone','Metric','Mean delta','CI95 low','CI95 high','Seed SD','Seed min','Seed max'],rows,
        'N3 minus Base; difference of medians/quantiles, not median paired difference. Session bootstrap10000 seed20260917, same resamples across seeds; mean formed within each draw. Reused-DEV posthoc analysis, no multiplicity correction, geometry-derived reference.',DOC/'PAIRED_POSE_ANALYSIS.json','DEV319','13 sessions; success sets per backbone')
    rows=[]
    for b,d in paired['backbones'].items():
        for seed,p in d['per_seed'].items():
            s=p['success_sets']
            for field in ['translation_cm','rotation_deg','yaw_deg']:
                h=p['metrics'][field+'_median'];rows.append([b,int(seed),field,s['common_success'],h['median_after_minus_before_on_common_success'],h['improved'],h['unchanged'],h['worsened'],len(s['new_failure_ids']),len(s['recovered_ids'])])
    table('tab:pose_common_success',['Backbone','Seed','Metric','Common success','Median paired delta','Improved','Same','Worse','New failures','Recoveries'],rows,
        'Secondary common-success analysis, distinct from difference of marginal medians. Success IDs identical before/after for all seeds: YOLO319,DOPE210,ResNet319. DOPE109 failures remain in full319 AUC/coverage.',DOC/'PAIRED_POSE_ANALYSIS.json','DEV319 common-success secondary','319/210/319 success; full319 retained',seeds='separate seed1/2/3')
    extra=C.read(DOC/'ABLATION_POSE_UNCERTAINTY.json');rows=[]
    for contrast,d in extra['comparisons'].items():
        for metric,h in d.items():rows.append([contrast,metric,h['mean_seed_delta'],*h['CI95'],h['seed_min'],h['seed_max']])
    table('tab:ablation_uncertainty',['Contrast','Metric','Mean delta','CI95 low','CI95 high','Seed min','Seed max'],rows,'Same locked session paired bootstrap for pose ablations. No extra learning or favorable seed selection.',DOC/'ABLATION_POSE_UNCERTAINTY.json','DEV319','13 sessions;319 frames')
    extra=C.read(DOC/'PAIRED_2D_UNCERTAINTY.json');rows=[]
    for b,contrasts2 in extra['comparisons'].items():
        for contrast,d in contrasts2.items():
            for metric,h in d.items():rows.append([b,contrast,metric,h['mean_seed_delta'],*(h['CI95'] or [None,None]),h['seed_min'],h['seed_max']])
    table('tab:corner_uncertainty',['Backbone','Contrast','Metric (PCK fraction)','Mean delta','CI95 low','CI95 high','Seed min','Seed max'],rows,'Existing 2D bootstrap definitions, same session resamples as pose; full PCK is fraction here. Seed mean formed inside each draw.',DOC/'PAIRED_2D_UNCERTAINTY.json','DEV319','13 sessions;full2499/conditional method-specific')
    sub=C.read(DOC/'CROSS_BACKBONE_SUBGROUPS.json');rows=[]
    for b,d in sub.items():
        for group in ['clean','moderate','severe','unclassified']:
            for method in ['base','N3']:
                h=d[method]['mean']['occlusion'][group];rows.append([b,group,method,h['frames'],h['full_supervised_corners'],*vals(h),h['pose_rotation_deg_P90'],100*h['pose_coverage']])
    table('tab:backbone_occlusion',['Backbone','Group','Method','Frames','Full corners','Median px','P90 px','PCK10 %','T med cm','R med deg','R P90 deg','Pose %'],rows,'Joined to same direct-review128 and unknown191. Earlier DOPE/ResNet reports had all319 unclassified. All overall values unchanged.',DOC/'CROSS_BACKBONE_SUBGROUPS.json','DEV319','current shared label version')
    # Main manuscript copy: replace exact original labels and local adjacent stale prose.
    original=C.DOC/'contract/source_context/manuscript_ko_v3.md';text=original.read_text()
    lifter=text[text.index('# 기록된 리프터 주행 영상의 사례 연구'):text.index('# 논의')]
    replacements={}
    for label,(content,insert) in tables.items():
        if not insert:continue
        pattern=r'<div id="'+re.escape(label)+r'">.*?</div>'
        if re.search(pattern,text,flags=re.S):
            text=re.sub(pattern,lambda m:'<div id="'+label+'">\n\n'+content+'\n</div>',text,count=1,flags=re.S);replacements[label]='existing table replaced'
    # Replace whole directly related result sections so abandoned x statements do not survive.
    def replace_section(heading,body):
        nonlocal text
        start=text.index(heading);next_heading=text.find('\n## ',start+len(heading));major=text.find('\n# ',start+len(heading))
        stop=min(x for x in [next_heading,major,len(text)] if x!=-1)
        text=text[:start]+heading+'\n\n'+body.strip()+'\n\n'+text[stop:].lstrip('\n')
    def block(label):return '<div id="'+label+'">\n\n'+tables[label][0]+'\n</div>'
    replace_section('## 치수 정보와 대칭 감독의 추가 효과',
        '동일 계약의 N0/N1 원시 코너에 누락되어 있던 6D 평가를 추가했다. OLD_P는 과거 실행이며 재현 통제 N0와 구분한다. 치수만 추가한 N2−N0와 대칭만 추가한 N1−N0, 결합 구성 N3−N2/N1을 따로 보고한다. N3가 N2보다 모든 지표에서 좋지는 않다. N3−N2의 코너 중앙값은 약 +0.000489px이며 PCK10 차이는 0이다. 대칭 감독이 일반적으로 필수라는 결론은 지지하지 않는다.\n\n'+block('tab:ablation_results')+'\n\n'+block('tab:ablation_deltas'))
    replace_section('## 형상·재질별 보정 효과와 정사각형 신규 평가',
        '직사각형 플라스틱194장과 목재125장을 원시 행에서 재집계했다. 정사각형은119장·한 촬영 세션·한 치수(1.10×1.10×0.15m)의 고정 모델 적용 결과이다. manual_declared602점과 manual_in_frame600점을 별도 패널로 보존한다. 제외되는 두 좌표는029710/0번(-42,289),029844/4번(-7,310)이다. 같은 패널의 모든 방법은 동일 모드와 분모를 사용했다. 모델 선택 없이 두 결과를 제시하며, 사후 유리한 모드를 주 모드로 선정하지 않았다. 정사각형 이동·회전 정확도는 독립적인 표준 자세 참조가 없어 x로 유지한다. 한 세션의 일반화 CI는 NA이다.\n\n'+block('tab:type_results')+'\n\n'+block('tab:square_manual_declared')+'\n\n'+block('tab:square_manual_in_frame')+'\n\n'+block('tab:square_contrasts')+'\n\nN3−R0는 전체 보정 패키지의 전이이고 N3−N2는 대칭 감독의 추가 차이다. C4 학습 행은0이므로 C4 학습으로 얻은 전이라고 주장하지 않는다. 원고의 과거 S0/S1 실사 지도 결과와 합치지 않는다.')
    replace_section('## 외부 가림 정도별 성능',
        '직접 검수 manifest의 가림 없음29·중간20·심함79와 미분류191장을 모두 보존했다. 과거29/21/78과의 차이는 eval_pallet09:1778653661653195264 한 프레임으로, 이전 SPLIT_LOCK은 중간, 현재 직접 검수 기록은 심함이다. 변경 이유와 결과 열람 전후 시점은 미확인이다. 기존 주석을 개수에 맞추어 수정하지 않았다. 가림 층별 차이는 시점·재질·참조 불확실성을 함께 포함하며 독립적인 인과 효과가 아니다.\n\n'+block('tab:occlusion_results')+'\n\n<figure><img src="figures/occlusion_pck.png"/><figcaption>Full-denominator corner accuracy by recorded occlusion label. Unknown labels remain a separate panel; seed curves are not independent sessions.</figcaption></figure>')
    replace_section('## 가시 코너와 비가시 코너의 구분',
        '외부 가림 등급은 영상 단위, 코너 가시성은 점 단위이다. 기존 FINAL_V2의 직접 수동 상태 검수에서 가시66점과 외부 가림5점을 연결했고, 나머지2428점은 미확인으로 남겼다. 프레임 이미지 해시와 원래 코너 번호를 대조했다. 재클릭 좌표는 고정 DEV 참조와 달라 교체하지 않았다. 따라서 아래 표는 기존 참조 오차의 가시성별 분석이며, 새 클릭 좌표에 대한 정확도나 독립 참조 평가가 아니다. 상태와 좌표 출처를 구분하며 PnP 보조 검수 이력·사전 예측 노출 미확인을 기록한다.\n\n'+block('tab:visibility'))
    replace_section('## 이동 상한과 큰 오류의 보정 가능 범위',
        '원본 영상 대각선1% 이동 상한을 고정한다. 고정 대응에서 e_after ≥ max(0,e_before−0.01D)이며, 이 기하학적 하한은 실패 원인의 인과적 규명이 아니다. 아래 진단은 초기 예측이 선택한 전체 코너 대응을 보정 후에도 유지한다. 방법별 최적 대칭 대응을 사용하는 주 지표와 구분하며, cap이나 평가 대상을 바꾸지 않았다. 정수 count는 seed별로 표시한다.\n\n'+block('tab:tail'))
    replace_section('## 검출 출력의 유지와 계산 비용',
        '박스·점수·선택 객체·중심점·결측 마스크를 보존하는 보정 경로를 검증했다. 이는 검출 성능 자체를 높였다는 주장이 아니다. YOLO N3 추가 파라미터는20,259개이고 DOPE/ResNet 연결은각23,331개이다. 같은 RTX3080의 고정26장, 준비20회,5반복,대표seed1로 계측했다. DOPE/ResNet의 동일 계약 측정을 재사용하고 YOLO 누락 측정을 추가했다. 라이브러리 환경이 달라 별도 패널로 공개하며 기반 간 절대 속도 순위를 주장하지 않는다. 전체 경로와 보정만의 시간은 계측 경계가 달라 증가량과 일치할 필요가 없다. D/L/PoseFix와 동일 조건의 속도 우월성, Jetson 실시간성은 주장하지 않는다.\n\n'+block('tab:cost'))
    replace_section('## 다른 보정 방법과의 전체 성능 비교',
        'D/L/PoseFix의 동일319장·8코너 원시 점수에서 통계를 다시 계산해 검산했다. D는 직접 좌표 잔차 회귀, L은 영상 선 구조 보정, PoseFix는 합성 전용 PRIOR 세 seed의 마지막6000 update·첫 raw pass이다. 후속 실사 Replay 결과로 대체하지 않았다. 입력·구조·손실이 다른 전체 방법 비교이며 단일 구성 요소의 인과 효과로 해석하지 않는다. N3보다 좋은 비교군 수치도 그대로 보고한다.\n\n'+block('tab:comparators'))
    replace_section('## 동일 N3의 여러 기반 추정기 적용',
        'YOLO·DOPE/VGG·ResNet-18 각각에서 고정 기본 추정기와 N3의 같은 프레임 출력을 비교했다. 기본 추정기는 RGB를 입력받고, N3는 이미지 특징·초기점·박스·W,D,H를 입력받으며 전체 객체 대칭 감독을 사용한다. DOPE/ResNet 각3seed의6000 update 완료 기록과 가중치·예측을 재사용했다. ResNet은10-epoch CONSTANT-fold RGB 모델로, 조건 z=0의 FiLM을 고정 연산으로 접었다. CONSTANT는 출력 좌표가 상수라는 뜻이 아니다. 공간 softmax 기댓값 디코더를 사용하며60-epoch/argmax 설명은 유효 명세가 아니다. 기본 모델의 총학습량은 같지 않다.\n\n'+block('tab:backbones')+'\n\n세 기반 모두 코너·T·R 중앙값의 평균은 감소했지만 큰 오류 개선은 일관되지 않았다. DOPE 회전P90은 증가했고 ResNet 이동P90도 증가했다. ResNet의 T 중앙값 차이 CI는0을 포함한다. 자세 산출 ID는 각 기반의 모든 seed에서 전후 동일하며 신규 실패·복구는0이었다. DOPE의109 실패는 전체319장 분모에 남는다. ResNet 비항등 대칭 목표 선택은0, C4 학습 행은0이다. 치수 logits 민감도는 경로 활성 증거이며 독립적 치수 정확도 이득의 증명은 아니다. 세 기반에서 각각 학습한 적용 결과를 임의 기반·동일 가중치 전이·심한 가림의 보편적 강건성으로 확대하지 않는다.\n\n<figure><img src="figures/backbone_results.png"/><figcaption>Frozen Base versus N3 within each backbone. Conditional errors and full-population PCK/pose coverage use explicit denominators; tails and failures are retained.</figcaption></figure>')
    replace_section('## 추정기 업데이트라는 대안',
        '원래319장 학생 비교는 공통 비노출 계약이 성립하지 않아 BLOCKED_CONTRACT로 남긴다. 아래 표는 별도 공통 플라스틱128장·7 recording 집단이며 R0/N3도 같은128장에서 집계했다. 학생/교사와 해당 집단의 영상·촬영 겹침은 기존 잠금 감사에서0이다. 다만LR5 선택은 반복 사용한194장 DEV에서 이루어졌으므로 독립 시험이 아니다. 보정 의사 레이블 교사는 별도 수동9장·38점 감독을 사용하며 N3라고 부르지 않는다. 이번 새 자기학습은0회이다.\n\n'+block('tab:update_alternative'))
    # Insert supplementary results before static qualitative section, never inside lifter chapter.
    appendix='## 정지 이미지 짝지은 불확실성 보조표\n\n주 변화량은 median(after)−median(before) 또는 P90(after)−P90(before)이다. 공통 성공 프레임의 median(after−before)는 별도 보조표로 보고한다. 세션 전체를10000회,seed20260917로 짝지어 재표집하고 매 표본 안에서 세seed 통계 차이를 평균했다. CI 끝점 평균이나 코너 독립 재표집을 사용하지 않았다. 재사용DEV의 사후 분석·적은 세션·다중 비교 미보정·기하 재구성 참조 한계는 남는다.\n\n'
    for label in ['tab:pose_uncertainty','tab:pose_common_success','tab:ablation_uncertainty','tab:corner_uncertainty','tab:backbone_occlusion']:appendix+=block(label)+'\n\n'
    appendix+='<figure><img src="figures/pose_uncertainty.png"/><figcaption>Session-paired 95% intervals for N3 minus Base. Zero is shown; negative error deltas indicate lower error. Reused-DEV exploratory uncertainty.</figcaption></figure>\n\n'
    text=text.replace('## 정성 사례',appendix+'## 정성 사례') if '## 정성 사례' in text else text.replace('# 기록된 리프터 주행 영상의 사례 연구',appendix+'# 기록된 리프터 주행 영상의 사례 연구')
    # Model/training pending statements outside result sections: replace specific paragraphs.
    paragraphs=text.split('\n\n');edited=[]
    for para in paragraphs:
        if para.startswith('**\\[추정·미검증 / 결과 x\\]**') and ('DOPE' in para or 'ResNet' in para) and '리프터' not in para:
            para='[실행 확인] DOPE·ResNet의 치수·대칭 N3 각3seed가 완료되었다. 두 기반은 원본 좌표축별 역변환 후1% cap과 검출 출력 보존을 검증했다. 같은 추가 학습 예산(6000 update×batch16)을 적용하며 기본 추정기의 총학습량까지 동일하지는 않다. ResNet은10-epoch CONSTANT-fold RGB/DSNT 모델이다. 원본 잠금 기록의 옛 설명은 별도 유효 명세 정정 기록에 보존한다.'
        edited.append(para)
    text='\n\n'.join(edited)
    text=text.replace('이는 연결부 구현의 값이지 후자의 N3 학습 완료 근거가 아니다.','연결부 구현과 별도로 각 N3 학습·추론 완료 기록 및 원시 예측을 검증했다.')
    text=text.replace('확인된 YOLO N2/N3는 세 seed','확인된 YOLO N2/N3와 DOPE·ResNet N3는 각각 세 seed')
    # Correct remaining local setup/summary statements while preserving every excluded lifter sentence.
    text=text.replace('자료와 감독의 역할. 새 정사각형셋의 어노테이션 완료는 사용자 제공 정보이고, 규모와 평가 결과는 x이다.', '자료와 감독의 역할. 정사각형은119장·한 세션·두 수동 코너 모드의 고정 평가를 검산했다.')
    text=text.replace('| 직사각형 / 플라스틱 | x       | 유형별 수·성능 재집계 x     |','| 직사각형 / 플라스틱 | 194     | 고정 DEV 유형별 재집계 완료 |')
    text=text.replace('| 직사각형 / 목재     | x       | 유형별 수·성능 재집계 x     |','| 직사각형 / 목재     | 125     | 고정 DEV 유형별 재집계 완료 |')
    text=text.replace('| 새 정사각형         | x       | 주석 완료; 고정 모델 평가 x |','| 새 정사각형         | 119     | 한 세션; 선언602/화면내600점 |')
    text=re.sub(r'이번에 추가하는 정사각형 데이터셋은 사용자 설명상.*?두 자료의 동일성이나 독립성도 현재 단정하지 않는다\.',
        '추가 정사각형 GREEN0918은119장·한 세션·한 치수이다. 고정 R0/P/N2/N3와 DOPE/ResNet Base/N3의 두 모드를 검산했다. 독립6D 참조와 개체 동일성/과거 S0/S1 실사 지도 이력의 완전한 대응은 미확인이다. 과거696장 학습·155장 평가 수치를 새 평가로 전용하지 않는다.',text)
    text=text.replace('분류·대응 확인과 재집계는 아직 수행 전이며 결과는 x이다.','기존 확정 주석으로 재집계했으며 미검수 영상191장과 코너2428점은 별도 미확인 집단으로 유지한다.')
    text=text.replace('현재 주 표와 동일한 8코너 평가로 재확인하지 않은 N0/N1 수치는 x이며, 과거 다른 집계값을 대체 입력하지 않는다.','N0/N1 원시 코너와 누락6D를 같은8코너·PnP 계약으로 검산했으며 과거P를 N0로 대체하지 않았다.')
    text=text.replace('| N3 학습·평가 완료       | 확인                       | x                          | x                          |','| N3 학습·평가 완료       | 3seed 확인                 | 3seed 확인                 | 3seed 확인                 |')
    text=text.replace('**동일하게 사용해야 함**','가로·깊이·높이 사용').replace('**동일 규칙 적용**','전체 순열 선택 + 분포 손실')
    text=re.sub(r'\*주: 특징 채널·간격은 연결 코드에서 확인한 값이다\..*?분리한다\.\*',
        '*주: 특징 채널·간격과 원본 좌표 역변환을 확인하고 각 N3의3seed 완료 가중치·원시 예측을 대조했다. ResNet은10-epoch CONSTANT-fold RGB/DSNT이다. 기본 학습 이력과 추가 보정 학습량을 분리한다.*',text)
    text=text.replace('동일 평가와 감독 예산 대조가 x인 동안 업데이트보다 N3가 항상 우월하다는 결론은 내리지 않는다.','동일128장 패널을 검산했으나 추가 감독 예산과 선택 이력이 달라 업데이트보다 N3가 항상 우월하다는 결론은 내리지 않는다.')
    text=text.replace('새 정사각형셋은 주석 완료·평가 전이며, 부록의 과거 정사각형 실사 지도 자료와 혼합하지 않는다. 가림별 수치와 코너 가시성 분해도 x이므로,','새 정사각형셋은 두 모드 평가를 완료했으며 부록의 과거 실사 지도 자료와 혼합하지 않는다. 기존 확정 주석의 가림·가시성 분해를 제시했으나 미검수 집단이 남아 있으므로,')
    text=text.replace('새로운 세션에서의 확인과 여러 기반 추정기에 대한 효과가 확보되기 전까지는','세 기반에서 각각 학습한 효과는 확보했으나 새로운 세션의 독립 확인 전까지는')
    text=text.replace('동일 N3의 DOPE·ResNet-18 적용, 새 정사각형, 가림·가시성 및 큰 오류 분해, 동등 평가의 비교 보정기·업데이트 대안, 기록 주행 영상의 결과는 x이다.',
        '동일 N3의 DOPE·ResNet-18 적용, 정사각형 두 모드, 확정 주석의 가림·가시성 및 큰 오류 분해, 비교 보정기와 별도128장 업데이트 대안의 계산을 완료했다. 정사각형6D와 미검수 집단의 확정 주석은 x이다. 기록 주행 영상의 결과는 x이다.')
    text=text.replace('이 결과가 확보된 이후에만 기반 추정기 적용 범위와 실제 주행 관측에서의 효과를 최종 결론에 반영한다.','기반 추정기 적용 범위는 세 조건의 관측 결과로 한정한다. 실제 주행 관측에서의 효과는 해당 결과가 확보된 이후에만 최종 결론에 반영한다.')
    text=text.replace('현재 수치는 YOLO 기반 보정의 개발 자료 내 효과를 보여준다. 다른 기반 추정기의 N3, 새 정사각형, 가림별 분석 및 주행 사례에서 아직 확보하지 않은 결과는 x로 남기며,','현재 수치는 YOLO·DOPE·실제 ResNet 기준에서 각각 학습한 N3의 개발 자료 내 효과와 한계를 보여준다. 정사각형2D와 확정 주석별 분석을 추가했으며 주행 사례에서 아직 확보하지 않은 결과는 x로 남기고,')
    text=text.replace('동일한 제안 보정기의 여러 추정기 적용성과 기록 주행 영상에서의 효과는 별도의 검증 대상으로 남는다.','동일한 제안 보정기를 DOPE·ResNet에도 각각 적용한 결과 중앙 오차는 감소했지만 일부 P90은 악화되었다. 기록 주행 영상에서의 효과는 별도의 검증 대상으로 남는다.')
    text=text.replace('현재 확인한 YOLO 기반 실사 319장 평가에서','고정된 실사319장 개발 평가의 YOLO 경로에서')
    text=text.replace('현재 수치의 원천은 2026년 10월 1일 게시된 저장소 고정본 `6c66b23`의 코너·자세 표와 해당 결과를 제시한 상담 자료이다.','원 수치는 저장소 `6c66b23`에 연결되어 있으며 본 마감은 로컬 `a7fb680`의 완료 실험·원시 예측과 별도 누락 계산을 검산했다. 각 표의 현재 원천 해시는 PAPER_CELL_MAP과 SOURCE_BINDINGS에 연결했다.')
    text=text.replace('DOPE·ResNet-18의 N3는 완료 전이므로 x로 표시하였다.','DOPE·ResNet-18의 N3는 각3seed 완료 결과를 본문 표에 제시한다.')
    # Use current raw-backed examples in the static chapter only.
    start=text.index('## 보정 사례와 남는 오류');stop=text.index('## 정지 이미지 짝지은 불확실성 보조표',start)
    text=text[:start]+'## 보정 사례와 남는 오류\n\n다음 사례는 정량 모집단을 변경하지 않는 사후 설명 자료이다. seed1과 기존 직접 검수 영상만 사용하고, 프레임 평균 코너 변화가 개선·무차이·악화인 각 집단의 ID 사전순 첫 영상을 골랐다. 해당 사례가 없는 칸은 없다고 표시한다. 실제 코너 예측을 그렸으며 PnP 재투영점이 아니다.\n\n<figure id="fig:examples"><img src="figures/static_examples.png"/><figcaption>Deterministic posthoc static examples: first approved-label frame ID per signed change category. Seed1 only for illustration; all seeds remain in quantitative tables. Cyan: Base, magenta: N3, yellow: image/geometry-derived reference.</figcaption></figure>\n\n'+text[stop:]
    text=text.replace('DOPE·ResNet-18의 N3 연결·학습·평가는 추가 검증 대상이며 완료 결과는 x이다.','DOPE·ResNet-18의 N3 연결·학습·평가는 각3seed의 완료 기록과 원시 예측으로 검증했다.')
    # Keep the excluded chapter byte-identical; write only a new copy.
    assert text[text.index('# 기록된 리프터 주행 영상의 사례 연구'):text.index('# 논의')]==lifter
    dest=DOC/'manuscript_ko_v3_static_closeout.md';dest.write_text(text)
    # Mark table insertion based on presence in the written copy, not result availability.
    for cell in cells:
        present=('<div id="'+cell['table_label']+'">') in text
        cell['markdown_inserted']=present
        if cell['status'] in ('INSERTED','READY_TO_INSERT'):cell['status']='INSERTED' if present else 'READY_TO_INSERT'
    write(DOC/'PAPER_CELL_MAP.json',cells)
    fields=list(cells[0])
    with (DOC/'PAPER_GAP_MATRIX.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(cells)
    (DOC/'PAPER_GAP_MATRIX.md').write_text('# 원고 칸 연결\n\n국문 v3 Markdown 복사본에 삽입한 표와 LaTeX 교체 조각을 구분한다. 동일 국문 v3의 LaTeX 원본은 발견되지 않아 실제 LaTeX 삽입은 하지 않았다. 원천 해시와 개별 칸은 CSV/JSON에 있다.\n\n'+md(['Table label','Cells','Markdown','LaTeX'],[[label,sum(c['table_label']==label for c in cells),'INSERTED' if '<div id="'+label+'">' in text else 'READY_TO_INSERT','READY_TO_INSERT fragment'] for label in tables]))
    write(DOC/'PAPER_PATCH_RECEIPT.json',{'original':bind(original),'copy':bind(dest),'existing_tables_replaced':replacements,'table_labels':list(tables),
        'lifter_chapter_byte_identical':True,'latex_source_found':False,'latex_fragment_count':len(tables),'pdf_compiled':False,'cell_count':len(cells)})
    patch='# 국문 v3 결과 반영\n\n원본은 보존하고 manuscript_ko_v3_static_closeout.md에 실제 표와 관련 본문을 반영했다. matching v3 LaTeX 원본은 없어 generated_tables의 동일 label별 .tex는 교체 준비 상태이다. 다른 self-training/P-only 원고는 수정하지 않았다.\n\n'
    patch+='\n'.join('- `'+label+'`: '+('기존 표 교체' if label in replacements else '보조표 추가') for label in tables)
    patch+='\n\nResNet 실제10epoch/DSNT 명세, 공통128장 학생 표, 두 정사각형 모드, 직접 검수71점 상태와 참조 좌표 차이, 중앙값/P90 및 CI의 다른 해석을 본문에 반영했다. 리프터 장은 원본과 byte 단위로 같으며 이번 검증 대상이 아니다. 원래319장 학생 칸은 BLOCKED_CONTRACT; 정사각형T/R은 x; 일반화CI는 NA이다.\n'
    (DOC/'PAPER_RESULTS_PATCH_KO.md').write_text(patch)
    # Requested flat exports.
    write(DOC/'N0_N1_POSE_RESULTS.json',{k:core[k] for k in ['N0_BASE_REPLAY','N1_SYM_ONLY']})
    flat=[]
    for k,p in core.items():
        if k=='R0':continue
        for seed,v in p['per_seed'].items():flat.append({'method':k,'seed':seed,**v['headline']})
    with (DOC/'N0_N1_POSE_RESULTS.csv').open('w',newline='') as f:
        fields=['method','seed']+[k for k in flat[0] if k.startswith('pose_')];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(flat)
    with (DOC/'PAIRED_POSE_ANALYSIS.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['backbone','metric','mean_delta','CI95_low','CI95_high','seed_sd','seed_min','seed_max'])
        for b,d in paired['backbones'].items():
            for metric,h in d['mean_of_seed_statistics'].items():w.writerow([b,metric,h['mean_seed_delta'],*h['CI95'],h['seed_sd_ddof1'],h['seed_min'],h['seed_max']])

if __name__=='__main__':main()
