"""Build V199 source report from sealed generated JSON summaries only.
No source input/target arrays, checkpoint/log bodies, model import or forward.
"""
from pathlib import Path
from html import escape
import hashlib,json,os,re,stat,datetime
V=Path(__file__).absolute().parent.parent;P=V.parent
RUN=V/'runs/C58_C59_NATIVE_CONDITIONED_GROUPED_SOURCE_FIXED8000'
ARMS=('C58_RAW96_DIRECT_RESIDUAL_ZERO_NATIVE_CONTROL','C59_RAW96_DIRECT_RESIDUAL_NATIVE_ONEHOT')
SHORT={ARMS[0]:'C58 zero8',ARMS[1]:'C59 native onehot8'}
inputs={}
def need(value,msg):
    if not value:raise AssertionError(msg)
def raw(path):
    p=Path(path);need(p.is_absolute() and p.is_relative_to(P) and 'v112'not in str(p).casefold(),'Finite project lexical path')
    need(p.suffix in('.json','.md','.py'),'Only authorized generated JSON/report/code texts')
    fd=os.open(p.anchor,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[1:-1]:
            end=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=end
        last=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    need(stat.S_ISREG(os.fstat(last).st_mode),'Ordinary source metadata file')
    with os.fdopen(last,'rb')as f:data=f.read()
    inputs[str(p)]=dict(path=str(p),sha256=hashlib.sha256(data).hexdigest(),size_bytes=len(data));return data
def jd(path):return json.loads(raw(path))
def bound(b):
    data=raw(b['path']);need(hashlib.sha256(data).hexdigest()==b['sha256'],'Sealed JSON/report binding');return json.loads(data)
def write(path,data):
    mode='xb'if isinstance(data,bytes)else'x'
    with path.open(mode)as f:f.write(data)
def percent(value):return '—'if value is None else f'{100*value:.1f}%'
def tbl(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |'for row in rows])
def render_md(md):
    lines=md.splitlines();out=[];i=0
    def inline(text):
        text=escape(text);text=re.sub(r'!\[([^]]*)\]\(([^)]+)\)',r'<img alt="\1" src="\2" style="max-width:100%;height:auto">',text);text=re.sub(r'\[([^]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',text);return re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
    while i<len(lines):
        line=lines[i]
        if line.startswith('|'):
            block=[]
            while i<len(lines)and lines[i].startswith('|'):block.append(lines[i]);i+=1
            rows=[[q.strip()for q in s.strip().strip('|').split('|')]for s in block];out.append('<div class="scroll"><table><thead><tr>'+''.join('<th>'+inline(q)+'</th>'for q in rows[0])+'</tr></thead><tbody>')
            for row in rows[2:]:out.append('<tr>'+''.join('<td>'+inline(q)+'</td>'for q in row)+'</tr>')
            out.append('</tbody></table></div>');continue
        if line.startswith('# '):out.append('<h1>'+inline(line[2:])+'</h1>')
        elif line.startswith('## '):out.append('<h2>'+inline(line[3:])+'</h2>')
        elif line:out.append('<p>'+inline(line)+'</p>')
        i+=1
    return '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V199 source native conditioning</title><style>body{font:16px/1.75 system-ui,sans-serif;max-width:1220px;margin:36px auto;padding:0 22px;color:#192638;background:#fafbfd}h1{font-size:30px}h2{font-size:23px;margin-top:32px}table{border-collapse:collapse;width:100%;background:#fff;margin:18px 0}td,th{border:1px solid #cbd5df;padding:9px;text-align:left;vertical-align:top}th{background:#eaf1f7}.scroll{overflow:auto}a{color:#185fa5}</style><main>'+''.join(out)+'</main></html>\n'

def main():
    review_path=V/'independent_saved_fit_review_c/actual_review_v1/REVIEW.json';review=jd(review_path)
    need(inputs[str(review_path)]['sha256']=='9a2e02c97ac754434480dee8af4aba2288c393fab616fa86ab533ccbbd60036e'and review['completed']and review['PASS'],'Exact actual saved-fit auditor PASS')
    readout=bound(review['source_readout']);fit=bound(review['source_fit_complete']);validation=jd(RUN/'FIT_INPUT_VALIDATION.json');prior=bound(fit['native_class_prior_census']);collisions=bound(fit['raw_identity_collision_census']);barrier=bound(fit['all2_freeze']);proto=bound(fit['protocol']);scope=bound(proto['prospective_scope'])
    # Older closed reports only; no old data/pixel/score containers.
    old197=jd(P/'query_visibility_train_only_target_contract_readout_20261024_v197/ROOT_RESULTS_KO.json');old198=jd(P/'query_visibility_native_query_conditioning_feasibility_20261024_v198/ROOT_RESULTS_KO.json')
    need(fit['completed']and fit['PASS']and fit['source_only']and fit['steps_each']==8000 and fit['total_updates']==16000 and fit['parameters_each']==116407,'Fixed source trial complete')
    need(fit['all2_frozen_before_any_TRAIN_or_HOLDOUT_outputs']and barrier['source_TRAIN_or_HOLDOUT_prediction_emitted']is False,'Both final8000 first')
    need(prior==review['native_class_prior_census']and collisions==review['raw_identity_collision_census'],'Independent census identity')
    need(set(readout['models'])==set(ARMS)and validation['cohorts']==fit['cohort_denominators'],'Full source metadata closure')
    models=readout['models'];cohorts=fit['cohort_denominators'];direct=cohorts['DIRECT_CONTROL'];expected_H=[[[28,2,0],[10,5,0]],[[28,2,0],[11,4,0]]]
    for arm,expected in zip(ARMS,expected_H):
        m=models[arm];need(m['TRAIN']['CM_D_E_by_D_E_U']==[[218,0,0],[0,109,0]]and m['HOLDOUT']['CM_D_E_by_D_E_U']==expected,'Actual source TRAIN/HOLDOUT CMs')
        need(fit['arms'][arm]['use_sam']is False and fit['arms'][arm]['native_condition']==(arm==ARMS[1]),'Both RAW exact native flag')
        need(m['HOLDOUT']['truth_UNKNOWN']==315 and m['HOLDOUT']['known']==45 and m['HOLDOUT']['known_abstentions']==0,'Source HOLDOUT denominators')
        need(m['paired_D_E_D_triplets']['TRAIN']['all3_correct']==109 and m['paired_D_E_D_triplets']['TRAIN']['known_D_E_D_triplets']==109,'All TRAIN known triplets')
    holds=models[ARMS[0]]['paired_D_E_D_triplets']['HOLDOUT']['triplets'];bases=sorted({int(re.search(r'BASE(\d+)',q['attempt_id']).group(1))for q in holds});need(bases==[42,50,74,75,83,104],'Six existing synthetic HOLDOUT bases')
    need(collisions['known_available_TRAIN_rows']==327 and collisions['RAW_only']['unique_identities']==collisions['RAW_plus_native']['unique_identities']==312 and collisions['RAW_only']['conflicting_identities']==collisions['RAW_plus_native']['conflicting_identities']==0,'TRAIN RAW exactidentity census')
    prior_all=[q for q in prior['rows']if q['domain']=='ALL'];need([q['direct_E']for q in prior_all]==old197['all109_native_known_triplets0to7']==[35,26,13,22,6,7,0,0],'Native support unchanged')
    full=models[ARMS[0]]['paired_D_E_D_triplets']['ALL'];need(full['incomplete_variant_groups']==168 and full['known_D_E_D_triplets']*3==models[ARMS[0]]['ALL']['known']==372,'Incomplete groups contain no extra known target')
    need(full['all_attempt_native_groups']==168+868+124 and 168+3*(868+124)==3144,'Full attempt/native and row denominators')
    # One static scientific figure from saved scalar metrics; not a source image.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    figdir=V/'scientific_readout_v1';figdir.mkdir(exist_ok=False);png=figdir/'NATIVE_CONDITION_SOURCE_HOLDOUT.png';svg=figdir/'NATIVE_CONDITION_SOURCE_HOLDOUT.svg'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(11.4,6.6));x=np.arange(3);width=.33;colors=['#286b9f','#d77c2d'];den=[30,15,15]
    for j,arm in enumerate(ARMS):
        m=models[arm];correct=[m['HOLDOUT']['classes'][0]['correct'],m['HOLDOUT']['classes'][1]['correct'],m['paired_D_E_D_triplets']['HOLDOUT']['all3_correct']];values=[a/b for a,b in zip(correct,den)];pos=x+(j-.5)*width
        bars=ax.bar(pos,values,width,label=SHORT[arm],color=colors[j])
        for bar,a,n in zip(bars,correct,den):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.025,f'{a}/{n}\n{100*a/n:.1f}%',ha='center',va='bottom',fontsize=11)
    ax.set_xticks(x,['Visible-end D recall\nknown D=30','External-end E recall\nknown E=15','Complete D/E/D response\nknown triplets=15']);ax.set_ylim(0,1.22);ax.set_yticks([0,.25,.5,.75,1.],['0%','25%','50%','75%','100%']);ax.set_ylabel('Fraction (separate fixed denominators)');ax.grid(axis='y',alpha=.20);ax.set_axisbelow(True);ax.legend(loc='upper right',frameon=False,ncol=2);ax.set_title('V199: native one-hot did not improve this fixed source holdout',loc='left',pad=18,fontsize=15)
    fig.text(.075,.075,'360 source rows: 45 known + 315 truth UNKNOWN; input U=0. 15 triplets from 6 synthetic bases.\nPreviously exposed source design; real-end holdout=0. Both TRAIN: 327/327; no arm selected.',fontsize=10,color='#364659');fig.subplots_adjust(left=.085,right=.98,top=.86,bottom=.25);fig.savefig(png,dpi=160,metadata={'Description':'Saved source scalar summaries only; no source images or model execution.'});fig.savefig(svg);plt.close(fig)
    metric_rows=[]
    for role in ('TRAIN','HOLDOUT'):
        for arm in ARMS:
            m=models[arm][role];D,E=m['classes'];tri=models[arm]['paired_D_E_D_triplets'][role]
            metric_rows.append([role,SHORT[arm],m['known'],m['truth_UNKNOWN'],f'{D["correct"]}/{D["truth"]}',percent(D['precision']),percent(D['recall']),f'{E["correct"]}/{E["truth"]}',percent(E['precision']),percent(E['recall']),m['known_abstentions'],f'{tri["all3_correct"]}/{tri["known_D_E_D_triplets"]}'])
    native_rows=[]
    for k in range(8):
        a=models[ARMS[0]]['by_native'][str(k)];b=models[ARMS[1]]['by_native'][str(k)];cm=a['TRAIN']['CM_D_E_by_D_E_U'];hc=a['HOLDOUT']['CM_D_E_by_D_E_U'];native_rows.append([k,f'{sum(cm[0])}/{sum(cm[1])}',a['TRAIN']['truth_UNKNOWN'],prior_all[k]['original_inputU'],f'{sum(hc[0])}/{sum(hc[1])}',a['HOLDOUT']['truth_UNKNOWN'],f'{hc[1][1]}/{sum(hc[1])}'if sum(hc[1])else'—',f'{b["HOLDOUT"]["CM_D_E_by_D_E_U"][1][1]}/{sum(hc[1])}'if sum(hc[1])else'—'])
    pop_rows=[]
    for domain,q in cohorts.items():pop_rows.append([domain,q['TRAIN']['rows'],q['TRAIN']['original_inputU'],q['HOLDOUT']['rows'],q['HOLDOUT']['original_inputU']])
    links=[('고정 source 학습 COMPLETE',str((RUN/'COMPLETE.json').relative_to(V))),('독립 저장값 감사','independent_saved_fit_review_c/actual_review_v1/REVIEW.json'),('전체 source readout','independent_saved_fit_review_c/actual_review_v1/SOURCE_GROUPED_DIRECT_READOUT.json'),('입력/분할 검증',str((RUN/'FIT_INPUT_VALIDATION.json').relative_to(V))),('TRAIN native prior',str(Path(fit['native_class_prior_census']['path']).relative_to(V))),('TRAIN RAW identity',str(Path(fit['raw_identity_collision_census']['path']).relative_to(V))),('학습 프로토콜','fit_preparation_a/PROTOCOL_BEFORE_NATIVE_CONDITIONED_SOURCE_CACHE_OR_FIT_BODIES.json'),('native head 코드','fit_preparation_a/native_conditioned_head.py'),('모든 두 final8000 봉인',str(Path(fit['all2_freeze']['path']).relative_to(V))),('V197 source 부위·원 q 대응','../query_visibility_train_only_target_contract_readout_20261024_v197/PORTABLE_REPORT_KO.md'),('V198 사전 가설·입력 계약','../query_visibility_native_query_conditioning_feasibility_20261024_v198/PORTABLE_REPORT_KO.md'),('결과 JSON','ROOT_RESULTS_KO.json'),('HTML 읽기','ROOT_READOUT_KO.html'),('보고서 생성 코드','source_report_preparation_c/build_source_native_report.py')]
    md='''# V199: native 번호 조건의 source 대조 결과

**native 번호를 더한 팔은 이번 고정 source HOLDOUT에서 개선되지 않았다.** C58 zero8과 C59 native onehot8은 TRAIN327개 known을 모두 맞혔지만, HOLDOUT E recall은5/15(33.3%) 대4/15(26.7%), D/E/D 세 상태를 모두 맞힌 triplet은4/15 대3/15였다. 다른 실패 원인이나 native 조건 전반의 불가능성을 확정하는 결과는 아니다. 모델·팔·문턱을 채택하거나 선택하지 않았으며 전체 목표는 미해결이다.

## 고정한 대조

두 팔 모두 RAW96이고 새 SAM 추론은 없다. 원 backbone/shared64/base direct는 유지하고 Linear(64+8,16)→SiLU→Linear(16,1) residual의 마지막 층을0으로 초기화했다. 각116,407개 parameter·36개 상태 텐서를 저장하며, C58에는 zero8, C59에는 원 native_slot onehot8을 준다. 같은 seed1810·초기 상태·두 draw stream으로 각8000회 학습한 뒤, 두 final 상태를 먼저 봉인하고 TRAIN/HOLDOUT 출력을 저장했다.

기존 다섯 loss의 평균과192개 draw에 별도16D/16E direct32·weight1을 더했다. FP32 AdamW lr3e-4·weight_decay1e-4·clip5를 유지했다. 원 q/L·native 순서·U·타깃·초기화기·원96 sampler를 바꾸거나 PnP·재crop을 수행하지 않았다. support2·owner3·품질과 direct-end weak D/E는 다른 과제다. native는 direct residual에만 명시적으로 들어가지만 direct loss가 shared backbone을 바꾸므로 support/owner 학습이 같다고 할 수는 없다.

같은 저장 구조와 초기 출력은 통제했지만 동일한 유효 함수족을 주장하지 않는다. zero8 condition 열의 gradient는0이고 onehot 열은 학습 가능하다. 마지막 Linear에 onehot만 붙이는 native bias와, 영상×native 상호작용이 가능한 비선형 residual도 구분한다.

## TRAIN 적합도와 source HOLDOUT

'''+tbl(['분리','팔','known','truth UNKNOWN','D 정답/N','D precision','D recall','E 정답/N','E precision','E recall','known 예측 U','완전 D/E/D 정답/N'],metric_rows)+'''

CM 행은 truth D/E, 열은 prediction D/E/U다. C58 HOLDOUT은[[28,2,0],[10,5,0]], C59는[[28,2,0],[11,4,0]]이며 두 TRAIN CM은[[218,0,0],[0,109,0]]다. precision의 분모도 known truth 안의 예측만 사용한다. UNKNOWN315개를 E·D 정답에 합치지 않는다. HOLDOUT 전체360행의 input U는0이고 known 예측 U도0이다.

TRAIN은2784행=known327+UNKNOWN2457이고 input U112다. HOLDOUT은360행=known45(D30/E15)+UNKNOWN315다. HOLDOUT15 triplet은 합성 base42·50·74·75·83·104의6개 base에서 나온 것으로, 독립15영상이나 자연실사15가림 정답이 아니다. real weak end HOLDOUT은0이다. 이 source 설계와 기존 HOLDOUT은 이미 노출된 반복 진단이므로 fresh blind 일반화로 부르지 않는다. TRAIN100%는 apparent 적합도이며 실사 검증이 아니다.

![Fixed source HOLDOUT comparison](scientific_readout_v1/NATIVE_CONDITION_SOURCE_HOLDOUT.png)

그림은 저장 scalar 요약만으로 만든 과학 도표다. 원 source 사진·마스크·새 모델 응답을 만들지 않았다. 오차막대나 독립 표본에 대한 유의성 주장을 하지 않는다.

## native0..7과 UNKNOWN을 유지한 분모

'''+tbl(['native','TRAIN D/E','TRAIN UNKNOWN','TRAIN input U','HOLDOUT D/E','HOLDOUT UNKNOWN','C58 HOLD E정답/N','C59 HOLD E정답/N'],native_rows)+'''

두 팔의 각 native TRAIN known은 모두 정답이다. native6·7은 TRAIN/HOLDOUT direct known0이며 번호 입력이나 보조 owner 감독을 새 direct 권위로 승격하지 않았다. native4의 HOLDOUT known도0이므로 성능값을 채우지 않는다. 모든 native 원행과 UNKNOWN을 그대로 남겼다.

## full source와 입력 충돌 진단

'''+tbl(['source domain','TRAIN 행','TRAIN input U','HOLDOUT 행','HOLDOUT input U'],pop_rows)+'''

DIRECT 전체3144행에는 D248/E124/UNKNOWN2772가 있다. 완전 known D/E/D124 triplet(372행) 이외에 완전 UNKNOWN868 attempt/native 그룹과 불완전168 attempt/native 그룹도 남겼다. 불완전168은 추가 정답이 없는 UNKNOWN 대조 그룹이며, 구성된 variant만 보존한 기록이다. 전체1160그룹=124+868+168이고 행수는(124+868)×3+168=3144다. 누락 variant를 만들어 정확도를 늘리지 않았다.

TRAIN direct known327개를 원 RGB96uint8 HWC bytes+valid96bool bytes로 비교하면 exact input identity는312개이며 D/E 충돌 identity는0이다. native를 identity에 추가해도312개·충돌0이다. 이는 independent frame/component/triplet312개를 뜻하지 않으며, 다른 입력에서 단서가 충분하다는 증거도 아니다. TRAIN input U112와 available UNKNOWN2345개는 이 충돌 검사에서 별도 제외됐다. 행 삭제나 재라벨링은 없었다.

온전한 같은 native D/E/D와 원 component mask를 유지한 TRAIN prior는 지원 native마다 raw E=1/3, balanced16D/16E 기대 E=1/2로 독립 검산됐다. 유한8000 draw의 실현비율, owner/public 보조 과제의 prior, 영상/source-factor shortcut과는 다른 진단이다.

## 권위·감사 범위·다음 단계

TRAIN known327은 구성적 합성 finite-end246행과 공개실사 RGB-weak end+YCB cutout 제어차폐81행이다. 공개실사 source 학습도 허용된 사용자 계약을 따르며 synthetic-only 실험이라고 부르지 않는다. 공개실사 weak는 자연 외부가림 human gold나 physical native 인증이 아니다. 번호는 배포 때의 원 입력 slot 식별자이고, weak semantic end·canonical 등록·self/좌표품질은 별도 권위다.

독립 감사는 모든16000개 저장 draw/로그의 수치·TRAIN mask·RNG, 두 CPU final checkpoint의36텐서/AdamW8000, 전체147952 saved row의 META/decoder와 TRAIN/HOLDOUT 부분집합 byte 일치, prior/RAW identity를 확인했다. 실제 loss·gradient tensor를 모델로 재계산한 것은 아니며 초기 상태는 봉인 SHA 영수증, 파일 시간은 순서 보조 증거다. 전체 source target container는 원 권위와 draw를 검산하기 위해 감사에서 읽었지만 HOLDOUT 행은 최적화 draw/loss/정규화 학습·선택에 쓰지 않았다. 본 보고서 생성은 기존 JSON 요약만 읽었고 원 배열·가중치·로그·픽셀을 다시 열지 않았다.

이번 source 결과만으로 번호 정보가 문제를 해결했다고 말할 수 없다. contrast·부위 정체성·미세 cue·3×3 압축·SAM 등 다른 원인의 인과는 이 대조로 확정하지 않는다. 새 실사 효과는 아직 측정하지 않았다. 후속 V200에서는 C58/C59 두 팔 모두 같은 고정 real 평가에 넣도록 사전 고정하며, 이번 결과로 더 나은 팔·문턱·자료를 골라 진행하지 않는다. 실사 결과는 별도 phase의 평가일 뿐 이 source 보고서에 포함하지 않는다. 모델 미채택·전체 목표 미해결을 유지한다.

'''+ ' · '.join('['+label+']('+target+')'for label,target in links)+'\n'
    tables=dict(metrics=dict(headers=['분리','팔','known','truth UNKNOWN','D 정답/N','D precision','D recall','E 정답/N','E precision','E recall','known 예측 U','완전 D/E/D 정답/N'],rows=metric_rows),native=dict(headers=['native','TRAIN D/E','TRAIN UNKNOWN','TRAIN input U','HOLDOUT D/E','HOLDOUT UNKNOWN','C58 HOLD E정답/N','C59 HOLD E정답/N'],rows=native_rows),source_cohorts=dict(headers=['source domain','TRAIN 행','TRAIN input U','HOLDOUT 행','HOLDOUT input U'],rows=pop_rows))
    result=dict(completed=True,PASS=True,phase='V199',result_kind='SOURCE_ONLY_PAIRED_NATIVE_CONDITIONING_FIT_AND_GROUPED_SOURCE_READOUT',created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs=list(inputs.values()),trial=dict(arms=[dict(id=arm,use_sam=False,native_condition=bool(i))for i,arm in enumerate(ARMS)],parameters_each=116407,state_tensors_each=36,steps_each=8000,total_updates=16000,seed=1810,direct_rng_seed=1811,old_loss_draw192=True,direct_D16_E16=True,weight1=True,same_initial_state_and_draws=True,effective_function_class_identical=False,shared_backbone_can_change=True,all2_before_any_saved_outputs=True),cohort_denominators=cohorts,models={arm:{k:value for k,value in models[arm].items()if k!='paired_D_E_D_triplets'}for arm in ARMS},paired_D_E_D={arm:{role:{k:value for k,value in q.items()if k!='triplets'}for role,q in models[arm]['paired_D_E_D_triplets'].items()}for arm in ARMS},HOLDOUT_bases=bases,HOLDOUT_direct_known_synthetic_only_real_end0=True,TRAIN_apparent_not_validation=True,source_design_and_holdout_previously_exposed_not_freshblind=True,native_class_prior_census=prior,RAW_identity_collision_census=collisions,full_attempt_native_groups=dict(all1160=True,known_complete_DED=124,complete_unknown=868,incomplete_all_unknown=168,no_extra_known_labels_from_incomplete=True),tables=tables,figure=dict(PNG=dict(path=str(png),sha256=hashlib.sha256(png.read_bytes()).hexdigest()),SVG=dict(path=str(svg),sha256=hashlib.sha256(svg.read_bytes()).hexdigest()),scalar_summary_only=True,source_images_decoded=0),links=[dict(label=label,target=target)for label,target in links],conclusion='Native onehot did not improve this fixed source HOLDOUT over the matched zero8 control. No general impossibility, other failure cause, real-effect or adoption claim.',next=dict(phase='V200',both_arms_always_frozen_real_evaluation=True,selection_from_source_or_real_results=False,actual_real_effect_read_in_this_report=False),report_generation_access=dict(saved_generated_JSON_and_closed_reports_only=True,source_input_target_arrays_or_headers=0,weights_logs_or_source_pixels_reopened=0,model_instantiation_forward_fit=0,protected_real_or_HOLDOUT_body_reopened=0,new_scientific_chart_only=True),selected_model=None,model_adopted=False,fully_solved=False,report_QA_pending_root_and_independent=True)
    data=(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();write(V/'ROOT_RESULTS_KO.json',data);write(V/'ROOT_REPORT_KO.md',md);write(V/'PORTABLE_REPORT_KO.md',md);write(V/'ROOT_READOUT_KO.html',render_md(md))
    # Verify only output text links exist; never follow into arrays/weight bodies.
    for label,target in links:
        path=Path(os.path.normpath(str(V/target)));need('v112'not in str(path).casefold()and path.is_file()and not path.is_symlink(),'Existing relative report target '+target)
    print(json.dumps(dict(completed=True,PASS=True,files={name:dict(path=str(V/name),sha256=hashlib.sha256((V/name).read_bytes()).hexdigest())for name in ['ROOT_RESULTS_KO.json','ROOT_REPORT_KO.md','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html']},figure=result['figure'],relative_document_links=len(links)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
