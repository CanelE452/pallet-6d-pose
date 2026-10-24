"""Build report from sealed scalar source readout only; no NumPy/model inputs."""
from pathlib import Path
from html import escape
import csv,hashlib,json,os,re,stat
H=Path(__file__).absolute().parent;V=H.parent
os.environ['MPLCONFIGDIR']=str(H/'matplotlib_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
E=V/'saved_source_direct_end_evaluation_v2_PARTIAL_UNKNOWN_PRESERVED'
EXPECTED_COMPLETE='b97ee7af1f7c187ff18c4439ff00f16ab452a94baaefe7780679bee01a580dd6'
REVIEW={'path':str(V/'independent_saved_fit_review_c/actual_review_v1/REVIEW.json'),'sha256':'7def791a9f153bc8491825a3af7406e1871718e97f9354a7411b2c536d3e168f'}
ARMS=('C60_RAW96_NATIVE_MASK_ZERO_XY_NOMINAL_CONTROL','C61_RAW96_NATIVE_MASK_Q8_XY_NOMINAL');SHORT={ARMS[0]:'C60 · zero XY',ARMS[1]:'C61 · original q8 XY'}

def safe(p):
    p=Path(p);assert p.is_absolute()and'v112'not in str(p).casefold()and'..'not in p.parts
    cur=Path(p.anchor)
    for part in p.parts[1:]:cur/=part;assert not stat.S_ISLNK(cur.lstat().st_mode)
    return p

def read_bound(b):
    p=safe(b['path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==b['sha256'];return json.loads(raw)
def binding(p):
    p=safe(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,s):
    assert not p.exists()
    with p.open('x')as f:f.write(s)
def jwrite(p,o):write(p,json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def percent(v):return '—'if v is None else f'{100*v:.2f}%'
def cmtext(m):return str(m['CM_D_E_by_D_E_U']).replace(' ','')
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n'for r in rows)

def figure(m):
    fig,axs=plt.subplots(2,4,figsize=(15.5,7),gridspec_kw={'width_ratios':[1,1,1,1.2]});fig.subplots_adjust(left=.055,right=.985,top=.86,bottom=.12,hspace=.65,wspace=.42)
    cohorts=('FULL','TRAIN','HOLDOUT')
    for row,arm in enumerate(ARMS):
        for col,cohort in enumerate(cohorts):
            rec=m['cohorts'][cohort]['all']['models'][arm];cm=rec['CM_D_E_by_D_E_U'];norm=[[v/sum(rr)if sum(rr)else 0 for v in rr]for rr in cm];ax=axs[row,col];ax.imshow(norm,vmin=0,vmax=1,cmap='Blues',aspect='auto')
            for y,rr in enumerate(cm):
                for x,v in enumerate(rr):ax.text(x,y,str(v),ha='center',va='center',fontsize=13,color='white'if norm[y][x]>.6 else'black',fontweight='bold')
            ax.set_xticks([0,1,2],['D','E','U']);ax.set_yticks([0,1],['weak D','weak E']);ax.set_xlabel('Prediction');ax.set_title(f'{arm[:3]} / {cohort}\nD={rec["D"]}, E={rec["E"]}',fontsize=11)
    ax=axs[0,3];xs=list(range(3));recovered=[m['cohorts'][c]['all']['paired_correctness']['recovered_by_XY']for c in cohorts];damaged=[m['cohorts'][c]['all']['paired_correctness']['damaged_by_XY']for c in cohorts]
    ax.bar([x-.18 for x in xs],recovered,.36,label='Recovered',color='#177c74');ax.bar([x+.18 for x in xs],damaged,.36,label='Damaged',color='#ba5b36')
    for x,a,b in zip(xs,recovered,damaged):ax.text(x-.18,a+.08,str(a),ha='center');ax.text(x+.18,b+.08,str(b),ha='center')
    ax.set_xticks(xs,cohorts,fontsize=8);ax.set_ylim(0,max(recovered+damaged)+1.5);ax.set_ylabel('Known-row count');ax.set_title('C60 -> C61 paired changes');ax.legend(fontsize=8,frameon=False)
    ax=axs[1,3];den=[m['triplets'][c]['known_D_E_D_triplets']for c in ('TRAIN','HOLDOUT')]
    for j,arm in enumerate(ARMS):
        vals=[m['triplets'][c]['models'][arm]['all3_correct']for c in ('TRAIN','HOLDOUT')];xx=[x+(-.18 if j==0 else.18)for x in range(2)];ax.bar(xx,[a/b for a,b in zip(vals,den)],.36,label=arm[:3],color=('#4477aa','#cc6677')[j])
        for x,v,n in zip(xx,vals,den):ax.text(x,v/n+.025,f'{v}/{n}',ha='center',fontsize=9)
    ax.set_xticks([0,1],['TRAIN','HOLDOUT']);ax.set_ylim(0,1.22);ax.set_ylabel('All 3 correct / known triplets');ax.set_title('BEFORE D / COVER E / SHAM D');ax.legend(fontsize=8,frameon=False)
    fig.suptitle('V202 source weak finite-end readout | 2 x 4000 updates | RAW + native + same mask',fontsize=15,y=.975)
    fig.text(.5,.925,'C60: zero other-query XY    C61: original-initializer relative XY    |    118,359 parameters each',ha='center',fontsize=11)
    fig.text(.5,.035,'Counts are source weak targets, not human native gold. HOLDOUT: 45 known / 15 triplets / 6 synthetic bases; real known = 0. No adoption.',ha='center',fontsize=9)
    png=H/'SOURCE_DIRECT_CM_AND_PAIRED.png';svg=H/'SOURCE_DIRECT_CM_AND_PAIRED.svg';fig.savefig(png,dpi=180);fig.savefig(svg,metadata={'Date':None});plt.close(fig);return png,svg

def build():
    complete=read_bound(dict(path=str(E/'COMPLETE.json'),sha256=EXPECTED_COMPLETE));m=read_bound(complete['outputs']['metrics']);audit=read_bound(REVIEW);assert audit['completed']is True and audit['PASS']is True
    sources={k:complete['outputs'][k]for k in('metrics','table','ledger','triplets')}
    for b in sources.values():safe(b['path']);assert hashlib.sha256(Path(b['path']).read_bytes()).hexdigest()==b['sha256']
    ledger=[json.loads(x)for x in Path(sources['ledger']['path']).read_bytes().splitlines()];assert len(ledger)==3144
    held={r['attempt_id']for r in ledger if r['role']=='HOLDOUT'and r['direct_weak_target']>=0};bases={re.search(r'BASE(\d+)',x).group(1)for x in held};assert len(held)==15 and len(bases)==6
    assert m['cohorts']['HOLDOUT']['all']['paired_correctness']['net_correct_change']==0
    png,svg=figure(m)
    result=dict(completed=True,report_only=True,phase='V202',source_only=True,headline='source HOLDOUT E정답4→6, D정답28→26; 정답합계32/45→32/45, 순증0. 실사 개선 근거 아님.',arms=list(ARMS),architecture_parameters_each=118359,updates_each=4000,total_updates=8000,runtime_seconds=382.95,runtime_authority='Root actual fit completion message',source_evaluation_complete=binding(E/'COMPLETE.json'),source_fit_complete=complete['source_fit_complete'],all2_freeze=complete['all2_freeze'],independent_actual_fit_audit=dict(status='PASS',binding=REVIEW),saved_source_bindings=sources,cohorts=m['cohorts'],triplets=m['triplets'],HOLDOUT_known_synthetic_bases=6,HOLDOUT_known_triplets=15,HOLDOUT_real_known=0,original_rows=3144,original_inputU=112,target_UNKNOWN=2772,full_source_rows_each=73976,source_context='Original q/L/native/U and pixels unchanged. RAW+native+mask same; only other-initializer XY is zero versus real. Both use NOMINAL eligibility; original component roles preserved.',limitations=['weak finite-end D/E/D is not human native D/E gold or exact physical registration certification','TRAIN metrics are apparent; HOLDOUT is already exposed source design, synthetic six bases only, not freshblind/natural-real generalization','known predictionU0 does not solve uncertainty; 2772 targetUNKNOWN and112 inputU retained','V199 scores are not compared as causal q8 effect because recipe/budget/fit eligibility changed','No data/model/threshold/checkpoint/arm choice or adoption; both arms predeclared to later V203 real benchmark'],known_failure=dict(v1='Legal BEFORE-only UNKNOWN incorrectly required all3variants; v1 not executed on source by root',v2='Known D/E/D requires all3; allUNKNOWN partial variants retained; 1160native-attemptgroups=992complete+168partial; all3144rows preserved'),figure=dict(png=binding(png),svg=binding(svg)),model_adopted=False,fully_solved=False,new_forward_fit_or_input_array_reads_for_report=0)
    jwrite(V/'ROOT_RESULTS.json',result)
    rows=[]
    for cohort in('FULL','TRAIN','HOLDOUT'):
        for arm in ARMS:
            r=m['cohorts'][cohort]['all']['models'][arm];rows.append([cohort,arm[:3],f'{r["D"]}/{r["E"]}',cmtext(r),percent(r['D_precision']),percent(r['E_precision']),percent(r['D_recall']),percent(r['E_recall']),r['known_prediction_U']])
    primary=table(['분모','팔','D/E수','CM [[D→D,E,U],[E→D,E,U]]','D P','E P','D R','E R','known예측U'],rows)
    domains=[]
    for cohort in('TRAIN','HOLDOUT'):
        for domain,z in m['cohorts'][cohort]['by_domain'].items():
            for arm in ARMS:
                r=z['models'][arm];domains.append([cohort,domain,arm[:3],r['known'],cmtext(r),percent(r['D_recall']),percent(r['E_recall'])])
    domain_table=table(['분모','source domain','팔','known','CM D/E×D/E/U','D R','E R'],domains)
    natives=[]
    for k,z in m['cohorts']['HOLDOUT']['by_native'].items():
        a,b=[z['models'][arm]for arm in ARMS];natives.append([k,f'{a["D"]}/{a["E"]}',cmtext(a),cmtext(b),percent(a['E_recall']),percent(b['E_recall']),a['target_UNKNOWN']])
    native_table=table(['native','H D/E수','C60 CM','C61 CM','C60 E R','C61 E R','H targetUNKNOWN'],natives)
    pairtable=table(['분모','known행','회복','손상','순증','known triplet','C60 all3','C61 all3'],[[c,m['cohorts'][c]['all']['paired_correctness']['known'],m['cohorts'][c]['all']['paired_correctness']['recovered_by_XY'],m['cohorts'][c]['all']['paired_correctness']['damaged_by_XY'],m['cohorts'][c]['all']['paired_correctness']['net_correct_change'],m['triplets'][c]['known_D_E_D_triplets'],m['triplets'][c]['models'][ARMS[0]]['all3_correct'],m['triplets'][c]['models'][ARMS[1]]['all3_correct']]for c in('FULL','TRAIN','HOLDOUT')])
    body='# V202 · 원 initial q8 공간문맥 source 비교\n\n**Source HOLDOUT에서 E 정답은 4/15→6/15, D 정답은 28/30→26/30입니다. 합계는 두 팔 모두32/45로 순증0입니다. 실사 개선을 뜻하지 않으며 두 팔을 모두 후속 V203 고정 평가로 넘깁니다.**\n\n![source CM와 paired 비교](report_b/SOURCE_DIRECT_CM_AND_PAIRED.png)\n\n두 RAW 팔은 원 backbone·shared64·direct scalar에 같은 zero-final residual MLP를 더한118,359파라미터 구조입니다. Native onehot8과 다른query의 원가용성mask8은 같고, C60은 다른query XY16을0으로, C61은 원 initializer q8의 상대좌표를 제공합니다. F64 차/나눗셈 후[-8,8] clip→FP32이며 self XY/mask는0입니다. q/L/native/U·원RGB96과 target은 바뀌지 않았습니다.\n\n각4,000회·총8,000회 from-scratch, seed1810/direct RNG1811, 원192개5loss평균+별도 D16/E16 direct loss 가중치1, AdamW lr3e-4/wd1e-4/clip5를 고정했습니다. 두 final4000을 먼저 동결한 후 source 출력을 저장했고 실제 fit시간은382.95초입니다. 모든 loss에 원 component TRAIN mask와 별도 NOMINAL eligibility를 교차했습니다. 제외 SELF 비NOMINAL 행은 원 TRAIN 역할을 유지하며 HOLDOUT으로 바뀌지 않습니다. V199와는 recipe·budget·fit eligibility가 달라 이전 점수와의 차이를 q8의 인과효과로 해석하지 않습니다.\n\n## 원분모와 해석범위\n\n모든source domain 입력행73,976/팔을 보존합니다. 이 보고서의 primary는 DIRECT_CONTROL3,144행 중 기존 weak finite-end known372(D248/E124)이며 TRAIN327(D218/E109), HOLDOUT45(D30/E15)입니다. Known HOLDOUT은 이미 노출된 synthetic6base·15triplet뿐이며 real-end HOLDOUT known은0입니다. TRAIN에는 public real weak 감독도 포함됩니다. Weak finite-end D/E/D는 인간 native D/E gold나 물리 exactvertex 인증과 같지 않습니다.\n\nTarget UNKNOWN2,772와 원 inputU112를 별도로 보존합니다. HOLDOUT360행 중 known45·UNKNOWN315·inputU0입니다. known predictionU0은 불확실성이 해결됐다는 뜻이 아닙니다. HOLDOUT native4/6/7의 directknown은0이며, 이것을 canonical6/7 부재로 바꾸지 않습니다. Native6/7 directknown 부재만으로 관련 D/E coverage 실패라고 단정하지 않으며 upstream SELF·canonical 대응을 이 표에서 재분류하지 않습니다.\n\n## D/E 혼동행렬·precision/recall\n\n'+primary+'\nTRAIN은 학습자료 apparent 성능입니다. 정사각형/목재 실사 성능이나 새로운 자연실사 일반화 결론을 이 수치로 내리지 않습니다.\n\n## Paired 변화와 D/E/D 통제\n\n'+pairtable+'\nHOLDOUT known행은 C61이3개를 회복하고3개를 손상해 순증0입니다. 세 변형 전체가 맞는 triplet은3/15→4/15이며, triplet 기준 회복2·손상1입니다. 이 변화는 소수의 이미 알려진 source 통제에서의 연관이며 채택·모델선택 근거로 사용하지 않습니다.\n\n## Source domain 분리\n\n'+domain_table+'\n## 원 native별 HOLDOUT\n\n'+native_table+'\nFULL/TRAIN/HOLD의 모든 native·authority·UNKNOWN 예측분포 원표는 [저장 metrics JSON](saved_source_direct_end_evaluation_v2_PARTIAL_UNKNOWN_PRESERVED/SOURCE_DIRECT_WEAK_METRICS.json) 및 [CSV](saved_source_direct_end_evaluation_v2_PARTIAL_UNKNOWN_PRESERVED/SOURCE_DIRECT_WEAK_CONFUSIONS_AND_STRATA.csv)에 있습니다.\n\n## 보존된 실패와 검산\n\n초기 evaluator v1은 합법적인 BEFORE-only UNKNOWN에도 세 변형을 요구하는 guard 문제가 있어 실제 평가에 사용하지 않았습니다. v2는 anyknown이면 원D/E/D 세 변형을 요구하고, allUNKNOWN은 present/missing 변형을 그대로 보존합니다. 원3,144행은 native-attempt1,160묶음(complete992+partialUNKNOWN168)으로 남았고 누락 COVER/SHAM을 만들지 않았습니다. [v1 보존·수정기록](source_evaluation_preparation_b_v2/V1_PRESERVATION_AND_MINIMAL_REPAIR.json), [A 독립 preflight](evaluation_preflight_a/REVIEW.json), [평가 COMPLETE](saved_source_direct_end_evaluation_v2_PARTIAL_UNKNOWN_PRESERVED/COMPLETE.json)를 연결합니다.\n\nC 실제 saved-fit 감사는 [PASS REVIEW](independent_saved_fit_review_c/actual_review_v1/REVIEW.json)로 확인됐습니다. C는 drawindices/hash와 logged scalar loss 산술·finitegrad, checkpoint2·sourcepacket24·META·저장수치를 검사했습니다. 신경망 loss/gradient값·activationtrace를 재실행한 전체훈련 독립재현은 아닙니다. 이 보고서는 저장 scalar/CSV/rowledger만 읽었고 모델·가중치·입력배열·새픽셀·실사/held/VAL/COMMON·새forward를 열지 않았습니다. 그림은 저장수치에서 만든 도식이며 실제source 이미지 또는 browser render를 보았다는 주장이 아닙니다. [원상세 JSON](ROOT_RESULTS.json), [SVG](report_b/SOURCE_DIRECT_CM_AND_PAIRED.svg), [보고서 생성 코드](report_b/build_source_report.py), [사실검사](report_b/FACTUAL_QA.json).\n\n**두 팔 모두 후속 고정 실사 비교로 진행하며, 데이터·문턱·checkpoint·팔 선택이나 모델 채택은 하지 않았습니다. 전체 목표는 미해결입니다.**\n'
    write(V/'PORTABLE_REPORT_KO.md',body)
    # Standalone offline HTML: escaped report text preserves all prose/tables; actual href/src only link known artifacts.
    links=[('PORTABLE_REPORT_KO.md','원문 Markdown'),('ROOT_RESULTS.json','상세 JSON'),('report_b/SOURCE_DIRECT_CM_AND_PAIRED.svg','SVG'),('saved_source_direct_end_evaluation_v2_PARTIAL_UNKNOWN_PRESERVED/SOURCE_DIRECT_WEAK_CONFUSIONS_AND_STRATA.csv','저장 CSV'),('independent_saved_fit_review_c/actual_review_v1/REVIEW.json','C actual audit PASS')]
    html='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>V202 source q8 readout</title><style>body{max-width:1220px;margin:30px auto;padding:0 20px;font:16px/1.7 system-ui;color:#17212a}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f9fb;padding:18px;font:14px/1.7 ui-monospace,monospace}a{color:#075985}</style><h1>V202 원 initial q8 source 비교</h1><p><b>HOLDOUT E 4/15→6/15, D 28/30→26/30, 합계32/45→32/45. 실사 개선·채택 결론 없음.</b></p><img src="report_b/SOURCE_DIRECT_CM_AND_PAIRED.png" alt="Source weak D/E CM and paired comparison"><p>'+ ' · '.join('<a href="'+a+'">'+escape(b)+'</a>'for a,b in links)+'</p><pre>'+escape(body)+'</pre></html>\n'
    write(V/'ROOT_READOUT_KO.html',html)
    jwrite(H/'REPORT_BUILD_COMPLETE.json',dict(completed=True,PASS=True,code=binding(__file__),source_complete=binding(E/'COMPLETE.json'),source_metrics=sources['metrics'],independent_actual_fit_audit=REVIEW,outputs={k:binding(V/k)for k in ['ROOT_RESULTS.json','PORTABLE_REPORT_KO.md','ROOT_READOUT_KO.html']},figure=dict(png=binding(png),svg=binding(svg)),no_input_model_pixel_private_or_new_eval=True,scientific_phase_frozen=False))
    print(json.dumps(dict(completed=True,PASS=True,build_complete=binding(H/'REPORT_BUILD_COMPLETE.json')),ensure_ascii=False))

if __name__=='__main__':build()
