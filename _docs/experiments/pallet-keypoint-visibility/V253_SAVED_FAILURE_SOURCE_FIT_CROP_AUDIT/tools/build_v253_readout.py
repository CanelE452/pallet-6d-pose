"""Render only saved Source diagnosis tables and fixed FIT case links."""
from pathlib import Path
import csv,hashlib,html,io,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).absolute().parent
H=R/'V253_ROOT_SAVED_FAILURE_DIAGNOSIS'
D=H/'saved_failure_diagnosis_v1'

def safe(p):
    p=Path(p)
    if not p.is_absolute() or '..' in p.parts or any(t in str(p).lower() for t in ('v112','cache','.pyc')):raise ValueError(str(p))
    for a in reversed([p,*p.parents]):
        if a.is_symlink():raise ValueError('Artifact symlink')
    return p
def raw(p):return safe(p).read_bytes()
def load(p):return json.loads(raw(p))
def bound(b):
    x=raw(b['path'])
    if len(x)!=b['size_bytes'] or hashlib.sha256(x).hexdigest()!=b['sha256']:raise ValueError('Saved output binding')
    return x
def write(p,s):
    with safe(p).open('xb') as f:f.write(s.encode())
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
def htable(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
def pct(x):return 'NA' if x is None else f'{100*x:.2f}%'

def run():
    complete=load(D/'COMPLETE.json');assert complete['completed'] and complete['PASS']
    summary=json.loads(bound(complete['outputs']['SUMMARY.json']));selection=json.loads(bound(complete['outputs']['FIT_TILE_SELECTION.json']))
    event=list(csv.DictReader(io.StringIO(bound(complete['outputs']['FIXED_D_DAMAGE_COUNTS.csv']).decode())))
    stats=list(csv.DictReader(io.StringIO(bound(complete['outputs']['REFERENCE_DISTRIBUTIONS.csv']).decode())))
    crops=load(H/'source_fit_crop_cards_v1/COMPLETE.json');assert crops['completed'] and crops['PASS']
    def row(f,role,native=''):return next(r for r in event if r['family']==str(f) and r['role']==role and r['original_native_address']==str(native))
    native=[]
    for k in range(8):
        a=row(0,'FIT',k);b=row(0,'DEV',k)
        native.append([k,int(a['D']),int(a['E']),int(a['D_TREATfalseE_total']),pct(float(a['D_TREATfalseE_rate_originalD'])),int(b['D']),int(b['E']),int(b['D_TREATfalseE_total']),pct(float(b['D_TREATfalseE_rate_originalD']))])
    changes=[]
    for f in (0,1):
        for role in ('FIT','DEV'):
            r=row(f,role);changes.append(['SYN MAIN' if f==0 else 'YCB AUX',role,int(r['D']),int(r['D_TREATfalseE_total']),int(r['D_BASEcorrect_to_TREATfalseE_new']),int(r['D_BASEfalseE_to_TREATfalseE_existing']),int(r['E_BASEmiss_to_TREATcorrect']),int(r['E_BASEcorrect_to_TREATmiss'])])
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),constrained_layout=True)
    x=np.arange(8)
    for role,shift,color in [('FIT',-.18,'#397da5'),('DEV',.18,'#d47d44')]:
        rr=[row(0,role,k) for k in range(8)];v=[float(r['D_TREATfalseE_rate_originalD']) for r in rr]
        axes[0].bar(x+shift,v,width=.35,label=role,color=color)
        for xx,yy in zip(x+shift,v):axes[0].text(xx,yy+.009,f'{yy*100:.1f}',ha='center',fontsize=8)
        axes[1].bar(x+shift,[int(r['E']) for r in rr],width=.35,label=role,color=color)
    axes[0].set(title='SYN TREAT: visible D predicted E',ylabel='False E / all original D',ylim=(0,.52));axes[1].set(title='SYN E label support',ylabel='Number of inherited E rows')
    for ax in axes:ax.set_xticks(x);ax.set_xlabel('Original native input address (not new physical gold)');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend()
    fig.suptitle('V253 saved diagnosis: native addresses 0/1 dominate false E; E support differs')
    fig.savefig(safe(H/'NATIVE_ADDRESS_FAILURE_AND_SUPPORT.png'),dpi=165);plt.close(fig)
    for score,name in [('selectedcos','TARGET_SELECTED_COSINE_ALL_VIEWS.png'),('margin','TARGET_MARGIN_ALL_VIEWS.png')]:
        fig,axes=plt.subplots(2,3,figsize=(12.5,7),constrained_layout=True)
        for f in (0,1):
            for j,view in enumerate(('full_CLS','fixed_centre_Q','centre_CLS')):
                boxes=[]
                for role in ('FIT','DEV'):
                    for truth in ('D','E'):
                        r=next(r for r in stats if r['family']==str(f) and r['role']==role and r['original_native_address']=='' and r['truth_filter']==truth and r['failure_group']=='ALL' and r['reference']=='TARGET' and r['view']==view and r['score']==score)
                        boxes.append(dict(label=role+' '+truth,whislo=float(r['p10']),q1=float(r['p25']),med=float(r['median']),q3=float(r['p75']),whishi=float(r['p90']),fliers=[]))
                ax=axes[f,j];ax.bxp(boxes,showfliers=False);ax.set_title(('SYN MAIN' if f==0 else 'YCB AUX')+' / '+view);ax.set_ylabel('Selected cosine' if score=='selectedcos' else 'Selected minus fixed next cosine');ax.grid(axis='y',alpha=.15)
        fig.suptitle('All fixed views; boxes p25/median/p75, whiskers p10/p90; descriptive only\n'+('Target reference similarity' if score=='selectedcos' else 'Fixed-next contrast is not a certified wrong target'))
        fig.savefig(safe(H/name),dpi=165);plt.close(fig)
    dev=row(0,'DEV');fit=row(0,'FIT');topdev=sum(int(row(0,'DEV',k)['D_TREATfalseE_total']) for k in (0,1));topfit=sum(int(row(0,'FIT',k)['D_TREATfalseE_total']) for k in (0,1))
    paras=[
        f"V252의 실패는 실사 전이에서만 생긴 것이 아닙니다. 합성 학습 FIT의 보이는 D4,565개 중 {fit['D_TREATfalseE_total']}개({pct(float(fit['D_TREATfalseE_rate_originalD']))})를 이미 E로 판정했습니다. 합성 DEV에서는 D1,288개 중 {dev['D_TREATfalseE_total']}개를 E로 판정했고, 이는 기존 BASE D정답을 새로 틀린 {dev['D_BASEcorrect_to_TREATfalseE_new']}개와 기존부터 E오판이던 {dev['D_BASEfalseE_to_TREATfalseE_existing']}개로 나뉩니다.",
        f"오판은 원 입력 주소0·1에 집중됩니다. 두 주소가 SYN DEV falseE {topdev}/{dev['D_TREATfalseE_total']}개({100*topdev/int(dev['D_TREATfalseE_total']):.2f}%)와 FIT falseE {topfit}/{fit['D_TREATfalseE_total']}개를 차지합니다. 주소6·7은 FIT E가 각각2개·4개, DEV E가 모두0개입니다. 따라서 전체 SYN 평균 성능으로 모든 부위의 가림 판정을 입증할 수 없습니다.",
        "고정 FIT 사례 S000057·S000119는 목재처럼 보이는 모서리를 E로 오판했고, 같은 주소의 S000028·S000029는 어두운 사각 구조를 D로 판정했습니다. 이는 선택된 사례의 시각 관찰이며 재질 정답이나 전체 데이터의 재질별 오판율이 아닙니다. 단일 평균 참조가 형태·시점 차이를 충분히 보존하는지 의심할 근거가 되지만, 재질 편향이나 원인을 확정하지 않습니다.",
        "S000035와 S000755처럼 팔레트 표면이 보이면서 inherited E인 사례도 보존했습니다. 추가 CODE 추적으로 SYN target-alone은 전체 등록 pallet 기하이며, witness ray의 앞을 별도 foreign BOX(owner999)가 막을 때 E임을 확인했습니다. 자기 기하가 막는 상태는 S로 구분하므로 SELF를 E에 합친 규칙은 아닙니다. point0는 각 native record의 첫 witness이며 physical native0의 별칭이 아닙니다. 이 정답을 q 중심 한 픽셀·부위 전체·crop 안의 팔레트 존재·실사 native 정답으로 확대하지 않습니다. YCB는 associated-r 주변 라벨입니다. 카드만 보고 라벨 오류나 새 물리 부위 정답을 만들지 않았습니다.",
        "YCB 라벨은 output1의 마지막 readout을 직접 감독하고, 팔레트용 SYN output0에는 공유 trunk를 통해서만 영향을 줍니다. YCB associated-r AUX 개선은 실제 팔레트 MAIN 개선을 입증하지 않습니다. 공통 출력·native residual·COVER/SHAM은 이전 CODE에 이미 존재하므로 단순히 처음 시도하는 해결책이라고 설명하지 않습니다.",
        f"36개 고정 셀에서 sample_index가 작은 두 개까지 선택해 FIT {selection['unique_selected_rows']}개를 표시했고 빈 셀 {selection['empty_cells']}개를 유지했습니다. 원96 RGB·valid·선택 miniNPZ·픽셀 hash를 함께 보존했습니다. q 표식은 원96 좌표(47.5,47.5)의 별도 표시이며 입력·좌표를 보정하지 않습니다. 원 Source 배열 전체의 hash에는 DEV bytes도 포함되지만 숫자 pixel 접근은 선택 FIT66행뿐입니다.",
        "이번에는 저장 점수·참조와 Source에서 유래한 라벨·입력 조건을 읽었습니다. 새 학습·모델/CP forward·bank 적합·문턱 선택·현재/보호 평가 자료·PnP는0입니다. S/U7,423행은 E나 성공 보류로 바꾸지 않았습니다. Source DEV는 반복 노출된 검증 자료입니다. 이 진단으로 모델 성능이나 현재 팔레트 전이가 개선됐다고 주장하지 않습니다.",
        "다음에 입증해야 할 것은 원 q/L에서 의도한 대상 부위에 대한 반응과 참조의 형태·시점 대응입니다. 기존 Source COVER/SHAM의 정확한 frozen input과 원 q/L·현재 입력 변환의 대응부터 닫아야 합니다. 이번에는 새 recipe를 선택하거나 현재 평가에 진입하지 않았습니다."
    ]
    visual=load(H/'VERIFIED_REPORT_LINKS.json');links=visual['links']
    nh=['원 주소','FIT D','FIT E','FIT falseE','FIT falseE/D','DEV D','DEV E','DEV falseE','DEV falseE/D']
    ch=['라벨 권한','분할','원 D','TREAT falseE 전체','BASE정답→새오판','기존오판 유지','E 새검출','E 새누락']
    images=[('NATIVE_ADDRESS_FAILURE_AND_SUPPORT.png','주소별 오판과 가림 정답 수'),('TARGET_SELECTED_COSINE_ALL_VIEWS.png','세 view 대상 참조 cosine 분포'),('TARGET_MARGIN_ALL_VIEWS.png','세 view 고정 next 대비 margin 분포')]
    md='# V253 — 저장 오판·Source FIT crop·라벨 연결 진단\n\n'+'\n\n'.join(paras)+'\n\n## 기존 정답의 손상과 회복\n\n'+table(ch,changes)+'\n## 원 입력 주소별 전체 분모\n\n'+table(nh,native)+'\n'+''.join(f'\n![{a}]({p})\n' for p,a in images)+'\n## 전체 고정 FIT 사례와 재현 자료\n\n'+'\n'.join(f'- [{a}]({p})' for a,p in links)+'\n'
    write(H/'PUBLIC_README_KO.md',md)
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V253 저장 오판·FIT crop 진단</title><style>body{font:16px/1.7 system-ui;max-width:1280px;margin:30px auto;padding:20px;color:#1c303c;background:#f7f9fb}.scroll{overflow:auto}img{max-width:100%}table{border-collapse:collapse;background:white;font-size:13px;width:100%}td,th{border:1px solid #d4dee4;padding:8px;white-space:nowrap}th{background:#e5edf2}a{color:#126a86}</style><h1>V253: 저장 오판과 Source FIT crop 진단</h1>'+''.join('<p>'+html.escape(p)+'</p>' for p in paras)+'<h2>손상과 회복</h2>'+htable(ch,changes)+'<h2>원 입력 주소별 전체 분모</h2>'+htable(nh,native)+''.join('<h2>'+a+'</h2><img src="'+p+'" alt="'+a+'">' for p,a in images)+'<h2>고정 FIT 사례와 재현 자료</h2><ul>'+''.join('<li><a href="'+p+'">'+html.escape(a)+'</a></li>' for a,p in links)+'</ul></html>\n'
    write(H/'ROOT_READOUT_KO.html',page)
    write(H/'PUBLIC_READOUT_TABLES.json',json.dumps({'native_headers':nh,'native_rows':native,'transition_headers':ch,'transition_rows':changes,'top_addresses':[0,1],'SYN_DEV_falseE_top':topdev,'SYN_FIT_falseE_top':topfit,'selected_FIT':selection['unique_selected_rows'],'empty_cells':selection['empty_cells'],'new_model_training':0},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'completed':True,'PASS':True,'SYN_DEV_falseE':int(dev['D_TREATfalseE_total']),'new_damage':int(dev['D_BASEcorrect_to_TREATfalseE_new']),'selected_FIT':selection['unique_selected_rows'],'new_training':0}))

if __name__=='__main__':run()
