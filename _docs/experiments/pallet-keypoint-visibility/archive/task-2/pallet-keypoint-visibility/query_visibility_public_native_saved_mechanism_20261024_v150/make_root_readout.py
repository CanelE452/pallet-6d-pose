"""Render saved source evidence and label-fidelity findings; no models or RGB."""
from pathlib import Path
import json,hashlib,html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
V=Path(__file__).resolve().parent;P=V.parent
def read(p):return json.loads(p.read_text())
def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def write(p,x):
    with p.open('x')as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def table(head,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>'for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>'for x in r)+'</tr>'for r in rows)+'</table></div>'
if __name__=='__main__':
    r=read(V/'source_saved_diagnosis/RESULTS.json');s=read(V/'ROOT_PUBLIC_SCALAR_DIAGNOSIS.json');review=read(V/'independent_saved_review/REVIEW.json')
    assert r['completed']and r['PASS']and review['PASS']and s['PASS']
    byfold=r['fold_headlines'];rows=[]
    for pool in ['OLD_NATIVE','NEW_NATIVE','PUBLIC_NATIVE_WEAK']:
        for fold in ['TRAIN','VALIDATION']:
            key=pool+'::'+fold
            if key not in byfold:continue
            a=byfold[key]['states'];x={}
            for state in ['D','E']:
                x[state]=a[state]['comparisons']['C33_FINAL2000->C34_FINAL2000']['delta_raw_logit']['mean']
            rows.append(dict(pool=pool,fold=fold,D=x['D'],E=x['E']))
    # Public names are retained exactly by the independent scalar readout.
    pub=s['C34_minus_C33_public_valid_shifts']
    groups=[('공개 TRAIN',pub['TRAIN_LAB::D']['mean'],pub['TRAIN_LAB::E']['mean']),('공개 HALL',pub['VAL_HALL::D']['mean'],pub['VAL_HALL::E']['mean'])]
    for pool,label in [('OLD_NATIVE','합성 OLD VAL'),('NEW_NATIVE','합성 NEW VAL')]:
        rr=next(x for x in rows if x['pool']==pool and x['fold']=='VALIDATION');groups.append((label,rr['D'],rr['E']))
    b=s['initial_public_TRAIN43_BCE'];top=max(s['initial_TRAIN43_all_rows'],key=lambda x:x['weighted_BCE'])
    assert top['global_index']==67
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf');plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False,'font.size':11})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    vals=[b['weighted_D'],b['weighted_E']];bars=axes[0].bar(['보임 D','외부 가림 E'],vals,color=['#168778','#c56b42']);axes[0].bar_label(bars,labels=[f'{x:.5f}'for x in vals],padding=3);axes[0].set_ylim(0,.16);axes[0].set_title('초기 공개 TRAIN43 · class-balanced BCE\nD가 loss의94.9%');axes[0].set_ylabel('각 class 가중 contribution')
    xx=list(range(len(groups)));w=.35
    axes[1].bar([x-w/2 for x in xx],[a[1]for a in groups],width=w,label='실제 D',color='#168778');axes[1].bar([x+w/2 for x in xx],[a[2]for a in groups],width=w,label='실제 E',color='#c56b42');axes[1].axhline(0,color='#667085',lw=1);axes[1].set_xticks(xx,[a[0]for a in groups]);axes[1].set_ylabel('평균 raw logit 차이 C34-C33\n음수: 보임 방향 / 양수: 가림 방향');axes[1].set_title('공개 안에서는 분리 개선 · 합성 변화는 작음');axes[1].legend()
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.suptitle('V150 · 저장 source 진단: 단순 공통 bias 원인으로 설명할 수 없음',fontsize=16)
    image=V/'SOURCE_MECHANISM_KO.png';fig.savefig(image,dpi=130,bbox_inches='tight');plt.close(fig)
    out=dict(completed=True,goal_status='active',full_goal_solved=False,source_only=True,selected_model=None,
        conclusion_KO='공개 동일 조립체 안에서는 D/E 분별이 개선됐지만 넓은 native 검증으로 이어지지 않았다. 공개 가림 종류와 역할의 범위가 좁고, 원 crop 밖의 scene 역할 보임도 같은 D 감독에 들어간다. 공통 bias만 막는 새 ranking fit보다 실제 팔레트 source의 역할·외부가림 범위를 확장하고 query 증거와 역할 상태를 분리할 근거를 확보한다.',
        all_source_rows=27136,all_native_base_slot_pairs=6656,native35_pass={'C32':20,'C33':22,'C34':22},
        public_initial_weighted_BCE=b,largest_initial_public_loss_row=top,largest_row_share=float(top['weighted_BCE']/b['total']),
        public_initial_scalar_bias_derivative=s['initial_public_TRAIN43_scalar_bias_derivative'],C34_C33_mean_logit_change=groups,
        label_fidelity={'known_D':70,'known_E':14,'unknown_U':428,'E_all_same_cardboard_native3_single_assembly':True,'TRAIN43':{'native2_D':19,'native3_D':16,'native3_E':8},'TRAIN_D_full96_reference_outside':2,'TRAIN_D_central32_reference_outside':19,'virtual_reference_not_exact_pixel_gold':True,'labels_changed_or_filtered':False},
        decisions={'new_ranking_fit':False,'reason':'All4 initial rank margin1 violations concern far-floor D67; would not add missing E types or resolve query-vs-role target gap.','next_source':'MR6D flat Euro pallet public dataset intake, native8 role/occlusion truth still to audit.','no_new_current_evaluation':True},
        evidence={k:bind(V/x)for k,x in [('saved_diagnosis','source_saved_diagnosis/DIAGNOSIS_SEAL.json'),('independent_saved_review','independent_saved_review/REVIEW.json'),('source_fidelity','source_fidelity_review/REVIEW_COMPLETE.json'),('new_public_source_review','public_source_gap_review/REVIEW_COMPLETE.json'),('root_scalar','ROOT_PUBLIC_SCALAR_DIAGNOSIS.json'),('plot','SOURCE_MECHANISM_KO.png')]},
        model_feature_checkpoint_or_new_fit_current223_PnP_calls=0,
        limitations=['Scalar derivative is not full-parameter causation.','Source crop-locality reference is virtual and provisional, not exact physical gold.','Public HALL is development exposed and limited to one assembly/cardboard.','No new model success or source qualification pass is claimed.'])
    write(V/'ROOT_RESULTS_KO.json',out)
    h='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V150 source 목표·학습 신호 진단</title><style>body{font:17px/1.7 sans-serif;max-width:1200px;margin:30px auto;padding:0 22px;color:#233044;background:#f7f9fc}h1{font-size:30px}h2{margin-top:32px}.lead{background:#eaf1fa;padding:18px;border-left:5px solid #4e7198}img{max-width:100%}table{border-collapse:collapse;background:white}td,th{border:1px solid #d8e0eb;padding:9px 13px}.scroll{overflow-x:auto}a{color:#245c97}.note{background:#fff2da;padding:15px}</style><body><h1>V150 · 공개 학습 신호와 crop 목표를 대조했습니다</h1>'''
    h+='<p class="lead">'+out['conclusion_KO']+'</p><img src="SOURCE_MECHANISM_KO.png" alt="공개 초기 loss와 공개/합성 클래스별 logit 변화">'
    h+='<h2>확인된 것과 단정할 수 없는 것</h2><p>초기 공개 loss는 D가94.9%를 차지하고, 공통 bias에 대한 D gradient 크기가 E의5.74배였습니다. 그러나 최종 공개 TRAIN에서는 D score가 평균−4.114, E score는+1.549로 갈라졌습니다. 따라서 실제 효과를 모든 점의 일괄 보임 이동으로 설명할 수 없습니다. 공개 안의 분별 개선과 다른 source로의 일반화 미개선을 구분해야 합니다.</p>'
    h+=table(['자료','D 평균 logit 변화','E 평균 logit 변화'],[[g[0],f'{g[1]:+.6f}',f'{g[2]:+.6f}']for g in groups])
    h+='<h2>공개 가림 유형의 범위</h2><p>512행 중 의미가 정해진 것은 D70/E14입니다. E14 모두 동일 목재 통합 조립체의 native3 하부를 종이상자가 가린 사례입니다. TRAIN43은 native2-D19, native3-D16, native3-E8이며, upper/rear는 U로 남아 있습니다. 이 자료가 다양한 자연 외부 가림을 대표하지는 않습니다.</p>'
    h+='<h2>실제로 crop이 바닥인 D 사례</h2><p>아래는 원 공개 프레임17입니다. 자홍 q3는 바닥으로 빗나갔고 원96 crop도 대부분 바닥입니다. 청록 R3는 제공 pose의 제한적 하부 역할 참조이며, 그 역할 자체는 보입니다. 기존 D 라벨은 이 장면의 역할 보임 판정이므로 원 q 픽셀의 대상 존재와 같은 정답으로 해석할 수 없습니다. 라벨을 E로 바꾸거나 이 행을 버리지 않았습니다.</p>'
    frame=P/'query_visibility_public_native_source_expansion_20261024_v146/saved_source64_contact_v1/FRAME_17_WHOLESCENE_AND_FIXED96.png'
    h+='<img src="'+str(frame)+'" alt="원공개 프레임17, 원query와 원96crop 및 별도하부참조">'
    h+='<p class="note">TRAIN D35 중 참조가 원 full96 밖인 행2개, central32 밖인 행19개가 있습니다. 이 거리는 virtual AABB 참조의 locality 진단이며 정확한 physical GT 오차가 아닙니다. 초기 rank margin1 위반4개는 모두 다른 far-floor D67 하나에 집중돼 있습니다.</p>'
    h+='<h2>다음 수집 행동</h2><p>실제 flat Euro pallet mesh·제공 K/pose·instance별 full/visible mask가 있는 MR6D 공개 source를 확인했습니다. 이 파일들은 native8 D/E가 자동으로 정답이라는 뜻은 아닙니다. 최소 원본 묶음을 고정 선택해 형상·축·객체 대응·외부/자기 가림의 주석 정의부터 검토합니다. 공식 validation/test를 학습에 쓰면 우리 source 개발 자료로 취급하며 MR6D blind benchmark 성능을 주장하지 않습니다.</p><p><a href="https://huggingface.co/datasets/anas-gouda/mr6d">MR6D 공식 데이터</a> · <a href="https://openaccess.thecvf.com/content/ICCV2025W/R6D/html/Gouda_MR6D_Benchmarking_6D_Pose_Estimation_for_Mobile_Robots_ICCVW_2025_paper.html">원 논문</a></p>'
    h+='<h2>저장 근거</h2><ul>'
    for label,path in [('전수27136행 source 진단',V/'source_saved_diagnosis/RESULTS.json'),('독립 전수 검산',V/'independent_saved_review/REVIEW.json'),('라벨·YCB 이력·ranking 한계',V/'source_fidelity_review/FINDINGS_KO.txt'),('공개 source 실제 획득 경로 검토',V/'public_source_gap_review/PUBLIC_SOURCE_GAP_AND_ACTION_EVIDENCE.json'),('독립 public43 scalar 수치',V/'ROOT_PUBLIC_SCALAR_DIAGNOSIS.json')]:h+='<li><a href="'+str(path)+'">'+label+'</a></li>'
    h+='</ul><p>새 학습·모델·현재223 재평가·PnP·좌표 보정은 없습니다. 전체 목표는 미해결이며 모델 채택도 없습니다.</p></body></html>'
    (V/'ROOT_READOUT_KO.html').write_text(h)
    print(json.dumps(dict(results=bind(V/'ROOT_RESULTS_KO.json'),plot=bind(image),report=bind(V/'ROOT_READOUT_KO.html'),largest_row_share=out['largest_row_share']),ensure_ascii=False))
