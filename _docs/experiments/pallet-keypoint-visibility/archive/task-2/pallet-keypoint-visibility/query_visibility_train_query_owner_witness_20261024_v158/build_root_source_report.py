"""Source-only owner-witness diagnostic. Does not train or select a model."""
from pathlib import Path
import datetime, hashlib, html, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

V = Path(__file__).resolve().parent
J = V / 'joined_owner_witness_v1'
STATES = ('TARGET_VISIBLE_SUPPORTED','FOREIGN_VISIBLE_SUPPORTED','BACKGROUND_SUPPORTED','UNKNOWN')
LABELS = ('대상 팔레트 표면','다른 물체 표면','배경','미확정')
COLORS = ('#21875a','#d86b31','#447fba','#a0a7b2')
def read(p): return json.loads(Path(p).read_text())
def bind(p):
    p=Path(p).resolve()
    assert '_v112' not in str(p).lower()
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def write(p,obj):
    with Path(p).open('x') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def main():
    s=read(J/'OWNER_WITNESS_JOIN_COMPLETE.json')
    assert s['PASS'] and s['all390_retained'] and s['original_roles']=={'D':341,'E':49}
    assert sum(s['counts']['ALL'].values())==390
    next_action={
        'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_evidence':[bind(J/'OWNER_WITNESS_JOIN_COMPLETE.json')],
        'decision_basis':'Public SOURCE_DEVELOPMENT observations only; no private metric used to choose the next source action.',
        'action_KO':'정확한 native 점의 보임과 query 중심 표면 소유를 분리한 상태로, 공개 TRAIN에서 native 부위의 실제 보임 근거를 추가로 확인하는 feasibility audit.',
        'why_KO':'중심에 다른 물체·배경이 보이는 상태만으로 native 외부 가림이나 좌표 오류 정답을 만들 수 없다. 가상 모서리·외곽점은 중심에 배경이 있어도 정상일 수 있다.',
        'constraints':[
            'Keep all original source rows, splits, native order, q/L and D/E states.',
            'Do not automatically admit the 53 supported target centers or any other new weak observations to training.',
            'Supported center pixels are only coarse owner witnesses, not exact native identity or coordinate gold.',
            'A future controlled cover must have before-visible native support and after-covered native support independently justified; a pixel cover alone gets no primary E label.',
            'Record ambiguous native-part support as UNKNOWN; do not silently drop it from the original evaluation scope.',
            'Full original WOOD/SQUARE/native8 scope remains required; this wood-source audit does not solve square coverage.',
            'Any future actual fit needs its own source-only frozen protocol and public source checks before current evaluation.',
            'No current/private data, held45 relabeling, V112, PnP or q/L correction for this source preparation.'
        ],
        'new_fit_or_model_forward_authorized_by_this_file':False,
        'selected_model':None,'goal_status':'active_unsolved'
    }
    write(V/'ROOT_NEXT_SOURCE_ACTION_AFTER_OWNER_WITNESS.json',next_action)
    joined_rows=[json.loads(t) for t in (J/'ALL390_SEPARATE_QUERY_VISIBLE_OWNER_WITNESS.jsonl').read_text().splitlines()]
    cross_rows=[x for x in joined_rows if x['cross_review']]
    r={
        'schema':'V158_PUBLIC_TRAIN390_SEPARATE_VISIBLE_OWNER_WITNESS',
        'completed':True,'stage_complete':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'goal_status':'active_unsolved','fully_solved':False,'selected_model':None,
        'lead_KO':'공개 TRAIN 390개의 실제 입력 중심을 확인했다. 원래 native 보임 역할과 입력 중심의 표면 소유는 서로 다른 질문이며, 두 정보를 섞어 학습하면 목표가 어긋날 수 있다. 이 감사는 원인 근거를 보탰으며 성능 개선을 입증하지 않는다.',
        'source_scope':{'available_weak_role_rows':390,'source_frames':s['source_frames'],'original_D':341,'original_E':49,'all_rows_retained':True,'q_L_native_roles_splits_unchanged':True,'material':'MR6D public Euro wood pallet; not square coverage','original_D_E_provenance':'Prior root/assistant-adjudicated weak structural-native roles on public MR6D frames; not official human per-keypoint visibility annotations.','original_D_E_human_or_physical_native_gold':False},
        'literal_saved_center':'RGB96[47:49,47:49], all four valid samples required; whole frame and central32 used for weak context.',
        'owner_states':dict(zip(STATES,LABELS)),
        'final_counts':{k:s['counts'][k] for k in ('ALL','ROLE_D','ROLE_E')},
        'raw_primary_counts':s['raw_primary_counts'],
        'review_scope':{'primary_actual_visual_cases':390,'blinded_reviewers':2,'partition_cases_each':195,'fixed_cross_cases':40,'cross_two_reviewer_agree':s['cross40_two_reviewer_agree'],'root_independently_proposed_before_new_reviewer_labels':40,'cross_three_observations_agree':s['cross40_three_observations_agree'],'cross40_disagreement_policy_rows':s['cross40_conservative_UNKNOWN_rows'],'cross40_final_unknown':sum(x['final_visible_owner_support']=='UNKNOWN'for x in cross_rows),'cross40_triple_agree_unknown':sum(x['final_visible_owner_support']=='UNKNOWN'and x['cross_review']['three_observations_agree']for x in cross_rows),'cross40_new_unknown_from_supported_primary':sum(x['primary_owner']!='UNKNOWN'and x['final_visible_owner_support']=='UNKNOWN'for x in cross_rows),'other_rows_single_weak_review':350,'invalid_center_rows':s['any_center_invalid_rows'],'accuracy_or_human_gold_claimed':False},
        'interpretation_KO':[
            '원래 D341개 중 중심이 대상 표면으로 지지되는 경우50, 다른 물체15, 배경146, 미확정130이다. D341개의 정답 오류율이라는 뜻이 아니다.',
            '원래 E49개 중 중심에 다른 물체가 지지되는 경우37, 대상 표면3, 배경2, 미확정7이다. 대상 표면3개도 native E의 오표기라고 단정할 수 없다.',
            'query 중심이 외부 물체여도 실제 native 점은 다른 곳에서 보일 수 있다. 중심에 대상 목재가 보여도 정확한 native 점이거나 실제 native 점이 보인다는 보장은 없다.',
            '모서리·가상 외곽점은 정상이어도 중심 샘플이 바닥이나 벽일 수 있다. 배경을 외부 가림이나 좌표 오류로 바꾸면 새로운 라벨 오류가 된다.',
            '40개 두 검토 일치34/40, 세 관찰 일치30/40은 관찰 반복성이다. 정확도 검증이나 독립적인 물리 정답이 아니다.',
            '이번390개는 공개 TRAIN 중 기존 D/E와 입력이 있는 전체이며, 다른 source나 현재 평가셋의 표본을 대체하지 않는다. 별도 native8/정사각형 범위는 유지한다.'
        ],
        'progress_and_limits_KO':{
            'confirmed':'실제 crop을 보면서 역할 라벨과 literal query 중심 관찰 사이의 의미 차이를 정량적으로 기록했다.',
            'not_confirmed':'이 의미 차이가 이전 모델의 모든 실패를 일으킨다는 인과관계, 새 방법의 성능 향상, 정확한 좌표 판별.',
            'still_open':'native 부위 보임의 확실한 공개 실사 학습 근거, 새로운 가림 물체에 대한 전이, 정사각형/모든 native 순서의 공개 실사 범위.',
            'feasibility':'현재 실험만으로 성공 확률을 수치로 정할 수 없다. 목표가 불가능하다는 근거는 없지만 실용 성능은 아직 미입증이다.'
        },
        'original_visibility_roles_changed':False,'new_annotations_training_admitted':False,
        'new_human_physical_native_pixel_coordinate_gold':0,
        'new_training_model_encoder_R0_private_inputs_private_predictions_V112_PnP_qL_calls':0,
        'status_summary_access_note':'Root read the existing parent status and stored aggregate benchmark summary to answer the user status question; no private input, label, per-query output or model was newly opened in this source audit. Next source action is based on source evidence only.',
        'evidence':{'join':bind(J/'OWNER_WITNESS_JOIN_COMPLETE.json'),'rows390':bind(J/'ALL390_SEPARATE_QUERY_VISIBLE_OWNER_WITNESS.jsonl'),'cross40':bind(J/'ALL40_BLIND_CROSSREVIEW_AND_ROOT_PROPOSALS.jsonl'),'root_visual_QA':bind(V/'ROOT_ACTUAL_CROSS40_VISUAL_QA.json'),'next_source_action':bind(V/'ROOT_NEXT_SOURCE_ACTION_AFTER_OWNER_WITNESS.json')},
        'live_processes':[]
    }
    write(V/'ROOT_RESULTS_KO.json',r)
    font_manager.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    plt.rcParams.update({'font.family':'NanumGothic','axes.unicode_minus':False})
    fig,ax=plt.subplots(figsize=(14,7.2),dpi=130)
    fig.set_facecolor('#f6f8fb');ax.set_facecolor('#f6f8fb')
    roles=('ROLE_D','ROLE_E');left=[0.,0.]
    for st,lab,color in zip(STATES,LABELS,COLORS):
        nn=[s['counts'][k][st] for k in roles];den=[341,49];vv=[100*n/d for n,d in zip(nn,den)]
        ax.barh([1,0],vv,left=left,height=.47,label=lab,color=color)
        for y,l,w,n in zip([1,0],left,vv,nn):
            if w>=8:ax.text(l+w/2,y,f'{n}개\n{w:.1f}%',ha='center',va='center',color='white',fontsize=13,fontweight='bold')
        left=[a+b for a,b in zip(left,vv)]
    ax.set_yticks([1,0],['원래 보임 D\n341개','원래 외부 가림 E\n49개'],fontsize=14)
    ax.set_xlim(0,100);ax.set_ylim(-.6,1.65);ax.set_xlabel('각 원래 역할 라벨 안에서의 비율 (%)',fontsize=12)
    for sp in ax.spines.values():sp.set_visible(False)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.03),ncol=4,frameon=False,fontsize=11)
    fig.suptitle('모서리의 보임 역할 ≠ 초기 query 중심의 표면 소유',fontsize=22,fontweight='bold',x=.51,y=.96)
    fig.text(.08,.075,'공개 TRAIN 390개 전부 유지 · 원래 q/L/native/D/E 변경 없음\n중심 2×2 샘플의 약한 시각 관찰: 대상 53 / 다른 물체 52 / 배경 148 / 미확정 137\n배경은 가림·좌표 오류 정답이 아님. 사람/물리 native 정답 및 모델 성능 검증이 아님.',fontsize=12,color='#334155',linespacing=1.6)
    fig.subplots_adjust(left=.17,right=.96,top=.8,bottom=.25)
    fig.savefig(V/'SOURCE_ROLE_VS_QUERY_OWNER_KO.png');plt.close(fig)
    repair={
        'completed':True,'kind':'syntax_only_pre_execution_repair',
        'failure':'Initial join script failed parsing because original_roles keyword appeared twice; no code body executed.',
        'repair':'Renamed only the metadata binding key to original_roles_binding; counts remain original_roles.',
        'preserved_before':bind(V/'provenance/JOIN_BEFORE_DUPLICATE_KEY_SYNTAX_REPAIR.py'),
        'executed_code':bind(V/'join_train_owner_witness.py'),
        'successful_join':bind(J/'OWNER_WITNESS_JOIN_COMPLETE.json'),
        'semantic_label_or_selection_change':False
    }
    write(V/'ROOT_JOIN_SYNTAX_REPAIR_RECEIPT.json',repair)
    print(json.dumps({'completed':True,'results':bind(V/'ROOT_RESULTS_KO.json'),'plot':bind(V/'SOURCE_ROLE_VS_QUERY_OWNER_KO.png')},ensure_ascii=False))

if __name__=='__main__':main()
