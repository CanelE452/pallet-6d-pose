"""Saved scores / root weak labels only. No model, feature, checkpoint or pixels."""
from pathlib import Path
import json, hashlib, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/C35_C36_SEPARATE_QUERY_FIXED2000'
ARMS = {'C35': 'C35_MATCHED_VISIBILITY_ONLY_CONTROL',
        'C36': 'C36_MATCHED_VISIBILITY_PLUS_ENTITY_AND_POINT_QUALITY'}
FONT = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family': font_manager.FontProperties(fname=FONT).get_name(),
                     'axes.unicode_minus': False, 'font.size': 11,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})

def binding(p):
    return {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}

def write(name, value):
    with (OUT/name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def load_json(p):
    return json.loads(p.read_text())

def load_packet(p):
    with np.load(p, allow_pickle=False) as x:
        # Explicit score / identity whitelist; no encoder feature or model loading.
        return {k:x[k] for k in ('raw_three_logits', 'primary_visibility_prediction',
          'secondary_entity_gated_visibility', 'global_index', 'frame_index',
          'native_slot', 'query_id', 'frame_id', 'source_id', 'split',
          'query_xy', 'L', 'query_available')}

def metric(y, pred, mask):
    cm=np.array([[np.sum(mask&(y==t)&(pred==s)) for s in (0,1,-1)] for t in (0,1)], dtype=int)
    def ratio(n,d): return float(n/d) if d else None
    return {'CM_rows_D_E_columns_D_E_U':cm.tolist(), 'known_DE':int(cm.sum()),
      'D_recall':ratio(cm[0,0],cm[0].sum()),'E_recall':ratio(cm[1,1],cm[1].sum()),
      'D_precision':ratio(cm[0,0],cm[:,0].sum()),'E_precision':ratio(cm[1,1],cm[:,1].sum()),
      'input_or_output_U_known':int(cm[:,2].sum()),'U_rate_known':ratio(cm[:,2].sum(),cm.sum()),
      'original_rows':int(mask.sum()),'truth_outside_DE':int(np.sum(mask&(y<0)))}

def transitions(y, a, b, mask):
    return {name:{'CM_previous_D_E_U_rows_next_D_E_U_columns':
      [[int(np.sum(mask&(y==t)&(a==u)&(b==v))) for v in (0,1,-1)] for u in (0,1,-1)],
      'recovered':int(np.sum(mask&(y==t)&(a!=t)&(b==t))),
      'damaged':int(np.sum(mask&(y==t)&(a==t)&(b!=t)))} for name,t in (('D',0),('E',1))}

control=load_json(RUN/'CONTROL_COMPLETE.json')
assert control['completed'] and control['PASS'] and control['updates_each']==2000
assert control['selected_model'] is None and control['source_only']
seal=load_json(ROOT/'ROOT_SUPERVISION_SEAL.json')
label_path=Path(seal['labels']['path'])
assert binding(label_path)['sha256']==seal['labels']['sha256']
rows=[json.loads(s) for s in label_path.read_text().splitlines()]
assert len(rows)==1840
y=np.array([r['role_target'] for r in rows],dtype=np.int8)
ey=np.array([r['entity_target'] for r in rows],dtype=np.int8)
states=np.array([r['role_state'] for r in rows])
packets={}; results={}; inputs=[binding(RUN/'CONTROL_COMPLETE.json'),binding(ROOT/'ROOT_SUPERVISION_SEAL.json'),binding(label_path)]
for arm,dirname in ARMS.items():
    d=RUN/dirname; results[arm]=load_json(d/'SOURCE_RESULTS.json')
    inputs.append(binding(d/'SOURCE_RESULTS.json'))
    for prefix in ('INITIAL','FINAL'):
        p=d/f'{prefix}_MR1840_OUTPUTS.npz'
        packets[(arm,prefix)]=load_packet(p);inputs.append(binding(p))
base=packets['C35','INITIAL'];available=base['query_available'];split=base['split']
for packet in packets.values():
    for k in ('global_index','frame_index','native_slot','query_id','frame_id','source_id','split','query_xy','L','query_available'):
        assert np.array_equal(packet[k],base[k],equal_nan=True) if packet[k].dtype.kind in 'fc' else np.array_equal(packet[k],base[k])
    expected=np.where(available,(packet['raw_three_logits'][:,0]>=0).astype(np.int8),-1)
    assert np.array_equal(expected,packet['primary_visibility_prediction'])
    expected_gated=np.where(available&(packet['raw_three_logits'][:,1]<0),-1,expected)
    assert np.array_equal(expected_gated,packet['secondary_entity_gated_visibility'])
for i,row in enumerate(rows):
    assert row['global_index']==i and row['native_slot']==i%8 and row['frame_index']==i//8
    assert row['query_id']==base['query_id'][i] and row['split']==split[i]
    assert row['original_input_valid']==bool(available[i])
    jq=np.array([np.nan,np.nan] if row['query_xy'] is None else row['query_xy'],dtype=np.float64)
    assert np.array_equal(jq,base['query_xy'][i],equal_nan=True)
    assert (row['L']==base['L'][i]) or (row['L'] is None and np.isnan(base['L'][i]))
    assert row['point_quality_label']==-1 and row['human_gold'] is False
assert packets['C35','INITIAL']['raw_three_logits'].tobytes()==packets['C36','INITIAL']['raw_three_logits'].tobytes()
models={'C32_initial':base,'C35':packets['C35','FINAL'],'C36':packets['C36','FINAL']}
groups={'ALL':np.ones(1840,bool),'SOURCE_DEVELOPMENT':split=='SOURCE_DEVELOPMENT','SOURCE_VALIDATION':split=='SOURCE_VALIDATION'}
mr={};paired={}
for group,mask in groups.items():
    mr[group]={'inventory':{'original_rows':int(mask.sum()),'input_valid':int(np.sum(mask&available)),
      'semantic_states':{s:int(np.sum(mask&(states==s))) for s in ('D','E','S','OOF','U')},
      'outside_DE':int(np.sum(mask&(y<0)))},'models':{}}
    for name,packet in models.items():
        pred=packet['primary_visibility_prediction']
        ep=np.where(available,(packet['raw_three_logits'][:,1]>=0).astype(np.int8),-1)
        model={'primary_all':metric(y,pred,mask),'primary_input_valid':metric(y,pred,mask&available),
          'secondary_entity_gated_all':metric(y,packet['secondary_entity_gated_visibility'],mask),
          'entity_all':metric(ey,ep,mask),'entity_input_valid':metric(ey,ep,mask&available),
          'entity_false_reject_available_correct_entity':{
            'numerator':int(np.sum(mask&available&(ey==1)&(ep==0))),
            'denominator':int(np.sum(mask&available&(ey==1)))},
          'additional_gate_U_all_original_rows':int(np.sum(mask&available&(ep==0)))}
        mr[group]['models'][name]=model
        if name!='C32_initial':
            ref=results[name]['MR'][group]
            assert model['primary_all']['CM_rows_D_E_columns_D_E_U']==ref['visibility']['CM_rows0_1_columns0_1_U']
            assert model['primary_input_valid']['CM_rows_D_E_columns_D_E_U']==ref['visibility_inputvalid']['CM_rows0_1_columns0_1_U']
            assert model['secondary_entity_gated_all']['CM_rows_D_E_columns_D_E_U']==ref['secondary_entity_gated_visibility']['CM_rows0_1_columns0_1_U']
            assert model['entity_all']['CM_rows_D_E_columns_D_E_U']==ref['entity']['CM_rows0_1_columns0_1_U']
    paired[group]={}
    for a,b in (('C32_initial','C35'),('C32_initial','C36'),('C35','C36')):
        paired[group][f'{a}_to_{b}']=transitions(y,models[a]['primary_visibility_prediction'],models[b]['primary_visibility_prediction'],mask)

quality={}
for name in ('C35','C36'):
    selected={k:v['VALIDATION'] for k,v in results[name]['synthetic_point_quality'].items()}
    cm=sum((np.array(v['CM_rows0_1_columns0_1_U'],dtype=int) for v in selected.values()),np.zeros((2,3),int))
    quality[name]={'per_corpus_held_metrics_from_sealed_SOURCE_RESULTS':selected,
      'combined_held_CM_rows_far_near_columns_far_near_U':cm.tolist(),
      'far_correct':int(cm[0,0]),'far_total':int(cm[0].sum()),
      'near_correct':int(cm[1,1]),'near_total':int(cm[1].sum()),
      'original_rows':sum(v['all_mask_rows'] for v in selected.values()),
      'unknown_quality_rows':sum(v['unknown_truth'] for v in selected.values())}
    assert cm.sum()==1222
native={name:{'passed':results[name]['native']['native35_passed'],'total':35,
  'all_pass':results[name]['native']['native35_all_pass'],
  'original_AUX29_not_retested':True} for name in ('C35','C36')}
pallox={name:results[name]['Pallox']['groups']['VAL_HALL'] for name in ('C35','C36')}
summary={'completed':True,'PASS':True,'source_only':True,'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'inputs':inputs,'actual_run':{'updates_each':2000,'source_CONTROL_complete':binding(RUN/'CONTROL_COMPLETE.json'),
  'actual_fit_completed':True,'historical_V153_prototype_was_untrained_at_its_own_timestamp':True},
 'design':{'initial':'frozen C32 update8000 visibility route plus zero auxiliary routes',
  'C35':'common visibility losses + auxiliary coefficient0',
  'C36':'identical common visibility losses + entity/point-quality auxiliary coefficient1',
  'thresholds_fixed_logit0':True,'primary_visibility_never_gated_by_entity_quality_or_GTerror':True,
  'initial_to_final_not_a_pure_MR_data_ablation':'Both arms also continue fixed synthetic and Pallox visibility losses.'},
 'MR':mr,'MR_paired_primary':paired,'synthetic_point_quality_held':quality,
 'native35':native,'Pallox_HALL':pallox,
 'conclusions_KO':[
  '새 공개 MR 약한 라벨을 포함한 C35 학습 후 held 장면 외부가림 정답은 초기7/52→13/52, 보이는 점은86/93→89/93이다. 다른 공통 loss도 함께 계속하므로 MR 데이터 단독 인과효과라고 단정하지 않는다.',
  'C36 보조학습은 C35 대비 외부가림 정답13/52→31/52이지만 보이는 점89/93→64/93으로 손상된다. 양쪽을 함께 만족한 성공이 아니다.',
  'held 전체360행은 D102/E52/그외206을 모두 유지했다. D9개 원입력U를 평가에서 삭제하지 않으며 입력가능 보조표는 D93/E52이다.',
  'C36 entity는 held 입력가능 실제 대응244개 중57개를 잘못 거부한다. 원입력U84개를 이57개의 분모와 혼합하지 않는다.',
  '좌표품질 near/far 성과는 합성 held 알려진1222개(near1156/far66)에서만 측정했다. 실제 MR 좌표품질 gold는 없다.',
  'native35는 C35 22/35, C36 17/35이며 둘 다 기준 미달이다. Pallox HALL 보이는 점도32/33→29/33으로 감소했다.',
  '라벨은 고정된 source assistant weak 제안이다. 현재223 성능·사람 gold·모델 채택·목표 해결을 주장하지 않는다.'
 ],'selected_model':None,'current_allowed':False,'adopted':False,'fully_solved':False,
 'access':{'saved_source_scores_and_root_labels_results_only':True,'encoder_model_fit_calls':0,
  'weights_actualfeatures_RGB_private_bodies_read':0,'label_threshold_or_selection_changes':0}}
write('SOURCE_RESULTS_SUMMARY_KO.json',summary)

names=['C32 초기','C35 가시성','C36 보조학습'];keys=list(models)
colors=['#3875ad','#cc7043']
fig=plt.figure(figsize=(16,11));grid=fig.add_gridspec(2,6,height_ratios=[1,1.05],hspace=.58,wspace=.65)
for j,(group,title) in enumerate((('SOURCE_DEVELOPMENT','학습 장면: 알려진 역할 중 원입력 가능'),('SOURCE_VALIDATION','분리한 held 장면: 알려진 역할 중 원입력 가능'))):
    ax=fig.add_subplot(grid[0,j*3:(j+1)*3]); xx=np.arange(3)
    for t,(label,col) in enumerate(zip(('D 보임','E 외부가림'),colors)):
        vals=[];counts=[]
        for key in keys:
            cm=np.array(mr[group]['models'][key]['primary_input_valid']['CM_rows_D_E_columns_D_E_U'])
            counts.append((cm[t,t],cm[t].sum()));vals.append(cm[t,t]/cm[t].sum()*100)
        bars=ax.bar(xx+(t-.5)*.32,vals,.30,color=col,label=label)
        for bar,(n,d) in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{n}/{d}\n{n/d:.1%}',ha='center',fontsize=10)
    ax.set(xticks=xx,xticklabels=names,ylim=(0,120),ylabel='정답률 (%)',title=title);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='lower left')
for j,key in enumerate(keys):
    ax=fig.add_subplot(grid[1,j*2:(j+1)*2]);cm=np.array(mr['SOURCE_VALIDATION']['models'][key]['primary_all']['CM_rows_D_E_columns_D_E_U'])
    ax.imshow(cm/cm.sum(axis=1,keepdims=True),cmap='Blues',vmin=0,vmax=1,aspect='auto')
    for r in range(2):
        for c in range(3):ax.text(c,r,str(cm[r,c]),ha='center',va='center',fontsize=20,color='white' if cm[r,c]/cm[r].sum()>.6 else '#16263a')
    ax.set(xticks=[0,1,2],xticklabels=['예측 D','예측 E','원입력 U'],yticks=[0,1],yticklabels=['실제 D 102','실제 E 52'],title=f'{names[j]} · held 전체 혼동행렬')
fig.suptitle('공개 MR6D 저장 결과: 외부가림 개선과 보이는 점 손상이 함께 발생',fontsize=21,y=.985)
fig.text(.06,.035,'held 360개 역할 유지: D102 · E52 · S9 · 화면밖77 · 불확실120. D9개는 원입력 불가.\n고정 source 약한 라벨 / raw0 / 각 2000회 학습 · 현재 평가223 결과 아님 · 모델 채택 없음',fontsize=12)
fig.subplots_adjust(top=.89,bottom=.17,left=.07,right=.98)
fig.savefig(OUT/'MR_PRE_POST_COMPARISON_KO.png',dpi=150);plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(16,11));fig.subplots_adjust(top=.89,bottom=.15,hspace=.63,wspace=.35)
ax=axs[0,0];bars=ax.bar(['C35','C36'],[22,17],color=['#3875ad','#c86d4c'],width=.5)
for bar,n in zip(bars,[22,17]):ax.text(bar.get_x()+bar.get_width()/2,n+.7,f'{n}/35',ha='center',fontsize=17)
ax.axhline(35,color='#686868',linestyle='--');ax.set(ylim=(0,39),ylabel='통과 조건 수',title='기존 합성 native 35개 조건: 둘 다 미달')
ax=axs[0,1];xx=np.arange(2)
for t,(label,col) in enumerate(zip(('D 보임','E 외부가림'),colors)):
    vals=[];counts=[]
    for key in ('C35','C36'):
        cm=np.array(pallox[key]['CM_D_E_rows_D_E_U_columns']);n=cm[t,t];d=cm[t,:2].sum();vals.append(n/d*100);counts.append((n,d))
    bars=ax.bar(xx+(t-.5)*.30,vals,.28,label=label,color=col)
    for bar,(n,d) in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{n}/{d}',ha='center',fontsize=13)
ax.set(xticks=xx,xticklabels=['C35','C36'],ylim=(0,115),title='Pallox HALL: 알려진 원입력 가능 D33/E6',ylabel='정답률 (%)');ax.legend(loc='lower left')
ax=axs[1,0];xx=np.arange(2)
for t,(label,col) in enumerate(zip(('near ≤3px','far >20px'),('#3875ad','#c59026'))):
    vals=[];counts=[]
    for corpus in ('OLD6144','NEW20480'):
        cm=np.array(quality['C36']['per_corpus_held_metrics_from_sealed_SOURCE_RESULTS'][corpus]['CM_rows0_1_columns0_1_U']);r=1-t;n=cm[r,r];d=cm[r].sum();counts.append((n,d));vals.append(n/d*100)
    bars=ax.bar(xx+(t-.5)*.30,vals,.28,color=col,label=label)
    for bar,(n,d) in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{n}/{d}',ha='center',fontsize=12)
ax.set(xticks=xx,xticklabels=['기존 합성 held','연속시점 합성 held'],ylim=(0,118),ylabel='정답률 (%)',title='C36 좌표품질: 합성에서만 정답이 있음');ax.legend(loc='lower left')
ax.text(0,-.24,'near 1156 / far 66 / 품질미정 4922 · 총6144행 유지\n실제 MR 좌표정확도 성능으로 해석하지 않음',transform=ax.transAxes,fontsize=10)
ax=axs[1,1];em=mr['SOURCE_VALIDATION']['models']['C36']['entity_false_reject_available_correct_entity']; cm=np.array(mr['SOURCE_VALIDATION']['models']['C36']['entity_all']['CM_rows_D_E_columns_D_E_U'])
ax.imshow(cm/cm.sum(axis=1,keepdims=True),cmap='Oranges',vmin=0,vmax=1,aspect='auto')
for r in range(2):
    for c in range(3):ax.text(c,r,str(cm[r,c]),ha='center',va='center',fontsize=19,color='white' if cm[r,c]/cm[r].sum()>.65 else '#352518')
ax.set(xticks=[0,1,2],xticklabels=['거부','대응 인정','원입력 U'],yticks=[0,1],yticklabels=['실제 다른대상 8','실제 대응 328'],title=f'C36 대상 판별: 입력가능 실제대응 {em["numerator"]}/{em["denominator"]} 오거부')
fig.suptitle('보조 출력이 학습돼도 전체 기준은 충족하지 못함',fontsize=21,y=.98)
fig.text(.055,.035,'source 약한 라벨 기반 사후 진단 · threshold 고정 · 주 가시성 출력은 entity/quality로 걸러내지 않음\n보조 entity gate 적용 시 held D 정답64 유지, E 정답31→15 및 추가 보류 발생 · 현재223 측정/선택/채택 없음',fontsize=12)
fig.savefig(OUT/'AUXILIARY_AND_SOURCE_GATES_KO.png',dpi=150);plt.close(fig)
qa={'completed':True,'PASS':True,'numerical_checks':{
 'MR_root_rows_original_identity_q_L_availability_joins':True,
 'initial_C35_C36_score_bytes_equal':True,'raw0_primary_and_secondary_decoders_exact':True,
 'recomputed_MR_primary_secondary_entity_CM_match_saved_SOURCE_RESULTS':True,
 'all1840_MR_rows_and360held_rows_preserved':True,
 'quality_metric_source':'Saved SOURCE_RESULTS only; quality target bodies not loaded',
 'original_labels_unchanged':binding(label_path)['sha256']==seal['labels']['sha256']},
 'outputs':[binding(OUT/n) for n in ('SOURCE_RESULTS_SUMMARY_KO.json','MR_PRE_POST_COMPARISON_KO.png','AUXILIARY_AND_SOURCE_GATES_KO.png')],
 'render':{'matplotlib':matplotlib.__version__,'Korean_font':FONT,'actual_image_view_QA':'PENDING separate receipt'},
 'access':summary['access'],'code':binding(Path(__file__))}
write('SUMMARY_NUMERIC_AND_RENDER_QA.json',qa)
print(json.dumps({'completed':True,'summary':binding(OUT/'SOURCE_RESULTS_SUMMARY_KO.json'),
 'numeric_QA':binding(OUT/'SUMMARY_NUMERIC_AND_RENDER_QA.json'),
 'initial_MR_TRAIN':mr['SOURCE_DEVELOPMENT']['models']['C32_initial']['primary_all'],
 'initial_MR_HELD':mr['SOURCE_VALIDATION']['models']['C32_initial']['primary_all'],
 'C36_quality_held':quality['C36']['combined_held_CM_rows_far_near_columns_far_near_U']},ensure_ascii=False))
