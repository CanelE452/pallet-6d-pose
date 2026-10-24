"""Inert saved-only V155 summary. Requires the root-pinned COMPLETE seal.

No trainer/head import, Torch, feature/checkpoint, image or private file reads.
"""
from pathlib import Path
import argparse, hashlib, json, datetime
import numpy as np

V = Path(__file__).resolve().parents[1]
P = V.parent
OUT = Path(__file__).resolve().parent
RUN = V/'runs/C37_C38_INDEPENDENT_AUX_WARM_FRESH_FIXED8000'
PROTOCOL = V/'source_fit_preparation/PROTOCOL_BEFORE_SOURCE_FEATURE_OR_CHECKPOINT_BODIES.json'
PROTOCOL_SHA = 'a042ca1c7daa02959f08ccc45ee65920870d40090518b8ec61c685b7e50910d8'
PREV = P/'query_visibility_mr6d_source_supervision_bridge_20261024_v154'
ARMS = {'C37':'C37_WARM_VISIBILITY_INDEPENDENT_AUX_FIXED8000',
        'C38':'C38_FRESH_VISIBILITY_INDEPENDENT_AUX_FIXED8000'}
SCORE_KEYS = ('raw_three_logits','primary_visibility_prediction',
    'secondary_entity_gated_visibility','global_index','frame_index','native_slot',
    'query_id','frame_id','source_id','split','query_xy','L','query_available')

def bind(p):
    return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def read(p):
    return json.loads(p.read_text())

def write(p,x):
    with p.open('x') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def checked(b):
    p=Path(b['path']);assert bind(p)==b;return p

def packet(p):
    with np.load(p,allow_pickle=False) as x:
        return {k:x[k] for k in SCORE_KEYS}

def metric(y,pred,mask):
    cm=[[int(np.sum(mask&(y==t)&(pred==v))) for v in (0,1,-1)] for t in (0,1)]
    def div(n,d):return float(n/d) if d else None
    a=np.array(cm);n=int(a.sum())
    return {'CM_rows0_1_columns0_1_U':cm,'N':n,
      'negative_recall':div(a[0,0],a[0].sum()),'positive_recall':div(a[1,1],a[1].sum()),
      'negative_precision':div(a[0,0],a[:,0].sum()),'positive_precision':div(a[1,1],a[:,1].sum()),
      'U_rate':div(a[:,2].sum(),n),'all_mask_rows':int(mask.sum()),
      'unknown_truth':int(np.sum(mask&~np.isin(y,[0,1])))}

def v154_metric(m):
    cm=m['CM_rows0_1_columns0_1_U']
    return {'CM_rows_D_E_columns_D_E_U':cm,'known_DE':m['N'],
      'D_recall':m['negative_recall'],'E_recall':m['positive_recall'],
      'D_precision':m['negative_precision'],'E_precision':m['positive_precision'],
      'input_or_output_U_known':sum(row[2] for row in cm),'U_rate_known':m['U_rate'],
      'original_rows':m['all_mask_rows'],'truth_outside_DE':m['unknown_truth']}

def paired(y,a,b,mask):
    return {name:{'N':int(np.sum(mask&(y==t))),
      'previous_D_E_U_rows_next_D_E_U_columns':[[int(np.sum(mask&(y==t)&(a==u)&(b==v))) for v in (0,1,-1)] for u in (0,1,-1)],
      'recovered':int(np.sum(mask&(y==t)&(a!=t)&(b==t))),
      'damaged':int(np.sum(mask&(y==t)&(a==t)&(b!=t)))} for name,t in (('D',0),('E',1))}

def ranking(y,z,mask):
    d=z[mask&(y==0)];e=z[mask&(y==1)];assert np.isfinite(d).all() and np.isfinite(e).all()
    auc=None if not len(d) or not len(e) else float(((e[:,None]>d).sum()+.5*(e[:,None]==d).sum())/(len(d)*len(e)))
    # Enumerate tied boundaries, exporting count pairs only, never numeric cuts.
    zy=np.r_[d,e];yy=np.r_[np.zeros(len(d),np.int8),np.ones(len(e),np.int8)]
    points=[{'D_correct':len(d),'E_correct':0}]
    dc,ec=len(d),0
    for val in np.unique(zy)[::-1]:
        ix=zy==val;dc-=int(np.sum(ix&(yy==0)));ec+=int(np.sum(ix&(yy==1)))
        points.append({'D_correct':dc,'E_correct':ec})
    capacity={}
    for floor in (.90,.95):
        minimum=int(np.ceil(floor*len(d)));best=max((x['E_correct'] for x in points if x['D_correct']>=minimum),default=0)
        capacity[str(floor)]={'minimum_D_correct':minimum,'D_N':len(d),
          'maximum_achievable_E_correct':best,'E_N':len(e),
          'maximum_achievable_E_recall':best/len(e) if len(e) else None}
    return {'ROC_AUC_E_positive_tie_half':auc,'D':len(d),'E':len(e),
      'E_capacity_at_D_recall_floor':capacity,'count_points_no_thresholds':points,
      'raw0_original_point_unchanged':{'D_correct':int(np.sum(d<0)),'D_N':len(d),
        'E_correct':int(np.sum(e>=0)),'E_N':len(e)},
      'retrospective_diagnostic_only':True,'threshold_or_model_selected':False}

def frame_metrics(rows,ey,pred,mask,av,frames,raw):
    truth=[];answer=[];with_input_truth=[];with_input_answer=[];support={}
    for f in np.unique(frames[mask]):
        m=mask&(frames==f);known=np.unique(ey[m&(ey>=0)])
        assert len(known)<=1,'Opposite known entity labels within one frame'
        if not len(known):
            support[str(int(f))]={'truth':-1,'known_query_rows':0};continue
        y=int(known[0]);idx=np.flatnonzero(m&av&(ey>=0))
        decision=-1
        if len(idx):
            assert np.all(raw[idx,1]==raw[idx[0],1]),'Tglobal output not frame-shared'
            assert np.all(pred[idx]==pred[idx[0]])
            decision=int(pred[idx[0]]);with_input_truth.append(y);with_input_answer.append(decision)
        truth.append(y);answer.append(decision)
        support[str(int(f))]={'truth':y,'known_query_rows':int(np.sum(m&(ey>=0))),
          'available_known_query_rows':len(idx),'decision':decision}
    ya=np.array(truth,np.int8);pa=np.array(answer,np.int8)
    yv=np.array(with_input_truth,np.int8);pv=np.array(with_input_answer,np.int8)
    return {'all_frames_with_known_truth':metric(ya,pa,np.ones(len(ya),bool)),
      'frames_with_available_known_query':metric(yv,pv,np.ones(len(yv),bool)),
      'all_original_frames':len(np.unique(frames[mask])),
      'unknown_truth_frames':sum(x['truth']<0 for x in support.values()),'support':support}

def main(control_sha):
    # No label, result or score body opens occur before this terminal-seal guard.
    control_path=RUN/'CONTROL_COMPLETE.json';cb=bind(control_path);assert cb['sha256']==control_sha
    control=read(control_path)
    assert control['completed'] and control['PASS'] and control['updates_each']==8000
    assert control['source_only'] and control['selected_model'] is None
    assert control['equal_auxiliary_initial_and_final_states_and_outputs_and_common_draws']
    assert control['all_original_native26624_public512_MR1840_retained']
    assert bind(PROTOCOL)['sha256']==PROTOCOL_SHA and control['protocol']==bind(PROTOCOL)
    protocol=read(PROTOCOL);labelseal=read(checked(protocol['source_supervision_seal']))
    assert labelseal['labels']==protocol['source_labels']
    lp=checked(protocol['source_labels']);rows=[json.loads(s) for s in lp.read_text().splitlines()]
    assert len(rows)==1840
    y=np.array([r['role_target'] for r in rows],np.int8);ey=np.array([r['entity_target'] for r in rows],np.int8)
    states=np.array([r['role_state'] for r in rows]);results={};packets={};inputs=[cb,bind(PROTOCOL),bind(lp)]
    for model,dirname in ARMS.items():
        fitpath=checked(control['arms'][dirname]['fit']);fit=read(fitpath)
        assert fit['completed'] and fit['optimizer_updates']==8000
        rp=checked(fit['source_results']);results[model]=read(rp);inputs.extend([bind(fitpath),bind(rp)])
        assert results[model]['update']==8000 and results[model]['selected_model'] is None
        for prefix in ('INITIAL','FINAL'):
            q=RUN/dirname/f'{prefix}_MR1840_OUTPUTS.npz';packets[model,prefix]=packet(q);inputs.append(bind(q))
    base=packets['C37','INITIAL'];av=base['query_available'];split=base['split'];frames=base['frame_index']
    for a in packets.values():
        for k in SCORE_KEYS[3:]:
            assert a[k].shape==base[k].shape and a[k].dtype==base[k].dtype
            assert a[k].tobytes()==base[k].tobytes(),k
        assert np.isfinite(a['raw_three_logits'][av]).all()
        pred=np.where(av,(a['raw_three_logits'][:,0]>=0).astype(np.int8),-1)
        assert np.array_equal(pred,a['primary_visibility_prediction'])
        assert np.array_equal(np.where(av&(a['raw_three_logits'][:,1]>=0),pred,-1),a['secondary_entity_gated_visibility'])
    assert packets['C37','FINAL']['raw_three_logits'][:,1:].tobytes()==packets['C38','FINAL']['raw_three_logits'][:,1:].tobytes()
    for i,r in enumerate(rows):
        assert r['global_index']==i and r['frame_index']==i//8 and r['native_slot']==i%8
        assert r['query_id']==base['query_id'][i] and r['split']==split[i] and bool(av[i])==r['original_input_valid']
        q=np.array([np.nan,np.nan] if r['query_xy'] is None else r['query_xy'],dtype=np.float64)
        assert np.array_equal(q,base['query_xy'][i],equal_nan=True)
        assert r['L']==base['L'][i] or r['L'] is None and np.isnan(base['L'][i])
        assert r['point_quality_label']==-1 and r['human_gold'] is False
    folds={'ALL':np.ones(1840,bool),'SOURCE_DEVELOPMENT':split=='SOURCE_DEVELOPMENT','SOURCE_VALIDATION':split=='SOURCE_VALIDATION'}
    mr={};pairs={};ranks={}
    for group,mask in folds.items():
        mr[group]={'inventory':{'original_rows':int(mask.sum()),'input_valid':int(np.sum(mask&av)),
          'semantic_states':{s:int(np.sum(mask&(states==s))) for s in ('D','E','S','OOF','U')},
          'outside_DE':int(np.sum(mask&~np.isin(y,[0,1])))},'models':{}}
        for model in ARMS:
            a=packets[model,'FINAL'];raw=a['raw_three_logits'];pred=a['primary_visibility_prediction'];gated=a['secondary_entity_gated_visibility']
            ep=np.where(av,(raw[:,1]>=0).astype(np.int8),-1)
            m={'visibility':metric(y,pred,mask),'visibility_inputvalid':metric(y,pred,mask&av),
              'entity':metric(ey,ep,mask),'entity_inputvalid':metric(ey,ep,mask&av),
              'secondary_entity_gated_visibility':metric(y,gated,mask),
              'entity_frames':frame_metrics(rows,ey,ep,mask,av,frames,raw),
              'additional_entity_rejection_U':int(np.sum(mask&av&(ep==0))),
              'trueE':{'all':int(np.sum(mask&(y==1))),'available':int(np.sum(mask&av&(y==1))),
                'entity_kept':int(np.sum(mask&av&(y==1)&(ep==1))),
                'entity_rejected':int(np.sum(mask&av&(y==1)&(ep==0))),
                'primary_correct':int(np.sum(mask&(y==1)&(pred==1))),
                'gate_kept_correct':int(np.sum(mask&(y==1)&(gated==1)))},
              'entity_false_rejection':{'n':int(np.sum(mask&av&(ey==1)&(ep==0))),
                'N':int(np.sum(mask&av&(ey==1)))}}
            for k in ('visibility','visibility_inputvalid','entity','entity_inputvalid','secondary_entity_gated_visibility'):
                assert m[k]==results[model]['MR'][group][k],(model,group,k)
            for new,previous in (('primary_all','visibility'),('primary_input_valid','visibility_inputvalid'),
              ('secondary_entity_gated_all','secondary_entity_gated_visibility'),('entity_all','entity'),('entity_input_valid','entity_inputvalid')):
                m[new]=v154_metric(m[previous])
            m['entity_false_reject_available_correct_entity']={'numerator':m['entity_false_rejection']['n'],'denominator':m['entity_false_rejection']['N']}
            m['additional_gate_U_all_original_rows']=m['additional_entity_rejection_U']
            mr[group]['models'][model]=m
        pairs[group]=paired(y,packets['C37','FINAL']['primary_visibility_prediction'],packets['C38','FINAL']['primary_visibility_prediction'],mask)
    held=folds['SOURCE_VALIDATION'];assert int(held.sum())==360 and np.sum(held&av&(y==0))==93 and np.sum(held&av&(y==1))==52
    for model in ARMS:ranks[model]=ranking(y,packets[model,'FINAL']['raw_three_logits'][:,0],held&av&np.isin(y,[0,1]))
    quality={};gates={};pallox={}
    for model in ARMS:
        selected={k:v['VALIDATION'] for k,v in results[model]['synthetic_point_quality'].items()}
        cm=sum((np.array(v['CM_rows0_1_columns0_1_U'],int) for v in selected.values()),np.zeros((2,3),int));assert cm.sum()==1222
        far_pred=int(cm[:,0].sum())
        quality[model]={'per_corpus':selected,'combined_rows_far_near_columns_far_near_U':cm.tolist(),
          'far_precision':float(cm[0,0]/far_pred) if far_pred else None,
          'far_correct':int(cm[0,0]),'far_N':int(cm[0].sum()),
          'near_correct':int(cm[1,1]),'near_N':int(cm[1].sum()),
          'original_held_rows':sum(v['all_mask_rows'] for v in selected.values()),
          'unknown_quality':sum(v['unknown_truth'] for v in selected.values())}
        gates[model]={'passed':results[model]['native']['native35_passed'],'N':35,'total':35,
          'all_pass':results[model]['native']['native35_all_pass'],'original_AUX29_not_retested':True}
        pallox[model]=results[model]['Pallox']['groups']['VAL_HALL']
    assert quality['C37']==quality['C38'],'Independent auxiliary outputs must agree'
    oldpath=PREV/'source_result_summary_agent/SOURCE_RESULTS_SUMMARY_KO.json';old=read(oldpath);inputs.append(bind(oldpath))
    for group,mask in folds.items():
        assert mr[group]['inventory']==old['MR'][group]['inventory']
        assert v154_metric(metric(y,base['primary_visibility_prediction'],mask))==old['MR'][group]['models']['C32_initial']['primary_all']
        mr[group]['models']={**old['MR'][group]['models'],**mr[group]['models']}
    oldrankpath=PREV/'independent_saved_fit_review/RANKING_DIAGNOSTIC.json';oldrank=read(oldrankpath);inputs.append(bind(oldrankpath))
    context={'older_models':old['MR']['SOURCE_VALIDATION'],'older_ranking':{k:{z:v[z] for z in ('ROC_AUC_E_positive_tie_half','raw0_original_point_unchanged','E_recall_capacity_at_D_recall_floor')} for k,v in oldrank['results'].items()},
      'earlier_C35_C36':{'updates_each':2000,'seed':1540,'shared_hidden_aux':True},
      'new_C37_C38':{'updates_each':8000,'seed':1550,'independent_aux':True},
      'causal_scope':'C37/C38 matching isolates visibility initialization. Earlier shared-hidden2000 fits are context, not branch-only architecture ablation; warm has prior learning history.'}
    summary={'completed':True,'PASS':True,'source_only':True,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'inputs':inputs,'MR':mr,'warm_to_fresh_paired_primary':pairs,'ranking_diagnostic':ranks,
      'MR_paired_primary':{group:{'C37_to_C38':value} for group,value in pairs.items()},
      'synthetic_quality':quality,'native35':gates,'Pallox_HALL':pallox,'earlier_context_unequal_budget':context,
      'independent_aux_final_scores_identical':True,'source_labels_unchanged':bind(lp)==protocol['source_labels'],
      'selected_model':None,'current_allowed':False,'fully_solved':False,
      'limits':['Fixed source assistant weak labels, not human gold or exact real pixel quality.',
        'All360 held roles include D102/E52/S9/OOF77/U120; knownD9 original inputU retained.',
        'Entity frame and query denominators both reported; eight held negative queries are one frame.',
        'Synthetic quality has1222 known held cases; precision and4922unknowns remain explicit.',
        'AUC/attainable counts are retrospective source diagnostics; no numerical threshold exported or selected.',
        'Current223 was not measured; no model adoption or full-goal success follows from source gains.'],
      'access':{'saved_source_scores_root_labels_results_metadata_only':True,'weights_features_RGB_or_private_reads':0,'learned_model_encoder_fit_PnP_calls':0}}
    write(OUT/'SOURCE_RESULTS_SUMMARY_KO.json',summary)
    render(summary)
    write(OUT/'SUMMARY_NUMERIC_RENDER_QA.json',{'completed':True,'PASS':True,
      'code':bind(Path(__file__)),'control':cb,'summary':bind(OUT/'SOURCE_RESULTS_SUMMARY_KO.json'),
      'MR_identity_qL_availability_join_and_saved_metric_equality':True,'primary_raw0_and_secondary_decoders_exact':True,
      'all1840_and360held_preserved':True,'all1222_quality_known_precision_and4922unknowns_preserved':True,
      'aux_final_raw_outputs_byteexact_across_arms':True,'original_labels_unchanged':True,
      'figures':[bind(OUT/n) for n in ('WARM_FRESH_SOURCE_COMPARISON_KO.png','AUXILIARY_SOURCE_SCOPE_KO.png')],
      'actual_image_view_QA':'PENDING root/agent actual view after rendering','access':summary['access']})
    print(json.dumps({'completed':True,'summary':bind(OUT/'SOURCE_RESULTS_SUMMARY_KO.json'),
      'held':{k:mr['SOURCE_VALIDATION']['models'][k]['visibility'] for k in ARMS},'native35':gates},ensure_ascii=False))

def render(s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font='/usr/share/fonts/truetype/nanum/NanumGothic.ttf';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':11,'figure.facecolor':'white'})
    old=s['earlier_context_unequal_budget']['older_models']['models']
    keys=['C32_initial','C35','C36','C37','C38'];names=['C32 초기','C35\n2000회','C36\n2000회','C37 warm\n8000회','C38 fresh\n8000회']
    cms=[old[k]['primary_input_valid']['CM_rows_D_E_columns_D_E_U'] for k in keys[:3]]+[s['MR']['SOURCE_VALIDATION']['models'][k]['visibility_inputvalid']['CM_rows0_1_columns0_1_U'] for k in keys[3:]]
    fig=plt.figure(figsize=(16,11));gs=fig.add_gridspec(2,2,height_ratios=[1.15,1],hspace=.52,wspace=.30)
    ax=fig.add_subplot(gs[0,:]);xx=np.arange(5)
    for r,(label,color) in enumerate(zip(('D 보임','E 외부가림'),('#3875ad','#c87842'))):
        counts=[(np.array(cm)[r,r],sum(cm[r])) for cm in cms];bars=ax.bar(xx+(r-.5)*.30,[n/d*100 for n,d in counts],.28,label=label,color=color)
        for bar,(n,d) in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{n}/{d}\n{n/d:.1%}',ha='center',fontsize=10)
    ax.axvline(2.5,ls='--',color='#777');ax.set(xticks=xx,xticklabels=names,ylim=(0,119),ylabel='정답률 (%)',title='held 입력가능 D93/E52 · 이전 결과는 학습량·seed가 다른 참고값');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    handles, labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper right',bbox_to_anchor=(.97,.945),ncol=2,frameon=False)
    for j,k in enumerate(('C37','C38')):
        ax=fig.add_subplot(gs[1,j]);cm=np.array(s['MR']['SOURCE_VALIDATION']['models'][k]['visibility']['CM_rows0_1_columns0_1_U']);ax.imshow(cm/cm.sum(axis=1,keepdims=True),cmap='Blues',vmin=0,vmax=1,aspect='auto')
        for r in range(2):
            for c in range(3):ax.text(c,r,str(cm[r,c]),ha='center',va='center',fontsize=22,color='white' if cm[r,c]/cm[r].sum()>.65 else '#253449')
        auc=s['ranking_diagnostic'][k]['ROC_AUC_E_positive_tie_half'];ax.set(xticks=[0,1,2],xticklabels=['예측 D','예측 E','원입력 U'],yticks=[0,1],yticklabels=['실제 D102','실제 E52'],title=f'{k} 전체 알려진154개 · AUC {auc:.3f}')
    fig.suptitle('같은 8000회 학습에서 visibility 초기화만 비교',fontsize=21,y=.98)
    fig.text(.06,.035,'held 전체360역할 유지: D102 / E52 / S9 / 화면밖77 / 불확실120 · D9개 원입력 U\nC35/C36: seed1540·2000회 / C37/C38: seed1550·8000회 → 과거 모델과 경로 분리만의 인과 비교 불가\n고정 source 약한 라벨 · 문턱0 · 현재223 신규 평가/모델 선택 없음',fontsize=11)
    fig.subplots_adjust(top=.88,bottom=.18,left=.08,right=.98);fig.savefig(OUT/'WARM_FRESH_SOURCE_COMPARISON_KO.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(16,11));fig.subplots_adjust(top=.88,bottom=.17,hspace=.65,wspace=.38)
    ax=axes[0,0];g=s['native35'];bars=ax.bar(['C37','C38'],[g[k]['passed'] for k in ARMS],color=['#3875ad','#c87842'],width=.5)
    for bar,k in zip(bars,ARMS):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.5,f'{g[k]["passed"]}/35',ha='center',fontsize=17)
    ax.axhline(35,color='#777',ls='--');ax.set(ylim=(0,39),title='기존 native35: 목재·정사각형·native8 범위 유지',ylabel='통과 조건 수')
    ax=axes[0,1];q=s['synthetic_quality']['C37'];counts=[(q['near_correct'],q['near_N']),(q['far_correct'],q['far_N'])];bars=ax.bar(['near ≤3px','far >20px'],[n/d*100 for n,d in counts],color=['#3875ad','#c59026'],width=.5)
    for bar,(n,d) in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{n}/{d}',ha='center',fontsize=14)
    fp=q['far_precision'];ax.set(ylim=(0,118),ylabel='정답률 (%)',title=f'양 arm 동일 합성 품질 · far precision {fp:.1%}' if fp is not None else '합성 품질 · far 예측 없음')
    ax.text(0,-.22,f'known1222 / 품질미정{q["unknown_quality"]} / 전체{q["original_held_rows"]}\n실사 좌표정확도 정답이 아니며 visibility 필터로 사용하지 않음',transform=ax.transAxes,fontsize=10)
    m=s['MR']['SOURCE_VALIDATION']['models']['C37'];ax=axes[1,0];ecm=np.array(m['entity_inputvalid']['CM_rows0_1_columns0_1_U']);ax.imshow(ecm[:,:2]/ecm.sum(axis=1,keepdims=True),cmap='Oranges',vmin=0,vmax=1,aspect='auto')
    for r in range(2):
        for c in range(2):ax.text(c,r,str(ecm[r,c]),ha='center',va='center',fontsize=22,color='white' if ecm[r,c]/ecm[r].sum()>.65 else '#352718')
    fm=m['entity_frames']['frames_with_available_known_query']['CM_rows0_1_columns0_1_U'];ax.set(xticks=[0,1],xticklabels=['대상 거부','대응 인정'],yticks=[0,1],yticklabels=['실제 다른대상8','실제 대응244'],title='양 arm 동일 entity · 원입력 가능252 query')
    ax.text(0,-.22,f'독립 frame 분모: 다른대상 {sum(fm[0])} / 대응 {sum(fm[1])}\nquery8개가 서로 다른 개체8개를 뜻하지 않음',transform=ax.transAxes,fontsize=10)
    ax=axes[1,1];xx=np.arange(2)
    for t,(label,col) in enumerate(zip(('D 보임','E 외부가림'),('#3875ad','#c87842'))):
        counts=[(np.array(s['Pallox_HALL'][k]['CM_D_E_rows_D_E_U_columns'])[t,t],sum(s['Pallox_HALL'][k]['CM_D_E_rows_D_E_U_columns'][t][:2])) for k in ARMS]
        bars=ax.bar(xx+(t-.5)*.30,[n/d*100 for n,d in counts],.28,label=label,color=col)
        for bar,(n,d) in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f'{n}/{d}',ha='center',fontsize=14)
    ax.set(xticks=xx,xticklabels=list(ARMS),ylim=(0,115),ylabel='정답률 (%)',title='Pallox HALL: 입력가능 D33/E6');ax.legend(loc='lower left')
    fig.suptitle('별도 보조 출력과 원래 source 기준을 함께 확인',fontsize=21,y=.98)
    fig.text(.06,.035,'보조 출력은 arm 간 동일한 입력·초기값·draw·optimizer로 학습되어 최종 출력까지 동일함.\nprimary visibility는 entity·품질로 보류시키지 않음. 보조 gate의 trueE 손실·추가 U는 JSON에 별도 보존.\n약한 source 라벨 / 원입력·원순서 유지 / 사람 gold·실사3px·현재223 성능 주장 없음',fontsize=11)
    fig.savefig(OUT/'AUXILIARY_SOURCE_SCOPE_KO.png',dpi=150);plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--control-sha',required=True)
    args=parser.parse_args();main(args.control_sha)
