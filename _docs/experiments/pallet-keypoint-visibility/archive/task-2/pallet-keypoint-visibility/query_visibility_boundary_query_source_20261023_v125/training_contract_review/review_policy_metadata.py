"""Frozen C17 policy + PUBLIC train metadata only; no features/scores/weights/model imports."""
from pathlib import Path
from collections import Counter
import hashlib,json
V=Path(__file__).resolve().parent.parent;O=Path(__file__).resolve().parent

def bind(p):return dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
def check(b):assert bind(Path(b['path']))==b;return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(s)for s in Path(p).read_text().splitlines()if s.strip()]
def main():
 assert not(O/'REVIEW.json').exists();d=read(V/'ROOT_C17_DESIGN_BEFORE_SOURCE_FIT.json');pol=read(check(d['decisions_before_baseline_read']));r=rows(check(d['wide_input_occurrences']));g=rows(check(d['wide_GT_occurrences']));assert len(r)==len(g)==4896
 freeze=read(check(d['warmstart_model_freeze']));assert freeze['selected_update']==4500 and freeze['checkpoint']==d['warmstart_checkpoint']and freeze['threshold_logit']==0 and freeze['head_class_code']==d['head_class_code']and freeze['trainable_parameters']==221825 and freeze['encoder_trainable_parameters']==0 and freeze['current_used_for_fit_or_selection']is False
 assert pol['warmstart']==d['warmstart_model_freeze']and d['decisions_before_baseline_read']==bind(V/'ROOT_C17_DECISIONS_BEFORE_BASELINE_READ.json')
 assert pol['optimizer']==dict(type='AdamW',fresh_optimizer_for_changed_objective=True,lr=.0001,weight_decay=.001,clip_norm=5,updates=1000,early_stop=False,extensions_inside_run=False,torch_seed=219)
 quotas=pol['sampling']['additional_wide64'];assert quotas=={'E_alpha_outside':16,'E_alpha_inside':16,'D_SHAM_alpha_inside':16,'D_SHAM_alpha_outside':8,'D_BEFORE':8}
 assert pol['sampling']['wide_generator_seed']==220 and pol['sampling']['replacement']and not pol['sampling']['S_U_in_loss']and not pol['sampling']['known_unavailable_in_loss']
 assert sum(quotas.values())==64 and sum(quotas[k]for k in quotas if k.startswith('E_'))==32 and sum(quotas[k]for k in quotas if k.startswith('D_'))==32
 assert pol['selection']['evaluate_updates']==list(range(0,1001,100))and not pol['selection']['current_used']and pol['selection']['preserve_exact_old15_regression_gate_definitions']
 pools={k:[]for k in quotas};foldcounts={};known_train=[];excluded=Counter()
 for i,(a,b)in enumerate(zip(r,g)):
  assert a['occurrence_index']==b['occurrence_index']==i and a['occurrence_id']==b['occurrence_id']and a['cue_index']==b['cue_index']and a['jitter_index']==b['jitter_index']and a['variant']==b['variant']and a['donor_rank']==b['donor_rank']
  y=b['candidate_aux_hidden_label_GT_only'];assert y in(-1,0,1)and b['native_mapping']=='UNKNOWN'and not b['human_gold']
  assert a['query_available']==(a['factual_query_available']and a['context_available'])
  key=a['source']+'::'+a['raw_split'];fc=foldcounts.setdefault(key,Counter());fc['all_rows']+=1;fc['teacher_D'if y==0 else'teacher_E'if y==1 else'teacher_U']+=1
  if y in(0,1):fc['known_available'if a['query_available']else'known_input_U']+=1
  if y<0:excluded['teacher_U']+=1
  if a['split']!='TRAIN':excluded['nonTRAIN_rows']+=1
  if y in(0,1)and not a['query_available']:excluded['known_input_U']+=1
  if a['split']!='TRAIN'or y not in(0,1)or not a['query_available']:continue
  assert a['source']=='LOCO'and a['raw_split']=='TRAIN';known_train.append(i);alpha=b['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only']
  if y==1:
   assert a['variant']=='OPAQUE_COVER'and type(alpha)is bool;pools['E_alpha_inside'if alpha else'E_alpha_outside'].append(i)
  elif a['variant']=='BEFORE':assert alpha is None;pools['D_BEFORE'].append(i)
  else:assert a['variant']=='CUE_PRESERVING_SHAM'and type(alpha)is bool;pools['D_SHAM_alpha_inside'if alpha else'D_SHAM_alpha_outside'].append(i)
 assert all(pools.values())and len(known_train)==795 and len(set(sum(pools.values(),[])))==795 and sorted(sum(pools.values(),[]))==known_train
 expected={'E_alpha_outside':156,'E_alpha_inside':216,'D_SHAM_alpha_inside':77,'D_SHAM_alpha_outside':295,'D_BEFORE':51};assert {k:len(v)for k,v in pools.items()}==expected
 summaries={k:dict(N=len(ix),quota=quotas[k],D=sum(g[j]['candidate_aux_hidden_label_GT_only']==0 for j in ix),E=sum(g[j]['candidate_aux_hidden_label_GT_only']==1 for j in ix),unique_cues=len({r[j]['cue_index']for j in ix}),unique_original_frames=len({r[j]['source_frame_group']for j in ix}),unique_cue_rank_pairs=len({(r[j]['cue_index'],r[j]['donor_rank'])for j in ix if r[j]['donor_rank']is not None}))for k,ix in pools.items()}
 selector={};checks={}
 for source,fold,name in [('LOCO','VALIDATION','LOCO_VALIDATION'),('TOMIE','TOMIE_DEVELOPMENT_STRESS','TOMIE_DEVELOPMENT')]:
  ix=[i for i,a in enumerate(r)if a['source']==source and a['raw_split']==fold];
  for y,state in[(0,'D'),(1,'E')]:selector[name+'_'+state]=dict(count=sum(g[i]['candidate_aux_hidden_label_GT_only']==y for i in ix),known_input_U=sum(g[i]['candidate_aux_hidden_label_GT_only']==y and not r[i]['query_available']for i in ix))
  checks[name]=dict(known_D=selector[name+'_D']['count'],known_E=selector[name+'_E']['count'],E_q_alpha_outside=sum(g[i]['candidate_aux_hidden_label_GT_only']==1 and g[i]['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only']is False for i in ix),SHAM_D_q_alpha_inside=sum(g[i]['candidate_aux_hidden_label_GT_only']==0 and r[i]['variant']=='CUE_PRESERVING_SHAM'and g[i]['nearest_q_inside_pasted_foreground_alpha_GT_diagnostic_only']is True for i in ix))
 result=dict(completed=True,PASS=True,blockers=[],scope='Prospective policy and actual PUBLIC metadata label/pool review only. Trainer implementation/generated is outside this review.',root_design=bind(V/'ROOT_C17_DESIGN_BEFORE_SOURCE_FIT.json'),decisions=d['decisions_before_baseline_read'],input_manifest=d['wide_input_occurrences'],GT_sidecar=d['wide_GT_occurrences'],warmstart_model_freeze=d['warmstart_model_freeze'],warmstart_checkpoint_descriptor=d['warmstart_checkpoint'],script=bind(Path(__file__)),all4896_joined=True,wide_TRAIN_pool_counts=summaries,wide_known_available_TRAIN_unique_rows=795,wide_TRAIN_D423_E372=True,by_source_rawfold={k:dict(v)for k,v in foldcounts.items()},exclusions=dict(excluded),wide_S_rows=0,loss_known_AVAILABLE_TRAIN_only=True,pools_exhaustively_partition_eligible_rows=True,wide64_D32_E32=True,combined256_D128_E128='Old192D96/E96 retained from exact C15 contract +wide64D32/E32; old192 helper implementation not re-executed here.',selector_new4=selector,source_diagnostic_denominators=checks,old14_cohorts_unchanged_plus_new4_equals18=True,selector_meaning_KO='기존14와 wide4의 cohort 평균BCE를 각1/18로 동일 가중한다. 같은 원 장면/weakcue/합성물을 좁은q와 넓은q로 다시 보므로 새 독립18개 데이터셋이 아니다. Known inputU는 safe0BCE와 평가분모에 남고 loss pool에서만 제외한다.',loss_meaning_KO='old192 loss(원old128평균BCE+.25원24pair hinge+원V12264평균BCE)에 wide64평균BCE를 더한다. 전체256행의 단일 평균BCE가 아니며 wide64는 old128 대비 행당BCE 가중이2배다. 새 hinge/S/U/VAL loss는 없다.',advantages_KO=['동일 고정 cue를 숨기는 외부물체와 q pixel의 alpha를 분리해 경계 위치 shortcut을 직접 대조한다.','원 uncertainty/physicalrecipe/fold와 centre 대조가 보존되며 falseD/E는 conditional component semantics로 한정된다.'],limits_KO=['학습 가능 wide795행은6LOCO weakcues에 반복 offsets/donor를 적용한 상관 자료다. 새 독립 원본이나 자연 E gold가 아니다.','D_SHAM-q안77행을16개씩 반복추출하므로 작은 stratum의 반복노출이 의도적으로 커진다. 불균형/독립표본 해소로 표현하면 안 된다.','q-alpha밖은 target pixel 인증이 아니고, semantic cue D/E는 native8/정확pixel/좌표오차 gold가 아니다.','TOMIE는 모두 개발 validation이며 TRAIN에 들어가지 않는다. 기존 sourceVAL 노출은 계속 development 성격이다.'],new_gate_or_approval_added=False,actual_feature_score_weight_model_current_V112_reads_or_calls=0)
 (O/'REVIEW.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(review=bind(O/'REVIEW.json'),pools=expected,selector_new4=selector),ensure_ascii=False))
if __name__=='__main__':main()
