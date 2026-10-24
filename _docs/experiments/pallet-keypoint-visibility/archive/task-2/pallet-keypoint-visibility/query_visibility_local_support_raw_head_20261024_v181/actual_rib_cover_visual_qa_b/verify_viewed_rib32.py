"""Verify saved display/ledger bytes for the four actually viewed fixed cards.

No source scene, GT buffer, input cache, checkpoint, feature or model is loaded.
"""
from pathlib import Path
import hashlib,json
from datetime import datetime,timezone
import numpy as np
from PIL import Image

H=Path(__file__).resolve().parent;V=H.parent;G=V/'fixed_source_visual288_v1'
IDS=['VAL_SCENE_000641','VAL_SCENE_000645','VAL_SCENE_000705','VAL_SCENE_000709']
ARMS=['C50_RAW96_DUAL_SUPPORT_OWNER_AUX','C51_RAW96_PLUS_FROZEN_SAM_DUAL_SUPPORT_OWNER_AUX']
V61=V.parent/'query_visibility_actual_initializer_source_20261024_v132/frozen_v61_actual_initializer_v1/PREDICTION_SEAL.json'
READ_FROM_ACTUAL_IMAGES={'VAL_SCENE_000641':['D','E','E','E','E','E','D','D'],
 'VAL_SCENE_000645':['D','E','E','E','E','E','D','D'],
 'VAL_SCENE_000705':['D','E','E','E','E','E','D','E'],
 'VAL_SCENE_000709':['E','E','E','E','E','E','D','E']}
def req(ok,msg):
 if not bool(ok):raise AssertionError(msg)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bind(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def check(b):req(sha(b['path'])==b['sha256'],'Binding '+b['path']);return Path(b['path'])
def read(p):return json.loads(Path(p).read_text())
def ash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def run():
 complete=read(G/'COMPLETE.json');req(sha(G/'COMPLETE.json')=='26647fd290a967de3a2b7f679eb30eb60fd6e92256a669cc17306d3fc12e1eaf','Root terminal gallery binding')
 req(complete['completed'] and complete['PASS'],'Gallery complete')
 inventory=read(check(complete['card_contact_inventory']));cards={r['group_id']:r for r in inventory['cards']}
 ledger=[json.loads(s) for s in check(complete['manifest']).read_text().splitlines()]
 selected=[r for r in ledger if r['group_id'] in IDS];req(len(selected)==32,'Fixed four native8 groups')
 fit=read(check(complete['fit_complete']));req(fit['completed'] and fit['PASS'] and fit['all2_frozen_before_outputs_or_VAL_access'],'Both heads frozen')
 outputs={}
 for arm in ARMS:
  with np.load(check(fit['arms'][arm]['outputs']['SYNTH_VAL2048']),allow_pickle=False) as z:
   keys=['raw_support_logits','raw_owner_logits','support_prediction','owner_prediction','regional_support_proxy_D0_E1_Uminus1','original_global_index','native_slot','query_xy','L']
   outputs[arm]={k:z[k].copy() for k in keys}
 old=read(V61)
 with np.load(check(old['predictions']),allow_pickle=False) as z:
  v61={k:z[k].copy() for k in ['global_index','raw_visible_logit','prediction_D0_E1_Uminus1','query_available','native_slot','query_xy','L']}
 vmap={int(g):i for i,g in enumerate(v61['global_index'])};checks=[];viewed=[]
 for group in IDS:
  rec=cards[group];card=np.asarray(Image.open(check(rec['image'])).convert('RGB'))
  viewed.append(dict(**rec['image'],actual_tool='tools.view_image',detail='original',all_native0_to7_actually_seen=True))
  rows=sorted([r for r in selected if r['group_id']==group],key=lambda r:r['native_slot'])
  req([r['native_slot'] for r in rows]==list(range(8)),'All native slots')
  for k,r in enumerate(rows):
   tx=20+(k%2)*1140;ty=125+(k//2)*480;y=ty+85
   clean=np.asarray(Image.open(check(r['clean96'])).convert('RGB'));req(clean.shape==(96,96,3) and ash(clean)==r['rgb96_array_sha256'],'Original clean96 digest')
   req(np.array_equal(card[y:y+192,tx+14:tx+206],clean.repeat(2,0).repeat(2,1)),'Raw RGB display is exact nearest2 unchanged pixels')
   for field,offset,size in [('valid96_array_sha256',226,96),('sam96_array_sha256',342,192),('sam_available96_array_sha256',554,96)]:
    panel=card[y:y+size,tx+offset:tx+offset+size]
    req(np.array_equal(panel[:,:,0],panel[:,:,1]) and np.array_equal(panel[:,:,0],panel[:,:,2]) and np.isin(panel,[0,255]).all(),'Boolean display panel')
    sampled=panel[::2,::2,0] if size==192 else panel[:,:,0];arr=sampled==255
    req(ash(arr)==r[field],'Displayed valid/SAM panel digest')
    if size==192:req(np.array_equal(panel,np.repeat(np.repeat((arr.astype(np.uint8)*255)[:,:,None],3,2),2,0).repeat(2,1)),'SAM exact nearest2')
   for arm in ARMS:
    z=outputs[arm];i=r['cache_index'];m=r['models'][arm]
    req(int(z['original_global_index'][i])==r['original_global_index'] and int(z['native_slot'][i])==k,'Saved output original ID')
    req(np.array_equal(z['query_xy'][i],np.asarray(r['query_xy'],np.float64)) and z['L'][i]==r['L'],'Saved original q/L')
    for key in ['raw_support_logits','raw_owner_logits']:
     req(np.array_equal(z[key][i],np.asarray(m[key],np.float32)),'Ledger matches frozen raw logits')
    req(z['support_prediction'][i].tolist()==m['support_prediction'] and int(z['owner_prediction'][i])==m['owner_prediction'] and int(z['regional_support_proxy_D0_E1_Uminus1'][i])==m['proxy'],'Ledger saved responses')
    req((z['raw_support_logits'][i]>0).astype(np.int8).tolist()==m['support_prediction'],'Fixed strict0 support decoder')
   j=vmap[r['original_global_index']];label='U' if not v61['query_available'][j] else 'D' if v61['raw_visible_logit'][j]>=0 else 'E'
   req(label==READ_FROM_ACTUAL_IMAGES[group][k],'Actually read v61 caption equals inherited saved packet')
   req(int(v61['native_slot'][j])==k and np.array_equal(v61['query_xy'][j],np.asarray(r['query_xy'],np.float64)) and v61['L'][j]==r['L'],'v61 original same q/native/L')
   checks.append(dict(case_id=r['case_id'],original_global_index=r['original_global_index'],native_slot=k,raw_valid_SAM_panel_byte_hash_verified=True,support_truth=r['support_truth']['support_target'],C50=r['models'][ARMS[0]]['support_prediction'],C51=r['models'][ARMS[1]]['support_prediction'],historical_v61=label))
 eval_summary=V/'saved_local_support_source_evaluation_v1/FIXED_SUPPORT_OWNER_AND_STRATA.json'
 es=read(eval_summary);scope_limits={}
 for arm in ARMS:
  tr=es['results'][arm]['SYNTH_TRAIN4096']['ALL'];va=es['results'][arm]['SYNTH_VAL2048']['ALL']
  scope_limits[arm]=dict(TRAIN_apparent_visible_known=tr['visible_end_support']['primary_known_available']['class_support'],TRAIN_apparent_visible_recall=tr['visible_end_support']['primary_known_available']['recall'],TRAIN_apparent_foreign_known=tr['foreign_front_support']['primary_known_available']['class_support'],TRAIN_apparent_foreign_recall=tr['foreign_front_support']['primary_known_available']['recall'],VAL_foreign_known=va['foreign_front_support']['primary_known_available']['class_support'],VAL_foreign_precision_YES=va['foreign_front_support']['primary_known_available']['precision'][1],VAL_foreign_recall_YES=va['foreign_front_support']['primary_known_available']['recall'][1])
 report=dict(completed=True,PASS=True,UTC=datetime.now(timezone.utc).isoformat(),code=bind(__file__),gallery_complete=bind(G/'COMPLETE.json'),gallery_manifest=complete['manifest'],card_inventory=complete['card_contact_inventory'],fit_complete=complete['fit_complete'],source_eval_summary=bind(eval_summary),historical_v61_seal=bind(V61),actual_card_images_viewed=4,native_tiles_actually_viewed=32,actually_viewed_images=viewed,per_case_binding_checks=checks,
  observations_KO=[
   '32개는 RIB COVER 고정 네 scene의 원native0..7 전부이며 대표 표본이 아니다. 원RGB·valid·SAM·SAMavailability의 card panel과 clean96/ledger 해시를 대조했다. 겹침 없이 q/L·두 head·GT·v61 캡션을 읽을 수 있었다.',
   'g5131/5163/5642/5673/5674 등은 중심 owner BG 질량이 크거나 1이어도 근처 끝 부위의 녹색 support가 있다. 두 head는 visible YES를 내고 v61은 E를 보여, literal q-owner/pixel presence와 regional end support의 차이가 드러난다. 이 관찰은 q 오류나 physical native 가시성 gold 인증이 아니다.',
   '같은 재질 작은 갈색 COVER(g5129)와 대비 재질 COVER(g5161), 큰 빨강 COVER(g5643/g5675)는 고정 foreign-support YES에 두 head 모두 [0,1]로 응답했다. SAM 흰 실루엣은 일부 작은 COVER 경계나 구멍을 합쳐 보이므로 SAM 자체를 정답 마스크로 해석할 수 없다.',
   'SAM 추가가 일관된 개선은 아니다. g5640의 fixed [1,0]에 C51 [0,0]으로 proxy U, g5641 [1,0]에 C51 [1,1]로 proxy E다. g5672는 fixed [1,0]인데 두 head 모두 [0,1]이다. RGB에는 끝 부분과 그 주변 빨강 foreign이 함께 보이며 literal owner F와 end-support V가 다르다.',
   '파랑 UNKNOWN_SELF_HIDDEN 표시 및 target[-1,-1] 타일(예 g5134/g5646/g5678)은 모델의 YES/E 응답을 새로운 정답이나 오류 수로 바꾸지 않았다. 비배타 출력 [1,1] 및 input-available이지만 support-proxy U인 경우를 그대로 보존했다.'
  ],source_TRAIN_and_VAL_limit_from_saved_summary=scope_limits,
  limitations=['Actual view scope is these four cards/32 native tiles only, not all288 gallery cases.','TRAIN known-support recall1.0 is apparent fit on named supervision; source VAL foreign precision/recall remain moderate. No public-held/private/current improvement is measured by this QA.','Synthetic VAL has historical research exposure; no architecture/data/threshold/arm/checkpoint choice follows from these pictures.','GT is fixed constructive source-end support; original CF/V4/native identity and reference3px diagnostic remain uncertified physical gold.','HTML was not browser rendered.'],source_scene_or_GT_buffer_inputcache_feature_weight_or_model_read=False,saved_output_members_loaded='Two final SYNTH_VAL2048 NPZ members and historical v61 source6144 prediction/metadata members; selected32 compared, no model forward.',new_labels_qL_training_admission_fit_or_model_selection=0,selected_model=None,fully_solved=False)
 out=H/'ACTUAL_VISUAL_QA.json';req(not out.exists(),'Preserve prior QA')
 out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps(bind(out)),flush=True)

if __name__=='__main__':run()
