"""SOURCE-only saved scalar YCB query/reference + object/aux join; Root entry."""
import argparse,csv,hashlib,io,json,math,os,stat,traceback
from pathlib import Path
R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation');H=Path(__file__).absolute().parent
BASE='BASELINE_V239_CANONICAL_CONTEXT';TREAT='TREAT_SCENE_CF_MATCHED_DATA'
META_PIN={'path':str(R/'V236_PUBLICATION_PACKAGE/artifacts/V234_ROOT_LABEL_DIAG/YCB_KNOWN1521_LABEL_QUERY_MODEL_DIAGNOSTIC.csv'),'sha256':'8eab4fe1f9c58ba55ab308437605cc64a2c30dc92f07d21416759b455a342b08','size_bytes':474209}
PLAN_SHA='50e547b4c07afdee690727d895278ad7b82fe80eab52551dd00b9f5eb7a5d0e3'
LEDGER_SHA='b2c1c3bff32981b2570f0330353c3c4f54fcfa4fb41d209b0dacfccfc1ac0dae'


def lexical(value):
 p=Path(value)
 if any(t in str(p).casefold() for t in ('v112','__pycache__','.pyc','current223','publicheld','oldval','v177','v178','source_holdout','v182','private_evaluation')) or not p.is_absolute() or '..' in p.parts:raise ValueError('Forbidden lexical path before FS')
 return p


def safe(value):
 p=lexical(value)
 for a in reversed((p,*p.parents)):
  try:mode=os.lstat(a).st_mode
  except FileNotFoundError:continue
  if stat.S_ISLNK(mode):raise ValueError('Symlink ancestor')
 return p


def ident(b):
 if not isinstance(b,dict) or not {'path','sha256'}<=set(b):raise ValueError('Exact metadata binding required')
 return str(lexical(b['path'])),b['sha256']


def read(p):
 with os.fdopen(os.open(safe(p),os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:return f.read()


def binding(p):
 raw=read(p);return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':len(raw)}


def checked(b,white,pin=None):
 if ident(b) not in {ident(v) for v in white} or (pin is not None and ident(b)!=ident(pin)):raise ValueError('Exact whitelist before FS')
 raw=read(b['path'])
 if hashlib.sha256(raw).hexdigest()!=b['sha256'] or ('size_bytes'in b and len(raw)!=b['size_bytes']):raise ValueError('Saved metadata bytes changed')
 return raw


def exact(a,b):
 if ident(a)!=ident(b):raise ValueError('Metadata graph mismatch')


def require_PASS(value):
 if value.get('completed') is not True or value.get('PASS') is not True:raise ValueError('Actual completed/PASS required')


def write_json(path,v):
 with safe(path).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 return binding(path)


def write_csv(path,rows):
 with safe(path).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return binding(path)


def saved_YCB_metadata(raw):
 rows={};sample_ids=set()
 for v in csv.DictReader(io.StringIO(raw.decode())):
  rid=v['row_id'];i=int(v['sample_index']);state=v['associated_r_state'];role=v['role'];obj=int(v['object_id']);aux=int(v['aux_slot'])
  if rid in rows or i in sample_ids or state not in ('D','E') or role not in ('FIT','DEV'):raise ValueError('Unique all1521 SOURCE knownD/E role metadata')
  value=v['q_reference_distance_px'];distance=None if value.strip().casefold() in ('','nan','none','null') else float(value)
  if distance is not None and not math.isfinite(distance):distance=None
  if distance is not None and distance<0:raise ValueError('Invalid negative saved q-reference distance; no substitute')
  bin_='UNKNOWN_DISTANCE' if distance is None else 'LE3' if distance<=3 else 'GT3'
  flag=(obj==7 and aux in (3,7)) or (obj==2 and aux==2)
  rows[rid]={'row_id':rid,'sample_index':i,'role':role,'target':0 if state=='D' else 1,'associated_r_state':state,'object_id':obj,'logical_aux_slot':aux,'saved_q_reference_distance_px':distance,'q_reference_bin':bin_,'object_aux_group':'FIXED_FLAGGED' if flag else 'OTHER'};sample_ids.add(i)
 if len(rows)!=1521:raise ValueError('All1521 YCB known source rows; no exclusions')
 fixed={(role,y):sum(v['role']==role and v['target']==y for v in rows.values()) for role in ('FIT','DEV') for y in (0,1)}
 if fixed!={('FIT',0):841,('FIT',1):283,('DEV',0):340,('DEV',1):57}:raise ValueError('Original SOURCE split/target counts unchanged')
 return rows


def natural_YCB_rows(raw,metadata):
 found={};count=0
 for i,v in enumerate(csv.DictReader(io.StringIO(raw.decode()))):
  count+=1
  if int(v['sample_index'])!=i:raise ValueError('Original7617 natural CSV ordering')
  rid=v['row_id']
  if rid not in metadata:continue
  old=metadata[rid]
  if rid in found or i!=old['sample_index'] or int(v['family'])!=1 or v['role']!=old['role'] or int(v['target'])!=old['target'] or v['available'] not in ('True','False'):raise ValueError('Exact originalYCB row/sample/family/role/teacher join')
  available=v['available']=='True';z=None if v['selected_family_logit']=='' else float(v['selected_family_logit']);prediction=v['prediction']
  if available:
   if z is None or not math.isfinite(z) or z!=float(v['logit1']) or prediction!=('E' if z>=0 else 'D'):raise ValueError('Family1 binary selected logit threshold0')
  elif z is not None or prediction!='U':raise ValueError('Original inputU remainsU; no confidence U')
  found[rid]={'available':available,'selected_logit':z,'prediction':prediction}
 if count!=7617 or set(found)!=set(metadata):raise ValueError('Exact all1521 rowIDs inside original7617 natural output')
 return found


def reduce(rows,arm):
 rr=[{'target':r['target'],'pred':r[arm+'_prediction']} for r in rows]
 cm=[[sum(v['target']==y and v['pred']==p for v in rr) for p in ('D','E','U')] for y in (0,1)]
 out={'N':len(rr),'DD':cm[0][0],'DE':cm[0][1],'DU':cm[0][2],'ED':cm[1][0],'EE':cm[1][1],'EU':cm[1][2],'D_truth':sum(cm[0]),'E_truth':sum(cm[1]),'U_count':cm[0][2]+cm[1][2]}
 for k,label in enumerate(('D','E')):
  truth=sum(cm[k]);pred=cm[0][k]+cm[1][k];tp=cm[k][k]
  out[label+'_precision']=tp/pred if pred else None;out[label+'_recall']=tp/truth if truth else None
 out['balanced_recall']=None if any(out[k+'_recall'] is None for k in ('D','E')) else .5*(out['D_recall']+out['E_recall'])
 return out


def strata_tables(rows):
 metrics=[];paired=[]
 for role in ('ALL','FIT','DEV'):
  for bin_ in ('ALL','LE3','GT3','UNKNOWN_DISTANCE'):
   for group in ('ALL','FIXED_FLAGGED','OTHER'):
    take=[r for r in rows if (role=='ALL' or r['role']==role) and (bin_=='ALL' or r['q_reference_bin']==bin_) and (group=='ALL' or r['object_aux_group']==group)]
    for arm in (BASE,TREAT):metrics.append({'source_role':role,'q_reference_bin':bin_,'object_aux_group':group,'arm':arm,**reduce(take,arm)})
    recovered=damaged=0
    for r in take:
     truth='D' if r['target']==0 else 'E';a=r[BASE+'_prediction']==truth;b=r[TREAT+'_prediction']==truth
     recovered+=not a and b;damaged+=a and not b
    paired.append({'source_role':role,'q_reference_bin':bin_,'object_aux_group':group,'N':len(take),'D_truth':sum(r['target']==0 for r in take),'E_truth':sum(r['target']==1 for r in take),'recovered':recovered,'damaged':damaged,'net':recovered-damaged})
 return metrics,paired


def run(entry_path):
 e=json.loads(read(entry_path))
 if e.get('approved_SOURCE_saved_query_error_join') is not True or e.get('source_only') is not True or e.get('automatic_retry') is not False:raise PermissionError('Separate Root savedSOURCE scalar join entry required')
 code,white=e['code_whitelist'],e['metadata_whitelist']
 checked(e['code'],code,binding(Path(__file__).absolute()));checked(e['implementation_protocol'],code)
 fit=json.loads(checked(e['fit_complete'],white));qa=json.loads(checked(e['fit_QA'],white));freeze=json.loads(checked(e['freeze'],white))
 for v in (fit,qa,freeze):require_PASS(v)
 exact(qa['fit_complete'],e['fit_complete']);exact(e['freeze'],fit['freeze'])
 if fit['source_plan_sha256']!=PLAN_SHA or fit['draw_ledger']['sha256']!=LEDGER_SHA or fit['new_optimizer_updates']!={BASE:0,TREAT:1000}:raise ValueError('Actual matched1000 plan frozen before SOURCE analysis')
 for arm in (BASE,TREAT):exact(e['CSV'][arm],fit['outputs'][arm]);exact(freeze['checkpoints'][arm],fit['checkpoints'][arm]);exact(freeze['logs'][arm],fit['logs'][arm])
 if fit['new_DEV_head_output_before_finalfreeze']!=0 or e['CSV'][BASE]['sha256']!='5403ba12c269abe4222e57c5534aa6d97b77fba13e0d34ca55efea0ad8f13ecb':raise ValueError('Exact oldV239TREAT baseline, no earlyDEV')
 out=safe(e['output_directory'])
 if out.parent!=H or out.name!='results_v1' or out.exists():raise ValueError('Fresh own scalar output only')
 out.mkdir()
 try:
  metadata=saved_YCB_metadata(checked(e['YCB_metadata'],white,META_PIN))
  predictions={arm:natural_YCB_rows(checked(e['CSV'][arm],white),metadata) for arm in (BASE,TREAT)}
  rows=[]
  for v in sorted(metadata.values(),key=lambda r:r['sample_index']):
   row=dict(v);rid=v['row_id']
   if predictions[BASE][rid]['available']!=predictions[TREAT][rid]['available']:raise ValueError('Same originalinputavailability')
   row['original_available']=predictions[BASE][rid]['available']
   for arm in (BASE,TREAT):row[arm+'_prediction']=predictions[arm][rid]['prediction'];row[arm+'_logit']=predictions[arm][rid]['selected_logit']
   truth='D' if row['target']==0 else 'E';a=row[BASE+'_prediction']==truth;b=row[TREAT+'_prediction']==truth
   row['paired_change']='RECOVERED' if not a and b else 'DAMAGED' if a and not b else 'BOTH_CORRECT' if b else 'BOTH_WRONG';rows.append(row)
  tables,pairs=strata_tables(rows)
  joined=write_csv(out/'ALL1521_YCB_QUERY_ERROR_ROWS.csv',rows);metrics=write_csv(out/'METRICS.csv',tables);changes=write_csv(out/'PAIRED_CHANGES.csv',pairs)
  summary=write_json(out/'SUMMARY.json',{'completed':True,'PASS':True,'YCB_rows':1521,'FIT':1124,'DEV':397,'distance_bins':{b:sum(r['q_reference_bin']==b for r in rows) for b in ('LE3','GT3','UNKNOWN_DISTANCE')},'fixed_failure_combos':[[7,3],[7,7],[2,2]],'strata_metrics':tables,'paired':pairs,'metadata_used':['row_id','sample_index','role','associated_r_state','object_id','aux_slot','q_reference_distance_px'],'new_targets_created':False,'new_threshold_or_model_choice':False,'limits':['Saved YCB source q-reference displacement is constructed query jitter relative to associated-r, NOT real pallet detector localization error.','Labels concern associated-r surface visibility and native remainsUNKNOWN, not physical pallet native-point gold or literalq-owner targets.','SOURCE DEV is exposed/reused. Fine-grained strata are descriptive checks, not new model/data/checkpoint/threshold selection.','Distance-missing rows are retained in UNKNOWN_DISTANCE; empty bins have explicit zero counts and null precision/recall, no substitution.','Only row identity/role/target/object/aux/saved-distance fields from oldmetadata are used; old score, coordinates/reference/owner/depth columns are not used.','No PNG/RGB/feature/weight/GT/newmodel/optimizer/currentheld access.']})
  lines=['# 저장 SOURCE YCB 거리·object/aux 비교','', 'YCB의 저장 q–r 거리를 `≤3px`/`>3px`로 나누고, 사전 고정 조합 obj7 aux3·7 또는 obj2 aux2와 나머지를 비교했습니다. 같은1521행을 보존했으며 SOURCE FIT1124/DEV397을 분리했습니다.','', '이 거리는 associated-r 주변에서 만든 SOURCE query의 변위입니다. 실제 pallet 초기 검출 오차나 native physical gold로 해석하지 않습니다. SOURCE DEV는 이미 노출·재사용됐으며, 세부 표는 새 모델·데이터·문턱 선택에 사용하지 않습니다.','', '| role | q–r bin | group | arm | N | D recall | E recall |','|---|---|---|---|---:|---:|---:|']
  for row in tables:
   if row['source_role']!='DEV':continue
   val=lambda v:'N/A' if v is None else f'{v:.4f}'
   lines.append(f"| {row['source_role']} | {row['q_reference_bin']} | {row['object_aux_group']} | {row['arm']} | {row['N']} | {val(row['D_recall'])} | {val(row['E_recall'])} |")
  lines.extend(['','[전체1521행](ALL1521_YCB_QUERY_ERROR_ROWS.csv) · [모든 분모/CM/P/R/U](METRICS.csv) · [회복·악화](PAIRED_CHANGES.csv) · [SUMMARY](SUMMARY.json)'])
  with (out/'KO.md').open('x') as f:f.write('\n'.join(lines)+'\n')
  complete=write_json(out/'COMPLETE.json',{'completed':True,'PASS':True,'entry':binding(entry_path),'code':e['code'],'protocol':e['implementation_protocol'],'fit_complete':e['fit_complete'],'fit_QA':e['fit_QA'],'freeze':e['freeze'],'CSV':e['CSV'],'YCB_metadata':e['YCB_metadata'],'summary':summary,'joined':joined,'metrics':metrics,'paired_changes':changes,'report':binding(out/'KO.md'),'all1521_rowids_exact':True,'new_RGB_features_weights_GT_models_or_currentheld_reads':0,'new_targets_or_model_data_threshold_selection':False,'automatic_retry':False});print(json.dumps({'completed':True,'PASS':True,'COMPLETE':complete}),flush=True)
 except BaseException:
  write_json(out/'FAILED_EXECUTION.json',{'completed':False,'PASS':False,'automatic_retry':False,'traceback':traceback.format_exc()});raise


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--entry',required=True);run(p.parse_args().entry)
