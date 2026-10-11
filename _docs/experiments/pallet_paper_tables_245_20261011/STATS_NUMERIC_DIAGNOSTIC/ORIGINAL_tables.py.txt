"""Render the predeclared paper tables from saved statistics only."""
from __future__ import annotations
import csv
from pathlib import Path
from . import stats as S

MAIN_COLUMNS=('scope','backbone','method','paper_name','frames','seeds','pose_available_n','pose_all_seed_n',
 'PCK10_percent','T_mean_cm','T_SD_cm','R_mean_deg','R_SD_deg','success_percent','confusion_percent',
 'latency_mean_ms','latency_SD_ms','T_median_cm','T_P90_cm','R_median_deg','R_P90_deg','ADDsym_AUC','IoU3D_mean')

def fmt(value,precision=2):return '—' if value is None else f'{value:.{precision}f}'
def pm(mean,sd):return fmt(mean)+' ± '+fmt(sd)
def md(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'
def write_text(name,value):
 with (S.DOC/name).open('x') as f:f.write(value)
def write_csv(name,columns,rows):
 with (S.DOC/name).open('x',newline='') as f:
  writer=csv.DictWriter(f,columns);writer.writeheader();writer.writerows({k:r.get(k) for k in columns} for r in rows)

def latency_data():
 path=S.DOC/'LATENCY.json'
 if not path.is_file():return {}
 value=S.read(path)
 # Receipts can use method keys or explicit row records; no inference of timing.
 arms=value.get('arms',value.get('methods',value.get('results',{})))
 if isinstance(arms,list):arms={r.get('method',r.get('name')):r for r in arms}
 result={}
 for method,p in arms.items():
  if not isinstance(p,dict):continue
  stats=p.get('statistics_ms',p.get('summary',p.get('wall_ms',p)))
  if not isinstance(stats,dict):continue
  result[method]=dict(mean=stats.get('mean_ms',stats.get('mean')),std=stats.get('SD_ms',stats.get('sd_ms',stats.get('std_ms',stats.get('std')))))
 return result

def cell(metrics,scope,backbone,method,latency):
 group=metrics['scopes'][scope]['groups'][backbone+'::'+method];m=group['seed_mean'];p=m['metrics']
 time=latency.get(method,{}) if backbone=='YOLO' and scope!='SQUARE118' else {}
 return dict(scope=scope,backbone=backbone,method=method,paper_name=S.ENGLISH[method],frames=group['frames'],
  seeds=','.join(map(str,group['seeds'])),pose_available_n=m['available_frames_any_seed'],pose_all_seed_n=m['available_frames_all_seeds'],
  PCK10_percent=100*m['PCK10']['rate'],T_mean_cm=p['T_cm']['mean'],T_SD_cm=p['T_cm']['std'],
  R_mean_deg=p['R_deg']['mean'],R_SD_deg=p['R_deg']['std'],success_percent=100*p['success_rate']['mean'],
  confusion_percent=100*p['confusion_rate']['mean'] if p['confusion_rate']['mean'] is not None else None,
  latency_mean_ms=time.get('mean'),latency_SD_ms=time.get('std'),
  T_median_cm=p['T_cm']['median'],T_P90_cm=p['T_cm']['P90'],R_median_deg=p['R_deg']['median'],R_P90_deg=p['R_deg']['P90'],
  ADDsym_AUC=p['ADDsym_AUC']['mean'],IoU3D_mean=p['IoU3D']['mean'])

def main_md(title,records,show_time=False):
 headers=['Scope','Backbone','Method','n / N','Seeds','PCK@10px (%)','Translation (cm)','Rotation (°)','5cm·5° (%)','W/D confusion (%)']
 if show_time:headers.append('Time (ms)')
 headers+=['T med / P90','R med / P90','ADDsym AUC','IoU3D']
 rendered=[]
 for r in records:
  row=[r['scope'],r['backbone'],r['paper_name'],f"{r['pose_available_n']} / {r['frames']}",r['seeds'],fmt(r['PCK10_percent']),
   pm(r['T_mean_cm'],r['T_SD_cm']),pm(r['R_mean_deg'],r['R_SD_deg']),fmt(r['success_percent']),fmt(r['confusion_percent'])]
  if show_time:row.append(pm(r['latency_mean_ms'],r['latency_SD_ms']) if r['latency_mean_ms'] is not None else '측정 결과 대기')
  row +=[fmt(r['T_median_cm'])+' / '+fmt(r['T_P90_cm']),fmt(r['R_median_deg'])+' / '+fmt(r['R_P90_deg']),fmt(r['ADDsym_AUC'],4),fmt(r['IoU3D_mean'],4)]
  rendered.append(row)
 return f'{title}\n\n'+md(headers,rendered)+'\n평균±표본 SD는 영상별 seed 평균의 분포다. `n`은 자세가 한 seed 이상 산출된 영상 수이며, 이진 지표와 PCK는 전체 고정 분모를 사용한다. PCK는 자세 미산출 영상의 성공 코너를 0으로 센다. 정사각형의 가로·깊이 교환은 C4 대칭이므로 W/D confusion을 산출하지 않는다.\n'

def paired_tables(paired):
 columns=('scope','backbone','left','right','metric','delta','CI95_low','CI95_high','paired_frames','full_frames',
  'p_t','p_t_holm','p_t_holm_family_size','t_df','t_target','p_bootstrap','p_bootstrap_holm','p_bootstrap_holm_family_size','bootstrap_target')
 for family,rr in paired['tables'].items():
  records=[];rendered=[]
  for r in rr:
   interval=r.get('CI95') or [None,None];tt=r.get('t_test',{})
   record={**r,'CI95_low':interval[0],'CI95_high':interval[1],'t_df':tt.get('df'),'t_target':tt.get('target'),
    'bootstrap_target':r.get('bootstrap_p_target')};records.append(record)
   rendered.append([r['scope'],r['backbone'],S.ENGLISH[r['left']]+' − '+S.ENGLISH[r['right']],r['metric'],
    fmt(r.get('delta'),6),fmt(interval[0],6)+' / '+fmt(interval[1],6),str(r.get('paired_frames','—')),
    fmt(r.get('p_t'),6),fmt(r.get('p_t_holm'),6),fmt(r.get('p_bootstrap'),6),fmt(r.get('p_bootstrap_holm'),6)])
  write_csv('PAIRED_'+family+'.csv',columns,records)
  text=md(['Scope','Backbone','Comparison','Metric','Δ','95% CI','Paired n','t p','t Holm p','Bootstrap p','Bootstrap Holm p'],rendered)
  text+='\n직사각형 t 검정은 세션 평균 차이의 동일 가중 평균(13개, df=12)을, bootstrap은 영상 수 가중 차이(PCK는 참조 코너 수 가중)를 대상으로 한다. 두 p값의 대상이 다르다. 정사각형 t 검정과 frame CI는 단일 세션의 기술용 결과다. Holm은 각 원래 표 안에서 t와 bootstrap을 각각 보정하며, 표 1은 두 자료의 비교를 함께 묶는다. 표 3은 사전 지정된 중복 방향 비교도 그대로 포함한다.\n'
  write_text('PAIRED_'+family+'.md',text)

def main():
 assert (S.DOC/'METHOD_LOCK.json').is_file()
 metrics=S.read(S.DOC/'METRICS.json');paired=S.read(S.DOC/'PAIRED.json');latency=latency_data()
 table1=[cell(metrics,scope,b,m,latency) for scope in ('RECT245','SQUARE118') for b in ('YOLO','DOPE','ResNet-18') for m in ('BASE','N3_THEN_SUBPIX')]
 table2=[cell(metrics,'RECT245','YOLO',m,latency) for m in ('BASE','SUBPIX','POSEFIX','POSEFIX_SUBPIX','N3_THEN_SUBPIX')]
 table3=[cell(metrics,'RECT245','YOLO',m,latency) for m in S.TABLE3_ORDER]
 supp=[cell(metrics,'RECT245','YOLO',m,latency) for m in ('P','D','L')]
 supp +=[cell(metrics,scope,'YOLO',m,latency) for scope in ('CLEAN153','MODERATE92') for m in ('BASE','N3_THEN_SUBPIX')]
 for name,records,title,show_time in (
  ('TABLE_1',table1,'Table 1. Base and Two-stage across backbones; rectangle and square references remain separate.',False),
  ('TABLE_2',table2,'Table 2. YOLO external comparisons on RECT245.',True),
  ('TABLE_3',table3,'Table 3. YOLO fixed ablations on RECT245.',False),
  ('SUPP',supp,'Supplement. Existing P/D/L comparators and annotation-grade groups.',False)):
  write_csv(name+'.csv',MAIN_COLUMNS,records);write_text(name+'.md',main_md(title,records,show_time))
 supp1=[cell(metrics,'RECT319',b,m,latency) for b in ('YOLO','DOPE','ResNet-18') for m in ('BASE','N3_THEN_SUBPIX')]
 supp2=[cell(metrics,'RECT319','YOLO',m,latency) for m in ('BASE','SUBPIX','POSEFIX','POSEFIX_SUBPIX','N3_THEN_SUBPIX')]
 for name,records in (('SUPPLEMENT_319_TABLE_1',supp1),('SUPPLEMENT_319_TABLE_2',supp2)):
  write_csv(name+'.csv',MAIN_COLUMNS,records);write_text(name+'.md',main_md('All319 frames, including severe74, with the same definitions.',records,'TABLE_2' in name))
 S.write('SUPPLEMENT_319.json',dict(status='COMPLETE',frames=319,severe_frames_retained=74,table1=supp1,table2=supp2,
  source_scope='METRICS.json:scopes.RECT319',manuscript_use='user decides'))
 # All seed-specific statistics, including methods outside main paper tables.
 columns=('scope','backbone','method','seed','frames','pose_available_n','PCK10_percent','T_mean_cm','T_SD_cm','R_mean_deg','R_SD_deg','success_percent','confusion_percent','ADDsym_AUC','IoU3D_mean')
 seedrecords=[]
 for scope,pop in metrics['scopes'].items():
  for group,packet in pop['groups'].items():
   b,m=group.split('::')
   for seed,p in packet['per_seed'].items():
    mm=p['metrics'];seedrecords.append(dict(scope=scope,backbone=b,method=m,seed=seed,frames=packet['frames'],
     pose_available_n=p['available_frames_any_seed'],PCK10_percent=100*p['PCK10']['rate'],
     T_mean_cm=mm['T_cm']['mean'],T_SD_cm=mm['T_cm']['std'],R_mean_deg=mm['R_deg']['mean'],R_SD_deg=mm['R_deg']['std'],
     success_percent=100*mm['success_rate']['mean'],confusion_percent=100*mm['confusion_rate']['mean'] if mm['confusion_rate']['mean'] is not None else None,
     ADDsym_AUC=mm['ADDsym_AUC']['mean'],IoU3D_mean=mm['IoU3D']['mean']))
 write_csv('SEED_TABLES.csv',columns,seedrecords)
 paired_tables(paired)
 S.write('TABLES_EXECUTION.json',dict(status='COMPLETE',source_sha256=S.sha(Path(__file__)),latency_loaded=bool(latency),
  rows=dict(TABLE_1=len(table1),TABLE_2=len(table2),TABLE_3=len(table3),SUPP=len(supp),SEED_TABLES=len(seedrecords)),
  additional_model_forwards=0,additional_PnP_calls=0))
 print('PAPER_TABLES_COMPLETE',len(seedrecords),flush=True)
if __name__=='__main__':main()
