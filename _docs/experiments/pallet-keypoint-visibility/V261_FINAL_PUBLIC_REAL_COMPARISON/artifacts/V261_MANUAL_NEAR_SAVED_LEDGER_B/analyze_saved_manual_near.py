"""Descriptive frozen-ledger counts only. No model, pixels or fitting."""
import argparse,csv,hashlib,io,json,math,os,stat
from collections import Counter
from pathlib import Path

R=Path('/home/minjae/Documents/Codex/2026-10-20/task-2/pallet-keypoint-visibility/publication_root_preparation')
ROOT=R/'V261_ROOT_FINAL_PUBLIC_REAL_COMPARISON'
LEDGER={'path':str(ROOT/'frozen_current_comparison_v1/ALL1784_FINAL_COMPARISON_LEDGER.csv'),'sha256':'c1fa5fc123ac79ba4512b385a21442d9a8eb98df47bc7ed8677b322138fcd409','size_bytes':633576}
CASES={'path':str(ROOT/'final_current_crop_gallery_v1/CASE_FACTS.json'),'sha256':'56bcb01283e93dce73c102ca49860d9853c16b4489928fb2bfb07694e6109a54','size_bytes':26456}
MODELS=('v61','legacy','CONTROL_ORIGINAL_SOURCE_FIT','TREAT_ORIGINAL_PLUS_PUBLIC_SURFACE_AUX')
D='DIRECT_VISIBLE';E='EXTERNAL_OCCLUDED';S='SELF_OCCLUDED'
GROUPS=('WOOD_PRIMARY105','SQUARE118')
BIN_NAMES=('[0,1]','(1,3]','(3,6]','(6,12]','(12,INF)')

def safe(value,exists=True):
    s=str(value)
    if any(t in s.casefold()for t in('v112','cache','.pyc')):raise ValueError('Forbidden path before FS')
    p=Path(s)
    if not p.is_absolute()or'..'in p.parts:raise ValueError('Absolute closed path')
    for a in[*reversed(p.parents),p]:
        try:mode=os.lstat(a).st_mode
        except FileNotFoundError:
            if exists:raise
            continue
        if stat.S_ISLNK(mode):raise ValueError('Symlink artifact/ancestor')
    return p

def checked(binding,path):
    b=safe(path).read_bytes()
    if hashlib.sha256(b).hexdigest()!=binding['sha256']or len(b)!=binding['size_bytes']:raise ValueError('Exact admitted saved bytes required')
    return b

def number(value):
    try:v=float(value)
    except(TypeError,ValueError):return None
    return v if math.isfinite(v)else None

def primary(row):return row['paired_same_query_RGB_D_vs_E']=='True'and row['human_state']in(D,E)

def near_d(row):
    e=number(row['initial_reference_error_px'])
    return row['human_state']==D and row['original_query_valid']=='True'and e is not None and 0<=e<=3

def scaled(row):
    e=number(row['initial_reference_error_px']);L=number(row['roi_side'])
    return None if e is None or e<0 or L is None or L<=0 else e*96/L

def bucket(v):
    if v is None:return'MISSING_OR_INVALID'
    for edge,name in zip((1,3,6,12,float('inf')),BIN_NAMES):
        if v<=edge:return name
    raise AssertionError(v)

def stats(values):
    x=sorted(v for v in values if v is not None)
    def q(p):
        if not x:return None
        z=(len(x)-1)*p;i=int(math.floor(z));j=int(math.ceil(z));return x[i]+(x[j]-x[i])*(z-i)
    return{'N_known':len(x),'minimum':x[0]if x else None,'mean':sum(x)/len(x)if x else None,'p10':q(.1),'p25':q(.25),'p50':q(.5),'p75':q(.75),'p90':q(.9),'p95':q(.95),'maximum':x[-1]if x else None}

def analyze(rows,cases):
    if len(rows)!=1784 or[int(r['global_index'])for r in rows]!=list(range(1784)):raise ValueError('Original1784 order')
    pp=[r for r in rows if primary(r)]
    if(len(pp),sum(r['human_state']==D for r in pp),sum(r['human_state']==E for r in pp))!=(1237,1147,90):raise ValueError('Frozen primary')
    near=[r for r in rows if near_d(r)];manual=[r for r in near if r['annotation_coordinate_source']=='manual_click'];manual_p=[r for r in manual if primary(r)]
    cohorts={'ALL1784':rows,'PRIMARY1237':pp,'HUMAN_D_ALL':[r for r in rows if r['human_state']==D],
             'HUMAN_D_LE3_VALID_ALL_AUTHORITY':near,'MANUAL_CLICK_HUMAN_D_LE3_VALID_ALL1784':manual,
             'MANUAL_CLICK_HUMAN_D_LE3_VALID_PRIMARY':manual_p,'NONMANUAL_HUMAN_D_LE3_VALID':[r for r in near if r['annotation_coordinate_source']!='manual_click']}
    counts=[]
    for scope,rr in(('ALL1784',manual),('PRIMARY1237',manual_p)):
        strata=[('ALL','ALL',rr)]+[(g,'ALL',[r for r in rr if r['group']==g])for g in GROUPS]+[('ALL',k,[r for r in rr if int(r['native_corner'])==k])for k in range(8)]
        for group,native,qq in strata:
            for model in MODELS:
                cc=Counter(r[model+'_D_E_U']for r in qq)
                counts.append({'scope':scope,'material_group':group,'native_corner':native,'model':model,'N_human_D':len(qq),'D':cc['D'],'E_false_absent':cc['E'],'U_or_unresolved':cc['U'],'D_recall':cc['D']/len(qq)if qq else None,'E_false_absent_fraction':cc['E']/len(qq)if qq else None})
    distributions=[];bins=[]
    for name,rr in cohorts.items():
        for measure,vals in(('original_reference_error_px',[number(r['initial_reference_error_px'])for r in rr]),('original_L_pixels',[number(r['roi_side'])for r in rr]),('reference_error_RAW96_pixels',[scaled(r)for r in rr])):
            distributions.append(dict(cohort=name,measure=measure,N_rows=len(rr),N_missing_or_invalid=sum(v is None for v in vals),**stats(vals)))
        for name_bin in(*BIN_NAMES,'MISSING_OR_INVALID'):
            qq=[r for r in rr if bucket(scaled(r))==name_bin]
            for model in MODELS:
                cc=Counter(r[model+'_D_E_U']for r in qq)
                bins.append({'cohort':name,'RAW96_distance_bin':name_bin,'model':model,'N':len(qq),'human_D':sum(r['human_state']==D for r in qq),'human_E':sum(r['human_state']==E for r in qq),'human_SELF':sum(r['human_state']==S for r in qq),'pred_D':cc['D'],'pred_E':cc['E'],'pred_U_or_unresolved':cc['U'],'false_E_on_human_D':sum(r['human_state']==D and r[model+'_D_E_U']=='E'for r in qq)})
    authority=[]
    for source in sorted({r['annotation_coordinate_source']for r in near}):
        for scope,rr in(('ALL1784',near),('PRIMARY1237',[r for r in near if primary(r)])):
            qq=[r for r in rr if r['annotation_coordinate_source']==source];authority.append({'scope':scope,'annotation_coordinate_source':source,'human_D_LE3_originalvalid_N':len(qq)})
    case_by_i={c['global_index']:c for c in cases['cases']};examples=[]
    for i in(684,1680):
        r=rows[i];c=case_by_i[i];err=number(r['initial_reference_error_px']);L=number(r['roi_side'])
        if c['row_id']!=r['row_id']or c['L_unchanged']!=L or c['initial_reference_error_px']!=err:raise ValueError('Already selected gallery facts exact row/L/error join')
        examples.append({'global_index':i,'row_id':r['row_id'],'material_group':r['group'],'native_corner':r['native_corner'],'human_state':r['human_state'],'coordinate_authority':r['annotation_coordinate_source'],'original_reference_error_px':err,'original_L_pixels':L,'RAW96_pixels_per_original_pixel':96/L,'reference_error_RAW96_pixels':scaled(r),'RAW96_distance_bin':bucket(scaled(r)),'v61':r['v61_D_E_U'],'CONTROL':r[MODELS[2]+'_D_E_U'],'TREAT':r[MODELS[3]+'_D_E_U'],'new_crop_or_case_selection':False})
    manual_bins={b:sum(bucket(scaled(r))==b for r in manual)for b in BIN_NAMES}
    return{'manual_counts':counts,'distributions':distributions,'bins':bins,'authority':authority,'examples':examples,
           'summary':{'completed':True,'PASS':True,'all1784':1784,'primary_N':1237,'near_human_D_originalvalid_any_authority':len(near),'near_authority_counts':dict(Counter(r['annotation_coordinate_source']for r in near)),'manual_near_all1784':len(manual),'manual_near_primary':len(manual_p),'manual_all_equal_primary_row_ids':{r['row_id']for r in manual}=={r['row_id']for r in manual_p},'manual_model_D_E_U':{m:{s:sum(r[m+'_D_E_U']==s for r in manual)for s in('D','E','U')}for m in MODELS},'RAW96_distance_formula':'frozen_original_reference_error_px * 96 / frozen_original_L_pixels','RAW96_bins_closed_right':list(BIN_NAMES),'manual_RAW96_distance_bins':manual_bins,'manual_RAW96_distance_GT3':sum(scaled(r)is not None and scaled(r)>3 for r in manual),'manual_RAW96_distance_GT6':sum(scaled(r)is not None and scaled(r)>6 for r in manual),'manual_RAW96_distance_GT12':sum(scaled(r)is not None and scaled(r)>12 for r in manual),'hypotheses_not_proof':['Different L geometrically changes distance in RAW96. This does not prove small L causes false E.','Manual visible reference with original error<=3 does not certify visible target at unchanged q pixel; no new mask/coordinategold.','High public conditional AUX results do not prove current native generalization.','Legacy U/unresolved can include its out-of-task geometry/abstention namespace; not necessarily original-input U.'],'new_models_crops_training_threshold_or_recipe_selection':0,'RGB_or_model_or_checkpoint_access':0}}

def csv_write(p,rows):
    with p.open('x',newline='',encoding='utf-8')as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--ledger',default=LEDGER['path']);ap.add_argument('--case-facts',default=CASES['path']);ap.add_argument('--output-dir',required=True);a=ap.parse_args()
    rows=list(csv.DictReader(io.StringIO(checked(LEDGER,a.ledger).decode('utf-8-sig'))));cases=json.loads(checked(CASES,a.case_facts));result=analyze(rows,cases)
    out=safe(a.output_dir,False)
    if out.exists():raise ValueError('Fresh result directory')
    out.mkdir()
    files={'MANUAL_NEAR_MODEL_COUNTS.csv':result['manual_counts'],'ERROR_SCALE_DISTRIBUTIONS.csv':result['distributions'],'ERROR_RAW96_FIXED_BINS.csv':result['bins'],'ANNOTATION_AUTHORITY_LE3_HUMAN_D_COUNTS.csv':result['authority'],'GALLERY_ERROR_SCALE_EXAMPLES.csv':result['examples']}
    for n,rr in files.items():csv_write(out/n,rr)
    s=result['summary'];s['inputs']={'ledger':dict(LEDGER,path=a.ledger),'gallery_case_facts':dict(CASES,path=a.case_facts)}
    with(out/'REVIEW.json').open('x',encoding='utf-8')as f:json.dump(s,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    text='''# 동결된 current 표의 수동 좌표 근접 사례\n\n원영상 오차 ≤3px인 human D·원입력 valid 사례는 461개다. 좌표 권위는 manual_click 376개, pnp_projected 52개, unknown 33개로 나뉜다. 기존 LE3 집계 전체를 수동 클릭 정답으로 부를 수 없다. 수동 클릭 376개는 모두 primary1237에도 포함되어 두 범위의 이번 분모와 응답 수가 같다.\n\n수동 클릭 376개에서 D/E/U 응답은 v61 85/291/0, legacy 205/108/63, CONTROL 316/60/0, TREAT 308/68/0이다. 이 D-only 부분집합의 E는 기존 human D에 대한 false-absent 응답이다. Legacy의 U/미해결에는 기존 out-of-task·abstention 구분이 포함될 수 있어 원입력 불가와 동일하게 해석하지 않는다.\n\n원영상 오차를 변경하지 않고 `오차×96/L`로 RAW96 좌표 거리만 계산했다. 고정 구간 [0,1], (1,3], (3,6], (6,12], (12,∞)의 수동 클릭 사례 수는 35/181/134/23/3이다. 따라서 원영상에서 ≤3px인 376개 중 160개는 RAW96에서 3px보다 멀다. 이것은 분포 설명용 구간이며 분류 문턱이나 새 학습 조건이 아니다.\n\n이미 선정된 사례 0684는 L=17.795578, 원오차=2.334941이므로 RAW96 거리=12.596으로 확대된다. 사례 1680은 L=81.424974, 원오차=1.964749로 RAW96 거리=2.316이다. 같은 원영상 ≤3px 기준이 crop에서 같은 근접도를 뜻하지 않는다는 기하학적 차이다. 두 사례는 모두 v61 D, CONTROL E, TREAT E이며, 이 두 사례만으로 작은 L을 오판 원인으로 확정할 수 없다.\n\n수동 클릭의 visible 기준점이 q에 가깝다는 사실은 변경하지 않은 q 픽셀 자체의 target-presence 정답을 인증하지 않는다. 새 mask·native gold를 만들거나 q를 기준점으로 이동하지 않았다. 높은 공개 conditional AUX 점수도 current native 일반화의 증거로 대체할 수 없다. 반복 노출된 current·기존 CAL 한계는 그대로다.\n\n새 crop·모델 실행·학습·문턱 또는 데이터/recipe 선택은 하지 않았다. 모든 수치의 입력은 동결된 current ledger와 이미 선정된 gallery CASE_FACTS 두 파일뿐이다.\n\n- [수동 근접 응답 수](MANUAL_NEAR_MODEL_COUNTS.csv)\n- [원오차/L/RAW96 거리 분포](ERROR_SCALE_DISTRIBUTIONS.csv)\n- [고정 거리 구간](ERROR_RAW96_FIXED_BINS.csv)\n- [좌표 권위 구분](ANNOTATION_AUTHORITY_LE3_HUMAN_D_COUNTS.csv)\n- [이미 선정된 두 사례](GALLERY_ERROR_SCALE_EXAMPLES.csv)\n'''
    # The exact example values are emitted in CSV; prose is rounded for reading.
    (out/'KO.md').write_text(text,encoding='utf-8')
    outputs={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}for p in out.iterdir()}
    (out/'OUTPUT_BINDINGS.json').write_text(json.dumps({'completed':True,'PASS':True,'files':outputs,'no_pixel_or_model_access':True},sort_keys=True,indent=2)+'\n')
    print(json.dumps({'completed':True,'PASS':True,'output_dir':str(out),'manual_counts':s['manual_model_D_E_U'],'manual_RAW96_bins':s['manual_RAW96_distance_bins']}))

if __name__=='__main__':main()
