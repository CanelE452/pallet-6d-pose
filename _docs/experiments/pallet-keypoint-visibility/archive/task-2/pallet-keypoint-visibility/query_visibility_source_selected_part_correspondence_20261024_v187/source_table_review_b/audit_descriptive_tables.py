"""Independent scalar census of sealed V187 metadata; no producer imports."""
from pathlib import Path
from collections import Counter
import csv,hashlib,json,math,os,stat

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
SUMMARY=V/'descriptive_source_tables_v1/SUMMARY.json'
SCENE_CSV=V/'descriptive_source_tables_v1/PER_SCENE_INSTANCE_MATCH.csv'
NATIVE_CSV=V/'descriptive_source_tables_v1/PER_NATIVE_ORIGINAL_CROP_COVERAGE.csv'
C=V/'selected_instance_metadata_diagnostic_v2_nested_R/COMPLETE.json'
A=V/'nominal_source_native_projection_v1/COMPLETE.json'
NOMINAL_REVIEW=V/'independent_nominal_review_b/actual_review_v1/REVIEW.json'
SCENES=('000000','000002','000005','000006')

def need(ok,message):
    if not ok:raise ValueError(message)
def open_safe(path):
    p=Path(path);need(p.is_absolute() and P in p.parents and '..' not in p.parts and all('v112' not in s.lower() for s in p.parts),'Forbidden name precheck')
    need(p.suffix in ('.json','.jsonl','.csv','.py'),'Only fixed metadata/source')
    fd=os.open(P,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.relative_to(P).parts[:-1]:
            new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=new
        actual=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    try:need(stat.S_ISREG(os.fstat(actual).st_mode),'Regular metadata')
    except BaseException:os.close(actual);raise
    return os.fdopen(actual,'rb')
def bind(path):
    h=hashlib.sha256()
    with open_safe(path) as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return dict(path=str(path),sha256=h.hexdigest())
def checked(b):
    need(bind(b['path'])==b,'Closed metadata SHA mismatch');return Path(b['path'])
def read(path):
    with open_safe(path) as f:return json.load(f)
def rows(path):
    with open_safe(path) as f:return [json.loads(line) for line in f if line.strip()]
def csvrows(path):
    with open_safe(path) as f:return list(csv.DictReader(line.decode('utf-8') for line in f))
def write(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def quantile_record(values):
    ordered=sorted(values);n=len(ordered);out={'n':n}
    for name,q in (('min',0.),('p25',.25),('median',.5),('p75',.75),('max',1.)):
        if not n:out[name]=None;continue
        index=q*(n-1);lo=math.floor(index);hi=math.ceil(index)
        out[name]=ordered[lo]*(1-(index-lo))+ordered[hi]*(index-lo)
    return out

def independent_query_census(rr):
    total=available=eligible=inside=sample=domain=0;all_status=Counter();available_status=Counter();dist=[];scaled=[]
    for r in rr:
        total+=1;status=r['nominal_correspondence']['status'];all_status[status]+=1
        if not r['original_query_available']:continue
        available+=1;available_status[status]+=1
        if status!='WEAK_INSTANCE_NOMINAL_REFERENCE':continue
        q,L=r['original_query_xy'],r['original_L'];reference=r['nominal_correspondence']['nominal_reference']
        valid=(isinstance(q,list) and len(q)==2 and all(type(x) in (int,float) and math.isfinite(x) for x in q)
               and q!=[-1.,-1.] and type(L) in (int,float) and math.isfinite(L) and L>0
               and reference is not None and isinstance(reference['nominal_reference_xy'],list)
               and all(math.isfinite(x) for x in reference['nominal_reference_xy']))
        saved=r['nominal_correspondence']['nominal_discrepancy_and_crop_diagnostic']
        need(saved['valid_nominal_discrepancy']==valid,'Saved numeric validity agrees independent finite/sentinel rule')
        if not valid:continue
        eligible+=1;x,y=reference['nominal_reference_xy'];delta=(x-q[0],y-q[1]);t=(96.*delta[0]/L+47.5,96.*delta[1]/L+47.5)
        inside+=int(all(-.5<=v<95.5 for v in t));sample+=int(all(0<=v<=95 for v in t))
        h,w=r['raw_hw'] if 'raw_hw' in r else reference['_independent_raw_hw']
        domain+=int(-.5<=x<w-.5 and -.5<=y<h-.5)
        distance=math.sqrt(delta[0]*delta[0]+delta[1]*delta[1]);dist.append(distance);scaled.append(distance/L)
    return dict(total=total,input_available=available,inputU=total-available,all_row_lookup_status=dict(all_status),available_lookup_status=dict(available_status),
        eligible_available_weak_nominal=eligible,eligible_inside_original_crop_extent=inside,eligible_outside_original_crop_extent=eligible-inside,
        eligible_inside_sample_center_rectangle=sample,eligible_reference_in_original_sampler_domain=domain,
        nominal_distance_px=quantile_record(dist),nominal_distance_over_L=quantile_record(scaled))

def compare(a,b,path='root',counter=None):
    need(type(a) is type(b),'Type mismatch '+path)
    if isinstance(a,dict):
        need(set(a)==set(b),'Fields mismatch '+path)
        for k,v in a.items():compare(v,b[k],path+'/'+str(k),counter)
    elif isinstance(a,list):
        need(len(a)==len(b),'List size mismatch '+path)
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i),counter)
    elif isinstance(a,float):
        tolerance=1e-10 if 'over_L' in path else 1e-8
        need(math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tolerance,'Float mismatch '+path)
    else:need(a==b,'Scalar mismatch '+path)
    if counter is not None and not isinstance(a,(dict,list)):counter['scalar_fields']+=1

def main():
    need(bind(C)['sha256']=='8541767371df1d5976ce9b44b8491fa23ee5cec37d01bd76fc16d975ba4f1deb','Sealed source C')
    need(bind(A)['sha256']=='43005b0ea5e8e674a5345a05fe4054eb3a52db5e815f94f08ffc41e7a5243568','Sealed nominal A')
    need(bind(NOMINAL_REVIEW)['sha256']=='76549fdb5d1a25f97e84caf93063872d268f1a4b90b6396f3238dabb8c8c1261','Independent nominal actual PASS')
    c,a,report=read(C),read(A),read(SUMMARY);need(read(NOMINAL_REVIEW)['PASS'],'Prior independent actual review')
    frames=rows(checked(c['outputs']['ALL185_FRAME_INSTANCE_ASSOCIATIONS.jsonl']));queries=rows(checked(a['outputs']['queries']))
    instances=rows(checked(a['outputs']['instances']));refs=rows(checked(a['outputs']['references']))
    need((len(frames),len(queries),len(instances),len(refs))==(185,1480,258,2064),'Full source denominators')
    need(all(r['semantic_state']=='UNKNOWN' and r['native_visibility_target']==r['point_quality_target']==-1 and r['training_admission'] is False for r in refs),'Nominal states remainUNKNOWN/newtargets−1')
    byfi={f['frame_index']:f for f in frames};need(set(f['scene'] for f in frames)==set(SCENES),'All4 TRAIN scenes')
    for r in queries:
        # Add a private local convenience key only on a copied reference for sampling bounds.
        ref=r['nominal_correspondence']['nominal_reference']
        if ref is not None:ref['_independent_raw_hw']=byfi[r['original_frame_index']]['raw_hw']
    expected_all=independent_query_census(queries)
    expected_native={str(k):independent_query_census([r for r in queries if r['native_slot']==k]) for k in range(8)}
    expected_scene=[]
    for s in SCENES:
        ff=[f for f in frames if f['scene']==s];objects=Counter(str(f['diagnostic_association']['selected_obj_id']) for f in ff)
        expected_scene.append(dict(scene=s,frames=len(ff),selected_obj1=objects['1'],selected_other=sum(v for k,v in objects.items() if k not in ('1','None')),
            selected_UNKNOWN=objects['None'],selected_obj_ids=dict(objects)))
    compared=Counter();compare(expected_all,report['all_queries'],'all_queries',compared);compare(expected_native,report['per_native'],'per_native',compared);compare(expected_scene,report['per_scene'],'per_scene',compared)
    compare(dict(Counter(f['diagnostic_association']['status'] for f in frames)),report['frame_status'],'frame_status',compared)
    compare(dict(Counter(str(f['diagnostic_association']['selected_obj_id']) for f in frames)),report['selected_obj_ids'],'selected_obj_ids',compared)
    native_csv=csvrows(NATIVE_CSV);scene_csv=csvrows(SCENE_CSV)
    need(len(native_csv)==8 and len(scene_csv)==4,'CSV allnative/allscene rows')
    for row in native_csv:
        native=row['native_slot'];need(native in expected_native,'CSV nativekey')
        for key,text in row.items():
            if key!='native_slot':need(int(text)==expected_native[native][key],'CSV native integer disagreement')
    for row,z in zip(scene_csv,expected_scene):
        need(row['scene']==z['scene'],'CSV originalscene order')
        for key,text in row.items():
            if key!='scene':need(int(text)==z[key],'CSV scene integer disagreement')
    need((expected_all['total'],expected_all['inputU'],expected_all['eligible_available_weak_nominal'],expected_all['eligible_inside_original_crop_extent'],expected_all['eligible_outside_original_crop_extent'])==(1480,194,924,650,274),'Overall coverage census')
    receipt=dict(completed=True,PASS=True,code=bind(Path(__file__).absolute()),descriptive_summary=bind(SUMMARY),descriptive_CSVs=[bind(SCENE_CSV),bind(NATIVE_CSV)],
        source_complete=bind(C),nominal_complete=bind(A),independent_nominal_review=bind(NOMINAL_REVIEW),
        all185_1480_U194_258_2064_denominators_preserved=True,per_native_summary_groups_verified=8,per_scene_summary_groups_verified=4,native_CSV_rows_verified=8,scene_CSV_rows_verified=4,
        summary_scalar_fields_compared=compared['scalar_fields'],all_queries_independent=expected_all,per_native_independent=expected_native,per_scene_independent=expected_scene,
        states_semanticUNKNOWN_newnative_quality_minus1_trainingfalse=True,producer_helper_imports=0,source_RGB_depth_mask_mesh_NPZ_model_or_evaluation_body_reads=0,
        nominal_outside274_is_not_physical_coordinate_error=True,no_new_labels_training_model_selection_or_definition_changes=True,
        scope='Closed source metadata descriptive counts and fixed quantiles only. The chart/summary are not identity accuracy, native gold, physical registration or model improvement.')
    write(H/'REVIEW.json',receipt);print(json.dumps(dict(completed=True,PASS=True,scalar_fields=compared['scalar_fields'],all_counts=[1480,194,924,650,274],review=bind(H/'REVIEW.json'))))

if __name__=='__main__':main()
