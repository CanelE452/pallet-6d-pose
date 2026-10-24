"""Independent saved nominal metadata verification; actual reads need root entry."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,math,os,stat,traceback
from independent_nominal_geometry import project_instance,selected_query_diagnostic,invented_checks

H=Path(__file__).absolute().parent;V=H.parent;P=V.parent
KERNEL=H/'independent_nominal_geometry.py';PROTOCOL=H/'PROTOCOL_BEFORE_ACTUAL_NOMINAL_REVIEW.json';CHECKS=H/'GENERATED_INVENTED_CHECKS.json'
SOURCES=('ALL185_FRAME_INSTANCE_ASSOCIATIONS.jsonl','ALL1480_QUERY_INSTANCE_DIAGNOSTICS.jsonl','ALL_GT_OBSERVATIONS_IN185_SOURCE_FRAMES.jsonl')
OUTPUTS={'instances':'ALL258_OBJ1_NOMINAL_INSTANCES.jsonl','references':'ALL2064_OBJ1_NOMINAL_NATIVE_REFERENCES.jsonl','queries':'ALL1480_ORIGINAL_QUERY_NOMINAL_CORRESPONDENCE.jsonl'}
SCENES={'000000','000002','000005','000006'}

def require(value,message):
    if not value:raise ValueError(message)
def safe(path):
    p=Path(path);require(p.is_absolute() and P in p.parents and '..' not in p.parts and all('v112' not in z.lower() for z in p.parts),'Forbidden path precheck')
    require(p.suffix in ('.json','.jsonl','.py'),'Metadata/source only')
    return p
def open_safe(path):
    p=safe(path);parts=p.relative_to(P).parts;fd=os.open(P,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            new=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=new
        actual=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
    finally:os.close(fd)
    try:require(stat.S_ISREG(os.fstat(actual).st_mode),'Regular metadata file only')
    except BaseException:os.close(actual);raise
    return os.fdopen(actual,'rb')
def bind(path):
    h=hashlib.sha256()
    with open_safe(path) as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return dict(path=str(safe(path)),sha256=h.hexdigest())
def checked(b):
    require(bind(b['path'])==b,'Frozen metadata binding changed');return Path(b['path'])
def read(path):
    with open_safe(path) as f:return json.load(f)
def lines(path):
    with open_safe(path) as f:return [json.loads(line) for line in f if line.strip()]
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def compare(expected,actual,path='root',errors=None,numeric=True):
    require(type(expected) is type(actual),'Type mismatch '+path)
    if isinstance(expected,dict):
        require(set(expected)==set(actual),'Field mismatch '+path)
        for key,value in expected.items():
            if key=='geometry_error' and value is not None:
                require(isinstance(actual[key],str) and bool(actual[key]),'Invalid geometry explanation retained')
            else:compare(value,actual[key],path+'/'+key,errors,numeric)
    elif isinstance(expected,list):
        require(len(expected)==len(actual),'Length mismatch '+path)
        for index,(a,b) in enumerate(zip(expected,actual)):compare(a,b,path+'/'+str(index),errors,numeric)
    elif isinstance(expected,float):
        require(math.isfinite(expected) and math.isfinite(actual),'Nonfinite numerical output '+path)
        normalized=any(s in path for s in ('normal_face_scores','front_runner_margin','R_orthogonality','R_determinant','over_L'))
        tolerance=(1e-10 if normalized else 1e-8) if numeric else 0.
        error=abs(expected-actual)
        if errors is not None:
            category='normalized' if normalized else 'coordinate'
            if error>errors[category]['absolute']:errors[category]=dict(absolute=error,path=path)
        require(error<=tolerance,'Independent numerical bound exceeded '+path)
    else:require(expected==actual,'Value mismatch '+path)

def self_tests():
    result=invented_checks();tests=list(result['checks'])
    try:compare([1.,2.],[1.,2.1])
    except ValueError:tests.append('Injected_projection_or_crop_drift_rejected')
    else:raise AssertionError('Numerical drift accepted')
    try:compare(dict(inputU=194),dict(inputU=193))
    except ValueError:tests.append('Injected_population_or_U_count_rejected')
    else:raise AssertionError('Wrong denominator accepted')
    try:compare(dict(original_query_xy=[1.,2.]),dict(original_query_xy=[1.,2.000000001]),numeric=False)
    except ValueError:tests.append('Original_q_L_metadata_compared_exact_not_numerical_tolerance')
    else:raise AssertionError('Original q drift accepted')
    return dict(completed=True,PASS=True,invented_only=True,count=len(tests),checks=tests,actual_source_or_nominal_result_body_reads=0)

def run(entry_path):
    e=read(entry_path)
    require(e.get('approved_independent_nominal_review') is True and e.get('source_TRAIN_only') is True,'Separate root actual entry required')
    require(e['code']==bind(__file__) and e['kernel']==bind(KERNEL) and e['protocol']==bind(PROTOCOL),'Frozen independent code')
    p=read(checked(e['protocol']));require(e['checks']==p['checks'] and read(checked(p['checks']))['PASS'],'Invented checks')
    for field in ('source_complete','nominal_complete'):require(e[field]==p[field],'Exact closed source/result binding')
    c=read(checked(e['source_complete']));a=read(checked(e['nominal_complete']))
    require(c['completed'] is True and c['PASS'] is True and a['completed'] is True and a['PASS'] is True,'Both source stages completed')
    require(a['code']==p['nominal_code'] and a['protocol']==p['nominal_protocol'] and a['C_complete']==e['source_complete'],'Nominal lineage closure')
    require(set(c['outputs'])==set(SOURCES) and set(a['outputs'])==set(OUTPUTS),'All sealed serialized metadata outputs')
    output=Path(e['output_directory']);require(output==H/'actual_review_v1','Fixed fresh independent output')
    output.mkdir(exist_ok=False);write(output/'STARTED.json',dict(completed=False,entry=bind(entry_path),automatic_retry=False))
    try:
        frames=lines(checked(p['original_frame_plan']));rows=lines(checked(p['original_row_plan']))
        cf=lines(checked(c['outputs'][SOURCES[0]]));cr=lines(checked(c['outputs'][SOURCES[1]]));allgt=lines(checked(c['outputs'][SOURCES[2]]))
        saved={k:lines(checked(v)) for k,v in a['outputs'].items()}
        require(len(frames)==len(cf)==185 and len(rows)==len(cr)==len(saved['queries'])==1480,'Original185/1480 retained')
        require(len(saved['instances'])==258 and len(saved['references'])==2064,'All258/2064 retained')
        byfi={f['frame_index']:f for f in frames};associations={f['frame_index']:f for f in cf}
        require(len(byfi)==len(associations)==185 and set(byfi)==set(associations),'Same source frame IDs')
        for f,z in zip(frames,cf):
            require(f['split']=='SOURCE_DEVELOPMENT' and f['scene'] in SCENES,'Four TRAIN scenes only')
            compare(f,{k:z[k] for k in f},'original frame',numeric=False)
        for g in allgt:
            f=byfi[g['frame_index']]
            require(g['scene']==f['scene'] and g['frame_key']==f['frame_key'],'Every GT record source-frame provenance')
        instances=[project_instance(g,byfi[g['frame_index']]['raw_hw']) for g in allgt if g['obj_id']==1]
        require(len(instances)==258,'Every obj1 candidate included')
        refs=[r for instance in instances for r in instance['references']]
        lookup={(g['frame_index'],g['gt_instance_index']):g for g in instances}
        require(len(lookup)==258 and len(refs)==2064,'Unique source instance lookup and all8 refs')
        joined=[]
        for row,z in zip(rows,cr):
            compare(row,{k:z[k] for k in row},'original query',numeric=False)
            require(row['split']=='SOURCE_DEVELOPMENT' and 0<=row['native_slot']<8 and row['original_global_index']==row['original_frame_index']*8+row['native_slot'],'Original native identity')
            association=associations[row['original_frame_index']]['diagnostic_association']
            compare(association,z['selected_instance_diagnostic'],'source association',numeric=False)
            nominal=lookup.get((row['original_frame_index'],association['selected_gt_instance_index'])) if association['selected_obj_id']==1 else None
            joined.append(selected_query_diagnostic(row,association,nominal,byfi[row['original_frame_index']]['raw_hw']))
        errors=dict(coordinate=dict(absolute=0.,path=None),normalized=dict(absolute=0.,path=None))
        for name,expected in (('instances',instances),('references',refs),('queries',joined)):
            compare(expected,saved[name],name,errors)
        for expected,z in zip(instances,saved['instances']):compare(expected['provided_values'],z['provided_values'],'literal provided K/R/t',numeric=False)
        for row,z in zip(rows,saved['queries']):compare(row,{k:z[k] for k in row},'preserved final input',numeric=False)
        counts=dict(frames=185,original_rows=1480,original_inputU=sum(not r['original_query_available'] for r in joined),obj1_instances=258,nominal_reference_rows=2064,
            nominal_instance_status=dict(Counter(r['status'] for r in instances)),query_lookup_status=dict(Counter(r['nominal_correspondence']['status'] for r in joined)))
        require(counts['original_inputU']==194 and Counter(r['native_slot'] for r in joined)==Counter({k:185 for k in range(8)}),'Original U/native denominators')
        compare(counts,a['counts'],'Nominal serialized counts',numeric=False)
        eligible=[r for r in joined if r['original_query_available'] and r['nominal_correspondence']['status']=='WEAK_INSTANCE_NOMINAL_REFERENCE'
                  and r['nominal_correspondence']['nominal_discrepancy_and_crop_diagnostic']['valid_nominal_discrepancy']]
        coverage=dict(original_available_weakobj1_valid_nominal=len(eligible),inside_original_crop_extent=sum(r['nominal_correspondence']['nominal_discrepancy_and_crop_diagnostic']['inside_outer_crop_extent'] for r in eligible))
        coverage['outside_original_crop_extent']=len(eligible)-coverage['inside_original_crop_extent']
        input_bindings=[e['source_complete'],e['nominal_complete'],p['original_frame_plan'],p['original_row_plan'],*c['outputs'].values(),*a['outputs'].values()]
        for binding in input_bindings:checked(binding)
        review=dict(completed=True,PASS=True,entry=bind(entry_path),code=bind(__file__),kernel=bind(KERNEL),protocol=bind(PROTOCOL),source_complete=e['source_complete'],nominal_complete=e['nominal_complete'],counts=counts,
            independently_verified_all2064refs_all1480selectedlookups=True,independent_native_order='Literal signs; analytic vertical normals; rear opposite-axis topology; no imported A or V74 function',
            maximum_numerical_differences=errors,fixed_bounds=dict(coordinate_absolute_mm_or_px=1e-8,normalized_absolute=1e-10),original_q_L_native_flags_inputU_exact=True,
            nominal_crop_coverage_diagnostic=coverage,all_new_semantic_states_UNKNOWN_and_native_quality_minus1=True,new_training_admission=False,
            raw_RGB_depth_mask_mesh_NPZ_model_or_evaluation_reads=0,upstream_GT_camera_asset_body_reads=0,producer_imports=0,automatic_retry=False,
            geometry_error_exception_text_not_compared=True,scope='Nominal arithmetic/source weak instance lookup only; nominal coverage/discrepancy is not physical coordinate error, new semantic gold, or model success.',bindings_rechecked=input_bindings)
        write(output/'REVIEW.json',review);print(json.dumps(dict(completed=True,PASS=True,counts=counts,max_errors=errors,nominal_coverage=coverage,review=bind(output/'REVIEW.json'))))
    except Exception:
        write(output/'FAILED_REVIEW_PRESERVED.json',dict(completed=False,automatic_retry=False,entry=bind(entry_path),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--self-test',action='store_true');g.add_argument('--entry',type=Path);p.add_argument('--report',type=Path);a=p.parse_args()
    if a.self_test:
        result=self_tests()
        if a.report:write(a.report,result)
        print(json.dumps(result))
    else:run(a.entry)
