"""Invented BASE+V187..V197 -> V200 roundtrip; no actual phase/repo inputs."""
from pathlib import Path
import copy,hashlib,importlib.util,json,tempfile
H=Path(__file__).absolute().parent
spec=importlib.util.spec_from_file_location('invented_v200_delta_builder',H/'build_completed_phase_delta.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def b(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def w(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False)+'\n')
def raw(relative,sha='b'*64):return dict(status='PUBLISHED_RAW',published_relative=relative,published_sha256=sha,source_sha256=sha,size_bytes=1,gzipchunks=None)

def pure_checks():
    checks={};bb={'path':'/invented/PUBLICATION_MANIFEST.json','sha256':'a'*64};base={'entries':[raw('archive/BASE.json')]};records=[];refs=[]
    for i,(tag,phase) in enumerate(m.pure.ANCESTOR_PHASES):
        dr='publication/deltas/'+tag+'_20261024_v1/DELTA_MANIFEST.json';cr='MANIFEST_CHAIN_AFTER_'+tag+'.json';ir='EXPERIMENT_INDEX_THROUGH_'+tag+'_KO.md'
        d={'path':'/invented/'+dr,'sha256':hashlib.sha256(('delta'+str(i)).encode()).hexdigest()};c={'path':'/invented/'+cr,'sha256':hashlib.sha256(('chain'+str(i)).encode()).hexdigest()};ix={'path':'/invented/'+ir,'sha256':hashlib.sha256(('index'+str(i)).encode()).hexdigest()}
        manifest=dict(format='immutable-completed-phase-delta-v1',phase=phase,previous_base_manifest=bb,entries=[raw(ir,ix['sha256']),dict(status='PUBLISHED_GZIP_CHUNKS',published_relative='archive/'+tag+'.jsonl',gzip_descriptor_path='archive/'+tag+'.jsonl.parts.json',gzipchunks={'parts':[{'path':'archive/'+tag+'.jsonl.parts/p.gz'}]})])
        if records:manifest.update(previous_delta_manifest=records[-1]['delta_manifest'],previous_chain=records[-1]['chain'])
        refs.append(dict(path=dr,sha256=d['sha256']));chain=dict(format='immutable-publication-manifest-chain-v1',base=dict(path='PUBLICATION_MANIFEST.json',sha256=bb['sha256']),deltas=copy.deepcopy(refs),index=dict(path=ir,sha256=ix['sha256']))
        if records:chain['previous_chain']=dict(path='MANIFEST_CHAIN_AFTER_'+records[-1]['tag']+'.json',sha256=records[-1]['chain']['sha256'])
        records.append(dict(tag=tag,delta_manifest=d,chain=c,index=ix,manifest=manifest,chain_data=chain))
    entries=m.pure.validate_parent_closure(bb,base,records);checks['full_BASE_through_V199_positive_closure']=bool(entries)
    negatives={
      'missing_earlier_V187':lambda r:r.pop(0),
      'missing_latest_V199':lambda r:r.pop(),
      'reversed_order':lambda r:r.reverse(),
      'ancestor_base_SHA_drift':lambda r:r[-1]['manifest']['previous_base_manifest'].update(sha256='f'*64),
      'earlier_delta_SHA_drift_in_latest_chain':lambda r:r[-1]['chain_data']['deltas'][0].update(sha256='f'*64),
      'previous_chain_SHA_drift':lambda r:r[-1]['chain_data']['previous_chain'].update(sha256='f'*64),
      'latest_index_SHA_drift':lambda r:r[-1]['chain_data']['index'].update(sha256='0'*64),
      'explicit_previous_index_drift':lambda r:r[-1]['manifest'].update(previous_index={'path':'/wrong','sha256':'f'*64})}
    for name,mutate in negatives.items():
        r=copy.deepcopy(records);mutate(r)
        try:m.pure.validate_parent_closure(bb,base,r);checks[name+'_rejected']=False
        except (AssertionError,KeyError):checks[name+'_rejected']=True
    for name in ['archive/BASE.json','archive/V187.jsonl','archive/V187.jsonl.parts/p.gz','archive/V188.jsonl.parts.json','MANIFEST_CHAIN_AFTER_V187.json','publication/deltas/V188_20261024_v1/DELTA_MANIFEST.json']:
        try:m.pure.assert_new_paths(entries,[raw(name)]);checks['old_collision_'+name]=False
        except AssertionError:checks['old_collision_'+name]=True
    repository='/invented/repo';closing=[]
    for tag in m.pure.CLOSING_TAGS:
        plan=dict(authorized_main_push=True,repository=repository,branch='main',remote='https://github.com/CanelE452/pallet-6d-pose.git',batches=[dict(files=[dict(path=m.REPO_PREFIX+'closing/'+tag+'.json',sha256='a'*64,size_bytes=3)])])
        closing.append(dict(tag=tag,plan_data=plan))
    cp=m.pure.closing_protected_entries(closing,repository);checks['all_fourteen_closing_plans_protected']=len(cp)==len(m.pure.CLOSING_TAGS)
    with_readme=copy.deepcopy(closing)
    with_readme[0]['plan_data']['batches'][0]['files'].append(dict(path='readme.md',sha256='c'*64,size_bytes=19))
    exceptions=[];protected=m.pure.closing_protected_entries(with_readme,repository,exceptions)
    checks['BASE_historical_README_accepted_reported_notprotected']=protected==cp and len(exceptions)==1 and exceptions[0]['tag']=='BASE' and exceptions[0]['path']=='readme.md' and exceptions[0]['sha256']=='c'*64 and exceptions[0]['size_bytes']==19
    for name,record_index,path in [('nonBASE_README',1,'readme.md'),('BASE_otheroutside',0,'OTHER.md'),('nonBASE_otheroutside',2,'OTHER.md')]:
        negative=copy.deepcopy(closing);negative[record_index]['plan_data']['batches'][0]['files'].append(dict(path=path,sha256='c'*64,size_bytes=19))
        try:m.pure.closing_protected_entries(negative,repository);checks[name+'_rejected']=False
        except AssertionError:checks[name+'_rejected']=True
    for name,field,value in [('README_badSHA','sha256','x'*64),('README_badsize','size_bytes',-1)]:
        negative=copy.deepcopy(with_readme);negative[0]['plan_data']['batches'][0]['files'][-1][field]=value
        try:m.pure.closing_protected_entries(negative,repository);checks[name+'_rejected']=False
        except AssertionError:checks[name+'_rejected']=True
    for label,negative in [('missing_closing',closing[:-1]),('reversed_closing',list(reversed(closing)))]:
        try:m.pure.closing_protected_entries(negative,repository);checks[label+'_rejected']=False
        except AssertionError:checks[label+'_rejected']=True
    try:m.pure.assert_new_paths(entries+cp,[raw('closing/V189.json')]);checks['old_closing_overwrite_rejected']=False
    except AssertionError:checks['old_closing_overwrite_rejected']=True
    assert all(checks.values()),checks
    return checks

if __name__=='__main__':
    x=pure_checks()
    out=dict(completed=True,PASS=all(x.values()),checks=x,count=len(x),invented_only=True,actual_phase_repo_hash_copy_Git=0)
    with(H/'INVENTED_PARENT_AND_CLOSING_ROUNDTRIP.json').open('x')as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(out))
