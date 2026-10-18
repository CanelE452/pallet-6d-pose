"""Independent P0 action/F0 reconstruction and read-only archived-model replay.

Only the comparison inside run() imports old production inputs. The independent
action/input/linear algebra implementation below does not call production helpers.
Real reference files are hashed outside inference and never JSON-decoded here.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math,os,resource,sys,time,traceback
from collections import Counter
from copy import deepcopy
from pathlib import Path
import numpy as np
from .prediction_guard import PredictionGuard

REUSED_ARMS=('D1_H1_MLP','D1_H1_PAIR_MLP','D1_H1_PAIR_ROLE_AFFINE')
SEEDS=(20261018,20261019,20261020)
FRAME_FIELDS=frozenset(('candidate_sources_before_dedup','candidates','cluster_id','covariance_summary','e0_regenerated','fallback','feature_names','id','object_scale_m','observation_coverage','partition','projection_rejections','proof','session','split','state_counts','transformation'))
ACTION_FIELDS=frozenset(('actual_pose','baseline_alternative_not_joint_certificate','candidate_id','eligible','feature_context_family','features','object_scale_m','source_aliases','source_baseline_aliases'))


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(path):
    with open(path)as f:return json.load(f)
def rows(path):
    opener=gzip.open if str(path).endswith('.gz')else open
    with opener(path,'rt')as f:
        for line in f:
            if line.strip():yield json.loads(line)
def write(path,value):
    with Path(path).open('x')as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')


def finite_pose(value):
    if not isinstance(value,dict)or not value.get('available'):return False
    try:
        r=np.asarray(value['R_physical'],float);t=np.asarray(value['centroid'],float)
        return r.shape==(3,3)and t.shape==(3,)and bool(np.isfinite(r).all()and np.isfinite(t).all())
    except(KeyError,TypeError,ValueError):return False
def numerical_pose(value):return tuple(np.asarray(value['R_physical'],float).flatten())+tuple(value['centroid'])if finite_pose(value)else()
def equivalent(a,b,scale):
    if not finite_pose(a)or not finite_pose(b):return False
    rmax=max(abs(x-y)for x,y in zip(numerical_pose(a)[:9],numerical_pose(b)[:9]))
    tmax=max(abs(x-y)for x,y in zip(a['centroid'],b['centroid']))
    return rmax<=1e-7 and tmax<=1e-7*max(scale,1e-9)
def aliases(c):
    values=set(c.get('source_aliases',()))|set(c.get('source_baseline_aliases',()))
    if c['candidate_id']in('N0','E0','E0_REGENERATED'):values.add(c['candidate_id'])
    if 'E0_REGENERATED'in values:values.add('E0')
    return values


def independent_actions(frame,pair=False):
    scale=float(frame.get('object_scale_m',1.));n0=frame.get('fallback',{});e0=frame.get('e0_regenerated',{})
    prepared=[]
    for original in frame['candidates']:
        p=original['actual_pose']
        if not finite_pose(p):continue
        n0_match=equivalent(p,n0,scale)
        if not(original['eligible']or n0_match):continue
        c=deepcopy(original);roles=aliases(c)
        if n0_match:roles.add('N0')
        c['source_baseline_aliases']=sorted(roles);prepared.append(c)
    if finite_pose(n0)and not any('N0'in aliases(c)for c in prepared):
        prepared.append(dict(candidate_id='N0',actual_pose=deepcopy(n0),eligible=True,features=[0.]*62,source_baseline_aliases=['N0'],feature_absence='N0 was not in original feature bank'))
    if finite_pose(e0)and not any(equivalent(c['actual_pose'],e0,scale)for c in prepared):
        prepared.append(dict(candidate_id='E0_REGENERATED',actual_pose=deepcopy(e0),eligible=True,features=[0.]*62,source_baseline_aliases=['E0'],feature_absence='E0 omitted from original feature bank'))
    for c in prepared:
        if equivalent(c['actual_pose'],e0,scale):c['source_baseline_aliases']=sorted(aliases(c)|{'E0'})
    prepared=sorted(prepared,key=lambda c:('N0'not in aliases(c),numerical_pose(c['actual_pose'])))
    groups=[]
    for candidate in prepared:
        matches=[i for i,c in enumerate(groups)if equivalent(c['actual_pose'],candidate['actual_pose'],scale)]
        if matches:
            match=groups[matches[0]];match['source_baseline_aliases']=sorted(aliases(match)|aliases(candidate))
            match['candidate_aliases']=sorted(set(match['candidate_aliases'])|{candidate['candidate_id']})
        else:
            candidate['candidate_aliases']=[candidate['candidate_id']];candidate['is_N0']='N0'in aliases(candidate);groups.append(candidate)
    if pair:groups=[c for c in groups if {'N0','E0'}&aliases(c)]
    for c in groups:c['action_role']='N0'if c['is_N0']else'E0'if'E0'in aliases(c)else'OTHER'
    return groups


def independent_inputs(frame,normalizer,pair=False):
    actions=independent_actions(frame,pair)
    if not actions:return np.zeros((0,188),np.float32),[]
    count=len(actions);matrix=np.zeros((count,62),np.float32)
    for i,c in enumerate(actions):
        for j in range(31):
            v=c['features'][j+31]
            if v>0 and normalizer['valid_train_counts'][j]>0:
                matrix[i,j]=(float(c['features'][j])-normalizer['mean'][j])/normalizer['scale'][j]
            matrix[i,j+31]=v
    out=np.empty((count,188),np.float32)
    out[:,:62]=matrix;out[:,62:124]=np.mean(matrix,axis=0);out[:,124:186]=np.max(matrix,axis=0)
    out[:,186]=np.log1p(count);out[:,187]=[float(c['is_N0'])for c in actions]
    return out,actions


def independent_choose(scores,actions):
    if not actions:return None
    if len(scores)!=len(actions)or not all(math.isfinite(float(v))for v in scores):raise ValueError('Nonfinite/misaligned action scores')
    best=0
    for i in range(1,len(actions)):
        if scores[i]>scores[best]:best=i
    return best


def independent_logits(x,state,architecture):
    import torch
    import torch.nn.functional as F
    x=torch.as_tensor(x,dtype=torch.float32)
    with torch.inference_mode():
        if architecture=='MLP':
            for prefix in('0','2','4'):
                x=F.linear(x,state[prefix+'.weight'],state[prefix+'.bias'])
                if prefix!='4':x=F.relu(x)
        elif architecture=='ROLE_AFFINE':
            shared=x[:,:187];other=F.linear(shared,state['other.weight'],state['other.bias']);n0=F.linear(shared,state['n0.weight'],state['n0.bias'])
            x=torch.where(x[:,-1:]>0.5,n0,other)
        else:raise ValueError(architecture)
    return x.flatten().numpy()


def _descriptor(path,role,shape=None):
    p=Path(path);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,role=role,shape=shape)


def run(archive,upstream,original_score,output):
    import torch
    from scripts.research.pallet_boundary_action_repair_v1.action_data import inputs as production_inputs
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    start=time.monotonic();cpu=time.process_time();archive=Path(archive).resolve();upstream=Path(upstream).resolve();score=Path(original_score).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False);checks=Counter();failures=[];inventory={};counts=Counter();source=Path(__file__)
    def check(name,condition,detail=None):
        checks[name]+=1
        if not bool(condition):failures.append(dict(check=name,detail=detail))
    def add(name,path,role,shape=None):
        inventory[name]=_descriptor(path,role,shape);return Path(path)
    try:
        # Real references are hash-only inventory, deliberately outside inference.
        ref=add('real_reference',score/'FIXED_REFERENCES.json','evaluation-only; bytes hashed, never decoded')
        refseal=add('real_reference_seal',score/'SCORE_SEAL.json','original evaluation binding')
        check('original_reference_seal',inventory['real_reference']['sha256']==read(refseal)['artifacts']['FIXED_REFERENCES.json'])
        trainmanifest=add('training_manifest',archive/'action_training_v1/TRAINING_MANIFEST.json','reused checkpoint inventory')
        normalizerpath=add('normalizer',archive/'d1_action_labels_v1/TRAIN_ONLY_NORMALIZER.json','frozen train-only scalar normalization',dict(mean=31,scale=31,valid_train_counts=31))
        normalizer=read(normalizerpath);labelsseal=add('synthetic_label_seal',archive/'d1_action_labels_v1/LABEL_SEAL.json','separate synthetic supervision binding')
        labelpath=add('synthetic_action_labels',archive/'d1_action_labels_v1/ACTION_TARGETS.jsonl.gz','training/validation supervision; not inference input',dict(frames=1003))
        target=add('synthetic_references',upstream/'synthetic_v1/SYNTH_GT_LABELS.jsonl.gz','original renderer supervision; not prediction input',dict(frames=1003))
        splitpath=add('split',upstream/'synthetic_v1/DATA_SPLIT_LOCK.json','fixed768/235 asset split',dict(train=768,validation=235))
        split=read(splitpath);expected_split={r['id']:r['split']for r in split['all_rows']}
        check('split_count',Counter(expected_split.values())==Counter(train=768,validation=235))
        check('split_asset_disjoint',not(set(split['train_assets'])&set(split['validation_assets'])))
        ls=read(labelsseal)
        for key,actual in [('labels_sha256',inventory['synthetic_action_labels']['sha256']),('normalizer_sha256',inventory['normalizer']['sha256']),('targets_sha256',inventory['synthetic_references']['sha256']),('split_lock_sha256',inventory['split']['sha256'])]:check('synthetic_label_binding/'+key,ls[key]==actual)
        frames={};seals={}
        for domain,n in(('real',238),('synthetic',1003)):
            p=add(domain+'_features',archive/f'd1_{domain}_candidates_v1/HEAD_FEATURES.jsonl.gz','prediction-only original action bank',dict(frames=n,candidate_features=62))
            s=add(domain+'_feature_seal',p.with_name('FEATURE_SEAL.json'),'original D1 feature and independently executed E0 binding')
            seals[domain]=read(s);check('D1_feature_sha/'+domain,seals[domain]['head_features_sha256']==inventory[domain+'_features']['sha256'])
            check('E0_independently_executed/'+domain,seals[domain]['e0_independently_executed']is True)
            data=list(rows(p));frames[domain]=data;check('frame_count/'+domain,len(data)==n and len({r['id']for r in data})==n)
            if domain=='synthetic':check('original_ID_split_join',{r['id']:r['split']for r in data}==expected_split)
            else:check('real_partition_counts',Counter(r['partition']for r in data)==Counter(TEST=49,REMAINDER=143,CAL=46))
        predpath=add('old_predictions',archive/'action_predictions_v1/PREDICTIONS.jsonl.gz','sealed old predictor outputs; no error labels',dict(frames=238,all_old_selectors=21,replayed_selectors=9))
        predseal=add('old_prediction_seal',predpath.with_name('PREDICTION_SEAL.json'),'archived no-reference inference binding')
        ps=read(predseal);check('archived_predictions_bound',ps['predictions_sha256']==inventory['old_predictions']['sha256'])
        check('archived_real_bank_bound',ps['features_sha256']==inventory['real_features']['sha256'])
        check('archived_training_bound',ps['training_manifest_sha256']==inventory['training_manifest']['sha256'])
        archived={r['id']:r for r in rows(predpath)};check('archived_prediction_IDs',set(archived)=={r['id']for r in frames['real']})
        models={}
        for record in read(trainmanifest)['runs']:
            if record['arm']not in REUSED_ARMS:continue
            name=record['arm']+'_s'+str(record['seed']);p=add('checkpoint/'+name,record['checkpoint'],'reused tiny selector checkpoint')
            check('checkpoint_SHA/'+name,sha(p)==record['checkpoint_sha256']==ps['checkpoints'][name]['checkpoint_sha256'])
            ck=torch.load(p,map_location='cpu',weights_only=False)
            check('checkpoint_normalizer/'+name,ck['normalizer']==normalizer)
            check('checkpoint_contract/'+name,ck['input_dim']==188 and ck['objective']=='H1'and ck['seed']in SEEDS)
            models[name]=ck
            inventory['checkpoint/'+name]['shape']={k:list(v.shape)for k,v in ck['model_state_dict'].items()}
        check('nine_checkpoints',len(models)==9)
        # Fresh process imported model libraries before guard, but loaded no real
        # reference objects. Only strict frame whitelist fields feed model inputs.
        denyroots=[score,archive/'evaluation_v1',output.parent/'evaluation_v1',output.parent/'b_reference_support_v1']
        guardspec=dict(allowed_files=[v['path']for k,v in inventory.items()if k not in('real_reference','real_reference_seal','synthetic_action_labels','synthetic_references','synthetic_label_seal')],allowed_read_roots=[Path(sys.prefix),source.parents[3],Path('/usr/lib')],output_root=output,denied_files=[ref],denied_roots=denyroots)
        input_digest=hashlib.sha256();pose_digest=hashlib.sha256();replays=Counter();maxdiff=0.
        with PredictionGuard(**guardspec)as guard:
            for domain,data in frames.items():
                for frame in data:
                    check('strict_frame_feature_whitelist',set(frame)<=FRAME_FIELDS,dict(id=frame['id'],extra=sorted(set(frame)-FRAME_FIELDS)))
                    for c in frame['candidates']:
                        check('strict_action_feature_whitelist',set(c)<=ACTION_FIELDS)
                        raw=np.asarray(c['features'],float);check('62_finite_binary_validity',raw.shape==(62,)and np.isfinite(raw).all()and np.isin(raw[31:],[0.,1.]).all())
                        check('explicit_eligibility',type(c['eligible'])is bool);counts[domain+'/archived_candidates']+=1;counts[domain+'/ineligible_candidates']+=int(not c['eligible'])
                    for pair in(False,True):
                        x,a=independent_inputs(frame,normalizer,pair);y,b=production_inputs(frame,normalizer,'H1',pair)
                        check('independent_F0_bitwise/'+domain+('/pair'if pair else'/full'),np.array_equal(x,y),frame['id'])
                        check('independent_actions_exact/'+domain,a==b,frame['id'])
                        input_digest.update(canonical([domain,frame['id'],pair]));input_digest.update(x.tobytes())
                        pose_digest.update(canonical([domain,frame['id'],pair,[dict(pose=c['actual_pose'],aliases=c['source_baseline_aliases'],eligible=c['eligible'])for c in a]]))
                        counts[domain+('/pair_actions'if pair else'/full_actions')]+=len(a)
                        counts[domain+('/pair_zero_actions'if pair else'/full_zero_actions')]+=int(not a)
                        counts[domain+('/pair_N0_absent'if pair else'/full_N0_absent')]+=int(not any('N0'in aliases(c)for c in a))
                        if not pair:
                            counts[domain+'/N0_E0_same_action']+=int(any({'N0','E0'}<=aliases(c)for c in a))
                            for c in a:
                                if 'N0'in aliases(c):check('actual_N0_alias_pose',equivalent(c['actual_pose'],frame['fallback'],frame['object_scale_m']))
                                if 'E0'in aliases(c):check('actual_E0_alias_pose',equivalent(c['actual_pose'],frame['e0_regenerated'],frame['object_scale_m']))
                        if domain!='real':continue
                        for name,ck in models.items():
                            if bool(ck['pair'])!=pair:continue
                            scores=independent_logits(x,ck['model_state_dict'],ck['architecture']);old=archived[frame['id']]['predictions'][name]
                            prior=np.asarray(old['logits']);equal=np.array_equal(scores,prior)
                            if len(scores):maxdiff=max(maxdiff,float(np.max(np.abs(scores-prior))))
                            check('independent_checkpoint_logits_bitwise',equal,dict(id=frame['id'],arm=name))
                            chosen=independent_choose(scores,a);c=None if chosen is None else a[chosen]
                            check('replay_exact_selected_ID',old['selected_candidate_id']==(c['candidate_id']if c else None))
                            check('replay_exact_pose',old['actual_pose']==(c['actual_pose']if c else{'available':False}))
                            check('replay_exact_alias',old['selected_source_aliases']==(sorted(aliases(c))if c else[]))
                            replays[name]+=1
            # Positive negative-control: original reference open must fail even
            # though its harmless-looking alternate name could pass a name filter.
            try:
                with open(ref,'rb'):pass
            except PermissionError:check('actual_real_reference_open_denied',True)
            else:check('actual_real_reference_open_denied',False)
        guardreceipt=guard.receipt();write(output/'PREDICTION_GUARD.json',guardreceipt)
        # Historical-only baseline regression. No new predictions or reference
        # decoding is used; this reads old, sealed numeric row journals.
        oldrows=add('old_metric_rows',archive/'evaluation_v1/ROW_METRICS.csv.gz','historical sealed regression only; isolated from new inference')
        evalpath=add('old_evaluation_seal',archive/'evaluation_v1/EVALUATION_SEAL.json','historical row hash binding')
        check('old_metric_rows_bound',sha(oldrows)==read(evalpath)['artifacts'][oldrows.name])
        with gzip.open(oldrows,'rt')as f:metricrows=list(csv.DictReader(f))
        baseline={};expected={'N0':108,'E0':114,'POSEFIX_SUBPIX_PF_ANCHOR':112,'POSEFIX_SUBPIX_YOLO_ANCHOR':111,
            'D1_H1_MLP_s20261018':120,'D1_H1_MLP_s20261019':117,'D1_H1_MLP_s20261020':121,
            'D1_H1_PAIR_MLP_s20261018':117,'D1_H1_PAIR_MLP_s20261019':113,'D1_H1_PAIR_MLP_s20261020':114,
            'D1_H1_PAIR_ROLE_AFFINE_s20261018':118,'D1_H1_PAIR_ROLE_AFFINE_s20261019':119,'D1_H1_PAIR_ROLE_AFFINE_s20261020':118}
        for arm,wanted in expected.items():
            selected=[r for r in metricrows if r['arm']==arm and r['partition']in('TEST','REMAINDER')]
            success=sum(str(r['success']).lower()in('true','1')for r in selected)
            baseline[arm]=dict(denominator=len(selected),success=success,mean_translation_cm=math.fsum(float(r['translation_cm'])for r in selected)/len(selected),mean_rotation_deg=math.fsum(float(r['rotation_deg'])for r in selected)/len(selected))
            check('historical_baseline_regression/'+arm,len(selected)==192 and success==wanted)
        # Hash all inputs a second time: no archive mutations by this command.
        for name,item in inventory.items():check('read_only_input_bytes_unchanged',sha(item['path'])==item['sha256'],name)
        write(output/'INPUTS_PRIVATE.json',dict(schema='RISK_P0_RESOLVED_INPUTS_V1',inputs=inventory,reference_content_decodes=0))
        def public_item(item):
            value=deepcopy(item)
            for root,alias in((archive,'${D1_RUN}'),(upstream,'${ORIGINAL_RUN}'),(score,'${ORIGINAL_SCORE}')):
                if Path(value['path']).is_relative_to(root):value['path']=alias+'/'+str(Path(value['path']).relative_to(root));break
            return value
        public=dict(schema='RISK_P0_PUBLIC_INPUT_SUMMARY_V1',inputs={k:public_item(v)for k,v in inventory.items()},input_dimension=188,
            splits=dict(real=238,PRIMARY192=192,TEST=49,REMAINDER=143,CAL=46,synthetic=1003,train=768,validation=235),counts=dict(counts),historical_baseline_regression=baseline)
        write(output/'INPUTS_PUBLIC_SUMMARY.json',public)
        receipt=dict(schema='RISK_INDEPENDENT_P0_CHECK_V1',status='PASS'if not failures else'FAIL',checks=sum(checks.values()),checks_by_kind=dict(checks),failure_count=len(failures),failures=failures,
            counts=dict(counts),checkpoint_replayed_rows=dict(replays),tiny_checkpoint_forward_count=sum(replays.values()),new_backbone_forward_count=0,new_PnP_optimizer_solve_count=0,new_fits=0,
            F0_input_stream_sha256=input_digest.hexdigest(),actual_pose_action_stream_sha256=pose_digest.hexdigest(),maximum_logit_absolute_difference=maxdiff,all_F0_comparisons_bitwise=True if not any('F0'in f['check']for f in failures)else False,
            prediction_guard_sha256=sha(output/'PREDICTION_GUARD.json'),private_input_manifest_sha256=sha(output/'INPUTS_PRIVATE.json'),public_input_summary_sha256=sha(output/'INPUTS_PUBLIC_SUMMARY.json'),source_sha256=sha(source),
            eligibility_semantics='Reuse original eligible flag; available means finite3x3R/3t only, no new proper-R/depth filter. N0 may be restored even if original flag is false. Upstream producer owns original numerical eligibility.',
            independent_scope='Action/dedup/188input/argmax implementation and saved affine/MLP matrix replay are independent; production inputs imported only for side-by-side comparison. Historical numeric regression reads old sealed rows, never new evaluation.',
            reference_content_decodes=0,real_reference_hash_only=True,cost=dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,threads=1,device='CPU'))
        write(output/'VERIFICATION.json',receipt);return receipt
    except BaseException as exc:
        write(output/'FAILED_ATTEMPT.json',dict(error=repr(exc),traceback=traceback.format_exc(),checks=dict(checks),failures=failures,source_sha256=sha(source),wall_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--original-score',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=run(a.archive,a.upstream,a.original_score,a.output);print(json.dumps({k:result[k]for k in('status','checks','failure_count','tiny_checkpoint_forward_count')}));raise SystemExit(0 if result['status']=='PASS'else 1)
