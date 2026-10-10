"""Independent public arithmetic checks for the fixed corrected-supervision replay.

Python >= 3.9, standard library only. This reads published JSON/CSV/gzip rows;
it never imports an experiment module, Torch, OpenCV or a private source tree.
Stored scorer metrics and reference coordinates retain their documented proxy
limits. Model weights, physical GT and first-surface engine authenticity are
not independently certified by this public arithmetic replay.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import time

sys.dont_write_bytecode = True
NAME = 'pallet_kp_corrected_supervision_20261010_v1'
OLD = 'pallet_observation_refiner_20261009_v1'
REPAIR = 'pallet_kp_supervision_repair_20261010_v1'
RUNTIME_PREP = 'pallet_kp_repair_runtime_20261010_v1'
MODEL_ARMS = ('GEOMETRY_ONLY', 'IMAGE_NO_ROLE', 'IMAGE_ROLE')
METHODS = MODEL_ARMS + ('IMAGE_ROLE_NO_MASK_ROBUST', 'IMAGE_ROLE_STANDARD', 'IMAGE_ROLE_POINT_LINE')
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
METRICS = {'translation_cm': ('translation_cm',1.,'cm'),
           'rotation_deg': ('rotation_deg',1.,'degree'),
           'ADDsym_cm': ('ADDsym_m',100.,'cm')}
LIMITS = [
    'Public arithmetic and saved-data joins only: no new inference, updates, pose fits, PnP, rays or rendering.',
    'Physical GT/scorer authenticity is not established; stored GEOMETRIC_PROXY metrics and references retain their published limitations.',
    'Unpublished checkpoint tensor bytes are not opened. Public checkpoint metadata and weight SHA declarations are linked, not treated as independent tensor certification.',
    'Runtime record arithmetic cannot recreate elapsed time, GPU state or transient interference. It checks the actual saved execution receipt and per-call records.',
]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def finite(value):
    return isinstance(value, (int,float)) and not isinstance(value,bool) and math.isfinite(value)


def close(a, b, label='', atol=1e-8, rtol=1e-9):
    if a is None or b is None:
        require(a is b, label + ': null mismatch')
    else:
        require(finite(a) and finite(b), label + ': nonfinite number')
        require(abs(a-b) <= atol + rtol * max(abs(a),abs(b)), label + ': numeric mismatch')


def same(a, b, label='', atol=1e-7, rtol=1e-10):
    if isinstance(a,list) and isinstance(b,list):
        require(len(a)==len(b),label+': list length mismatch')
        for x,y in zip(a,b):same(x,y,label,atol,rtol)
    elif finite(a) or finite(b):
        close(a,b,label,atol,rtol)
    else:
        require(a==b,label+': value mismatch')


def quantile(values,q):
    if not values:return None
    values=sorted(values);position=(len(values)-1)*q
    low=int(math.floor(position));high=int(math.ceil(position));fraction=position-low
    return values[low]+(values[high]-values[low])*fraction


def distribution(values):
    require(all(finite(v) for v in values),'nonfinite distribution input')
    n=len(values);mean=math.fsum(values)/n if n else None
    variance=math.fsum((v-mean)**2 for v in values)/(n-1) if n>1 else None
    return dict(n=n,mean=mean,sample_variance=variance,
                sample_std=math.sqrt(variance) if variance is not None else None,
                median=quantile(values,.5),P90=quantile(values,.9),max=max(values) if n else None)


def check_distribution(stored,values,label):
    result=distribution(values)
    for key,value in result.items():
        if key=='n':require(stored[key]==value,label+': n differs')
        else:close(stored[key],value,label+'/'+key)
    if 'ddof' in stored:require(stored['ddof']==1,label+': ddof differs')
    return result


def point_valid(point):
    return (isinstance(point,list) and len(point)==2 and all(finite(v) for v in point)
            and point!=[-1,-1])


def in_frame(point,hw):
    return point_valid(point) and 0<=point[0]<hw[1] and 0<=point[1]<hw[0]


def cube(dimensions):
    a,b,c=[x/2 for x in dimensions]
    return [[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],
            [-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]]


def project(pose,K):
    R,t=pose['R_cf'],pose['centroid'];result=[]
    require(len(R)==3 and all(len(row)==3 for row in R) and len(t)==3,'pose shape invalid')
    for X in cube(pose['cf_extents']):
        camera=[math.fsum(R[i][j]*X[j] for j in range(3))+t[i] for i in range(3)]
        require(all(finite(v) for v in camera) and camera[2]>1e-9,'projection depth invalid')
        uv=[math.fsum(K[i][j]*camera[j] for j in range(3)) for i in range(3)]
        result.append([uv[0]/uv[2],uv[1]/uv[2]])
    return result


def hidden(pose):
    if not pose.get('available'):return []
    R,t=pose['R_cf'],pose['centroid']
    camera=[-math.fsum(R[j][i]*t[j] for j in range(3)) for i in range(3)]
    result=[]
    for i,X in enumerate(cube(pose['cf_extents'])):
        ray=[camera[j]-X[j] for j in range(3)];length=math.sqrt(math.fsum(v*v for v in ray))
        cosine=max((1 if X[j]>=0 else -1)*ray[j]/length for j in range(3))
        if cosine < -math.sin(math.radians(2)):result.append(i)
    return result


def errors(points,reference,valid_ids):
    return [math.hypot(points[i][0]-reference[i][0],points[i][1]-reference[i][1])
            if i in valid_ids and point_valid(points[i]) else None for i in range(8)]


def schedule():
    arms=('BASE','N3_SUBPIX','N3_SUBPIX_GEOM_NOSELF_ROBUST','IMAGE_ROLE');jobs=[]
    for index in range(20):
        order=list(arms[index%4:]+arms[:index%4]);order=order[::-1] if index%2 else order
        for position,arm in enumerate(order):
            jobs.append(dict(phase='warmup',arm=arm,warmup_index=index,arm_position=position,image_index=index%26))
    for repeat in range(5):
        order=list(arms[repeat%4:]+arms[:repeat%4]);order=order[::-1] if repeat%2 else order
        for image_index in range(26):
            for position,arm in enumerate(order):
                jobs.append(dict(phase='measured',arm=arm,repeat=repeat,arm_position=position,image_index=image_index))
    return jobs


class Review:
    def __init__(self,root,require_manifest=False,allow_pending_runtime=False):
        self.root=Path(root).resolve();self.doc=self.root/'_docs/experiments'/NAME
        self.old=self.root/'_docs/experiments'/OLD;self.repair=self.root/'_docs/experiments'/REPAIR
        self.runtime_prep=self.root/'_docs/experiments'/RUNTIME_PREP
        self.require_manifest=require_manifest;self.allow_pending_runtime=allow_pending_runtime
        self.used=set();self.counts=Counter();self.data={}

    def path(self,path):
        path=(self.root/path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
        require(path.is_relative_to(self.root),'public path escapes repository')
        require(path.is_file(),'missing public input: '+str(path.relative_to(self.root)))
        self.used.add(str(path.relative_to(self.root)));return path

    def json(self,path):
        path=self.path(path)
        with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as stream:return json.load(stream)

    def rows(self,path):
        path=self.path(path)
        with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as stream:
            for line in stream:
                if line.strip():yield json.loads(line)

    def artifact(self,name):
        path=self.doc/name
        if path.exists():return self.path(path)
        if name.endswith('.jsonl') and path.with_suffix('.jsonl.gz').exists():return self.path(path.with_suffix('.jsonl.gz'))
        return self.path(path)

    def bind(self,item):
        name=Path(item['path'])
        candidate=self.root/name if not name.is_absolute() else None
        if candidate is None or not candidate.is_file():candidate=self.doc/name.name
        candidate=self.path(candidate)
        require(sha(candidate)==item['sha256'] and candidate.stat().st_size==item['bytes'],'public binding mismatch: '+name.name)
        return candidate

    def population(self,rows,methods):
        lookup={}
        for row in rows:
            key=row['method'],row['id']
            require(key not in lookup,'duplicate row identity')
            require(row['method'] in methods and row['id'] in self.frames,'unknown row identity')
            require(row['session']==self.frames[row['id']]['session'],'session mismatch')
            lookup[key]=row
        require(len(lookup)==self.n*len(methods) and all((m,i) in lookup for m in methods for i in self.ids),'complete scoped per-arm population mismatch')
        return lookup

    def protected(self):
        prior=self.json(self.doc/'PRIOR_PUBLICATION_BINDINGS.json');entries=prior['files']
        require(prior['protected_prior_files']==len(entries)==331,'protected331 population mismatch')
        require(len({b['path'] for b in entries})==331,'duplicate protected path')
        for item in entries:self.bind(item)
        inputs=self.json(self.old/'INPUTS.json')['frames'];all_ids=[f['id'] for f in inputs]
        require(len(all_ids)==len(set(all_ids))==319,'frozen legacy319 IDs mismatch')
        self.all_frames={f['id']:f for f in inputs}
        require(len({f['session'] for f in inputs})==13,'frozen13 sessions mismatch')
        self.cohort=self.json(self.doc/'COHORT.json');self.ids=self.cohort['ids'];self.n=len(self.ids)
        require(self.n==len(set(self.ids)) and set(self.ids)<=set(all_ids),'scoped cohort identities invalid')
        self.frames={fid:self.all_frames[fid] for fid in self.ids}
        self.original_initial={r['id']:r['initial_pose'] for r in self.rows(self.old/'POSE_DIAGNOSTICS.jsonl.gz') if r['method']=='BASE_NO_MASK_STANDARD'}
        require(set(self.original_initial)==set(all_ids),'fixedBase legacy initial319 mismatch')
        self.controls={(r['method'],r['id']):r for r in self.rows(self.old/'FIXED_CONTROLS.jsonl.gz')}
        return dict(prior_files=331,SHA_and_bytes_checked=True,legacy_frames=319,scoped_frames=self.n,
                    scoped_sessions=len({f['session'] for f in self.frames.values()}))

    def training(self):
        protocol=self.json(self.repair/'RETRAINING_PROTOCOL.json')
        completion=self.json(self.artifact('TRAINING_COMPLETION.json'))
        require(completion['complete'] and completion['formal_updates']==completion['total_updates']==9000 and completion['throwaway_updates']==0,'training9000 completion mismatch')
        require(completion['batch']==16 and completion['seed']==1 and completion['formal_RGB_exposures']==144000,'training settings/exposures mismatch')
        require(completion['source_probe_head_calls']==168 and completion['source_probe_image_exposures']==2688 and completion['total_head_forward_calls']==9168,'training/probe forwards accounting mismatch')
        require(all(completion[k] is True for k in ('same_initial_tensor_sha','same_batch_order','same_update_budget','input_hashes_unchanged')) and completion['source_scores_model_selection'] is False,'frozentraining parity/modelselection mismatch')
        self.bind(completion['protocol']);auth_path=self.bind(completion['authorization']);authorization=self.json(auth_path)
        require(authorization['status']=='APPROVED' and authorization['additional_formal_updates']==9000 and bool(authorization['user_authorization_evidence']),'additional9000 authorization mismatch')
        require(authorization['protocol_sha256']==sha(self.repair/'RETRAINING_PROTOCOL.json') and authorization['checks_sha256']==sha(self.repair/'ZERO_UPDATE_CPU_CHECKS.json'),'authorization protocol/checks mismatch')
        require(protocol['arms']==list(MODEL_ARMS) and protocol['updates_per_arm']==3000 and protocol['batch']==16 and protocol['parameters']==5890,'fixedtiny protocol mismatch')
        require(protocol['initial_state_sha256']==self.json(self.old/'LEARNING_PROTOCOL.json')['initial_state_sha256'],'original initialization binding mismatch')
        formal=list(self.rows(self.artifact('FORMAL_UPDATE_ROWS.jsonl')))
        require(len(formal)==9000,'formal9000 rawrows mismatch')
        ordered={a:[] for a in MODEL_ARMS}
        for index,row in enumerate(formal):
            arm=MODEL_ARMS[index//3000];step=index%3000+1
            require(row['arm']==arm and row['step']==step,'formalupdate ordering mismatch')
            ids=row['source_indices'];require(len(ids)==16 and all(type(i) is int and 0<=i<768 for i in ids),'training source-only batch invalid')
            ordered[arm].append(ids)
            expected=.001*(step/100 if step<=100 else .5*(1+math.cos(math.pi*(step-100)/2900)))
            close(row['lr'],expected,'originalLR',atol=1e-15,rtol=1e-12)
            require(finite(row['loss']) and finite(row['gradient_norm']) and row['gradient_norm']>=0,'nonfinite training loss/gradient')
        require(ordered[MODEL_ARMS[0]]==ordered[MODEL_ARMS[1]]==ordered[MODEL_ARMS[2]],'three models batch order differs')
        digest=hashlib.sha256(b''.join(struct.pack('<16q',*batch) for batch in ordered[MODEL_ARMS[0]])).hexdigest()
        require(digest==protocol['batch_order_sha256'],'frozen original batch tensor SHA mismatch')
        checkpoints=completion['checkpoints'];require([r['arm'] for r in checkpoints]==list(MODEL_ARMS),'lastcheckpoint population mismatch')
        checkpoint_lookup={r['arm']:r for r in checkpoints}
        for arm,row in checkpoint_lookup.items():
            require(row['updates']==3000 and row['exposures']==48000 and len(row['checkpoint']['sha256'])==64 and row['checkpoint']['bytes']>0,'lastcheckpoint count/binding invalid')
            first=row['first_step'];require(all(finite(v) for v in first['parameter_gradient_norms'].values()),'first gradients invalid')
            if arm=='GEOMETRY_ONLY':close(first['image_channel_gradient'],0.,'geometry imagegate gradient',0.,0.)
            if arm=='IMAGE_NO_ROLE':close(first['role_channel_gradient'],0.,'no-role rolegate gradient',0.,0.)
        metadata=self.json(self.doc/'CHECKPOINT_METADATA.json')
        require(metadata['model_forwards']==metadata['optimizer_updates']==0 and metadata['state_tensors_published'] is False,'checkpoint export scope mismatch')
        require([r['arm'] for r in metadata['rows']]==list(MODEL_ARMS),'checkpoint export arm population mismatch')
        for row in metadata['rows']:
            require(row['steps']==3000 and row['parameters']==5890 and row['config']==protocol['model'],'last checkpoint model/step/parameter metadata mismatch')
            require(row['initial_state_sha256']==protocol['initial_state_sha256'] and row['batch_order_sha256']==digest,'checkpoint original initialization/order mismatch')
            require(row['protocol_sha256']==sha(self.repair/'RETRAINING_PROTOCOL.json') and row['repaired_target_sha256']==protocol['fixed_inputs']['repaired_targets']['sha256'],'checkpoint protocol/target metadata mismatch')
            cp=checkpoint_lookup[row['arm']]['checkpoint']
            require(row['checkpoint']['sha256']==cp['sha256'] and row['checkpoint']['bytes']==cp['bytes'],'checkpoint metadata export weight binding mismatch')
        logs=list(self.rows(self.artifact('TRAIN_LOGS.jsonl')))
        formal_logs={(r['arm'],r['step']):r for r in logs if r['kind']=='formal'}
        source_logs={(r['arm'],r['step']):r for r in logs if r['kind']=='source_curve'}
        require(len(logs)==102 and len(formal_logs)==93 and len(source_logs)==9,'training log population mismatch')
        for arm in MODEL_ARMS:
            for step in [1]+list(range(100,3001,100)):
                row=formal_logs[arm,step];raw=formal[MODEL_ARMS.index(arm)*3000+step-1]
                close(row['loss_image_average'],raw['loss'],'formal log loss');close(row['lr'],raw['lr'],'formal log lr')
                close(row['gradient_norm'],raw['gradient_norm'],'formal log gradient')
                require(row['batch_images']==16 and row['positive_queries']+row['no_match_queries']==row['valid_queries'] and row['valid_queries']+row['ignored_queries']==1344,'formal log querycounts mismatch')
            require(all((arm,step) in source_logs for step in [1000,2000,3000]),'sourceprobe curve steps mismatch')
        self.data.update(protocol=protocol,completion=completion,checkpoints=checkpoint_lookup)
        self.counts['formal_update_row_replays']+=9000
        return dict(formal_updates=9000,models=3,updates_per_model=3000,exposures=144000,source_probe_forwards=168,
                    same_initial_SHA=True,same_3000x16_batch_order=True,batch_order_sha256=digest,
                    last_checkpoint_bindings=3,private_checkpoint_tensors_read=False,learning_rerun=False)

    def cohort_selection(self):
        cohort=self.cohort;source=self.json(self.doc/'COHORT_SOURCE_LABELS.json');rows=source['rows']
        require(len(rows)==len({r['id'] for r in rows})==319 and {r['id'] for r in rows}==set(self.all_frames),'existing319 severity source population mismatch')
        labels={};explicit=retained=changed=0;approved=Counter()
        for row in rows:
            fid=row['id'];s=row['source_row'];record=row['recorded_input'];frame=self.all_frames[fid]
            require(s['frame_id']==fid and s['case_id']=='DEV319::'+fid and s['population']=='DEV319' and s['session']==frame['session'],'severity identity mismatch')
            require(s['image']['sha256']==frame['image_sha256'] and s['image']['path']==frame['image'],'severity original image binding mismatch')
            label=s['severity'];require(label in ('clean','moderate','severe'),'unknown severity label')
            old=s['original_approved'];require(old['locked'] is True and old['source']=='HUMAN_DIRECT_CLASS','prior approved label scope mismatch')
            approved[old['status']]+=1;changed+=label!=old['status']
            if record is None:
                retained+=1;require(s['source']=='existing_approved_label' and label==old['status'],'retained label differs from approved original')
            else:
                explicit+=1;require(s['source']=='new_explicit_human_input' and record['frame_id']==s['case_id'] and record['severity']==label and record['image_sha256']==frame['image_sha256'] and record['session_id']==frame['session'],'stored explicit UI label linkage mismatch')
                require(record['review_scope']=='static_frame_severity_only' and record['human_identity_confirmed'] is False,'severity evidence falsely certifies broader truth')
            labels[fid]=label
        require(explicit==source['explicit_records']==316 and retained==source['retained_approved_records']==3 and changed==source['label_changes_from_original_approved']==42,'severity provenance counters mismatch')
        require(dict(approved)==source['original_approved_counts'],'original approved severity counts mismatch')
        expected=sorted(fid for fid,label in labels.items() if label in ('clean','moderate'))
        excluded=sorted(fid for fid,label in labels.items() if label=='severe')
        require(self.ids==expected and cohort['excluded_ids']==excluded and cohort['count']==self.n==245 and cohort['excluded_count']==74,'easy/medium cohort filtering mismatch')
        require(cohort['included_labels']==['clean','moderate'] and cohort['excluded_labels']==['severe'],'user scoped classes mismatch')
        require(cohort['freeze_stage']=='before_corrected_real_observations_or_accuracy_read' and cohort['selection_uses_prediction_or_pose_error'] is False,'cohort selected by new performance')
        require(cohort['disjoint'] and cohort['complete_original_population_partition'] and not cohort['reference_or_corner_truth_newly_certified'] and not cohort['human_identity_confirmed'],'cohort coverage/truth-scope claim mismatch')
        require([r['id'] for r in cohort['frames']]==self.ids,'cohort frame ordering mismatch')
        for row in cohort['frames']:
            fid=row['id'];require(row['label']==labels[fid] and row['session']==self.frames[fid]['session'] and row['image']['sha256']==self.frames[fid]['image_sha256'],'selected frame label/image/session mismatch')
        counts=Counter(labels[i] for i in self.ids)
        require(counts==Counter(clean=153,moderate=92) and cohort['counts']==dict(clean=153,moderate=92,severe_excluded=74,original=319),'cohort class counters mismatch')
        require(cohort['session_counts']==dict(Counter(self.frames[i]['session'] for i in self.ids)) and cohort['sessions']==sorted({f['session'] for f in self.frames.values()}),'cohort session counters mismatch')
        for item in cohort['bindings']:self.bind(item)
        require(source['authority_metadata']['source_sha256']==source['source_bindings'][1]['sha256'] and source['authority_metadata']['human_identity_confirmed'] is False,'severity external provenance linkage mismatch')
        panel=self.json(self.doc/'RUNTIME_PANEL.json');self.bind(panel['cohort'])
        selected=[]
        sessions=cohort['sessions'];by_session={s:sorted(i for i in self.ids if self.frames[i]['session']==s) for s in sessions}
        for index in range(2):
            for s in sessions:
                if index<len(by_session[s]):selected.append(by_session[s][index])
        selected=selected[:26]
        for fid in self.ids:
            if len(selected)==26:break
            if fid not in selected:selected.append(fid)
        require([f['frame_id'] for f in panel['frames']]==selected and len(set(selected))==26,'fixed eligible runtime panel selection mismatch')
        require(Counter(self.frames[i]['session'] for i in selected)==Counter({s:2 for s in sessions}),'runtime panel requires two eligible frames per session')
        for row in panel['frames']:
            f=self.frames[row['frame_id']]
            require(row['session_id']==f['session'] and row['label']==labels[f['id']] and row['image_key']==f['image'] and row['image']['sha256']==f['image_sha256'],'runtimepanel identity/image/label mismatch')
            require(row['camera_intrinsics']==f['K'] and row['original_hw']==f['raw_hw'] and row['dimensions_wdh_m']==[f['xyz'][0],f['xyz'][2],f['xyz'][1]] and row['input_selected_index']==f['selected_index'],'runtimepanel K/dimensions/selection mismatch')
        self.data.update(labels=labels,runtime_panel=panel)
        scope=self.json(self.doc/'SUBSET_PROTOCOL.json')
        for item in scope['fixed_inputs'].values():self.bind(item)
        self.bind(scope['cohort']);self.bind(scope['training_protocol']);self.bind(scope['subset_driver'])
        require(scope['frames']==self.n and scope['excluded_frames']==74 and scope['observation_rows']==self.n*3 and scope['pose_rows']==self.n*6 and scope['point_solver_paths']==self.n*5 and scope['point_line_paths']==self.n,'frozen scope population mismatch')
        require(scope['evaluation_scope']=='easy_medium_only' and scope['cohort_selection_uses_accuracy'] is False and scope['model_settings_changed'] is False and scope['new_training_updates']==9000 and scope['new_N3_updates']==scope['source_RGB_regenerated']==0,'scope algorithm or cost changed')
        return dict(original_labels=319,explicit_existing_records=316,retained_existing_approved=3,changed_from_older_labels=42,
                    selected=245,clean=153,moderate=92,severe_excluded=74,runtime_panel=26,
                    selection_replayed=True,zero_external_occlusion_not_claimed=True)

    def observations(self):
        seal=self.json(self.doc/'OBSERVATION_SEAL.json');path=self.doc/'LEARNED_OBSERVATIONS.jsonl.gz'
        require(seal['complete'] and seal['frames']==self.n and seal['rows']==self.n*3 and seal['models']==list(MODEL_ARMS),'scoped learned observation seal population mismatch')
        require(seal['GT_open_allowed'] is False,'GT used during learned observation inference')
        self.bind(seal['records']);require(len(seal['checkpoints'])==3,'observation checkpoint population mismatch')
        for arm,b in zip(MODEL_ARMS,seal['checkpoints']):
            c=self.data['checkpoints'][arm]['checkpoint']
            require(b['sha256']==c['sha256'] and b['bytes']==c['bytes'],'observation does not use corrected last checkpoint')
        lookup=self.population(self.rows(path),MODEL_ARMS);queries_replayed=0;lines_checked=0
        for (arm,fid),row in lookup.items():
            require(row.get('GT_input') is False,'learned inference GT contract mismatch')
            if not row['detector_available']:
                require(row['queries']==row['corners']==row['lines']==[],'absent detector fabricated observations')
                continue
            base=row['original_base_points'];same(base,self.frames[fid]['points']['BASE'],'fixed detector parity',atol=.001)
            queries=row['queries'];require(len(queries)==84,'original84 query population mismatch')
            accepted=[];raw=[]
            for qi,query in enumerate(queries):
                edge=qi//7;a,b=EDGES[edge];u=(qi%7+1)/8
                require(query['query']==qi and query['edge']==edge and query['endpoints']==[a,b] and query['fraction']==u,'query identity/geometry convention mismatch')
                vector=[base[b][j]-base[a][j] for j in range(2)] if point_valid(base[a]) and point_valid(base[b]) else [0.,0.]
                length=math.hypot(*vector);valid=point_valid(base[a]) and point_valid(base[b]) and length>1e-6
                if valid:
                    center=[(1-u)*base[a][j]+u*base[b][j] for j in range(2)];normal=[-vector[1]/length,vector[0]/length]
                    same(query['center'],center,'query center');same(query['normal'],normal,'query normal')
                scores=query['candidate_logits'];require(len(scores)==66 and all(finite(v) for v in scores),'66way logits invalid')
                raw.extend(scores);choice=max(range(66),key=lambda j:scores[j])
                require(query['chosen_candidate']==choice and query['no_match']==(choice==65 or not valid),'original66wayMAP decode mismatch')
                expected=[query['center'][j]+(choice-32)*query['normal'][j] for j in range(2)] if choice<65 and valid else None
                same(query['selected_xy'],expected,'query selected1px candidate')
                if expected is not None:accepted.append(qi)
                queries_replayed+=1
            digest=hashlib.sha256(struct.pack('<'+str(len(raw))+'f',*raw)).hexdigest()
            require(digest==row['raw_logits_sha256'],'storedFP32 rawlogit semantic SHA mismatch')
            require(row['selected_queries']==len(accepted),'selected query count mismatch')
            by_edge={line['edge']:line for line in row['lines']}
            require(len(by_edge)==len(row['lines']) and set(by_edge)<=set(range(12)),'duplicate or unknown physical line edge')
            expected_edges=[]
            for edge in range(12):
                ids=[i for i in accepted if i//7==edge]
                if len(ids)<2:continue
                points=[queries[i]['selected_xy'] for i in ids]
                center=[math.fsum(p[j] for p in points)/len(points) for j in range(2)]
                centered=[[p[j]-center[j] for j in range(2)] for p in points]
                xx=math.fsum(p[0]**2 for p in centered);yy=math.fsum(p[1]**2 for p in centered);xy=math.fsum(p[0]*p[1] for p in centered)
                large=(xx+yy+math.hypot(xx-yy,2*xy))/2
                if math.sqrt(max(0.,large))<1e-6:continue
                expected_edges.append(edge);require(edge in by_edge,'valid >=2 support edge lost by storedTLS')
                line=by_edge[edge];require(line['queries']==ids and line['endpoints']==list(EDGES[edge]),'line query/endpoints mismatch')
                same(line['support_points'],points,'line chosen support coordinates')
                normal=line['normal'];require(len(normal)==2 and all(finite(v) for v in normal),'line normal invalid')
                close(math.hypot(*normal),1.,'TLS line unitnormal')
                tangent=[normal[1],-normal[0]]
                close(line['offset'],math.fsum(normal[j]*center[j] for j in range(2)),'TLS mean offset')
                eigenvalue=xx*tangent[0]**2+2*xy*tangent[0]*tangent[1]+yy*tangent[1]**2
                close(eigenvalue,large,'TLS largest-axis arithmetic',atol=1e-6,rtol=1e-8)
                support=[math.fsum(p[j]*tangent[j] for j in range(2)) for p in centered]
                close(line['support_length_px'],max(support)-min(support),'TLS supportspan',atol=1e-6)
                rms=math.sqrt(math.fsum((math.fsum(p[j]*normal[j] for j in range(2))-line['offset'])**2 for p in points)/len(points))
                close(line['residual_rms_px'],rms,'TLS residualRMS',atol=1e-6)
                lines_checked+=1
            require(set(by_edge)==set(expected_edges),'stored physical line population mismatch')
            corners={c['id']:c for c in row['corners']};require(len(corners)==len(row['corners']),'duplicate cornerID')
            expected_corners=[]
            for corner in range(8):
                incident=[e for e,(a,b) in enumerate(EDGES) if corner in (a,b) and e in by_edge]
                pairs=[]
                for ii,a in enumerate(incident):
                    for b in incident[ii+1:]:
                        x,y=by_edge[a]['normal'],by_edge[b]['normal'];det=x[0]*y[1]-x[1]*y[0]
                        if abs(det)>1e-6:pairs.append((abs(det),a,b))
                if not pairs:continue
                expected_corners.append(corner);require(corner in corners,'eligible line intersection corner missing')
                c=corners[corner];a,b=c['edges'];require((a,b) in [(a,b) for _,a,b in pairs],'corner uses nonincident/degenerate edges')
                x,y=by_edge[a]['normal'],by_edge[b]['normal'];det=x[0]*y[1]-x[1]*y[0]
                require(abs(det)>=max(p[0] for p in pairs)-1e-10,'corner failed fixed maximum-det pair selection')
                oa,ob=by_edge[a]['offset'],by_edge[b]['offset']
                expected=[(oa*y[1]-x[1]*ob)/det,(x[0]*ob-oa*y[0])/det]
                same(c['xy'],expected,'physical-line corner intersection',atol=1e-6)
            require(set(corners)==set(expected_corners),'stored corner population mismatch')
            require(row['all_no_match']==(not row['lines']),'all-no-match line flag mismatch')
        require(seal['detector_forward']==self.n and seal['head_forward']==self.n*3 and seal['initial_feature_pose_calls']==self.n,'fresh inference call accounting mismatch')
        self.data.update(observations=lookup,observation_seal=seal)
        self.counts['stored66way_logit_queries_replayed']+=queries_replayed
        return dict(rows=self.n*3,frames=self.n,queries=queries_replayed,physical_lines_checked=lines_checked,
                    original66wayMAP=True,storedTLS_and_intersection_arithmetic_checked=True,
                    corrected_checkpoint_linked=True,actual_model_forwards_in_this_verifier=0)

    def geometry(self):
        execution=self.json(self.doc/'LEARNED_POSE_EXECUTION.json')
        require(execution['complete'] and execution['rows']==self.n*6 and execution['models']==3 and execution['point_solver_paths']==self.n*5 and execution['point_line_paths']==self.n,'scoped6path pose completion mismatch')
        require(execution['GT_canary'] is True,'pose inference GT canary missing')
        self.bind(execution['raw_observations']);self.bind(execution['final_sealed'])
        sealed=self.population(self.rows(self.doc/'LEARNED_GEOMETRY_SEALED.jsonl.gz'),METHODS)
        scored=self.population(self.rows(self.doc/'LEARNED_PREDICTIONS.jsonl.gz'),METHODS)
        projected_corners=0;statuses=Counter()
        for key,row in sealed.items():
            method,fid=key;frame=self.frames[fid];pred=scored[key];base=frame['points']['BASE']
            for name,value in row.items():require(name in pred and pred[name]==value,'sealed→scored changed deployable '+name)
            require(not any(k in row for k in ('pose','corner','mask_audit','baseline_pose','baseline_corner')),'scored GT fields in preGT geometryseal')
            require(pred['pose']['available']==row['pose_available'],'scored/operational availability mismatch')
            require(pred['baseline_pose']==self.controls['BASE',fid]['pose'] and pred['baseline_corner']==self.controls['BASE',fid]['corner'],'scoring same fixedBase baseline metric changed')
            require(row['K']==frame['K'] and row['xyz']==frame['xyz'] and row['raw_hw']==frame['raw_hw'],'frozen camera/registry/image dimensions changed')
            require(row['selected_index']==frame['selected_index'] and row['fixed_metadata']['preserved'],'candidate selection/metadata contract mismatch')
            require(row['fixed_metadata']['candidate_metadata']==frame['candidate_metadata'],'stored candidate metadata changed')
            require(row['initial_pose']==self.original_initial[fid],'fixedBase initial pose changed')
            arm=method if method in MODEL_ARMS else 'IMAGE_ROLE';observation=self.data['observations'][arm,fid]
            require(row['observation_raw_logits_sha256']==observation.get('raw_logits_sha256'),'geometry→observation rawlogit SHA mismatch')
            selected={c['id']:c['xy'] for c in observation['corners']}
            sparse=[selected.get(i,[None,None]) for i in range(8)]+[base[8]]
            same(row['input_points'],sparse,'sparse observed input without Base fill',atol=0.,rtol=0.)
            require(row['correspondence_absence_not_used_to_fill_pose'] and row['reprojections_reused_as_observations'] is False,'no-match filling/reprojection refit contract mismatch')
            H=[] if method=='IMAGE_ROLE_NO_MASK_ROBUST' else hidden(row['initial_pose'])
            require(row['hidden_initial']==H and row['excluded']==H,'prediction-only initial hidden mask mismatch')
            fit=row['solver'].get('fit_input_ids',[]);require(set(H).isdisjoint(fit) and set(fit)<=set(selected),'hidden/fabricated corner entered fit')
            new=bool(row['solver']['available']);fallback=not new and bool(row['initial_pose']['available'])
            require(row['new_pose_estimated']==new and row['fallback_used']==fallback and row['no_pose']==(not row['pose_available']),'new/fallback/failure flags mismatch')
            require(row['output_status']==('NEW_POSE' if new else 'BASELINE_FALLBACK' if fallback else 'POSE_FAILURE'),'output status mismatch')
            require(row['actual_pose']['available']==row['pose_available'],'operational pose object availability mismatch')
            require(row['hidden_reprojected']==bool(new and H) and row['reprojected_ids']==(H if new else []),'hidden replacement flags mismatch')
            require(len(row['native_points'])==9 and row['native_points'][8]==base[8],'center changed')
            expected=[list(p) for p in base]
            if new:
                for field in ('R_cf','R_physical','centroid','cf_extents'):
                    same(row['actual_pose'][field],row['solver'][field],'final actualPose→solver '+field,atol=0.,rtol=0.)
                projections=project(row['actual_pose'],row['K']);same(row['solver']['projected'],projections,'independent final R,t/K projection',atol=1e-6)
                for i,xy in selected.items():
                    if i not in H:expected[i]=xy
                for i in H:expected[i]=projections[i];projected_corners+=1
                require(row['hidden_after']==hidden(row['actual_pose']) and row['hidden_set_changed']==(set(row['hidden_after'])!=set(H)),'post-fit hidden mask diagnostic mismatch')
            else:require(row['actual_pose']==row['initial_pose'] and not row['hidden_after'] and not row['hidden_set_changed'],'fullBase fallback pose/mask mismatch')
            same(row['native_points'],expected,'new observed updates/H projection or fullBase fallback',atol=1e-6)
            if method=='IMAGE_ROLE_POINT_LINE':
                require(row['local_point_line_refinement'] and row['independent_four_point_PnP'] is False,'local line ablation misrepresented as independentPnP')
                allowed={c['id'] for c in observation['corners'] if c['id'] not in H and in_frame(c['xy'],frame['raw_hw'])}
                consumed={e for c in observation['corners'] if c['id'] in allowed for e in c['edges']}
                retained={l['edge'] for l in observation['lines']}-consumed
                for answer in row['solver'].get('attempts',[]):
                    if 'point_ids' not in answer:continue
                    require(set(answer['point_ids'])==allowed and answer['consumed_edges']==sorted(consumed) and set(answer['line_edges'])==retained,'point/line factors differ from same selected observations')
                    require(set(answer['line_edges']).isdisjoint(consumed) and answer['same_edge_point_line_double_count'] is False,'point/line duplicate physical edge')
            else:
                require(row['solver']['hidden']==H and row['solver']['excluded']==H,'final solver hidden/excluded mismatch')
                require(set(row['solver'].get('final_inliers',[]))<=set(row['solver']['used']),'final inliers outside observed pool')
            statuses[(method,row['output_status'])]+=1
        self.data.update(sealed=sealed,scored=scored,pose_execution=execution)
        for stage in ('infer','evaluate'):
            receipt=self.json(self.doc/('SUBSET_'+stage.upper()+'_ADAPTER_RECEIPT.json'))
            require(receipt['complete'] and receipt['schema']=='subset_original_path_supervision_repair_adapter_v1' and receipt['stage']==stage and receipt['frames']==self.n,'scoped actual stage receipt mismatch')
            require(receipt['excluded_frames_executed']==0 and receipt['protected_prior_files_preserved']==331 and receipt['original319_authority_verified'],'excluded frame execution/preservation claim mismatch')
            for name in ('cohort','training_protocol','completed_training','scope_protocol','driver'):self.bind(receipt[name])
            if stage=='evaluate':
                for name in ('geometry','predictions','pose_execution'):self.bind(receipt[name])
                require(receipt['reference_scope']['actual_cached_reference_frames']==self.n and receipt['reference_scope']['excluded_cached_reference_frames']==0,'reference read scope mismatch')
            else:
                for name in ('observations','observation_seal'):self.bind(receipt[name])
                require(receipt['detector_forwards']==self.n and receipt['head_forwards']==self.n*3 and receipt['feature_initial_pose_calls']==self.n,'actual scoped inference receipt count mismatch')
        self.counts['sealed_scored_pose_row_joins']+=self.n*6;self.counts['hidden_corner_projection_replays']+=projected_corners
        return dict(rows=self.n*6,frames_per_arm=self.n,arms=6,hidden_corner_coordinates_independently_projected=projected_corners,
                    fullBase_fallback_and_sparse_observation_contract=True,GT_scoring_not_repeated=True,
                    outcomes={m:{status:statuses[m,status] for status in ('NEW_POSE','BASELINE_FALLBACK','POSE_FAILURE')} for m in METHODS})

    def scoring_resume(self):
        protocol=self.json(self.doc/'SCORING_RESUME_PROTOCOL.json')
        interruption=self.json(self.doc/'EVALUATION_INTERRUPTION.json')
        receipt=self.json(self.doc/'SUBSET_EVALUATE_ADAPTER_RECEIPT.json')
        require(protocol['status']=='PREPARED_SCORING_ONLY' and protocol['frames']==self.n and protocol['rows']==self.n*6 and protocol['methods']==list(METHODS),'scoring-only resume scope mismatch')
        require(protocol['original_model_decoder_solver_unchanged'] and all(protocol[k]==0 for k in ('new_detector_forwards','new_head_forwards','new_PnP_calls','new_optimizer_calls','new_geometry_rows')),'resume repeated scientific inference')
        for key,item in protocol['fixed_inputs'].items():
            if key!='axis_manifest':self.bind(item)
        require(interruption['geometry_complete'] and interruption['geometry_rows']==self.n*6 and interruption['scored_rows_before_failure']==interruption['reference_cached_frames_opened_before_failure']==0,'guarded interruption scope mismatch')
        require(all(interruption[k]==0 for k in ('new_detector_forwards','new_head_forwards','new_fit_calls')),'interruption receipt declares new fits')
        for key in ('preserved_attempt','original_subset_driver','original_scope_protocol','resume_protocol'):self.bind(interruption[key])
        require('original319 reference authority differs' in interruption['error'],'recorded guard failure changed')
        for key in ('scoring_resume_code','scoring_resume_protocol','original_guard_failure_preserved'):self.bind(receipt[key])
        require(receipt['geometry_recomputed'] is False and receipt['actual_scoring_rows']==self.n*6 and all(receipt[k]==0 for k in ('new_detector_forwards','new_head_forwards','new_PnP_calls','new_optimizer_calls')),'scoring receipt fit-free resume mismatch')
        mapping=self.json(self.doc/'REFERENCE_ID_MAPPING.json');rows=mapping['rows']
        require(mapping['authority_frames']==len(rows)==319 and mapping['selected_frames']==self.n and mapping['selection_uses_reference_values'] is False,'reference alias scope mismatch')
        require(len({r['source_cache_id'] for r in rows})==len({r['public_id'] for r in rows})==319 and {r['public_id'] for r in rows}==set(self.all_frames),'reference alias mapping not bijective')
        aliases={}
        for row in rows:
            fid=row['public_id'];f=self.all_frames[fid]
            require(row['session']==f['session'] and row['image']==f['image'] and row['image_sha256']==f['image_sha256'],'reference alias image/session/SHA mismatch')
            require(row['source_cache_id']==fid.replace(':','__',1) and row['selected']==(fid in self.frames),'source-cache/public alias or selected mapping mismatch')
            aliases[fid]=row['source_cache_id']
        scope=receipt['reference_scope'];require(scope['axis_aliases']==aliases and scope['authority_metadata_frames']==319 and scope['actual_scored_frames']==self.n,'scoring alias receipt mismatch')
        self.bind(scope['mapping'])
        ledger=Counter();reasons=Counter();starts=optimizer=nfev=0
        for (method,fid),row in self.data['sealed'].items():
            if method!='IMAGE_ROLE_POINT_LINE':ledger.update(row['solver']['operation_counts'])
            else:
                for answer in row['solver'].get('attempts',[]):
                    starts+=1;reasons[answer['reason']]+=1
                    if 'nfev' in answer:optimizer+=1;nfev+=answer['nfev']
        execution=self.data['pose_execution'];counts=execution['fit_counts']
        require(dict(ledger)==counts['point_solver_ledger'] and dict(reasons)==counts['point_line_attempt_reasons'],'stored per-solve primitive ledger mismatch')
        require(counts['point_line_starts_checked']==starts and counts['point_line_optimizer_calls']==optimizer and counts['point_line_scipy_reported_nfev']==nfev,'stored point-line actual optimizer ledger mismatch')
        expected=dict(ledger);expected.update(point_line_paths=self.n,point_line_optimizer_starts=optimizer,point_line_nfev=nfev)
        require(execution['counts']==expected and receipt['primitive_counts_reconstructed']==counts,'execution ledger/receipt joins mismatch')
        require(execution['partial_original_evaluation_resumed_scoring_only'] and execution['new_geometry_on_resume']==0 and execution['seconds'] is None,'missing full failed-prefix time invented or geometry repeated')
        require(counts['optimizer_residual_callback_calls'] is None and 'NA' in counts['optimizer_residual_callback_count_status'],'unsaved callback count invented')
        require(counts['unit_check_optimizer_calls']==2 and 'derived' in counts['unit_check_count_basis'],'unit-check count provenance mismatch')
        require(mapping['axis_manifest']['sha256']==protocol['fixed_inputs']['axis_manifest']['sha256'],'external axis provenance SHA mismatch')
        self.data['reference_mapping']=mapping
        self.counts['saved_per_solve_operation_ledgers_replayed']+=self.n*6
        return dict(source_cache_public_aliases=319,selected_reference_caches=self.n,sealed_rows_scored=self.n*6,
                    new_geometry_on_resume=0,point_solver_ledger=dict(ledger),point_line_optimizer_calls=optimizer,
                    scipy_reported_nfev=nfev,actual_callback_count=None,failed_prefix_wall_seconds=None,
                    unsaved_counters_not_certified=True,external_axis_contents_not_opened=True)

    def metrics(self):
        metric=self.json(self.doc/'METRICS.json');wanted=set(metric['methods']);lookup={}
        raw_sources={}
        for name in ('FIXED_CONTROLS.jsonl.gz','PREDICTIONS.jsonl.gz','LEARNED_PREDICTIONS.jsonl.gz'):
            for row in self.rows(self.old/name):
                raw_sources[name,row['method'],row['id']]=row
        declarations=metric.get('method_sources',{})
        for method in wanted:
            for fid in self.ids:
                if method.startswith('CORRECTED_'):
                    raw=method[len('CORRECTED_'):];row=self.data['scored'][raw,fid]
                elif method in declarations:
                    source=declarations[method];path=source['path'];raw=source.get('raw_method',method)
                    if Path(path).name=='LEARNED_PREDICTIONS.jsonl.gz' and NAME in path:
                        row=self.data['scored'][raw,fid]
                    else:row=raw_sources[Path(path).name,raw,fid]
                else:
                    raw=method[len('ORIGINAL_'):] if method.startswith('ORIGINAL_') else method
                    candidates=[r for (name,m,i),r in raw_sources.items() if m==raw and i==fid]
                    require(len(candidates)==1,'metric raw source ambiguous or absent: '+method)
                    row=candidates[0]
                lookup[method,fid]=row
        require(all('CORRECTED_'+m in wanted for m in METHODS) and {'BASE','N3_SUBPIX'}<=wanted,'required corrected and simple comparisons omitted')
        checked=0
        def arm_summary(method,stored,ids):
            nonlocal checked
            rows=[lookup[method,i] for i in ids]
            require(stored['total_frames']==len(ids),'metric scope denominator mismatch')
            for flag in ('pose_available','new_pose_estimated','fallback_used','no_pose','hidden_reprojected'):
                require(stored[flag]==sum(bool(r[flag]) for r in rows),'metric operational count mismatch: '+method+'/'+flag)
            for field,flag in (('available_ids','pose_available'),('new_pose_ids','new_pose_estimated'),('fallback_ids','fallback_used'),('no_pose_ids','no_pose')):
                if field in stored:require(stored[field]==[i for i in ids if lookup[method,i][flag]],'metric '+field+' population mismatch')
            if 'output_status_counts' in stored:require(stored['output_status_counts']==dict(Counter(r['output_status'] for r in rows)),'metric status counts mismatch')
            for scope,selected in (('operational',[r for r in rows if r['pose_available']]),('new_pose',[r for r in rows if r['new_pose_estimated']])):
                for name,(field,factor,unit) in METRICS.items():
                    values=[r['pose'][field]*factor for r in selected]
                    check_distribution(stored['metrics'][scope][name],values,method+'/'+scope+'/'+name)
                    require(stored['metrics'][scope][name]['unit']==unit,'pose metric unit mismatch');checked+=1
        strata=metric['strata'];require(set(strata)=={'combined','easy','medium'},'three fixed strata missing')
        require(metric['methods']==strata['combined']['methods'] and metric['contrasts']==strata['combined']['contrasts'],'topline/combined metrics differ')
        strata_ids={'combined':self.ids,'easy':[i for i in self.ids if self.data['labels'][i]=='clean'],
                    'medium':[i for i in self.ids if self.data['labels'][i]=='moderate']}
        contrast_sets={};contrast_scopes=0
        for stratum,block in strata.items():
            selected=strata_ids[stratum]
            require(block['ids']==selected and block['count']==len(selected) and set(block['methods'])==wanted,'stratum IDs/denominator/method coverage differs')
            for method,stored in block['methods'].items():arm_summary(method,stored,selected)
            require(set(block['contrasts'])==set(metric['contrasts']),'stratum contrasts differ')
            for key,scopes in block['contrasts'].items():
                a,b=key.split('_minus_',1);require(a in wanted and b in wanted,'contrast unknown method')
                require(set(scopes)=={'common_operational','candidate_new_pose','both_new_pose'},'paired scope omitted')
                for scope,stored in scopes.items():
                    ids=[i for i in selected if lookup[a,i]['pose']['available'] and lookup[b,i]['pose']['available']
                         and (scope=='common_operational' or lookup[a,i]['new_pose_estimated'])
                         and (scope!='both_new_pose' or lookup[b,i]['new_pose_estimated'])]
                    require(stored['denominator']==len(selected) and stored['common_frames']==len(ids) and stored['common_ids']==stored['pair_ids']==ids,'paired complete/common/newset denominator mismatch')
                    require(stored['excluded_ids']==[i for i in selected if i not in set(ids)],'paired excluded population mismatch')
                    contrast_sets[stratum,key,scope]=ids
                    for name,(field,factor,unit) in METRICS.items():
                        va=[lookup[a,i]['pose'][field]*factor for i in ids];vb=[lookup[b,i]['pose'][field]*factor for i in ids]
                        delta=[(lookup[a,i]['pose'][field]-lookup[b,i]['pose'][field])*factor for i in ids]
                        summary=stored['metrics'][name];check_distribution(summary,delta,key+'/'+scope+'/'+name)
                        close(summary['mean_delta'],distribution(delta)['mean'],'paired mean_delta')
                        require(summary['improved_frames']==sum(d < -1e-9 for d in delta) and summary['worsened_frames']==sum(d > 1e-9 for d in delta) and summary['unchanged_frames']==sum(abs(d)<=1e-9 for d in delta),'paired improvement/worsening counters mismatch')
                        for values,fieldname in ((va,'new_marginals'),(vb,'comparator_marginals')):
                            check_distribution(stored[fieldname][name],values,key+'/'+scope+'/'+fieldname+'/'+name)
                        checked+=3
                    contrast_scopes+=1
        csv_rows=list(csv.DictReader(self.path(self.doc/'METRICS.csv').open(newline='',encoding='utf-8')))
        require(len(csv_rows)==len(wanted)*18 and len({(r['stratum'],r['method'],r['scope'],r['metric']) for r in csv_rows})==len(csv_rows),'metric CSV population mismatch')
        for row in csv_rows:
            stored=strata[row['stratum']]['methods'][row['method']]['metrics'][row['scope']][row['metric']]
            for key,cell in row.items():
                if key in ('stratum','method','scope','metric'):continue
                expected=stored[key]
                if expected is None:require(cell=='','CSV null mismatch')
                elif isinstance(expected,str):require(cell==expected,'CSV string/unit mismatch')
                else:close(float(cell),expected,'CSV '+key,atol=1e-12,rtol=1e-12)
        filtered=list(self.rows(self.doc/'HISTORICAL_FILTERED_ROWS.jsonl.gz'))
        require(len(filtered)==4900,'historical filtered population mismatch')
        for row in filtered:
            require(row['id'] in self.frames,'severe entered historical filtered rows')
            if row.get('historical_stress'):continue
            original=lookup[row['alias'],row['id']]
            require({k:v for k,v in row.items() if k!='alias'}==original,'filtered historical row modified')
        self.data.update(metrics=metric,metric_lookup=lookup,contrast_sets=contrast_sets,strata_ids=strata_ids)
        self.counts['raw_pose_distributions_replayed']+=checked
        return dict(methods=len(wanted),frames_per_method=self.n,distributions=checked,CSV_rows=len(csv_rows),
                    paired_scopes=contrast_scopes,means_samplevariance_SD_median_linearP90=True,
                    stored_proxy_scorer_metrics=True,private_GT_recomputed=False)

    def bootstrap(self):
        draws=self.json(self.root/'_docs/experiments/pallet_kp_difficulty_20261010_v1/BOOTSTRAP_SESSION_DRAWS.json.gz')
        matrix=draws['counts'];sessions=draws['sessions'];metric=self.data['metrics']
        require(draws['shape']==[10000,13] and len(matrix)==10000 and len(sessions)==13,'frozen10000x13 bootstrap shape mismatch')
        require(all(len(row)==13 and all(type(v) is int and 0<=v<=13 for v in row) and sum(row)==13 for row in matrix),'bootstrap multiplicity invalid')
        digest=hashlib.sha256(b''.join(struct.pack('<13H',*row) for row in matrix)).hexdigest()
        require(digest==draws['serialized_raw_sha256']==metric['bootstrap']['serialized_raw_sha256']=='63e288a51d7b0612616beefac28fcc625e76e8c5d7b5a0ecc5e8b85c73048fa5','original frozen bootstrap digest mismatch')
        require(metric['bootstrap']['sessions']==sessions and metric['bootstrap']['new_draws_generated']==0,'bootstrap session ordering/RNG claim mismatch')
        require(metric['bootstrap']['session_frame_counts']==dict(Counter(self.frames[i]['session'] for i in self.ids)),'bootstrap cohort session denominator mismatch')
        lookup=self.data['metric_lookup'];index={s:j for j,s in enumerate(sessions)};cache={};ci_count=mean_replays=0
        for (stratum,key,scope),ids in self.data['contrast_sets'].items():
            a,b=key.split('_minus_',1);stored=metric['strata'][stratum]['contrasts'][key][scope];cachekey=a,b,tuple(ids)
            if cachekey not in cache:
                population=[0]*13;values=[[[] for _ in METRICS] for _ in sessions]
                for fid in ids:
                    j=index[self.frames[fid]['session']];population[j]+=1
                    for k,(field,factor,unit) in enumerate(METRICS.values()):values[j][k].append((lookup[a,fid]['pose'][field]-lookup[b,fid]['pose'][field])*factor)
                sums=[[math.fsum(x) for x in row] for row in values];means=[[] for _ in METRICS]
                if ids:
                    for draw in matrix:
                        denominator=sum(n*c for n,c in zip(population,draw))
                        if denominator:
                            for k in range(3):means[k].append(math.fsum(sums[j][k]*draw[j] for j in range(13))/denominator)
                    mean_replays+=10000*3
                cache[cachekey]=means
            means=cache[cachekey];require(stored['bootstrap_nonempty_resamples']==len(means[0]),'bootstrap empty resample accounting mismatch')
            for k,name in enumerate(METRICS):
                ci=[quantile(means[k],.025),quantile(means[k],.975)] if means[k] else None
                same(stored['metrics'][name]['CI95'],ci,'independent frozen13session pairedCI95',atol=2e-8);ci_count+=1
        self.counts['frozen_resample_metric_means_replayed']+=mean_replays;self.counts['pairedCI95_replayed']+=ci_count
        return dict(existing_draws=10000,sessions=13,new_random_draws=0,paired_metric_CIs=ci_count,scalar_resample_mean_replays=mean_replays)

    def posthoc(self):
        metric=self.data['metrics'];lookup=self.data['metric_lookup'];methods=set(metric['methods'])
        prior={(r['method'],r['id']):r for r in self.rows(self.old/'REAL_CORRESPONDENCE_ROWS.jsonl.gz')}
        rows=list(self.rows(self.doc/'POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz'))
        records=self.population(rows,methods);references={};pairs_replayed=0
        for (method,fid),row in records.items():
            pred=lookup[method,fid];arm='N3_SUBPIX' if method.startswith('N3_SUBPIX') else 'BASE'
            baseline=lookup[arm,fid];audit=prior[arm+'_NO_MASK_ROBUST',fid]
            require(row['raw_method']==pred['method'] and row['label']==self.data['labels'][fid] and row['frozen_baseline_arm']==arm,'posthoc row/source/label/phase join mismatch')
            require(row['reference_kind']=='GEOMETRIC_PROXY' and row['GT_used_after_geometry_seal'] is True,'posthoc reference scope mismatch')
            for key in ('permutation_native_to_canonical','human_states_native','reference_matched','reference_valid_native_ids'):
                require(row[key]==audit[key],'posthoc inherited phase/reference valid/human state mismatch')
            require(row['accuracy_threshold_px']==8. and row['raw_hw']==self.frames[fid]['raw_hw'],'reference threshold/image convention changed')
            reference=row['reference_native_points_px'];valid=set(row['reference_valid_native_ids'])
            require(len(reference)==8 and valid<=set(range(8)) and all(point_valid(reference[i]) for i in valid),'stored reference point population invalid')
            refkey=arm,fid
            if refkey in references:require(references[refkey]==(reference,sorted(valid)),'same-phase reference differs across methods')
            references[refkey]=reference,sorted(valid)
            require(row['frozen_BASE_native_points']==self.frames[fid]['points']['BASE'] and row['frozen_initial_native_points']==baseline['native_points'],'posthoc fixed initial coordinates changed')
            require(row['input_native_points']==pred.get('input_points',pred['native_points']) and row['output_native_points']==pred['native_points'],'posthoc input/output coordinates not linked to scored row')
            actual={}
            for name,pointkey in (('input','input_native_points'),('initial','frozen_initial_native_points'),('output','output_native_points')):
                actual[name]=errors(row[pointkey],reference,valid)
                same(row['reference_error_'+name+'_native_px'],actual[name],'public reference-error '+name,atol=1e-7)
            same(actual['initial'],audit['reference_error_input_native_px'],'inherited proxy reference-error anchor',atol=1e-7)
            known={i for i,v in enumerate(actual['input']) if v is not None};correct={i for i in known if actual['input'][i]<=8.}
            require(row['reference_valid_input_ids']==sorted(known) and row['correct_input_native_ids']==sorted(correct),'reference input correctness threshold mismatch')
            paired=[i for i in range(8) if actual['initial'][i] is not None and actual['output'][i] is not None]
            require(row['paired_corner_valid_ids']==paired,'paired corner population mismatch');pairs_replayed+=len(paired)
            solver=pred.get('solver',{});local=bool(pred.get('local_point_line_refinement') or method.endswith('POINT_LINE'))
            pool=set(solver.get('used',[]));inliers=set() if local else set(solver.get('final_inliers',solver.get('inliers',[])))
            require(row['pool_ids']==sorted(pool) and row['final_inlier_ids']==sorted(inliers),'posthoc final solver pool/inlier join mismatch')
            for prefix,ids in (('pool',pool),('final_inlier',inliers)):
                require(row['correct_'+prefix+'_ids']==sorted(ids&correct) and row['correct_'+prefix+'_count']==len(ids&correct),'correct count/ID mismatch')
                require(row['wrong_'+prefix+'_ids']==sorted((ids&known)-correct) and row['unknown_'+prefix+'_ids']==sorted(ids-known),'reference correctness partitions mismatch')
            states=row['human_states_native'];direct={i for i,s in enumerate(states) if s=='DIRECT_VISIBLE'}
            selfH={i for i,s in enumerate(states) if s=='SELF_OCCLUDED'};labelknown={i for i,s in enumerate(states) if s!='UNANNOTATED'}
            H=set(pred.get('hidden_initial',[]));reprojected=pred.get('reprojected_ids',[])
            fields={'human_direct_pool_ids':pool&direct,'human_direct_correct_pool_ids':pool&direct&correct,
                    'human_direct_correct_final_inlier_ids':inliers&direct&correct,'false_excluded_direct_ids':H&direct,
                    'false_excluded_accurate_ids':H&correct,'false_retained_human_self_ids':(pool&selfH)-H}
            for name,ids in fields.items():require(row[name]==sorted(ids),'posthoc human mask/correspondence subset mismatch')
            require(row['mask_applied']==('NO_MASK' not in method and method not in ('BASE','N3_SUBPIX')) and row['mask_wrong_on_known']==bool((H^selfH)&labelknown),'mask classification versus no-mask diagnostic relation mismatch')
            require(row['initial_hidden_ids']==sorted(H) and row['hidden_reprojected_ids']==reprojected and row['hidden_set_changed']==bool(pred.get('hidden_set_changed')),'posthoc H/reprojection join mismatch')
            require(row['point_PnP_inliers_applicable']==(not local) and row['local_point_line_refinement']==local,'local point-line consensus meaning mismatch')
            require(row['point_line_rank']==(solver.get('rank') if local else None) and row['retained_line_edges']==(solver.get('line_edges',[]) if local else []),'posthoc retained line/rank metadata mismatch')
            dimensions=(pred.get('actual_pose') or {}).get('cf_extents') or pred.get('xyz') or (baseline.get('actual_pose') or {}).get('cf_extents') or self.frames[fid]['xyz']
            model=cube(dimensions);same(row['correct_pool_layout_m'],[model[i] for i in sorted(pool&correct)],'correct correspondence actual3D layout',atol=0.,rtol=0.)
            require(row['solver_local_jacobian']==solver.get('geometry',{}).get('jacobian') and row['normalized_jacobian_condition']==(solver.get('geometry',{}).get('jacobian') or {}).get('condition_number'),'posthoc local conditioning metadata join mismatch')
            for flag in ('new_pose_estimated','fallback_used','output_status'):require(row[flag]==pred[flag],'posthoc pose status mismatch')
            require(row['no_pose']==(not pred['pose']['available']),'posthoc complete failure mismatch')
            for field in ('translation_cm','rotation_deg','ADDsym_m'):close(row[field],pred['pose'].get(field),'posthoc pose metric')
            delta=[pred['pose'][f]-baseline['pose'][f] if pred['pose']['available'] and baseline['pose']['available'] else None for f in ('translation_cm','rotation_deg')]
            for d,f in zip(delta,('translation_delta_cm','rotation_delta_deg')):close(row[f],d,'posthoc frozen coordinate baseline delta')
            outcome=('no_pose' if not pred['pose']['available'] else 'fallback' if not pred['new_pose_estimated'] else
                     'both_improved' if all(d is not None and d < -1e-9 for d in delta) else
                     'both_worsened' if all(d is not None and d > 1e-9 for d in delta) else 'mixed_or_equal')
            require(row['paired_pose_outcome_vs_same_coordinate_initial']==outcome,'mask/pose outcome separated incorrectly')
        damage_distributions=0
        def subset(stored,rr):
            expected=dict(frames=len(rr),new_pose=sum(r['new_pose_estimated'] for r in rr),fallback=sum(r['fallback_used'] for r in rr),
                          no_pose=sum(r['no_pose'] for r in rr),pose_outcomes=dict(Counter(r['paired_pose_outcome_vs_same_coordinate_initial'] for r in rr)),
                          correct_pool_count_histogram=dict(Counter(str(r['correct_pool_count']) for r in rr)),
                          correct_final_inlier_count_histogram=dict(Counter(str(r['correct_final_inlier_count']) for r in rr)),
                          direct_correct_pool_count_histogram=dict(Counter(str(len(r['human_direct_correct_pool_ids'])) for r in rr)))
            for key,value in expected.items():require(stored[key]==value,'posthoc subset count '+key)
            for field in ('translation_cm','rotation_deg'):check_distribution(stored[field],[r[field] for r in rr if r[field] is not None],'posthoc subset '+field)
        for stratum,block in metric['strata'].items():
            for method in methods:
                rr=[records[method,i] for i in self.data['strata_ids'][stratum]];stored=block['posthoc_correspondence'][method]
                subset(stored['all'],rr)
                subset(stored['correct_pool_lt4'],[r for r in rr if r['correct_pool_count']<4]);subset(stored['correct_pool_ge4'],[r for r in rr if r['correct_pool_count']>=4])
                require(stored['correct_pool_ge4_object_rank_counts']==dict(Counter(str(r['correct_pool_shape_rank']) for r in rr if r['correct_pool_count']>=4)),'correct-pool saved-rank group counts mismatch')
                for field,rowfield in (('false_excluded_direct_total','false_excluded_direct_ids'),('false_excluded_accurate_total','false_excluded_accurate_ids'),('final_reference_inaccurate_inliers_total','wrong_final_inlier_ids'),('final_unknown_inliers_total','unknown_final_inlier_ids')):
                    require(stored[field]==sum(len(r[rowfield]) for r in rr),'posthoc error subset total mismatch')
                require(stored['point_PnP_inliers_applicable']==all(r['point_PnP_inliers_applicable'] for r in rr),'inlier applicability summary mismatch')
                relation=stored['mask_known_relation']
                if rr[0]['mask_applied']:
                    subset(relation['wrong_mask'],[r for r in rr if r['mask_wrong_on_known']]);subset(relation['matching_known_human_self_states'],[r for r in rr if not r['mask_wrong_on_known']])
                else:require(relation['applicable'] is False,'no-mask mismatch falsely reported as classifier error')
                for scope,summary in block['visibility_damage'][method].items():
                    pairs=defaultdict(list);frame_means=defaultdict(list)
                    for row in rr:
                        if scope=='new_pose' and not row['new_pose_estimated']:continue
                        local_pairs=defaultdict(list)
                        for i in row['paired_corner_valid_ids']:
                            categories=[row['human_states_native'][i]]
                            if i in row['hidden_reprojected_ids']:categories.append('ALGORITHM_REPROJECTED_IDS')
                            if i in row['false_excluded_direct_ids']:categories.append('DIRECT_VISIBLE_FALSE_EXCLUDED')
                            pair=row['reference_error_initial_native_px'][i],row['reference_error_output_native_px'][i]
                            for category in categories:pairs[category].append(pair);local_pairs[category].append(pair)
                        for category,values in local_pairs.items():frame_means[category].append((math.fsum(a for a,b in values)/len(values),math.fsum(b for a,b in values)/len(values)))
                    require(set(summary)==set(pairs),'visibility damage category coverage mismatch')
                    for category,values in pairs.items():
                        before=[a for a,b in values];after=[b for a,b in values];stored=summary[category]
                        require(stored['corners']==len(values) and stored['frames']==len(frame_means[category]),'DIRECT/self/H point or paired-frame denominator mismatch')
                        for name,values_ in (('before',before),('after',after),('paired_delta',[b-a for a,b in values]),('before_frame_mean',[a for a,b in frame_means[category]]),('after_frame_mean',[b for a,b in frame_means[category]])):
                            check_distribution(stored[name],values_,'DIRECT/self/H '+category+'/'+name);damage_distributions+=1
                        expected=dict(improved=sum(b<a-1e-9 for a,b in values),worsened=sum(b>a+1e-9 for a,b in values),
                                      good5_to_bad10=sum(a<5 and b>10 for a,b in values),bad20_to_good10=sum(a>20 and b<=10 for a,b in values))
                        for key,value in expected.items():require(stored[key]==value,'DIRECT/self/H damage counter mismatch')
        require(metric['posthoc_correspondence']==metric['strata']['combined']['posthoc_correspondence'] and metric['visibility_damage']==metric['strata']['combined']['visibility_damage'],'topline visibility summary differs')
        receipt=self.json(self.doc/'STATISTICS_RECEIPT.json')
        for item in receipt['inputs']+receipt['outputs']+[receipt['script']]:self.bind(item)
        require(receipt['GT_inputs_opened_after_seal'] and receipt['eligible_ids_identical_all24methods'] and receipt['geometry_raw_fields_unchanged_in_scored'],'statistics scope/order mismatch')
        require(receipt['rows']==dict(new_scored=self.n*6,posthoc=self.n*24,historical_filtered=self.n*18,historical_stress=self.n*2),'statistics raw population receipt mismatch')
        require(all(receipt[k]==0 for k in ('new_detector_head_calls','new_PnP_optimizer_calls','new_training_updates','new_RGB_or_ray_renders','new_bootstrap_draws')),'statistics receipt new scientific work')
        self.data['posthoc']=records
        self.counts['posthoc_coordinate_error_pairs_replayed']+=pairs_replayed;self.counts['visibility_damage_distributions_replayed']+=damage_distributions
        return dict(rows=len(rows),stored_proxy_coordinate_error_pairs=pairs_replayed,damage_distributions=damage_distributions,
                    DIRECT_self_hidden_reprojection_separate=True,solver_pool_final_inlier_join=True,
                    wrong_mask_good_pose_and_matching_mask_bad_pose_separate=True,reference_authenticity_certified=False,
                    saved_rank_values_joined_and_grouped=True,rank_SVD_not_recomputed=True)

    def source_curves(self):
        curves=self.json(self.doc/'SOURCE_CURVES.json');logs=list(self.rows(self.artifact('TRAIN_LOGS.jsonl')))
        formal=[r for r in logs if r['kind']=='formal'];probes=[r for r in logs if r['kind']=='source_curve']
        require(curves['complete'] and curves['formal_updates']==9000 and curves['formal_exposures']==144000 and curves['seed']==1 and curves['model_selection'] is False,'source curve cost/selection contract mismatch')
        require(curves['formal_logged_rows']==formal and curves['source_curves']==probes and len(probes)==9,'source curve saved log join mismatch')
        require(curves['initial_probes']=={r['arm']:r['initial_probe'] for r in self.data['completion']['checkpoints']},'initial zero-update probe summaries changed')
        require(curves['final_source_test']=={r['arm']:r['source_test'] for r in probes if r['step']==3000},'last source-test log selected differently')
        return dict(initial_probes=3,fixed_source_curve_records=9,model_selection=False,
                    source_coordinate_error_conditional_and_not_real_pose_accuracy=True,forward_reruns=0)

    def N3_supplements(self):
        mask=self.json(self.doc/'MASK_OUTCOME_N3.json');corners=self.json(self.doc/'CORNER_N3_AND_REPROJECTED_SELF.json')
        phase=self.json(self.doc/'PHASE_ALIGNMENT_AUDIT.json');post=self.data['posthoc'];lookup=self.data['metric_lookup']
        for item in mask['bindings']+corners['bindings']:self.bind(item)
        require(mask['complete'] and mask['comparator']=='N3_SUBPIX' and mask['status_and_pose_quality_separate'] and mask['new_detector_head_PnP_ray_calls']==0,'supplement mask comparator/cost mismatch')
        require(corners['complete'] and corners['new_models_fits_rays_truth_reads']==0,'supplement corner cost/scope mismatch')
        methods={'CORRECTED_'+m for m in METHODS};rows=mask['rows'];record={}
        require(len(rows)==self.n*6,'supplement mask row population mismatch')
        for row in rows:
            key=row['method'],row['id'];require(key not in record and key in post and row['method'] in methods,'supplement mask identity mismatch')
            record[key]=row;ar=post[key];pred=lookup[key];n3=lookup['N3_SUBPIX',row['id']]
            for field in ('label','mask_applied','mask_wrong_on_known','no_pose'):require(row[field]==ar[field],'supplement mask/human scope join mismatch')
            require(row['raw_status']==pred['output_status'] and row['new_pose']==pred['new_pose_estimated'] and row['fallback']==pred['fallback_used'],'supplement statuses changed')
            delta=[pred['pose'][f]-n3['pose'][f] if pred['pose']['available'] and n3['pose']['available'] else None for f in ('translation_cm','rotation_deg')]
            for field,v in zip(('translation_delta_cm','rotation_delta_deg'),delta):close(row[field],v,'supplement delta vs fixedN3')
            quality='no_common_pose' if any(v is None for v in delta) else 'both_improved' if all(v < -1e-9 for v in delta) else 'both_worsened' if all(v > 1e-9 for v in delta) else 'mixed_or_equal'
            require(row['operational_pose_quality_vs_N3']==quality,'supplement mask/pose quality category mismatch')
        different=[]
        for fid in self.ids:
            base=post['BASE',fid]['permutation_native_to_canonical'];n3=post['N3_SUBPIX',fid]['permutation_native_to_canonical']
            if base!=n3:different.append(dict(id=fid,BASE_phase=base,N3_phase=n3))
        require(phase['complete'] and phase['same_phase_frames']==self.n-len(different) and phase['different_phase_frames']==len(different) and phase['different_phase_ids_and_values']==different,'BASE/N3 phase alignment audit mismatch')
        require(phase['source']['sha256']==sha(self.doc/'POSTHOC_CORRESPONDENCE_ROWS.jsonl.gz') and phase['new_scientific_calls']==0,'phase audit source/cost mismatch')
        distributions=missing=0
        for stratum,ids in self.data['strata_ids'].items():
            for method in methods:
                rr=[record[method,i] for i in ids];groups={'all':rr}
                if rr[0]['mask_applied']:
                    groups.update(wrong_mask=[r for r in rr if r['mask_wrong_on_known']],matching_known_mask=[r for r in rr if not r['mask_wrong_on_known']])
                stored=mask['strata'][stratum].get(method)
                require((stored is not None)==rr[0]['mask_applied'],'supplement no-mask group applicability mismatch')
                if stored is not None:require(set(stored)==set(groups),'supplement mask group coverage mismatch')
                for name,values in (groups.items() if stored is not None else []):
                    expected=dict(frames=len(values),new_pose=sum(r['new_pose'] for r in values),fallback=sum(r['fallback'] for r in values),no_pose=sum(r['no_pose'] for r in values),
                                  quality_operational=dict(Counter(r['operational_pose_quality_vs_N3'] for r in values)),
                                  quality_new_only=dict(Counter(r['operational_pose_quality_vs_N3'] for r in values if r['new_pose'])),
                                  quality_fallback_only=dict(Counter(r['operational_pose_quality_vs_N3'] for r in values if r['fallback'])))
                    require(stored[name]==expected,'supplement wrong-mask/matching-mask quality counts mismatch')
                pairs=defaultdict(list);frame_pairs=defaultdict(list);counts=Counter();replaced=[]
                for fid in ids:
                    ar=post[method,fid];ref=ar['reference_native_points_px'];valid=set(ar['reference_valid_native_ids'])
                    before=errors(lookup['N3_SUBPIX',fid]['native_points'],ref,valid);after=ar['reference_error_output_native_px'];base=errors(ar['frozen_BASE_native_points'],ref,valid)
                    counts['frames_same_BASE_and_N3_reference_phase']+=post['N3_SUBPIX',fid]['permutation_native_to_canonical']==ar['permutation_native_to_canonical']
                    local_pairs=defaultdict(list)
                    for i,category in enumerate(ar['human_states_native']):
                        if before[i] is not None and after[i] is not None:pairs[category].append((before[i],after[i]));local_pairs[category].append((before[i],after[i]))
                        if category!='SELF_OCCLUDED':continue
                        counts['human_SELF_ids']+=1
                        if not ar['new_pose_estimated']:counts['SELF_without_new_pose']+=1;continue
                        if i not in ar['hidden_reprojected_ids']:counts['SELF_not_reprojected_despite_new_pose']+=1;continue
                        counts['SELF_reprojected_after_new_pose']+=1
                        if any(x is None for x in (base[i],before[i],after[i])):counts['SELF_reprojected_missing_valid_reference_or_coordinates']+=1;missing+=1;continue
                        replaced.append((base[i],before[i],after[i]))
                    for category,values in local_pairs.items():frame_pairs[category].append((math.fsum(a for a,b in values)/len(values),math.fsum(b for a,b in values)/len(values)))
                saved=corners['visibility_damage_vs_N3'][stratum][method];require(set(saved)==set(pairs),'N3 sameBASE-phase damage categories differ')
                for category,values in pairs.items():
                    s=saved[category];require(s['corners']==len(values) and s['frames']==len(frame_pairs[category]),'N3 samephase paired denominator differs')
                    for name,vs in (('before_N3',[a for a,b in values]),('after',[b for a,b in values]),('paired_delta_vs_N3',[b-a for a,b in values]),('before_N3_frame_mean',[a for a,b in frame_pairs[category]]),('after_frame_mean',[b for a,b in frame_pairs[category]])):
                        check_distribution(s[name],vs,'N3 samephase damage '+name);distributions+=1
                    for name,value in dict(improved=sum(b<a-1e-9 for a,b in values),worsened=sum(b>a+1e-9 for a,b in values),good5_to_bad10=sum(a<5 and b>10 for a,b in values),bad20_to_good10=sum(a>20 and b<=10 for a,b in values)).items():require(s[name]==value,'N3 samephase damage counts mismatch')
                s=corners['actual_reprojected_SELF_only'][stratum][method]
                require(s['counts']==dict(counts) and s['evaluable_reprojected_SELF_corners']==len(replaced),'actual newly reprojectedSELF scope count mismatch')
                for name,vs in (('before_BASE',[a for a,b,c in replaced]),('before_N3_same_BASE_phase',[b for a,b,c in replaced]),('after_reprojection',[c for a,b,c in replaced]),('delta_vs_BASE',[c-a for a,b,c in replaced]),('delta_vs_N3',[c-b for a,b,c in replaced])):
                    check_distribution(s[name],vs,'actual reprojectedSELF '+name);distributions+=1
        require(phase['strict_SELF_missing_valid_reference_or_coordinates']==sum(corners['actual_reprojected_SELF_only']['combined'][m]['counts'].get('SELF_reprojected_missing_valid_reference_or_coordinates',0) for m in methods),'phase audit strictSELF missing counter mismatch')
        self.counts['N3_supplement_distributions_replayed']+=distributions
        return dict(mask_vs_N3_rows=len(rows),BASE_N3_same_phase=self.n-len(different),BASE_N3_different_phase=len(different),
                    distributions=distributions,strict_new_SELF_reprojection_scope_checked=True,
                    N3_corner_errors_use_fixedBASEphase=True,no_best_branch_reselection=True)

    def real_stress(self):
        protocol=self.json(self.doc/'REAL_STRESS_PROTOCOL.json');execution=self.json(self.doc/'REAL_STRESS_EXECUTION.json')
        self.bind(execution['protocol']);self.bind(execution['raw_rows'])
        for item in protocol['fixed_inputs'].values():self.bind(item)
        families=['HUMAN_NOSELF','KNOWN_ACCURATE_OBSERVATIONS'];conditions=[('ANCHOR',0,0),('DROP_ONE_GOOD',1,0),('DROP_TWO_GOOD',2,0),('RETAIN_ONE_BAD',0,1),('RETAIN_TWO_BAD',0,2),('BOTH_ONE_ONE',1,1),('BOTH_TWO_TWO',2,2)]
        methods={a+'::'+f+'::'+c for a in ('BASE','N3_SUBPIX') for f in families for c,d,k in conditions}
        require(protocol['conditions']==[dict(name=c,drop_good=d,retain_bad=k) for c,d,k in conditions] and protocol['families']==families and protocol['residual_threshold_px']==8. and protocol['maximum_refinement_starts']==3,'fixed real stress selection/settings differ')
        require(execution['complete'] and execution['frames']==self.n and execution['rows']==self.n*28 and execution['banks']==self.n*2 and execution['protected_prior_files_preserved']==331,'real stress actual population mismatch')
        require(execution['nontransformable_not_pose_failure'] and execution['oracle_diagnostic_not_deployment'] and all(execution[k]==0 for k in ('new_detector_forwards','new_head_forwards','new_training_updates','new_RGB')),'oracle stress scope/cost mismatch')
        lookup=self.population(self.rows(self.doc/'REAL_STRESS_ROWS.jsonl.gz'),methods)
        audits={(r['method'],r['id']):r for r in self.rows(self.old/'REAL_CORRESPONDENCE_ROWS.jsonl.gz')}
        poses={(r['method'],r['id']):r['initial_pose'] for r in self.rows(self.old/'POSE_DIAGNOSTICS.jsonl.gz') if r['method'] in ('BASE_NO_MASK_STANDARD','N3_SUBPIX_NO_MASK_STANDARD')}
        ledgers=Counter();yields=defaultdict(Counter);nontransformable=Counter();reprojected=0
        by_bank={(r['coordinate'],r['id']):r for r in execution['bank_generation']}
        require(len(by_bank)==self.n*2 and all((a,i) in by_bank for a in ('BASE','N3_SUBPIX') for i in self.ids),'real stress coordinate bank population mismatch')
        for (arm,fid),bank in by_bank.items():
            frame=self.frames[fid];q=frame['points'][arm];eligible={i for i in range(8) if in_frame(q[i],frame['raw_hw'])}
            require(bank['eligible']==sorted(eligible),'real stress bank eligibility differs')
            raw=[v for p in q for v in p]+[v for p in frame['K'] for v in p]+frame['xyz']
            digest=hashlib.sha256(struct.pack('<'+str(len(raw))+'d',*raw)+str((frame['raw_hw'][1],frame['raw_hw'][0])).encode()).hexdigest()
            require(bank['input_hash']==digest,'shared original coordinate/K/dimension bank hash mismatch')
            dimensions=1 if abs(frame['xyz'][0]-frame['xyz'][2])<1e-9 else 2
            subset_count=math.comb(len(eligible),4)*dimensions if len(eligible)>=4 else 0
            require(bank['counts']['subset_generic_calls']==bank['counts']['subsets_considered'] and bank['counts']['subsets_considered'] in (0,subset_count) and subset_count<=140,'finite4-ID generators were rebuilt per mask or partially repeated')
            mask_ledger=Counter()
            for family in families:
                for c,d,k in conditions:mask_ledger.update(lookup[arm+'::'+family+'::'+c,fid]['solver']['operation_counts'])
            require(dict(mask_ledger)==bank['counts'],'same-bank per-mask operation delta ledger does not sum to single coordinate bank')
            ledgers.update(bank['counts']);a=audits[arm+'_NO_MASK_STANDARD',fid]
            valid=set(a['reference_valid_native_ids']) if a['reference_matched'] else set();err=a['reference_error_input_native_px']
            correct={i for i in valid if finite(err[i]) and err[i]<=8.};bad={i for i in valid if finite(err[i]) and err[i]>8.}
            states=a['human_states_native'];H={i for i,s in enumerate(states) if s=='SELF_OCCLUDED'};good=eligible&correct&{i for i,s in enumerate(states) if s=='DIRECT_VISIBLE'}
            for family in families:
                anchor=H if family=='HUMAN_NOSELF' else set(range(8))-good
                good_pool=good-anchor;bad_pool=eligible&H if family=='HUMAN_NOSELF' else eligible&bad
                choose=lambda role,pool,count:sorted(pool,key=lambda i:hashlib.sha256((fid+':'+arm+':'+family+':'+role+':'+str(i)).encode()).hexdigest())[:count]
                for condition,d,k in conditions:
                    row=lookup[arm+'::'+family+'::'+condition,fid];solver=row['solver']
                    require(row['coordinate']==arm and row['family']==family and row['condition']==condition and row['requested_drop']==d and row['requested_retain']==k,'real stress condition identity mismatch')
                    for field in ('human_states_native','permutation_native_to_canonical','reference_error_input_native_px','reference_matched'):
                        require(row[field]==a[field],'real stress frozen reference/human phase changed')
                    require(row['reference_valid_native_ids']==sorted(valid) and row['accurate_threshold_px']==8. and row['oracle'] and row['oracle_diagnostic_not_deployment'],'real stress reference/oracle scope changed')
                    require(row['initial_coordinates_changed'] is False and row['GT_pose_or_coordinates_input_to_solver'] is False and row['input_points']==q and row['K']==frame['K'] and row['xyz']==frame['xyz'],'real stress injected coordinates or GT into solver')
                    possible=len(good_pool)>=d and len(bad_pool)>=k;drop=choose('drop',good_pool,d) if possible else [];keep=choose('keep',bad_pool,k) if possible else []
                    excluded=(anchor|set(drop))-set(keep);hidden_ids=sorted(H&excluded)
                    require(row['transformation_possible']==possible and row['drop_good_ids']==drop and row['retain_bad_ids']==keep and row['anchor_excluded']==sorted(anchor),'fixed nested SHA-selected error perturbation differs')
                    require(row['nontransformable_reason']==(None if possible else dict(good_candidates=len(good_pool),bad_candidates=len(bad_pool))),'nontransformable cause mismatch')
                    require(row['excluded']==solver['excluded']==sorted(excluded) and row['hidden_initial']==solver['hidden']==hidden_ids,'oracle fit exclusion/reprojection H mismatch')
                    require(solver['input_hash']==digest and solver['used']==sorted(eligible-excluded) and set(solver.get('fit_input_ids',[])).isdisjoint(excluded),'masked coordinates included in final fit or bankchanged')
                    require(row['reprojections_reused_as_observations'] is False and solver.get('reprojected_points_reused_as_observations',False) is False,'real stress refit reused final projections')
                    pool=set(solver['used']);inliers=set(solver['inliers']);require(inliers<=pool,'real stress inliers outside pool')
                    for prefix,ids in (('pool',pool),('final_inlier',inliers)):
                        require(row['correct_'+prefix+'_ids']==sorted(ids&correct) and row['wrong_'+prefix+'_ids']==sorted(ids&bad) and row['unknown_'+prefix+'_ids']==sorted(ids-valid),'real stress known accurate/wrong/unknown partition differs')
                    require(row['correct_pool_count']==len(pool&correct) and row['remaining_direct_correct_ids']==sorted(pool&good),'real stress remaining correct/DIRECT count differs')
                    for i in keep:
                        require(row['retained_bad_reasons'][str(i)]==dict(human_state=states[i],self_observation_invalid=i in H,coordinate_outlier_over8px=i in bad,coordinate_accurate_at8px=i in correct,reference_error_px=err[i]),'SELF invalidity conflated with coordinate error')
                    require(set(row['retained_bad_reasons'])=={str(i) for i in keep},'retained bad reason population differs')
                    require(row['initial_pose']==poses[arm+'_NO_MASK_STANDARD',fid],'real stress initial posechanged')
                    new=solver['available'];fallback=not new and row['initial_pose']['available']
                    require(row['new_pose_estimated']==new and row['fallback_used']==fallback and row['no_pose']==(not row['pose_available']) and row['pose']['available']==row['pose_available'],'real stress status/failure flags differ')
                    require(row['output_status']==('NEW_POSE' if new else 'BASELINE_FALLBACK' if fallback else 'POSE_FAILURE'),'real stress operational outcome differs')
                    require(row['reprojected_ids']==(hidden_ids if new else []),'real stress H replace flags differ')
                    expected=[list(p) for p in q]
                    if new:
                        projection=project(solver,frame['K']);same(solver['projected'],projection,'real stress saved pose projection',atol=1e-6)
                        for i in hidden_ids:expected[i]=projection[i];reprojected+=1
                        for field in ('R_cf','R_physical','centroid','cf_extents'):same(row['actual_pose'][field],solver[field],'real stress finalpose join',atol=0.,rtol=0.)
                    else:require(row['actual_pose']==row['initial_pose'],'real stress full fallback posechanged')
                    same(row['native_points'],expected,'real stress H-only replacement/full fallback coordinates',atol=1e-6)
                    require(row['corresponding_simple_pose']==self.controls[arm,fid]['pose'] and row['fixed_N3_SUBPIX_simple_pose']==self.controls['N3_SUBPIX',fid]['pose'],'real stress comparator changed')
                    delta={f:row['pose'][f]-row['corresponding_simple_pose'][f] if row['pose']['available'] and row['corresponding_simple_pose']['available'] else None for f in ('translation_cm','rotation_deg','ADDsym_m')}
                    same(row['delta_vs_corresponding_simple'],delta,'real stress corresponding baseline delta',atol=1e-9)
                    wrong_mask=bool((excluded^H)&{i for i,s in enumerate(states) if s!='UNANNOTATED'});t,r=delta['translation_cm'],delta['rotation_deg']
                    improved=t is not None and t<0 and r<0;worsened=t is not None and (t>0 or r>0)
                    outcome='mask_wrong_pose_improved' if wrong_mask and improved else 'mask_correct_pose_worse' if not wrong_mask and worsened else 'other_or_mixed'
                    require(row['mask_wrong_on_known']==wrong_mask and row['mask_pose_outcome']==outcome,'real stress mask versus pose outcome conflated')
                    yields[row['method']].update(rows=1,new_pose=int(new),fallback=int(fallback),failure=int(row['no_pose']),transformable=int(possible),nontransformable=int(not possible))
                    nontransformable[family]+=not possible
        require(dict(ledgers)==execution['counts'] and {k:dict(v) for k,v in yields.items()}==execution['condition_yields'],'real stress bank/yield ledger mismatch')
        require(execution['actual_OpenCV_counts']['real']==dict(solvePnPGeneric=ledgers['generic_calls'],solvePnPRefineLM=ledgers['lm_calls']) and execution['actual_OpenCV_counts']['unit_checks']=={},'real stress actual CV primitives differ from saved bankledger')
        self.data['stress_rows']=lookup;self.counts['real_mask_stress_row_replays']+=len(lookup);self.counts['hidden_corner_projection_replays']+=reprojected
        return dict(rows=len(lookup),coordinate_banks=len(by_bank),masks_per_bank=14,nontransformable=dict(nontransformable),
                    finite4_ID_generators_reused_between_masks=True,correct_wrong_unknown_correspondences_separate=True,
                    retained_SELF_not_assumed_coordinate_wrong=True,hidden_actual_projection_replays=reprojected,
                    real_pose_metrics_saved_proxy_only=True,SVD_rank_not_recomputed=True,new_pose_fits=0)

    def stress_statistics(self):
        book=self.json(self.doc/'REAL_STRESS_STATISTICS.json');lookup=self.data['stress_rows'];methods={k[0] for k in lookup};checked=0
        for item in book['bindings']:self.bind(item)
        require(book['complete'] and book['rows']==self.n*28 and book['frames']==self.n and book['new_statistical_pose_calls']==0 and book['oracle_diagnostic_not_deployment'] and book['absent_transformations_not_counted_as_solver_failures'],'stress aggregate population/cost/scope mismatch')
        require(book['nontransformable_total']==sum(not r['transformation_possible'] for r in lookup.values()) and book['family_nontransformable']=={f:sum(not r['transformation_possible'] for r in lookup.values() if r['family']==f) for f in book['families']},'stress nontransformable aggregate mismatch')
        def quality(r):
            if not r['pose']['available']:return 'no_pose'
            t,rotation=[r['delta_vs_corresponding_simple'][k] for k in ('translation_cm','rotation_deg')]
            if t is None or rotation is None:return 'no_common_comparator'
            return 'both_improved' if t < -1e-9 and rotation < -1e-9 else 'both_worsened' if t > 1e-9 and rotation > 1e-9 else 'mixed_or_equal'
        def aggregate(stored,rr):
            nonlocal checked
            pose=stored['pose'];available=[r for r in rr if r['pose']['available']];fresh=[r for r in available if r['new_pose_estimated']]
            counts=dict(total_frames=len(rr),denominator=len(rr),pose_available=len(available),new_pose_estimated=len(fresh),fallback_used=sum(r['fallback_used'] for r in rr),no_pose=len(rr)-len(available),fixed_control_outputs=0,
                        hidden_reprojected=sum(r['hidden_reprojected'] for r in rr),hidden_set_changed=sum(r['hidden_set_changed'] for r in rr),
                        available_ids=[r['id'] for r in available],new_pose_ids=[r['id'] for r in fresh],fallback_ids=[r['id'] for r in rr if r['fallback_used']],no_pose_ids=[r['id'] for r in rr if not r['pose']['available']],
                        output_status_counts=dict(Counter(r['output_status'] for r in rr)),failure_reasons=dict(Counter(r['solver']['reason'] for r in rr if not r['new_pose_estimated'])))
            for key,value in counts.items():require(pose[key]==value,'stress aggregate operational/status '+key)
            for scope,values in (('operational',available),('new_pose',fresh)):
                for name,(field,factor,unit) in METRICS.items():check_distribution(pose['metrics'][scope][name],[r['pose'][field]*factor for r in values],'stress operational/new moments '+name);checked+=1
            histograms={'correct_pool_count_histogram':[str(r['correct_pool_count']) for r in rr],
                        'direct_correct_pool_count_histogram':[str(len(r['remaining_direct_correct_ids'])) for r in rr],
                        'correct_final_inlier_count_histogram':[str(len(r['correct_final_inlier_ids'])) for r in rr],
                        'wrong_final_inlier_count_histogram':[str(len(r['wrong_final_inlier_ids'])) for r in rr],
                        'correct_pool_registry_rank_counts':[str(r['correct_pool_registry_shape']['numerical_rank']) for r in rr],
                        'correct_pool_image_rank_counts':[str(r['correct_pool_image_shape']['numerical_rank']) for r in rr],
                        'pose_quality_operational':[quality(r) for r in rr],'pose_quality_new_only':[quality(r) for r in rr if r['new_pose_estimated']]}
            for field,values in histograms.items():require(stored[field]==dict(Counter(values)),'stress correct-point arrangement/inlier/quality distribution '+field)
            for name,vs in (('mean_correct_pool',[r['correct_pool_count'] for r in rr]),('mean_wrong_pool',[len(r['wrong_pool_ids']) for r in rr]),('mean_correct_final_inliers',[len(r['correct_final_inlier_ids']) for r in rr]),('mean_wrong_final_inliers',[len(r['wrong_final_inlier_ids']) for r in rr])):
                close(stored[name],math.fsum(vs)/len(vs) if vs else None,'stress pointcount mean')
            require(stored['correct_pool_ge4']==sum(r['correct_pool_count']>=4 for r in rr),'stress >=4 correct-pool count differs')
        for stratum,ids in self.data['strata_ids'].items():
            block=book['strata'][stratum];require(set(block)==methods,'stress all28 condition coverage differs')
            for method in methods:
                rr=[lookup[method,i] for i in ids];stored=block[method]
                possible=[r for r in rr if r['transformation_possible']];non=[r for r in rr if not r['transformation_possible']]
                require(stored['rows']==len(rr) and stored['transformation_possible']==len(possible) and stored['nontransformable']==len(non),'stress transformed/nontransformed scope count differs')
                require(stored['nontransformable_reasons']==dict(Counter(json.dumps(r['nontransformable_reason'],sort_keys=True,separators=(',',':')) for r in non)),'stress nontransformable reasons differ')
                for field,values in (('all_operational',rr),('transformable_only',possible),('nontransformable_anchor_retained',non)):
                    aggregate(stored[field],values)
                for label,wrong in (('wrong_known_mask',True),('matching_known_mask',False)):aggregate(stored['mask_groups'][label],[r for r in rr if r['mask_wrong_on_known']==wrong])
        self.counts['real_mask_stress_distributions_replayed']+=checked
        return dict(rows=self.n*28,condition_groups=len(methods),strata=3,operational_and_transformable_moments=checked,
                    nontransformable_anchor_retained_separate=True,mask_and_final_pose_quality_separate=True,
                    stored_arrangement_rank_histograms_checked=True,SVD_ranks_not_recomputed=True)

    def runtime(self):
        if not (self.doc/'RUNTIME.json').is_file():
            require(self.allow_pending_runtime,'actual600 runtime records missing')
            return dict(pending=True,complete=False,reason='Actual fresh600 timing has not yet been published; accuracy-only review does not certify latency.')
        runtime=self.json(self.doc/'RUNTIME.json');rows=list(self.rows(self.doc/'RUNTIME_ROWS.jsonl.gz'))
        arms=['BASE','N3_SUBPIX','N3_SUBPIX_GEOM_NOSELF_ROBUST','IMAGE_ROLE']
        require(runtime['arms']==arms and runtime['complete'] and runtime['status']=='DONE' and len(rows)==600,'fresh original four-arm600 runtime incomplete')
        require(runtime['frames']==26 and runtime['panel_sessions']==13 and runtime['warmup_per_arm']==20 and runtime['repeats']==5 and runtime['measured_per_arm']==130 and runtime['max_pipeline_calls']==600,'runtime schedule budget differs')
        require(runtime['evaluation_scope']=='easy_medium_only' and runtime['original_timed_pipeline_body_unchanged'],'runtime route or user evaluation scope changed')
        self.bind(runtime['cohort']);self.bind(runtime['panel_selection']);self.bind(runtime['metadata_scope_adapter']);self.bind(runtime['runtime_code']);self.bind(runtime['raw_rows'])
        selected=self.data['runtime_panel']['frames'];panel={r['frame_id']:r for r in selected}
        require(runtime['panel']==[dict(id=r['frame_id'],session=r['session_id'],image_key=r['image_key']) for r in selected],'runtime actual panel/order differs from frozen eligible panel')
        diagnostics={(r['method'],r['id']):r for r in self.rows(self.old/'POSE_DIAGNOSTICS.jsonl.gz')}
        counts=Counter();per_arm=defaultdict(Counter);measured=defaultdict(Counter);actual_calls=Counter()
        for row,job in zip(rows,schedule()):
            require(all(row[k]==v for k,v in job.items()),'runtime original600 job order differs')
            require(row['id']==selected[job['image_index']]['frame_id'] and row['session']==panel[row['id']]['session_id'],'runtime eligible image/session identity differs')
            arm=row['arm'];fid=row['id'];per_arm[arm][row['phase']]+=1
            if row['phase']=='measured':measured[arm][fid]+=1
            require(all(row[k] is True for k in ('GT_canary_active','RAW_cached_bitexact','candidate_center_score_box_preserved')) and row['parity_status']=='PASS','runtime canary/fresh detector/candidate parity failed')
            stages=('detector','correction','initial_pose','robust_and_reprojection')
            require(all(finite(row[k+'_ms']) and row[k+'_ms']>=0 for k in ('full',)+stages),'invalid actual timing interval')
            close(row['full_ms'],math.fsum(row[k+'_ms'] for k in stages),'contiguous runtime intervals',atol=1e-5,rtol=0.)
            require(row['final_points_parity']['atol_px']==row['final_pose_parity']['atol']==1e-7 and row['final_points_parity']['max_abs_px']<=1e-7 and row['final_points_parity']['finite_support_equal'],'runtime coordinate saved parity contract mismatch')
            require(all(finite(v) and v<=1e-7 for v in row['final_pose_parity']['max_abs_by_field'].values()),'runtime pose saved parity contract mismatch')
            require(row['learned_initial_pose_in_correction_interval']==(arm=='IMAGE_ROLE'),'fresh feature-role pose outside correction interval')
            frame=self.frames[fid];same(row['raw_points'],frame['points']['BASE'],'fresh raw detector public parity',atol=0.,rtol=0.)
            require(row['selected_index']==frame['selected_index'],'runtime detector candidate changed')
            if arm in ('BASE','N3_SUBPIX'):
                expected_points=frame['points'][arm];expected_pose=diagnostics[arm+'_NO_MASK_STANDARD',fid]['initial_pose']
                require(row['output_status']=='HISTORICAL_FIXED_PATH','fixed runtime output status changed')
            elif arm=='N3_SUBPIX_GEOM_NOSELF_ROBUST':
                expected=diagnostics[arm,fid];expected_points=expected['native_points'];expected_pose=expected['actual_pose']
                require(row['output_status']==expected['output_status'] and row['hidden_initial']==expected['hidden_initial'],'N3 robust runtime status/H parity differs')
            else:
                expected=self.data['sealed']['IMAGE_ROLE',fid];expected_points=expected['native_points'];expected_pose=expected['actual_pose']
                require(row['output_status']==expected['output_status'] and row['hidden_initial']==expected['hidden_initial'],'corrected ROLE runtime status/H parity differs')
            same(row['final_points'],expected_points,'fresh runtime public final coordinate parity',atol=1e-7,rtol=0.)
            require((row['actual_pose'] or {}).get('available')==(expected_pose or {}).get('available'),'runtime final pose availability differs')
            for field in ('R_cf','R_physical','centroid','cf_extents'):
                same((row['actual_pose'] or {}).get(field),(expected_pose or {}).get(field),'fresh runtime final pose parity '+field,atol=1e-7,rtol=0.)
            require(all(type(v) is int and v>=0 for v in row['call_counts'].values()),'runtime call primitive invalid')
            actual_calls.update(row['call_counts']);counts['detector_calls']+=1;counts['initial_pose_calls']+=1
            counts['robust_solver_calls']+=int(arm in arms[2:]);counts['learned_observation_calls']+=int(arm=='IMAGE_ROLE');counts['feature_role_pose_calls']+=int(arm=='IMAGE_ROLE')
        for arm in arms:
            require(per_arm[arm]==Counter(warmup=20,measured=130) and measured[arm]==Counter({fid:5 for fid in panel}),'runtime four-arm/warmup/per-image repeat counts differ')
            for stage in ('full','detector','correction','initial_pose','robust_and_reprojection'):
                values=[r[stage+'_ms'] for r in rows if r['arm']==arm and r['phase']=='measured'];calculated=distribution(values);stored=runtime['summaries'][arm][stage]
                for source,target in dict(n='n',mean='mean_ms',sample_variance='sample_variance_ms2',sample_std='sample_std_ms',median='median_ms',P90='p90_ms').items():
                    close(calculated[source],stored[target],'raw runtime distribution '+arm+'/'+stage+'/'+source)
                require(stored['ddof']==1,'runtime sample variance ddof differs')
        for name,value in (counts+actual_calls).items():require(runtime['execution'][name]==value,'runtime recorded actual call count '+name)
        execution=runtime['execution'];require(execution['pipeline_calls_started']==execution['pipeline_calls_complete']==600 and execution['image_decode_calls']==26 and execution['image_decode_outside_intervals'] and execution['updates']==execution['extra_GT_evaluation_calls']==0,'runtime extra work/full-route count differs')
        forwards=runtime['model_forwards'];require(forwards['detector']-forwards['detector_internal_initialization_calls']==600 and forwards['fixed_N3']==300 and forwards['learned']==150 and forwards['duplicate_N3_backbone']==0,'runtime whole detector/head forward accounting differs')
        require(runtime['GT_inference_access'] is False and runtime['cached_coordinate_replay_used_for_timing'] is False and runtime['other_benchmarks_parallel'] is False,'runtime canary/cache/interference contract mismatch')
        require(runtime['boundaries']['all_initial_and_final_PnP_included'] and runtime['boundaries']['accuracy_cache_replay_used'] is False,'runtime full actual path boundary mismatch')
        last=self.data['checkpoints']['IMAGE_ROLE']['checkpoint'];weights=[b for b in runtime['input_bindings'] if Path(b['path']).name=='IMAGE_ROLE.pt']
        require(len(weights)==1 and weights[0]['sha256']==last['sha256'] and weights[0]['bytes']==last['bytes'],'runtime uses different ROLE weight than actual corrected last checkpoint')
        environment=runtime['environment'];require(environment['thread_contract']==dict(torch=4,opencv=1),'runtime fixed CPU thread contract differs')
        require(all(s['quiet'] and s['gpu_temperature_under_80'] and not s['foreign_gpu_pids'] and not s['foreign_cpu_workloads'] for s in environment['interference_snapshots']),'runtime declared quiet window compromised')
        adapter=self.json(self.doc/'RUNTIME_ADAPTER_RECEIPT.json')
        for item in adapter['fixed_inputs'].values():self.bind(item)
        self.bind(adapter['runtime']);self.bind(adapter['raw_rows'])
        require(adapter['complete'] and adapter['status']=='DONE' and adapter['original_prior_files_preserved']==331 and adapter['old_context_restored'] and adapter['cached_coordinate_timing'] is False and adapter['new_training_updates']==0,'actual corrected runtime adapter receipt mismatch')
        require(adapter['prerequisites']['cohort']['sha256']==sha(self.doc/'COHORT.json'),'runtime prerequisite cohort differs')
        self.counts['runtime_record_distribution_replays']+=20
        return dict(rows=600,warmup_calls=80,measured_calls=520,eligible_frames=26,sessions=13,
                    stored_full_path_distributions_recomputed=20,actual_saved_call_counts=dict(counts+actual_calls),
                    fresh_route_saved_pose_coordinate_parity_checked=True,latency_remeasured=False,
                    transient_hardware_interference_independently_certified=False)

    def manifest(self):
        path=self.doc/'REVIEW_MANIFEST.json'
        if not path.is_file():
            require(not self.require_manifest,'required final publication manifest missing')
            return dict(pending=True,complete=False,reason='Final manifest not yet published.')
        manifest=self.json(path);files=manifest['files'];paths=[f['path'] for f in files]
        require(len(paths)==len(set(paths)),'duplicate manifest path')
        require(all(p not in {str((self.doc/n).relative_to(self.root)) for n in ('REVIEW_CHECKS.json','REVIEW_MANIFEST.json','PUBLICATION.json')} for p in paths),'receipt/manifest hash cycle')
        for item in files:self.bind(item)
        prior={f['path']:f for f in self.json(self.doc/'PRIOR_PUBLICATION_BINDINGS.json')['files']}
        bindings={f['path']:f for f in files};protected=set(prior)
        uncovered=self.used-set(paths)-protected-{str(path.relative_to(self.root))}
        require(not uncovered,'reviewed current input absent from final manifest: '+','.join(sorted(uncovered)))
        return dict(files=len(files),bytes=sum(b['bytes'] for b in files),protected_prior_files=len(protected),
                    every_public_math_input_bound=True,manifest_self_and_generated_receipt_excluded=True)


def output_guard(root,output):
    root=Path(root).resolve();doc=root/'_docs/experiments'/NAME
    output=Path(output).absolute();canonical=doc.resolve()/'REVIEW_CHECKS.json'
    require(not output.is_symlink(),'refuse symlink receipt output')
    target=output.resolve()
    if target.is_relative_to(root):require(target==canonical,'refuse overwriting repository artifact except own REVIEW_CHECKS.json')
    elif target.exists():
        try:existing=json.loads(target.read_text())
        except (OSError,ValueError):raise ValueError('refuse overwriting existing non-review artifact')
        require(existing.get('schema')=='corrected_supervision_public_review_v1' and existing.get('reviewer_kind')=='stdlib_saved_data_arithmetic','refuse overwriting other existing output')
    if target.exists():
        existing=json.loads(target.read_text())
        require(existing.get('schema')=='corrected_supervision_public_review_v1','refuse overwriting foreign receipt')
    target.parent.mkdir(parents=True,exist_ok=True)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3],help='Public repository root; no source environment or private data needed.')
    parser.add_argument('--output',type=Path,help='Own REVIEW_CHECKS.json by default; external new review receipt allowed.')
    parser.add_argument('--require-manifest',action='store_true')
    parser.add_argument('--allow-pending-runtime',action='store_true',help='Accuracy-only development check; missing actual600 timing explicitly remains pending.')
    args=parser.parse_args();root=args.root.resolve();target=output_guard(root,args.output or root/'_docs/experiments'/NAME/'REVIEW_CHECKS.json')
    started=time.monotonic();review=Review(root,args.require_manifest,args.allow_pending_runtime);groups=[]
    for name in ('protected','training','cohort_selection','observations','geometry','scoring_resume','metrics','bootstrap','posthoc','N3_supplements','real_stress','stress_statistics','source_curves','runtime','manifest'):
        try:
            detail=getattr(review,name)();status='PENDING' if detail.get('pending') else 'PASS'
            groups.append(dict(check=name,status=status,details=detail))
        except Exception as exc:
            groups.append(dict(check=name,status='FAIL',error=type(exc).__name__+': '+str(exc)));break
    failed=any(g['status']=='FAIL' for g in groups);pending=any(g['status']=='PENDING' for g in groups)
    result=dict(schema='corrected_supervision_public_review_v1',reviewer_kind='stdlib_saved_data_arithmetic',
                status='FAIL' if failed else 'PARTIAL' if pending else 'PASS',complete=not(failed or pending),
                require_manifest=args.require_manifest,allow_pending_runtime=args.allow_pending_runtime,checks=groups,
                verifier=dict(path=str(Path(__file__).resolve().relative_to(root)) if Path(__file__).resolve().is_relative_to(root) else Path(__file__).name,
                              sha256=sha(Path(__file__)),bytes=Path(__file__).stat().st_size),
                inputs=[dict(path=p,sha256=sha(root/p),bytes=(root/p).stat().st_size) for p in sorted(review.used)],
                mathematical_replay_counts=dict(review.counts),actual_execution_in_this_verifier=dict(detector=0,heads=0,
                    optimizer_updates=0,pose_fits=0,PnP=0,rays=0,private_GT_reads=0,timing_intervals=0),
                limits=LIMITS,wall_seconds=time.monotonic()-started)
    print(json.dumps(dict(status=result['status'],checks={g['check']:g['status'] for g in groups},output=str(target)),ensure_ascii=False))
    temporary=None
    try:
        with tempfile.NamedTemporaryFile('w',dir=target.parent,prefix='.'+target.name+'.',suffix='.pending',delete=False,encoding='utf-8') as stream:
            temporary=Path(stream.name);json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,target)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()
    return 1 if failed else 0


if __name__=='__main__':raise SystemExit(main())
