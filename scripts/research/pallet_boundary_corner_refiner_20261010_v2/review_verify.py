"""Independent standard-library arithmetic review of saved boundary-refiner rows.

Python >=3.9. No experiment imports, private GT, model, PnP, optimizer or rays.
Use --input for the published result directory or the isolated execution output.
Existing receipts are never overwritten.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

sys.dont_write_bytecode = True
NAME = 'pallet_boundary_corner_refiner_20261010_v2'
OLD = 'pallet_observation_refiner_20261009_v1'
CORRECTED = 'pallet_kp_corrected_supervision_20261010_v1'
METHODS = ('VALIDATED_ROLE_ONLY', 'N3_VALIDATED_ROLE', 'N3_VALIDATED_ROLE_NO_MASK', 'N3_BASIN_ROBUST', 'N3_BASIN_STANDARD')
PRIMARY = 'N3_VALIDATED_ROLE'
CONTROLS = ('BASE', 'N3_SUBPIX')
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))
METRICS = {'translation_cm': ('translation_cm',1.,'cm'), 'rotation_deg': ('rotation_deg',1.,'degree'), 'ADDsym_cm': ('ADDsym_m',100.,'cm')}
LIMITS = [
    'Saved public arithmetic and identity joins only; no new detector/head, training, PnP, optimizer, rays, rendering or private GT reads.',
    'Stored GEOMETRIC_PROXY scorer values are aggregated, not independently reproduced from physical pose GT.',
    'Calibration references are actual-wire-supported virtual line intersections, not certified physical vertices or real-domain precision guarantees.',
    'The initial N3 projection/dimension prior may inherit initial errors. Local rank and numerical-tie handling do not certify global uniqueness.',
    'Private weight/source/image dependencies are SHA declarations where unavailable in the public bundle; their tensors and image content are not opened.',
    'Saved runtime records cannot independently recreate GPU execution, elapsed time or transient interference.',
    'Bootstrap confidence intervals are checked for finite ordered metadata here; their numeric endpoints are not replayed by this verifier.',
]


def require(value, message):
    if not value: raise ValueError(message)


def numeric(v):
    return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)


def close(a,b,label='',atol=1e-7,rtol=1e-9):
    if a is None or b is None: require(a is b,label+': null mismatch')
    else:
        require(numeric(a) and numeric(b),label+': nonfinite')
        require(abs(a-b)<=atol+rtol*max(abs(a),abs(b)),label+': numeric mismatch')


def same(a,b,label='',atol=1e-7,rtol=1e-9):
    if isinstance(a,list) and isinstance(b,list):
        require(len(a)==len(b),label+': length mismatch')
        for x,y in zip(a,b): same(x,y,label,atol,rtol)
    elif numeric(a) or numeric(b): close(a,b,label,atol,rtol)
    else: require(a==b,label+': value mismatch')


def dot(a,b): return math.fsum(x*y for x,y in zip(a,b))
def norm(a): return math.sqrt(dot(a,a))
def dist(a,b): return norm([x-y for x,y in zip(a,b)])
def valid(p): return isinstance(p,list) and len(p)==2 and all(numeric(x) for x in p) and p!=[-1,-1]
def inside(p,hw): return valid(p) and 0<=p[0]<hw[1] and 0<=p[1]<hw[0]
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def cube(d):
    a,b,c=[x/2 for x in d]
    return [[-a,-b,-c],[a,-b,-c],[a,b,-c],[-a,b,-c],[-a,-b,c],[a,-b,c],[a,b,c],[-a,b,c]]


def project(pose,K):
    R,t=pose['R_cf'],pose['centroid'];out=[]
    require(len(R)==3 and all(len(r)==3 and all(numeric(x) for x in r) for r in R),'rotation shape/nonfinite')
    require(len(t)==3 and all(numeric(x) for x in t),'translation shape/nonfinite')
    for X in cube(pose['cf_extents']):
        z=[dot(r,X)+v for r,v in zip(R,t)];require(z[2]>1e-9,'projection behind camera')
        q=[dot(r,z) for r in K];out.append([q[0]/q[2],q[1]/q[2]])
    return out


def physical(pose,xyz):
    d=pose['cf_extents'];R=pose['R_cf'];swapped=[xyz[2],xyz[1],xyz[0]]
    same_dim=lambda x:all(abs(a-b)<=1e-9 for a,b in zip(d,x))
    if same_dim(xyz): Q=[[1,0,0],[0,1,0],[0,0,1]]
    else:
        require(same_dim(swapped),'unregistered fit dimensions');Q=[[0,0,1],[0,1,0],[-1,0,0]]
    for i in range(3):
        for j in range(3): close(dot(R[i],R[j]),float(i==j),'Rcf orthogonality',1e-7,0.)
    determinant=R[0][0]*(R[1][1]*R[2][2]-R[1][2]*R[2][1])-R[0][1]*(R[1][0]*R[2][2]-R[1][2]*R[2][0])+R[0][2]*(R[1][0]*R[2][1]-R[1][1]*R[2][0])
    close(determinant,1.,'proper Rcf',1e-7,0.)
    same(pose['R_physical'],[[math.fsum(R[i][k]*Q[k][j] for k in range(3)) for j in range(3)] for i in range(3)],'physical frame mapping',1e-7,0.)


def hidden(pose):
    if not pose.get('available'): return []
    R,t=pose['R_cf'],pose['centroid'];camera=[-math.fsum(R[j][i]*t[j] for j in range(3)) for i in range(3)];H=[]
    for i,X in enumerate(cube(pose['cf_extents'])):
        ray=[camera[j]-X[j] for j in range(3)];l=norm(ray)
        if max((1 if X[j]>=0 else -1)*ray[j]/l for j in range(3)) < -math.sin(math.radians(2)): H.append(i)
    return H


def rms(a,b): return math.sqrt(math.fsum(dist(x,y)**2 for x,y in zip(a,b))/len(a))


def weighted_tls(points,radii):
    """Independent closed-form two-dimensional TLS/covariance arithmetic."""
    weights=[1/max(.5,r)**2 for r in radii];total=math.fsum(weights)
    center=[math.fsum(w*p[j] for w,p in zip(weights,points))/total for j in range(2)]
    delta=[[p[j]-center[j] for j in range(2)] for p in points]
    scatter=[[math.fsum(w*p[i]*p[j] for w,p in zip(weights,delta))/total for j in range(2)] for i in range(2)]
    theta=.5*math.atan2(2*scatter[0][1],scatter[0][0]-scatter[1][1]);tangent=[math.cos(theta),math.sin(theta)]
    biggest=max(range(2),key=lambda i:abs(tangent[i]))
    if tangent[biggest]<0:tangent=[-x for x in tangent]
    normal=[-tangent[1],tangent[0]];offset=dot(normal,center)
    coordinate=[dot(p,tangent) for p in delta];residual=[dot(p,normal)-offset for p in points]
    aa=math.fsum(w*s*s for w,s in zip(weights,coordinate));bb=math.fsum(w*s for w,s in zip(weights,coordinate));cc=total;det=aa*cc-bb*bb
    require(det>1e-12,'line information singular')
    scaled=math.fsum((v/max(.5,r))**2 for v,r in zip(residual,radii))/len(points);scale=max(1.,scaled)
    covariance=[[scale*cc/det,-scale*bb/det],[-scale*bb/det,scale*aa/det]]
    return dict(center=center,tangent=tangent,normal=normal,offset=offset,parameter_covariance=covariance,
                residual_rms_px=math.sqrt(math.fsum(v*v for v in residual)/len(points)),standardized_residual_rms=math.sqrt(scaled))


def quantile(v,q):
    if not v:return None
    v=sorted(v);x=(len(v)-1)*q;a=int(math.floor(x));b=int(math.ceil(x));return v[a]+(v[b]-v[a])*(x-a)


def distribution(v,unit):
    require(all(numeric(x) for x in v),'nonfinite metric input');n=len(v);mean=math.fsum(v)/n if n else None
    variance=math.fsum((x-mean)**2 for x in v)/(n-1) if n>1 else None
    return dict(n=n,mean=mean,sample_variance=variance,sample_std=math.sqrt(variance) if variance is not None else None,
                median=quantile(v,.5),P90=quantile(v,.9),max=max(v) if n else None,unit=unit,ddof=1,quantile_method='numpy linear')


def check_distribution(stored,v,unit,label):
    for key,value in distribution(v,unit).items():
        if numeric(value) or value is None: close(stored[key],value,label+'/'+key)
        else: require(stored[key]==value,label+'/'+key)


class Review:
    def __init__(self,root,folder,require_runtime,runtime_folder=None,failed_runtime_folder=None):
        self.root=root.resolve();self.doc=self.root/'_docs/experiments'/NAME;self.folder=folder.resolve()
        self.runtime_folder=Path(runtime_folder).resolve() if runtime_folder else self.folder
        self.failed_runtime_folder=Path(failed_runtime_folder).resolve() if failed_runtime_folder else self.runtime_folder.parent/'runtime' if self.runtime_folder.name=='runtime_resume' else self.runtime_folder
        self.old=self.root/'_docs/experiments'/OLD;self.corrected=self.root/'_docs/experiments'/CORRECTED
        self.require_runtime=require_runtime;self.used={};self.unopened=[];self.counts=Counter()

    def track(self,path):
        path=Path(path).resolve();require(path.is_file(),'missing input '+path.name)
        require(any(path.is_relative_to(p) for p in (self.root,self.folder,self.runtime_folder,self.failed_runtime_folder)),'input outside public root/execution folders')
        name=str(path.relative_to(self.root)) if path.is_relative_to(self.root) else path.name
        self.used[str(path)]=dict(path=name,sha256=sha(path),bytes=path.stat().st_size)
        return path

    def json(self,path):
        path=self.track(path)
        with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:return json.load(f)

    def rows(self,path):
        path=self.track(path)
        with gzip.open(path,'rt') as f:return [json.loads(s) for s in f if s.strip()]

    def bind(self,item,optional=False,alias=None):
        name=Path(item['path']);candidates=[self.root/name] if not name.is_absolute() else []
        candidates += [self.folder/name.name,self.runtime_folder/name.name,self.doc/name.name]
        if alias:candidates=[self.doc/alias,self.folder/alias,self.runtime_folder/alias,self.failed_runtime_folder/name.name]+candidates
        path=next((p for p in candidates if p.is_file()),None)
        if path is None:
            require(optional,'missing bound public artifact '+name.name)
            self.unopened.append(dict(path=name.name,sha256=item['sha256'],bytes=item['bytes'],content_opened=False));return
        self.track(path);require(sha(path)==item['sha256'] and path.stat().st_size==item['bytes'],'SHA binding differs '+name.name)

    def population(self,rows,methods):
        out={}
        for r in rows:
            key=r['method'],r['id'];require(key not in out and key[0] in methods and key[1] in self.frames,'unknown/duplicate row')
            require(r['session']==self.frames[r['id']]['session'],'session identity differs');out[key]=r
        require(len(out)==245*len(methods) and all((m,i) in out for m in methods for i in self.ids),'245 per-arm population differs')
        return out

    def inputs(self):
        self.cohort=self.json(self.corrected/'COHORT.json');self.ids=self.cohort['ids']
        require(len(self.ids)==len(set(self.ids))==245 and self.cohort['counts']==dict(clean=153,moderate=92,severe_excluded=74,original=319),'scope differs')
        authority=self.json(self.old/'INPUTS.json')['frames'];require(len(authority)==319 and len({f['id'] for f in authority})==319,'319 metadata authority differs')
        allframes={f['id']:f for f in authority};self.frames={i:allframes[i] for i in self.ids}
        require(len({f['session'] for f in self.frames.values()})==13,'eligible13 sessions differ')
        require(set(self.ids).isdisjoint(self.cohort['excluded_ids']) and set(self.ids)|set(self.cohort['excluded_ids'])==set(allframes),'cohort disjoint complete partition differs')
        self.labels={f['id']:f['label'] for f in self.cohort['frames']}
        require(Counter(self.labels.values())==Counter(clean=153,moderate=92),'severity labels differ')
        protection=self.json(self.doc/'PRIOR_MATERIALIZED_BINDINGS.json')
        for item in protection['files']:self.bind(item)
        self.protocol=self.json(self.doc/'PROTOCOL.json')
        require(self.protocol['methods']==list(METHODS) and self.protocol['primary']==PRIMARY and self.protocol['frames']==245 and self.protocol['GT_tuning'] is False and self.protocol['new_training_updates']==0,'frozen method contract differs')
        for name,item in self.protocol['fixed_inputs'].items():
            self.bind(item,optional=name in {'training_completion','corrected_role_last','base_weights','N3_weights','N3_configuration','N3_normalization','original_N3_runtime'})
        self.cal=self.json(self.doc/'CALIBRATION.json');checks=self.json(self.doc/'CALIBRATION_CHECKS.json')
        require(checks['complete'] and checks['passed'] and checks['calibration_sha256']==sha(self.doc/'CALIBRATION.json'),'source calibration receipt mismatch')
        require(checks['real_rows']==checks['source_test_rows']==checks['new_head_forwards']==checks['new_PnP']==0,'calibration verifier scope changed')
        return dict(eligible_frames=245,clean=153,moderate=92,severe_excluded=74,protected_files=len(protection['files']),public_code_bindings_checked=True,source_calibration_receipt_linked_not_model_replayed=True)

    def observation(self):
        rows=self.rows(self.folder/'OBSERVATIONS.jsonl.gz');require(len(rows)==245 and {r['id'] for r in rows}==set(self.ids),'observation245 identity differs')
        self.observations={r['id']:r for r in rows}
        for r in rows:
            fid=r['id'];f=self.frames[fid];require(r['session']==f['session'] and r['GT_input'] is False,'observation identity/GT input differs')
            same(r['original_base_points'],f['points']['BASE'],'Base detector parity',1e-7,0.);same(r['native_N3_points'],f['points']['N3_SUBPIX'],'N3 parity',1e-7,0.)
            require(not any(k in r for k in ('pose','corner','mask_audit','target','reference_native_points_px')),'posthoc truth in deployable observation')
            records=r['queries'];require(len(records) in (0,84),'query count differs')
            if not records:require(not r['corners'] and not r['lines'],'missing logits with manufactured observations');continue
            raw=[]
            for i,q in enumerate(records):
                z=q['candidate_logits'];require(len(z)==66 and all(numeric(v) for v in z),'rawlogit shape/nonfinite')
                raw.extend(z);best=max(range(65),key=lambda k:z[k]);m=z[best]-z[65];confidence=1/(1+math.exp(-max(-700,min(700,m))))
                p=[math.exp(v-max(z[:65])) for v in z[:65]];total=math.fsum(p);sigma=max(.5,math.sqrt(math.fsum(v*(k-best)**2 for k,v in enumerate(p))/total))
                close(q['confidence'],confidence,'confidence from raw logits',1e-10);close(q['sigma_mode_px'],sigma,'mode second moment',1e-8)
                close(q['radius_px'],sigma*self.cal['uncertainty']['query_scale'],'query radius',1e-8)
                require(q['query']==i and q['edge']==i//7 and q['endpoints']==list(EDGES[i//7]) and q['chosen_candidate']==best,'query/candidate identity differs')
                a,b=EDGES[i//7];A,B=r['original_base_points'][a],r['original_base_points'][b];u=(i%7+1)/8
                expected_center=[(1-u)*A[j]+u*B[j] for j in range(2)];d=[B[j]-A[j] for j in range(2)];length=norm(d);normal=[-d[1]/max(length,1e-6),d[0]/max(length,1e-6)]
                same(q['center'],expected_center,'query Base anchor');same(q['normal'],normal,'query Base normal')
                xy=[expected_center[j]+(best-32)*normal[j] for j in range(2)]
                geomvalid=all(numeric(v) for v in d) and length>1e-6 and A!=[-1,-1] and B!=[-1,-1]
                accepted=bool(geomvalid and q['edge'] in self.cal['supported_edges'] and inside(xy,f['raw_hw']) and self.cal['confidence']['enabled'] and confidence>=self.cal['confidence']['threshold'])
                require(q['no_match']==(not accepted) and (q['selected_xy'] is not None)==accepted and q['physical_absence_inferred'] is False,'query gate differs')
                if accepted:same(q['selected_xy'],xy,'MAP candidate coordinate')
            require(hashlib.sha256(struct.pack('<'+'f'*len(raw),*raw)).hexdigest()==r['raw_logits_sha256'],'raw float32 logit SHA mismatch')
            selected=sum(q['selected_xy'] is not None for q in records);require(r['selected_queries']==selected and r['all_no_match']==(selected==0),'query selection/absence count differs')
            lines={l['edge']:l for l in r['lines']};require(len(lines)==len(r['lines']),'duplicate semantic line')
            for e,line in lines.items():
                ids=line['queries'];require(len(ids)>=3 and len(ids)==len(set(ids)) and all(i//7==e and records[i]['selected_xy'] is not None for i in ids),'line supports differ')
                require(max(i%7 for i in ids)-min(i%7 for i in ids)>=2,'nonspanning query IDs')
                same(line['support_points'],[records[i]['selected_xy'] for i in ids],'line support coordinates');same(line['query_radii_px'],[records[i]['radius_px'] for i in ids],'line radii')
                fitted=weighted_tls(line['support_points'],line['query_radii_px'])
                for key,value in fitted.items():same(line[key],value,'independent weighted TLS '+key,1e-6)
                close(norm(line['normal']),1.,'line unit normal');close(norm(line['tangent']),1.,'line unit tangent')
                close(dot(line['normal'],line['tangent']),0.,'line tangent orthogonality')
                support=[dot([p[j]-line['center'][j] for j in range(2)],line['tangent']) for p in line['support_points']]
                same(line['support_interval_px'],[min(support),max(support)],'line support interval');close(line['support_length_px'],max(support)-min(support),'line span')
                close(line['nominal_query_spacing_px'],dist(r['original_base_points'][EDGES[e][0]],r['original_base_points'][EDGES[e][1]])/8,'nominal spacing')
                require(line['support_length_px']>=2*line['nominal_query_spacing_px']*(1-64*sys.float_info.epsilon),'line span gate differs')
            cornerids=[]
            for c in r['corners']:
                require(c['id'] not in cornerids,'duplicate corner ID');cornerids.append(c['id']);aa,bb=c['edges'];A,B=lines[aa],lines[bb]
                require(c['id'] in EDGES[aa] and c['id'] in EDGES[bb],'nonincident lines used as corner')
                a,b=A['normal'];d,e=B['normal'];det=a*e-b*d;require(abs(det)>1e-6,'parallel corner lines')
                xy=[(A['offset']*e-b*B['offset'])/det,(a*B['offset']-A['offset']*d)/det];same(c['xy'],xy,'independent line intersection')
                variances=[];gaps=[]
                for line in (A,B):
                    s=dot([xy[j]-line['center'][j] for j in range(2)],line['tangent']);lo,hi=line['support_interval_px'];gaps.append(max(lo-s,s-hi,0.)/max(line['support_length_px'],1e-6))
                    cov=line['parameter_covariance'];variances.append(cov[0][0]*s*s+(cov[0][1]+cov[1][0])*s+cov[1][1])
                inv=[[e/det,-b/det],[-d/det,a/det]];cov=[[math.fsum(inv[i][k]*variances[k]*inv[j][k] for k in range(2)) for j in range(2)] for i in range(2)]
                same(c['covariance_px2'],cov,'corner covariance surrogate',1e-6);vmax=(cov[0][0]+cov[1][1]+math.sqrt((cov[0][0]-cov[1][1])**2+4*cov[0][1]*cov[1][0]))/2
                sigma=math.sqrt(max(0.,vmax));radius=sigma*self.cal['uncertainty']['corner_scale'];spacing=min(A['nominal_query_spacing_px'],B['nominal_query_spacing_px'])
                close(c['sigma_px'],sigma,'corner sigma',1e-6);close(c['radius_px'],radius,'corner radius',1e-6);same(c['extrapolation_ratios'],gaps,'corner extrapolation',1e-6)
                require(self.cal['geometry']['enabled'] and inside(xy,f['raw_hw']) and radius/max(spacing,1e-6)<=1.+1e-9 and max(gaps)<=self.cal['geometry']['max_extrapolation_ratio']+1e-9,'accepted corner violates geometry gate')
            self.counts.update(observation_rows=1,raw_query_logits=len(records),observed_lines=len(lines),decoded_corners=len(cornerids))
        return dict(self.counts,confidence_and_MAP_recomputed=True,virtual_intersection_not_physical_vertex_certification=True)

    def geometry(self):
        seal=self.json(self.folder/'GEOMETRY_SEAL.json');require(seal['complete'] and seal['frames']==245 and seal['rows']==1225 and seal['fixed_rows']==490 and seal['GT_read_allowed'] is False,'geometry seal scope differs')
        for name in ('protocol','observations','geometry','fixed_geometry','parity'):self.bind(seal[name])
        rows=self.population(self.rows(self.folder/'GEOMETRY_SEALED.jsonl.gz'),METHODS);controls=self.population(self.rows(self.folder/'FIXED_GEOMETRY_SEALED.jsonl.gz'),CONTROLS)
        for (m,fid),r in controls.items():
            obs=self.observations[fid];expected=obs['original_base_points'] if m=='BASE' else obs['native_N3_points'];same(r['native_points'],expected,'fresh fixed controls');require(r['output_status']=='FRESH_FIXED_CONTROL' and not r['new_pose_estimated'] and not r['fallback_used'],'fixed control status differs')
            if m=='N3_SUBPIX':require(r['actual_pose']==obs['initial_N3_pose'],'fresh initial/control pose differs')
        bankledger=defaultdict(Counter)
        for (m,fid),r in rows.items():
            obs=self.observations[fid];f=self.frames[fid];native=obs['native_N3_points'];initial=obs['initial_N3_pose'];s=r['solver'];contract=r['observation_contract']
            require(not any(k in r for k in ('pose','corner','mask_audit','baseline_pose','reference_native_points_px','evaluation_reference')),'GT-scored fields in geometry seal')
            require(r['initial_pose']==initial and r['K']==f['K'] and r['xyz']==f['xyz'] and r['raw_hw']==f['raw_hw'],'frozen metadata/initial pose differs')
            require(r['observation_raw_logits_sha256']==obs['raw_logits_sha256'] and r['selected_index']==f['selected_index'],'observation linkage/candidate selection differs')
            H=[] if m=='N3_VALIDATED_ROLE_NO_MASK' else hidden(initial);require(r['hidden_initial']==H and r['excluded']==H and s['hidden']==H and s['excluded']==H,'initial hidden/excluded set differs')
            rawcorners={c['id']:c for c in obs['corners']};invalid=[]
            for line,check in zip(obs['lines'],contract['final_line_consensus_checks']):
                residual=[abs(dot(p,line['normal'])-line['offset']) for p in line['support_points']];ok=all(numeric(x) and numeric(v) and v>=0 and x<=v+1e-9 for x,v in zip(residual,line['query_radii_px']))
                require(check['edge']==line['edge'] and check['support_queries']==line['queries'] and check['accepted']==ok and check['tolerance_px']==1e-9,'final line evidence veto differs');same(check['absolute_residuals_px'],residual,'final line residual');same(check['query_radii_px'],line['query_radii_px'],'final line evidence radius')
                if not ok:invalid.append(line['edge'])
            require(len(contract['final_line_consensus_checks'])==len(obs['lines']) and contract['invalid_final_line_edges']==sorted(invalid),'line veto population differs')
            selected={}
            for admission in contract['corner_admission']:
                c=rawcorners[admission['id']];radius=c['radius_px'];ok=not(set(c['edges'])&set(invalid)) and numeric(radius) and 0<=radius<=8.
                require(admission['accepted']==ok and admission['edges']==c['edges'] and admission['radius_px']==radius,'corner evidence/radius admission differs')
                if ok:selected[c['id']]=c['xy']
            require(sorted(a['id'] for a in contract['corner_admission'])==sorted(rawcorners) and contract['validated_boundary_corner_ids']==sorted(selected) and contract['computed_decoder_corner_ids']==sorted(rawcorners),'corner admission coverage differs')
            sparse=[[None,None] for _ in range(8)]+[native[8]];hybrid=[list(p) for p in native];replaced=[]
            for k,xy in sorted(selected.items()):
                sparse[k]=xy
                if k not in obs['predicted_N3_hidden'] and valid(native[k]) and dist(xy,native[k])<=8.:hybrid[k]=xy;replaced.append(k)
            require(contract['hybrid_boundary_corner_ids']==replaced and contract['replacement_cap_px']==contract['corner_uncertainty_cap_px']==8.,'hybrid fixed8px admission differs')
            require(contract['hybrid_rejected_boundary_corner_ids']==sorted(set(selected)-set(replaced)) and contract['native_N3_RGB_corner_ids']==[k for k in range(8) if k not in replaced],'hybrid provenance ID partition differs')
            require(len(contract['per_validated_corner'])==len(selected),'validated corner distance population differs')
            for c in contract['per_validated_corner']:
                require(c['id'] in selected and c['selected_for_hybrid']==(c['id'] in replaced),'hybrid distance selection differs')
                same(c['validated_xy'],selected[c['id']],'validated corner coordinate');same(c['native_N3_xy'],native[c['id']],'independent native N3 coordinate')
                close(c['distance_to_native_N3_px'],dist(selected[c['id']],native[c['id']]) if valid(native[c['id']]) else None,'native8px replacement distance')
            points=sparse if m=='VALIDATED_ROLE_ONLY' else hybrid if m.startswith('N3_VALIDATED_ROLE') else native
            same(r['input_points'],points,'pose input provenance',1e-7,0.);require(r['native_points'][8]==native[8],'detector center changed')
            require(not r['reprojections_reused_as_observations'] and not r['no_match_filled_as_boundary_observation'] and not r['local_point_line_refinement'] and r['partial_lines_not_pose_inputs'],'pseudo observations/extra line path introduced')
            eligible=[i for i in range(8) if inside(points[i],f['raw_hw'])];used=[i for i in eligible if i not in H];require(s['eligible']==eligible and s['used']==used and set(s['fit_input_ids'])<=set(used),'hidden/nonobserved fit input')
            new=bool(s['available']);fallback=not new and bool(initial['available']);require(r['new_pose_estimated']==new and r['fallback_used']==fallback and r['pose_available']==bool(new or initial['available']) and r['actual_pose']['available']==r['pose_available'] and r['no_pose']==not_bool(r['pose_available']),'new/fallback/failure availability differs')
            require(s['refit_count']<= (1 if m=='N3_BASIN_STANDARD' else 3) and s['operation_counts']['lm_calls']<=s['refit_count'],'LM fixed-start bound violated')
            require(r['output_status']==('NEW_POSE' if new else 'N3_BASELINE_FALLBACK' if fallback else 'POSE_FAILURE'),'output status differs')
            expected=[list(p) for p in native]
            if new:
                physical(r['actual_pose'],f['xyz']);require(r['actual_pose']['cf_extents']==initial['cf_extents'],'dimension branch changed')
                for field in ('R_cf','R_physical','centroid','cf_extents'):same(r['actual_pose'][field],s[field],'solver actual pose',0.,0.)
                projection=project(r['actual_pose'],f['K']);prior=project(initial,f['K']);same(s['projected'],projection,'independent final projection',1e-6);same(s['initial_identity_prior']['projected'],prior,'independent initial prior projection',1e-6)
                value=rms(projection,prior);close(s['prior_projection_rms_px'],value,'all8 prior RMS',1e-6);require(value<=8.+1e-9 and s['prior_projection_used_as_observation'] is False and not s['global_uniqueness_proven'],'projection prior/global uniqueness contract differs')
                residual=[dist(projection[i],points[i]) for i in used];inliers=[i for i,e in zip(used,residual) if e<=8.];require(s['final_inliers']==inliers and len(used)>=4,'final observed inliers differ')
                require(s['geometry']['jacobian']['numerical_rank']==6 and s['unresolved_ambiguity'] is False,'numerical rank/ambiguity failed pose accepted')
                if m!='N3_BASIN_STANDARD':require(len(inliers)>=4,'robust insufficient consensus accepted')
                for i in range(8):
                    if i not in H and valid(points[i]):expected[i]=points[i]
                for i in H:expected[i]=projection[i];self.counts['hidden_projected_coordinates']+=1
                require(r['hidden_after']==hidden(r['actual_pose']),'postfit hidden diagnostic differs')
            else:require(r['actual_pose']==initial and not r['hidden_after'],'fallback pose altered')
            require(r['reprojected_ids']==(H if new else []) and r['hidden_reprojected']==bool(new and H) and r['hidden_set_changed']==bool(new and set(r['hidden_after'])!=set(H)),'H replacement/change diagnostic differs')
            same(r['native_points'],expected,'observed nonH/final H/fullN3 fallback',1e-6)
            boundary=set(selected if m=='VALIDATED_ROLE_ONLY' else replaced if m.startswith('N3_VALIDATED_ROLE') else [])
            sources=['FINAL_POSE_REPROJECTION' if new and i in H else 'VALIDATED_BOUNDARY_INTERSECTION' if new and i in boundary else 'NATIVE_N3_RGB' for i in range(8)]+['UNCHANGED_DETECTOR_CENTER']
            require(r['output_coordinate_sources']==sources and r['native_N3_used_as_independent_RGB_observations']==(m!='VALIDATED_ROLE_ONLY'),'output source labels differ')
            key='ROLE_ONLY' if m=='VALIDATED_ROLE_ONLY' else 'HYBRID' if m.startswith('N3_VALIDATED_ROLE') else 'N3';bankledger[fid,key].update(s['operation_counts'])
            self.counts['pose_contract_rows']+=1
        for fid in self.ids:
            same(rows['N3_VALIDATED_ROLE',fid]['input_points'],rows['N3_VALIDATED_ROLE_NO_MASK',fid]['input_points'],'same coordinates masked/no-mask',0.,0.)
            same(rows['N3_BASIN_ROBUST',fid]['input_points'],rows['N3_BASIN_STANDARD',fid]['input_points'],'same N3 standard/robust',0.,0.)
            require(rows['N3_VALIDATED_ROLE',fid]['solver']['input_hash']==rows['N3_VALIDATED_ROLE_NO_MASK',fid]['solver']['input_hash'] and rows['N3_BASIN_ROBUST',fid]['solver']['input_hash']==rows['N3_BASIN_STANDARD',fid]['solver']['input_hash'],'same-coordinate numeric bank hash differs')
        for bank in seal['banks']:
            for key,value in bank['coordinate_banks'].items():
                require(dict(bankledger[bank['id'],key])==value,'per-solve vs shared-bank ledger differs')
                method='VALIDATED_ROLE_ONLY' if key=='ROLE_ONLY' else PRIMARY if key=='HYBRID' else 'N3_BASIN_ROBUST';r=rows[method,bank['id']]
                n=len(r['solver']['eligible']);dimensions=1 if abs(r['xyz'][0]-r['xyz'][2])<1e-9 else 2
                bound=dimensions*math.comb(n,4) if n>=4 else 0
                require(value['subsets_considered'] in (0,bound) and value['subset_generic_calls']<=value['subsets_considered'] and value['generic_calls']==value['subset_generic_calls']+value['refit_generic_calls']+value['standard_generic_calls'],'finite bank generation bound/primitive ledger differs')
        require(seal['actual_calls']['detector_calls']==seal['actual_calls']['N3_route_calls']==seal['actual_calls']['initial_pose_calls']==seal['actual_calls']['base_control_pose_calls']==seal['actual_calls']['ROLE_head_calls']==seal['actual_calls']['feature_initial_pose_calls']==245 and seal['actual_calls']['final_pose_paths']==1225,'whole fresh-path stage counts differ')
        self.geometry_rows=rows;self.fixed_geometry=controls;self.geometry_seal=seal
        return dict(rows=1225,fixed_rows=490,hidden_projection_coordinates=self.counts['hidden_projected_coordinates'],fit_exclusion_and_no_pseudo_observations=True,full_N3_fallback_verified=True,shared_mask_bank_ledgers_verified=True)

    def scored(self):
        new=self.population(self.rows(self.folder/'PREDICTIONS.jsonl.gz'),METHODS);fixed=self.population(self.rows(self.folder/'FIXED_PREDICTIONS.jsonl.gz'),CONTROLS)
        for raw,scored in ((self.geometry_rows,new),(self.fixed_geometry,fixed)):
            for key,r in raw.items():
                for field,value in r.items():require(scored[key].get(field)==value,'sealed field changed by scoring '+field)
                require(scored[key]['pose']['available']==r['pose_available'],'scored availability differs')
        receipt=self.json(self.folder/'SCORING_RECEIPT.json')
        require(receipt['complete'] and receipt['frames']==245 and receipt['scored_method_rows']==1225 and receipt['fixed_control_rows']==490,'posthoc score scope differs')
        for name in ('geometry_seal','geometry','predictions','fixed_predictions','reference_image_ID_mapping'):self.bind(receipt[name])
        require(receipt['GT_access_only_after_complete_seal'] and receipt['scoring_fitting_entries_forbidden'] and receipt['frozen_reference_phase']=='fresh fixed N3_SUBPIX','posthoc truth-phase guard differs')
        require(all(receipt[k]==0 for k in ('new_detector_forwards','new_head_forwards','new_pose_fits','new_rays')),'posthoc stage has new fit/model/ray count')
        for (m,fid),r in {**new,**fixed}.items():
            ref=r['evaluation_reference'];base=fixed['BASE',fid];n3=fixed['N3_SUBPIX',fid]
            require(ref['phase_control']=='N3_SUBPIX' and r['mask_audit']['reference_phase_control']=='N3_SUBPIX','posthoc coordinate identity phase differs')
            same(ref['initial_N3_native_points'],n3['native_points'],'posthoc initialN3');same(ref['initial_BASE_native_points'],base['native_points'],'posthoc initialBase')
            require(len(ref['native_points_px'])==8 and len(ref['human_states_native'])==8 and len(set(ref['valid_native_ids']))==len(ref['valid_native_ids']) and all(0<=i<8 and valid(ref['native_points_px'][i]) for i in ref['valid_native_ids']),'proxy reference shape/valid IDs differ')
            audit=r['mask_audit'];states=ref['human_states_native'];require(audit['human_states_native']==states and audit['references_used_for_observation_selection'] is False and audit['classification_difference_is_pose_failure'] is False,'human-state audit inference separation differs')
            H=set(r['hidden_initial']);humanH={i for i,s in enumerate(states) if s=='SELF_OCCLUDED'};known={i for i,s in enumerate(states) if s!='UNANNOTATED'}
            sol=r.get('solver') or {};eligible=set(sol.get('eligible',range(8)));used=set(sol.get('used',[]))
            require(audit['known_ids']==sorted(known) and audit['false_excluded_visible']==sorted(H & {i for i,s in enumerate(states) if s=='DIRECT_VISIBLE'}) and audit['false_retained_self']==sorted((humanH-H)&eligible) and audit['mask_wrong_on_known']==bool((H^humanH)&known) and audit['remaining_human_direct_visible']==len(used & {i for i,s in enumerate(states) if s=='DIRECT_VISIBLE'}),'posthoc mask versus pose separation differs')
        for m in METHODS:require(receipt['status_counts'][m]==dict(Counter(r['output_status'] for (mm,_),r in new.items() if mm==m)),'scoring status receipt mismatch')
        self.all_scored={**new,**fixed}
        return dict(scored_rows=1225,fixed_scored_rows=490,sealed_deployable_fields_unchanged=True,stored_proxy_values_not_private_GT_recomputed=True)

    def summary(self,stored,rows):
        available=[r for r in rows if r['pose']['available']];fresh=[r for r in available if r['new_pose_estimated']]
        values=dict(total_frames=len(rows),denominator=len(rows),pose_available=len(available),new_pose_estimated=len(fresh),fallback_used=sum(r['fallback_used'] for r in rows),no_pose=len(rows)-len(available),fixed_control_outputs=sum(r['output_status']=='FRESH_FIXED_CONTROL' for r in rows),hidden_reprojected=sum(r['hidden_reprojected'] for r in rows),hidden_set_changed=sum(r['hidden_set_changed'] for r in rows))
        for key,value in values.items():require(stored[key]==value,'summary count '+key)
        require(stored['output_status_counts']==dict(Counter(r['output_status'] for r in rows)),'summary status counts')
        require(stored['available_ids']==[r['id'] for r in available] and stored['new_pose_ids']==[r['id'] for r in fresh] and stored['fallback_ids']==[r['id'] for r in rows if r['fallback_used']] and stored['no_pose_ids']==[r['id'] for r in rows if not r['pose']['available']],'summary status IDs differ')
        for scope,rr in [('operational',available),('new_pose',fresh)]:
            for metric,(field,factor,unit) in METRICS.items():check_distribution(stored['metrics'][scope][metric],[r['pose'][field]*factor for r in rr],unit,scope+'/'+metric);self.counts['distributions']+=1

    def metrics(self):
        metrics_path=self.folder/'METRICS.json'
        if not metrics_path.exists():metrics_path=self.doc/'METRICS.json'
        metrics=self.json(metrics_path);require(metrics['complete'] and metrics['primary_method']==PRIMARY and metrics['comparator']=='N3_SUBPIX','metrics contract differs')
        for item in metrics['bindings'].values():self.bind(item)
        groups={m:{fid:self.all_scored[m,fid] for fid in self.ids} for m in CONTROLS+METHODS}
        contrasts=[(m,'N3_SUBPIX') for m in METHODS]+[(PRIMARY,'N3_BASIN_ROBUST'),(PRIMARY,'N3_VALIDATED_ROLE_NO_MASK'),('N3_BASIN_ROBUST','N3_BASIN_STANDARD')]
        for name,ids in [('combined',self.ids),('easy',[i for i in self.ids if self.labels[i]=='clean']),('medium',[i for i in self.ids if self.labels[i]=='moderate'])]:
            stored=metrics['strata'][name];require(stored['frames']==len(ids) and set(stored['methods'])==set(groups) and set(stored['contrasts'])=={a+'_minus_'+b for a,b in contrasts},'stratum methods/contrasts/population differ')
            for m in groups:self.summary(stored['methods'][m],[groups[m][i] for i in ids])
            for a,b in contrasts:
                for scope in ('common_operational','candidate_new_pose'):
                    pair=stored['contrasts'][a+'_minus_'+b][scope];common=[i for i in ids if groups[a][i]['pose']['available'] and groups[b][i]['pose']['available'] and (scope=='common_operational' or groups[a][i]['new_pose_estimated'])]
                    require(pair['common_ids']==pair['pair_ids']==common and pair['common_frames']==len(common) and pair['denominator']==len(ids) and pair['excluded_ids']==[i for i in ids if i not in set(common)],'paired cohort differs')
                    for metric,(field,factor,unit) in METRICS.items():
                        delta=[(groups[a][i]['pose'][field]-groups[b][i]['pose'][field])*factor for i in common];p=pair['metrics'][metric];check_distribution(p,delta,unit,'paired '+metric);close(p['mean_delta'],distribution(delta,unit)['mean'],'paired mean')
                        require(p['improved_frames']==sum(x<-1e-9 for x in delta) and p['worsened_frames']==sum(x>1e-9 for x in delta) and p['unchanged_frames']==sum(abs(x)<=1e-9 for x in delta),'paired outcome counts')
                        ci=p['CI95'];require((not common and ci is None) or (bool(common) and isinstance(ci,list) and len(ci)==2 and all(numeric(x) for x in ci) and ci[0]<=ci[1]),'CI95 finite/ordered metadata differs')
                        for who,label in [(a,'new_marginals'),(b,'comparator_marginals')]:check_distribution(pair[label][metric],[groups[who][i]['pose'][field]*factor for i in common],unit,'paired marginal')
                        self.counts['paired_mean_distributions']+=1
        require(metrics['methods']==metrics['strata']['combined']['methods'] and metrics['contrasts']==metrics['strata']['combined']['contrasts'],'combined aliases differ')
        require(metrics['population']==dict(frames=245,clean=153,moderate=92,severe_excluded=74,all_frames_retained=True,sessions=13) and metrics['zero_new_detector_head_PnP_ray_or_optimizer_calls'] is True,'statistics scope/execution declaration differs')
        bootstrap=self.json(self.root/'_docs/experiments/pallet_kp_difficulty_20261010_v1/BOOTSTRAP_SESSION_DRAWS.json.gz')
        draws=bootstrap['counts'];sessions=bootstrap['sessions'];require(len(draws)==10000 and len(sessions)==len(set(sessions))==13 and all(len(v)==13 and all(isinstance(x,int) and 0<=x<=13 for x in v) and sum(v)==13 for v in draws),'fixed bootstrap shape/counts differ')
        drawhash=hashlib.sha256(struct.pack('<'+'H'*(10000*13),*(x for row in draws for x in row))).hexdigest()
        require(drawhash==bootstrap['serialized_raw_sha256']==metrics['bootstrap']['serialized_raw_sha256'] and metrics['bootstrap']['resamples']==10000 and metrics['bootstrap']['sessions']==sessions and metrics['bootstrap']['new_draws_generated']==0,'frozen bootstrap byte/metadata binding differs')
        primary=metrics['contrasts'][PRIMARY+'_minus_N3_SUBPIX']['common_operational'];full=primary['common_frames']==245;improved=full and primary['metrics']['translation_cm']['mean_delta']<0 and primary['metrics']['rotation_deg']['mean_delta']<0
        require(metrics['verdict']['all245_paired_outputs_available']==full and metrics['verdict']['primary_full_operational_T_and_R_improved']==improved and metrics['verdict']['implementation_success_is_performance_success'] is False and metrics['verdict']['independent_generalization_established'] is False,'primary verdict differs')
        return dict(distributions=self.counts['distributions'],paired_distributions=self.counts['paired_mean_distributions'],sample_variance_ddof1=True,linear_quantiles=True,CI95_numeric_replay=False,bootstrap_metadata_only=True,primary_verdict_recomputed=True)

    def runtime(self):
        path=self.runtime_folder/'RUNTIME.json'
        if not path.exists():path=self.doc/'RUNTIME.json'
        if not path.exists():
            require(not self.require_runtime,'actual fresh600 runtime missing')
            return dict(pending=True,reason='Fresh runtime is not yet available; no latency certification.')
        run=self.json(path)
        require(run['schema']=='validated_boundary_actual_complete_runtime_v2' and run['complete'] and run['status']=='DONE' and run['statistics_official'],'fresh runtime incomplete')
        arms=('BASE','N3_SUBPIX','N3_BASIN_ROBUST',PRIMARY)
        stages=('detector','N3_correction','initial_pose','boundary_observation','final_pose_reprojection_and_metadata','return')
        require(run['arms']==list(arms) and run['frames']==26 and run['panel_sessions']==13 and run['warmup_per_arm']==20 and run['repeats']==5 and run['measured_per_arm']==130 and run['configured_pipeline_calls']==600,'fresh runtime configuration differs')
        require(run['cached_coordinate_replay_used_for_timing'] is False and run['GT_inference_access'] is False and run['other_benchmarks_parallel'] is False and run['new_training_updates']==run['new_GT_evaluations']==0,'runtime route/access declaration differs')
        require(run['boundaries']['all_initial_and_final_PnP_included'] and run['boundaries']['all_observation_and_admission_work_included'],'runtime excludes required computation')
        for name in ('protocol','accuracy_seal','runtime_code','raw_rows'):self.bind(run[name])
        resume_path=self.doc/'RUNTIME_RESUME_PROTOCOL.json'
        if resume_path.exists():
            plan=self.json(resume_path);receipt=self.json(path.parent/'RUNTIME_RESUME_RECEIPT.json');fixed=plan['fixed_inputs'];previous=fixed['previous_attempt']
            require(plan['schema']=='validated_boundary_runtime_namespace_resume_v2' and plan['fixed_timed_calls']==600 and plan['arms']==list(arms) and plan['model_decoder_solver_and_timed_body_unchanged'] and plan['no_accuracy_repeated'] and plan['no_training_repeated'] and plan['no_GT_tuning'] and plan['previous_failure_preserved'],'runtime resume policy changed')
            for k in ('protocol','frozen_runtime','frozen_pipeline','resume_code','accuracy_seal'):self.bind(fixed[k])
            require(fixed['protocol']==run['protocol'] and fixed['frozen_runtime']==run['runtime_code'] and fixed['accuracy_seal']==run['accuracy_seal'],'runtime resume frozen body/seal link differs')
            names=previous['archived_public_names'];require(names==dict(runtime='FAILED_RUNTIME_PREFIX.json',raw_rows='FAILED_RUNTIME_ROWS.jsonl.gz',journal='FAILED_RUNTIME_STARTED.json'),'failure artifact alias map differs')
            for key,alias in names.items():self.bind(previous[key],alias=alias)
            failure_path=self.doc/names['runtime']
            if not failure_path.exists():failure_path=self.failed_runtime_folder/'RUNTIME.json'
            failure=self.json(failure_path);failed_rows_path=self.doc/names['raw_rows']
            if not failed_rows_path.exists():failed_rows_path=self.failed_runtime_folder/'RUNTIME_ROWS.jsonl.gz'
            failed_rows=self.rows(failed_rows_path);journal_path=self.doc/names['journal']
            if not journal_path.exists():journal_path=self.failed_runtime_folder/'RUNTIME_STARTED.json'
            journal=self.json(journal_path)
            reason=dict(type='ModuleNotFoundError',message="No module named 'scripts.research.pallet_joint_action_handoff_20261006_v1'")
            require(failure['status']==journal['status']=='FAILED_RUNTIME' and not failure['complete'] and not failure['statistics_official'] and not failure['model_initialization_complete'] and failure['reason']==previous['reason']==reason,'unrecognized previous runtime failure')
            require(not failed_rows and failure['execution'].get('pipeline_calls_started',0)==failure['execution'].get('pipeline_calls_complete',0)==0 and not journal['counts'] and previous['actual_timed_calls']==previous['actual_timed_rows']==previous['successful_model_initializations']==0,'previous timed calls incorrectly restarted')
            require(failure['protocol']==run['protocol'] and failure['runtime_code']==run['runtime_code'] and failure['configured_pipeline_calls']==600,'previous failure body/schedule drift')
            require(receipt['complete'] and receipt['previous_failed_attempt']==previous and receipt['resumed_actual_timed_calls']==600 and receipt['frozen_runtime_body_unchanged'] and receipt['new_accuracy_frames']==receipt['new_training_updates']==0,'runtime resume execution receipt differs')
            for k in ('resume_protocol','result','rows'):self.bind(receipt[k])
            self.counts['preserved_failed_runtime_timed_calls']=0
        panel=self.json(self.corrected/'RUNTIME_PANEL.json')['frames'];expect_panel=[dict(id=f['frame_id'],session=f['session_id']) for f in panel]
        require(run['panel']==expect_panel and len({f['id'] for f in expect_panel})==26 and all(f['id'] in self.frames for f in expect_panel) and Counter(f['session'] for f in expect_panel)==Counter({s:2 for s in {f['session'] for f in expect_panel}}),'eligible runtime panel differs')
        jobs=[]
        for i in range(20):
            order=list(arms[i%4:]+arms[:i%4]);order=order[::-1] if i%2 else order
            jobs.extend(dict(phase='warmup',arm=a,warmup_index=i,arm_position=p,image_index=i%26) for p,a in enumerate(order))
        for repeat in range(5):
            order=list(arms[repeat%4:]+arms[:repeat%4]);order=order[::-1] if repeat%2 else order
            for image in range(26):jobs.extend(dict(phase='measured',arm=a,repeat=repeat,arm_position=p,image_index=image) for p,a in enumerate(order))
        rows=self.rows(path.parent/'RUNTIME_ROWS.jsonl.gz');require(len(rows)==len(jobs)==600,'runtime row count differs')
        primitive=Counter();refs={**self.geometry_rows,**self.fixed_geometry}
        for r,j in zip(rows,jobs):
            require(all(r[k]==v for k,v in j.items()),'runtime exact job ordering differs')
            frame=expect_panel[j['image_index']];require(r['id']==frame['id'] and r['session']==frame['session'],'runtime frame/session differs')
            ref=refs[j['arm'],r['id']]
            require(r['GT_canary_active'] and r['parity_status']=='PASS' and r['candidate_center_score_box_preserved'],'runtime canary/parity declaration failed')
            require(all(r[k]==ref[k] for k in ('hidden_initial','output_status','new_pose_estimated','fallback_used','selected_index')),'runtime outcome versus sealed output differs')
            same(r['final_points'],ref['native_points'],'runtime final point parity',1e-7,0.)
            if ref['actual_pose']['available']:
                for key in ('R_cf','R_physical','centroid','cf_extents'):same(r['actual_pose'][key],ref['actual_pose'][key],'runtime pose parity',1e-7,0.)
            require(r['actual_pose']['available']==ref['actual_pose']['available'],'runtime availability differs')
            for stage in ('full',)+stages:require(numeric(r[stage+'_ms']) and r[stage+'_ms']>=0,'negative/nonfinite timing')
            close(r['full_ms'],math.fsum(r[s+'_ms'] for s in stages),'runtime interval partition',1e-7,1e-9)
            entries=r['primitive_entry_calls'];require(all(isinstance(v,int) and v>=0 for v in entries.values()),'invalid primitive counter');primitive.update(entries)
            if j['arm']==PRIMARY:require(r['observation_raw_logits_sha256']==self.observations[r['id']]['raw_logits_sha256'],'runtime ROLE raw logits changed')
            if r['new_pose_estimated']:
                s=r['solver'];H=r['hidden_initial'];require(set(H).isdisjoint(s['fit_input_ids']) and s['reprojected_points_reused_as_observations'] is False,'runtime hidden fit/pseudoobservation leak')
                projection=project(r['actual_pose'],self.frames[r['id']]['K']);same(s['projected'],projection,'runtime independent projection',1e-6)
                same([r['final_points'][k] for k in H],[projection[k] for k in H],'runtime H final projection',1e-6)
        require(dict(primitive)==run['actual_OpenCV_entry_calls'],'runtime primitive totals differ')
        expected=dict(detector_calls=600,N3_route_calls=450,initial_pose_calls=600,ROLE_head_calls=150,feature_initial_pose_calls=150,final_pose_paths=300)
        require(run['actual_pipeline_calls']==expected and run['execution']==dict(pipeline_calls_started=600,pipeline_calls_complete=600),'runtime fresh stage/call ledger differs')
        require(run['model_forwards']['N3']==450 and run['model_forwards']['ROLE']==150 and run['model_forwards']['detector']==600+run['model_forwards']['detector_internal_initialization_calls'],'runtime model forward ledger differs')
        require(run['model_initialization_complete'] and run['actual_image_decode_calls']==26 and run['image_decode_outside_intervals'],'runtime initialization/decode declaration differs')
        probes=run['environment']['interference_snapshots'];require(probes and all(p['quiet'] and p['gpu_temperature_under_80'] for p in probes),'saved interference/thermal snapshots failed')
        for arm in arms:
            require(run['warmup_accounting'][arm]==dict(configured=20,completed=20),'runtime warmup count differs')
            for stage in ('full',)+stages:
                values=[r[stage+'_ms'] for r in rows if r['arm']==arm and r['phase']=='measured']
                d=distribution(values,'ms');d['maximum']=d.pop('max');d.pop('quantile_method')
                for key,value in d.items():
                    if numeric(value) or value is None:close(run['summaries'][arm][stage][key],value,'runtime summary '+arm+'/'+stage+'/'+key)
                    else:require(run['summaries'][arm][stage][key]==value,'runtime summary metadata differs')
                self.counts['runtime_distributions']+=1
        return dict(rows=600,warmup=80,measured=520,distributions=self.counts['runtime_distributions'],exact_schedule_and_fresh_saved_call_ledger_verified=True,full_interval_partition_verified=True,all_saved_outputs_match_geometry=True,actual_GPU_or_clock_not_recreated=True)


def not_bool(value): return not bool(value)


def output_guard(root,path):
    path=Path(path).absolute();require(not path.is_symlink(),'receipt output is a symlink');target=path.resolve();doc=root/'_docs/experiments'/NAME
    require(not target.exists(),'preserve existing receipt/output')
    if target.is_relative_to(root):require(target.parent==doc.resolve() and target.name.startswith('REVIEW_') and target.suffix=='.json','repository output must be a new own REVIEW_*.json receipt')
    for base in (Path('/home/minjae/Documents/github/pallet-pose'),Path('/home/minjae/Documents/github/pallet-pose-handoff-20261006'),Path('/dev/shm/pallet-kp-supervision-repair-private-20261010')):
        require(not target.is_relative_to(base) or target.is_relative_to(doc.resolve()),'receipt output overlaps protected source/baseline/weights')
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);parser.add_argument('--input',type=Path);parser.add_argument('--runtime-input',type=Path);parser.add_argument('--failed-runtime-input',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--require-runtime',action='store_true')
    args=parser.parse_args();root=args.root.resolve();doc=root/'_docs/experiments'/NAME;folder=args.input or doc;target=output_guard(root,args.output or doc/'REVIEW_CHECKS.json')
    started=time.monotonic();review=Review(root,folder,args.require_runtime,args.runtime_input,args.failed_runtime_input);checks=[]
    for group in ('inputs','observation','geometry','scored','metrics','runtime'):
        try:
            detail=getattr(review,group)();checks.append(dict(name=group,status='PENDING' if detail.get('pending') else 'PASS',details=detail))
        except Exception as exc:checks.append(dict(name=group,status='FAIL',error=type(exc).__name__+': '+str(exc)));break
    failed=any(r['status']=='FAIL' for r in checks);pending=any(r['status']=='PENDING' for r in checks)
    result=dict(schema='boundary_refiner_public_arithmetic_review_v2',status='FAIL' if failed else 'PARTIAL' if pending else 'PASS',complete=not(failed or pending),checks=checks,
        verifier=dict(path=str(Path(__file__).relative_to(root)),sha256=sha(__file__),bytes=Path(__file__).stat().st_size),verified_inputs=list(review.used.values()),unopened_external_declarations=review.unopened,
        arithmetic_counts=dict(review.counts),limits=LIMITS,actual_execution_in_this_review=dict(detector_forwards=0,head_forwards=0,optimizer_updates=0,PnP=0,rays=0,private_GT_reads=0,timing_calls=0),wall_seconds=time.monotonic()-started)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],checks={r['name']:r['status'] for r in checks},output=str(target)),ensure_ascii=False))
    return 1 if failed else 0


if __name__=='__main__':raise SystemExit(main())
