"""저장 SOURCE 관측 산술만 재집계: python VERIFY_SAVED_QUERY_OBSERVATIONS.py

Package/results의 COMPLETE.json, SUMMARY.json, OLD_STATE_BY_Q_OWNER.csv,
QUERY_OBSERVATIONS8000.jsonl.gz(우선) 또는 .jsonl만 읽는다. 절대 source
pointer를 따르지 않고 GT/NPZ/PNG/모델을 열지 않는다. Semantic classifier
CM이나 원 teacher 권위의 재인증이 아니다. 표준 Python만 필요하다.
"""
import csv
import gzip
import hashlib
import io
import json
import math
import os
import stat
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).absolute().parent/'results'
STATES = ('D','E','S','U')
OWNERS = ('QUERY_CONTINUOUS_OOB','QUERY_NEAREST_OOB','UNIQUE_OWN',
          'UNIQUE_FOREIGN','OVERLAP','UNASSIGNED')

def need(ok, message):
    if not ok:
        raise AssertionError(message)

def read_file(name):
    p = ROOT/name
    need('v112' not in str(p).lower(), 'Forbidden literal before filesystem')
    for x in reversed((p,*p.parents)):
        need(not stat.S_ISLNK(os.lstat(x).st_mode), 'Symlink path')
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        return f.read()

def sha(data): return hashlib.sha256(data).hexdigest()
def close(a,b): return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-9)
def finite_pair(x): return isinstance(x,list) and len(x)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in x)
def matrix(counts): return {s:{o:counts[s,o] for o in OWNERS} for s in STATES}

def window_counts(w,own,scale):
    # 양성 depth와 mask-only owner 관측은 별도 축이다.
    for k in ('inbounds9','rawdepth9','depth_mm9','ownfull9','ownvis9','ownercount9','ownerbits9'):
        need(isinstance(w[k],list) and len(w[k])==9,'Nine saved window values '+k)
    flags,depth,full,vis,counts,bits = (w[k] for k in ('inbounds9','rawdepth9','ownfull9','ownvis9','ownercount9','ownerbits9'))
    for j in range(9):
        need(all(type(a[j]) is bool for a in (flags,full,vis)),'Boolean window values')
        need(type(depth[j]) is int and 0<=depth[j]<=65535,'uint16 raw depth')
        need(type(counts[j]) is int and 0<=counts[j]<=64,'Owner count')
        need(type(bits[j]) is int and 0<=bits[j]<2**64 and bits[j].bit_count()==counts[j],'Owner bits/count')
        need(vis[j] == bool(bits[j] & (1<<own)),'Own visible bit identity')
        need(close(w['depth_mm9'][j],depth[j]*scale),'Raw depth × saved scale')
        if not flags[j]:
            need(depth[j]==counts[j]==bits[j]==0 and not full[j] and not vis[j],'Outside window zero padding')
    expected = [sum(flags),sum(a and d>0 for a,d in zip(flags,depth)),
        sum(a and c==1 and v for a,c,v in zip(flags,counts,vis)),
        sum(a and c==1 and not v for a,c,v in zip(flags,counts,vis)),
        sum(a and c>1 for a,c in zip(flags,counts)),sum(a and c==0 for a,c in zip(flags,counts)),sum(full)]
    need(w['counts']==expected and sum(expected[2:6])==expected[0],'Seven evidence counts/partition')
    need(w['unique_own_and_positive_count']==sum(a and c==1 and v and d>0 for a,c,v,d in zip(flags,counts,vis,depth)),'Own AND positive depth count')
    need(w['same_foreign9'] is (all(flags) and all(c==1 for c in counts) and not any(vis) and len(set(bits))==1),'Same foreign owner across nine')
    need(type(w['continuous_valid']) is bool and w['nearest_valid'] is flags[4],'Saved validity/center flag')
    return flags,depth,full,vis,counts,bits

def main():
    complete=json.loads(read_file('COMPLETE.json'))
    need(complete['completed'] is True and complete['PASS'] is True,'Completed builder output')
    output=complete['outputs']  # 절대 path 필드는 inert; 고정 basename만 사용한다.
    def owned(name):
        data=read_file(name);binding=output[name]
        need(sha(data)==binding['sha256'],'Saved output SHA '+name)
        if 'size_bytes' in binding: need(len(data)==binding['size_bytes'],'Saved output size '+name)
        return data
    summary=json.loads(owned('SUMMARY.json'))
    csv_data=list(csv.DictReader(io.StringIO(owned('OLD_STATE_BY_Q_OWNER.csv').decode('utf-8-sig'))))
    try:
        encoded=read_file('QUERY_OBSERVATIONS8000.jsonl.gz')
        stream=gzip.GzipFile(fileobj=io.BytesIO(encoded),mode='rb'); encoding='gzip'
    except FileNotFoundError:
        encoded=read_file('QUERY_OBSERVATIONS8000.jsonl')
        stream=io.BytesIO(encoded);encoding='identity'
    table=Counter();labels=Counter();bins=defaultdict(Counter);frame_keys={};instance_key=None
    raw_digest=hashlib.sha256();raw_size=0;rows=0;nearest_oob=continuous_oob=unique_own=0
    for line in stream:
        raw_digest.update(line);raw_size+=len(line)
        if not line.strip(): continue
        r=json.loads(line);i,k=divmod(rows,8)
        need(r['slot_index']==rows and r['instance_row_index']==i and r['logical_aux_slot']==k,'Original1000×8 order')
        need(r['input_id']==f'Y109_SOURCE_{i:06d}','Source input identity')
        identity=tuple(r[x] for x in ('input_id','instance_row_index','frame_index','scene_id','source_frame_id','source_instance_id','object_id'))
        if k==0: instance_key=identity
        need(identity==instance_key,'Eight slots share original instance/frame')
        frame=int(r['frame_index']);fk=(r['scene_id'],r['source_frame_id'])
        need(0<=frame<700 and (frame not in frame_keys or frame_keys[frame]==fk),'Original source frame identity')
        frame_keys[frame]=fk
        need(r['native_mapping']=='UNKNOWN' and r['new_native_visibility_target'] is None and r['training_admission'] is False,'No native relabel/admission')
        need(r['GT_teacher_only_no_deployment_input_change'] is True,'Teacher/input authority separation')
        q,ref=r['q'],r['r_GT_teacher_only'];need(finite_pair(q) and finite_pair(ref),'Finite q/reference')
        old=r['original_teacher_immutable'];state=old['raw_surface_teacher_state'];need(state in STATES,'Original D/E/S/U')
        need(old['query_xy_input_copy']==q and old['reference_xy_GT_only']==ref,'Saved immutable q/reference copies')
        need(type(r['L_input']) in (int,float) and math.isfinite(r['L_input']) and r['L_input']>0,'Saved positive L')
        if 'roi_side_input_copy' in old: need(old['roi_side_input_copy']==r['L_input'],'Saved L copy')
        need(old['instance_row_index']==i and old['frame_index']==frame and old['logical_aux_slot']==k,'Original teacher identity fields')
        distance=math.sqrt((q[0]-ref[0])**2+(q[1]-ref[1])**2)
        d=r['constructed_jitter_distance_px']
        need(0<=d<=4 and close(d,distance) and close(d,old['query_reference_error_px']),'Constructed q−r distance')
        own=r['source_instance_id'];need(type(own) is int and 0<=own<64,'Own instance bit index')
        scale=old['depth_scale_to_mm'];need(type(scale) in (int,float) and math.isfinite(scale) and scale>0,'Saved depth scale')
        w=r['q_observation_not_semantic_target'];flags,depth,full,vis,counts,bits=window_counts(w,own,scale)
        need(w['nearest_xy']==[math.floor(v+.5) for v in q],'Nearest floor(q+.5), no query snapping')
        need(w['continuous_valid'] is old['query_continuous_valid'] and w['nearest_valid'] is old['query_nearest_valid'],'Immutable query validity flags')
        owner=('QUERY_CONTINUOUS_OOB' if not w['continuous_valid'] else 'QUERY_NEAREST_OOB' if not flags[4]
            else 'OVERLAP' if counts[4]>1 else 'UNASSIGNED' if counts[4]==0 else 'UNIQUE_OWN' if vis[4] else 'UNIQUE_FOREIGN')
        need(w['state']==owner,'Center owner enum from saved flags/bits')
        expected_center=dict(ownership=owner,raw_depth=depth[4],depth_mm=w['depth_mm9'][4],
            positive_depth=bool(flags[4] and depth[4]>0),own_full=full[4],own_visible=vis[4],owner_count=counts[4],owner_bits=bits[4])
        need(r['q_centre_observation']==expected_center,'Center/window saved-value identity')
        table[state,owner]+=1;labels[state]+=1
        b=next(j for j,edge in enumerate((1,2,3,4)) if d<=edge)
        bins['bin_'+str(b)][state,owner]+=1
        if d<=3: bins['distance_le3'][state,owner]+=1
        nearest_oob+=not w['nearest_valid'];continuous_oob+=not w['continuous_valid'];unique_own+=owner=='UNIQUE_OWN'
        rows+=1
    stream.close()
    need(rows==8000 and len(frame_keys)==len(set(frame_keys.values()))==700,'8000 slots/700 unique frames')
    original=output['QUERY_OBSERVATIONS8000.jsonl']
    need(raw_digest.hexdigest()==original['sha256'],'Exact decompressed original JSONL SHA')
    if 'size_bytes' in original: need(raw_size==original['size_bytes'],'Exact decompressed original byte size')
    need(dict(labels)=={'D':1181,'E':340,'S':1917,'U':4562},'Original associated-r D/E/S/U census')
    need(summary['completed'] is True and summary['PASS'] is True,'Summary completion')
    need((summary['instances'],summary['slots'],summary['frames'])==(1000,8000,700),'Summary census')
    need(summary['original_state_counts']==dict(labels) and summary['old_associated_r_state_x_literal_q_observation']==matrix(table),'Summary old state × q-owner arithmetic')
    need(len(csv_data)==24 and len({(r['original_associated_r_state'],r['literal_q_observation']) for r in csv_data})==24,'24 unique CSV cells')
    need({(r['original_associated_r_state'],r['literal_q_observation']) for r in csv_data}=={(s,o) for s in STATES for o in OWNERS},'CSV full cell coverage')
    for r in csv_data: need(int(r['rows'])==table[r['original_associated_r_state'],r['literal_q_observation']],'CSV cell recount')
    need(summary['jitter_bins_definition']==['[0,1]','(1,2]','(2,3]','(3,4]'],'Original jitter cuts')
    need(set(summary['jitter_bin_tables'])==set(bins),'Jitter stratum keys')
    need(sum(sum(bins['bin_'+str(k)].values()) for k in range(4))==8000,'Four-bin partition')
    need(bins['distance_le3']==sum((bins['bin_'+str(k)] for k in range(3)),Counter()),'≤3 exact subset partition')
    for key,cc in bins.items():
        need(summary['jitter_bin_tables'][key]==dict(rows=sum(cc.values()),table=matrix(cc)),'Jitter summary '+key)
    need((summary['q_nearest_OOB'],summary['q_continuous_OOB'],summary['q_mask_only_unique_own_count'])==(nearest_oob,continuous_oob,unique_own),'Summary validity/unique-own counts')
    need(summary['native_mapping']=='UNKNOWN' and summary['new_training_admission'] is False and summary['observation_table_is_not_semantic_classifier_confusion'] is True,'No semantic classification/admission promotion')
    print(json.dumps(dict(completed=True,PASS=True,rows=rows,instances=1000,frames=700,
        original_state_counts=dict(labels),old_state_by_q_owner=matrix(table),
        jitter_bin_rows={k:sum(v.values()) for k,v in bins.items()},
        JSONL_encoding=encoding,package_JSONL_sha256=sha(encoded),
        decompressed_original_JSONL_sha256=raw_digest.hexdigest(),decompressed_original_size_bytes=raw_size,
        scope='Saved SOURCE observation arithmetic, not semantic classifier CM or original teacher recertification',
        image_dimensions_not_reparsed=True,original_raw_GT_NPZ_PNG_or_model_reads=0,new_labels=0,fit=0),allow_nan=False))

if __name__=='__main__':
    main()
