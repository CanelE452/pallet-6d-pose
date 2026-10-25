"""가상 작은 packet/CSV만. 실제 SOURCE/feature/model 파일 접근 없음."""
import io
import csv
import json
import numpy as np
from verify_feature_packet import check_arrays, relative
from summarize_saved_pair import parse_rows, COLS
from source_metrics_math import self_test
from bound_io import lexical

def rejects(fn):
    try: fn()
    except ValueError: return
    raise AssertionError('negative fixture accepted')

q=np.array([[[j,j/2] for j in range(8)]]*2,np.float64)
g=dict(q8=q,L8=np.full((2,8),10.,np.float64),box_xyxy=np.array([[0.,0.,7.,3.5]]*2),
    original_available8=np.ones((2,8),bool),node_available=np.ones((2,8),bool),
    node_fraction=np.ones((2,8),np.float32),context_fraction=np.ones(2,np.float32))
group=np.array([0,1],np.int64);slot=np.array([0,1],np.int64);r=relative(g,group,slot)
condition=np.zeros((2,32),np.float32);condition[0,0]=1;condition[0,8:16]=1;condition[0,8]=0;condition[0,16:]=r[0].ravel()
d=dict(full_view=np.array([0,2],np.int64),central_view=np.array([1,3],np.int64),
    target=np.array([0,1],np.int8),family=np.array([0,1],np.int8),fit=np.array([True,False]),
    available=np.ones(2,bool),central4=np.ones(2,bool),row_id=np.array(['a','b']),condition=condition,
    fractions=np.ones((2,2),np.float32))
pixel=np.full((2,8,1154),.2,np.float32);pixel[:,:,1152:]=1
f={'NODE_PIXEL_FEATURES.npy':pixel,'NODE_CLS.npy':pixel[:,:,:384].copy(),
    'CONTEXT_CLS.npy':np.ones((2,384),np.float32),'ORIGINAL_ALL_FEATURES.npy':np.concatenate((pixel[group,slot],condition),1),
    'SAMPLE_GROUP_INDEX.npy':group,'SAMPLE_NODE_INDEX.npy':slot,'RELATIVE_XY8.npy':r}
rows=[dict(row_id=d['row_id'][i].item(),group_index=int(group[i]),active_slot=int(slot[i]),fit=bool(d['fit'][i]),
    q=q[group[i],slot[i]].tolist(),L=10.,original_query_available=True,crop_available=True,
    sample_index=i,target=int(d['target'][i]),family_code=int(d['family'][i])) for i in range(2)]
rows.append(dict(row_id='masked',group_index=0,active_slot=2,fit=True,q=q[0,2].tolist(),L=10.,
    original_query_available=True,crop_available=True,sample_index=None,target=-1,family_code=0))
counts=dict(groups=2,supported_nodes=16,known=2,FIT=1,DEV=1,all_original_rows=3,masked_target_minus1=1,context_supported_groups=2)
assert check_arrays(f,d,g,rows,counts)['all_known_active_NODE_first384_byte_exact']
bad={k:v.copy() for k,v in f.items()};bad['ORIGINAL_ALL_FEATURES.npy'][0,0]+=.001
rejects(lambda:check_arrays(bad,d,g,rows,counts))
bad={k:v.copy() for k,v in f.items()};bad['RELATIVE_XY8.npy'][1,0,0]=7
rejects(lambda:check_arrays(bad,d,g,rows,counts))
badrows=[dict(x) for x in rows];badrows[-1]['fit']=False
rejects(lambda:check_arrays(f,d,g,badrows,counts))
rejects(lambda:lexical('/invented/V112/forbidden.npy'))
rejects(lambda:lexical('/invented/tools/__pycache__/forbidden.pyc'))
s=io.StringIO(newline='');w=csv.DictWriter(s,fieldnames=COLS);w.writeheader()
w.writerow(dict(sample_index=0,row_id='toy',family=1,role='DEV',available=True,target=1,
    logit0=-1,logit1=0,selected_family_logit=0,prediction='E'))
assert parse_rows(s.getvalue().encode())[0]['prediction']=='E'
rejects(lambda:parse_rows(s.getvalue().replace(',0,E',',0,D').encode()))
self_test()
print(json.dumps(dict(completed=True,PASS=True,invented_packet_groups=4,csv_and_lexical_checks=4,
    frozen_gate_checks=4,actual_SOURCE_array_image_weight_or_result_reads=0)))
