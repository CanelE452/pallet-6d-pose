"""Original frozen feature caches, with supervision separated from inference."""
from pathlib import Path
import numpy as np,torch
from .a_common import read,context,POSE,dcp_env
from .geometry import build_bank

def affine(canvas_shape,input_shape):
    h,w=canvas_shape;ih,iw=input_shape;gain=min(640/h,640/w)
    offset=np.array([round((iw-round(w*gain))/2-.1),round((ih-round(h*gain))/2-.1)])
    return np.array([gain,gain]),offset

class Data:
    def __init__(self,source_root):
        self.root=Path(source_root);self.line=self.root/'data/pallet/results/pallet_line_pose_v1';self.dim=self.root/'data/pallet/results/pallet_dim_conditioned_p_v1'
        self.source=read(self.line/'SOURCE_MANIFEST.json');manifest=read(self.line/'cache/CACHE_MANIFEST.json')
        completion=read(self.line/'cache/CACHE_COMPLETE.json');assert completion['complete'] and completion['PASS']
        assert np.load(self.line/'cache/done.npy',mmap_mode='r').all()
        self.arrays={k:np.load(self.line/'cache'/v['file'],mmap_mode='r') for k,v in manifest['arrays'].items()}
        self.indices=np.array(manifest['record_indices']);self.side=dict(np.load(self.dim/'DIMENSION_SIDECAR.npz'));assert np.array_equal(self.indices,self.side['record_index'])
        self.norm=read(self.root/'_docs/experiments/pallet_dim_conditioned_p_v1/DIM_NORMALIZATION_LOCK.json')
        self.geometry=dict(np.load(self.root/'challenge/yolo_pose_one_model/pallet_translation_loss_v1/GEOMETRY_SIDETABLE.npz'))
        self.gindex={str(s):i for i,s in enumerate(self.geometry['stems'])}
        self.partitions=np.array([self.source['records'][i]['partition'] for i in self.indices])
        self.train_rows=np.flatnonzero((self.partitions=='train')&self.arrays['matched']&self.arrays['gt_valid'][:,:8].any(-1))
        self.oracle_rows=np.flatnonzero(self.partitions=='selection');self.eval_rows=np.flatnonzero(self.partitions=='heldout')
    def source_frame(self,row):
        r=self.source['records'][self.indices[row]];i=self.gindex[r['id']];g=self.geometry
        fx,fy,cx,cy=g['K'][i];K=np.array([[fx,0,cx-g['pad'][i]],[0,fy,cy-g['pad'][i]],[0,0,1.]])
        scale,offset=affine(r['prepared_shape_hw'],self.arrays['input_shape'][row]);pad=r['reflect_pad_px']
        q=np.array(self.arrays['points'][row],float);valid=np.array(self.arrays['point_valid'][row]);q=(q-offset)/scale-pad
        invalid=~valid;q[invalid]=np.array(self.arrays['points'][row],float)[invalid]
        xyz=g['dims'][i]
        truth=dict(R=g['R'][i],t=g['t'][i],xyz=xyz,body_R=g['R'][i],body_xyz=xyz,order=int(self.side['order'][row]))
        return dict(id=r['id'],session=r['scenario_id'],q=q,K=K,xyz=xyz,source=True,raw_hw=r['raw_shape_hw'],scale=scale,offset=offset+pad*scale,point_valid=valid,truth=truth,cache_row=int(row),partition=r['partition'])
    def batch(self,rows,device='cuda',supervision=True):
        keys=('p3','p4','points','boxes','point_valid','input_shape')+(('gt_points','gt_valid') if supervision else ())
        b={k:torch.from_numpy(np.array(self.arrays[k][rows],copy=True)).to(device) for k in keys}
        b['context']=torch.from_numpy(context(self.side['dimensions'][rows],self.side['order'][rows],self.norm,False)).to(device)
        if supervision:
            for k in ['permutations','group_valid']:b[k]=torch.from_numpy(self.side[k][rows].copy()).to(device)
        return b
    def real_frames(self):
        p=self.root/'data/pallet/results/paper_pose_metric_closure_v1';axis=read(p/'AXIS_REVIEW_MANIFEST.json')['frames_list'];gts=read(p/'GEOMETRY_RESOLVED_POSE_GT.json')['frames']
        frames=[]
        for r in sorted(axis,key=lambda r:r['frame_id']):
            captured=torch.load(self.dim/'DEV_cache'/f"{r['frame_id']}.pt",map_location='cpu',weights_only=False)
            qpack=captured['captured'];idx=qpack['selected_index'];q=None if idx is None else np.array(qpack['candidates'][idx]['keypoints_xy'],float)
            ann=read(self.root/r['annotation']);raw=ann['camera_data']['intrinsics'];K=np.array([[raw['fx'],0,raw['cx']],[0,raw['fy'],raw['cy']],[0,0,1.]])
            xyz=np.array(captured['dimensions'])[[0,2,1]];gt=gts[r['frame_id']];d=gt['physical_dimensions_m'];cf=np.array([d['across'],d['height'],d['along']]);Rcf=np.array(gt['R_gt_representative'])
            Q=np.eye(3) if abs(cf[0]-xyz[0])<1e-6 else POSE.rotations(4)[1]
            truth=dict(R=Rcf@Q,t=np.array(gt['t_gt']),xyz=xyz,body_R=Rcf,body_xyz=cf,order=captured['order'])
            scale,offset=affine(qpack['canvas_shape'],qpack['input_shape']);pad=qpack['added_border']
            valid=np.zeros(9,bool) if q is None else np.isfinite(q).all(-1)&~(q==-1).all(-1)
            frames.append(dict(id=captured['id'],session=captured['session'],q=q,K=K,xyz=xyz,source=False,raw_hw=captured['raw_hw'],scale=scale,offset=offset+pad*scale,point_valid=valid,truth=truth,captured=captured,axis=r,annotation=ann))
        return frames
    def real_batch(self,frames,device='cuda'):
        batches=[]
        for f in frames:
            c=f['captured']['captured'];idx=c['selected_index']
            if idx is None:raise ValueError('Use NoOp for absent detection, do not run scorer')
            cand=c['candidates'][idx];q=np.array(cand['keypoints_xy'])*f['scale']+f['offset'];box=(np.array(cand['box_xyxy']).reshape(2,2)*f['scale']+f['offset']).reshape(4)
            batch={k:torch.as_tensor(v)[None] for k,v in dict(points=q.astype('float32'),boxes=box.astype('float32'),point_valid=f['point_valid'],input_shape=c['input_shape']).items()}
            batch.update(p3=c['p3'],p4=c['p4']);batch['context']=torch.from_numpy(context(f['captured']['dimensions'][None],[f['captured']['order']],self.norm,False))
            batches.append(batch)
        return {k:torch.cat([b[k] for b in batches]).to(device) for k in batches[0]}

def network_bank(bank,frame,raw_network_q):
    a=bank['points'].copy();valid=frame['point_valid'];a[:,valid]=a[:,valid]*frame['scale']+frame['offset']
    a[0]=raw_network_q  # NoOp is exactly the input tensor, including its FP32 rounding.
    return a.astype('float32')
