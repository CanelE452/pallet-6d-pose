"""Does the fixed Hough target range contain enough true incident directions?"""
from context29 import *
from eval27 import population,conditions,inputs
from model29 import INCIDENT,ROLE_PAIRS,geometry
from plpose_local.geometry import EDGES
from plpose_local.losses import line_targets

def main():
    torch.set_num_threads(2);rows=[]
    for split in EVAL_SPLITS:
        for c in conditions(split):
            m,ts=population(split,c);det=np.array([r['detected'] for r in m['rows']]);_,O,r=inputs(split,c,1);del r
            gt=torch.tensor([t['points'] for t in ts],dtype=torch.float32)[det];valid=torch.tensor([t['valid'] for t in ts])[det];kind=C.P.target_type(ts)[det]
            center,half=geometry(O);uv=(gt-center[:,None])/half[:,None];u,v=torch.tensor(EDGES).T;d=uv[:,v]-uv[:,u];n=torch.stack((-d[...,1],d[...,0]),-1);n=n/n.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            _,rangeok=line_targets(gt,O['box'],36,65);known=valid[:,u]&valid[:,v];ik=known[:,INCIDENT];ir=rangeok[:,INCIDENT];inn=n[:,INCIDENT]
            pairs=[]
            for a,b in ROLE_PAIRS:
                detn=inn[:,:,a,0]*inn[:,:,b,1]-inn[:,:,a,1]*inn[:,:,b,0]
                pairs.append(ik[:,:,a]&ik[:,:,b]&ir[:,:,a]&ir[:,:,b]&(detn.abs()>=.1))
            recoverable=torch.stack(pairs,-1).any(-1)&(uv.abs()<=4).all(-1)
            for typ,k in (('intact_proxy',0),('hidden',1),('outside',2)):
                mask=valid&(kind==k);complete=mask&ik.all(-1)
                if not mask.any():continue
                rows.append(dict(split=split,condition=c,type=typ,detected_target_points=int(mask.sum()),all_incident_roles_GT_known=int(complete.sum()),true_lines_sufficient_pair_fraction=float(recoverable[complete].float().mean()) if complete.any() else None,mean_incident_roles_out_of_Hough_target_range=float((~ir).float().sum(-1)[complete].mean()) if complete.any() else None,corner_beyond_decoder_range_fraction=float((uv.abs()>4).any(-1)[complete].float().mean()) if complete.any() else None))
    write(RUN/'SUPPORT_DIAGNOSIS.json',dict(rows=rows,no_training=True,scope='Uses true structural lines and exact training target-range/pair-angle masks; not physical visibility or measured DHT recall. Fractions require all3 incident roles to have valid GT endpoints, avoiding treating missing manual labels as absent lines.'))
    print('SUPPORT29 COMPLETE',flush=True)

if __name__=='__main__':main()
