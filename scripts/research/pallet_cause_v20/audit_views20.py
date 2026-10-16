"""Independent geometric and frozen-proposal verification of synthetic base-quality views."""
from context20 import *
from model18 import VectorRefiner

def main():
    setup();rows=[]
    for seed in SEEDS:
        O,rep,gt,valid,ids,kind=proposal_training(seed);view=torch.load(RUN/f'QUALITY_VIEWS_seed{seed}.pt',weights_only=True);assert view['ids']==ids;v=valid.cpu().numpy();b=view['bases'].numpy().astype(float);d=view['deltas'].numpy();truth=gt.cpu().numpy().astype(float);original=rep['base'].cpu().numpy();diag=(O['box'][:,2:]-O['box'][:,:2]).clamp_min(1).norm(dim=-1).cpu().numpy();assert np.array_equal(b[0],original);assert np.array_equal(b[1][v],truth[v]);assert all(np.array_equal(z[~v],original[~v]) for z in b)
        radius_error=0.
        for j,radius in enumerate((0.,.01,.03),1):
            error=np.abs(np.linalg.norm(b[j]-truth,axis=-1)-radius*diag[:,None]);radius_error=max(radius_error,float(error[v].max()));assert error[v].max()<.0002
        proposer=VectorRefiner(seed).cuda().eval().requires_grad_(False);proposer.load_state_dict(torch.load(Q.RUN/'models'/f'line_vector_seed{seed}'/'final.pt',weights_only=True)['state_dict']);maximum=0.
        with torch.no_grad():
            for j in range(4):
                sl=slice(0,16);r=subset(rep,sl);r['base']=view['bases'][j,sl].cuda();out=proposer(subset(O,sl),r,'line_vector');error=float(np.abs((out['points']-r['base']).cpu().numpy()-d[j,sl]).max());maximum=max(maximum,error);assert error<.001
        rows.append(dict(seed=seed,original_ids_and_base_equal=True,correct_base_exact_GT=True,invalid_points_unchanged=True,maximum_radius_error_px=radius_error,proposal_replay_max_px=maximum,observations=len(ids)));del proposer
    write(RUN/'QUALITY_VIEW_AUDIT.json',dict(status='PASS',rows=rows,scope='Synthetic training augmentation only; global frozen parent context retained, no claim that augmented base reproduces full real parent behavior.'));print('QUALITY VIEW AUDIT PASS',flush=True)

if __name__=='__main__':main()
