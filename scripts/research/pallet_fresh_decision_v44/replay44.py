from features44 import *
sys.path.insert(0,str(KIT.parent/'pallet_continuous_lines_v36'))
from model36 import Model
from context36 import setup36
from data32 import render_data
import torch

def main():
 setup36();rows=[]
 for seed in SEEDS:
  m,O,S,targets=render_data(seed,'check','clean');ids=[r['id'] for r in m['rows'] if r['detected']];old=torch.load(L.V36/'cache/render_check/clean'/f'seed{seed}.pt',weights_only=True,map_location='cpu');assert ids==old['ids'];net=Model(seed).cuda().eval();net.load_state_dict(torch.load(L.V36/'models'/f'both_seed{seed}'/'final.pt',weights_only=True,map_location='cpu')['state_dict']);oo={k:O[k][:8].cuda() for k in ('features','content','edge','box','image_hw')};ss={k:S[k][:8].cuda() for k in ('base','strong')}
  with torch.no_grad():x=net.core.capture(oo,ss);out=net(x,'both')
  diffs={k:float((x[k].cpu()-old['x'][k][:8]).abs().max()) for k in ('features','query','v34','context')};assert max(diffs.values())<1e-3,diffs
  saved=np.load(L.V36/'predictions/render_check/clean'/f'both_seed{seed}.npz');det=saved['detected'];errs={k:float(abs(out[k].cpu().numpy()-saved[k][det][:8]).max()) for k in ('normal','offset','score','points')};assert max(errs.values())<1e-3,errs;rows.append(dict(seed=seed,frames=8,capture_max=diffs,prediction_max=errs));print('REPLAY44',rows[-1],flush=True);del net
 write(RUN/'UPSTREAM_REPLAY.json',dict(status='PASS',rows=rows,tolerance=1e-3,scope='8 existingcleancheck frames perseed fixedmodelforward parity on current processlocaldriver; not new performance evaluation.'))
if __name__=='__main__':main()
