"""Verify cache bindings and replay the public integrated module after reporting."""
from context25 import *
from prepare25 import ARMS
from eval25 import inputs
from network25 import ObservationCorrectionNetwork

def main():
    setup();bindings={};old=read(PREV.RUN/'TRAIN_PROTOCOL.json')
    for key in ('parents','targets','new_parent_bindings','new_target_bindings'):
        for p,h in old[key].items():assert sha(ROOT/p)==h,p;bindings[p]=h
    edges=[]
    for path in sorted((RUN/'edge_cache').glob('*.json')):
        meta=read(path);tensorpath=path.with_suffix('.pt');assert sha(tensorpath)==meta['sha256'];key=path.stem
        split=next(p for p in ('new_train','new_check','heldout','synth_val','real') if key.startswith(p+'_'));c=key[len(split)+1:]
        if c=='opposite20':mp=RUN/'cache'/split/c/'manifest.json'
        elif split in ('new_train','new_check'):mp=PREV.RUN/'cache'/split/c/'manifest.json'
        elif split=='heldout':mp=C.RUN/'cache'/split/c/'manifest.json'
        elif c in ('crop10','crop30'):mp=V.RUN/'cache'/split/c/'manifest.json'
        else:mp=C.V11/'cache'/split/c/'manifest.json'
        m=read(mp);assert digest(json.dumps(m,sort_keys=True))==meta['manifest_digest'];assert [r['id'] for r in m['rows'] if r['detected']]==meta['rows'];bindings[str(mp.relative_to(ROOT))]=sha(mp)
        for name,h in m['chunk_sha256'].items():assert sha(mp.parent/name)==h;bindings[str((mp.parent/name).relative_to(ROOT))]=h
        bindings[str(tensorpath.relative_to(ROOT))]=sha(tensorpath);edges.append(dict(key=key,detected=len(meta['rows'])))
    replay=[]
    for seed in SEEDS:
        m,O,_=inputs('real','crop20',seed);O={k:v[:3].cuda() for k,v in O.items()}
        for arm in ARMS:
            model=ObservationCorrectionNetwork(seed,arm)
            with torch.inference_mode():out=model(O)
            p=np.load(RUN/'predictions/real/crop20'/f'{arm}_seed{seed}.npz');err=float(np.abs(out['points'].cpu().numpy()-p['points'][p['detected']][:3]).max());assert err<.001;replay.append(dict(seed=seed,arm=arm,max_error_px=err));del model
    write(RUN/'PROVENANCE_CHECK.json',dict(status='PASS',files=bindings,edge_caches=edges,public_module_replays=replay,network_sha256=sha(KIT/'network25.py'),maximum_module_error_px=max(r['max_error_px'] for r in replay)));print('PROVENANCE25 PASS',len(bindings),len(replay),flush=True)

if __name__=='__main__':main()
