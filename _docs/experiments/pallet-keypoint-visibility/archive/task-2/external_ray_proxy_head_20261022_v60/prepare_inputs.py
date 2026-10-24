from common import *
import numpy as np
from collections import Counter
assert not(ROOT/'PREPARED.json').exists();u=rows(B/'pallet-boundary-action-repair-20261018-v1/d1_upstream_v1/SYNTHETIC1003_UPSTREAM.jsonl.gz');assert len(u)==1003;lock=read(B/'pallet-statewise-end-to-end-recovery-v1/synthetic_v1/DATA_SPLIT_LOCK.json');scenario={r['id']:r['scenario_id']for r in lock['all_rows']};tr=[i for i,x in enumerate(u)if x['split']=='train'];va=[i for i,x in enumerate(u)if x['split']=='validation'];assert len(tr)==768 and len(va)==235
assert not({scenario[u[i]['id']]for i in tr}&{scenario[u[i]['id']]for i in va});assert not({u[i]['image_sha256']for i in tr}&{u[i]['image_sha256']for i in va})
inv={(r['id'],int(r['native_corner'])):r for r in csv.DictReader(open(V59/'PROPOSED_PROXY_LABEL_INVENTORY.csv'))};y=np.array([[int(inv[(x['id'],k)]['proposed_blockage_target'])for k in range(8)]for x in u],np.float32);valid=np.array([[inv[(x['id'],k)]['proposed_supervision_valid']=='True'for k in range(8)]for x in u]);np.savez_compressed(ROOT/'SYNTH_LABELS.npz',y=y,valid=valid,train=np.array(tr),validation=np.array(va));save('PREPARED_INPUTS_SYNTH.json.gz',u)
# Inference-only real inputs: no references/labels/real-CAL model/statistics imported.
wood=[x for x in rows(B/'pallet-boundary-action-repair-20261018-v1/d1_upstream_v1/REAL238_UPSTREAM.jsonl.gz')if x['partition']!='CAL'and abs(x['known_physical_dimensions_xyz_m'][0]-.8)<1e-9];square=[x for x in rows(ROOT.parent/'ocpr_wood_square_transfer_20261022_v50/ACTUAL_INPUTS122.jsonl.gz')if x['partition']=='SQUARE_YOLO'];assert len(wood)==105 and len(square)==118
real=[]
for group,rr in [('WOOD_PRIMARY105',wood),('SQUARE118',square)]:
 for x in rr:
  r={k:x[k]for k in ['id','image','image_sha256','raw_hw','K','initial_points','initial_pose','native_xyz','known_physical_dimensions_xyz_m','upstream_provenance']};r['group']=group;r['recording']=x['session'];r['partition']=x['partition'];r['symmetry_order']=2 if group.startswith('WOOD')else 4;r['reference_kind']='historical reconstructed geometric reference, not independent physical GT';real.append(r)
save('PREPARED_INPUTS_REAL.json.gz',real)
save('PREPARED.json',dict(UTC=utc(),synthetic_frames=1003,TRAIN=768,VAL=235,train_queries=int(valid[tr].sum()),train_positive=int(y[tr][valid[tr]].sum()),VAL_queries=int(valid[va].sum()),VAL_positive=int(y[va][valid[va]].sum()),source_scenario_ids_disjoint=True,source_RGB_hashes_disjoint=True,unverified='full original paired-render family/assets/background lineage not independently reconstructed; repeated VAL and upstream detector validation exposure remain',real_inputs_only=dict(wood105=True,square118=True,no_reference_labels_in_input_file=True),protocol_sha256=sha(ROOT/'PROTOCOL.json'),source_split_hash=sha(B/'pallet-statewise-end-to-end-recovery-v1/synthetic_v1/DATA_SPLIT_LOCK.json'),label_inventory_hash=sha(V59/'PROPOSED_PROXY_LABEL_INVENTORY.csv'),training_updates=0))
print('INPUTS_PREPARED_SYNTH1003_REAL223; TRAINING0',flush=True)
