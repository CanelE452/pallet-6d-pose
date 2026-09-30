"""Apply explicit pre-fit matched251 amendment; no new model inference."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.pallet_pose_stable_improvement_20261001_v1 import prepare_diverse as P


def main():
    p=P.read(P.DOC/'DATA_PREPARATION_PROTOCOL.json');selection=P.read(P.RAW/'DATA_SELECTION.json');gpu=P.read(P.DOC/'DATA_PREPARATION_GPU.json')
    amendment=P.DOC/'PROTOCOL_AMENDMENT_01.json'
    assert amendment.exists(),'Root must freeze explicit pre-training amendment first'
    failure=gpu['failures'];assert len(failure)==1 and failure[0]['reason']=='bbox_iou_below_existing_0.5'
    failed=failure[0]['id'];assert failed=='PLASTIC__10957de42de4bd188099b48c43743677039e60825001704bf19e1ca5e7ff3ff5'
    removed=max(selection['SINGLE252'],key=lambda r:(r['image']['sha256'],r['id']))
    assert removed['id']=='DAY264__005627'
    single=[r for r in selection['SINGLE252'] if r['id']!=removed['id']]
    receipts={r['id']:r for r in gpu['receipts']}
    diverse=[dict(r,pair=receipts[r['id']]['pair']) for r in selection['DIVERSE252'] if r['id']!=failed]
    assert len(single)==len(diverse)==251
    arms=dict(SINGLE251=single,DIVERSE251=diverse);summaries={}
    excluded=set(p['excluded_eval_sha'])
    for arm,rows in arms.items():
        errors=[];support=0;patched=0;min_iou=1.;per_frame=[]
        assert len({r['id'] for r in rows})==len({r['image']['sha256'] for r in rows})==251
        assert not excluded&{r['image']['sha256'] for r in rows}
        for r in rows:
            P.verify(r['image']);P.verify(r['pair']);x=torch.load(ROOT/r['pair']['path'],map_location='cpu',weights_only=False)
            pair=x['pair'];meta=x['metadata'];assert meta['id']==r['id'] and meta['R0_actually_rerun']
            assert np.array_equal(pair['CLEAN']['target_valid'],pair['OCC']['target_valid'])
            assert pair['OCC']['target_valid'].any() and not pair['OCC']['target_valid'][8]
            for z in pair.values():
                inverse=P.E.N.C.transform_points(z['target'],np.linalg.inv(z['matrix']))
                np.testing.assert_allclose(inverse[z['target_valid']],np.array(meta['target_original'])[z['target_valid']],atol=1e-4,rtol=0)
                assert z['rgb'].shape==(3,384,288) and z['points'].shape==(9,2)
            raw=P.E.P.top(meta['raw_prediction']);occ=P.E.P.top(meta['occluded_R0_prediction']);match=P.iou(raw['box_xyxy'],occ['box_xyxy']);assert match>=.5;min_iou=min(min_iou,match)
            z=pair['OCC'];valid=z['target_valid']&z['valid'];err=np.linalg.norm(z['points'][valid]-z['target'][valid],axis=1)/z['matrix'][0,0]
            errors.extend(err.tolist());n=int(z['target_valid'].sum());support+=n;patch=bool(meta['plans']);patched+=patch
            frame=dict(id=r['id'],recording_id=r['recording_id'],supervised_corners=n,input_error_gt_kind='fixed_pseudo_target_not_physical_GT',
                paired_bbox_iou=match,patched=patch,hard20_input_corners=int((err>20).sum()),target_sha=meta['target_sha'],mask_sha=meta['mask_sha'])
            r['support']=n;r['patched']=patch;r['paired_bbox_iou']=match;per_frame.append(frame)
        summaries[arm]=dict(images=251,recordings=dict(Counter(r['recording_id'] for r in rows)),supervised_corners=support,
            patched=patched,minimum_paired_bbox_iou=min_iou,pseudo_error_valid_corners=len(errors),
            input_pseudo_error_median_px=float(np.median(errors)),input_pseudo_error_P90_px=float(np.quantile(errors,.9)),
            hard20_input_corners=int((np.array(errors)>20).sum()),per_frame=per_frame)
    orders={}
    for seed in (1,2,3):
        array=np.random.default_rng(6400+seed).integers(0,251,size=(300,8));assert array.min()>=0 and array.max()<251
        path=P.RAW/f'REAL_ORDER_251_SEED{seed}.pt';P.tensor_save(path,torch.from_numpy(array));orders[str(seed)]=P.bind(path)
    final=dict(arms=arms,paired_count=251,selection=dict(algorithm=p['selection'],seed=20261001,original_count=252,
        original_selection=P.bind(P.RAW/'DATA_SELECTION.json'),candidate_counts=selection['target_eligible_recordings'],
        excluded_eval_sha=p['excluded_eval_sha'],preselected_ids=[r['id'] for r in selection['DIVERSE252']],
        pair_failures=failure,removed_single=removed,amendment=P.bind(amendment),resampling=False,guard_relaxed=False),
        source_orders=selection['source_orders'],real_order=orders['1'],real_orders=orders,initialization=selection['initialization'],
        target_recipe_bindings=p['target_recipe_bindings'],cache_bindings=p['cache_bindings'],data_audit=p['data_audit'],
        data_protocol=P.bind(P.DOC/'DATA_PREPARATION_PROTOCOL.json'),data_selection=P.bind(P.RAW/'DATA_SELECTION.json'),
        actual_masked_R0_forwards=gpu['actual_masked_R0_forwards'],smoke_forwards=gpu['smoke_forwards'],model_forward_total=gpu['model_forward_total'],
        parity=gpu['parity'],GT_input=False,summaries=summaries,finalizer=P.bind(Path(__file__)))
    P.save(P.DOC/'INPUTS.json',final)
    P.save(P.DOC/'DATA_PREPARATION_FINAL.json',dict(complete=True,matched_unique=251,arms=list(arms),amendment=P.bind(amendment),
        inputs=P.bind(P.DOC/'INPUTS.json'),source_order_shared=True,real_orders_shared_across_arms=orders,
        raw_eval_image_overlap=0,evaluation_coordinates_opened=0,optimizer_updates=0,new_forward_after_amendment=0,
        original_selected=252,original_ready=251,failed_id=failed,removed_control_id=removed['id'],
        summaries={k:{a:b for a,b in v.items() if a!='per_frame'} for k,v in summaries.items()}))
    print('FINAL_INPUTS_READY', {k:v['recordings'] for k,v in summaries.items()},flush=True)


if __name__=='__main__':main()
