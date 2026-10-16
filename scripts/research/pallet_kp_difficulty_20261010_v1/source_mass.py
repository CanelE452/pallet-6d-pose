"""One fixed CPU head set; compare joint class MAP and existence MAP.

Existence MAP chooses a correspondence iff summed probability of the 65
position bins exceeds probability of NONE. The position remains one argmax
bin; no averaging, calibration threshold search or model selection occurs.
"""
import base64
import hashlib
from pathlib import Path
import time
import zlib
from collections import Counter

import numpy as np
from .source_ceiling import C,DOC,CACHE,decode_choices,query_geometry,stats

PRIVATE=Path('/dev/shm/pallet-kp-difficulty-private-20261010')
DECODERS=('ORIGINAL_66_CLASS_MAP','FIXED_BINARY_EXISTENCE_MAP')


def decode_mass(logits):
    a=np.asarray(logits,np.float64);c=a[:,:65];peak=c.max(1)
    logmass=peak+np.log(np.exp(c-peak[:,None]).sum(1));logodds=logmass-a[:,65]
    probability=1/(1+np.exp(-logodds));choice=c.argmax(1)
    return np.where(logodds>0,choice,65),probability,logodds


def decode_logits(row):
    data=zlib.decompress(base64.b64decode(row['logits_f32_zlib_base64']))
    assert hashlib.sha256(data).hexdigest()==row['logits_uncompressed_sha256']
    return np.frombuffer(data,dtype='<f4').reshape(row['logits_shape'])


def analyze(logits,source,choices,probability=None):
    targets=source['targets'];lo=np.asarray(targets['lo']);weight=np.asarray(targets['weight']);valid=np.asarray(targets['valid'],bool)
    role=np.asarray(source['predicted_role_query_ids']);pos=valid&(lo<65);none=valid&(lo==65);ignored=~valid
    points=np.asarray(source['frozen_selected_points']);center,normal,queryvalid=query_geometry(points);accept=(choices<65)&queryvalid;ap=accept&pos
    target=lo+weight-32;offset=choices-32;decoded=decode_choices(points,choices);h,w=source['raw_hw']
    inframe=[c['id'] for c in decoded['corners'] if 0<=c['xy'][0]<w and 0<=c['xy'][1]<h];hidden=source['source_annotation_oracle_H']
    return dict(choices=choices.tolist(),selected_queries=int(accept.sum()),positive=int(pos.sum()),positive_accepted=int(ap.sum()),
        no_match=int(none.sum()),no_match_false_accepted=int((accept&none).sum()),ignored=int(ignored.sum()),ignored_accepted_without_truth=int((accept&ignored).sum()),
        selected_by_role=np.bincount(role[accept],minlength=3).tolist(),target_positive_by_role=np.bincount(role[pos],minlength=3).tolist(),
        positive_accepted_by_role=np.bincount(role[ap],minlength=3).tolist(),no_match_by_role=np.bincount(role[none],minlength=3).tolist(),
        no_match_false_accepted_by_role=np.bincount(role[accept&none],minlength=3).tolist(),
        accepted_positive_query_ids=np.flatnonzero(ap).tolist(),
        zero_offset_error_same_accepted_px=np.abs(target[ap]).tolist(),head_offset_error_same_accepted_px=np.abs(offset[ap]-target[ap]).tolist(),
        decoded_line_edges=[r['edge'] for r in decoded['lines']],decoded_corner_ids=[r['id'] for r in decoded['corners']],
        decoded_corners=decoded['corners'],in_frame_corner_ids=inframe,
        after_source_annotation_oracle_H_in_frame_corner_ids=sorted(set(inframe)-set(hidden)),
        input_mask_is_original_query_geometry_only=True,source_ignored_acceptance_not_claimed_valid=True)


def summary(rows):
    out={}
    for arm in ['GEOMETRY_ONLY','IMAGE_NO_ROLE','IMAGE_ROLE']:
        out[arm]={}
        for split in ['calibration','source_test']:
            rr=[r for r in rows if r['arm']==arm and r['partition']==split];out[arm][split]={}
            for decoder in DECODERS:
                a=[r['decoders'][decoder] for r in rr];positive=sum(r['positive'] for r in a);none=sum(r['no_match'] for r in a)
                baseline=[v for r in a for v in r['zero_offset_error_same_accepted_px']];learned=[v for r in a for v in r['head_offset_error_same_accepted_px']]
                out[arm][split][decoder]=dict(families=len(a),positive_targets=positive,positive_adoption_rate=sum(r['positive_accepted'] for r in a)/positive,
                    no_match_targets=none,no_match_false_acceptance=sum(r['no_match_false_accepted'] for r in a)/none,
                    ignored_accepted_without_truth=sum(r['ignored_accepted_without_truth'] for r in a),selected_queries=sum(r['selected_queries'] for r in a),
                    positive_accepted_by_role=np.sum([r['positive_accepted_by_role'] for r in a],0).tolist(),target_positive_by_role=np.sum([r['target_positive_by_role'] for r in a],0).tolist(),
                    no_match_false_accepted_by_role=np.sum([r['no_match_false_accepted_by_role'] for r in a],0).tolist(),no_match_by_role=np.sum([r['no_match_by_role'] for r in a],0).tolist(),
                    before_mask_corner_histogram=dict(Counter(len(r['decoded_corner_ids']) for r in a)),
                    before_mask_ge4=sum(len(r['decoded_corner_ids'])>=4 for r in a),in_frame_ge4=sum(len(r['in_frame_corner_ids'])>=4 for r in a),
                    after_source_annotation_oracle_H_in_frame_ge4=sum(len(r['after_source_annotation_oracle_H_in_frame_corner_ids'])>=4 for r in a),
                    zero_offset_same_accepted_error_px=stats(baseline),head_same_accepted_error_px=stats(learned),head_minus_zero_same_accepted_error_px=stats(np.asarray(learned)-baseline))
            original=[r['decoders'][DECODERS[0]] for r in rr];binary=[r['decoders'][DECODERS[1]] for r in rr]
            out[arm][split]['class_none_but_match_mass_gt_half_query_count']=sum(r['none_argmax_but_match_mass_gt_half_count'] for r in rr)
            out[arm][split]['rescued_positive_boundary_query_count']=sum(r['rescued_positive_boundary_count'] for r in rr)
    return out


def run():
    import torch
    from scripts.research.pallet_observation_refiner_20261009_v1 import model as M
    torch.set_num_threads(1);torch.set_num_interop_threads(1);begin=time.monotonic();PRIVATE.mkdir(parents=True,exist_ok=True)
    path=PRIVATE/'source_mass_logits.npz';assert not path.exists()
    sources={r['index']:r for r in C.iter_rows(DOC/'SOURCE_CEILING_ROWS.jsonl.gz')};features=np.load(CACHE/'features.npy',mmap_mode='r');published=C.read(C.DOC/'TRAINING_COMPLETION.json')
    all_logits=np.empty((3,256,84,66),dtype='<f4');records=[];bindings=[];forwards=0
    for arm_index,arm in enumerate(M.ARMS):
        checkpoint_path=C.SCRATCH/'learned_fits'/(arm+'.pt');binding=C.binding(checkpoint_path);expected=next(r['checkpoint'] for r in published['checkpoints'] if r['arm']==arm)
        assert binding['sha256']==expected['sha256'];checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        assert checkpoint['steps']==3000 and checkpoint['arm']==arm and checkpoint['protocol_sha256']==C.sha(C.DOC/'LEARNING_PROTOCOL.json')
        head=M.CorrespondenceHead().eval();head.requires_grad_(False);head.load_state_dict(checkpoint['model']);bindings.append(binding)
        with torch.no_grad():
            for start in range(768,1024,16):
                x=torch.tensor(np.asarray(features[start:start+16]),dtype=torch.float32);logits=head(x,arm).numpy();forwards+=1;all_logits[arm_index,start-768:start-768+16]=logits
                for local,index in enumerate(range(start,start+16)):
                    source=sources[index];scores=np.asarray(logits[local],dtype='<f4');raw=scores.tobytes();original=scores.argmax(1);binary,pmatch,logodds=decode_mass(scores)
                    targets=source['targets'];valid=np.asarray(targets['valid']);lo=np.asarray(targets['lo']);role=np.asarray(source['predicted_role_query_ids']);rescued=(original==65)&(binary<65)
                    row=dict(id=source['id'],index=index,family=source['family'],partition=source['partition'],arm=arm,
                        source_ceiling_row=dict(file='SOURCE_CEILING_ROWS.jsonl.gz',line=index+1,sha256=C.digest(source)),
                        logits_shape=[84,66],logits_dtype='little-endian float32',logits_f32_zlib_base64=base64.b64encode(zlib.compress(raw,6)).decode(),
                        logits_uncompressed_sha256=hashlib.sha256(raw).hexdigest(),Pmatch=pmatch.tolist(),log_match_mass_minus_none=logodds.tolist(),
                        none_argmax_but_match_mass_gt_half_count=int(rescued.sum()),rescued_positive_boundary_count=int((rescued&valid&(lo<65)&(role==0)).sum()),
                        decoders={DECODERS[0]:analyze(scores,source,original),DECODERS[1]:analyze(scores,source,binary,pmatch)})
                    assert np.array_equal(decode_logits(row),scores);records.append(row)
    np.savez_compressed(path,logits=all_logits,source_indices=np.arange(768,1024),arms=np.asarray(M.ARMS))
    assert forwards==48 and len(records)==768
    C.save_rows(DOC/'SOURCE_MATCH_MASS_LOGITS.jsonl.gz',records)
    result=dict(schema='fixed_match_existence_decision_causal_diagnostic_v1',summary=summary(records),
        decoders={DECODERS[0]:'argmax over65coordinate bins and NONE',DECODERS[1]:'Pmatch=sum65joint-softmax bins > Pnone, exact0.5; argmaxcoordinate bin, integer1px; no averaging'},
        calibration_use='reported separately; no threshold, model, checkpoint or configuration chosen',
        same_outputs_for_both_decoders=True,batch=16,CPU_batch_head_forwards=48,image_head_exposures=768,
        cumulative_new_source_CPU_batch_head_forwards=72,cumulative_new_source_image_head_exposures=1152,
        new_detector_forwards=0,new_optimizer_updates=0,new_PnP_calls=0,new_auxiliary_rays=0,new_RGB=0,device='CPU float32',
        original9000updates_and_original66decoder_unchanged=True,no_model_reselection=True,no_source_pose_metrics=True,
        rows=C.binding(DOC/'SOURCE_MATCH_MASS_LOGITS.jsonl.gz'),private_logits=C.binding(path),source_ceiling=C.binding(DOC/'SOURCE_CEILING_ROWS.jsonl.gz'),
        checkpoint_bindings=bindings,code=C.binding(Path(__file__)),wall_seconds=time.monotonic()-begin)
    C.write(DOC/'SOURCE_MATCH_MASS_ANALYSIS.json',result)
    import json
    print('SOURCE_MATCH_MASS_COMPLETE',json.dumps(result['summary']),flush=True)


if __name__=='__main__':run()
