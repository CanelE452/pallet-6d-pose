"""Reference-free saved-bank contract checks, using direct numeric arithmetic.

This is an observation/source contract audit. It is not independent pose or
reference evaluation and does not call the extraction/feature implementation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def read(path):
    path=Path(path)
    with (gzip.open if path.suffix=='.gz' else open)(path,'rt') as stream:
        return json.load(stream)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def audit(run_files, extraction_lock, output):
    start=time.monotonic();lock=read(extraction_lock);counts=Counter();domains={}
    scalar_count=lock['feature_contract']['scalar_count'];sigma=lock['configuration']['detector_sigma_px']
    source_hashes={}
    for run_file in run_files:
        run=read(run_file);domain_counts=Counter();source_hashes[str(run_file)]=sha(run_file)
        assert run['binding']['extraction_lock_sha256']==sha(extraction_lock)
        with gzip.open(run['feature_file'],'rt') as stream:aggregate=[json.loads(s) for s in stream if s.strip()]
        assert len(aggregate)==run['frames']
        assert sha(run['feature_file'])==run['feature_sha256']
        for seal_path,aggregate_feature in zip(run['frame_seals'],aggregate):
            seal_path=Path(seal_path);seal=read(seal_path);folder=seal_path.parent
            assert seal['complete'] and seal['binding']==run['binding']|{'inputs':seal['binding']['inputs']}
            assert seal['reference_reads']==0 and seal['neural_forward_calls']==0
            for name,digest in seal['outputs'].items():assert sha(folder/name)==digest
            bank=read(folder/'BANK.json.gz');feature=read(folder/'FEATURES.json.gz');legacy=read(folder/'LEGACY.json.gz')
            assert feature==aggregate_feature
            assert set(feature)=={'schema','id','split','partition','corner_rows'}
            assert bank['configuration']==lock['configuration']
            assert feature['id']==bank['frame_id']==seal['id']
            assert len(bank['corners'])==8 and bank['states']==legacy['states']
            feature_by_id={c['native_corner']:c for c in feature['corner_rows']}
            assert len(feature_by_id)==len(feature['corner_rows'])
            for k,corner in enumerate(bank['corners']):
                assert corner['native_corner']==k
                if bank['states'][k] not in ('B','I'):
                    assert corner['candidates']==[] and not corner['pixel_queried'] and k not in feature_by_id
                    domain_counts['excluded_corners']+=1;continue
                assert corner['pixel_queried'] and 1<=len(corner['candidates'])<=13 and len(corner['rays'])<=6
                assert corner['candidates'][0]['kind']=='keep'
                rays={r['source_line_id']:r for r in corner['rays']}
                sampling={s['native_neighbor']:s for s in corner['samplings']}
                for mate,trace in sampling.items():
                    paired=[r for r in corner['rays'] if r['native_neighbor']==mate]
                    assert {r['cue'] for r in paired}=={'B','I'} and len(paired)==2
                    assert all(r['query_grid_sha256']==trace['query_grid_sha256'] for r in paired)
                    domain_counts['paired_cue_grids']+=1
                cfeature=feature_by_id[k]
                assert set(cfeature)=={'native_corner','state','candidate_ids','applied_xy','x_values','x_validity','partial_replaces_full_likelihood'}
                assert cfeature['candidate_ids']==[c['candidate_id'] for c in corner['candidates']]
                assert cfeature['applied_xy']==[c['applied_xy'] for c in corner['candidates']]
                assert cfeature['state']==bank['states'][k]
                assert cfeature['partial_replaces_full_likelihood']==(corner['partial_line'] is not None)
                x=np.asarray(cfeature['x_values']);v=np.asarray(cfeature['x_validity'])
                assert x.shape==(len(corner['candidates']),scalar_count) and v.shape==x.shape
                assert np.isfinite(x).all() and np.isin(v,[0,1]).all() and np.all(x[v==0]==0)
                domain_counts['finite_allowlisted_feature_scalars']+=int(x.size)
                assert corner['full_alternative_count']==len(corner['candidates'])-1
                if corner['partial_line'] is not None:
                    assert corner['full_alternative_count']==0 and corner['partial_original_visible'] is not None
                    assert corner['partial_line']['replaces_complete_BI_likelihood']
                    assert corner['partial_line']['explicit_complete_BI_likelihood_revocation']
                    assert corner['partial_line']['association_provenance']['original_raw_xy']==corner['q0']
                    domain_counts['fixed_partial_corners']+=1
                elif corner['full_alternative_count']:
                    domain_counts['full_alternative_corners']+=1
                else:domain_counts['q0_only_corners']+=1
                for candidate in corner['candidates']:
                    covariance=np.asarray(candidate['covariance_px2']);np.linalg.cholesky(covariance)
                    assert np.allclose(covariance,covariance.T,atol=1e-12,rtol=0)
                    q=np.asarray(candidate['applied_xy']);h,w=bank['image_hw']
                    assert 0<=q[0]<w and 0<=q[1]<h
                    if candidate['kind']=='intersection':
                        raw=np.asarray(candidate['raw_intersection_xy']);q0=np.asarray(candidate['q0']);alpha=candidate['blend_alpha']
                        conditional=np.asarray(candidate['conditional_covariance']['covariance_px2'])
                        np.testing.assert_allclose(q,q0+alpha*(raw-q0),atol=1e-12,rtol=0)
                        np.testing.assert_allclose(covariance,(1-alpha)**2*np.eye(2)*sigma**2+alpha**2*conditional+np.eye(2),atol=1e-12,rtol=0)
                        assert 0<alpha<=lock['configuration']['maximum_alpha']
                        assert np.linalg.norm(raw-q0)<=lock['configuration']['maximum_corner_shift_px']
                        assert len(set(candidate['source_neighbors']))==2
                        assert all(rays[source]['supported'] for source in candidate['source_line_ids'])
                        assert candidate['physical_source_ids']==[rays[s]['physical_source_id'] for s in candidate['source_line_ids']]
                        domain_counts['own_covariance_applied_intersections']+=1
                    else:
                        np.testing.assert_array_equal(candidate['applied_xy'],corner['q0'])
                        expected_sigma=lock['configuration']['fallback_boundary_sigma_px' if bank['states'][k]=='B' else 'fallback_interior_sigma_px']
                        np.testing.assert_array_equal(covariance,np.eye(2)*expected_sigma**2)
                    domain_counts['candidates']+=1
                domain_counts['visible_corners']+=1
            domain_counts['frames']+=1
        assert domain_counts['frames']==run['frames']
        domains[run['domain']]=dict(domain_counts);counts.update(domain_counts)
    result=dict(schema='SAVED_COMMON_OBSERVATION_CONTRACT_AUDIT_V1',status='PASS',domains=domains,counts=dict(counts),
        source_files_sha256=source_hashes,extraction_lock_sha256=sha(extraction_lock),audit_source_sha256=sha(__file__),
        reference_reads=0,pose_solves=0,neural_forwards=0,elapsed_seconds=time.monotonic()-start,
        scope='direct saved-array/source/feature contract arithmetic; not independent pose or reference evaluation')
    Path(output).write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(result,sort_keys=True));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='append',required=True);p.add_argument('--extraction-lock',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.run,a.extraction_lock,a.output)
