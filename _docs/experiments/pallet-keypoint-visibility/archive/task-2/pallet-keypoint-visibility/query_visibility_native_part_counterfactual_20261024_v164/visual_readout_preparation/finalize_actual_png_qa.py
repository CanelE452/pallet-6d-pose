#!/usr/bin/env python3
"""Finalize recorded direct PNG views; metadata/PNG bytes only, no NPZ loads."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

V = Path(__file__).resolve().parent.parent
O = V / 'visual_readout_v1'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def binding(p):
    return {'path': str(p), 'sha256': sha(p)}

def read(p):
    return json.loads(Path(p).read_text())

def lines(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x]

def check_binding(b):
    p = Path(b['path']).resolve()
    assert p.is_relative_to(O.resolve()), p
    assert sha(p) == b['sha256'], p

class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.refs=[]; self.ids=[]; self.images=[]
    def handle_starttag(self, tag, attrs):
        d=dict(attrs)
        if 'id' in d: self.ids.append(d['id'])
        for k in ('href','src'):
            if k in d: self.refs.append(d[k])
        if tag == 'img': self.images.append(d.get('src'))

def main():
    receipt_path=O/'ACTUAL_PNG_VIEW_QA_COMPLETE.json'
    assert not receipt_path.exists(), 'single finalization only'
    rendered=read(O/'COMPLETE.json')
    protocol=read(V/'visual_readout_preparation/PROTOCOL_BEFORE_SAVED_PIXEL_VISUAL_READOUT.json')
    fixed_binding=protocol['fixed_inputs']
    assert sha(fixed_binding['path']) == fixed_binding['sha256']
    fixed=read(fixed_binding['path'])
    rows=lines(O/'ALL66_VISUAL_MANIFEST.jsonl')
    notes=lines(O/'ACTUALLY_VIEWED51_CARD_NOTES.jsonl')
    assert rendered['PASS'] and rendered['completed']
    assert sha(rendered['manifest']['path']) == rendered['manifest']['sha256']
    assert [r['pair_id'] for r in rows] == [a['pair_id'] for a in fixed['attempts']]
    assert len(rows)==66 and len({r['case_id'] for r in rows})==22
    assert set(Counter(r['case_id'] for r in rows).values())=={3}
    frames={f['frame_index']:f for f in fixed['frames']}
    by_note={n['pair_id']:n for n in notes}
    assert len(notes)==len(by_note)==51
    counts=Counter(); variants=Counter(); assets=0
    for r,a in zip(rows,fixed['attempts']):
        region=a['region']; frame=frames[a['frame_index']]; slot=r['native_slot']
        for k in ('case_id','query_id','query_xy','L','source_id','split','selected_gt_instance_index','native_slot','native_name','region_status'):
            assert r[k] == region[k], (r['pair_id'],k)
        assert r['donor_id']==a['donor_id'] and r['frame_index']==a['frame_index']
        assert r['query_xy']==frame['query_xy'][slot] and r['L']==frame['L'][slot]
        assert r['query_id']==frame['query_id'][slot] and r['split']=='SOURCE_DEVELOPMENT'
        assert r['new_native_quality_owner_targets']==-1 and not r['training_admitted'] and not r['native_or_coordinate_gold']
        assert r['display_overlays_separate_never_model_inputs']
        present=[k for k,v in r['variants'].items() if v['constructed']]
        if r['region_status']=='UNKNOWN':
            counts['UNKNOWN_no_payload']+=1
            assert present==[] and not r['assets'] and r['card'] is None and not r['generator_PASS']
            assert r['coarse_weak_original_fullRGB_bbox_xyxy'] is None
        elif r['generator_PASS']:
            counts['PASS_three_variants']+=1
            assert present==['BEFORE','COVER','ADJACENT_SHAM']
        else:
            counts['placement_failure_BEFORE_only']+=1
            assert present==['BEFORE']
        assert set(r['assets'])==set(present)
        for variant in present:
            variants[variant]+=1
            v=r['variants'][variant]
            assert v['all8_native_quality_owner_targets_unknown']
            for key,fkey in [('all8_query_available','native_continuous_valid'),('all8_crop_constructed','feature_constructed'),('all8_nearest_valid','nearest_valid'),('all8_center4_available','local_central4_available'),('all8_context_available','context_available')]:
                assert v[key]==frame[fkey], (r['pair_id'],variant,key)
            assert set(r['assets'][variant])=={'full_RGB','selected_raw96','selected_central32','target224_DISPLAY','full_RGB_DISPLAY_OVERLAY'}
            for b in r['assets'][variant].values(): check_binding(b); assets+=1
        if r['card']:
            check_binding(r['card']); n=by_note.pop(r['pair_id'])
            assert n['actually_viewed'] and n['tool']=='tools.view_image' and n['layout_PASS']
            assert n['image']==r['card'] and len(n['note'])>25
    assert not by_note
    assert dict(counts)=={'PASS_three_variants':46,'UNKNOWN_no_payload':15,'placement_failure_BEFORE_only':5}
    assert dict(variants)=={'BEFORE':51,'COVER':46,'ADJACENT_SHAM':46} and assets==715
    check_binding(rendered['census']); check_binding(rendered['gallery'])
    parser=Links(); parser.feed((O/'index.html').read_text())
    assert set(parser.ids)=={r['pair_id'] for r in rows} and len(parser.ids)==66
    local_link_checks=0
    provenance_links=0
    allowed_npz={v['saved_NPZ']['path'] for r in rows for v in r['variants'].values() if v['constructed']}
    for ref in parser.refs:
        if ref.startswith('#'): assert ref[1:] in parser.ids
        else:
            u=urlparse(ref)
            if u.scheme=='file':
                assert not u.netloc and unquote(u.path) in allowed_npz, ref
                p=Path(unquote(u.path)); assert p.is_file(), p
                provenance_links+=1
            else:
                assert not u.scheme and not u.netloc, ref
                p=(O/unquote(u.path)).resolve(); assert p.is_relative_to(O.resolve()) and p.is_file(), p
            local_link_checks+=1
    assert len(parser.images)==52
    result={'completed':True,'PASS':True,'scope':'Rendered PNG layout/direct actual views and all66 saved-evidence completeness, not generator66 PASS or scientific qualification',
        'actual_view_QA_completed':True,'direct_card_views':51,'direct_census_views':1,
        'census_actually_viewed':True,'census_tool':'tools.view_image',
        'attempt_count':66,'fixed_cases':22,'status_counts':dict(counts),'variant_counts':dict(variants),
        'potential_variant_slots':198,'real_variant_payloads':143,'missing_variant_slots':55,'rendered_variant_assets_hashed':assets,
        'all51_card_hashes_verified':True,'all66_fixed_order_and_geometry_input_identity_PASS':True,
        'all_original8_input_flags_preserved_from_fixed_metadata':True,'no_NEW_source_or_donor_pixels_NPZ_features_models_loaded_by_finalizer':True,
        'all_new_targets_minus1':True,'training_admitted':False,'native_or_coordinate_gold':False,
        'render_complete':binding(O/'COMPLETE.json'),'generator_complete':rendered['generator_complete'],'protocol':rendered['protocol'],
        'fixed_inputs':fixed_binding,'code':binding(Path(__file__).resolve()),'notes':binding(O/'ACTUALLY_VIEWED51_CARD_NOTES.jsonl'),
        'manifest':rendered['manifest'],'census':rendered['census'],'gallery':rendered['gallery'],
        'HTML_source_and_links':{'PASS':True,'article_ids':66,'image_links':52,'local_file_references_checked':local_link_checks,'bound_generated_NPZ_provenance_file_links_existence_only':provenance_links,'browser_render_verified':False,'reason':'Parent browser file-URL security block; no workaround attempted. Source/link checks and direct PNG QA are separate.'},
        'finalizer_adapter_repair':{'initial_attempt':'exit1 after all66/715PNG checks at strict relative-only HTML link parser; existing file:// generated-NPZ provenance link rejected','preserved_original_code':binding(V/'visual_readout_preparation/FINALIZER_BEFORE_FILE_LINK_ADAPTER_REPAIR.py'),'minimal_repair':'Only exact143 saved-NPZ manifest paths accepted for file:// provenance existence checks; no NPZ bytes loaded; no renderer/output/scientific gate changes.'},
        'layout_findings':{'blocking_issues':[],'nonblocking':['Bottom local central32/target224 captions are tightly spaced but remained legible in direct views; no rerender required.'],'raw_vs_display':'Clean raw panels and separate labeled overlays are distinguishable on all51 cards. Float32 target224 rounding is explicitly display-only.','failed_UNKNOWN':'Five BEFORE-only cards explicitly mark missing interventions;15 UNKNOWN attempts have census/HTML metadata only.'},
        'full_task_limit':'Coarse weak Euro WOOD deck pilot; no exact native corner/part mask/physical visibility/coordinate gold, full native8 or real SQUARE qualification.',
        'finished_utc':datetime.now(timezone.utc).isoformat()}
    receipt_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'completed':True,'PASS':True,'receipt':binding(receipt_path),'statuses':dict(counts),'direct_views':52}))

if __name__=='__main__': main()
