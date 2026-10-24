#!/usr/bin/env python3
"""Final saved report/table/link check only. No pixels or numeric join rerun."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import collections
import json
import os
import posixpath
import re
import stat
from check_saved_join import rb,bind,checked,put

H=Path(__file__).absolute().parent
V=H.parent
PINS={'ROOT_REPORT_KO.md':'b6ba98b1140e863acc38faf39bea56d289f83d63e9fc5267b01051244f176619',
      'PORTABLE_REPORT_KO.md':'b6ba98b1140e863acc38faf39bea56d289f83d63e9fc5267b01051244f176619',
      'ROOT_READOUT_KO.html':'8e756d7e44fc4925682e7bd4ab06e135a433fe1ab18c1b6d37bc4e10fccfe2bd',
      'ROOT_RESULTS_KO.json':'5c61df552bc305977632941b2c8bd2758e8f07770ff4703d82b88887ade8b401'}


class DOM(HTMLParser):
    def __init__(self):
        super().__init__();self.articles=[];self.article=None;self.links=[];self.table=[];self.row=None;self.cell=None;self.option=None;self.option_data=[];self.options={}
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='option':self.option=a.get('value');self.option_data=[]
        for key in ('href','src'):
            if key in a:self.links.append(a[key])
        if tag=='article':self.article={'attrs':a,'text':[],'images':[]};self.articles.append(self.article)
        if tag=='img' and self.article is not None:self.article['images'].append(a['src'])
        if tag=='tr':self.row=[];self.table.append(self.row)
        if tag in ('td','th'):self.cell=[]
    def handle_data(self,data):
        if self.option is not None:self.option_data.append(data)
        if self.article is not None:self.article['text'].append(data)
        if self.cell is not None:self.cell.append(data)
    def handle_endtag(self,tag):
        if tag=='option' and self.option is not None:self.options[self.option]=''.join(self.option_data);self.option=None
        if tag=='article':self.article=None
        if tag in ('td','th') and self.cell is not None:self.row.append(''.join(self.cell).strip());self.cell=None
        if tag=='tr':self.row=None


def local_link(p,value):
    u=urlsplit(value)
    if u.scheme in ('https','http'):return None
    assert not u.scheme and not u.netloc and not value.startswith('/'),value
    if not u.path:return None
    target=Path(posixpath.normpath(str(p.parent/unquote(u.path))))
    assert target.is_absolute() and V in target.parents and 'v112' not in str(target).casefold(),value
    assert target.suffix.lower() not in {'.npz','.npy','.pt','.pth','.bin','.safetensors','.pkl','.pickle'}
    # Only link existence/type metadata; PNG bytes/headers are never opened.
    cur=Path(target.anchor)
    for part in target.parts[1:]:
        cur=cur/part;mode=os.lstat(cur).st_mode;assert not stat.S_ISLNK(mode)
    assert stat.S_ISREG(mode)
    return str(target)


def run():
    raw={name:checked({'path':str(V/name),'sha256':h}) for name,h in PINS.items()}
    assert raw['ROOT_REPORT_KO.md']==raw['PORTABLE_REPORT_KO.md']
    result=json.loads(raw['ROOT_RESULTS_KO.json']);s=result['summary']
    numerical=json.loads(rb(H/'actual_join_v1/REVIEW.json'));assert numerical['completed'] and numerical['PASS']
    n=numerical['summary'];wr=numerical['weak_support_by_role']
    weakdata=json.loads(rb(V/'independent_actual_data_review_b/REVIEW.json'))['summaries']
    expected={'source_frames':12,'declared_GT_candidates':15,'original_native_queries':96,'original_inputU':11,
        'original_roles':{'SOURCE_FIT':64,'SOURCE_DEV':32},'weak_triplets':weakdata['weak']['triplets'],
        'weak_variant_RGB':weakdata['weak']['variants'],'weak_triplet_roles':weakdata['weak']['roles'],
        'separate_GT_derived_training_RGB96':weakdata['GT_family']['training_RGB96_PNGs'],
        'generated_roles':numerical['role_counts'],'generated_query_available':weakdata['GT_family']['generated_query_available'],
        'nominal_target_in_frame_variants':weakdata['GT_family']['nominal_target_in_frame'],
        'full_strict_valid_crops':weakdata['GT_family']['all_strict_crop_pixels_valid_rows'],
        'nominal_part_observations':120,'observer_agreement':n['agreements'],'observer_disagreement':n['disagreements'],
        'in_frame_agreement':n['in_frame_agreements'],'in_frame_denominator':n['in_frame_nominal_rows'],
        'nominal_consensus':n['consensus_state_counts'],'nominal_weak_D':n['consensus_state_counts']['D_VISIBLE_EXPECTED_PART_WEAK'],
        'nominal_weak_E':n['consensus_state_counts']['E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK'],
        'nominal_weak_D_E':n['nominal_weak_eligible'],'support_checked_weak_variants':n['support_checked_weak_candidates'],
        'weak_variant_D':n['generated_state_counts']['D_VISIBLE_EXPECTED_PART_WEAK'],
        'weak_variant_E':n['generated_state_counts']['E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK'],
        'FIT_weak_candidates':n['FIT_weak_loss_candidates'],'FIT_D':wr['SOURCE_FIT']['D_VISIBLE_EXPECTED_PART_WEAK'],
        'FIT_E':wr['SOURCE_FIT']['E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK'],'DEV_weak_candidates':n['DEV_weak_evaluation_candidates'],
        'DEV_D':wr['SOURCE_DEV']['D_VISIBLE_EXPECTED_PART_WEAK'],'DEV_E':wr['SOURCE_DEV']['E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK'],
        'physical_native_gold':0,'actual_fit_updates':0,'new_model_forwards':0,'new_protected_evaluations':0,'PnP_runs':0}
    assert all(s[k]==v for k,v in expected.items())
    assert result['training_admission'] is False and result['model_improvement_established'] is False
    assert result['selected_model'] is None and result['adopted'] is False
    assert result['latest_real_benchmark_phase_unchanged']=='V213' and result['fully_solved'] is False
    assert result['GT_derived_training_query_itself'] is True
    assert result['no_additional_GT_pose_ref_ROI_mask_deployment_feature'] is True
    assert result['all_SOURCE_DEV_descendants_excluded_fit_and_normalizers'] is True
    assert result['original_initializer_GT_instance_association']=='UNKNOWN'
    md=raw['PORTABLE_REPORT_KO.md'].decode();html=raw['ROOT_READOUT_KO.html'].decode()
    dom=DOM();dom.feed(html)
    summary_table=dict(dom.table)
    scalar_summary={k:v for k,v in s.items() if not isinstance(v,(dict,list))}
    assert set(summary_table)==set(scalar_summary) and all(summary_table[k]==str(v) for k,v in scalar_summary.items())
    complete=json.loads(rb(V/'joined_weak_intended_part_targets_v1/COMPLETE.json'))
    joined=[json.loads(x) for x in checked(complete['outputs']['ALL120_JOINED_NOMINAL_OBSERVATIONS.jsonl']).decode().splitlines()]
    assert len(dom.articles)==len(joined)==120
    for article,r in zip(dom.articles,joined):
        assert article['attrs']['data-state']==r['consensus_state'] and article['attrs']['data-role']==r['source_role']
        assert article['images']==['public_source_gt_query_jitter_v1/'+r['nominal_row_id']+'_RGB96.png']
        text=' '.join(''.join(article['text']).split())
        assert dom.options[r['A']['state']] in text and dom.options[r['C']['state']] in text
    # Compare the two principal Markdown numeric tables without browser rendering.
    tables=[];current=[]
    for line in md.splitlines():
        if line.startswith('|'):current.append([x.strip() for x in line.strip().strip('|').split('|')])
        elif current:tables.append(current);current=[]
    if current:tables.append(current)
    assert tables[0][2:]==[['보존한 원 query','96','64','32'],['15후보의 nominal 부위 질문','120','80','40'],
        ['별도 GT 유래 RGB96 입력','360','240','120'],['support 확인된 weak D/E 변형','57','45','12'],['위57 중 D / E','48 / 9','39 / 6','9 / 3']]
    diag=n['A_rows_C_columns_confusion'];order=['D_VISIBLE_EXPECTED_PART_WEAK','E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK','SELF_OR_OWN_STRUCTURE_UNOBSERVABLE','WRONG_PART_OR_GEOMETRY_UNRESOLVED','UNKNOWN','OUT_OF_FRAME']
    assert [[int(x) for x in row[1:]] for row in tables[1][2:]]==[[n['A_state_counts'].get(key,0),n['C_state_counts'].get(key,0),diag[i][i]] for i,key in enumerate(order)]
    role_nominal={role:collections.Counter(r['consensus_state'] for r in joined if r['source_role']==role and r['nominal_weak_eligible']) for role in ('SOURCE_FIT','SOURCE_DEV')}
    assert role_nominal['SOURCE_FIT']=={'D_VISIBLE_EXPECTED_PART_WEAK':15,'E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK':5}
    assert role_nominal['SOURCE_DEV']=={'D_VISIBLE_EXPECTED_PART_WEAK':4,'E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK':3}
    assert {r['native_slot'] for r in joined if r['nominal_weak_eligible']}.isdisjoint({2,6,7})
    assert {r['native_slot'] for r in joined if r['consensus_state']=='E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK'}=={4,5}
    assert len({r['frame_position'] for r in joined if r['source_role']=='SOURCE_FIT' and r['consensus_state']=='E_FOREIGN_COVER_OF_EXPECTED_PART_WEAK'})==5
    mdlinks=re.findall(r'\]\(([^)]+)\)',md)
    expected_condition='deployable_query_condition_b/actual_conditions_v1/ALL96_DEPLOYABLE_CONDITION32.jsonl'
    assert expected_condition in mdlinks
    links={}
    for name,values in [('ROOT_REPORT_KO.md',mdlinks),('PORTABLE_REPORT_KO.md',mdlinks),('ROOT_READOUT_KO.html',dom.links)]:
        local=[x for x in (local_link(V/name,value) for value in values) if x is not None]
        links[name]={'local_targets':len(local),'distinct_local_targets':len(set(local))}
    for phrase in ['GT에서 유래','UNKNOWN','SOURCE_DEV','V213','U0','fresh blind','zero broad path enumeration','렌더 확인은 하지 못했습니다','약한 부위']:
        assert phrase in md,phrase
    assert 'ROOT_RESUMPTION_METADATA_DISCOVERY_DEVIATION.json' in mdlinks and 'ROOT_BROWSER_RENDER_LIMITATION.json' in mdlinks
    deviation=json.loads(checked(result['filename_metadata_enumeration_deviation']))
    limitation=json.loads(rb(V/'ROOT_BROWSER_RENDER_LIMITATION.json'))
    qa={'completed':True,'PASS':True,'code':bind(Path(__file__)),'four_reports':{k:{'path':str(V/k),'sha256':v} for k,v in PINS.items()},
        'independent_actual_join_QA':bind(H/'actual_join_v1/REVIEW.json'),'root_summary_count_fields_compared':len(expected),
        'HTML_summary_rows':len(dom.table),'HTML_nominal_articles':len(dom.articles),'Markdown_numeric_tables_checked':2,
        'local_links':links,'original96_condition_link_exact':expected_condition,
        'nominal_role_D_E':{r:dict(c) for r,c in role_nominal.items()},'metadata_discovery_deviation_disclosed':True,
        'browser_render_unverified_disclosed':True,'PNG_body_reads':0,'new_join_recomputation':False,
        'historical_V213_metrics':'Only existing quoted report rows/algebra read; no original protected scores/ledgers reopened or fresh benchmark claims.',
        'semantic_limits_checked':['Assistant weakpart consensus, not human/native physical gold/blind','GT-derived q itself disclosed, original96 association UNKNOWN','DEV all descendant fit/normalizer exclusion','no fit/inference/model adoption/improvement','OOF/self/wrongpart/U separated from external E'],
        'actual_browser_or_tool_image_views_by_B':0}
    put(H/'FINAL_REPORT_TEXT_AND_LINK_QA.json',qa)
    print(json.dumps({'QA':bind(H/'FINAL_REPORT_TEXT_AND_LINK_QA.json'),'links':links,'summary_fields':len(expected)}))


if __name__=='__main__':run()
