"""Publication arithmetic/prose check from sealed source readouts; no model calls."""
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime,timezone
import ast,hashlib,json
V=Path(__file__).resolve().parents[1];H=Path(__file__).resolve().parent
class Tables(HTMLParser):
 def __init__(self):super().__init__();self.tables=[];self.table=None;self.row=None;self.cell=None
 def handle_starttag(self,t,a):
  if t=='table':self.table=[]
  if t=='tr':self.row=[]
  if t in ('td','th'):self.cell=''
 def handle_data(self,d):
  if self.cell is not None:self.cell+=d
 def handle_endtag(self,t):
  if t in ('td','th')and self.cell is not None:self.row.append(self.cell);self.cell=None
  if t=='tr'and self.row is not None:self.table.append(self.row);self.row=None
  if t=='table':self.tables.append(self.table);self.table=None

def bind(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text())
def pct(n):return '—'if n is None else f'{100*n:.1f}%'
def main(output_name='FINAL_PUBLICATION_REVIEW.json'):
 root=read(V/'ROOT_RESULTS_KO.json');summary=read(V/'source_result_readout/ARM_DATASET_SUMMARY.json')
 aux=read(V/'source_result_readout/ORIGINAL_GATE_AND_SEPARATE_AUXILIARY_DIAGNOSTICS.json')
 tables=read(V/'source_result_readout/FULL_NATIVE_MATERIAL_ASPECT_TABLES.json')
 arms=list(summary['arms']);armmap={'C43':arms[0],'C44':arms[1]}
 folds={'OLD6144':'VALIDATION','NEW20480':'VALIDATION','PALLOX512':'VAL_HALL','MR1840':'SOURCE_VALIDATION'}
 html=(V/'ROOT_READOUT_KO.html').read_text();parser=Tables();parser.feed(html)
 data_rows=0;main_full_rows=0;conditional_rows=0;cohort_rows=0;owner_rows=0;native_rows=0
 for table in parser.tables:
  for row in table[1:]:
   data_rows+=1
   if row[0]in armmap and row[1].startswith('[['):
    arm=armmap[row[0]];cm=ast.literal_eval(row[1]);conditioning='all_original_rows'if cm[0][2]==9 else'input_valid_diagnostic'
    m=summary['arms'][arm]['MR1840']['SOURCE_VALIDATION'][conditioning]
    assert cm==m['CM_D_E_rows_D_E_U_columns']
    u_cell=f"{m['prediction_U_known']}/{m['known_DE']} ({pct(m['prediction_U_rate_known'])})"if'보류율'in table[0][-1]else str(m['prediction_U_known'])
    assert row[2:]==[pct(m['D_recall']),pct(m['E_precision']),pct(m['E_recall']),u_cell]
    if conditioning=='all_original_rows':main_full_rows+=1
    else:conditional_rows+=1
   elif row[0]in armmap:
    arm=armmap[row[0]];a=aux['arms'][arm]
    cm=a['CF_owner_weak_TRAIN314_diagnostic']['CM_rows_truth_T_F_BG_columns_prediction_T_F_BG'];correct=sum(cm[k][k]for k in range(3))
    assert row[1:]==[str(a['owner_rows_used']),'253/253',f'{correct}/314',str(summary['arms'][arm]['gates']['native35_passed']),'False'];owner_rows+=1
   elif row[0]in folds:
    family=row[0];arm=armmap[row[1]];m=summary['arms'][arm][family][folds[family]]['all_original_rows']
    assert row[2:]==[str(m['known_D']),str(m['known_E']),pct(m['D_recall']),pct(m['E_precision']),pct(m['E_recall']),str(m['prediction_U_known']),str(m['unknown_truth'])];cohort_rows+=1
   else:
    slot=int(row[0]);mm=[tables['arms'][arm]['MR1840'][f'SOURCE_VALIDATION::native{slot}']['all_original_rows']for arm in arms]
    assert row[1:]==[str(mm[0]['known_D']),str(mm[0]['known_E']),f"{mm[0]['CM_D_E_rows_D_E_U_columns'][1][1]}/{mm[0]['known_E']}",f"{mm[1]['CM_D_E_rows_D_E_U_columns'][1][1]}/{mm[1]['known_E']}",str(mm[0]['prediction_U_known'])];native_rows+=1
 assert (main_full_rows,conditional_rows,owner_rows,cohort_rows,native_rows)==(2,2,2,8,8)
 assert data_rows==22
 assert any('D/E 입력 U'in cell for table in parser.tables for cell in table[0])
 for i,arm in enumerate(arms):
  assert root['MR_held_full'][i]==summary['arms'][arm]['MR1840']['SOURCE_VALIDATION']['all_original_rows']
  assert root['MR_train_full'][i]==summary['arms'][arm]['MR1840']['SOURCE_DEVELOPMENT']['all_original_rows']
 assert root['native35_passed']==[23,24]and root['full_source_qualification']==[False,False]
 assert root['fixed_effect_E_true_positives_delta']==1 and root['fixed_effect_D_true_positives_delta']==0
 assert root['original_source_rows_each']==28976 and root['original_source_inputU_each']==535 and root['CF_rows_preserved']==3816 and root['CF_inputU']==360
 assert root['selected_model']is None and not root['adopted']and not root['fully_solved']and root['current223_reads']==root['V112_reads']==root['PnP_q_L_changes']==0
 assert root['all_new_CF_native_visibility_and_quality_UNKNOWN']and root['BEFORE_not_duplicated']and root['owner_rows']==[253,567]
 aa=[aux['arms'][arm]for arm in arms]
 assert aa[0]['synthetic_point_quality_diagnostic']==aa[1]['synthetic_point_quality_diagnostic']
 for key in aa[0]['MR_entity_and_secondary_gate_diagnostics']:
  assert aa[0]['MR_entity_and_secondary_gate_diagnostics'][key]['entity']==aa[1]['MR_entity_and_secondary_gate_diagnostics'][key]['entity']
 for text in ('+1/52','+0/102','9개','UNKNOWN','native2/3','WOOD/SQUARE','한 seed','약라벨','전체 목표는 미해결'):
  assert text in html,text
 assert 'C43'in html and ('미사용'in html or '사용하지'in html or '학습하지'in html)
 census=read(V/'source_result_readout/MR_HELD_KNOWN_DE_CHANGE_CENSUS.json')
 assert census['changed_predictions']==1 and census['unchanged_known_predictions']==153 and census['known_E_true_positive_counts']==[2,3]
 receipt=dict(schema='V162_FOCUSED_FINAL_SOURCE_RESULT_PUBLICATION_REVIEW',completed=True,PASS=True,UTC=datetime.now(timezone.utc).isoformat(),findings=[],
  publication=[bind(V/'ROOT_RESULTS_KO.json'),bind(V/'ROOT_READOUT_KO.html')],review_code=bind(Path(__file__).resolve()),
  checked_visible_data_rows=data_rows,visible_row_types=dict(MR_full=main_full_rows,MR_input_valid=conditional_rows,owner=owner_rows,four_family_validation=cohort_rows,MR_native8=native_rows),
  verified=dict(all_visible_numeric_values_match_sealed_metrics=True,full_MR154_D102_E52_including9U_and_conditional145_separate=True,MR_native_U_label_is_known_DE_U=True,
   E_gain_one_of52_no_solution_or_general_success_claim=True,CF314_is_weak_TRAIN_diagnostic_C43_not_CF_trained_C44_CF_trained=True,
   native35_23_24_and_both_full_gate_false=True,entity_quality_unchanged=True,original28976_U535_and_CF3816_U360_preserved=True,
   all_native_material_aspect_cross_tables_bound_full_WOOD_SQUARE_native8_scope_preserved=True,no_new_current_eval_selection_threshold_or_native_quality_labels=True,
   exhaustive_known_MR_change_census_one_improvement_zero_regressions=True),
  evidence=[bind(V/'source_result_readout/ARM_DATASET_SUMMARY.json'),bind(V/'source_result_readout/FULL_NATIVE_MATERIAL_ASPECT_TABLES.json'),bind(V/'source_result_readout/ORIGINAL_GATE_AND_SEPARATE_AUXILIARY_DIAGNOSTICS.json'),bind(V/'source_result_readout/MR_HELD_KNOWN_DE_CHANGE_CENSUS.json')],
  source_feature_weight_model_encoder_forward_training_private_current_V112_reads_or_calls=0,
  visual_scope='HTML source/JSON numeric and prose review only. No browser render, plot or pending gallery visual verification asserted.',selected_model=None,fully_solved=False)
 out=H/output_name
 with out.open('x')as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(dict(PASS=True,rows=data_rows,review=bind(out))))
if __name__=='__main__':main()
