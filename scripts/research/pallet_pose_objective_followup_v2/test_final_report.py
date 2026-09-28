import inspect
from pathlib import Path
import unittest
from . import final_report as R
from . import final_audit as A


class AggregateReportContracts(unittest.TestCase):
    def source(self,value):return R.Source(Path('example.json'),value,dict(path='example.json',sha256='fixture'))

    def test_pointer_escape_and_display_format(self):
        source=self.source({'a/b':{'c~d':[-1.23456789]}})
        ptr=R.pointer('a/b','c~d',0)
        self.assertEqual(ptr,'/a~1b/c~0d/0')
        self.assertEqual(R.lookup(source.value,ptr),-1.23456789)
        self.assertEqual(R.number(source,ptr).text,'-1.234568')
        self.assertEqual(R.number(source,ptr).refs[0]['json_pointer'],ptr)

    def test_empty_and_failed_quantiles_do_not_become_zero(self):
        source=self.source({'median':None,'median_status':'POSITIVE_INFINITY','P90':None,'P90_status':'NA_EMPTY'})
        self.assertEqual(R.number(source,'/median').text,'+∞')
        self.assertEqual(R.number(source,'/P90').text,'NA')

    def test_every_rendered_number_records_exact_document_line(self):
        source=self.source({'n':99,'T':12.423010486617885});trace=[]
        doc=R.Document('test.md',trace);doc.row('primary',R.number(source,'/n','d'),R.number(source,'/T'))
        self.assertEqual(len(trace),2)
        for row in trace:
            self.assertEqual(row['line_number'],1)
            self.assertEqual(row['line_contains'],doc.lines[0])
            self.assertIn(row['rendered'],doc.lines[0])

    def test_pending_checks_seeds_not_scalar_legacy_seed(self):
        p=self.source(dict(cycle='REPEAT',materials=['PLASTIC'],seeds=[43],seed=42))
        r=self.source(dict(cycle='REPEAT',material='PLASTIC',seed=42))
        missing=R.pending_results([p],[r],self.source({}))
        self.assertEqual(missing,['REPEAT/PLASTIC/S43'])

    def test_exact_reexecution_not_independent_replication(self):
        d=self.source(dict(repeat_status='DIRECTION_REPEATED',repeat_outcome='JOINT_GAIN_ON_REUSED_DEV'))
        result=R.decision([],None,d,[],True,self.source({'effective_data_variation':False}))
        self.assertEqual(result['REPEAT'],'NOT_RUN')
        self.assertEqual(result['EXECUTION'],'PARTIAL_BUDGET')
        self.assertFalse(result['stochastic_training_variation_tested'])
        self.assertFalse(result['deployment_approval'])

    def test_dimension_metadata_allowed_but_actual_pixels_forbidden(self):
        self.assertEqual(A.privacy_paths({'pixels':[1728,671]}),[])
        self.assertEqual(A.privacy_paths({'pixels':[[0,128,255],[0,255,255]]}),['$.pixels'])
        self.assertEqual(A.privacy_paths({'xy':[12,34]}),['$.xy'])

    def test_report_builder_has_no_training_git_or_shared_write(self):
        source=inspect.getsource(R)
        for token in ('C.resource(', 'C.state(', '.cuda(', 'torch.load(', 'subprocess.', 'os.system('):
            self.assertNotIn(token,source)
        self.assertNotIn("'cycles/'+MAIN[0]+'/REPORT_KO.md'",source)


if __name__=='__main__':unittest.main()
