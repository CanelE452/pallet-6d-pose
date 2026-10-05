"""Synthetic validation fixtures stay in TemporaryDirectory; never evidence."""
import copy
import json
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zlib
from pathlib import Path

from serve import Context, ReviewServer, ValidationError, sha256


def png(width=20, height=10):
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    raw=b''.join(b'\0'+b'\0\0\0'*width for _ in range(height))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.plan=self.root/'plan.json';self.plan.write_text('{"synthetic_test_only":true}')
        self.contract=self.root/'contract.json'
        self.contract.write_text(json.dumps({'schema_version':'lifter_corner_contract_v1','version':'synthetic_test_contract','object_definition':'synthetic fixture, not human evidence','source':'unit test only','direct_click_policy':'human_confirmation_required','corners':[{'id':i,'name':str(i),'definition':'synthetic'} for i in range(8)]}))
        image=self.root/'raw.png';image.write_bytes(png())
        self.manifest=self.root/'manifest.json'
        self.manifest.write_text(json.dumps({'schema_version':'lifter_review_manifest_v1','plan_sha256':sha256(self.plan),'corner_contract_sha256':sha256(self.contract),'frames':[{'frame_id':'test:0','session_id':'test','saved_frame_index':0,'camera_sensor_timestamp_ms':1000,'width':20,'height':10,'image_path':'raw.png','image_sha256':sha256(image),'repeat_review':True}]}))
        self.store=self.root/'store.json'
        self.ctx=Context(self.manifest,self.plan,self.contract,self.store)
        self.reviewer={'id':'TEST_FIXTURE_NOT_HUMAN','entered_by':'human','machine_assistance':False,'previous_prediction_exposure':False,'previous_annotation_exposure':False,'exposure_notes':'SYNTHETIC UNIT TEST ONLY','confirmation':True}

    def sample(self, review_pass='primary'):
        return {'frame_id':'test:0','review_pass':review_pass,'status':'reviewed','reviewer':self.reviewer,'object':{'presence':'present','target_object_id':'synthetic_object','target_identity_confirmed':True},'corners':[{'id':i,'visibility':'direct_visible' if i==0 else 'not_direct_visible','external_occlusion':False,'self_occlusion':i>0,'out_of_frame':False,'definition_uncertain':False,'definition_confirmed':True,'x':4.25 if i==0 else None,'y':3.5 if i==0 else None,'reason':'synthetic test only'} for i in range(8)],'skip_reason':'','edit_reason':''}

    def saved(self,review_pass='primary'):
        start=self.ctx.start_session({'frame_id':'test:0','review_pass':review_pass,'reviewer':self.reviewer})
        return self.ctx.save({'token':start['token'],'record':self.sample(review_pass)})['record']

    def test_save_reload_export_roundtrip(self):
        record=self.saved()
        fresh=Context(self.manifest,self.plan,self.contract,self.store)
        self.assertEqual(fresh.record_for('test:0','primary'),record)
        self.assertEqual(fresh.export()['records'][0]['corners'][0]['x'],4.25)
        self.assertEqual(fresh.counts()['incomplete_repeat'],1)
        self.assertTrue(record['review_time']['finished_at'])

    def test_repeat_does_not_import_first_clicks(self):
        self.saved()
        start=self.ctx.start_session({'frame_id':'test:0','review_pass':'repeat','reviewer':self.reviewer})
        self.assertIsNone(start['record'])
        record=self.ctx.save({'token':start['token'],'record':self.sample('repeat')})['record']
        self.assertEqual(record['repeat_relationship'],'same_person_repeat')
        self.assertFalse(record['independent_repeat'])
        other=copy.deepcopy(self.reviewer);other['id']='OTHER_SYNTHETIC_REVIEWER'
        self.assertIsNone(self.ctx.start_session({'frame_id':'test:0','review_pass':'repeat','reviewer':other})['record'])

    def test_reject_bad_records(self):
        valid=self.saved()
        mutations=[lambda r:r.update(status='draft',source_kind='human_in_progress'),lambda r:r.update(source_kind='machine_proposed'),lambda r:r.update(image_sha256='bad'),lambda r:r.update(plan_sha256='bad'),lambda r:r['corners'][0].update(x=None),lambda r:r['corners'][0].update(x=20),lambda r:r['corners'][0].update(y=-1),lambda r:r['corners'][0].update(x=float('nan')),lambda r:r['corners'][1].update(id=0),lambda r:r['corners'][0].update(definition_confirmed=False),lambda r:r['corners'][0].update(external_occlusion=True),lambda r:r['corners'][1].update(x=2),lambda r:r['reviewer'].update(previous_prediction_exposure=None),lambda r:r['reviewer'].update(entered_by='machine'),lambda r:r['object'].update(presence='absent'),lambda r:r['object'].update(target_identity_confirmed=False)]
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index):
                record=copy.deepcopy(valid);mutate(record)
                with self.assertRaises(ValidationError):self.ctx.validate_record(record)
        bundle=self.ctx.export();bundle['records'].append(copy.deepcopy(valid))
        with self.assertRaises(ValidationError):self.ctx.validate_bundle(bundle)

    def test_draft_not_exported_and_missing_coordinate_final_rejected(self):
        start=self.ctx.start_session({'frame_id':'test:0','review_pass':'primary','reviewer':self.reviewer})
        record=self.sample();record['status']='draft';record['corners'][0]['x']=None;record['corners'][0]['y']=None
        self.ctx.save({'token':start['token'],'record':record})
        self.assertEqual(self.ctx.export()['records'],[])
        record['status']='reviewed'
        with self.assertRaises(ValidationError):self.ctx.save({'token':start['token'],'record':record})

    def test_skip_requires_reason_and_edit_preserves_history(self):
        self.saved()
        start=self.ctx.start_session({'frame_id':'test:0','review_pass':'primary','reviewer':self.reviewer})
        record=self.sample();record.update(status='skipped',corners=[])
        with self.assertRaises(ValidationError):self.ctx.save({'token':start['token'],'record':record})
        record.update(skip_reason='synthetic reason',edit_reason='synthetic edit reason')
        self.ctx.save({'token':start['token'],'record':record})
        previous=json.loads(self.store.with_name('store.history.jsonl').read_text())
        self.assertEqual(previous['record']['corners'][0]['x'],4.25)

    def test_binding_and_changed_input_rejected(self):
        start=self.ctx.start_session({'frame_id':'test:0','review_pass':'primary','reviewer':self.reviewer})
        self.plan.write_text('{}')
        with self.assertRaises(ValidationError):self.ctx.save({'token':start['token'],'record':self.sample()})
        with self.assertRaises(ValidationError):Context(self.manifest,self.plan,self.contract,self.store)

    def test_http_save_export_origin_and_independent_reload(self):
        server=ReviewServer(('127.0.0.1',0),self.ctx)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        self.addCleanup(server.server_close);self.addCleanup(server.shutdown)
        base='http://127.0.0.1:'+str(server.server_address[1])
        def post(path,data,origin=base):
            req=urllib.request.Request(base+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json','Origin':origin})
            with urllib.request.urlopen(req) as res:return json.load(res)
        start=post('/api/start',{'frame_id':'test:0','review_pass':'primary','reviewer':self.reviewer})
        result=post('/api/save',{'token':start['token'],'record':self.sample()})
        self.assertTrue(result['saved'])
        with urllib.request.urlopen(base+'/api/export') as response:
            bundle=json.load(response)
        self.assertEqual(self.ctx.validate_bundle(bundle)['records'][0]['width'],20)
        fresh=Context(self.manifest,self.plan,self.contract,self.root/'imported.json')
        fresh.store['records']=fresh.validate_bundle(bundle)['records'];fresh.atomic_write()
        self.assertEqual(Context(self.manifest,self.plan,self.contract,self.root/'imported.json').counts()['primary_reviewed'],1)
        with self.assertRaises(urllib.error.HTTPError):post('/api/start',{},origin='https://untrusted.example')

    def test_actual_cli_export_validate_import_and_resume(self):
        def cli(command,store,*extra):
            return subprocess.run([sys.executable,str(Path(__file__).with_name('serve.py')),command,'--manifest',str(self.manifest),'--plan',str(self.plan),'--contract',str(self.contract),'--store',str(store),*map(str,extra)],capture_output=True,text=True)
        export=self.root/'reviewed_export.json'
        empty=cli('export',self.store,'--output',export)
        self.assertEqual(empty.returncode,2)
        self.assertFalse(export.exists())
        self.saved()
        self.assertEqual(cli('export',self.store,'--output',export).returncode,0)
        self.assertEqual(cli('validate',self.store,'--input',export).returncode,0)
        imported=self.root/'imported_from_cli.json'
        self.assertEqual(cli('import',imported,'--input',export).returncode,0)
        self.assertEqual(Context(self.manifest,self.plan,self.contract,imported).counts()['primary_reviewed'],1)
        self.assertEqual(cli('import',imported,'--input',export,'--merge').returncode,0)
        self.assertEqual(cli('export',self.store,'--output',export).returncode,2)
        bad=json.loads(export.read_text());bad['records'][0]['corners'][0]['x']=999
        invalid=self.root/'invalid.json';invalid.write_text(json.dumps(bad))
        self.assertEqual(cli('import',self.root/'reject.json','--input',invalid).returncode,2)
        self.assertFalse((self.root/'reject.json').exists())


if __name__=='__main__':
    unittest.main(verbosity=2)
