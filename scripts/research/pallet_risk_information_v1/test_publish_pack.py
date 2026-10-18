"""Small isolated publication safety fixtures; no campaign/ref/score reads."""
from collections import Counter
import csv,gzip,hashlib,json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image,PngImagePlugin
from . import publish_pack as P


class PublicationSafety(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_private_path_keys_retain_identity(self):
        raw={'/home/alice/a/common.py':1,'/home/alice/b/common.py':2}
        value=P.sanitized(raw,Counter());self.assertEqual(sorted(value.values()),[1,2]);self.assertEqual(len(value),2)
        for key in value:self.assertNotIn('/home/alice',key)

    def test_key_collision_never_overwrites(self):
        with patch.object(P,'redact',return_value='same'):
            with self.assertRaisesRegex(ValueError,'collision'):P.sanitized({'a':1,'b':2},Counter())

    def test_all_failure_details_and_deltas_preserved(self):
        raw={'failure_count':500,'failures':[{'id':str(i),'translation_delta':i-250.,'rotation_delta':-i}for i in range(500)]}
        self.assertEqual(P.sanitized(raw,Counter()),raw)

    def test_private_bulk_summarized_not_scalar_failures(self):
        raw={'candidates':[{'R_fixed':[[1,0,0]]}], 'failure_count':1,'reason':'bad'}
        out=P.sanitized(raw,Counter());self.assertEqual(out['failure_count'],1)
        self.assertEqual(out['candidates']['items'],1);P.assert_no_bulk(out)
        with self.assertRaises(ValueError):P.assert_no_bulk(raw)

    def test_selected_pose_exact_no_logits_or_bank(self):
        pose={'available':True,'R_physical':[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],'centroid':[.1,.2,3.]}
        entry={'actual_pose':pose,'selected_candidate_id':'N0','selected_source_aliases':['N0'],
            'model_outputs':[123,456],'actions':[{'private':'bank'}],'selected_score':123,'fallback':False}
        row={'id':'a','partition':'TEST','session':'s','cluster_id':'c','predictions':{'arm':entry}}
        kept=P.selected_ledger([row],['arm'])[0]['predictions']['arm']
        self.assertEqual(kept['actual_pose'],pose);self.assertNotIn('model_outputs',kept);self.assertNotIn('actions',kept);self.assertNotIn('selected_score',kept)
        out=P.sanitized([{'predictions':{'arm':kept}}],Counter(),hide_bulk=False)
        self.assertEqual(out[0]['predictions']['arm']['actual_pose'],pose)

    def test_unavailable_pose_stays_failure(self):
        row=dict(id='x',partition='CAL',session='s',cluster_id='c',predictions={'a':{'actual_pose':{'available':False},'fallback':True}})
        self.assertEqual(P.selected_ledger([row],['a'])[0]['predictions']['a']['actual_pose'],{'available':False})

    def test_prediction_arm_omission_rejected(self):
        with self.assertRaises(ValueError):P.selected_ledger([{'predictions':{}}],['missing'])

    def test_csv_extremes_negative_deltas_all_rows_preserved(self):
        src=self.root/'r.csv.gz'
        with gzip.open(src,'wt')as f:f.write('id,translation_cm,delta\na,999999,-123.456\nb,0,0\n')
        out=list(csv.DictReader(P.io.StringIO(gzip.decompress(P.csv_bytes(src,Counter())).decode())))
        self.assertEqual(out,[{'id':'a','translation_cm':'999999','delta':'-123.456'},{'id':'b','translation_cm':'0','delta':'0'}])

    def test_duplicate_json_key_rejected(self):
        src=self.root/'bad.json';src.write_text('{"a":1,"a":2}')
        with self.assertRaises(ValueError):P.read(src)

    def test_sanitized_root_run_log_jsonl_allowed_all_rows(self):
        path=self.root/'RUN_LOG.jsonl';path.write_text('{"event":"FAILED","exit_code":1}\n{"event":"COMPLETED","wall_seconds":2.5}\n')
        self.assertEqual(P.public_jsonl_check(path,'RUN_LOG.jsonl'),2)

    def test_raw_candidate_jsonl_and_nested_log_rejected(self):
        path=self.root/'RUN_LOG.jsonl';path.write_text('{"event":"x"}\n')
        for name in ('RAW_CANDIDATES.jsonl','nested/RUN_LOG.jsonl'):
            with self.assertRaises(ValueError):P.public_jsonl_check(path,name)
        path.write_text('{"event":"x","candidates":[{"R_physical":[1,2,3]}]}\n')
        with self.assertRaises(ValueError):P.public_jsonl_check(path,'RUN_LOG.jsonl')

    def test_public_run_log_jsonl_duplicate_private_path_nonfinite_rejected(self):
        path=self.root/'RUN_LOG.jsonl'
        for content in ('{"x":1,"x":2}\n','{"source":"/home/alice/private"}\n','{"seconds":NaN}\n'):
            path.write_text(content)
            with self.assertRaises(ValueError):P.public_jsonl_check(path,'RUN_LOG.jsonl')

    def test_derivative_transform_preserves_negative_failure_scalars(self):
        src=self.root/'source.json';src.write_text(json.dumps({'status':'FAIL','failure_count':2,
            'failures':[{'delta':-2.5},{'delta':99999.}],'source':'/home/alice/private/source.py'}))
        raw,counts=P.derivative_bytes(src,'summary');out=json.loads(raw)
        self.assertEqual(out['failures'],[{'delta':-2.5},{'delta':99999.}]);self.assertEqual(out['status'],'FAIL')
        self.assertEqual(counts['private_path_replacements'],1)
        self.assertEqual(raw,P.derivative_bytes(src,'summary')[0])

    def test_original_derivative_numeric_tamper_changes_expected_bytes(self):
        src=self.root/'source.json';src.write_text('{"failure_count":2}')
        original=P.derivative_bytes(src,'summary')[0];src.write_text('{"failure_count":0}')
        self.assertNotEqual(original,P.derivative_bytes(src,'summary')[0])

    def test_late_metadata_names_cannot_hide_nested_or_core_payloads(self):
        from types import SimpleNamespace
        for name in ('nested/receipt.json','VERIFICATION.json','../receipt.json','x.csv'):
            with self.assertRaises(ValueError):P.metadata_exclusions(SimpleNamespace(late_metadata=[name]))
        self.assertIn('FINAL_REMOTE.json',P.metadata_exclusions(SimpleNamespace(late_metadata=['FINAL_REMOTE.json'])))

    def test_token_and_private_path_blocked(self):
        for text in ['/home/alice/private/bank.json','ghp_'+'A'*30,'-----BEGIN PRIVATE KEY-----']:
            with self.assertRaises(ValueError):P.scan_text(text,'fixture')

    def test_embedded_raw_image_data_blocked(self):
        with self.assertRaises(ValueError):P.scan_text('data:image/png;base64,AAAA','fixture')

    def test_relative_path_escape_rejected(self):
        for name in ['../bank.json','/etc/passwd','a/../../b']:
            with self.assertRaises(ValueError):P.relative(name)

    def test_relative_links_query_anchor_and_missing(self):
        src=self.root/'README.md';target=self.root/'report.md';target.write_text('hello')
        P.local_link(src,'report.md?q=1#section',self.root)
        with self.assertRaises(ValueError):P.local_link(src,'gone.md',self.root)
        with self.assertRaises(ValueError):P.local_link(src,'../escape.txt',self.root)

    def test_external_embedded_image_rejected(self):
        with self.assertRaises(ValueError):P.local_link(self.root/'a.md','https://example.com/a.png',self.root,image=True)

    def test_existing_root_report_not_overwritten(self):
        report=self.root/'REPORT_KO.md';report.write_bytes(b'original')
        with self.assertRaises(FileExistsError):P._backup_write(report,b'changed',self.root/'private',replace=True,known=False)
        self.assertEqual(report.read_bytes(),b'original')

    def test_first_audit_self_link_gets_explicit_pending_destination(self):
        public=self.root/'public';public.mkdir();private=self.root/'private'
        self.assertTrue(P.initialize_audit_destination(public,private))
        self.assertEqual(P.read(public/'PUBLIC_PACK_AUDIT.json')['status'],'NOT_YET_VERIFIED')
        P.local_link(public/'REPORT_KO.md','PUBLIC_PACK_AUDIT.json',self.root)
        with self.assertRaises(ValueError):P.local_link(public/'REPORT_KO.md','missing_other.json',self.root)

    def test_existing_failed_audit_is_not_reset_to_pending(self):
        public=self.root/'public';public.mkdir();path=public/'PUBLIC_PACK_AUDIT.json';path.write_text('{"status":"FAIL","failure_count":3}')
        old=path.read_bytes();self.assertFalse(P.initialize_audit_destination(public,self.root/'private'))
        self.assertEqual(path.read_bytes(),old)

    def test_generated_replacement_preserves_exact_prior_bytes(self):
        path=self.root/'derived.json';path.write_bytes(b'original');digest=P.sha(path)
        P._backup_write(path,b'next',self.root/'private',replace=True,known=True)
        self.assertEqual((self.root/'private/publication_prior'/digest/path.name).read_bytes(),b'original')

    def test_valid_png_and_jpeg_fresh_verify_decode(self):
        for extension in('png','jpg'):
            path=self.root/('valid.'+extension);Image.new('RGB',(13,11),(10,20,30)).save(path)
            info=P.image_check(path);self.assertEqual((info['width'],info['height']),(13,11))

    def test_corrupt_png_still_rejected(self):
        path=self.root/'corrupt.png';Image.new('RGB',(13,11)).save(path)
        raw=bytearray(path.read_bytes());raw[-5]^=0xff;path.write_bytes(raw)
        with self.assertRaises(Exception):P.image_check(path)

    def test_exif_rejected_in_public_but_private_original_is_untouched(self):
        path=self.root/'camera.jpg';image=Image.new('RGB',(13,11));exif=Image.Exif();exif[315]='Fixture Author';image.save(path,exif=exif)
        digest=P.sha(path)
        with self.assertRaisesRegex(ValueError,'EXIF'):P.image_check(path)
        self.assertEqual(P.image_check(path,public=False)['width'],13);self.assertEqual(P.sha(path),digest)

    def test_png_text_private_path_rejected(self):
        path=self.root/'metadata.png';meta=PngImagePlugin.PngInfo();meta.add_text('Comment','/home/alice/raw.png')
        Image.new('RGB',(13,11)).save(path,pnginfo=meta)
        with self.assertRaises(ValueError):P.image_check(path)

    def test_raw_rgb_copy_and_reencoding_rejected(self):
        raw=self.root/'original.png';copy=self.root/'renamed.png';image=Image.new('RGB',(13,11),(11,22,33));image.save(raw,compress_level=0);image.save(copy,compress_level=9)
        info=P.image_check(raw,public=False)
        with self.assertRaisesRegex(ValueError,'bytes'):P.public_raster(raw,{P.sha(raw)},set())
        self.assertNotEqual(P.sha(raw),P.sha(copy))
        with self.assertRaisesRegex(ValueError,'pixels'):P.public_raster(copy,{P.sha(raw)},{info['pixel_sha256']})


if __name__=='__main__':unittest.main()
