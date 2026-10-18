"""Constructed safe-pack contracts; no campaign data, training or network."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import csv
from PIL import Image, PngImagePlugin

from . import public_pack as P


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True);path.write_text(json.dumps(value, indent=2)+'\n')


def jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(''.join(json.dumps(r)+'\n'for r in records).encode(),mtime=0))


class PackTests(unittest.TestCase):
    def test_GO_guard_precedes_any_io(self):
        with self.assertRaises(PermissionError):P.context(SimpleNamespace(publication_GO=False))

    def test_decoded_linux_private_path(self):
        value='/'+'home'+'/person/data'
        for encoded in (value,value.replace('/','%2F'),value.replace('/','\\u002f'),value.replace('/','&#47;')):
            with self.assertRaises(ValueError):P.scan_text(encoded,'toy')

    def test_decoded_windows_drive_private_path(self):
        value='Q'+':'+chr(92)+'person'+chr(92)+'data'
        with self.assertRaises(ValueError):P.scan_text(value,'toy')
        with self.assertRaises(ValueError):P.scan_text(value.replace(chr(92),'%5C'),'toy')

    def test_tokens_are_rejected(self):
        for value in ('gh'+'p_'+'a'*36,'api'+'_key='+'a'*36):
            with self.assertRaises(ValueError):P.scan_text(value,'toy')

    def test_safe_paths_and_statuses_are_not_secret(self):
        P.scan_text('source_sha256='+'a'*64+'; ${PRIVATE_PATH}/receipt.json; NOT_RUN; REMOTE_NOT_CHECKED','toy')

    def test_direct_and_nested_reference_bank_logits_rejected(self):
        for key in ('actual_pose','references','actions','risk_scores','candidate_poses'):
            with self.assertRaises(ValueError):P.no_bulk({'nested':{key:[1,2]}})
        P.no_bulk({'actions':{'publication':'PRIVATE_BULK_NOT_COPIED','items':10,'canonical_payload_sha256':'a'*64}})

    def test_string_encoded_bank_payload_is_not_a_csv_escape(self):
        with self.assertRaises(ValueError):P.no_bulk(json.dumps({'actual_pose':{'centroid':[1,2,3]}}))

    def test_reference_input_binding_is_not_a_reference_pose_dump(self):
        P.no_bulk({'input_bindings':{'references':{'path':'${PRIVATE_PATH}/FIXED_REFERENCES.json','sha256':'a'*64}}})
        for item in ({'path':'safe','sha256':'bad'}, {'path':'safe','sha256':'a'*64,'poses':[1]}, {'scene1':{'centroid':[1,2,3]}}):
            with self.assertRaises(ValueError):P.no_bulk({'references':item})
        with self.assertRaises(ValueError):P.no_bulk({'actual_pose':{'path':'safe','sha256':'a'*64}})

    def test_symlink_is_not_silently_skipped(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);(root/'a').write_text('safe');(root/'link').symlink_to(root/'a')
            with self.assertRaises(ValueError):P.tree_inventory(root)

    def test_image_byte_and_pixel_identity_are_both_rejected(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);a=root/'a.png';b=root/'b.png';Image.new('RGB',(4,4),(12,13,14)).save(a)
            info=P.OLD.image_check(a,public=False);Image.open(a).save(b,compress_level=0)
            with self.assertRaises(ValueError):P.OLD.public_raster(a,{P.sha(a)},set())
            with self.assertRaises(ValueError):P.OLD.public_raster(b,set(),{info['pixel_sha256']})

    def test_exif_is_not_allowed_in_derived_image(self):
        with tempfile.TemporaryDirectory()as d:
            path=Path(d)/'image.jpg';im=Image.new('RGB',(4,4));exif=im.getexif();exif[0x010E]='private metadata';im.save(path,exif=exif)
            with self.assertRaises(ValueError):P.OLD.image_check(path)

    def test_png_metadata_also_scanned_for_windows_paths(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);path=root/'image.png';meta=PngImagePlugin.PngInfo();meta.add_text('source','Q'+':'+chr(92)+'person'+chr(92)+'data');Image.new('RGB',(4,4)).save(path,pnginfo=meta)
            with self.assertRaises(ValueError):P.scan_asset(path,root,root,set(),set())

    def test_forbidden_extension_and_compressed_expansion_limit(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);path=root/'weights.pt';path.write_bytes(b'x')
            with self.assertRaises(ValueError):P.scan_asset(path,root,root,set(),set())
            path=root/'too_big.csv.gz';path.write_bytes(gzip.compress(b'a'*(P.CAP+1)))
            with self.assertRaises(ValueError):P.raw_bytes(path)

    def test_relative_links_must_exist_without_pending_waiver(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);path=root/'doc.md';path.write_text('[later](PUBLICATION_RECEIPT.json)')
            with self.assertRaises(ValueError):P.check_links(path,root)
            (root/'PUBLICATION_RECEIPT.json').write_text('{}');P.check_links(path,root)

    def test_decoded_and_reference_style_markdown_links(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);path=root/'doc.md';(root/'a b.md').write_text('safe');path.write_text('[x](a%20b.md)\n[y]: a%20b.md\n')
            P.check_links(path,root);path.write_text('[y]: missing.md')
            with self.assertRaises(ValueError):P.check_links(path,root)

    def test_declared_transform_requires_actual_original_bytes(self):
        with tempfile.TemporaryDirectory()as d:
            run=Path(d)/'run';public=Path(d)/'public';run.mkdir();public.mkdir();(run/'a.md').write_text('original');(public/'a.md').write_text('altered')
            mapping=[dict(original_path=str(run/'a.md'),original_sha256=P.sha(run/'a.md'),public_path='a.md',public_sha256=P.sha(public/'a.md'),transform='byte_exact')]
            write(run/'MAP.json',mapping)
            with self.assertRaises(ValueError):P.derivative_map(run/'MAP.json',run,public)

    def test_only_exact_mapped_named_scalar_CSV_streams_past_general_decoded_limit(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);private=root/'private';public=root/'public';private.mkdir();public.mkdir()
            path=public/'B_QUERY_FACTORIAL.csv.gz';original=private/path.name
            payload=('scalar\n'+('a'*1024+'\n')*17000).encode();self.assertGreater(len(payload),P.CAP)
            original.write_bytes(gzip.compress(payload));shutil.copyfile(original,path)
            mapping=[dict(original_path=str(original),original_sha256=P.sha(original),public_path=path.name,public_sha256=P.sha(path),transform='byte_exact')];write(private/'MAP.json',mapping)
            P.derivative_map(private/'MAP.json',private,public);sources=P.stream_sources(private/'MAP.json',private,public)
            result=P.scan_asset(path,public,root,set(),set(),streamed_sources=sources)
            self.assertEqual(result['rows'],17000);self.assertEqual(result['decoded_bytes'],len(payload));self.assertTrue(result['original_header_identical'])
            with self.assertRaises(ValueError):P.scan_asset(path,public,root,set(),set())
            arbitrary=public/'arbitrary.csv.gz';shutil.copyfile(path,arbitrary)
            with self.assertRaises(ValueError):P.raw_bytes(arbitrary)

    def test_streamed_CSV_caps_cell_decode_and_source_header(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);original=root/'original.csv.gz';public=root/'public.csv.gz'
            original.write_bytes(gzip.compress(b'x\n1\n'));public.write_bytes(gzip.compress(b'y\n1\n'))
            with self.assertRaises(ValueError):P.stream_csv(public,original)
            oversized=b'x\n'+b'a'*(P.CELL_CAP+1)+b'\n';original.write_bytes(gzip.compress(oversized));shutil.copyfile(original,public)
            with self.assertRaises((ValueError,csv.Error)):P.stream_csv(public,original)
            repeated=b'x\n'+(b'a'*1024+b'\n')*100;original.write_bytes(gzip.compress(repeated));shutil.copyfile(original,public)
            with patch.object(P,'STREAM_CAP',64*1024):
                with self.assertRaises(ValueError):P.stream_csv(public,original)

    def test_streamed_CSV_rejects_missing_cells_and_nested_bank_strings(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);original=root/'original.csv.gz';public=root/'public.csv.gz'
            for payload in (b'a,b\n1\n',b'payload\n"{""actual_pose"": {""centroid"": [1,2,3]}}"\n'):
                original.write_bytes(gzip.compress(payload));shutil.copyfile(original,public)
                with self.assertRaises(ValueError):P.stream_csv(public,original)

    def fixture(self, directory):
        root=Path(directory);run=root/'private';repo=root/'repo';public=repo/'_docs/experiments'/P.NAMESPACE;evaluation=run/'evaluation_v1'
        public.mkdir(parents=True);evaluation.mkdir(parents=True);source=repo/'scripts/research'/P.NAMESPACE;source.mkdir(parents=True);(source/'safe.py').write_text('VALUE = 1\n')
        registered=list(P.arms(False));records=[];selected=[];inventory=[]
        for i in range(238):
            partition='TEST'if i<49 else'REMAINDER'if i<192 else'CAL';fid='TOY'+str(i)
            pose=dict(available=True,R_physical=[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],centroid=[0.,0.,float(i+1)],extra_original_field='exact')
            record=dict(id=fid,partition=partition,session='toy',cluster_id='one',predictions={a:dict(actual_pose=deepcopy(pose),actions=[{'private_logit':123.}])for a in registered})
            records.append(record);selected.append(dict(id=fid,partition=partition,predictions={a:dict(actual_pose=deepcopy(pose))for a in registered}))
            if i<192:
                image=run/'rgb'/(str(i)+'.png');image.parent.mkdir(exist_ok=True);Image.new('RGB',(4,4),(i,1,2)).save(image)
                inventory.append(dict(id=fid,path=str(image),sha256=P.sha(image),raw_hw=[4,4]))
        write(run/'EXPERIMENT_PLAN.json',dict(status='FROZEN_TOY'))
        jsonl(run/'PREDICTIONS.jsonl.gz',records);jsonl(evaluation/'SELECTED_POSES.jsonl.gz',selected);shutil.copyfile(evaluation/'SELECTED_POSES.jsonl.gz',public/'SELECTED_POSES.jsonl.gz')
        write(run/'PREDICTION_MANIFEST.json',dict(status='ALL_REGISTERED_OUTPUTS_SEALED_BEFORE_REAL_SCORING',real_reference_inputs=False,augmentation_executed=False,arms=registered,ordered_ids=[r['id']for r in records],experiment_plan_sha256=P.sha(run/'EXPERIMENT_PLAN.json'),artifacts={'PREDICTIONS.jsonl.gz':P.sha(run/'PREDICTIONS.jsonl.gz')}))
        write(evaluation/'EVALUATION.json',dict(arms=registered,populations={pop:{a:dict(denominator=n)for a in registered}for pop,n in P.EXPECTED_N.items()}))
        write(evaluation/'EVALUATION_SEAL.json',dict(status='COMPLETE',prediction_manifest_sha256=P.sha(run/'PREDICTION_MANIFEST.json'),artifacts={n:P.sha(evaluation/n)for n in('EVALUATION.json','SELECTED_POSES.jsonl.gz')}))
        write(run/'INVENTORY.json',dict(frames=inventory));Image.new('RGB',(5,5),(255,1,2)).save(public/'derived.png')
        for name in ('README.md','REPORT_KO.md','METHOD_AND_SOURCES.md','REPRODUCE.md'):(public/name).write_text('[derived](derived.png)\n')
        mapping=[dict(original_path=str(evaluation/'SELECTED_POSES.jsonl.gz'),original_sha256=P.sha(evaluation/'SELECTED_POSES.jsonl.gz'),public_path='SELECTED_POSES.jsonl.gz',public_sha256=P.sha(public/'SELECTED_POSES.jsonl.gz'),transform='byte_exact')]
        write(run/'MAP.json',mapping)
        return SimpleNamespace(publication_GO=True,run_dir=run,repo=repo,public_root=public,evaluation=evaluation,rgb_inventory=run/'INVENTORY.json',extra_map=run/'MAP.json',replace_metadata=False,pointer=[])

    def test_full238x26_prepare_verify_and_late_metadata_content_scan(self):
        with tempfile.TemporaryDirectory()as d:
            args=self.fixture(d);prepared=P.prepare(args);self.assertEqual(prepared['selected_output_check']['actual_selected_poses_exact'],238*26)
            result=P.verify(args);self.assertEqual(result['status'],'PASS');self.assertFalse(result['remote_verification_by_this_tool'])
            write(args.public_root/'PUBLICATION_RECEIPT.json',dict(status='ACTUAL_EXTERNAL_RECEIPT',result_commit_sha='a'*40,remote_main_sha='a'*40))
            args.replace_metadata=True;self.assertEqual(P.verify(args)['status'],'PASS')
            write(args.public_root/'PUBLICATION_RECEIPT.json',dict(bad='/'+'home'+'/person/data'))
            with self.assertRaises(ValueError):P.verify(args)
            self.assertEqual(read_json(args.public_root/'PUBLIC_PACK_AUDIT.json')['status'],'FAIL')

    def test_selected_pose_must_preserve_whole_original_dict_and_exclude_bank(self):
        with tempfile.TemporaryDirectory()as d:
            args=self.fixture(d);pm=read_json(args.run_dir/'PREDICTION_MANIFEST.json');data=list(P.OLD.rows(args.public_root/'SELECTED_POSES.jsonl.gz'))
            del data[0]['predictions'][pm['arms'][0]]['actual_pose']['extra_original_field'];jsonl(args.public_root/'SELECTED_POSES.jsonl.gz',data)
            with self.assertRaises(ValueError):P.selected_check(args.run_dir,args.public_root,pm,pm['arms'])
            data[0]['predictions'][pm['arms'][0]]['actual_pose']['extra_original_field']='exact';data[0]['predictions'][pm['arms'][0]]['actions']=[1];jsonl(args.public_root/'SELECTED_POSES.jsonl.gz',data)
            with self.assertRaises(ValueError):P.selected_check(args.run_dir,args.public_root,pm,pm['arms'])

    def test_dynamic_source_addition_invalidates_frozen_pack(self):
        with tempfile.TemporaryDirectory()as d:
            args=self.fixture(d);P.prepare(args);(args.repo/'scripts/research'/P.NAMESPACE/'late.py').write_text('VALUE = 2\n')
            with self.assertRaises(ValueError):P.verify(args)

    def test_selected_status_string_cannot_smuggle_private_bank(self):
        with tempfile.TemporaryDirectory()as d:
            args=self.fixture(d);pm=read_json(args.run_dir/'PREDICTION_MANIFEST.json');arm=pm['arms'][0]
            original=list(P.OLD.rows(args.run_dir/'PREDICTIONS.jsonl.gz'));selected=list(P.OLD.rows(args.public_root/'SELECTED_POSES.jsonl.gz'))
            payload=json.dumps({'candidate_poses':[[1,2,3]]})
            original[0]['predictions'][arm]['reason']=payload;selected[0]['predictions'][arm]['reason']=payload
            jsonl(args.run_dir/'PREDICTIONS.jsonl.gz',original);jsonl(args.public_root/'SELECTED_POSES.jsonl.gz',selected)
            with self.assertRaises(ValueError):P.selected_check(args.run_dir,args.public_root,pm,pm['arms'])

    def test_baseline_pointer_preserves_git_suffix_and_scans_only_exact_new_prefix(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);run=root/'private';repo=root/'repo';run.mkdir();repo.mkdir()
            def git(*args):return subprocess.run(['git','-C',str(repo),*args],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
            git('init');old=('Historical preserved bytes: '+'/'+'home'+'/previous/person\n').encode();(repo/'readme.md').write_bytes(old)
            git('add','readme.md');git('-c','user.name=Constructed','-c','user.email=constructed@example.invalid','commit','-m','toy baseline')
            base=git('rev-parse','HEAD').decode().strip();write(run/'EXPERIMENT_PLAN.json',dict(base_main=base));write(run/'INPUT_MANIFEST.json',dict(base_main=base));(run/'ROOT_README_BASE_BYTES').write_bytes(old)
            (repo/'safe.md').write_text('safe');prefix=b'## New introduction\n\n[Result](safe.md)\n\nNew results retained.\n\n';(repo/'readme.md').write_bytes(prefix+old)
            result=P.baseline_readme_pointer(run,repo);self.assertTrue(result['original_suffix_exact']);self.assertEqual(result['introduced_prefix_lines'],6)
            self.assertIn('NOT_RESCANNED_OR_CERTIFIED',result['preexisting_baseline_content_review'])
            (repo/'readme.md').write_bytes(prefix+old+b'changed')
            with self.assertRaises(ValueError):P.baseline_readme_pointer(run,repo)
            bad_prefix=prefix.replace(b'New results retained.',('/'+'home'+'/new/person').encode());(repo/'readme.md').write_bytes(bad_prefix+old)
            with self.assertRaises(ValueError):P.baseline_readme_pointer(run,repo)
            (repo/'readme.md').write_bytes(prefix+old);(run/'ROOT_README_BASE_BYTES').write_bytes(old+b'mutated')
            with self.assertRaises(ValueError):P.baseline_readme_pointer(run,repo)

    def test_only_root_synthetic_coverage_actions_column_is_typed_scalar_count(self):
        with tempfile.TemporaryDirectory()as d:
            root=Path(d);path=root/'SYNTHETIC_COVERAGE.csv';path.write_text('id,actions\none,0\ntwo,12\n')
            self.assertEqual(P.scan_asset(path,root,root,set(),set())['kind'],'decoded_text')
            for value in ('-1','1.5','NaN','[1]','{}'):
                path.write_text('id,actions\none,'+value+'\n')
                with self.assertRaises(ValueError):P.scan_asset(path,root,root,set(),set())
            other=root/'OTHER.csv';other.write_text('actions\n12\n')
            with self.assertRaises(ValueError):P.scan_asset(other,root,root,set(),set())


def read_json(path):return json.loads(path.read_text())


if __name__=='__main__':unittest.main()
