"""Isolated local bare-Git fixtures; never access the research remote or GT."""
from __future__ import annotations
import hashlib,json,os,subprocess,tempfile,unittest
from pathlib import Path
from . import publication_verify as V


def command(repo,*args):
    return subprocess.run(['git',*args],cwd=repo,env=dict(os.environ,GIT_OPTIONAL_LOCKS='0'),capture_output=True,check=True)


class PublicationFixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='publication-verify-fixture-');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.remote=self.root/'remote.git';self.original=self.root/'original';self.repo=self.root/'research';self.run=self.root/'private';self.run.mkdir()
        command(self.root,'init','--bare',str(self.remote));command(self.root,'init','-b','main',str(self.original))
        command(self.original,'config','user.name','Fixture');command(self.original,'config','user.email','fixture@example.invalid')
        (self.original/'readme.md').write_text('preserved original index\n')
        command(self.original,'add','readme.md');command(self.original,'commit','-m','fixture base')
        command(self.original,'remote','add','origin',str(self.remote));command(self.original,'push','origin','main')
        self.base=command(self.original,'rev-parse','HEAD').stdout.decode().strip()
        command(self.root,'clone','--branch','main',str(self.remote),str(self.repo))
        command(self.repo,'config','user.name','Fixture');command(self.repo,'config','user.email','fixture@example.invalid')
        self.save('START_BEFORE_FETCH.json',dict(original_HEAD=self.base,status_sha256=V.sha_bytes(command(self.original,'status','--porcelain=v1','-z').stdout),protected_files_sha256={'readme.md':V.sha(self.original/'readme.md')}))
        self.save('MIDRUN_GIT_STATE.json',dict(execution_base=self.base))
        self.save('FINAL_PUBLICATION_PROOF_SCHEMA_B.json',dict(schema='FINAL_PUBLICATION_PROOF_SCHEMA_SPEC_B_V1',is_schema_spec_only=True,remote_receipt_expected_shape={'schema':'REMOTE_RESULT_COMMIT_VERIFICATION_V1'},git_proof_expected_shape={'schema':'SCOPED_RESEARCH_PUBLICATION_GIT_PROOF_V1'}))
        self.public=self.repo/V.PUBLIC_ROOT;self.public.mkdir(parents=True);(self.public/'README.md').write_text('Scientific result fixture; no remote claim.\n')
        (self.public/'figure.png').write_bytes(b'fixture PNG blob; no scientific pixels')
        (self.public/'gallery').mkdir();(self.public/'gallery/index.html').write_text('<html>fixture</html>\n')
        code=self.repo/V.CODE_ROOT;code.mkdir(parents=True);(code/'fixture.py').write_text('"""Fixture source only."""\n')
        self.refresh_manifest()
        paths=sorted(str(p.relative_to(self.repo))for p in self.repo.rglob('*')if p.is_file()and '.git'not in p.parts and p!=self.repo/'readme.md')
        self.allow=dict(schema='RESULT_A_STAGE_ALLOWLIST_V1',repo_root=str(self.repo),public_root=V.PUBLIC_ROOT,allowed_code_roots=[V.CODE_ROOT],allowed_index_paths=[],paths=[dict(path=p,sha256=V.sha(self.repo/p))for p in paths])
        self.save('ALLOW.json',self.allow)
        command(self.repo,'add','--',*paths)
        raw=command(self.repo,'diff','--cached','--name-only','-z').stdout;(self.run/'cached_paths.nul').write_bytes(raw)
        command(self.repo,'commit','-m','fixture result A');self.result=command(self.repo,'rev-parse','HEAD').stdout.decode().strip()
        fetch=command(self.repo,'fetch','--no-tags','origin','main')
        self.pre=dict(schema='RESULT_A_PREPUSH_RECEIPT_V1',repo_root=str(self.repo),result_commit=self.result,base_commit=self.base,original_worktree_root=str(self.original),remote_main_before=self.base,fetch_exit_code=fetch.returncode,fetch_argv=['git','fetch','--no-tags','origin','main'],related_diff_reviewed=True,affected_rechecks_passed=True,failures_and_limitations_visible=True,stage_allowlist_sha256=V.sha(self.run/'ALLOW.json'),staged_paths=raw.rstrip(b'\0').decode().split('\0'),staged_paths_raw_path='cached_paths.nul',staged_paths_stdout_sha256=V.sha_bytes(raw))
        self.save('PREPUSH.json',self.pre)
        push=command(self.repo,'push','--porcelain','origin',self.result+':refs/heads/main');(self.run/'push.stdout').write_bytes(push.stdout);(self.run/'push.stderr').write_bytes(push.stderr)
        self.push=dict(schema='RESULT_A_PUSH_RECEIPT_V1',result_commit=self.result,source_commit=self.result,remote_main_before=self.base,push_argv=['git','push','--porcelain','origin',self.result+':refs/heads/main'],exit_code=push.returncode,force_used=False,normal_fast_forward=True,stdout_path='push.stdout',stderr_path='push.stderr',stdout_sha256=V.sha(self.run/'push.stdout'),stderr_sha256=V.sha(self.run/'push.stderr'))
        self.save('PUSH.json',self.push)

    def save(self,name,value):
        (self.run/name).write_bytes(V.encoded(value))

    def refresh_manifest(self):
        rows=[dict(path=str(p.relative_to(self.public)),sha256=V.sha(p))for p in sorted(self.public.rglob('*'))if p.is_file()and p.name not in('PUBLICATION_CHECK.json','PUBLICATION_MANIFEST.json')]
        manifest=dict(complete_preparation=True,artifacts=rows);(self.public/'PUBLICATION_MANIFEST.json').write_bytes(V.encoded(manifest))
        check=dict(status='PASS',manifest_sha256=V.sha(self.public/'PUBLICATION_MANIFEST.json'))
        (self.public/'PUBLICATION_CHECK.json').write_bytes(V.encoded(check))

    def verify(self,GO=True):
        return V.verify(self.repo,self.run,self.result,self.run/'PREPUSH.json',self.run/'PUSH.json',self.run/'ALLOW.json',GO=GO)

    def assert_failure(self,pattern):
        with self.assertRaisesRegex((ValueError,PermissionError),pattern):self.verify()
        self.assertFalse((self.run/'PUBLICATION_GIT_PROOF.json').exists())
        self.assertFalse((self.run/'REMOTE_PUBLICATION_RECEIPT.json').exists())
        failures=list((self.run/'publication_verification_attempts').glob('*/FAILURE.json'))
        self.assertEqual(len(failures),1);self.assertFalse(V.read(failures[0])['actual_remote_publication_verified'])

    def test_actual_network_and_immutable_A_blob_evidence(self):
        before=command(self.original,'status','--porcelain=v1','-z').stdout
        result=self.verify();self.assertEqual(result['status'],'PUSHED_MAIN_VERIFIED');self.assertEqual(result['remote_main_sha'],self.result)
        self.assertEqual(len(result['verified_artifacts']),len(self.allow['paths']));self.assertTrue(all(r['match']for r in result['verified_artifacts']))
        proof=V.read(self.run/'PUBLICATION_GIT_PROOF.json');self.assertTrue(proof['stage_only_allowlisted']);self.assertEqual(result['git_proof_sha256'],V.sha(self.run/'PUBLICATION_GIT_PROOF.json'))
        self.assertEqual(command(self.original,'status','--porcelain=v1','-z').stdout,before)
        self.assertEqual((self.run/'publication_snapshots/result_A/PUBLICATION_CHECK.json').read_bytes(),(self.public/'PUBLICATION_CHECK.json').read_bytes())
        commands=V.read(next((self.run/'publication_verification_attempts').glob('*/COMMANDS.json')))
        self.assertEqual(sum(r['argv'][:3]==['git','ls-remote','origin']for r in commands),2)
        self.assertIn(['git','fetch','--no-tags','origin','main'],[r['argv']for r in commands])

    def test_GO_required(self):
        with self.assertRaisesRegex(PermissionError,'verification-GO'):self.verify(GO=False)
        self.assertFalse((self.run/'publication_verification_attempts').exists())

    def test_allowlist_wrong_expected_A_hash(self):
        self.allow['paths'][0]['sha256']='0'*64;self.save('ALLOW.json',self.allow);self.pre['stage_allowlist_sha256']=V.sha(self.run/'ALLOW.json');self.save('PREPUSH.json',self.pre)
        self.assert_failure('blob differs')

    def test_actual_staging_raw_hash_required(self):
        (self.run/'cached_paths.nul').write_bytes(b'forged\0');self.assert_failure('cached-path bytes changed')

    def test_force_push_receipt_rejected(self):
        self.push['force_used']=True;self.save('PUSH.json',self.push);self.assert_failure('normal non-force')

    def test_original_protected_bytes_preserved(self):
        (self.original/'readme.md').write_text('external changed fixture\n');self.assert_failure('Original user worktree differs')

    def test_raw_actual_push_output_hash_required(self):
        (self.run/'push.stdout').write_bytes(b'forged push output');self.assert_failure('push output log hash')

    def test_out_of_scope_paths_rejected(self):
        self.allow['paths'][0]['path']='papers/untouched.tex';self.save('ALLOW.json',self.allow);self.pre['stage_allowlist_sha256']=V.sha(self.run/'ALLOW.json');self.save('PREPUSH.json',self.pre)
        self.assert_failure('outside the explicit publication scopes')

    def test_snapshot_conflict_preserved(self):
        snap=self.run/'publication_snapshots/result_A';snap.mkdir(parents=True);(snap/'PUBLICATION_CHECK.json').write_bytes(b'existing evidence')
        self.assert_failure('immutable A snapshot');self.assertEqual((snap/'PUBLICATION_CHECK.json').read_bytes(),b'existing evidence')

    def test_canonical_proofs_never_overwritten(self):
        self.verify();before=V.sha(self.run/'PUBLICATION_GIT_PROOF.json')
        with self.assertRaisesRegex(ValueError,'immutable canonical A'):self.verify()
        self.assertEqual(V.sha(self.run/'PUBLICATION_GIT_PROOF.json'),before)

    def test_later_B_metadata_does_not_require_A_hashes(self):
        (self.public/'README.md').write_text('Later B metadata may mention older A.\n')
        command(self.repo,'add','--',str((self.public/'README.md').relative_to(self.repo)));command(self.repo,'commit','-m','fixture metadata B')
        result=self.verify();self.assertEqual(result['result_commit'],self.result)
        proof=V.read(self.run/'PUBLICATION_GIT_PROOF.json');self.assertFalse(proof['current_worktree_comparison']['required_to_match_A'])
        self.assertTrue(any(not r['match']for r in proof['current_worktree_comparison']['files']))


if __name__=='__main__':unittest.main()
