# V259 public audit publication helper

CODE-only preparation. No real payload or Git operation was run during preparation.

Root supplies a separate true entry with complete `{path,sha256,size_bytes}` source bindings and strict package-relative paths. Include all named public raw/geometry/ledgers/QA/figures, preserved failed-parser V1 and Root documents explicitly; the helper does not discover files or choose outcomes. `README_KO.md` is required. The false template authorizes nothing.

Use the same exact entry bytes for both operations:

```sh
python -B /ABS/publish_public_point_teacher.py --entry /ABS/ROOT_TRUE_ENTRY.json --mode package
python -B /ABS/publish_public_point_teacher.py --entry /ABS/ROOT_TRUE_ENTRY.json --mode stage
```

The package is fresh, opaque raw byte copies plus `STARTED.json` and `MANIFEST.json`. The manifest embeds original entry UTF8 and its SHA/size; every copied file has a SHA/size. No gzip, source decoder, image/array/model loader, Git operation or network request is used in package mode. Maximum source leaf is 90MiB.

Stage verifies local HEAD and live `origin/main` against the fixed parent, clean tracked work/index, all package bytes and registry HEAD suffixes. It stages only the exact NUL-separated allowlist and compares every index blob. Root files/tools and the three registry additions receive whitespace checks; archived artifacts/assets/references retain exact original bytes. The stage receipt directory must be fresh and outside package/repository. It holds `STAGED_QA.json`, including staged tree, file hashes and original registry suffix hashes. It is not automatically added to Git.

No commit or push is performed. Root can inspect the staged tree and use a separate normal commit/push. A failure preserves partial output/staging state and writes `FAILURE.json` once a fresh run directory exists; no automatic retry, rollback or overwrite is attempted. Guard failures before authorization/fresh-output creation do not mutate existing artifacts.

Scientific claims are unchanged: the two public-frame D/E/U records are policy-derived surface AUX labels, not model correctness/recall, physical native gold or actual initializer query labels. Training and model inference remain zero for this audit; the latest model comparison remains V256.
