"""Verify only packaged files; never modify the repository or install dependencies."""
from pathlib import Path
import hashlib,json,sys

root=Path(__file__).resolve().parent
manifest=json.loads((root/'BUNDLE_MANIFEST.json').read_text(encoding='utf8'))
errors=[]
for relative,expected in manifest['files_sha256'].items():
    path=(root/relative).resolve()
    if not path.is_relative_to(root):errors.append(relative+': unsafe path');continue
    if not path.is_file():errors.append(relative+': missing');continue
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=expected:errors.append(relative+': digest mismatch')
if errors:
    print('\n'.join(errors));sys.exit(1)
print(f"Verified {len(manifest['files_sha256'])} packaged files. This verifies bytes, not model accuracy.")
