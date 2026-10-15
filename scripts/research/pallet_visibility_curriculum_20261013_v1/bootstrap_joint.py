"""Restore the hash-bound public ImageNet backbone for the optional RGB head."""
import argparse
from pathlib import Path
import urllib.request

from . import common as C

URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"


def restore(descriptor_path):
    descriptor_path = Path(descriptor_path).resolve()
    descriptor = C.read(descriptor_path)
    if descriptor["backbone"]["weights_sha256"] != SHA256:
        raise ValueError("Descriptor does not bind the declared official backbone")
    parameters = descriptor_path.parent / descriptor["arrays_file"]
    if C.sha(parameters) != descriptor["arrays_sha256"]:
        raise ValueError("Published fitted parameters changed")
    target = C.ROOT / descriptor["backbone"]["weights_repo_relative"]
    if target.exists():
        if C.sha(target) != SHA256:
            raise ValueError("Preserve an existing file with a different checksum: "+str(target))
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(target.suffix+".pending")
    if pending.exists():
        raise FileExistsError("Preserve a previous partial download: "+str(pending))
    with urllib.request.urlopen(URL, timeout=60) as source, pending.open("xb") as output:
        while True:
            block = source.read(1024*1024)
            if not block:
                break
            output.write(block)
    if C.sha(pending) != SHA256:
        raise ValueError("Downloaded public weights fail SHA256; partial file preserved")
    pending.replace(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, default=C.DOC/"nonsevere_v1/joint_model/MODEL.json")
    arguments = parser.parse_args()
    print("JOINT_BACKBONE_READY", restore(arguments.descriptor))
