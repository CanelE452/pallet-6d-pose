# Paper-release dataset archive

This is the dataset to be published with the paper.  It is kept here as the
authoritative copy.

Replaced on 2026-08-14.  The previous release (`pallet6d_v2_10k_run1.zip` /
`run2.zip`, 20,000 frames, subsets `v2_prod10k2_s810x` + `v2_prod10k3_s820x`)
was found to be faulty and has been deleted.  Do not reintroduce it.

## Contents

```
v2_prod40k_clean_merged/          29G
  rgb/           f{n}_rgb.png       40,000
  labels/        f{n}_label.json    40,000
  mask_visible/  f{n}.png           40,000
  mask_amodal/   f{n}.png           40,000
  records.jsonl                     40,000 lines
```

Frame ids run `f0` .. `f39999` with no zero padding beyond four digits
(`f0000_rgb.png`, `f39999_rgb.png`).  Masks drop the suffix: `f0000.png`.

The naming differs from this repository's `{i:06d}.json` + `{i:06d}.png`
convention, so any ingestion needs a mapping step rather than a straight copy.

`objects[0]` carries `pnp_conditioning`, a key the previous release did not
have.  Both `mask_visible` and `mask_amodal` ship per frame, which the previous
release lacked at the frame level.

`data/pallet/release/attribution/ATTRIBUTION.md` came with these archives.
Read it before publishing: it may carry licence terms on third-party assets.

## Integrity

Verified 2026-08-14 after unpacking and again after moving into this folder.
The frame-id sets of `rgb/`, `labels/`, `mask_visible/` and `mask_amodal/` are
identical to `f0` .. `f39999`: no missing frames, no extras.

```
ef7c8543ef94e13c8595c712af9646d221140ac800303923d54727524ccb06c7  v2_prod40k_clean_merged/records.jsonl
```

Verify at any time:

```
cd data/pallet/training_data/paper_release
sha256sum -c SHA256SUMS
for s in rgb labels mask_visible mask_amodal; do
  echo "$s $(ls v2_prod40k_clean_merged/$s | wc -l)"
done
```

Unlike the previous release this is an unpacked directory, not a sealed
archive, so the per-file checksums that `unzip -t` used to cover no longer
apply.  Only `records.jsonl` is checksummed.

## Provenance

Downloaded to `~/Downloads` as `pallet6d_v2_clean_10k_part1..4.zip` (4 x 6.8G,
2026-08-13 19:56), unpacked 2026-08-14 into
`data/pallet/runs/diagnostics/v2_prod40k_clean_merged/`, then moved here the
same day.  The four zips were deleted after verification, so this unpacked
directory is the only copy on this machine.

## Not under version control

`data/` is in `.gitignore`, so git does not protect these files.  A published
dataset with a single local copy and no version control has no recovery path if
this disk fails or a command deletes it.  The previous release was chmod 444 in
a chmod 555 folder to stop an accidental overwrite; that seal was lifted to
carry out this replacement and has not been reapplied, because the payload is
now a directory that ingestion code reads rather than a sealed archive.  Either
way, file permissions do not substitute for an off-machine backup.
