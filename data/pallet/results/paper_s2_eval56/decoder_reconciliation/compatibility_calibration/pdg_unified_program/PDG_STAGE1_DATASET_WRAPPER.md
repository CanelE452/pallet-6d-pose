# Source-space wrapper: design resolved, not yet built

The two unknowns that blocked the wrapper are answered.  Nothing about the
design is open now.

## HARD_BLOCKED #1 does not apply — sample identity resolves to source paths

```
CleanVisiiDopeLoader.imgs[index] == (image_path, name, json_path)   absolute
utils_dataset.py:433  path_img, img_name, path_json = self.imgs[index]
utils_dataset.py:434  img = np.array(Image.open(path_img).convert("RGB"))
```

No filename search, no root guessing: the dataset already carries the pair.

## The insertion point TACA needs already exists

```
utils_dataset.py:__getitem__
  _load_raw(index)                       source RGB at native resolution
  _collect_keypoints(data_json)          9 kp in source pixels
  if self.truncation_aug_prob > 0:
      apply_truncation_aug(img, kps9, rng)     <- ORIGINAL image, pre-albumentations
  A.Resize(400, 400) when aspect_resize     anisotropic squash
  CreateBeliefMap(size=50, sigma=self.sigma)
  GenerateMapAffinity(size=50, nb_vertex=8)
```

The legacy augmentation runs exactly where TACA has to run -- on the source
frame before the squash.  The seam is a module-level lookup, so a subclass can
substitute a TACA adapter for that one call and inherit the rest of the geometry
unchanged, which is what makes no-transform parity checkable rather than
asserted.

## Off-screen keypoints survive the transform

```
utils_dataset.py:575, 594
    keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
```

This was the remaining risk: had albumentations dropped out-of-bounds
keypoints, the nine-channel structure would break under TACA and the off-screen
target policy would have nothing to mask.  `remove_invisible=False` keeps them,
so a corner pushed outside the frame arrives at the target builder with its true
out-of-frame coordinate.

## Resolved wrapper design

```
subclass CleanVisiiDopeLoader
  A1: truncation_aug_prob = 0
  A2: substitute a TACA adapter at the apply_truncation_aug seam
  both: call super().__getitem__(index) for image, geometry and refine_keypoints
  then regenerate from the parent's own transformed keypoints:
      belief          pdg_targets.build_targets, sigma 2.0 / 2.5
      belief mask     0 on off-screen channels
      affinity        parent's map, channels 2i and 2i+1 masked with the corner
      visibility      three-state from the transformed coordinate
      palletness      pdg_targets.palletness_target
      truncation      any corner off-screen
      DiffPnP         gated to fully in-frame samples
```

Parity is then checkable against the parent directly: same image tensor, same
geometry, same affinity generator, and only the belief widths and the masks
differ by construction.

## State

```
wrapper implemented     no
trainer implemented     no
optimizer steps         0
checkpoints             0
E44 inference           0     SEALED
W45 inference           0     SEALED
```
