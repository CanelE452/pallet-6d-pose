# Self-training recovery — frozen detector, pose-only adaptation

The user goal is improved self-training, not merely finding a favourable diagnostic. All completed attempts are retained below. No existing release is replaced.

## Fixed population and method

Ordinary plastic194 and NEG2689. R0 initialization; freeze backbone, neck, box/class branches and all running buffers; train only pose branches/flow. Same217 unique pseudo images from249 candidates,512 real+512 synthetic exposures per epoch,5epochs/320updates. No IoU selection, no manual review masks, no inference-time refiner. RAW/REF targets have matched trusted-keypoint masks. Existing Replay teacher had prior manual9-image/38-corner supervision; this is not a claim of never using real supervision anywhere in the teacher pipeline.

Six predeclared source/raw/refined × learning-rate screens were followed by two actual order replicates of the selected1e-5 setting, with matched source/raw controls. The loader sorts paths and uses a fixed RNG generator, so indexed aliases ensure orders43/44 truly differ. Same data membership/multiplicities; first-batch image hashes verify RAW/REF pairing and distinct repeats.

## All attempts

| Phase | Arm | PCK20 % | median8 px | IoU3D | t cm | ADDsym |
|---|---|---:|---:|---:|---:|---:|
| bn_probe | STUDENT_R0BN | 75.611 | 8.498 | 0.54175 | 12.255 | 0.29589 |
| bn_probe | R0_STUDENTBN | 77.925 | 6.908 | 0.61581 | 8.919 | 0.38640 |
| pose_only | SYN_LR4 | 77.660 | 7.312 | 0.58963 | 9.587 | 0.36415 |
| pose_only | RAW_LR4 | 77.132 | 7.376 | 0.59079 | 9.533 | 0.36996 |
| pose_only | REF_LR4 | 78.255 | 6.954 | 0.60104 | 9.943 | 0.37048 |
| pose_only | SYN_LR5 | 78.057 | 7.450 | 0.59421 | 10.123 | 0.35643 |
| pose_only | RAW_LR5 | 77.859 | 7.288 | 0.59411 | 10.011 | 0.36199 |
| pose_only | REF_LR5 | 78.718 | 6.722 | 0.59405 | 10.227 | 0.36621 |
| pose_repeat | SYN_ORDER43 | 78.255 | 7.391 | 0.59038 | 10.093 | 0.35437 |
| pose_repeat | RAW_ORDER43 | 77.726 | 7.282 | 0.58939 | 10.095 | 0.36334 |
| pose_repeat | REF_ORDER43 | 78.652 | 6.565 | 0.59830 | 10.431 | 0.36668 |
| pose_repeat | SYN_ORDER44 | 78.321 | 7.381 | 0.59542 | 10.215 | 0.35881 |
| pose_repeat | RAW_ORDER44 | 77.792 | 7.329 | 0.59097 | 10.196 | 0.36245 |
| pose_repeat | REF_ORDER44 | 78.586 | 6.720 | 0.59430 | 10.454 | 0.36216 |

## Selected method: all three runs, not best repeat

| Model | PCK20 % | paper kp median px | paper kp P90 px | AP50–95 | FPR95 | IoU3D | ADDsym |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | 77.859 | 7.275 | 39.363 | 0.7194 | 0.0439 | 0.58572 | 0.34476 |
| REF_LR5 | 78.718 | 6.693 | 40.208 | 0.7194 | 0.0439 | 0.59405 | 0.36621 |
| REF_ORDER43 | 78.652 | 6.573 | 40.195 | 0.7194 | 0.0439 | 0.59830 | 0.36668 |
| REF_ORDER44 | 78.586 | 6.702 | 39.802 | 0.7194 | 0.0439 | 0.59430 | 0.36216 |

## Uncertainty and limits

The194 images are repeatedly used DEV, not independent test data. Session-clustered bootstrap intervals below are exploratory and not adjusted for the search. Seed/order repeatability is not new image-level validation. MAIN6D uses the frozen geometry-reconstructed reference, not externally measured ground-truth pose. PCK20 uses whole-object symmetry and8corners, while paper median/P90 uses fixed9-keypoint supervision on matched detections.

- REF_LR5: primary/control flags {'pck20_above_R0': True, 'keypoint_median_below_R0': True, 'ap_preserved': True, 'iou3d_preserved': True, 'pck20_above_SYN': True, 'pck20_above_RAW': True}; clustered PCK20 contrasts {'R0': {'delta_pp': 0.8592200925313945, 'CI95_pp': [0.0, 1.7651176876175505], 'clusters': 9, 'bootstrap_replicates': 10000, 'exploratory': True, 'multiple_search_adjustment': False}, 'SYN': {'delta_pp': 0.6609385327164574, 'CI95_pp': [-0.4709761262995764, 1.8805396426887118], 'clusters': 9, 'bootstrap_replicates': 10000, 'exploratory': True, 'multiple_search_adjustment': False}, 'RAW': {'delta_pp': 0.8592200925313945, 'CI95_pp': [-0.5848813209494323, 2.374065127307664], 'clusters': 9, 'bootstrap_replicates': 10000, 'exploratory': True, 'multiple_search_adjustment': False}}
- REF_ORDER43: primary/control flags {'pck20_above_R0': True, 'keypoint_median_below_R0': True, 'ap_preserved': True, 'iou3d_preserved': True, 'pck20_above_SYN': True, 'pck20_above_RAW': True}; clustered PCK20 contrasts {'R0': {'delta_pp': 0.7931262392597488, 'CI95_pp': [-0.3018184636520026, 1.884002398799241], 'clusters': 9, 'bootstrap_replicates': 10000, 'exploratory': True, 'multiple_search_adjustment': False}, 'SYN': {'delta_pp': 0.3965631196298744, 'CI95_pp': [-1.1545589826839824, 1.7567567567567568], 'clusters': 9, 'bootstrap_replicates': 10000, 'exploratory': True, 'multiple_search_adjustment': False}, 'RAW': {'delta_pp': 0.9253139458030403, 'CI95_pp': [-0.4360862678744084, 2.3504901960784306], 'clusters': 9, 'bootstrap_replicates': 10000, 'exploratory': True, 'multiple_search_adjustment': False}}
- REF_ORDER44: primary/control flags {'pck20_above_R0': True, 'keypoint_median_below_R0': True, 'ap_preserved': True, 'iou3d_preserved': True, 'pck20_above_SYN': True, 'pck20_above_RAW': True}; clustered PCK20 contrasts {'R0': {'delta_pp': 0.7270323859881032, 'CI95_pp': [-0.1481508936660901, 1.5957539090761437], 'clusters': 9, 'bootstrap_replicates': 10000, 'exploratory': True, 'multiple_search_adjustment': False}, 'SYN': {'delta_pp': 0.26437541308658297, 'CI95_pp': [-1.0624523183860302, 1.5068493150684932], 'clusters': 9, 'bootstrap_replicates': 10000, 'exploratory': True, 'multiple_search_adjustment': False}, 'RAW': {'delta_pp': 0.7931262392597488, 'CI95_pp': [-0.7692307692307693, 2.2928994082840237], 'clusters': 9, 'bootstrap_replicates': 10000, 'exploratory': True, 'multiple_search_adjustment': False}}

No assertion of uniformly improved tails or all viewpoints. P90 and subgroup harms must be disclosed even if primary central-accuracy metrics improve. No automatic final-model promotion.

## Evaluation provenance correction

The full194 evaluation includes 3 images previously used for manual supervision of the frozen Replay teacher: ['plastic_night_01:036918', 'plastic_night_01:039283', 'plastic_night_01:040717']. The student did not directly train on these labels, but the complete teacher/student pipeline is not independent of them. Preserve full194 results as exploratory development evidence, not independent test results.

Excluding exact teacher images or their entire overlapping session below is a posthoc sensitivity check, not a newly independent holdout and not a replacement of the original primary population.

```json
{
  "exclude_exact_teacher_images": {
    "images": 191,
    "R0_PCK20": 77.50167897918065,
    "runs": {
      "REF_LR5": {
        "pck20": 78.3747481531229,
        "contrast_R0": {
          "delta_pp": 0.8730691739422431,
          "CI95_pp": [
            0.0,
            1.7806361770741717
          ],
          "clusters": 9,
          "bootstrap_replicates": 10000,
          "exploratory": true,
          "multiple_search_adjustment": false
        }
      },
      "REF_ORDER43": {
        "pck20": 78.30758898589657,
        "contrast_R0": {
          "delta_pp": 0.8059100067159167,
          "CI95_pp": [
            -0.3105590062111801,
            1.9040924338879204
          ],
          "clusters": 9,
          "bootstrap_replicates": 10000,
          "exploratory": true,
          "multiple_search_adjustment": false
        }
      },
      "REF_ORDER44": {
        "pck20": 78.24042981867025,
        "contrast_R0": {
          "delta_pp": 0.7387508394895903,
          "CI95_pp": [
            -0.15267759180148002,
            1.6082239048325033
          ],
          "clusters": 9,
          "bootstrap_replicates": 10000,
          "exploratory": true,
          "multiple_search_adjustment": false
        }
      }
    }
  },
  "exclude_teacher_overlap_sessions": {
    "images": 172,
    "R0_PCK20": 76.06581899775617,
    "runs": {
      "REF_LR5": {
        "pck20": 77.03814510097233,
        "contrast_R0": {
          "delta_pp": 0.9723261032161555,
          "CI95_pp": [
            0.0,
            1.9113149847094801
          ],
          "clusters": 8,
          "bootstrap_replicates": 10000,
          "exploratory": true,
          "multiple_search_adjustment": false
        }
      },
      "REF_ORDER43": {
        "pck20": 76.96335078534031,
        "contrast_R0": {
          "delta_pp": 0.8975317875841436,
          "CI95_pp": [
            -0.3740648379052369,
            2.05285884860412
          ],
          "clusters": 8,
          "bootstrap_replicates": 10000,
          "exploratory": true,
          "multiple_search_adjustment": false
        }
      },
      "REF_ORDER44": {
        "pck20": 76.8885564697083,
        "contrast_R0": {
          "delta_pp": 0.8227374719521316,
          "CI95_pp": [
            -0.17857142857142858,
            1.739172483390799
          ],
          "clusters": 8,
          "bootstrap_replicates": 10000,
          "exploratory": true,
          "multiple_search_adjustment": false
        }
      }
    }
  }
}
```
