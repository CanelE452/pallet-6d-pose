# Field-only DeepLSD prototype

The loader uses the unmodified official VGGUNet source and exact inference heads from commit `f7d9d6258c0cd25d4f6eea882853565403d289be`. The MIT notice is retained in `field_model.py`. All 144 checkpoint entries load strictly. No package was installed and no official source was changed. The missing `pytlsd` import was replaced only in the reference-comparison import by a fail-closed function, which was never called.

`FieldOnlyDeepLSD(repository, checkpoint)` accepts FP32 gray `[B,1,H,W]` in `[0,1]` and returns `df_norm`, `df`, and `line_level` at that exact input resolution. `predict_canvas(gray640, [H,W])` performs one B1 rectangular inference and returns square fields: `df=5`, `line_level=0`, `input_extent_mask=false` outside the actual rectangle. The original-content mask from the existing gray cache should additionally exclude reflected and LetterBox regions when sampling pallet evidence.

`df=exp(-df_norm)*5` is a learned distance field in input pixels, capped at 5 by the official head. It is neither a correctness probability nor an estimate of distances beyond that neighborhood. `line_level` is the original pi-periodic line-direction value. Angle interpolation must respect that periodicity, for example through doubled-angle sine/cosine; no angle remapping or pallet role classification occurred in this prototype.

The official MegaDepth/MiniDepth checkpoint introduces generic real-image pretraining. Any matched point/Hough comparison that uses these fields must disclose and share that pretraining. This is not a newly synthetic-only backbone.

Three fixed SHA-selected training images were inferred at batch one. The field-only and official `detect_lines=False` outputs are bit-exact for all three fields, maximum difference0. No real image was inferred, no pallet GT was used, no model was trained, and no full field cache was prepared. Timing is not a benchmark because reference passes and another small task could share the device.

The saved diagnostic figure shows generic pallet boundaries together with fence, building, floor and texture responses. It does not establish semantic pallet-edge accuracy or improvement of the point model.
