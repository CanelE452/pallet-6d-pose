# Third-party data and model notices

YCB-Video imagery/annotations: Yu Xiang, Tanner Schmidt, Venkatraman Narayanan and Dieter Fox, PoseCNN (RSS 2018). Original project: https://rse-lab.cs.washington.edu/projects/posecnn/ . Author dataset/toolbox README: https://github.com/yuxng/YCB_Video_toolbox/blob/master/README.md . Official BOP YCB-V format/source/license: https://bop.felk.cvut.cz/datasets/ . Both author README and BOP YCB-V entry identify MIT licensing (checked2026-10-25). Author LICENSE: https://github.com/yuxng/YCB_Video_toolbox/blob/master/LICENSE .

MIT License

Copyright (c) 2017 Yu Xiang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The three clean public SOURCE RGB examples retain their original encoded bytes. Crops, overlays and opaque cutout composites are derivative experiment visualizations; masks/coordinates and SHA bindings preserve the source links. No new private/current223 image is included.

DINOv2 fixed base: Meta DINOv2, https://github.com/facebookresearch/dinov2 ; pinned vendor7764ea0f912e53c92e82eb78a2a1631e92725fc8 and official checkpoint SHA are recorded in experiment entries. Official base weights and vendor source tree are not redistributed in this package. The small experiment head checkpoints are new trained artifacts, not copies of the official base checkpoint. The rendered pallet SOURCE provenance is documented in the V234 renderer audit. This notice does not relicense unrelated files in the repository.

V239–V240 actual checkpoint: six frozen SOURCE feature arrays, two newly trained tiny V239 heads and derivative SOURCE crop/context visualizations are added. The heads are newly trained from their original seed17 initial states, not copies of a historical trained CONTROL. Official DINO weights/vendor tree and private evaluation images remain excluded. SOURCE YCB native correspondence remains UNKNOWN.

V246 includes selected KU Leuven Pallox public-source derivative certificates and four already-generated SOURCE gallery PNG copies. Dataset: Yanming Wu and Eric Demeester, DOI10.48804/DJLDFS, recorded CC-BY-4.0. No official weights, original mixed source manifests, raw depth/mesh/pose or held RGB are redistributed by this builder. Conditional teacher support is not native gold.
