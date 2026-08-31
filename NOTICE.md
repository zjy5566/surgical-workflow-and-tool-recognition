# Provenance and third-party notice

## Student-authored project source

The retained project code was copied unchanged from the user's specified inner
`cw/cw` folder. No outer-folder revision, personal path configuration, dataset,
checkpoint, log, marking file, submission archive, or course report is
included. `SOURCE_CHECKSUMS.sha256` records the preserved source files.

The repository owner states that all retained project code other than the
specific TF-Cholec80 exception below was completed independently and is sole
student-authored work. Several source files contain their own statements about
Google Gemini assistance; those statements remain unchanged.

The expanded `README.md`, public configuration example, ignore rules,
dependency list, notices, and curated documentation assets were added during
repository preparation and are not part of the checksum-preserved coursework
source set.

## Documentation figures

The following figures were selected from the repository owner's experiment and
report materials, renamed, flattened onto a white background, and converted to
JPEG for readable GitHub display:

- `assets/two-stage-workflow-guided-architecture.jpg`
- `assets/selected-tool-timeline-intervals.jpg`
- `assets/video49-timeline-part1.jpg`
- `assets/video49-timeline-part2.jpg`

The conversion did not alter the plotted results. No visible name, medical
record number, hospital identifier, face, account name, or local filesystem
path was identified in the selected figures during review. The architecture
figure contains small laparoscopic frame thumbnails, and the other figures
visualize predictions derived from Cholec80.

These figures are documentation assets, not MIT-licensed project source code.
They are based on the Cholec80 dataset released by the CAMMA Research Group,
University of Strasbourg, under Creative Commons
Attribution-NonCommercial-ShareAlike 4.0:

- https://camma.unistra.fr/datasets/
- https://creativecommons.org/licenses/by-nc-sa/4.0/

Modifications include frame sampling, resizing, prediction and annotation
visualization, figure composition, background flattening, and JPEG conversion.
The resulting Cholec80-derived figures are shared under CC BY-NC-SA 4.0. No
endorsement by CAMMA is implied. Users must preserve attribution, link the
license, identify modifications, comply with the non-commercial restriction,
and share adaptations under the same or a compatible license.

## Cholec80 data

No standalone Cholec80 video, extracted-frame file, annotation file, or model
checkpoint is distributed in this repository. A small number of resized frame
thumbnails are embedded in the architecture figure described above. Cholec80
access and use remain governed by CAMMA's process and CC BY-NC-SA 4.0 terms:

- https://camma.unistra.fr/datasets/

The dataset is associated with the following work, which users should cite:

A. P. Twinanda, S. Shehata, D. Mutter, J. Marescaux, M. de Mathelin, and
N. Padoy, “EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic
Videos,” *IEEE Transactions on Medical Imaging*, 2017.
https://doi.org/10.1109/TMI.2016.2593957

## TF-Cholec80

`tf_cholec80/dataset.py` is a verbatim third-party file from the University of
Strasbourg/CAMMA TF-Cholec80 project. It is not used by the main PyTorch
pipeline and is the only clearly identified vendored third-party source file:

- https://github.com/CAMMA-public/TF-Cholec80

It remains under Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
Its license is reproduced at
`THIRD_PARTY_LICENSES/TF-Cholec80-LICENSE.txt`. Preserve attribution, identify
changes, comply with the non-commercial and share-alike conditions, and do not
replace this file's license with a license selected for the student-authored
code.

PyTorch, torchvision, TensorFlow, NumPy, Pillow, tqdm, Matplotlib,
scikit-learn, SciPy, and setuptools are external dependencies and are not
vendored here. Each remains governed by its own license.

## License scope

The MIT License in `LICENSE` applies only to the following independently
authored project source files:

- `config.py`
- `dataset.py`
- `model.py`
- `run.py`
- `test.py`
- `utils.py`
- `visualization.py`
- `train/Timed_Label_Guided.py`
- `train/Timed_Multi_Task_Label-Guided.py`
- `train/Timed_Multi_Task_Pred-Guided.py`
- `train/train_A.py`
- `train/train_baseline.py`

The MIT License does not apply to `tf_cholec80/dataset.py`, any other
third-party material, Cholec80 videos or annotations, extracted medical-image
pixels, Cholec80-derived documentation figures in `assets/`, coursework or
assessment materials, checkpoints or model outputs, or external dependencies.

`tf_cholec80/dataset.py` and the Cholec80-derived documentation figures remain
governed by CC BY-NC-SA 4.0 as described above. The repository is therefore a
mixed-license research repository: independently authored project code is MIT,
while the identified Cholec80-derived and TF-Cholec80 materials are
CC BY-NC-SA 4.0. Nothing in the MIT License replaces or extends to the
CC-licensed components.
