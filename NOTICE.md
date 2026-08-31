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

These figures are documentation assets, not student-authored source code. They
are excluded from the MIT License applied to that code. Because they derive
from Cholec80 experiments, confirm the Cholec80 data terms, attribution
requirements, and applicable coursework conditions before reusing them.

## Cholec80 data

No Cholec80 video, extracted frame, annotation file, or model checkpoint is
distributed in this repository. Cholec80 access and use remain governed by
CAMMA's process and terms:

- https://camma.unistra.fr/datasets/

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
pixels, documentation figures in `assets/`, coursework or assessment
materials, checkpoints or model outputs, or external dependencies.

`tf_cholec80/dataset.py` remains governed solely by the CC BY-NC-SA 4.0 terms
described above. The documentation figures remain subject to the Cholec80 data
terms and applicable coursework conditions. Nothing in the MIT License grants
rights to those excluded materials.
