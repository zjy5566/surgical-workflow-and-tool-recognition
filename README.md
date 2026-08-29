# Surgical Workflow and Tool Recognition

This is a source-preserving, data-free staging copy based only on:

```text
MPHY0043_Artificial Intelligence for Surgery and Intervention/cw/cw
```

The retained Python files and `instruction.txt` preserve their original names,
directory structure, line endings, comments, model logic, training behavior,
and evaluation code. They have not been replaced with the newer outer-folder
version or with previously corrected code. See `SOURCE_CHECKSUMS.sha256`.

## Original project files retained

```text
config.py
dataset.py
instruction.txt
model.py
run.py
test.py
utils.py
visualization.py
train/*.py
tf_cholec80/__init__.py
tf_cholec80/configs/__init__.py
tf_cholec80/dataset.py
```

The original `tf_cholec80/configs/config.json` was the one exception: it
contained a personal Windows data path and is not included. Copy the public
placeholder before running locally:

```bash
cp tf_cholec80/configs/config.example.json tf_cholec80/configs/config.json
```

Install the retained dependencies with:

```bash
python -m pip install -r requirements.txt
```

The original `instruction.txt` describes the intended training sequence.

## Data and generated files are intentionally excluded

No Cholec80 videos, extracted frames, annotations, report, checkpoint, log,
result image, marking material, or submission archive is included. Obtain
authorized Cholec80 access from CAMMA and comply with its terms:

- https://camma.unistra.fr/datasets/

## Preserved limitations

- `dataset.py` uses the first occurrence returned by `sequence.index()` when a
  phase repeats in a video's phase sequence.
- Prediction-guided training logs a warning and continues with random Task A
  weights when its checkpoint is missing.
- Fixed splits, hyperparameters, relative output paths, and all original model
  behavior are preserved rather than silently corrected.
- Full training was not rerun during this data-free staging pass.
- This code is research/educational software and is not validated for clinical
  use.

## Third-party file

`tf_cholec80/dataset.py` is retained because it was present in the specified
source folder, but it comes from CAMMA's TF-Cholec80 project rather than the
sole student-authored PyTorch pipeline. It is the only clearly identified
vendored third-party source exception in this repository and remains subject to
CC BY-NC-SA 4.0; see `THIRD_PARTY_LICENSES/TF-Cholec80-LICENSE.txt` and
`NOTICE.md`.

The repository owner states that the remaining project code was completed
independently and is sole student-authored work. It still has no public license;
resolve the course and institutional conditions in `LICENSE_PENDING.md` before
publishing.
