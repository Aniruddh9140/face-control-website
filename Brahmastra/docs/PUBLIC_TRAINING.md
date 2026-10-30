# Public-data model

EyeSync now includes a NumPy head-pose regressor trained locally on the public
BIWI Kinect Head Pose RGB dataset, using fast.ai's RGB repack. MediaPipe remains
the pretrained landmark detector. The newly trained regressor maps 54 normalized
landmark coordinates to yaw and pitch; a session-neutral position then maps head
rotation to cursor position. Original speech recognition and click timing remain.

## Reproduction

Run from the Brahmastra folder with its Python 3.15 environment:

```
.venv\Scripts\python.exe training\download_biwi.py
.venv\Scripts\python.exe training\train_public_pose.py
.venv\Scripts\python.exe training\check_public_runtime.py
```

The archive is 452,316,199 bytes. SHA256:
`9cfefd53ed85f824c5908bc6eb21fc719583eec57a7df1d8141d3156645693cf`.
The dataset and extracted feature cache stay in `data/public-training`, excluded
from source control. Download URL, citation and terms are in its provenance file.
Model weights are in `models/biwi_head_pose.npz`. Full results and model SHA256
are in [PUBLIC_TRAINING_RESULTS.json](PUBLIC_TRAINING_RESULTS.json).

Every fifth frame was sampled. Input images were resized to 320×240 to match
runtime. Labels derive from the BIWI pose matrix forward vector. The predefined
working domain is yaw ±50°, pitch ±40°; this is a restricted internal experiment,
not the full BIWI benchmark. Frames with no detected face were excluded and counted.

The known repeated subjects were kept together across all their sequences,
using ETH Zurich's sequence-to-subject mapping. Training used 13 people,
validation 3, and the final test 4. Eight configurations were compared on
validation only. After selection, training plus validation were used to fit the
delivered model; the independent test was evaluated afterward.

There were 2,215 usable frames: 1,584 train, 299 validation, 332 test. Final test
MAE was 2.56° yaw, 3.60° pitch, average 3.08°. A constant training-mean prediction
averaged 15.63°. The 90th percentile angular vector error was 8.21°. Dataset
coverage, limited subject diversity, excluded undetected faces and the capture
environment limit generalization. These numbers do not establish real cursor
accuracy or accessibility suitability for an individual.

## Starting control

Connect through the dashboard, face the screen and keep your head steady for
two seconds. The app learns neutral and enables requested control after gesture
baselines settle. Turn your head gently; the camera preview is mirrored. Dashboard
Recenter resets neutral. Ctrl+Alt+E pauses. Nine-dot Calibrate pointer remains
available and a valid personal profile takes priority. No personal profile is
fabricated from the public dataset.

## Sources and terms

- [ETH Zurich BIWI dataset and terms](https://huggingface.co/datasets/ETHZurich/biwi_kinect_head_pose)
- [ETH sequence-to-subject mapping](https://huggingface.co/datasets/ETHZurich/biwi_kinect_head_pose/blob/main/biwi_kinect_head_pose.py)
- [fast.ai dataset catalog](https://github.com/fastai/fastai/blob/master/fastai/data/external.py)

BIWI is available for non-commercial use such as university research and
education. This local model is part of the user's non-commercial research setup;
review those terms before commercial use or redistribution.
