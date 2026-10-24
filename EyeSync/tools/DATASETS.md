# Public dataset evaluation

## GRID audiovisual speech corpus

- **Use here:** false-positive evaluation and conservative closure-model tuning on people reading sentences, with speaker-disjoint development and held-out test sets.
- **Authors:** Martin Cooke, Jon Barker, Stuart Cunningham and Xu Shao (2006).
- **Paper:** *An audio-visual corpus for speech perception and automatic speech recognition*, Journal of the Acoustical Society of America 120(5), 2421–2424.
- **Official record:** https://zenodo.org/records/3625687
- **License:** CC BY 4.0. Attribution and the official record are also embedded in every extracted NPZ sidecar.
- **Privacy/storage:** the workflow downloads the public videos temporarily, verifies Zenodo's MD5, and deletes them after extracting numerical features. Raw videos and face images are not committed.

Readers in GRID are not operating EyeSync, so every click, face gesture or scroll produced by the replay is a false positive. Natural short closures can provide weak positive examples of real blinks; long apparent closures are only **candidates** for downcast/open-eye negatives. Any learned rule must also pass the existing deliberate-blink tests and a separate real-appearance hold-open replay before it can ship.

## Other datasets considered

- **Eyeblink8** (https://www.blinkingmatters.com/research): useful labelled natural video, but GPL licensing and a small four-person corpus make it a better independent comparison than the first training source.
- **RT-BENE** (https://zenodo.org/records/3685316): more than 200,000 labelled eye patches, but the patches lack the full-face/head-pose inputs used here and the dataset/code are CC BY-NC-SA 4.0. It is not bundled or used by the current training pipeline.

No dataset can prove that an accessibility controller is perfect for every person, camera, posture or lighting condition. The goal is measurable improvement on unseen people without sacrificing deliberate gestures.
