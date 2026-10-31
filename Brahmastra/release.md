# EyeSync 2.0.3 for Windows x64

EyeSync combines local head-controlled pointer movement, blink/wink clicks,
head-tilt scrolling, facial shortcuts, offline voice typing, and the on-screen
keyboard command panel. It connects to the existing [EyeSync dashboard](https://aniruddh9140.github.io/face-control-website/#dashboard).

## Calibration result fix

- Failed calibration shows its validation error instead of being reported as cancelled.
- Late cancellation cannot overwrite a completed calibration.
- Pointer control resumes automatically only after successful training.

## Local stability update

- Windows mouse-input failures pause pointer control while keeping the camera worker alive.
- Invalid eye measurements and interrupted frame streams cancel pending blink clicks.
- Blink timing and personal calibration remain unchanged.

## What users download

**`EyeSync.exe` is the single application/installer file.** Its Python 3.15 runtime,
MediaPipe face model, Whisper English model, Silero speech detector, and the public
head-pose model are bundled. Users do not need Python or the original project folders.
Personal calibration, logs and registration backups are created locally and are
excluded from the executable. The app is named EyeSync; Brahmastra is the source folder.

Open the downloaded executable and complete its installation. Then open the dashboard
and click **Connect**. Camera and microphone processing runs on the user's PC.
The web page receives control/status updates over a local WebSocket connection;
GitHub Pages does not run or stream the Python AI. The microphone starts only when
voice typing or command listening is requested.

## Release assets

| File | Purpose |
|---|---|
| `EyeSync.exe` | Stable asset name used by the website's Download button |
| `latest.json` | Version, executable filename, exact size, SHA-256 and date |
| `SHA256SUMS.txt` | Checksums for the produced release files |
| `release.md` | Release notes and publishing instructions |

Use the **generated copies under `release/`** after a successful build. They describe
the actual executable, including its checksum. Keep `EyeSync.exe` named exactly that
way for every version.

The current site builds its download URL as:

```
https://github.com/Aniruddh9140/face-control-website/releases/latest/download/EyeSync.exe
```

It reads metadata from its own `downloads/latest.json`. Merely uploading this Markdown
file, or attaching `latest.json` to a Release, does not update the Pages metadata file.

## Build locally

Use Windows x64 and Python 3.15 from the Brahmastra project root:

```powershell
python -m pip install --only-binary=:all: -r requirements-python315.txt -r requirements-build.txt
python training/fetch_release_models.py
python -m unittest discover -s tests -q
python build_exe.py --output release
release\EyeSync.exe --selftest release\frozen-selftest.txt
python tests/check_frozen_release.py release\EyeSync.exe
```

The fetcher verifies hashes from `models/manifest.json`. Build output and tests contain
no personal recordings or calibration. A release is ready only after the source tests,
frozen self-test and frozen dashboard lifecycle check succeed. The checksum is an
integrity check; it does not represent a code-signing certificate.

## Confirmed website setup

The existing repository was checked read-only while preparing this release:
GitHub Pages publishes the `main` branch from `/` (the repository root). Use
`downloads/latest.json` in that root for manual publishing. The supplied release
workflow discovers this configuration automatically. No GitHub repository or
live website was changed while preparing these files.

`EyeSync-source.zip` in the generated release folder contains the small source
files and workflow for adding to the existing website repository. Extract it
locally, merge its contents into the repository root while keeping the website
files, and commit/push them. Large models are fetched and hash-checked by CI.

## Automatic publishing after one-time GitHub setup

Copy this project's release/build source into the **existing website repository root**,
retaining its website `index.html`, assets, account configuration and existing Pages
workflow. Include `.github/workflows/release.yml`, the requirement files, model manifest,
model fetcher, code, tests, documentation, and `models/biwi_head_pose.npz`. Do not upload
`.venv`, `data`, downloaded datasets, personal settings, logs, `build`, or the executable
as ordinary repository source. The fetcher supplies the large speech/face model files
during CI; the final executable goes into a GitHub Release asset.

1. Enable Actions in the website repository. The workflow needs permission to create
   Releases and commit `downloads/latest.json`. Repository or organization rules may
   require granting the workflow write access or allowing the metadata commit.
2. Confirm the Pages publishing source:
   - **Deploy from a branch:** use the repository default branch and either `/` or
     `/docs`. The workflow discovers that folder, updates its `downloads/latest.json`,
     and explicitly requests a Pages build.
   - **GitHub Actions:** keep the existing website deployment workflow. It must accept
     `workflow_dispatch`. Add repository variable **`EYESYNC_PAGES_WORKFLOW`** with that
     workflow's filename, for example `pages.yml`. If that workflow publishes `/docs`,
     also set **`EYESYNC_PAGES_SOURCE`** to `/docs`; the default is `/`.
3. Keep the release version in `version.py` consistent with the tag. Commit the source,
   then create and push a matching version tag such as **`v2.0.0`**.
4. The release workflow runs Python 3.15 source tests, builds `EyeSync.exe`, runs its
   frozen self-test and frozen dashboard lifecycle check, and retains the tested files
   as an Actions artifact. If those pass, it publishes the tagged GitHub Release,
   updates the website metadata, and requests the existing Pages deployment.
5. Check both the release workflow and Pages deployment run. Once Pages finishes,
   the website's Download page shows the new version and checksum. An already installed
   older EyeSync can show the website's update notice. Downloading/installing a new
   executable remains a user action.

An ordinary source upload does not publish an executable. A `v*` tag explicitly triggers
publication. To **build without publishing**, open Actions → Build and release EyeSync →
Run workflow, leaving `publish` false. To publish manually, select `publish: true` and
provide an existing `vX.Y.Z` tag matching `version.py`; the workflow verifies it.

The workflow supports this site's current repository address. If you move to another
GitHub repository, update `RELEASE_URL` in the website's JavaScript and the app's allowed
website origin/URL to the new host. A project-path change on the same GitHub Pages host
still requires the site URL to be updated where appropriate.

GitHub notes that commits made using `GITHUB_TOKEN` do not trigger branch-based Pages
builds. That is why this workflow explicitly requests a build, or dispatches the existing
Actions deployment, after committing metadata. See [Pages publishing sources](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site),
[requesting a Pages build](https://docs.github.com/en/rest/pages/pages#request-a-github-pages-build),
and [Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Manual GitHub publishing

If you do not use Actions, create a GitHub Release with the matching version tag and
attach the generated `EyeSync.exe`, `latest.json`, `SHA256SUMS.txt`, and `release.md`.
Copy the generated `latest.json` to `downloads/latest.json` in the actual Pages source
folder, then commit it and let the existing Pages publishing process complete.
Use the generated `release/release.md` as the Release body.

For this existing site the binary belongs in `Aniruddh9140/face-control-website` Releases.
Do not commit a large executable into Pages source or replace the website with the AI
source folder. Keep the executable and metadata from the same successful build.

## Local use and verification

Click a text field before starting dictation. Use **Ctrl+Alt+Shift+F5**, the voice button,
or both eyes closed for 2.5 seconds. Stop discards unfinished speech. A text-target
change while recognition is running stops the session so old queued speech is not
sent to the new window. Tilt the left/right ear toward its shoulder to scroll up/down,
and straighten to stop. **Ctrl+Alt+E** pauses pointer input.

Recorded automated tests do not establish live webcam accuracy or speech recognition
accuracy on every microphone. English speech decoding and local OS input limitations
remain documented in `README.md` and `docs/VOICE_SYSTEM.md`.

## Dataset and third-party terms

The public head model was trained using the BIWI Kinect Head Pose dataset. Its published
terms allow non-commercial use such as university research and education. This build
does not grant broader redistribution or commercial rights. Review the dataset terms
and `THIRD_PARTY_NOTICES.md` before publishing or distributing the bundled model;
the workflow does not establish legal permission. See the [BIWI dataset source](https://huggingface.co/datasets/ETHZurich/biwi_kinect_head_pose)
and `docs/PUBLIC_TRAINING.md`. Other bundled libraries and model notices must be retained.
