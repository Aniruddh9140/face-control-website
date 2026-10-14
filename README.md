# face-control-website

**EyeSync**, the control dashboard for the face control eye-tracker project. It is a single static page (`index.html`) with inline CSS and an inline JavaScript module, built to run on GitHub Pages with no build step and no server.

## What it does

- **Home page.** Visitors land on an introduction (`#home`) covering what EyeSync does, a simulated gaze preview, how to get started, and the privacy and accessibility approach. Buttons there lead to sign-in and registration.
- **Accounts in the browser.** Registration and sign-in use the Web Crypto API. Passwords are salted (16 random bytes) and hashed with PBKDF2-HMAC-SHA-256 at 600,000 rounds before anything is saved. Plain-text passwords are never stored.
- **Session lock.** A successful sign-in creates a random 32-byte session token, kept in `sessionStorage` for this tab only. The account record stores just the SHA-256 of that token. Protected views (`#dashboard`, `#calibration`) check the token on every navigation, and the session expires after 30 minutes without activity.
- **Brute-force slowdown.** After 5 failed sign-ins, that username is locked for 30 s, doubling on each further failure up to 15 minutes.
- **Cookie consent.** A banner appears on the first visit. Optional preference storage (`fc.prefs`) stays in memory until the visitor chooses **Accept**. Choosing **Essential only** deletes it. The Cookie Policy and Terms and Conditions open as pages (`#cookies`, `#terms`) or as dialogs from the registration form.
- **Mandatory agreement.** The registration form has an unchecked "I agree to the Terms and Conditions and Cookie Policy" checkbox. No account is created unless it is checked, and the versions agreed to are recorded with the account.
- **Live gaze.** A WebSocket client connects to `ws://localhost:8765` (configurable, loopback addresses only). It validates every frame, reconnects with exponential backoff, and flags a stale stream. The red crosshair is positioned with CSS `top`/`left` from each `{"x", "y", "blink"}` frame.
- **9-point calibration.** Targets in the corners, edges, and centre light up one at a time. While each target is showing, the dashboard measures the gaze error at that target and sends calibration events back to the tracker.
- **Assistive extras.** Every control is at least 56 px with solid colour-inversion hover states. Optional dwell-to-select lets you press any button by holding your gaze on it. Light and dark themes are both included, and reduced-motion preferences are respected.

## Run it locally

```bash
python -m http.server 8000
```

Then open http://localhost:8000. You can also open `index.html` directly from disk.

## Test without hardware

Click **Run simulator** on the Dashboard, or run the mock tracker, which uses the real WebSocket path:

```bash
pip install websockets
python tools/mock_tracker.py
```

The mock streams 60 frames per second on `ws://localhost:8765` and prints the calibration events it receives. It only accepts pages from the GitHub Pages site and `http://localhost:8000`. Use `--origin` to add others, or `--allow-file` for a page opened from disk.

## Tracker protocol

**Tracker → dashboard**, one JSON text frame per sample:

```json
{"x": 400, "y": 300, "blink": false}
```

`x` and `y` are numbers. By default they are pixels in an 800×600 tracker frame, which is scaled to the browser window; set the frame size on the Dashboard to match your tracker, or switch to "window pixels". `blink` is optional and defaults to `false`. The dashboard rejects frames over 4 KB, frames that aren't JSON, and frames without numeric `x`/`y`.

**Dashboard → tracker**, sent during calibration:

```json
{"type": "calibration", "event": "start", "points": 9, "viewport": {"w": 1280, "h": 800}, "frame": {"w": 800, "h": 600}}
{"type": "calibration", "event": "point", "index": 0, "label": "Top left", "viewport": {"x": 60, "y": 60}, "source": {"x": 38, "y": 45}, "normalized": {"x": 0.0469, "y": 0.075}}
{"type": "calibration", "event": "point_done", "index": 0, "samples": 50, "mean_error_px": 23}
{"type": "calibration", "event": "complete", "mean_error_px": 18, "points": [{"i": 0, "e": 23, "n": 50}]}
{"type": "calibration", "event": "cancel"}
```

`source` is the target position in tracker-frame coordinates. Your tracker can ignore these messages if it doesn't calibrate.

### Browser notes

- GitHub Pages is served over HTTPS. Chrome, Edge, and Firefox allow an HTTPS page to open `ws://localhost`; Safari blocks it, so use `wss://` or serve the page locally there.
- Newer Chrome versions may ask for permission to access devices on your local network the first time the page connects. Allow it.
- To accept a tracker on another machine, add its host to both `LOOPBACK_HOSTS` in the script and `connect-src` in the Content-Security-Policy meta tag.

## Security model

Everything runs in the browser, so this is local-first protection, not server-grade authentication. Anyone with access to the browser profile can read or clear `localStorage`, which also resets the sign-in throttle. Within that limit, the page:

- stores only salted PBKDF2 hashes, compares them in constant time, and hashes unknown usernames too, so timing doesn't reveal which accounts exist;
- builds all DOM with `textContent` and element APIs (no `innerHTML`), and enforces Trusted Types in Chromium browsers;
- sets a strict Content-Security-Policy: no external hosts, and WebSockets to loopback only;
- refuses to run sign-in when framed by another site (clickjacking guard), since GitHub Pages can't send `frame-ancestors` headers.

## Legal text

The Cookie Policy and Terms and Conditions in `index.html` (`#tpl-cookies`, `#tpl-terms`) are starting templates. Have them reviewed before relying on them, and bump the version constants (`TERMS_VERSION`, `COOKIE_POLICY_VERSION`) when they change.
