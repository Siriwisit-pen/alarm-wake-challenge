# Alarm Wake Challenge

An offline-first PWA prototype for an alarm that keeps ringing until the user
finishes a short exercise challenge.

## Run Locally

```powershell
npm.cmd run start
```

Open:

```text
http://127.0.0.1:5173
```

## What Works

- Installable PWA metadata for Add to Home Screen.
- Service worker cache for offline use after the first load.
- Local MediaPipe runtime, WASM files, and pose model.
- A challenge builder with 1-4 exercises.
- Squat, jumping jack, push-up, and plank detection.
- Alarm-style ringing flow that clears only after the full challenge completes.

## Notes

Web/PWA alarms are best-effort when the browser or operating system fully kills
the app. For the strongest real-world alarm behavior, the future Android version
should use native alarm APIs.
