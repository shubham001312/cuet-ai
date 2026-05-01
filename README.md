---
title: cuet_AI
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
fullWidth: true
header: mini
short_description: Public CUET UG assistant with official-link-first answers and shared-key guardrails.
---

# cuet_AI

`cuet_AI` is a browser version of the CUET monitor from `A:\C programs\CA PROJECT 2ND SEMISTER\ai.py`. It is now packaged for public deployment on Hugging Face Spaces with a server-side Gemini key, rate limiting, and bounded context to reduce free-tier abuse.

## Local run

```powershell
cd C:\Users\shubh\cuet_AI
python app.py --open-browser
```

If you prefer to inject the key from the terminal:

```powershell
$env:GEMINI_API_KEY="AIza..."
python app.py --open-browser
```

You can also use [launch_cuet_ai.bat](/C:/Users/shubh/cuet_AI/launch_cuet_ai.bat).

## Hugging Face Spaces deployment

1. Create a new Hugging Face Space.
2. Choose `Docker` as the SDK.
3. Make the Space `Public` if you want a public URL with no paywall.
4. Upload or push the contents of `C:\Users\shubh\cuet_AI`.
5. In the Space `Settings`, add a secret named `GEMINI_API_KEY`.
6. Let the Space build. The app will serve on port `7860`.

## Recommended runtime settings

Optional environment variables:

- `GEMINI_MODEL=gemini-2.5-flash-lite`
- `CUET_AI_RATE_LIMIT_MAX_REQUESTS=20`
- `CUET_AI_RATE_LIMIT_WINDOW_SECONDS=3600`
- `CUET_AI_CONTEXT_MESSAGES=8`
- `CUET_AI_MAX_OUTPUT_TOKENS=1000`

## Important constraints

- A free public Space is not the same as "free forever". Hosting plans and Gemini quotas can change.
- On Hugging Face free hardware, unused Spaces go to sleep.
- On free public Hugging Face Spaces, the source code is visible to everyone.
- Keep the Gemini key in a Space secret, not in the repository.
- Public users can consume your shared Gemini quota. This app now rate-limits shared-key requests, and users can paste their own key in the UI if needed.

## Notes

- No third-party Python packages are required.
- Browser-saved keys stay in local storage.
- Server-side keys are read from `GEMINI_API_KEY` first, then from the local config file at `~/.cuet_ai/config.json`.
- Default model selection prefers `gemini-2.5-flash-lite`, then falls back to `gemini-2.5-flash`, `gemini-2.0-flash`, and `gemini-flash-latest`.
