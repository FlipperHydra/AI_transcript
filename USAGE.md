# Audio Notes — Usage Guide

---

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running (whale icon in system tray)
- [Git](https://git-scm.com/download/win) installed
- A HuggingFace account with a **Write** token — [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- You must have accepted the license for both of these models on HuggingFace while logged in:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
  *(Without this, diarization is silently skipped — transcription still works.)*

---

## Part 1 — First-Time Setup

**Step 1 — Clone the repo**
```powershell
git clone https://github.com/FlipperHydra/AI_transcript.git
cd AI_transcript
```

**Step 2 — Create your `.env` file**
```powershell
copy .env.example .env
notepad .env
```
Replace `hf_your_token_here` with your real HuggingFace token. Save and close.

**Step 3 — Create the output folder**
```powershell
mkdir output
```

**Step 4 — Build the app image**
```powershell
docker compose build
```
This installs all Python packages including WhisperX, torch, and sounddevice.
Expect 10–20 minutes on first run.

**Step 5 — Start everything**
```powershell
docker compose up
```
On first run, ollama will pull phi3:mini (~2.3 GB). This happens once — the model
is saved to a Docker volume and reused on every subsequent start.

The app will not start until ollama passes its health check, so you won't see
"Models ready" until phi3:mini is fully downloaded and ollama is serving.

**Step 6 — Open the app**

Navigate to: **http://localhost:8000**

Wait for the green **"Models ready"** banner before recording.

---

## Part 2 — Every Day Use

**Starting the app (after first setup)**
```powershell
cd AI_transcript
docker compose up
```
Open **http://localhost:8000** — ollama starts in seconds since phi3:mini is already cached.

**Stopping the app**
```powershell
docker compose down
```
Your recordings and notes in `output/` are preserved.

---

## Part 3 — Recording a Session

**Step 1 — Select your microphone**
In the top bar, use the microphone dropdown to pick your input device.
Leave it on "Default" if you only have one mic.

**Step 2 — Select processing device**
Click **CPU**, **CUDA**, or **MPS** depending on your hardware.
- Most users: **CPU**
- NVIDIA GPU: **CUDA** (requires GPU build — see Part 5)
- Apple Silicon: **MPS**

**Step 3 — Hit Record**
Click the **● Record** button. The status dot turns red and the timer starts.
Speak clearly — the volume meter confirms audio is being captured.

**Step 4 — Hit Stop**
Click **■ Stop** when finished. Processing begins immediately.

---

## Part 4 — Processing Pipeline (What Happens After Stop)

You will see the status bar progress through these stages automatically:

| Stage | What's happening |
|---|---|
| **Transcribing** | WhisperX converts your audio to text with word-level timestamps |
| **Chunking** | Transcript is split into token-sized pieces for phi3:mini |
| **Summarizing (N of M)** | phi3:mini generates structured notes for each chunk |
| **Compiling** | phi3:mini assembles all chunks into a single master document |
| **Done** | Notes and transcript appear in the panels |

Processing time depends on recording length and your hardware. A 5-minute
recording typically takes 1–3 minutes to process on CPU.

---

## Part 5 — Viewing & Exporting Results

**Transcript panel (right)**
Shows every speaker segment with timestamps in `[MM:SS] SPEAKER: text` format.
Click **⬇ Export .txt** to download the raw transcript.

**Notes panel (left)**
Shows the compiled markdown document: Overview, Action Items, Speaker Breakdown,
and Full Transcript Log.
Click **⬇ Export .md** to download the notes file.

**History sidebar**
All past recordings are listed in the left sidebar. Click any entry to reload
its transcript and notes. Export buttons work for past recordings too.

---

## Part 6 — GPU Build (NVIDIA only)

To use your GPU for faster WhisperX processing:

**Step 1 — Edit `docker-compose.yml`**
Change `GPU: 0` to `GPU: 1` under the `app` build args.

**Step 2 — Rebuild**
```powershell
docker compose build --no-cache app
docker compose up
```

**Step 3 — Select CUDA in the UI**
Click the **CUDA** button in the device selector. Restart the container
if models were already loaded on CPU.

---

## Part 7 — Updating to a New Version

```powershell
cd AI_transcript
docker compose down
git pull origin main
docker compose build --no-cache
docker compose up
```

Your `output/` folder and the `ollama_data` volume (containing phi3:mini)
are preserved across updates. Only the app image is rebuilt.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Page won't load | Make sure Docker Desktop is running. Check `docker compose ps` — both services should be `Up`. |
| "Models loading" never changes to "ready" | WhisperX is still downloading alignment models on first run (~360MB). Wait 2–3 minutes. |
| Transcript is empty after processing | Check that your mic is selected and the volume meter moved during recording. |
| Notes say "Could not reach ollama" | ollama container may still be pulling phi3:mini. Wait for the pull to finish and try again. |
| No speaker labels (all show UNKNOWN) | HF_TOKEN is missing or the pyannote licenses haven't been accepted on HuggingFace. |
| Build fails on pip install | Run `docker compose build --no-cache` to force a clean install. |
