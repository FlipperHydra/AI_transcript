# Audio Notes System

Record system audio → transcribe with WhisperX → summarize with phi3:mini → save structured Markdown notes.

## Stack
| Layer | Technology |
|---|---|
| Capture | `soundcard` (loopback) + `soundfile` |
| Export | `pydub` → MP3 (atomic rename) |
| Transcription | WhisperX `medium` + alignment + pyannote diarization |
| Summarization | `ollama` / `phi3:mini` (rolling context) |
| Backend | FastAPI + asyncio pipeline queue |
| UI | Jinja2 + vanilla JS (WebSocket live updates) |
| Storage | SQLite (`notes.db`) + `.md` files on disk |
| Infra | Docker + docker-compose (ollama sidecar) |

## First-time setup

1. **HuggingFace account** — [huggingface.co](https://huggingface.co) (free)
2. **Accept pyannote license** — [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
3. **Generate a read token** — HuggingFace → Settings → Access Tokens → New Token (role: read)
4. **Create `.env`**:
   ```
   HF_TOKEN=hf_your_token_here
   ```
5. **Build and run**:
   ```bash
   docker compose up --build
   ```
6. Open [http://localhost:8000](http://localhost:8000)

## Usage

1. Set the **Output Path** field if you want files saved somewhere other than `./output/`
2. Click **● Record** — system loopback audio captures immediately
3. Click **■ Stop** — the pipeline runs automatically:
   - Transcribing → Dividing → Summarizing
4. The **Transcript Panel** fills as soon as WhisperX finishes
5. The **Notes Panel** fills after phi3:mini compiles the master notes

## File layout

```
project/
├── Dockerfile
├── docker-compose.yml
├── .env                     ← HF_TOKEN (never commit)
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── app/
│   ├── main.py              # FastAPI: routes, WebSocket, pipeline task
│   ├── recorder.py          # soundcard loopback + pydub export
│   ├── transcriber.py       # WhisperX 3-step pipeline
│   ├── notes_gen.py         # chunking, rolling context, ollama
│   ├── db.py                # SQLite helpers
│   ├── config.json          # persisted output path (gitignored)
│   └── templates/
│       └── index.html       # UI
└── output/                  # MP3s + .md notes (bind-mounted)
```

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Docker healthcheck |
| `POST` | `/start-recording` | Begin loopback capture |
| `POST` | `/stop-recording` | Stop capture, trigger pipeline |
| `POST` | `/set-output-path` | `{"path": "/some/dir"}` |
| `GET` | `/transcript/{id}` | Raw WhisperX JSON segments |
| `GET` | `/notes` | All completed notes |
| `GET` | `/notes/{id}` | Single note detail |
| `WS` | `/ws` | Live status events |

## WebSocket events

| Event field | Value | Meaning |
|---|---|---|
| `status` | `recording` | Capture started |
| `status` | `stopped` | Capture stopped |
| `status` | `transcribing` | WhisperX running |
| `status` | `transcript_ready` + `id` | Transcript done; UI fetches it |
| `status` | `chunking` | Splitting transcript |
| `status` | `summarizing` + `chunk` + `of` | phi3:mini chunk N of M |
| `status` | `compiling` | Final compile pass |
| `event` | `note_ready` + `id` | Notes done; UI fetches them |
| `status` | `error` + `detail` | Pipeline error |

## GPU acceleration

To use a GPU, change `DEVICE` and `COMPUTE_TYPE` in `transcriber.py`:
```python
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
```
And install the CUDA-enabled torch wheel before building.
