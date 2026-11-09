# Local dialogue TTS — English + Bengali

Turns a scripted dialogue (`script.json`) into one 24 kHz mono WAV (+ optional MP3), with
emotion, conversational pauses, multiple speakers, invented cartoon voices (English) and
voice cloning from a short clip. Everything runs locally; the network is only used once,
by `--setup`. Only `en` and `bn` are supported.

| Language | Primary backend | Fallback(s) |
|---|---|---|
| English | **Qwen3-TTS** (`qwen`): instruction-driven VoiceDesign + cloning | **Chatterbox** (`chatterbox`): `exaggeration` + inline tags |
| Bengali | **Chatterbox-Bangla** (`chatterbox-bn`, `jongy5/chatterbox-bangla`) | **XTTS v2 Bangla** (`xtts-bn`), gated and non-commercial, see Licenses |

## Install (Windows 11, CPU only — this machine: i5-8265U, 16 GB RAM, no NVIDIA GPU)

Needs Python **3.11** (3.10–3.12; PyTorch 2.6 has no 3.14 wheels), ffmpeg on PATH (MP3 export and
`make_ref.py`), about 15 GB of disk for models and about 6 GB for venvs.

```powershell
cd D:\audio\tts
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt   # light core deps
.\.venv\Scripts\python tts.py --setup                       # venvs + model downloads (one time, needs internet)
.\.venv\Scripts\python tts.py --check
```

`--setup` with no flags installs the two primary backends. Add fallbacks explicitly:

```powershell
.\.venv\Scripts\python tts.py --setup --backend en=chatterbox
.\.venv\Scripts\python tts.py --setup --backend bn=xtts      # gated repo, see below
```

**Why several venvs?** The model libraries have incompatible hard pins:
`qwen-tts 0.1.1` requires `transformers==4.57.3`, while `chatterbox-tts 0.1.7` requires
`transformers==5.2.0`, `torch==2.6.0` and `numpy<2`. So `--setup` creates `.venvs/qwen`,
`.venvs/chatterbox` and `.venvs/xtts`. The CLI runs each backend as a worker subprocess in its
venv, loads it only when a line needs it, and shuts it down before the next backend starts
(so two models are never in RAM at once). Direct pins are in `requirements/<env>.txt`, and torch is
pinned to 2.6.0 everywhere. The exact package sets that were tested are frozen in
`requirements/lock-<env>.txt` (`pip install -r requirements/lock-qwen.txt` inside a venv reproduces one).

Windows notes:
- Models are downloaded into plain folders (`models/repos/…`), because the Hugging Face cache
  needs symlinks, which Windows refuses without Developer Mode.
- `coqui-tts 0.28.0` imports `six` without declaring it, so `requirements/xtts.txt` adds it.

**With an NVIDIA GPU (Windows or Linux):** the same commands work. `--setup` detects `nvidia-smi` and
installs the CUDA 12.4 torch wheels, and inference picks CUDA automatically (`--device cuda` forces it
and fails clearly if CUDA isn't usable). On GPU, Qwen uses the 1.7B Base model in bf16 by default.
On Linux CPU, `--setup` uses the CPU torch index.

Models are stored in `tts/models/` (override with `TTS_MODELS_DIR`). At inference time
`HF_HUB_OFFLINE=1` is set, so a missing model fails with "run --setup", never a download.

## Usage

```powershell
python tts.py examples\scene1.json                  # -> out\scene1.wav
python tts.py examples\scene1.json --mp3            # + out\scene1.mp3
python tts.py script.json --backend en=chatterbox --backend bn=xtts
python tts.py script.json --no-cache                # regenerate everything
python tts.py script.json -o final.mp3
```

(`python` here means `.\.venv\Scripts\python`.)

### script.json

```json
{
  "output": "scene1.wav",
  "pause_ms": { "same_speaker": 250, "turn": 600, "paragraph": 1200 },
  "speakers": {
    "Rina": { "lang": "bn", "ref_wav": "voices/rina.wav" },
    "Narr": { "lang": "en", "ref_wav": "voices/narrator.wav", "ref_text": "exact words in narrator.wav" },
    "Boo":  { "lang": "en", "design": "small squeaky cartoon ghost, playful, high pitch, fast" }
  },
  "lines": [
    { "speaker": "Narr", "emotion": "calm", "text": "It was raining in Dhaka." },
    { "speaker": "Rina", "emotion": "angry", "text": "তুমি আবার দেরি করলে?" },
    { "speaker": "Boo", "emotion": "happy", "paragraph": true, "text": "[laugh] Nobody ever blames the ghost!" }
  ],
  "backend_options": { "qwen": { "base_model": "1.7b" } }
}
```

**Speakers.** Each speaker has either `ref_wav` (clone) or `design` (invented voice, English/Qwen only).
A Bengali speaker with `design` is rejected. Omit `lang` to detect it per line: text that is mostly
Bengali script (U+0980–U+09FF) is routed to `bn`, otherwise `en`. A line is never split across models.
`ref_text` is the transcript of `ref_wav`. Qwen clones much better with it. A `voices/x.txt` next to
`voices/x.wav` is picked up automatically.

**Line fields.** `speaker`, `text`, `emotion` (default `neutral`). Optional:
- `lang`: override detection.
- `paragraph: true`: use the paragraph pause before this line.
- `pause_before_ms`: explicit pause.
- `exaggeration` (0–2): override the emotion's Chatterbox value.
- `instruct`: extra English direction appended to the emotion instruction.

**Emotions.**

| Label | `exaggeration` (bn / Chatterbox) | Qwen instruction (abridged) |
|---|---|---|
| calm | 0.30 | "calmly and evenly, unhurried" |
| neutral | 0.50 | "naturally, relaxed" |
| happy | 0.70 | "happily and warmly" |
| angry | 0.90 | "angrily, tense, slightly rushed" |
| sad | 0.40 | "sadly, quietly and slowly" |

About 50 modes in total, from `calm` (0.30) to `sobbing`/`shouting` (1.30), plus aliases such as
"anger", "teary" and "heartbreak". Run `python tts.py --emotions` for the full list with values. Any other word (e.g. `"pirate-like"`) becomes a free-text instruction for Qwen and
neutral for Bengali. Expressive values also lower Chatterbox's `cfg_weight` (0.3–0.5), as Resemble recommends.

### How emotion actually reaches each model (read this)

- **Qwen3-TTS** only accepts written directions in its **VoiceDesign** model. The **Base** (clone) model
  has no `instruct` parameter. This tool therefore uses Qwen's documented *design → clone* workflow:
  1. For each designed speaker × emotion, VoiceDesign renders a reference clip from
     `"<design>. <emotion instruction>"` with a fixed seed. The clip is cached in `cache/voices/`.
  2. The Base model clones from that clip for every line.

  The timbre stays consistent, and the delivery follows the emotion.
  - **Cloned (`ref_wav`) English speakers:** emotion labels do not change Qwen's delivery. It comes
    from the reference clip and the wording. Use `--backend en=chatterbox` if you need an emotion
    slider for cloned voices.
  - **Inline tags** (`[laugh]`) are removed for Qwen, with a warning.
- **Chatterbox (en)** uses the `exaggeration` slider.
  - Tags are mapped to the tokenizer's `[laughter] [giggle] [sigh] [gasp] [whisper] [cough] [clear_throat]`.
    Resemble does not document these for the original model, so treat them as best-effort.
  - `"backend_options": {"chatterbox": {"variant": "turbo"}}` switches to Chatterbox-Turbo, which officially
    supports `[laugh] [sigh] [whispering] [gasp] [chuckle] [crying]…` but ignores `exaggeration` and needs a
    reference longer than 5 s. Run `python tts.py script.json --setup --backend en=chatterbox` after adding the option so Turbo's
    weights are downloaded.
- **Bengali** has no instruction-following model. Emotion = the reference clip's style + `exaggeration`.
  For strongly different moods, record a reference in that mood next to the main clip, named
  `<clip>_<mode>.wav` (e.g. `voices/rina_crying.wav` beside `voices/rina.wav`). Lines with that emotion
  (or an alias of it) then clone from the mood clip automatically; other lines keep the main clip.
  A `rina_crying.txt` transcript is picked up too. This works for English cloned speakers as well.

### Pipeline details

- One line per generation call. Lines longer than `--max-chars` (200) are split at `। ? ! .`, then at
  clause punctuation, then at spaces, and the pieces are joined with a 120 ms gap.
- Text normalization runs before synthesis:
  - `text/bn_norm.py`: Bangla numbers → words with হাজার/লাখ/কোটি grouping, years like ১৯৭১ → উনিশশো
    একাত্তর, decimals, ৳/% → words, NFC plus nukta and vowel-sign repair, `|` → `।`.
  - `text/en_norm.py`: numbers, years, money, ordinals, common abbreviations.
- Speaker voices (embeddings, conditionals, clone prompts) are computed once per speaker per run.
- Every line is trimmed, normalized to −20 LUFS (pyloudnorm, with an RMS fallback for clips under 0.4 s)
  and peak-limited, then joined with pauses: `same_speaker`, `turn` or `paragraph`.
- **Cache.** `cache/<sha1>.wav` per line. The key covers backend, voice fingerprint (ref clip hash or
  design text), speaker, emotion and text. Re-runs regenerate only lines that changed, and if every line
  is cached no model is loaded at all.

## Making reference clips

```powershell
python scripts\make_ref.py any_audio.mp3 voices\rina.wav
python scripts\make_ref.py interview.mp4 voices\narr.wav --start 30 --max 12 --text "Exact words spoken."
```

What it does:
1. Decodes any format with ffmpeg and converts to mono 24 kHz.
2. Applies a 70 Hz high-pass filter.
3. Trims silence and shortens long pauses.
4. Picks the most speech-dense 6–12 s window, ending on a pause.
5. Normalizes to −20 LUFS and writes 16-bit WAV.

For good clones, use clean speech from one speaker, with no music or reverb.
`--text` must match what remains in the final clip. Check the printed duration.

## Performance on this laptop (CPU)

Expect generation to be several times slower than real time. Qwen VoiceDesign (1.7B, fp32) needs about
7 GB of RAM and runs only once per designed speaker × emotion, because its output is cached. The English
default on CPU is the 0.6B Base model. Set `"backend_options": {"qwen": {"base_model": "1.7b"}}` for
higher quality at about 3× the time (that model is downloaded by `--setup` only on GPU machines; run
`python tts.py script.json --setup --skip-install --backend en=qwen` after setting the option). `threads` (default: all cores),
`seed`, `temperature` and `top_p` can also be set per backend in `backend_options`.

## Licenses

| Component | License | Notes |
|---|---|---|
| Qwen3-TTS (`qwen-tts`, `Qwen/Qwen3-TTS-12Hz-*`) | Apache-2.0 | permissive |
| Chatterbox (`chatterbox-tts`, `ResembleAI/chatterbox`, `-turbo`) | MIT | outputs carry Resemble's imperceptible **Perth watermark**. The repo is archived (no longer maintained). |
| Chatterbox-Bangla (`jongy5/chatterbox-bangla`) | MIT | trained on AI4Bharat Shrutilipi/Rasa and SPRINGLab IndicTTS-Bengali. ⚠ Dataset terms were **not reviewed**; check IndicTTS's terms before commercial use. |
| XTTS v2 Bangla (`EMTIAZZ/xtts-bangla-finetuned`) | card says "other" | ⚠ **Gated** repo. Derived from XTTS-v2, which is under the **Coqui Public Model License: non-commercial only**. Treat as non-commercial. |
| Coqui TTS library (`coqui-tts`, idiap fork) | MPL-2.0 | |

To use the XTTS fallback:
1. Accept the terms at https://huggingface.co/EMTIAZZ/xtts-bangla-finetuned.
2. Log in with `.venvs\xtts\Scripts\hf auth login`, or set `HF_TOKEN`.
3. Run `python tts.py --setup --backend bn=xtts`.

As the brief requires, it is called with `language="en"` even for Bangla text, because the checkpoint
was trained that way (override with `backend_options.xtts.language`). If the default `model.pth`
doesn't load, set `backend_options.xtts.checkpoint` to `best_model_496577.pth`; that file is also in the
repo, but `--setup` downloads only `model.pth`, so run `python tts.py script.json --setup --skip-install --backend bn=xtts`
after changing the option.

**Not implemented:** the Orpheus Bangla "no-GPU" fallback. `asif00/orpheus-bangla-tts-gguf-4bit` does not exist.
The related `asif00/orpheus-bangla-tts*` repos are proof-of-concept models (955 training samples) with
no usage instructions.

## Development

```powershell
.\.venv\Scripts\python -m pytest -q      # unit tests, dummy backend, no models needed
$env:TTS_DEBUG=1                          # full tracebacks (incl. from workers)
$env:TTS_IN_PROCESS=1                     # run backends in the current venv instead of workers
```

Layout:

| Path | Contents |
|---|---|
| `core/script.py` | load and validate the script, language detection, chunking |
| `core/stitch.py` | resample, trim, loudness, pauses, WAV/MP3 export |
| `core/pipeline.py` | cache → lazy backends → stitch |
| `core/emotion.py`, `core/cache.py`, `core/runtime.py` | emotion mapping, line cache, paths/offline/device |
| `backends/base.py` | the `TTSBackend` interface: `load()`, `prepare_voices()`, `synth(text, voice, emotion) -> np.ndarray` |
| `backends/worker.py`, `backends/remote.py` | per-venv subprocess workers |
| `backends/en_qwen.py`, `en_chatterbox.py`, `bn_chatterbox.py`, `bn_xtts.py` | model backends |
| `text/` | normalizers and tag handling |
| `scripts/make_ref.py` | reference-clip maker |
