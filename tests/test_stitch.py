import numpy as np
import soundfile as sf

from core.stitch import (
    Segment, assemble, normalize_loudness, prepare_clip, resample, rms_db, to_mono_f32, trim_silence, write_wav,
)

SR = 24000


def tone(sec, amp=0.3, sr=SR, f=220):
    t = np.arange(int(sr * sec)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_resample_length():
    a = tone(1.0, sr=22050)
    b = resample(a, 22050, SR)
    assert abs(b.size - SR) <= 1
    assert resample(a, 22050, 22050) is a or np.array_equal(resample(a, 22050, 22050), a)


def test_to_mono():
    st = np.stack([tone(0.5), tone(0.5)], axis=1)  # (n, 2)
    assert to_mono_f32(st).shape == (SR // 2,)
    assert to_mono_f32(st.T).shape == (SR // 2,)  # (2, n)


def test_trim_silence():
    a = np.concatenate([np.zeros(SR), tone(0.5), np.zeros(SR)])
    t = trim_silence(a, SR, keep_ms=40)
    assert 0.5 <= t.size / SR <= 0.6


def test_loudness_equalizes():
    quiet, loud = normalize_loudness(tone(2, 0.02), SR), normalize_loudness(tone(2, 0.6), SR)
    assert abs(rms_db(quiet) - rms_db(loud)) < 0.5


def test_loudness_short_clip_rms_fallback():
    out = normalize_loudness(tone(0.2, 0.01), SR)
    assert rms_db(out) > rms_db(tone(0.2, 0.01)) + 10


def test_loudness_silence_safe():
    z = np.zeros(SR, np.float32)
    assert np.array_equal(normalize_loudness(z, SR), z)


def test_assemble_pauses():
    segs = [Segment(tone(1), 0), Segment(tone(1), 600), Segment(tone(1), 250)]
    out = assemble(segs, SR, lead_ms=0, tail_ms=0)
    assert out.size == SR * 3 + int(SR * 0.85)
    assert np.max(np.abs(out)) <= 0.97 + 1e-6


def test_prepare_clip_and_write(tmp_path):
    raw = np.concatenate([np.zeros(5000), tone(1, sr=22050), np.zeros(5000)])
    clip = prepare_clip(raw, 22050, SR)
    p = write_wav(tmp_path / "x.wav", clip, SR)
    info = sf.info(p)
    assert info.samplerate == SR and info.channels == 1 and info.subtype == "PCM_16"
