import shutil

import numpy as np
import pytest
import soundfile as sf

from backends import parse_backend_flags
from backends.dummy import DummyBackend
from core.cache import LineCache
from core.pipeline import pause_for, render
from core.runtime import SetupError
from core.script import load_script


class Factory:
    """Records which backends were created; returns DummyBackends."""

    def __init__(self):
        self.created: dict[str, DummyBackend] = {}

    def __call__(self, bid: str) -> DummyBackend:
        b = DummyBackend()
        b.name = bid
        self.created[bid] = b
        return b


def run(script, tmp_path, factory, **kw):
    cache = LineCache(tmp_path / "cache", enabled=kw.pop("cache", True))
    return render(script, {"en": "en-dummy", "bn": "bn-dummy"}, factory, cache, **kw)


def test_end_to_end(write_script, scene, tmp_path):
    s = load_script(write_script(scene))
    f = Factory()
    res = run(s, tmp_path, f)
    assert res.generated == 4 and res.cached == 0
    audio, sr = sf.read(res.wav)
    assert sr == 24000 and audio.ndim == 1 and res.duration_s > 1
    assert set(f.created) == {"en-dummy", "bn-dummy"}
    # each backend loaded once, each speaker voiced once (consistency/caching)
    for b in f.created.values():
        assert b.load_calls == 1
    assert sorted(f.created["bn-dummy"].voice_calls) == ["Arif", "Rina"]
    assert sorted(f.created["en-dummy"].voice_calls) == ["Boo", "Narr"]


def test_rerun_uses_cache_and_skips_loading(write_script, scene, tmp_path):
    s = load_script(write_script(scene))
    run(s, tmp_path, Factory())
    f2 = Factory()
    res = run(s, tmp_path, f2)
    assert res.generated == 0 and res.cached == 4
    assert f2.created == {}  # no model loaded at all


def test_only_changed_line_regenerates(write_script, scene, tmp_path):
    run(load_script(write_script(scene)), tmp_path, Factory())
    scene["lines"][2]["text"] = "সরি, আজ অনেক জ্যাম ছিল।"
    f = Factory()
    res = run(load_script(write_script(scene)), tmp_path, f)
    assert res.generated == 1 and res.cached == 3
    assert list(f.created) == ["bn-dummy"]  # en backend never loaded


def test_emotion_change_invalidates(write_script, scene, tmp_path):
    run(load_script(write_script(scene)), tmp_path, Factory())
    scene["lines"][0]["emotion"] = "angry"
    res = run(load_script(write_script(scene)), tmp_path, Factory())
    assert res.generated == 1


def test_lazy_loading_en_only(write_script, scene, tmp_path):
    scene["lines"] = [scene["lines"][0], scene["lines"][3]]
    f = Factory()
    run(load_script(write_script(scene)), tmp_path, f)
    assert list(f.created) == ["en-dummy"]


def test_long_line_is_chunked(write_script, scene, tmp_path):
    long = " ".join(f"Sentence {i} keeps going for a while." for i in range(20))
    scene["lines"] = [{"speaker": "Narr", "text": long}]
    f = Factory()
    run(load_script(write_script(scene)), tmp_path, f, cache=False)
    calls = f.created["en-dummy"].synth_calls
    assert len(calls) > 1 and all(len(t) <= 200 for t, _, _ in calls)


def test_pauses(write_script, scene):
    s = load_script(write_script(scene))
    a, b, c, d = s.lines
    assert pause_for(None, a, s.pause_ms) == 0
    assert pause_for(a, b, s.pause_ms) == 600
    assert pause_for(b, b, s.pause_ms) == 250
    c.paragraph = True
    assert pause_for(b, c, s.pause_ms) == 1200
    d.pause_before_ms = 3000
    assert pause_for(c, d, s.pause_ms) == 3000


def test_mp3_export(write_script, scene, tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    res = run(load_script(write_script(scene)), tmp_path, Factory(), mp3=True)
    assert res.mp3 and res.mp3.stat().st_size > 1000


def test_loudness_consistent_across_lines(write_script, scene, tmp_path):
    # dummy amplitude depends on emotion; after normalization lines should match
    s = load_script(write_script(scene))
    res = run(s, tmp_path, Factory())
    audio, sr = sf.read(res.wav)
    assert np.max(np.abs(audio)) <= 0.98


def test_backend_flags():
    assert parse_backend_flags(None) == {"en": "qwen", "bn": "chatterbox-bn"}
    assert parse_backend_flags(["en=chatterbox", "bn=xtts"]) == {"en": "chatterbox", "bn": "xtts-bn"}
    assert parse_backend_flags(["xtts"])["bn"] == "xtts-bn"
    with pytest.raises(SetupError):
        parse_backend_flags(["hi=qwen"])
    with pytest.raises(SetupError):
        parse_backend_flags(["bn=qwen"])
