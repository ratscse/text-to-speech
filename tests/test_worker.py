"""Exercise the cross-venv worker protocol with the dummy backend in a real subprocess."""
import sys

import numpy as np

from backends.remote import RemoteBackend
from core.cache import LineCache
from core.emotion import resolve
from core.pipeline import render
from core.script import load_script


def test_remote_roundtrip(write_script, scene):
    s = load_script(write_script(scene))
    rb = RemoteBackend("dummy", sys.executable)
    try:
        rb.ensure_loaded()
        assert rb.sample_rate == 22050
        rb.prepare_voices([s.speakers["Narr"]], {"Narr": [resolve("calm")]})
        assert rb.normalize_text("hi") == "hi"
        a = rb.synth("Hello there", rb.voice(s.speakers["Narr"]), resolve("calm"))
        assert a.dtype == np.float32 and a.size > 1000
    finally:
        rb.unload()
    assert rb.proc is None


def test_remote_error_is_reported(write_script, scene):
    rb = RemoteBackend("dummy", sys.executable)
    try:
        rb.ensure_loaded()
        res = None
        try:
            rb._call(op="bogus")
        except Exception as e:  # noqa: BLE001
            res = str(e)
        assert res and "unknown op" in res
        # worker survives errors
        assert rb.normalize_text("ok") == "ok"
    finally:
        rb.unload()


def test_render_through_worker(write_script, scene, tmp_path):
    s = load_script(write_script(scene))
    res = render(s, {"en": "dummy", "bn": "dummy"}, lambda bid: RemoteBackend(bid, sys.executable),
                 LineCache(tmp_path / "c"))
    assert res.generated == 4 and res.wav.is_file()
