import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def ref_wav(tmp_path: Path) -> Path:
    p = tmp_path / "voices" / "ref.wav"
    p.parent.mkdir()
    t = np.arange(24000 * 6) / 24000
    sf.write(p, (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), 24000)
    return p


@pytest.fixture
def write_script(tmp_path: Path):
    def _write(data: dict, name: str = "script.json") -> Path:
        p = tmp_path / name
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return p

    return _write


@pytest.fixture
def scene(ref_wav: Path) -> dict:
    return {
        "output": "out/scene.wav",
        "pause_ms": {"same_speaker": 250, "turn": 600, "paragraph": 1200},
        "speakers": {
            "Rina": {"lang": "bn", "ref_wav": "voices/ref.wav"},
            "Arif": {"ref_wav": "voices/ref.wav"},
            "Narr": {"lang": "en", "ref_wav": "voices/ref.wav"},
            "Boo": {"lang": "en", "design": "small squeaky cartoon ghost"},
        },
        "lines": [
            {"speaker": "Narr", "emotion": "calm", "text": "It was raining in Dhaka."},
            {"speaker": "Rina", "emotion": "angry", "text": "তুমি আবার দেরি করলে?"},
            {"speaker": "Arif", "emotion": "sad", "text": "সরি, জ্যামে আটকে ছিলাম।"},
            {"speaker": "Boo", "emotion": "happy", "text": "[laugh] Nobody ever blames the ghost!"},
        ],
    }
