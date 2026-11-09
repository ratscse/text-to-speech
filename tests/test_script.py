import pytest

from core.script import ScriptError, chunk_text, detect_lang, load_script


def test_load_scene(write_script, scene, ref_wav):
    s = load_script(write_script(scene))
    assert [ln.lang for ln in s.lines] == ["en", "bn", "bn", "en"]  # Arif auto-detected
    assert s.speakers["Narr"].ref_wav == str(ref_wav.resolve())
    assert s.speakers["Boo"].design
    assert s.output.name == "scene.wav"
    assert s.langs_used() == {"en", "bn"}


def test_detect_lang_majority():
    assert detect_lang("তুমি আবার দেরি করলে?") == "bn"
    assert detect_lang("Hello there") == "en"
    assert detect_lang("আমি office যাচ্ছি এখন") == "bn"  # mostly Bengali
    assert detect_lang("I said ভালো to him yesterday") == "en"  # mostly Latin
    assert detect_lang("123 !!!") == "en"
    assert detect_lang("[laugh] হাহা") == "bn"  # tags ignored


def test_unsupported_language(write_script, scene):
    scene["speakers"]["Narr"]["lang"] = "hi"
    with pytest.raises(ScriptError, match="unsupported language 'hi'"):
        load_script(write_script(scene))


def test_line_lang_override_unsupported(write_script, scene):
    scene["lines"][0]["lang"] = "fr"
    with pytest.raises(ScriptError, match="only 'en'|Only 'en'"):
        load_script(write_script(scene))


def test_bn_design_rejected(write_script, scene):
    scene["speakers"]["Boo"]["lang"] = "bn"
    with pytest.raises(ScriptError, match="cloning only"):
        load_script(write_script(scene))


def test_design_speaker_with_bengali_line_rejected(write_script, scene):
    scene["speakers"]["Boo"].pop("lang")
    scene["lines"][3]["text"] = "আমি ভূত!"
    with pytest.raises(ScriptError, match="cloning only"):
        load_script(write_script(scene))


def test_missing_ref_wav(write_script, scene):
    scene["speakers"]["Rina"]["ref_wav"] = "voices/nope.wav"
    with pytest.raises(ScriptError, match="ref_wav not found"):
        load_script(write_script(scene))


def test_ref_and_design_exclusive(write_script, scene):
    scene["speakers"]["Narr"]["design"] = "deep voice"
    with pytest.raises(ScriptError, match="exactly one"):
        load_script(write_script(scene))


def test_unknown_speaker(write_script, scene):
    scene["lines"][0]["speaker"] = "Ghost"
    with pytest.raises(ScriptError, match="unknown speaker"):
        load_script(write_script(scene))


def test_sidecar_transcript(write_script, scene, ref_wav):
    ref_wav.with_suffix(".txt").write_text("hello transcript", encoding="utf-8")
    s = load_script(write_script(scene))
    assert s.speakers["Narr"].ref_text == "hello transcript"


def test_mood_clip_picked_up(write_script, scene, ref_wav):
    import shutil

    shutil.copy(ref_wav, ref_wav.with_name(f"{ref_wav.stem}_sad{ref_wav.suffix}"))
    scene["lines"][0]["emotion"] = "sadness"  # alias resolves to the 'sad' clip
    s = load_script(write_script(scene))
    ln = s.lines[0]
    assert ln.voice == f"{ln.speaker}@sad"
    assert s.speakers[ln.voice].ref_wav.endswith("_sad.wav")
    assert all(o.voice is None for o in s.lines if o.emotion not in ("sad", "sadness"))  # only _sad exists


def test_mood_clip_group_fallback(write_script, scene, ref_wav):
    import shutil

    shutil.copy(ref_wav, ref_wav.with_name(f"{ref_wav.stem}_soft{ref_wav.suffix}"))
    shutil.copy(ref_wav, ref_wav.with_name(f"{ref_wav.stem}_calm{ref_wav.suffix}"))
    scene["lines"][0]["emotion"] = "tired"  # no _tired clip -> soft group clip
    scene["lines"][1]["emotion"] = "calm"  # exact clip wins over the group clip
    scene["lines"][2]["emotion"] = "lonely"  # sad group, no _sad clip -> plain voice
    s = load_script(write_script(scene))
    assert s.lines[0].voice == f"{s.lines[0].speaker}@soft"
    assert s.speakers[s.lines[0].voice].ref_wav.endswith("_soft.wav")
    assert s.speakers[s.lines[1].voice].ref_wav.endswith("_calm.wav")
    assert s.lines[2].voice is None


def test_bad_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{oops", encoding="utf-8")
    with pytest.raises(ScriptError, match="invalid JSON"):
        load_script(p)


def test_chunk_short_untouched():
    assert chunk_text("Hello there.") == ["Hello there."]


def test_chunk_english_sentences():
    text = " ".join(f"This is sentence number {i} in a long paragraph." for i in range(12))
    chunks = chunk_text(text, 200)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks) == text
    assert all(c.endswith(".") for c in chunks)


def test_chunk_bangla_dari():
    sent = "আমি আজ সকালে বাজারে গিয়েছিলাম এবং অনেক কিছু কিনেছি।"
    text = " ".join([sent] * 8)
    chunks = chunk_text(text, 200)
    assert len(chunks) >= 2
    assert all(len(c) <= 200 for c in chunks)
    assert all(c.endswith("।") for c in chunks)
    assert " ".join(chunks) == text


def test_chunk_no_punctuation_hard_wraps():
    text = " ".join(["word"] * 120)
    chunks = chunk_text(text, 50)
    assert all(len(c) <= 50 for c in chunks)
    assert " ".join(chunks) == text
