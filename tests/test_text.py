from core.emotion import resolve
from text import bn_norm, en_norm
from text.tags import find_tags, map_tags, strip_tags


def test_bn_numbers():
    assert bn_norm.int_to_words(0) == "শূন্য"
    assert bn_norm.int_to_words(25) == "পঁচিশ"
    assert bn_norm.int_to_words(100) == "একশো"
    assert bn_norm.int_to_words(345) == "তিনশো পঁয়তাল্লিশ"
    assert bn_norm.int_to_words(1500) == "এক হাজার পাঁচশো"
    assert bn_norm.int_to_words(250000) == "দুই লাখ পঞ্চাশ হাজার"
    assert bn_norm.int_to_words(30_000_000) == "তিন কোটি"


def test_bn_normalize_text():
    # expected strings go through the same Unicode canonicalisation (precomposed nukta letters)
    u = bn_norm.normalize_unicode
    assert bn_norm.normalize("১৯৭১ সালে") == u("উনিশশো একাত্তর সালে")
    assert bn_norm.normalize("আমার বয়স ২৫ বছর।") == u("আমার বয়স পঁচিশ বছর।")
    assert bn_norm.normalize("দাম ৳৫০০") == u("দাম পাঁচশো টাকা")
    assert bn_norm.normalize("৫০% ছাড়") == u("পঞ্চাশ শতাংশ ছাড়")
    assert bn_norm.normalize("১০টা বাজে") == u("দশটা বাজে")
    assert bn_norm.normalize("৩.৫ কেজি") == u("তিন দশমিক পাঁচ কেজি")
    assert bn_norm.normalize("আমি যাব|তুমি?") == u("আমি যাব। তুমি?")
    assert bn_norm.normalize("কেন ?") == u("কেন?")


def test_bn_unicode_fixes():
    decomposed = "ড়"  # DDA + nukta
    assert bn_norm.normalize("বা" + decomposed + "ি") == "বাড়ি"
    assert bn_norm.normalize("কো") == "কো"  # e-kar + aa-kar -> o-kar
    assert bn_norm.normalize("a​b") == "ab"  # zero-width space removed
    assert bn_norm.normalize("ক্‍য") == "ক্‍য"  # ZWJ kept


def test_en_normalize():
    assert en_norm.normalize("I have 3 cats.") == "I have three cats."
    assert en_norm.normalize("In 1999 and 2024.") == "In nineteen ninety-nine and twenty twenty-four."
    assert en_norm.normalize("It costs $5.50") == "It costs five dollars and fifty cents"
    assert en_norm.normalize("Dr. Rahman won 1st place") == "Doctor Rahman won first place"
    assert en_norm.normalize("50% off") == "fifty percent off"
    assert en_norm.normalize("pi is 3.14") == "pi is three point one four"
    assert en_norm.normalize("[laugh] 2 ghosts") == "[laugh] two ghosts"
    assert en_norm.int_to_words(1_234_567) == "one million two hundred thirty-four thousand five hundred sixty-seven"


def test_tags():
    t = "[laugh] Nobody ever [sigh] blames me!"
    assert find_tags(t) == ["laugh", "sigh"]
    assert strip_tags(t) == "Nobody ever blames me!"
    assert map_tags(t, {"laugh": "[laughter]"}) == "[laughter] Nobody ever blames me!"


def test_emotion_mapping():
    assert resolve("calm").exaggeration == 0.3
    assert resolve("angry").exaggeration == 0.9 and resolve("angry").cfg_weight == 0.3
    assert resolve("happy").exaggeration == 0.7
    assert resolve("sad").exaggeration == 0.4
    assert resolve("neutral").exaggeration == 0.5
    assert "angrily" in resolve("ANGER").instruct  # alias + case
    e = resolve("pirate-like")
    assert e.exaggeration == 0.5 and "pirate-like" in e.instruct
    e = resolve("happy", exaggeration=0.95, instruct="Whisper the last word.")
    assert e.exaggeration == 0.95 and e.cfg_weight == 0.3 and e.instruct.endswith("Whisper the last word.")
