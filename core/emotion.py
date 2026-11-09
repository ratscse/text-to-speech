"""Emotion labels -> backend controls.

English (Qwen3-TTS): a natural-language instruction string.
Bengali / Chatterbox: an `exaggeration` float (+ a matching cfg_weight, per the
Chatterbox tip that expressive speech wants a lower cfg_weight).
"""
from __future__ import annotations

from dataclasses import dataclass

# label: (exaggeration, cfg_weight, english instruction)
# exaggeration drives Bengali / Chatterbox (0 = flat, ~1.3 = very intense; >1.5 gets unstable).
# Values are grouped from quiet to intense so a scene can move through a real range.
EMOTIONS: dict[str, tuple[float, float, str]] = {
    # --- neutral / quiet -------------------------------------------------
    "neutral":      (0.50, 0.50, "Speak naturally in a relaxed, conversational tone."),
    "calm":         (0.30, 0.50, "Speak calmly and evenly, unhurried, with a soft, steady voice."),
    "soft":         (0.35, 0.50, "Speak softly and gently, quiet and warm."),
    "whisper":      (0.35, 0.50, "Whisper softly, breathy and quiet, close to the listener."),
    "tired":        (0.35, 0.50, "Speak wearily, low energy, slow, with sighing breath."),
    "bored":        (0.30, 0.50, "Speak in a bored, flat, uninterested tone."),
    "cold":         (0.35, 0.50, "Speak coldly and distantly, flat and emotionally detached."),
    "serious":      (0.45, 0.50, "Speak seriously and firmly, measured and grave."),
    "shy":          (0.40, 0.50, "Speak shyly and hesitantly, quiet, a little unsure."),
    "tender":       (0.40, 0.50, "Speak tenderly and affectionately, soft and intimate."),
    "romantic":     (0.50, 0.50, "Speak romantically, warm, slow and intimate."),
    # --- sad family ------------------------------------------------------
    "sad":          (0.40, 0.50, "Speak sadly, quietly and slowly, with a heavy, downcast tone."),
    "lonely":       (0.40, 0.50, "Speak with loneliness, quiet, slow and empty."),
    "regretful":    (0.50, 0.50, "Speak with regret, slow, heavy, apologetic."),
    "disappointed": (0.55, 0.50, "Speak with disappointment, deflated, tired and let down."),
    "hurt":         (0.65, 0.40, "Speak in a hurt, wounded voice, holding back pain."),
    "emotional":    (0.75, 0.40, "Speak emotionally, voice trembling slightly, deeply moved."),
    "pleading":     (0.80, 0.30, "Speak pleadingly, begging, urgent and vulnerable."),
    "heartbroken":  (0.85, 0.30, "Speak heartbroken, voice cracking, slow and full of pain."),
    "desperate":    (0.95, 0.30, "Speak desperately, urgent, strained and breaking."),
    "crying":       (1.10, 0.30, "Speak while crying, tearful, voice breaking, with shaky breaths."),
    "sobbing":      (1.30, 0.30, "Speak while sobbing heavily, broken words, gasping between phrases."),
    # --- happy family ----------------------------------------------------
    "relieved":     (0.55, 0.50, "Speak with relief, exhaling, relaxed and grateful."),
    "hopeful":      (0.55, 0.50, "Speak hopefully, gentle and quietly optimistic."),
    "grateful":     (0.55, 0.50, "Speak gratefully, warm and sincere."),
    "proud":        (0.65, 0.40, "Speak proudly, confident and warm, chin up."),
    "confident":    (0.60, 0.40, "Speak confidently, clear, steady and assured."),
    "happy":        (0.70, 0.40, "Speak happily and warmly, with a bright, smiling tone."),
    "cheerful":     (0.70, 0.40, "Speak cheerfully, upbeat and friendly."),
    "playful":      (0.70, 0.40, "Speak playfully and teasingly, with a light, bouncy rhythm."),
    "excited":      (0.85, 0.35, "Speak with lively excitement, energetic and slightly fast."),
    "laughing":     (1.00, 0.30, "Speak while laughing, breathy, words broken by laughter."),
    # --- angry family ----------------------------------------------------
    "sarcastic":    (0.60, 0.45, "Speak sarcastically, dry and drawn-out, with exaggerated emphasis."),
    "annoyed":      (0.65, 0.40, "Speak with annoyance, irritated and impatient."),
    "jealous":      (0.70, 0.40, "Speak jealously, bitter and resentful."),
    "disgusted":    (0.70, 0.40, "Speak with disgust, sneering and repelled."),
    "frustrated":   (0.75, 0.40, "Speak with frustration, tense, exasperated, close to snapping."),
    "angry":        (0.90, 0.30, "Speak angrily, tense and forceful, slightly rushed."),
    "furious":      (1.20, 0.30, "Speak furiously, explosive, loud and sharp."),
    "shouting":     (1.30, 0.30, "Shout loudly and forcefully."),
    # --- fear / surprise family ------------------------------------------
    "curious":      (0.55, 0.50, "Speak curiously, inquisitive, rising intonation."),
    "confused":     (0.55, 0.50, "Speak confused, hesitant and puzzled."),
    "nervous":      (0.65, 0.40, "Speak nervously, hesitant, slightly shaky and quick."),
    "anxious":      (0.65, 0.40, "Speak anxiously, worried, tense and restless."),
    "scared":       (0.75, 0.40, "Speak fearfully, with a shaky, breathy, nervous voice."),
    "surprised":    (0.80, 0.40, "Speak with surprise, rising pitch, as if caught off guard."),
    "shocked":      (0.95, 0.30, "Speak in shock, stunned, breathless and disbelieving."),
    "terrified":    (1.10, 0.30, "Speak terrified, trembling, breathless and panicked."),
    "panicked":     (1.15, 0.30, "Speak in panic, very fast, breathless and frantic."),
}

ALIASES = {
    "happiness": "happy", "joy": "happy", "joyful": "happy", "glad": "happy",
    "anger": "angry", "mad": "angry", "rage": "furious", "yelling": "shouting", "screaming": "shouting",
    "irritated": "annoyed",
    "sadness": "sad", "upset": "sad", "melancholy": "sad",
    "heartbreak": "heartbroken", "broken": "heartbroken", "wounded": "hurt",
    "cry": "crying", "tearful": "crying", "teary": "crying", "weeping": "crying",
    "sob": "sobbing", "begging": "pleading",
    "fear": "scared", "afraid": "scared", "worried": "anxious",
    "surprise": "surprised", "terror": "terrified", "panic": "panicked",
    "relaxed": "calm", "gentle": "soft", "laugh": "laughing",
    "loving": "tender", "affectionate": "tender",
    "whispering": "whisper", "sleepy": "tired", "exhausted": "tired",
    "regret": "regretful", "sorry": "regretful",
}


# Voice-clip fallback: if voices/<name>_<emotion>.wav is missing, use the clip of the
# emotion's group (e.g. 'lonely' -> voices/<name>_sad.wav), then the plain voices/<name>.wav.
MOOD_GROUPS = {
    "sad": "sad lonely regretful disappointed hurt emotional pleading heartbroken desperate crying sobbing",
    "soft": "bored calm cold soft tired whisper shy tender serious neutral romantic",
    "happy": "relieved hopeful grateful confident proud happy cheerful playful excited laughing",
    "angry": "sarcastic annoyed jealous disgusted frustrated angry furious shouting",
    "surprised": "curious confused nervous anxious scared surprised shocked terrified panicked",
}
MOOD_FALLBACK = {label: group for group, labels in MOOD_GROUPS.items() for label in labels.split()}


def list_emotions() -> str:
    """Human-readable table for `tts.py --emotions`."""
    rows = [f"{'mode':<14}{'exaggeration':>13}  aliases"]
    for name, (exg, _cfg, _ins) in sorted(EMOTIONS.items(), key=lambda kv: (kv[1][0], kv[0])):
        al = ", ".join(sorted(a for a, t in ALIASES.items() if t == name))
        rows.append(f"{name:<14}{exg:>13.2f}  {al}")
    return "\n".join(rows)


# Inline tag -> phrase added to the Qwen instruction (Qwen has no tag tokens).
TAG_DIRECTIONS = {
    "laugh": "laughing as you speak",
    "laughs": "laughing as you speak",
    "chuckle": "with a light chuckle",
    "giggle": "giggling",
    "sigh": "with an audible sigh",
    "gasp": "with a sharp gasp",
    "whisper": "in a whisper",
    "cough": "clearing the throat with a small cough",
    "cry": "on the verge of tears",
    "sob": "sobbing",
    "groan": "with a groan",
    "shout": "shouting",
}


@dataclass(frozen=True)
class EmotionSpec:
    label: str
    exaggeration: float
    cfg_weight: float
    instruct: str  # full English instruction (emotion + any extra direction)

    def key(self) -> str:
        return f"{self.label}|{self.exaggeration:.3f}|{self.cfg_weight:.3f}|{self.instruct}"


def known_label(label: str) -> str | None:
    label = (label or "neutral").strip().lower()
    label = ALIASES.get(label, label)
    return label if label in EMOTIONS else None


def resolve(label: str, exaggeration: float | None = None, instruct: str | None = None) -> EmotionSpec:
    """Unknown labels are allowed: they become free-text instructions for English
    ("speak like a pirate") and neutral exaggeration for Bengali."""
    raw = (label or "neutral").strip().lower()
    norm = known_label(raw)
    if norm is not None:
        exg, cfg, ins = EMOTIONS[norm]
    else:
        exg, cfg, _ = EMOTIONS["neutral"]
        ins = f"Speak in a {raw} manner."
        norm = raw
    if exaggeration is not None:
        exg = float(exaggeration)
        cfg = 0.3 if exg >= 0.8 else (0.4 if exg >= 0.6 else 0.5)
    if instruct:
        ins = f"{ins} {instruct.strip()}"
    return EmotionSpec(label=norm, exaggeration=exg, cfg_weight=cfg, instruct=ins)
