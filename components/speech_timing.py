from config import DEFAULT_TTS_RATE, DEFAULT_TTS_WPM_AT_HALF


def estimate_tts_duration_sec(
    text: str,
    rate: float = DEFAULT_TTS_RATE,
    wpm_at_half: float = DEFAULT_TTS_WPM_AT_HALF,
) -> float:
    if not (text or "").strip():
        return 0.0
    r = max(0.01, min(1.0, float(rate)))
    wpm = wpm_at_half * (0.5 + 0.5 * r)
    word_count = max(1, len(text.split()))
    return word_count * (60.0 / wpm)
