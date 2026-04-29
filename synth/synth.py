"""
synth.py — Core synthesis engine.

Covers three foundational synthesis techniques:

1. ADDITIVE SYNTHESIS
   Build complex timbres by summing sine waves at harmonic (or arbitrary)
   frequencies. A pure 440 Hz sine is a flute-like tone; adding its overtones
   (880, 1320, 1760 … Hz) at decreasing amplitudes gives richer timbres like
   a piano, organ, or strings.

2. WAVETABLE / CLASSIC WAVEFORMS
   Instead of computing harmonics manually, generate classic synthesizer
   waveforms directly: sine, square, sawtooth, triangle. Each has a known
   harmonic profile that we can also derive analytically.

3. FM SYNTHESIS (Frequency Modulation)
   Modulate the frequency of one oscillator (carrier) with another (modulator).
   Even two sine waves can produce a huge variety of complex spectra.
   Key parameters: modulation index (depth) and ratio of modulator/carrier.

4. ADSR ENVELOPE
   Shape the amplitude of any sound over time:
     Attack  — time to ramp from 0 to peak after note-on
     Decay   — time to fall from peak to sustain level
     Sustain — amplitude held while key is pressed  (0–1)
     Release — time to fade from sustain to 0 after note-off

All functions return numpy arrays of float32 samples normalised to [-1, 1].
"""

import numpy as np
from typing import Sequence

SAMPLE_RATE = 44100  # Hz — CD quality


# ── Low-level helpers ──────────────────────────────────────────────────────

def _time(duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return a time axis array for the given duration in seconds."""
    n_samples = int(duration * sample_rate)
    return np.linspace(0, duration, n_samples, endpoint=False)


def _normalize(samples: np.ndarray) -> np.ndarray:
    """Scale samples to [-1, 1]. Returns unchanged if all-zero."""
    peak = np.max(np.abs(samples))
    return samples / peak if peak > 0 else samples


# ── 1. Basic oscillators ──────────────────────────────────────────────────

def sine(freq: float, duration: float, amplitude: float = 1.0,
         phase: float = 0.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Pure sine wave — the building block of all additive synthesis.

    y(t) = A * sin(2π * f * t + φ)

    A sine wave contains only the fundamental frequency — no harmonics.
    Perceptually it sounds very "pure" and flute-like.
    """
    t = _time(duration, sample_rate)
    return amplitude * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)


def square(freq: float, duration: float, amplitude: float = 1.0,
           n_harmonics: int = 40, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Square wave via additive synthesis (Fourier series).

    A square wave contains ONLY odd harmonics at amplitudes 1/n:
        y(t) = (4/π) * Σ  sin(2π * (2k-1) * f * t) / (2k-1)   k = 1, 2, 3 …

    Sounds bright and hollow — like a clarinet.
    Limiting n_harmonics avoids Gibbs phenomenon and aliasing.
    """
    t = _time(duration, sample_rate)
    wave = np.zeros(len(t), dtype=np.float32)
    for k in range(1, n_harmonics + 1):
        harmonic = 2 * k - 1          # 1, 3, 5, 7 …
        wave += (1.0 / harmonic) * np.sin(2 * np.pi * harmonic * freq * t)
    return amplitude * (4 / np.pi) * wave


def sawtooth(freq: float, duration: float, amplitude: float = 1.0,
             n_harmonics: int = 40, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Sawtooth wave via additive synthesis.

    Contains ALL harmonics (fundamental + overtones) at amplitudes 1/n:
        y(t) = (2/π) * Σ  (-1)^(k+1) * sin(2π * k * f * t) / k   k = 1, 2, 3 …

    The richest standard waveform — bright, buzzy, ideal for strings/brass.
    """
    t = _time(duration, sample_rate)
    wave = np.zeros(len(t), dtype=np.float32)
    for k in range(1, n_harmonics + 1):
        sign = (-1) ** (k + 1)
        wave += sign * (1.0 / k) * np.sin(2 * np.pi * k * freq * t)
    return amplitude * (2 / np.pi) * wave


def triangle(freq: float, duration: float, amplitude: float = 1.0,
             n_harmonics: int = 40, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Triangle wave via additive synthesis.

    Odd harmonics only (like square) but amplitudes fall as 1/n²:
        y(t) = (8/π²) * Σ  (-1)^k * sin(2π * (2k+1) * f * t) / (2k+1)²

    Softer than square, slightly hollow — like an ocarina.
    """
    t = _time(duration, sample_rate)
    wave = np.zeros(len(t), dtype=np.float32)
    for k in range(0, n_harmonics):
        sign = (-1) ** k
        harmonic = 2 * k + 1          # 1, 3, 5, 7 …
        wave += sign * (1.0 / harmonic ** 2) * np.sin(2 * np.pi * harmonic * freq * t)
    return amplitude * (8 / np.pi ** 2) * wave


# ── 2. Additive synthesis ─────────────────────────────────────────────────

def additive(
    freq: float,
    duration: float,
    harmonics: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8),
    amplitudes: Sequence[float] | None = None,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    General additive synthesizer: mix arbitrary harmonics at given amplitudes.

    Parameters
    ----------
    freq       : fundamental frequency (Hz)
    harmonics  : list of harmonic numbers (1 = fundamental, 2 = octave, etc.)
    amplitudes : relative amplitudes for each harmonic; defaults to 1/n rolloff

    Examples
    --------
    # Organ-like (first 4 harmonics, equal amplitude):
    additive(440, 1.0, harmonics=[1,2,3,4], amplitudes=[1, 0.5, 0.25, 0.125])

    # Bell-like (inharmonic partials):
    additive(440, 2.0, harmonics=[1, 2.756, 5.404, 8.933])
    """
    if amplitudes is None:
        amplitudes = [1.0 / h for h in harmonics]

    t = _time(duration, sample_rate)
    wave = np.zeros(len(t), dtype=np.float32)
    for h, a in zip(harmonics, amplitudes):
        wave += a * np.sin(2 * np.pi * h * freq * t)

    return _normalize(wave).astype(np.float32)


# ── 3. FM synthesis ───────────────────────────────────────────────────────

def fm(
    carrier_freq: float,
    duration: float,
    mod_ratio: float = 2.0,
    mod_index: float = 5.0,
    amplitude: float = 1.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Two-operator FM synthesis (Chowning algorithm, 1973).

    Carrier frequency:  fc
    Modulator frequency: fm = fc * mod_ratio
    Instantaneous frequency: fc + mod_index * fm * sin(2π * fm * t)

    The modulation INDEX (β) controls harmonic richness:
        β ≈ 0   → nearly pure sine (just carrier)
        β = 1   → slight overtones
        β = 5   → moderate brightness
        β > 10  → dense, metallic spectrum

    The mod_ratio determines which harmonics appear:
        ratio = 1   → harmonically related partials (organ-like)
        ratio = 1.5 → inharmonic partials (bell/gong-like)
        ratio = 3.5 → complex metallic textures
    """
    t = _time(duration, sample_rate)
    mod_freq = carrier_freq * mod_ratio
    modulator = mod_index * mod_freq * np.sin(2 * np.pi * mod_freq * t)
    carrier = amplitude * np.sin(2 * np.pi * carrier_freq * t + modulator)
    return carrier.astype(np.float32)


# ── 4. ADSR envelope ──────────────────────────────────────────────────────

def adsr_envelope(
    duration: float,
    attack: float = 0.01,
    decay: float = 0.1,
    sustain: float = 0.7,
    release: float = 0.2,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Build an ADSR amplitude envelope.

    The four stages:
        Attack  [0 … attack]          : ramp from 0 → 1
        Decay   [attack … a+d]        : ramp from 1 → sustain
        Sustain [a+d … duration-rel]  : hold at sustain level
        Release [duration-rel … end]  : ramp from sustain → 0

    Multiplying any waveform by this envelope makes it feel like a real note.

    Parameters (all in seconds except sustain which is 0–1 amplitude):
        attack   : ramp-up time
        decay    : fall time from peak to sustain
        sustain  : amplitude during the held portion (0 = no sustain)
        release  : fade-out time
    """
    n = int(duration * sample_rate)
    envelope = np.zeros(n, dtype=np.float32)

    a = int(attack * sample_rate)
    d = int(decay * sample_rate)
    r = int(release * sample_rate)
    s_start = a + d
    s_end = max(s_start, n - r)

    if a > 0:
        envelope[:a] = np.linspace(0, 1, a)
    if d > 0:
        envelope[a:a+d] = np.linspace(1, sustain, min(d, n - a))
    if s_end > s_start:
        envelope[s_start:s_end] = sustain
    if r > 0 and s_end < n:
        envelope[s_end:] = np.linspace(sustain, 0, n - s_end)

    return envelope


def apply_envelope(wave: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    """Multiply a wave by an envelope (trimmed/padded to match lengths)."""
    n = min(len(wave), len(envelope))
    return (wave[:n] * envelope[:n]).astype(np.float32)


# ── 5. Chord / polyphony mixing ──────────────────────────────────────────

def mix(waves: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """
    Mix multiple waves into one, padding shorter ones with zeros.
    Optionally weight each wave. Result is normalised to [-1, 1].
    """
    if not waves:
        return np.array([], dtype=np.float32)

    max_len = max(len(w) for w in waves)
    if weights is None:
        weights = [1.0] * len(waves)

    result = np.zeros(max_len, dtype=np.float32)
    for w, weight in zip(waves, weights):
        padded = np.pad(w, (0, max_len - len(w)))
        result += weight * padded

    return _normalize(result)


def chord(
    freqs: Sequence[float],
    duration: float,
    wave_fn=None,
    attack: float = 0.02,
    decay: float = 0.1,
    sustain: float = 0.6,
    release: float = 0.3,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Play a chord: multiple frequencies simultaneously, each with an ADSR envelope.

    wave_fn : oscillator function to use (defaults to sine)
              can be any of: sine, square, sawtooth, triangle, or a lambda
    """
    if wave_fn is None:
        wave_fn = sine

    env = adsr_envelope(duration, attack, decay, sustain, release, sample_rate)
    waves = [
        apply_envelope(wave_fn(f, duration, sample_rate=sample_rate), env)
        for f in freqs
    ]
    return mix(waves)


# ── 6. Sequencing ─────────────────────────────────────────────────────────

def sequence(
    notes: list[tuple[float, float]],
    wave_fn=None,
    attack: float = 0.01,
    decay: float = 0.05,
    sustain: float = 0.7,
    release: float = 0.1,
    gap: float = 0.01,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Render a melody: list of (frequency, duration) tuples played in order.

    Pass 0.0 as frequency for a rest.
    gap adds a brief silence between notes to avoid them blurring.
    """
    if wave_fn is None:
        wave_fn = sine

    segments = []
    silence = np.zeros(int(gap * sample_rate), dtype=np.float32)

    for freq, dur in notes:
        if freq == 0.0:
            segments.append(np.zeros(int(dur * sample_rate), dtype=np.float32))
        else:
            env = adsr_envelope(dur, attack, decay, sustain, release, sample_rate)
            w = apply_envelope(wave_fn(freq, dur, sample_rate=sample_rate), env)
            segments.append(w)
        segments.append(silence)

    return np.concatenate(segments).astype(np.float32)
