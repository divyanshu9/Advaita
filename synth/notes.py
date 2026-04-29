"""
notes.py — Musical note frequencies and scale helpers.

In 12-tone equal temperament (12-TET), the octave is divided into 12 equal
semitones. Each semitone is a frequency ratio of 2^(1/12) ≈ 1.0595.

Reference: A4 = 440 Hz  (MIDI note 69)

Formula:
    freq(n) = 440 * 2^((n - 69) / 12)
    where n is the MIDI note number.
"""

import numpy as np

# MIDI note number for A4
A4_MIDI = 69
A4_FREQ = 440.0

# Chromatic note names (sharps)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_freq(midi_note: int) -> float:
    """Convert a MIDI note number to frequency in Hz."""
    return A4_FREQ * 2 ** ((midi_note - A4_MIDI) / 12)


def note_to_midi(name: str, octave: int) -> int:
    """
    Convert note name + octave to MIDI number.
    e.g. note_to_midi("A", 4) -> 69
         note_to_midi("C", 4) -> 60 (middle C)
    """
    name = name.upper().replace("b", "#")  # treat flats as sharps
    semitone = NOTE_NAMES.index(name)
    return (octave + 1) * 12 + semitone


def note_to_freq(name: str, octave: int) -> float:
    """Convenience: note name + octave -> Hz."""
    return midi_to_freq(note_to_midi(name, octave))


# ── Common frequencies for quick reference ─────────────────────────────────
STANDARD_NOTES = {
    name + str(octave): note_to_freq(name, octave)
    for octave in range(0, 9)
    for name in NOTE_NAMES
}

# ── Scale builders ──────────────────────────────────────────────────────────

# Intervals (in semitones) for common scales
SCALE_INTERVALS = {
    "major":           [0, 2, 4, 5, 7, 9, 11, 12],
    "minor_natural":   [0, 2, 3, 5, 7, 8, 10, 12],
    "minor_harmonic":  [0, 2, 3, 5, 7, 8, 11, 12],
    "pentatonic_major":[0, 2, 4, 7, 9, 12],
    "pentatonic_minor":[0, 3, 5, 7, 10, 12],
    "blues":           [0, 3, 5, 6, 7, 10, 12],
    "chromatic":       list(range(13)),
}

# Intervals (semitones above root) for common chords
CHORD_INTERVALS = {
    "major":     [0, 4, 7],
    "minor":     [0, 3, 7],
    "diminished":[0, 3, 6],
    "augmented": [0, 4, 8],
    "major7":    [0, 4, 7, 11],
    "minor7":    [0, 3, 7, 10],
    "dominant7": [0, 4, 7, 10],
}


def build_scale(root_name: str, root_octave: int, scale: str = "major") -> list[float]:
    """
    Return a list of frequencies for a scale.

    Example:
        build_scale("C", 4, "major")  -> [261.6, 293.7, 329.6, 349.2, 392.0, 440.0, 493.9, 523.3]
    """
    root_midi = note_to_midi(root_name, root_octave)
    intervals = SCALE_INTERVALS[scale]
    return [midi_to_freq(root_midi + i) for i in intervals]


def build_chord(root_name: str, root_octave: int, chord: str = "major") -> list[float]:
    """Return a list of frequencies for a chord."""
    root_midi = note_to_midi(root_name, root_octave)
    intervals = CHORD_INTERVALS[chord]
    return [midi_to_freq(root_midi + i) for i in intervals]


if __name__ == "__main__":
    print("A4 =", note_to_freq("A", 4), "Hz")
    print("C4 (middle C) =", note_to_freq("C", 4), "Hz")
    print()
    print("C major scale (C4):")
    for f in build_scale("C", 4, "major"):
        print(f"  {f:.2f} Hz")
    print()
    print("A minor chord (A3):")
    for f in build_chord("A", 3, "minor"):
        print(f"  {f:.2f} Hz")
