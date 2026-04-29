# synth — Pure Python Music Synthesizer

A from-scratch audio synthesis engine and interactive terminal synthesizer built with NumPy and SoundDevice. No external audio libraries — every waveform is computed as raw floating-point samples.

---

## Features

- **Four classic waveforms:** Sine, Triangle, Square, Sawtooth
- **Additive synthesis:** Build timbres by layering harmonics with custom amplitudes
- **FM synthesis:** Chowning-algorithm frequency modulation (Yamaha DX7 style)
- **ADSR envelopes:** Shape amplitude over the note lifetime
- **Chord and melody sequencing:** Polyphonic playback from frequency lists
- **Interactive terminal piano:** Real-time polyphonic keyboard with 7 timbres
- **Guided demo:** 8-section tour with live waveform display and spectrum plots
- **Visualization:** Waveform, spectrum, ADSR, and multi-wave comparison plots (saved as PNG)

---

## Installation

```bash
pip install numpy sounddevice matplotlib scipy rich
```

Or from the requirements file:

```bash
pip install -r requirements.txt
```

> **macOS:** If `sounddevice` fails, install PortAudio first: `brew install portaudio`

---

## Project Structure

```
synth/
├── synth.py        # Core synthesis engine
├── notes.py        # Music theory — notes, scales, chords, MIDI
├── piano.py        # Interactive terminal piano (curses UI)
├── demo.py         # Guided synthesis demo with rich terminal UI
├── visualize.py    # Waveform and spectrum plots (saved as PNG)
├── plots/          # Auto-created — all generated PNGs go here
└── requirements.txt
```

---

## Modules

### `synth.py` — Synthesis Engine

All functions return a `numpy.ndarray` of `float32` samples normalised to `[-1, 1]`. Sample rate is 44 100 Hz.

| Function | Description |
|---|---|
| `sine(freq, duration)` | Pure sine wave |
| `square(freq, duration)` | Square wave (odd harmonics, 1/n rolloff) |
| `sawtooth(freq, duration)` | Sawtooth wave (all harmonics, 1/n rolloff) |
| `triangle(freq, duration)` | Triangle wave (odd harmonics, 1/n² rolloff) |
| `additive(freq, duration, harmonics, amplitudes)` | Custom harmonic mix |
| `fm(freq, duration, mod_ratio, mod_index)` | FM synthesis — carrier + modulator |
| `adsr_envelope(duration, attack, decay, sustain, release)` | ADSR amplitude envelope array |
| `apply_envelope(wave, envelope)` | Multiply wave by envelope |
| `mix(waves)` | Sum and normalise a list of waves |
| `chord(freqs, duration, wave_fn, ...)` | Polyphonic chord with ADSR |
| `sequence(notes, wave_fn, ...)` | Sequence of `(freq, duration)` pairs |

**FM synthesis formula:**

```
output(t) = sin(2π·fc·t  +  β · sin(2π·fm·t))

mod_ratio = fm / fc   →  harmonic relationships
mod_index = β         →  brightness / density
```

**Quick example:**

```python
from synth import sine, fm, adsr_envelope, apply_envelope, chord
from notes import build_chord
import sounddevice as sd

SAMPLE_RATE = 44100

# Sine wave — A4
wave = sine(440.0, 1.0)
sd.play(wave, samplerate=SAMPLE_RATE); sd.wait()

# FM bell
bell = fm(440.0, 2.0, mod_ratio=1.4, mod_index=5.0)
env  = adsr_envelope(2.0, attack=0.005, decay=0.4, sustain=0.1, release=1.5)
sd.play(apply_envelope(bell, env), samplerate=SAMPLE_RATE); sd.wait()

# C major chord
freqs = build_chord("C", 4, "major")
w = chord(freqs, 2.0)
sd.play(w, samplerate=SAMPLE_RATE); sd.wait()
```

---

### `notes.py` — Music Theory

```python
from notes import note_to_freq, build_scale, build_chord, SCALE_INTERVALS, CHORD_INTERVALS

note_to_freq("A", 4)          # → 440.0 Hz
note_to_freq("C", 4)          # → 261.63 Hz

build_scale("C", 4, "major")          # → [261.63, 293.66, 329.63, ...]
build_scale("A", 3, "pentatonic_minor")

build_chord("A", 3, "minor7")         # → [220.0, 261.63, 329.63, 392.0]
build_chord("C", 4, "dominant7")
```

**Available scales:** `major`, `minor_natural`, `minor_harmonic`, `pentatonic_major`, `pentatonic_minor`, `blues`, `chromatic`

**Available chords:** `major`, `minor`, `diminished`, `augmented`, `major7`, `minor7`, `dominant7`

MIDI utilities:

```python
from notes import midi_to_freq, note_to_midi
midi_to_freq(69)    # → 440.0 (A4)
note_to_midi("C", 4)  # → 60
```

---

### `piano.py` — Interactive Terminal Piano

```bash
python piano.py
```

A real-time polyphonic piano in your terminal. Press keyboard keys to play notes; sounds are rendered and streamed instantly via `sounddevice.OutputStream`.

**Keyboard layout:**

```
  Black keys:  W  E     T  Y  U     O  P
  White keys:  A  S  D  F  G  H  J  K  L  ;  =
               C4 D4 E4 F4 G4 A4 B4 C5 D5 E5
```

**Controls:**

| Key | Action |
|---|---|
| `A S D F G H J K L ; =` | Play notes C4–E5 (white keys) |
| `W E T Y U O P` | Play sharps/flats (black keys) |
| `1` – `7` | Switch waveform / timbre |
| `←` / `Z` | Octave down |
| `→` / `X` | Octave up |
| `↑` / `↓` | Volume up / down |
| `Q` / `Esc` | Quit |

**Available timbres:**

| # | Name | Description |
|---|---|---|
| 1 | Sine | Pure tone |
| 2 | Triangle | Soft, mellow |
| 3 | Square | Hollow, nasal |
| 4 | Sawtooth | Bright, buzzy |
| 5 | FM Bell | ratio=3.5, index=8 |
| 6 | FM Bass | ratio=1.0, index=3.5 |
| 7 | FM Metal | ratio=3.5, index=14.0 |

---

### `demo.py` — Guided Synthesis Tour

```bash
python demo.py
```

An interactive 8-section demonstration of audio synthesis concepts, with a rich terminal UI (progress bars, live waveform display). Each section plays examples and saves spectrum/waveform PNG plots to `plots/`.

| Key | Section |
|---|---|
| `1` | Pure sine waves — frequency and octaves |
| `2` | Harmonics — how overtones build timbre |
| `3` | Classic waveforms — sine / square / sawtooth / triangle |
| `4` | FM synthesis — Chowning algorithm, 5 presets |
| `5` | ADSR envelope — piano / organ / strings / pluck |
| `6` | Chords and polyphony |
| `7` | Melody sequencing — Twinkle Twinkle in two timbres |
| `8` | Full synth patch — warm pad, plucky bass, bell chord |
| `A` | Run all sections in order |
| `Q` | Quit |

---

### `visualize.py` — Plots

All plots are saved as PNG files in `synth/plots/` and never open GUI windows (uses the `Agg` backend).

| Function | Output |
|---|---|
| `plot_waveform(wave, title)` | Time-domain waveform |
| `plot_spectrum(wave, title, max_freq)` | Frequency spectrum (FFT magnitude) |
| `plot_waveform_and_spectrum(wave, title)` | Side-by-side waveform + spectrum |
| `plot_compare(waves_dict, title)` | Stacked comparison across multiple waves |
| `plot_adsr(attack, decay, sustain, release, duration)` | ADSR envelope shape with labelled phases |

```python
from synth import sine, sawtooth, fm
from visualize import plot_compare, plot_adsr

plot_compare({
    "Sine":     sine(440, 0.5),
    "Sawtooth": sawtooth(440, 0.5),
    "FM Bell":  fm(440, 0.5, mod_ratio=1.4, mod_index=5.0),
}, title="Three Timbres")

plot_adsr(attack=0.4, decay=0.1, sustain=0.8, release=0.6, duration=2.0)
```

---

## Synthesis Reference

### Tuning

12-tone equal temperament, A4 = 440 Hz:

```
freq = 440 × 2^((midi - 69) / 12)
```

### Waveform Harmonic Content

| Waveform | Harmonics | Rolloff | Character |
|---|---|---|---|
| Sine | Fundamental only | — | Pure, flute |
| Triangle | Odd (1, 3, 5, …) | 1/n² | Soft, ocarina |
| Square | Odd (1, 3, 5, …) | 1/n | Hollow, clarinet |
| Sawtooth | All (1, 2, 3, …) | 1/n | Bright, brass |

### FM Preset Parameters

| Preset | mod_ratio | mod_index | Sound |
|---|---|---|---|
| Organ | 1.0 | 1.0 | Mild harmonic |
| Brass | 1.0 | 4.0 | Bright buzz |
| Bell | 1.4 | 5.0 | Inharmonic ring |
| Metallic | 3.5 | 8.0 | Dense metal |
| Gong | 0.5 | 7.0 | Deep resonance |

### ADSR Preset Parameters

| Preset | Attack | Decay | Sustain | Release |
|---|---|---|---|---|
| Piano | 5 ms | 300 ms | 0.20 | 800 ms |
| Organ | 5 ms | 10 ms | 1.00 | 10 ms |
| Strings | 400 ms | 100 ms | 0.80 | 600 ms |
| Pluck | 2 ms | 500 ms | 0.00 | 100 ms |
