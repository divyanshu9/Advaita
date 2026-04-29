"""
demo.py — Musical Synthesis Explorer with rich terminal UI.

Controls:
  1–8   Run a specific section
  A     Run all sections in order
  Q     Quit
"""

import time
import numpy as np
import sounddevice as sd

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.rule import Rule
from rich.columns import Columns
from rich import box

from notes import note_to_freq, build_chord
from synth import (
    SAMPLE_RATE,
    sine, square, sawtooth, triangle,
    additive, fm,
    adsr_envelope, apply_envelope,
    chord, sequence, mix,
)
from visualize import plot_waveform_and_spectrum, plot_compare, plot_adsr

console = Console()

# ── Waveform display helpers ───────────────────────────────────────────────

BLOCKS = " ▁▂▃▄▅▆▇█"


def wave_to_bars(wave: np.ndarray, width: int = 62) -> str:
    """Downsample waveform amplitude to block-character bars."""
    n = len(wave)
    out = []
    for i in range(width):
        s = int(i * n / width)
        e = int((i + 1) * n / width)
        amp = float(np.max(np.abs(wave[s:e]))) if e > s else 0.0
        out.append(BLOCKS[min(int(amp * (len(BLOCKS) - 1)), len(BLOCKS) - 1)])
    return "".join(out)


def _playing_panel(label: str, bars: str, elapsed: float,
                   duration: float, info: str) -> Panel:
    progress = min(elapsed / duration, 1.0) if duration > 0 else 1.0
    filled = int(progress * 54)
    bar_str = "█" * filled + "░" * (54 - filled)

    body = Text()
    body.append(f"\n  {bars}\n\n", style="bold cyan")
    body.append(f"  {bar_str}  ", style="green")
    body.append(f"{elapsed:.1f}s", style="bold green")
    body.append(f" / {duration:.1f}s", style="dim green")
    if info:
        body.append(f"\n\n  {info}", style="dim white")
    body.append("\n")

    return Panel(
        body,
        title=f"[bold yellow]  ♪  {label}[/]",
        border_style="cyan",
        padding=(0, 1),
    )


def play(wave: np.ndarray, label: str = "", info: str = "",
         sample_rate: int = SAMPLE_RATE):
    """Play audio with a live animated waveform + progress bar."""
    duration = len(wave) / sample_rate
    bars = wave_to_bars(wave)

    sd.play(wave, samplerate=sample_rate)
    start = time.time()

    with Live(
        _playing_panel(label, bars, 0, duration, info),
        refresh_per_second=24,
        console=console,
        transient=False,
    ) as live:
        while True:
            elapsed = time.time() - start
            if elapsed >= duration:
                break
            live.update(_playing_panel(label, bars, elapsed, duration, info))
            time.sleep(0.04)
        live.update(_playing_panel(label, bars, duration, duration, info))

    sd.wait()


# ── UI chrome ──────────────────────────────────────────────────────────────

def print_header():
    console.clear()
    title = Text()
    title.append("  ♩ ♪ ♫  ", style="bold yellow")
    title.append("MUSICAL SYNTHESIS EXPLORER", style="bold white")
    title.append("  ♫ ♪ ♩  ", style="bold yellow")
    sub = Text("\n  From a single sine wave to a full synthesizer", style="dim")
    console.print(Panel(
        Align.center(Text.assemble(title, sub)),
        border_style="bold blue",
        padding=(1, 6),
    ))


def print_menu():
    table = Table(
        box=box.ROUNDED,
        border_style="blue",
        show_header=True,
        header_style="bold blue",
        padding=(0, 2),
        min_width=64,
    )
    table.add_column("Key",     style="bold yellow",  width=5)
    table.add_column("Section", style="bold white",   width=20)
    table.add_column("Covers",  style="dim",          width=38)

    rows = [
        ("1", "Sine Waves",    "Pure tones, octave = 2× frequency"),
        ("2", "Harmonics",     "Overtone series builds timbre"),
        ("3", "Waveforms",     "Sine / Square / Sawtooth / Triangle"),
        ("4", "FM Synthesis",  "Yamaha DX7 — 2 sines → complex timbres"),
        ("5", "ADSR Envelope", "Shape amplitude: attack, decay, sustain, release"),
        ("6", "Chords",        "Polyphony by mixing frequency waves"),
        ("7", "Melody",        "Sequence notes into a tune"),
        ("8", "Synth Patch",   "Everything combined into instruments"),
        ("", "", ""),
        ("A", "Run All",       "Full guided tour, all sections"),
        ("Q", "Quit",          ""),
    ]
    for k, n, d in rows:
        table.add_row(f"[{k}]" if k else "", n, d)

    console.print()
    console.print(Align.center(table))
    console.print()


def section_banner(number: str, title: str):
    console.print()
    console.print(Rule(
        f"[bold cyan]  SECTION {number} — {title}  [/]",
        style="cyan",
    ))
    console.print()


def explainer(text: str):
    console.print(Panel(
        text.strip(),
        border_style="dim blue",
        padding=(0, 3),
    ))
    console.print()


def done(name: str):
    console.print(f"\n  [bold green]✓[/] [dim]{name} complete[/]\n")


# ── Section 1: Sine waves ─────────────────────────────────────────────────

def demo_sine_waves():
    section_banner("1", "Pure Sine Waves")
    explainer("""
A [bold]sine wave[/] is the simplest possible sound — one frequency, zero harmonics.
It sounds very "clean" and flute-like to the human ear.

Key rule:  [bold yellow]octave = doubling the frequency[/]

  A3 =  220 Hz      A4 =  440 Hz  (concert pitch)      A5 =  880 Hz

A perfect fifth above A4 is E5 = 659 Hz  (ratio 3:2 — the most consonant interval).
""")
    env = adsr_envelope(1.5, attack=0.05, decay=0.1, sustain=0.8, release=0.3)
    notes = [
        ("A3 — 220 Hz",       220.0, "One octave below concert pitch"),
        ("A4 — 440 Hz",       440.0, "Concert pitch — standard tuning reference"),
        ("A5 — 880 Hz",       880.0, "One octave above — double the frequency"),
        ("E5 — 659 Hz",  659.255, "Perfect fifth above A4  (ratio 3:2)"),
    ]
    for label, freq, info in notes:
        play(apply_envelope(sine(freq, 1.5), env), label, info)

    console.print("  [dim]Saving spectrum plot → plots/[/]")
    plot_waveform_and_spectrum(sine(440.0, 0.5),
                               title="Pure Sine Wave — A4 (440 Hz)")
    done("Sine Waves")


# ── Section 2: Harmonics ──────────────────────────────────────────────────

def demo_harmonics():
    section_banner("2", "Harmonics & Timbre")
    explainer("""
When a real instrument plays 440 Hz, you also hear [bold]overtones[/] above it:

  1st harmonic (fundamental) :  440 Hz   ← what you "hear" as the pitch
  2nd harmonic (octave)      :  880 Hz
  3rd harmonic               : 1320 Hz
  4th harmonic               : 1760 Hz  …

Their [bold yellow]relative loudness = timbre[/] (tone colour).
Listen as each harmonic is added — the sound grows richer each time.
""")
    base, dur = 440.0, 2.0
    env = adsr_envelope(dur, attack=0.05, decay=0.1, sustain=0.8, release=0.3)
    acc = []
    for n in range(1, 7):
        acc.append(n)
        amps = [1.0 / h for h in acc]
        w = additive(base, dur, harmonics=acc, amplitudes=amps)
        play(apply_envelope(w, env),
             f"Adding harmonic {n}  ({base * n:.0f} Hz)",
             f"Active harmonics: {acc}  |  amplitudes: {[round(a, 2) for a in amps]}")

    console.print("  [dim]Saving harmonic comparison plot → plots/[/]")
    waves = {
        "1 harmonic  (pure sine)": additive(440, 0.5, [1], [1]),
        "2 harmonics":             additive(440, 0.5, [1,2], [1,.5]),
        "4 harmonics":             additive(440, 0.5, [1,2,3,4], [1,.5,.33,.25]),
        "6 harmonics":             additive(440, 0.5, list(range(1,7)), [1/n for n in range(1,7)]),
    }
    plot_compare(waves, max_freq=3500, title="Harmonic Series Build-Up")
    done("Harmonics")


# ── Section 3: Classic waveforms ──────────────────────────────────────────

def demo_waveforms():
    section_banner("3", "Classic Synthesizer Waveforms")
    explainer("""
Classic synth waveforms are defined entirely by their [bold]harmonic content[/]:

  [bold cyan]Sine[/]      — fundamental only.                      Pure, smooth, flute-like.
  [bold green]Triangle[/]  — odd harmonics, 1/n² amplitude rolloff.  Soft, mellow, ocarina.
  [bold yellow]Square[/]    — odd harmonics, 1/n amplitude rolloff.   Hollow, nasal, clarinet.
  [bold red]Sawtooth[/]  — ALL harmonics, 1/n amplitude rolloff.  Bright, buzzy, brass/strings.

Same pitch (A4 = 440 Hz) — completely different texture.
""")
    dur = 1.5
    env = adsr_envelope(dur, attack=0.02, decay=0.1, sustain=0.7, release=0.3)
    waveforms = [
        ("Sine",     sine,     "cyan",   "Fundamental only — the purest tone"),
        ("Triangle", triangle, "green",  "Odd harmonics, 1/n² rolloff — soft"),
        ("Square",   square,   "yellow", "Odd harmonics, 1/n rolloff — hollow"),
        ("Sawtooth", sawtooth, "red",    "All harmonics, 1/n rolloff — bright, buzzy"),
    ]
    for name, fn, color, info in waveforms:
        play(apply_envelope(fn(440.0, dur), env),
             f"[{color}]{name} Wave[/]  @ 440 Hz", info)

    console.print("  [dim]Saving waveform comparison plot → plots/[/]")
    plot_compare(
        {name: fn(440.0, 0.5) for name, fn, *_ in waveforms},
        max_freq=6000,
        title="Classic Waveforms Compared",
    )
    done("Waveforms")


# ── Section 4: FM synthesis ───────────────────────────────────────────────

def demo_fm():
    section_banner("4", "FM Synthesis")
    explainer("""
[bold]Frequency Modulation[/] synthesis — invented by John Chowning (1973),
commercialised in the Yamaha DX7 (1983), one of the best-selling synths ever.

Instead of adding sines, we [bold yellow]modulate the carrier's frequency[/] with a modulator:

  output(t) = sin(2π·fc·t  +  [bold]β[/] · sin(2π·fm·t))

  [bold]mod_ratio[/] = fm / fc  → which harmonics appear  (1 = harmonic, 1.4 = bell-like)
  [bold]mod_index[/] = β        → richness/brightness     (0 = pure sine, >5 = dense)

Two humble sine waves produce enormously varied timbres with tiny parameter changes.
""")
    dur = 2.5
    env = adsr_envelope(dur, attack=0.02, decay=0.2, sustain=0.6, release=0.4)
    fc = 220.0
    presets = [
        ("Organ",    1.0, 1.0,  "ratio=1.0  index=1.0  — harmonically related, mild"),
        ("Brass",    1.0, 4.0,  "ratio=1.0  index=4.0  — bright, harmonic buzz"),
        ("Bell",     1.4, 5.0,  "ratio=1.4  index=5.0  — inharmonic partials"),
        ("Metallic", 3.5, 8.0,  "ratio=3.5  index=8.0  — dense metallic texture"),
        ("Gong",     0.5, 7.0,  "ratio=0.5  index=7.0  — deep resonant boom"),
    ]
    fm_waves = {}
    for name, ratio, idx, info in presets:
        w = apply_envelope(fm(fc, dur, mod_ratio=ratio, mod_index=idx), env)
        play(w, f"FM — [bold yellow]{name}[/]", info)
        fm_waves[name] = fm(fc, 0.5, mod_ratio=ratio, mod_index=idx)

    console.print("  [dim]Saving FM spectrum comparison plot → plots/[/]")
    plot_compare(fm_waves, max_freq=4000, title="FM Synthesis Presets")
    done("FM Synthesis")


# ── Section 5: ADSR ───────────────────────────────────────────────────────

def demo_adsr():
    section_banner("5", "ADSR Envelope")
    explainer("""
A raw oscillator plays at constant volume — unnatural and mechanical.
An [bold]ADSR envelope[/] shapes amplitude over the lifetime of a note:

  [bold green]A — Attack[/]   time to ramp 0 → peak after note-on   (fast = sharp, slow = swell)
  [bold yellow]D — Decay[/]    time to fall from peak → sustain level
  [bold cyan]S — Sustain[/]  amplitude while key is held            (0–1 scale)
  [bold red]R — Release[/]  time to fade sustain → 0 after note-off

Same sawtooth at A3 — four completely different characters:
""")
    dur = 2.0
    base_wave = sawtooth(note_to_freq("A", 3), dur)
    presets = [
        ("Piano",   "fast attack · low sustain · long release",
         dict(attack=0.005, decay=0.3,  sustain=0.2, release=0.8)),
        ("Organ",   "instant on/off · full sustain",
         dict(attack=0.005, decay=0.01, sustain=1.0, release=0.01)),
        ("Strings", "slow attack pad · high sustain · long release",
         dict(attack=0.4,   decay=0.1,  sustain=0.8, release=0.6)),
        ("Pluck",   "instant attack · zero sustain",
         dict(attack=0.002, decay=0.5,  sustain=0.0, release=0.1)),
    ]
    for name, info, params in presets:
        env = adsr_envelope(dur, **params)
        play(apply_envelope(base_wave, env),
             f"ADSR — [bold green]{name}[/]", info)
        console.print("  [dim]Saving envelope shape plot...[/]")
        plot_adsr(duration=dur, **params)
    done("ADSR Envelope")


# ── Section 6: Chords ─────────────────────────────────────────────────────

def demo_chords():
    section_banner("6", "Chords & Polyphony")
    explainer("""
Chords = multiple frequencies playing simultaneously.
We generate separate waves and [bold yellow]add them together[/], then normalise to [-1, 1].

Consonance vs dissonance is all about [bold]frequency ratios[/]:
  Perfect 5th  3 : 2  →  very stable and open
  Major 3rd    5 : 4  →  bright and pleasant
  Minor 2nd  16 : 15  →  tense, dissonant clash
""")
    dur = 2.5
    chord_presets = [
        ("C Major",      "bright, happy",  build_chord("C", 4, "major")),
        ("A Minor",      "darker, sadder", build_chord("A", 3, "minor")),
        ("A Major 7",    "jazzy, warm",    build_chord("A", 3, "major7")),
        ("A Minor 7",    "mellow jazz",    build_chord("A", 3, "minor7")),
        ("B Diminished", "tense, spooky",  build_chord("B", 3, "diminished")),
    ]
    for name, mood, freqs in chord_presets:
        w = chord(freqs, dur, wave_fn=sawtooth,
                  attack=0.02, decay=0.1, sustain=0.6, release=0.4)
        freq_str = "  ".join(f"{f:.0f} Hz" for f in freqs)
        play(w, f"[bold magenta]{name}[/]  [dim]— {mood}[/]", freq_str)
    done("Chords")


# ── Section 7: Melody ─────────────────────────────────────────────────────

def demo_melody():
    section_banner("7", "Melody Sequencing")
    explainer("""
A melody is a list of [bold](frequency, duration)[/] pairs played in sequence.
Frequency = 0 inserts a rest. Each note gets its own ADSR envelope.

Playing [bold yellow]Twinkle Twinkle Little Star[/] in C major — same notes, two timbres:
  [bold cyan]Triangle[/] — soft, toy-box quality
  [bold red]Sawtooth[/] — brighter, classic synthesizer feel
""")
    C4,D4,E4,F4,G4,A4 = [note_to_freq(n, 4) for n in ["C","D","E","F","G","A"]]
    q, h = 0.3, 0.55
    melody = [
        (C4,q),(C4,q),(G4,q),(G4,q),(A4,q),(A4,q),(G4,h),
        (F4,q),(F4,q),(E4,q),(E4,q),(D4,q),(D4,q),(C4,h),
        (G4,q),(G4,q),(F4,q),(F4,q),(E4,q),(E4,q),(D4,h),
        (G4,q),(G4,q),(F4,q),(F4,q),(E4,q),(E4,q),(D4,h),
        (C4,q),(C4,q),(G4,q),(G4,q),(A4,q),(A4,q),(G4,h),
        (F4,q),(F4,q),(E4,q),(E4,q),(D4,q),(D4,q),(C4,h),
    ]
    for name, fn, color in [("Triangle", triangle, "cyan"), ("Sawtooth", sawtooth, "red")]:
        w = sequence(melody, wave_fn=fn,
                     attack=0.02, decay=0.05, sustain=0.8, release=0.08)
        play(w, f"Twinkle Twinkle — [{color}]{name}[/]",
             f"C major  |  {len(melody)} notes  |  {len(w)/SAMPLE_RATE:.1f}s")
    done("Melody")


# ── Section 8: Synth patch ────────────────────────────────────────────────

def demo_synth_patch():
    section_banner("8", "Full Synthesizer Patch")
    explainer("""
A [bold]synthesizer patch[/] chains modules together:
  oscillator → envelope → (filter) → output

Three instruments built from scratch using only sine-wave math:

  [bold green]Warm Pad[/]    FM oscillator + slow-attack envelope  → lush, evolving texture
  [bold yellow]Plucky Bass[/]  Sawtooth + sharp ADSR               → percussive low end
  [bold cyan]Bell Chord[/]   Inharmonic FM (ratio=1.4) + long release → bright ring-out
""")
    dur = 3.0
    fc = note_to_freq("A", 3)

    w = fm(fc, dur, mod_ratio=1.0, mod_index=2.0)
    env = adsr_envelope(dur, attack=0.5, decay=0.3, sustain=0.7, release=0.8)
    play(apply_envelope(w, env),
         "[bold green]Warm Pad[/]",
         f"FM  |  fc={fc:.0f} Hz  |  ratio=1.0  |  index=2  |  slow 500ms attack")

    bass = [(note_to_freq("E", 2), 0.25)] * 4 + \
           [(note_to_freq("A", 2), 0.25)] * 2 + \
           [(note_to_freq("D", 3), 0.5)]
    w = sequence(bass, wave_fn=sawtooth,
                 attack=0.005, decay=0.15, sustain=0.1, release=0.05)
    play(w, "[bold yellow]Plucky Bass[/]",
         "Sawtooth  |  E2 × 4 → A2 × 2 → D3  |  sharp pluck envelope")

    c_major = build_chord("C", 4, "major")
    env_bell = adsr_envelope(dur, attack=0.005, decay=0.4, sustain=0.1, release=1.5)
    bell_waves = [apply_envelope(fm(f, dur, mod_ratio=1.4, mod_index=5.0), env_bell)
                  for f in c_major]
    play(mix(bell_waves),
         "[bold cyan]Bell Chord[/]",
         f"FM ratio=1.4  |  {[f'{f:.0f} Hz' for f in c_major]}  |  long 1.5s release")
    done("Synth Patch")


# ── Main loop ──────────────────────────────────────────────────────────────

SECTIONS = {
    "1": ("Sine Waves",    demo_sine_waves),
    "2": ("Harmonics",     demo_harmonics),
    "3": ("Waveforms",     demo_waveforms),
    "4": ("FM Synthesis",  demo_fm),
    "5": ("ADSR Envelope", demo_adsr),
    "6": ("Chords",        demo_chords),
    "7": ("Melody",        demo_melody),
    "8": ("Synth Patch",   demo_synth_patch),
}

if __name__ == "__main__":
    print_header()
    try:
        while True:
            print_menu()
            choice = console.input("[bold yellow]  ›[/] ").strip().lower()

            if choice == "q":
                console.print(
                    "\n  [dim]Plots saved to [bold]synth/plots/[/] — open them in Finder.[/]\n"
                    "  [bold yellow]  ♪  Goodbye!  ♪[/]\n"
                )
                break
            elif choice == "a":
                for key in "12345678":
                    name, fn = SECTIONS[key]
                    fn()
            elif choice in SECTIONS:
                _, fn = SECTIONS[choice]
                fn()
            else:
                console.print("\n  [red]Unknown choice — enter 1–8, A, or Q[/]\n")

    except KeyboardInterrupt:
        sd.stop()
        console.print("\n\n  [dim]Interrupted. Plots saved to synth/plots/[/]\n")
