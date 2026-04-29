"""
visualize.py — Waveform and spectrum plots for understanding synthesis.

Uses the Agg (non-interactive) backend so plots never open GUI windows —
no event-loop conflicts with sounddevice or input() on macOS.

All functions save the figure as a PNG and print the file path.
"""

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from synth import SAMPLE_RATE, adsr_envelope

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

_plot_counter = 0


def _save(fig, name: str) -> str:
    global _plot_counter
    _plot_counter += 1
    safe = name.lower().replace(" ", "_").replace("/", "-")
    path = os.path.join(PLOTS_DIR, f"{_plot_counter:02d}_{safe}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot saved] {path}")
    return path


def plot_waveform(wave: np.ndarray, title: str = "Waveform",
                  duration_ms: float = 10.0, sample_rate: int = SAMPLE_RATE):
    n = int(duration_ms * sample_rate / 1000)
    t_ms = np.linspace(0, duration_ms, n)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_ms, wave[:n], lw=1.2, color="steelblue")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.set_ylim(-1.2, 1.2)
    plt.tight_layout()
    _save(fig, title)


def plot_spectrum(wave: np.ndarray, title: str = "Frequency Spectrum",
                  max_freq: float = 5000.0, sample_rate: int = SAMPLE_RATE):
    N = len(wave)
    spectrum = np.abs(np.fft.rfft(wave)) / N
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)
    mask = freqs <= max_freq

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs[mask], spectrum[mask], lw=1.0, color="darkorange")
    ax.fill_between(freqs[mask], spectrum[mask], alpha=0.3, color="darkorange")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    plt.tight_layout()
    _save(fig, title)


def plot_waveform_and_spectrum(wave: np.ndarray, title: str = "",
                                duration_ms: float = 10.0,
                                max_freq: float = 5000.0,
                                sample_rate: int = SAMPLE_RATE):
    n = int(duration_ms * sample_rate / 1000)
    t_ms = np.linspace(0, duration_ms, n)

    N = len(wave)
    spectrum = np.abs(np.fft.rfft(wave)) / N
    freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)
    mask = freqs <= max_freq

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(title, fontsize=13)

    ax1.plot(t_ms, wave[:n], lw=1.2, color="steelblue")
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Waveform")
    ax1.axhline(0, color="gray", lw=0.5, ls="--")
    ax1.set_ylim(-1.2, 1.2)

    ax2.plot(freqs[mask], spectrum[mask], lw=1.0, color="darkorange")
    ax2.fill_between(freqs[mask], spectrum[mask], alpha=0.3, color="darkorange")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Spectrum (harmonics)")

    plt.tight_layout()
    _save(fig, title or "waveform_and_spectrum")


def plot_compare(waves: dict[str, np.ndarray], duration_ms: float = 10.0,
                 max_freq: float = 5000.0, sample_rate: int = SAMPLE_RATE,
                 title: str = "Waveform + Spectrum Comparison"):
    n_waves = len(waves)
    n = int(duration_ms * sample_rate / 1000)
    fig = plt.figure(figsize=(14, 3 * n_waves))
    gs = gridspec.GridSpec(n_waves, 2, figure=fig)
    colors = plt.cm.tab10.colors

    for i, (label, wave) in enumerate(waves.items()):
        t_ms = np.linspace(0, duration_ms, n)
        N = len(wave)
        spectrum = np.abs(np.fft.rfft(wave)) / N
        freqs = np.fft.rfftfreq(N, d=1.0 / sample_rate)
        mask = freqs <= max_freq
        c = colors[i % len(colors)]

        ax1 = fig.add_subplot(gs[i, 0])
        ax1.plot(t_ms, wave[:n], lw=1.2, color=c)
        ax1.set_ylabel(label, fontsize=10)
        ax1.set_ylim(-1.2, 1.2)
        ax1.axhline(0, color="gray", lw=0.5, ls="--")
        if i == n_waves - 1:
            ax1.set_xlabel("Time (ms)")
        ax1.set_title("Waveform" if i == 0 else "")

        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(freqs[mask], spectrum[mask], lw=1.0, color=c)
        ax2.fill_between(freqs[mask], spectrum[mask], alpha=0.3, color=c)
        if i == n_waves - 1:
            ax2.set_xlabel("Frequency (Hz)")
        ax2.set_title("Spectrum" if i == 0 else "")

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, title)


def plot_adsr(attack: float = 0.05, decay: float = 0.1,
              sustain: float = 0.7, release: float = 0.3,
              duration: float = 1.0, sample_rate: int = SAMPLE_RATE):
    env = adsr_envelope(duration, attack, decay, sustain, release, sample_rate)
    t = np.linspace(0, duration, len(env))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, env, lw=2, color="mediumseagreen")
    ax.fill_between(t, env, alpha=0.2, color="mediumseagreen")

    a = attack
    d = attack + decay
    r_start = duration - release

    ax.axvspan(0, a,        alpha=0.08, color="blue",   label=f"Attack  ({attack*1000:.0f} ms)")
    ax.axvspan(a, d,        alpha=0.08, color="orange", label=f"Decay   ({decay*1000:.0f} ms)")
    ax.axvspan(d, r_start,  alpha=0.08, color="green",  label=f"Sustain ({sustain:.0%})")
    ax.axvspan(r_start, duration, alpha=0.08, color="red", label=f"Release ({release*1000:.0f} ms)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("ADSR Envelope")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    title = f"ADSR_a{int(attack*1000)}ms_s{int(sustain*100)}pct"
    _save(fig, title)
