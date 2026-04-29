"""
piano.py — Terminal piano synthesizer, playable with the keyboard.

  Keyboard layout (C4–E5, two octaves):

     W    E         T    Y    U         O    P
   ┌───┐┌───┐     ┌───┐┌───┐┌───┐     ┌───┐┌───┐
   │C# ││D# │     │F# ││G# ││A# │     │C# ││D# │
 ┌─┘   └┬─  └──┬─┘   └┬─  └┬─  └──┬─┘   └┬─  └──┐
 │  C   │  D   │  E   │  F  │  G   │  A   │  B   │  ...
 │  A   │  S   │  D   │  F  │  G   │  H   │  J   │
 └──────┴──────┴──────┴─────┴──────┴──────┴──────┘

Controls:
  ← / →  : octave down / up          Z / X : octave down / up (alt)
  ↑ / ↓  : volume up / down
  1/2/3/4/5/6/7: waveform  Sine/Triangle/Square/Sawtooth/FM-Bell/FM-Bass/FM-Metal
  Q/ESC  : quit
"""

import curses
import threading
import time
import sys
import os
import collections

import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(__file__))
from notes import note_to_freq

# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────
SR   = 44100
WW   = 7    # white key width  (chars)
WH   = 9    # white key height (rows)
BW   = 5    # black key width  (chars)
BH   = 5    # black key height (rows)
NW   = 10   # number of white keys shown

WHITE_KEYS = [
    ('a', 'C'), ('s', 'D'), ('d', 'E'), ('f', 'F'), ('g', 'G'),
    ('h', 'A'), ('j', 'B'), ('k', 'C'), ('l', 'D'), (';', 'E'),
]
BLACK_KEYS = [
    ('w', 'C#', 0), ('e', 'D#', 1),
    ('t', 'F#', 3), ('y', 'G#', 4), ('u', 'A#', 5),
    ('o', 'C#', 7), ('p', 'D#', 8),
]

# key → (note_name, octave_offset_from_base)
KEY_NOTE_MAP = {
    'a':('C', 0), 'w':('C#',0), 's':('D', 0), 'e':('D#',0),
    'd':('E', 0), 'f':('F', 0), 't':('F#',0), 'g':('G', 0),
    'y':('G#',0), 'h':('A', 0), 'u':('A#',0), 'j':('B', 0),
    'k':('C', 1), 'o':('C#',1), 'l':('D', 1), 'p':('D#',1),
    ';':('E', 1),
}

WAVEFORMS = ['Sine', 'Triangle', 'Square', 'Sawtooth', 'FM-Bell', 'FM-Bass', 'FM-Metal']
WF_ICONS  = ['∿',    '△',        '⊓',      '⊿',       '🔔',      '🎸',      '⚙'       ]

# FM presets: (mod_ratio, mod_index)
FM_PRESETS = {
    'FM-Bell':  (3.5,  8.0),
    'FM-Bass':  (1.0,  3.5),
    'FM-Metal': (3.5, 14.0),
}

NOTE_COLORS = {
    'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'A': 6, 'B': 7,
}

# ─────────────────────────────────────────────────────────────
#  Black key x-position helper
# ─────────────────────────────────────────────────────────────
def bx(after_white_idx: int) -> int:
    """Column of a black key centred over the gap after white key N."""
    return (after_white_idx + 1) * WW - BW // 2 - 1


# ─────────────────────────────────────────────────────────────
#  Real-time polyphonic synth engine
# ─────────────────────────────────────────────────────────────
class SynthEngine:
    _DECAY = {
        'Sine':     0.9994,
        'Triangle': 0.9994,
        'Square':   0.9991,
        'Sawtooth': 0.9989,
        'FM-Bell':  0.9985,   # faster natural decay like a bell
        'FM-Bass':  0.9992,
        'FM-Metal': 0.9988,
    }

    def __init__(self):
        self.lock     = threading.Lock()
        self.notes    = {}          # key_id → state dict
        self.volume   = 0.65
        self.waveform = 'Sine'
        self.stream   = sd.OutputStream(
            samplerate=SR, channels=1,
            callback=self._callback,
            blocksize=256, dtype='float32',
        )
        self.stream.start()

    # ── waveform generators (operate on phase arrays) ──────────
    @staticmethod
    def _gen(wf: str, ph: np.ndarray, freq: float = 440.0) -> np.ndarray:
        if wf == 'Sine':
            return np.sin(ph)
        if wf == 'Square':
            return np.sign(np.sin(ph))
        if wf == 'Sawtooth':
            return 2.0 * (ph / (2 * np.pi) % 1.0) - 1.0
        if wf in FM_PRESETS:
            mod_ratio, mod_index = FM_PRESETS[wf]
            mod_ph = ph * mod_ratio
            modulator = mod_index * np.sin(mod_ph)
            return np.sin(ph + modulator)
        # Triangle
        p = ph / (2 * np.pi) % 1.0
        return 4.0 * np.abs(p - 0.5) - 1.0

    def _callback(self, outdata, frames, time_info, status):
        t   = np.arange(frames, dtype=np.float64) / SR
        buf = np.zeros(frames, dtype=np.float32)

        with self.lock:
            dead = []
            for kid, s in self.notes.items():
                ph   = 2 * np.pi * s['freq'] * t + s['phase']
                smp  = self._gen(s['wf'], ph, s['freq']).astype(np.float32)
                env  = s['amp'] * (s['decay'] ** np.arange(frames, dtype=np.float32))
                buf += smp * env
                s['amp']   *= float(s['decay'] ** frames)
                s['phase']  = float(ph[-1]) % (2 * np.pi)
                if s['amp'] < 4e-4:
                    dead.append(kid)
            for k in dead:
                del self.notes[k]

        np.tanh(buf, buf)           # soft clip
        buf *= self.volume
        outdata[:, 0] = buf

    def play(self, key_id: str, freq: float):
        with self.lock:
            self.notes[key_id] = {
                'freq':  freq,
                'phase': 0.0,
                'amp':   1.0,
                'decay': self._DECAY.get(self.waveform, 0.9993),
                'wf':    self.waveform,
            }

    def release(self, key_id: str):
        with self.lock:
            if key_id in self.notes:
                self.notes[key_id]['decay'] = 0.96

    def close(self):
        with self.lock:
            self.notes.clear()
        self.stream.stop()
        self.stream.close()


# ─────────────────────────────────────────────────────────────
#  Color pair IDs
# ─────────────────────────────────────────────────────────────
CP_WHITE       = 1   # white key normal
CP_WHITE_PRESS = 2   # white key pressed
CP_BLACK       = 3   # black key normal
CP_BLACK_PRESS = 4   # black key pressed
CP_HEADER      = 5   # top header bar
CP_STATUS      = 6   # status row
CP_BRIGHT      = 7   # bright label
CP_DIM         = 8   # dim borders
CP_GREEN       = 9   # positive indicator
CP_CYAN        = 10  # cyan accent
CP_NOTE        = 11  # note name pop
CP_HELP        = 12  # help text


def setup_colors():
    curses.start_color()
    curses.use_default_colors()
    bg = -1  # transparent background

    curses.init_pair(CP_WHITE,       curses.COLOR_BLACK,  curses.COLOR_WHITE)
    curses.init_pair(CP_WHITE_PRESS, curses.COLOR_BLACK,  curses.COLOR_YELLOW)
    curses.init_pair(CP_BLACK,       curses.COLOR_WHITE,  curses.COLOR_BLACK)
    curses.init_pair(CP_BLACK_PRESS, curses.COLOR_BLACK,  curses.COLOR_CYAN)
    curses.init_pair(CP_HEADER,      curses.COLOR_BLACK,  curses.COLOR_BLUE)
    curses.init_pair(CP_STATUS,      curses.COLOR_BLACK,  curses.COLOR_WHITE)
    curses.init_pair(CP_BRIGHT,      curses.COLOR_YELLOW, bg)
    curses.init_pair(CP_DIM,         curses.COLOR_WHITE,  bg)
    curses.init_pair(CP_GREEN,       curses.COLOR_GREEN,  bg)
    curses.init_pair(CP_CYAN,        curses.COLOR_CYAN,   bg)
    curses.init_pair(CP_NOTE,        curses.COLOR_YELLOW, bg)
    curses.init_pair(CP_HELP,        curses.COLOR_WHITE,  bg)


# ─────────────────────────────────────────────────────────────
#  Safe draw helpers
# ─────────────────────────────────────────────────────────────
def saddch(scr, y, x, ch, attr=0):
    try:
        scr.addch(y, x, ch, attr)
    except curses.error:
        pass


def saddstr(scr, y, x, s, attr=0):
    try:
        scr.addstr(y, x, s, attr)
    except curses.error:
        pass


# ─────────────────────────────────────────────────────────────
#  Piano key drawing
# ─────────────────────────────────────────────────────────────
def draw_white_key(scr, ky, kx, note: str, kb: str, pressed: bool):
    """Draw one white key, top-left at (ky, kx)."""
    body = curses.color_pair(CP_WHITE_PRESS if pressed else CP_WHITE)
    bord = curses.color_pair(CP_DIM)

    # ── top border ───────────────────────────────────────────
    saddch(scr, ky, kx, '┌', bord)
    for c in range(1, WW - 1):
        saddch(scr, ky, kx + c, '─', bord)
    saddch(scr, ky, kx + WW - 1, '┐', bord)

    # ── body rows ────────────────────────────────────────────
    for r in range(1, WH - 1):
        saddch(scr, ky + r, kx, '│', bord)
        for c in range(1, WW - 1):
            saddch(scr, ky + r, kx + c, ' ', body)
        saddch(scr, ky + r, kx + WW - 1, '│', bord)

    # ── note label (row WH-4) ────────────────────────────────
    lbl = f' {note:<{WW - 3}} '
    saddstr(scr, ky + WH - 4, kx + 1, lbl[:WW - 2], body | curses.A_BOLD)

    # ── keyboard key label (row WH-3) ────────────────────────
    kb_attr = curses.color_pair(CP_WHITE_PRESS) | curses.A_BOLD if pressed \
              else curses.color_pair(CP_WHITE) | curses.A_BOLD
    kb_lbl = f'[{kb.upper()}]'
    pad    = (WW - 2 - len(kb_lbl)) // 2
    saddstr(scr, ky + WH - 3, kx + 1, ' ' * (WW - 2), body)
    saddstr(scr, ky + WH - 3, kx + 1 + pad, kb_lbl, kb_attr)

    # ── bottom border ────────────────────────────────────────
    saddch(scr, ky + WH - 1, kx, '└', bord)
    for c in range(1, WW - 1):
        saddch(scr, ky + WH - 1, kx + c, '─', bord)
    saddch(scr, ky + WH - 1, kx + WW - 1, '┘', bord)


def draw_black_key(scr, ky, kx, note: str, kb: str, pressed: bool):
    """Draw one black key (drawn on top of white keys)."""
    body = curses.color_pair(CP_BLACK_PRESS if pressed else CP_BLACK)
    bord = curses.color_pair(CP_BLACK)
    if pressed:
        body |= curses.A_BOLD

    # ── top border ───────────────────────────────────────────
    saddch(scr, ky, kx, '┌', bord)
    for c in range(1, BW - 1):
        saddch(scr, ky, kx + c, '─', bord)
    saddch(scr, ky, kx + BW - 1, '┐', bord)

    # ── body rows ────────────────────────────────────────────
    for r in range(1, BH - 1):
        saddch(scr, ky + r, kx, '│', bord)
        for c in range(1, BW - 1):
            saddch(scr, ky + r, kx + c, ' ', body)
        saddch(scr, ky + r, kx + BW - 1, '│', bord)

    # ── note + kb labels ─────────────────────────────────────
    inner = BW - 2
    saddstr(scr, ky + 1, kx + 1, f'{note:<{inner}}'[:inner], body | curses.A_BOLD)
    kb_lbl = f'[{kb.upper()}]'[:inner]
    saddstr(scr, ky + 2, kx + 1, f'{kb_lbl:<{inner}}'[:inner], body | curses.A_BOLD)

    # ── bottom border ────────────────────────────────────────
    saddch(scr, ky + BH - 1, kx, '└', bord)
    for c in range(1, BW - 1):
        saddch(scr, ky + BH - 1, kx + c, '─', bord)
    saddch(scr, ky + BH - 1, kx + BW - 1, '┘', bord)


def draw_piano(scr, py: int, px: int, pressed: set, base_oct: int):
    """Draw the full piano at screen position (py, px)."""
    # White keys first
    for wi, (kb, note) in enumerate(WHITE_KEYS):
        oct_off  = 1 if wi >= 7 else 0
        full     = f'{note}{base_oct + oct_off}'
        draw_white_key(scr, py, px + wi * WW, full, kb, kb in pressed)

    # Black keys on top
    for kb, note, after_wi in BLACK_KEYS:
        oct_off = 1 if after_wi >= 7 else 0
        full    = f'{note}{base_oct + oct_off}'
        draw_black_key(scr, py, px + bx(after_wi), full, kb, kb in pressed)


# ─────────────────────────────────────────────────────────────
#  Header & status bars
# ─────────────────────────────────────────────────────────────
def draw_header(scr, W: int):
    title = '  ♩ ♪ ♫   PYTHON SYNTHESIZER   ♫ ♪ ♩  '
    pad   = max(0, (W - len(title)) // 2)
    saddstr(scr, 0, 0, ' ' * W, curses.color_pair(CP_HEADER))
    saddstr(scr, 0, pad, title, curses.color_pair(CP_HEADER) | curses.A_BOLD)


def draw_controls(scr, y: int, W: int, wf_idx: int, base_oct: int, volume: float):
    saddstr(scr, y, 0, ' ' * W, curses.color_pair(CP_STATUS))

    # Waveforms
    x = 2
    saddstr(scr, y, x, 'Wave: ', curses.color_pair(CP_STATUS))
    x += 6
    for i, (name, icon) in enumerate(zip(WAVEFORMS, WF_ICONS)):
        lbl  = f' {icon} {name[:4]} '
        attr = (curses.color_pair(CP_WHITE_PRESS) | curses.A_BOLD
                if i == wf_idx else curses.color_pair(CP_STATUS))
        saddstr(scr, y, x, f'[{i+1}]', curses.color_pair(CP_STATUS))
        x += 3
        saddstr(scr, y, x, lbl, attr)
        x += len(lbl) + 1

    # Octave
    x += 2
    saddstr(scr, y, x, f'Oct: ◄ ', curses.color_pair(CP_STATUS))
    x += 7
    saddstr(scr, y, x, str(base_oct),
            curses.color_pair(CP_STATUS) | curses.A_BOLD)
    x += 1
    saddstr(scr, y, x, ' ►  ', curses.color_pair(CP_STATUS))
    x += 4

    # Volume bar
    saddstr(scr, y, x, 'Vol: ', curses.color_pair(CP_STATUS))
    x += 5
    filled  = int(volume * 10)
    vol_bar = '█' * filled + '░' * (10 - filled)
    saddstr(scr, y, x, vol_bar, curses.color_pair(CP_STATUS) | curses.A_BOLD)
    x += 11
    saddstr(scr, y, x, f'{int(volume*100):3d}%', curses.color_pair(CP_STATUS))


def draw_note_display(scr, y: int, W: int,
                      last_note: str, last_freq: float,
                      history: list, wf_name: str):
    scr.move(y, 0)
    scr.clrtoeol()
    scr.move(y + 1, 0)
    scr.clrtoeol()

    if last_note:
        saddstr(scr, y, 2, '♪  ', curses.color_pair(CP_BRIGHT) | curses.A_BOLD)
        saddstr(scr, y, 5, last_note, curses.color_pair(CP_NOTE) | curses.A_BOLD)
        saddstr(scr, y, 5 + len(last_note), f'  {last_freq:.2f} Hz',
                curses.color_pair(CP_CYAN))
        saddstr(scr, y, 5 + len(last_note) + 12, f'  {wf_name}',
                curses.color_pair(CP_DIM))

    if history:
        saddstr(scr, y + 1, 2, 'History: ', curses.color_pair(CP_DIM))
        x = 11
        for note in list(history)[-14:]:
            saddstr(scr, y + 1, x, note + ' ',
                    curses.color_pair(CP_GREEN) | curses.A_BOLD)
            x += len(note) + 1


def draw_help(scr, y: int, W: int):
    scr.move(y, 0)
    scr.clrtoeol()
    hint = ('  ← → or Z X : octave    ↑ ↓ : volume    '
            '1-7 : waveform (Sine/Tri/Sq/Saw/FM-Bell/FM-Bass/FM-Metal)    Q/ESC : quit')
    saddstr(scr, y, 0, hint[:W], curses.color_pair(CP_DIM))


def draw_separator(scr, y: int, W: int):
    saddstr(scr, y, 0, '─' * W, curses.color_pair(CP_DIM))


# ─────────────────────────────────────────────────────────────
#  Main curses loop
# ─────────────────────────────────────────────────────────────
def main(stdscr):
    setup_colors()
    curses.curs_set(0)
    stdscr.timeout(40)          # 25 fps, non-blocking getch
    stdscr.keypad(True)

    engine   = SynthEngine()
    state    = {
        'wf_idx':   0,
        'base_oct': 4,
        'volume':   0.65,
        'last_note': '',
        'last_freq':  0.0,
        'history':  collections.deque(maxlen=20),
        'pressed':  set(),          # keys visually highlighted
        'press_ts': {},             # key → time pressed
    }
    PRESS_SHOW_S = 0.5             # how long pressed highlight lasts

    def redraw():
        H, W = stdscr.getmaxyx()
        piano_width = NW * WW + 1
        px = max(0, (W - piano_width) // 2)
        piano_y = 4

        draw_header(stdscr, W)
        draw_separator(stdscr, 1, W)
        draw_controls(stdscr, 2, W,
                      state['wf_idx'], state['base_oct'], state['volume'])
        draw_separator(stdscr, 3, W)
        draw_piano(stdscr, piano_y, px, state['pressed'], state['base_oct'])
        draw_separator(stdscr, piano_y + WH, W)
        draw_note_display(stdscr, piano_y + WH + 1, W,
                          state['last_note'], state['last_freq'],
                          list(state['history']),
                          WAVEFORMS[state['wf_idx']])
        draw_separator(stdscr, piano_y + WH + 3, W)
        draw_help(stdscr, piano_y + WH + 4, W)
        stdscr.refresh()

    redraw()

    while True:
        # ── expire old pressed highlights ────────────────────
        now   = time.time()
        stale = [k for k, t in state['press_ts'].items()
                 if now - t > PRESS_SHOW_S]
        for k in stale:
            state['pressed'].discard(k)
            del state['press_ts'][k]
            engine.release(k)

        # ── get input ────────────────────────────────────────
        key = stdscr.getch()

        if key == -1:
            redraw()
            continue

        # Quit
        if key in (ord('q'), ord('Q'), 27):
            break

        # Waveform
        if ord('1') <= key <= ord('1') + len(WAVEFORMS) - 1:
            state['wf_idx']       = key - ord('1')
            engine.waveform       = WAVEFORMS[state['wf_idx']]
            redraw()
            continue

        # Octave shift
        if key in (curses.KEY_LEFT, ord('z'), ord('Z')):
            state['base_oct'] = max(1, state['base_oct'] - 1)
            redraw()
            continue
        if key in (curses.KEY_RIGHT, ord('x'), ord('X')):
            state['base_oct'] = min(7, state['base_oct'] + 1)
            redraw()
            continue

        # Volume
        if key == curses.KEY_UP:
            state['volume'] = min(1.0, round(state['volume'] + 0.05, 2))
            engine.volume   = state['volume']
            redraw()
            continue
        if key == curses.KEY_DOWN:
            state['volume'] = max(0.0, round(state['volume'] - 0.05, 2))
            engine.volume   = state['volume']
            redraw()
            continue

        # Piano key press
        if 32 <= key <= 126:
            ch = chr(key).lower()
            if ch in KEY_NOTE_MAP:
                note_name, oct_off = KEY_NOTE_MAP[ch]
                octave = state['base_oct'] + oct_off
                freq   = note_to_freq(note_name, octave)
                full   = f'{note_name}{octave}'

                engine.play(ch, freq)

                state['pressed'].add(ch)
                state['press_ts'][ch] = time.time()
                state['last_note'] = full
                state['last_freq'] = freq
                state['history'].append(full)

        redraw()

    engine.close()


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
    print('\n  Goodbye!  ♪\n')
