#!/usr/bin/env python3
"""
chtptsgbaemu.py

A single-file, blue-on-black Tkinter Game Boy Advance emulator workbench.

Run:
    python chtptsgbaemu.py

Optional acceleration:
    pip install cython
    python chtptsgbaemu.py

This file keeps the project self-contained.  When Cython is available it builds a
small temporary helper module for hot paths such as ROM checksuming and RGB frame
rendering.  If a compiler is not available, the pure-Python core is used.

Scope note:
    This is a much stronger workbench/core scaffold than the starter file: it has
    a 60 Hz frame scheduler, faster framebuffer presentation, a larger ARM/THUMB
    interpreter subset, memory mapped I/O, input register handling, DMA copies,
    basic BIOS SWI HLE, and bitmap/text-background PPU rendering.  It is still a
    compact educational emulator, not a full replacement for a mature GBA core.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox


# -----------------------------------------------------------------------------
# Constants and theme
# -----------------------------------------------------------------------------

APP_TITLE = "chtptsgbaemu - 60 FPS GBA Workbench"
BLACK = "#000000"
PANEL_BLACK = "#030912"
DEEP_BLUE = "#001a33"
MID_BLUE = "#004a99"
BLUE = "#008cff"
BRIGHT_BLUE = "#33b5ff"
DIM_BLUE = "#005a9e"
TEXT_BLUE = "#40c4ff"
WARNING_BLUE = "#79d8ff"
FONT_MONO = ("Consolas", 10)
FONT_MONO_SMALL = ("Consolas", 9)
FONT_MONO_BIG = ("Consolas", 14, "bold")

SCREEN_W = 240
SCREEN_H = 160
SCREEN_SCALE = 3
CANVAS_W = SCREEN_W * SCREEN_SCALE
CANVAS_H = SCREEN_H * SCREEN_SCALE
TARGET_FPS = 60.0
TARGET_FRAME_TIME = 1.0 / TARGET_FPS
GBA_MASTER_HZ = 16_777_216
GBA_FRAME_HZ = 59.7275005696
GBA_CYCLES_PER_FRAME = int(GBA_MASTER_HZ / GBA_FRAME_HZ)  # about 280,896

BIOS_START = 0x00000000
EWRAM_START = 0x02000000
IWRAM_START = 0x03000000
IO_START = 0x04000000
PAL_START = 0x05000000
VRAM_START = 0x06000000
OAM_START = 0x07000000
ROM_START = 0x08000000
SRAM_START = 0x0E000000

CPSR_T = 1 << 5
FLAG_N = 1 << 31
FLAG_Z = 1 << 30
FLAG_C = 1 << 29
FLAG_V = 1 << 28
MODE_SYSTEM = 0x1F

KEY_BITS = {
    "A": 0,
    "B": 1,
    "SELECT": 2,
    "START": 3,
    "RIGHT": 4,
    "LEFT": 5,
    "UP": 6,
    "DOWN": 7,
    "R": 8,
    "L": 9,
}


# -----------------------------------------------------------------------------
# Optional Cython acceleration
# -----------------------------------------------------------------------------

CYTHON_SOURCE = rf'''
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

cdef int W = {SCREEN_W}
cdef int H = {SCREEN_H}

cpdef int header_checksum(bytes data):
    cdef Py_ssize_t i
    cdef unsigned int s = 0
    if len(data) < 0xBE:
        return -1
    for i in range(0xA0, 0xBD):
        s += data[i]
    return (-(s + 0x19)) & 0xFF

cpdef unsigned int rom_fold(bytes data, int frame, int limit):
    cdef Py_ssize_t i
    cdef Py_ssize_t n = len(data)
    cdef unsigned int h = 2166136261
    cdef Py_ssize_t idx
    if n <= 0:
        return <unsigned int>frame
    if limit <= 0 or limit > n:
        limit = n
    for i in range(limit):
        idx = (i * 37 + frame * 131) % n
        h = (h ^ data[idx]) * 16777619
    return h

cdef inline unsigned char expand5(unsigned int v):
    v &= 31
    return <unsigned char>((v << 3) | (v >> 2))

cpdef bytes render_mode3(unsigned char[:] vram):
    cdef int x, y, off, out_i
    cdef unsigned int pix
    cdef bytearray out = bytearray(W * H * 3)
    out_i = 0
    for y in range(H):
        for x in range(W):
            off = ((y * W) + x) * 2
            if off + 1 < vram.shape[0]:
                pix = vram[off] | (vram[off + 1] << 8)
                out[out_i] = expand5(pix)
                out[out_i + 1] = expand5(pix >> 5)
                out[out_i + 2] = expand5(pix >> 10)
            out_i += 3
    return bytes(out)

cpdef bytes render_mode4(unsigned char[:] vram, unsigned char[:] pal, int frame_select):
    cdef int x, y, src, out_i, poff
    cdef unsigned int idx, pix
    cdef int base = 0xA000 if frame_select else 0
    cdef bytearray out = bytearray(W * H * 3)
    out_i = 0
    for y in range(H):
        for x in range(W):
            src = base + (y * W) + x
            if src < vram.shape[0]:
                idx = vram[src]
                poff = idx * 2
                if poff + 1 < pal.shape[0]:
                    pix = pal[poff] | (pal[poff + 1] << 8)
                    out[out_i] = expand5(pix)
                    out[out_i + 1] = expand5(pix >> 5)
                    out[out_i + 2] = expand5(pix >> 10)
            out_i += 3
    return bytes(out)

cpdef bytes render_mode5(unsigned char[:] vram, int frame_select):
    cdef int x, y, src, out_i
    cdef unsigned int pix
    cdef int base = 0xA000 if frame_select else 0
    cdef bytearray out = bytearray(W * H * 3)
    out_i = 0
    for y in range(H):
        for x in range(W):
            if x < 160 and y < 128:
                src = base + ((y * 160) + x) * 2
                if src + 1 < vram.shape[0]:
                    pix = vram[src] | (vram[src + 1] << 8)
                    out[out_i] = expand5(pix)
                    out[out_i + 1] = expand5(pix >> 5)
                    out[out_i + 2] = expand5(pix >> 10)
            out_i += 3
    return bytes(out)

cpdef bytes render_viz(bytes rom, int frame, unsigned int pc, int keys_mask):
    cdef int x, y, i, idx, n, blue, green, glow, band
    cdef unsigned int seed = 2166136261 ^ (<unsigned int>frame * 16777619) ^ pc
    cdef bytearray out = bytearray(W * H * 3)
    n = len(rom)
    i = 0
    if n <= 0:
        for y in range(H):
            for x in range(W):
                glow = (x * x + y * 13 + frame * 5) & 255
                out[i] = 0
                out[i + 1] = <unsigned char>(8 + (glow >> 5))
                out[i + 2] = <unsigned char>(24 + (glow >> 2))
                i += 3
        return bytes(out)
    for y in range(H):
        band = ((y + frame) // 8) & 1
        for x in range(W):
            idx = (x * 37 + y * 91 + frame * 53 + (pc & 4095)) % n
            seed = (seed ^ rom[idx]) * 16777619
            blue = 24 + ((rom[idx] ^ (seed >> 8) ^ (x * 3) ^ y) & 0xB7)
            if blue > 255:
                blue = 255
            green = 4 + (((seed >> 24) + band * 18 + (keys_mask * 3)) & 31)
            out[i] = 0
            out[i + 1] = <unsigned char>green
            out[i + 2] = <unsigned char>blue
            i += 3
    return bytes(out)
'''


def python_header_checksum(data: bytes) -> int:
    if len(data) < 0xBE:
        return -1
    return (-(sum(data[0xA0:0xBD]) + 0x19)) & 0xFF


def python_rom_fold(data: bytes, frame: int, limit: int) -> int:
    if not data:
        return frame & 0xFFFFFFFF
    n = len(data)
    if limit <= 0 or limit > n:
        limit = n
    h = 2166136261
    for i in range(limit):
        idx = (i * 37 + frame * 131) % n
        h = ((h ^ data[idx]) * 16777619) & 0xFFFFFFFF
    return h


def expand5(v: int) -> int:
    v &= 31
    return (v << 3) | (v >> 2)


def rgb555_to_tuple(value: int) -> Tuple[int, int, int]:
    return expand5(value), expand5(value >> 5), expand5(value >> 10)


def python_render_mode3(vram: bytearray) -> bytes:
    out = bytearray(SCREEN_W * SCREEN_H * 3)
    oi = 0
    for y in range(SCREEN_H):
        base = y * SCREEN_W * 2
        for x in range(SCREEN_W):
            off = base + x * 2
            if off + 1 < len(vram):
                pix = vram[off] | (vram[off + 1] << 8)
                out[oi], out[oi + 1], out[oi + 2] = rgb555_to_tuple(pix)
            oi += 3
    return bytes(out)


def python_render_mode4(vram: bytearray, pal: bytearray, frame_select: int) -> bytes:
    out = bytearray(SCREEN_W * SCREEN_H * 3)
    base = 0xA000 if frame_select else 0
    oi = 0
    for y in range(SCREEN_H):
        row = base + y * SCREEN_W
        for x in range(SCREEN_W):
            src = row + x
            if src < len(vram):
                poff = vram[src] * 2
                if poff + 1 < len(pal):
                    pix = pal[poff] | (pal[poff + 1] << 8)
                    out[oi], out[oi + 1], out[oi + 2] = rgb555_to_tuple(pix)
            oi += 3
    return bytes(out)


def python_render_mode5(vram: bytearray, frame_select: int) -> bytes:
    out = bytearray(SCREEN_W * SCREEN_H * 3)
    base = 0xA000 if frame_select else 0
    oi = 0
    for y in range(SCREEN_H):
        for x in range(SCREEN_W):
            if x < 160 and y < 128:
                off = base + (y * 160 + x) * 2
                if off + 1 < len(vram):
                    pix = vram[off] | (vram[off + 1] << 8)
                    out[oi], out[oi + 1], out[oi + 2] = rgb555_to_tuple(pix)
            oi += 3
    return bytes(out)


def python_render_viz(rom: bytes, frame: int, pc: int, keys_mask: int) -> bytes:
    out = bytearray(SCREEN_W * SCREEN_H * 3)
    seed = (2166136261 ^ (frame * 16777619) ^ pc) & 0xFFFFFFFF
    i = 0
    n = len(rom)
    if not n:
        for y in range(SCREEN_H):
            for x in range(SCREEN_W):
                glow = (x * x + y * 13 + frame * 5) & 255
                out[i] = 0
                out[i + 1] = 8 + (glow >> 5)
                out[i + 2] = 24 + (glow >> 2)
                i += 3
        return bytes(out)
    for y in range(SCREEN_H):
        band = ((y + frame) // 8) & 1
        for x in range(SCREEN_W):
            idx = (x * 37 + y * 91 + frame * 53 + (pc & 4095)) % n
            seed = ((seed ^ rom[idx]) * 16777619) & 0xFFFFFFFF
            blue = min(255, 24 + ((rom[idx] ^ (seed >> 8) ^ (x * 3) ^ y) & 0xB7))
            green = 4 + (((seed >> 24) + band * 18 + keys_mask * 3) & 31)
            out[i] = 0
            out[i + 1] = green
            out[i + 2] = blue
            i += 3
    return bytes(out)


@dataclass
class Accelerator:
    available: bool = False
    error: str = ""
    header_checksum: Callable[[bytes], int] = python_header_checksum
    rom_fold: Callable[[bytes, int, int], int] = python_rom_fold
    render_mode3: Callable[[bytearray], bytes] = python_render_mode3
    render_mode4: Callable[[bytearray, bytearray, int], bytes] = python_render_mode4
    render_mode5: Callable[[bytearray, int], bytes] = python_render_mode5
    render_viz: Callable[[bytes, int, int, int], bytes] = python_render_viz


def load_cython_accelerator() -> Accelerator:
    try:
        import pyximport  # type: ignore

        cache_root = Path(tempfile.gettempdir()) / "chtptsgbaemu_cython_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        pyx_path = cache_root / "chtptsgbaemu_fast.pyx"
        if not pyx_path.exists() or pyx_path.read_text(encoding="utf-8") != CYTHON_SOURCE:
            pyx_path.write_text(CYTHON_SOURCE, encoding="utf-8")

        if str(cache_root) not in sys.path:
            sys.path.insert(0, str(cache_root))

        pyximport.install(
            build_dir=str(cache_root / "build"),
            language_level=3,
            inplace=False,
        )
        import chtptsgbaemu_fast  # type: ignore

        return Accelerator(
            available=True,
            error="",
            header_checksum=chtptsgbaemu_fast.header_checksum,
            rom_fold=chtptsgbaemu_fast.rom_fold,
            render_mode3=chtptsgbaemu_fast.render_mode3,
            render_mode4=chtptsgbaemu_fast.render_mode4,
            render_mode5=chtptsgbaemu_fast.render_mode5,
            render_viz=chtptsgbaemu_fast.render_viz,
        )
    except Exception as exc:
        return Accelerator(
            available=False,
            error=f"{type(exc).__name__}: {exc}",
        )


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def clean_ascii(raw: bytes) -> str:
    text = raw.decode("ascii", errors="replace")
    text = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in text)
    return text.rstrip(" \x00") or "<blank>"


def u32(value: int) -> int:
    return value & 0xFFFFFFFF


def s32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value & 0x80000000 else value


def ror32(value: int, amount: int) -> int:
    amount &= 31
    value &= 0xFFFFFFFF
    if amount == 0:
        return value
    return ((value >> amount) | (value << (32 - amount))) & 0xFFFFFFFF


def sign_extend(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (value & (sign - 1)) - (value & sign)


def cond_name(cond: int) -> str:
    return [
        "EQ", "NE", "CS", "CC", "MI", "PL", "VS", "VC",
        "HI", "LS", "GE", "LT", "GT", "LE", "AL", "NV",
    ][cond & 0xF]


def make_ppm(rgb: bytes) -> bytes:
    return f"P6\n{SCREEN_W} {SCREEN_H}\n255\n".encode("ascii") + rgb


# -----------------------------------------------------------------------------
# ROM header helpers
# -----------------------------------------------------------------------------


@dataclass
class GBAHeader:
    title: str = "<no rom>"
    game_code: str = "----"
    maker_code: str = "--"
    fixed_value: int = 0
    unit_code: int = 0
    device_type: int = 0
    version: int = 0
    checksum: int = 0
    expected_checksum: int = 0
    checksum_ok: bool = False
    rom_size: int = 0
    save_type: str = "unknown"

    @classmethod
    def from_rom(cls, data: bytes, checksum_func: Callable[[bytes], int]) -> "GBAHeader":
        if len(data) < 0xBE:
            return cls(title="<file too small>", rom_size=len(data), save_type=detect_save_type(data))
        expected = checksum_func(data)
        actual = data[0xBD]
        return cls(
            title=clean_ascii(data[0xA0:0xAC]),
            game_code=clean_ascii(data[0xAC:0xB0]),
            maker_code=clean_ascii(data[0xB0:0xB2]),
            fixed_value=data[0xB2],
            unit_code=data[0xB3],
            device_type=data[0xB4],
            version=data[0xBC],
            checksum=actual,
            expected_checksum=expected,
            checksum_ok=(actual == expected),
            rom_size=len(data),
            save_type=detect_save_type(data),
        )

    def lines(self) -> List[str]:
        mib = self.rom_size / (1024 * 1024) if self.rom_size else 0.0
        return [
            f"TITLE       : {self.title}",
            f"GAME CODE   : {self.game_code}",
            f"MAKER CODE  : {self.maker_code}",
            f"ROM SIZE    : {self.rom_size:,} bytes ({mib:.2f} MiB)",
            f"SAVE TYPE   : {self.save_type}",
            f"FIXED 0xB2  : 0x{self.fixed_value:02X} {'OK' if self.fixed_value == 0x96 else 'WARN'}",
            f"UNIT CODE   : 0x{self.unit_code:02X}",
            f"DEVICE TYPE : 0x{self.device_type:02X}",
            f"VERSION     : {self.version}",
            f"CHECKSUM    : 0x{self.checksum:02X} expected 0x{self.expected_checksum:02X} {'OK' if self.checksum_ok else 'WARN'}",
        ]


def detect_save_type(data: bytes) -> str:
    # Common ASCII markers used by GBA ROMs for save hardware selection.
    markers = [
        b"EEPROM_V", b"SRAM_V", b"FLASH_V", b"FLASH512_V", b"FLASH1M_V",
        b"SIIRTC_V", b"GPIO", b"RTC_V",
    ]
    found = []
    for marker in markers:
        if marker in data:
            found.append(marker.decode("ascii", errors="ignore"))
    return ", ".join(found) if found else "not detected"


# -----------------------------------------------------------------------------
# GBA memory, DMA, and IO scaffold
# -----------------------------------------------------------------------------


class GBAMemory:
    """Compact GBA memory bus with useful mirroring and immediate DMA handling."""

    def __init__(self) -> None:
        self.bios = bytearray(16 * 1024)
        self.rom: bytes = b""
        self.ewram = bytearray(256 * 1024)
        self.iwram = bytearray(32 * 1024)
        self.io = bytearray(1024)
        self.palette = bytearray(1024)
        self.vram = bytearray(96 * 1024)
        self.oam = bytearray(1024)
        self.sram = bytearray(64 * 1024)
        self.last_open_bus = 0
        self.display_touched = False
        self.reset_io()

    def reset_io(self) -> None:
        self.io[:] = b"\x00" * len(self.io)
        self.write16_raw(0x000, 0x0080)  # DISPCNT forced blank bit-style friendly default.
        self.write16_raw(0x004, 0x0000)  # DISPSTAT
        self.write16_raw(0x006, 0x0000)  # VCOUNT
        self.write16_raw(0x130, 0x03FF)  # KEYINPUT, active-low, all released.
        self.display_touched = False

    def load_rom(self, data: bytes) -> None:
        self.rom = data

    def reset_ram(self) -> None:
        self.ewram[:] = b"\x00" * len(self.ewram)
        self.iwram[:] = b"\x00" * len(self.iwram)
        self.palette[:] = b"\x00" * len(self.palette)
        self.vram[:] = b"\x00" * len(self.vram)
        self.oam[:] = b"\x00" * len(self.oam)
        self.reset_io()

    def update_key_input(self, keys: Dict[str, bool]) -> None:
        value = 0x03FF
        for name, bit in KEY_BITS.items():
            if keys.get(name, False):
                value &= ~(1 << bit)
        self.write16_raw(0x130, value)

    def set_vcount(self, line: int) -> None:
        self.write16_raw(0x006, line & 0x1FF)
        dispstat = self.read16(IO_START + 0x004)
        if line >= 160:
            dispstat |= 1
        else:
            dispstat &= ~1
        if line == ((dispstat >> 8) & 0xFF):
            dispstat |= 4
        else:
            dispstat &= ~4
        self.write16_raw(0x004, dispstat)

    def write16_raw(self, io_offset: int, value: int) -> None:
        io_offset &= 0x3FE
        self.io[io_offset] = value & 0xFF
        if io_offset + 1 < len(self.io):
            self.io[io_offset + 1] = (value >> 8) & 0xFF

    def _region(self, addr: int) -> Tuple[Optional[bytearray], int, bool]:
        addr &= 0xFFFFFFFF
        top = addr & 0xFF000000
        if top == BIOS_START:
            return self.bios, addr & (len(self.bios) - 1), False
        if top == EWRAM_START:
            return self.ewram, addr & (len(self.ewram) - 1), True
        if top == IWRAM_START:
            return self.iwram, addr & (len(self.iwram) - 1), True
        if top == IO_START:
            return self.io, addr & (len(self.io) - 1), True
        if top == PAL_START:
            return self.palette, addr & (len(self.palette) - 1), True
        if top == VRAM_START:
            # GBA VRAM has awkward mirrors; modulo gives stable workbench behavior.
            return self.vram, (addr - VRAM_START) % len(self.vram), True
        if top == OAM_START:
            return self.oam, addr & (len(self.oam) - 1), True
        if top == SRAM_START:
            return self.sram, addr & (len(self.sram) - 1), True
        return None, 0, False

    def read8(self, addr: int) -> int:
        addr &= 0xFFFFFFFF
        if 0x08000000 <= addr <= 0x0DFFFFFF:
            if not self.rom:
                return 0xFF
            idx = (addr - ROM_START) % len(self.rom)
            self.last_open_bus = self.rom[idx]
            return self.last_open_bus
        region, off, _writable = self._region(addr)
        if region is None:
            return self.last_open_bus & 0xFF
        value = region[off]
        self.last_open_bus = value
        return value

    def read16(self, addr: int) -> int:
        addr &= 0xFFFFFFFE
        value = self.read8(addr) | (self.read8(addr + 1) << 8)
        self.last_open_bus = value
        return value

    def read32(self, addr: int) -> int:
        addr &= 0xFFFFFFFC
        value = (
            self.read8(addr)
            | (self.read8(addr + 1) << 8)
            | (self.read8(addr + 2) << 16)
            | (self.read8(addr + 3) << 24)
        )
        self.last_open_bus = value
        return value

    def read32_rotate(self, addr: int) -> int:
        aligned = addr & 0xFFFFFFFC
        value = self.read32(aligned)
        return ror32(value, (addr & 3) * 8)

    def read16_signed(self, addr: int) -> int:
        value = self.read16(addr)
        return sign_extend(value, 16) & 0xFFFFFFFF

    def read8_signed(self, addr: int) -> int:
        value = self.read8(addr)
        return sign_extend(value, 8) & 0xFFFFFFFF

    def write8(self, addr: int, value: int) -> None:
        addr &= 0xFFFFFFFF
        region, off, writable = self._region(addr)
        if region is not None and writable:
            region[off] = value & 0xFF
            if (addr & 0xFF000000) == IO_START:
                self._after_io_write(off & 0x3FF)

    def write16(self, addr: int, value: int) -> None:
        addr &= 0xFFFFFFFE
        region, off, writable = self._region(addr)
        if region is not None and writable:
            value &= 0xFFFF
            region[off] = value & 0xFF
            region[(off + 1) % len(region)] = (value >> 8) & 0xFF
            if (addr & 0xFF000000) == IO_START:
                self._after_io_write(off & 0x3FE)

    def write32(self, addr: int, value: int) -> None:
        addr &= 0xFFFFFFFC
        region, off, writable = self._region(addr)
        if region is not None and writable:
            value &= 0xFFFFFFFF
            for i in range(4):
                region[(off + i) % len(region)] = (value >> (8 * i)) & 0xFF
            if (addr & 0xFF000000) == IO_START:
                self._after_io_write(off & 0x3FE)
                self._after_io_write((off + 2) & 0x3FE)

    def _after_io_write(self, off: int) -> None:
        off &= 0x3FF
        if off in (0x000, 0x002):
            self.display_touched = True
        # Trigger immediate DMA when DMAxCNT_H enable is written with start timing 0.
        for ch in range(4):
            cnt_h = 0x0BA + ch * 12
            if off in (cnt_h, cnt_h + 1):
                control = self.read16(IO_START + cnt_h)
                if control & 0x8000 and ((control >> 12) & 3) == 0:
                    self._run_dma(ch)
                break

    def _run_dma(self, ch: int) -> None:
        base = 0x0B0 + ch * 12
        src = self.read32(IO_START + base)
        dst = self.read32(IO_START + base + 4)
        count = self.read16(IO_START + base + 8)
        control = self.read16(IO_START + base + 10)
        if count == 0:
            count = 0x10000 if ch == 3 else 0x4000
        word = bool(control & 0x0400)
        unit = 4 if word else 2
        dst_mode = (control >> 5) & 3
        src_mode = (control >> 7) & 3
        fixed_fill = bool(control & 0x0100)

        src_step = {0: unit, 1: -unit, 2: 0, 3: unit}.get(src_mode, unit)
        dst_step = {0: unit, 1: -unit, 2: 0, 3: unit}.get(dst_mode, unit)
        max_units = min(count, 0x20000)

        if fixed_fill:
            fill_value = self.read32(src) if word else self.read16(src)
        else:
            fill_value = 0

        for _ in range(max_units):
            value = fill_value if fixed_fill else (self.read32(src) if word else self.read16(src))
            if word:
                self.write32(dst, value)
            else:
                self.write16(dst, value)
            if not fixed_fill:
                src = (src + src_step) & 0xFFFFFFFF
            dst = (dst + dst_step) & 0xFFFFFFFF

        # Non-repeat immediate DMA disables itself after transfer.
        if not (control & 0x0200):
            self.write16_raw(base + 10, control & ~0x8000)


# -----------------------------------------------------------------------------
# CPU core
# -----------------------------------------------------------------------------


@dataclass
class CPUState:
    r: List[int] = field(default_factory=lambda: [0] * 16)
    cpsr: int = MODE_SYSTEM
    cycles: int = 0
    last_disasm: str = ""
    halted: bool = False
    waiting: bool = False


class ARM7TDMI:
    """Practical ARM7TDMI interpreter subset for GBA boot/homebrew inspection."""

    def __init__(self, memory: GBAMemory) -> None:
        self.mem = memory
        self.state = CPUState()
        self.reset()

    def reset(self) -> None:
        self.state = CPUState()
        s = self.state
        s.cpsr = MODE_SYSTEM
        s.r[13] = 0x03007F00
        s.r[14] = 0x00000000
        s.r[15] = ROM_START
        s.cycles = 0
        s.last_disasm = "RESET -> PC=08000000"
        s.halted = False
        s.waiting = False

    def flag_n(self) -> bool:
        return bool(self.state.cpsr & FLAG_N)

    def flag_z(self) -> bool:
        return bool(self.state.cpsr & FLAG_Z)

    def flag_c(self) -> bool:
        return bool(self.state.cpsr & FLAG_C)

    def flag_v(self) -> bool:
        return bool(self.state.cpsr & FLAG_V)

    def carry_bit(self) -> int:
        return 1 if self.flag_c() else 0

    def set_flag(self, flag: int, enabled: bool) -> None:
        if enabled:
            self.state.cpsr |= flag
        else:
            self.state.cpsr &= ~flag

    def set_nz(self, result: int) -> None:
        result &= 0xFFFFFFFF
        self.set_flag(FLAG_N, bool(result & 0x80000000))
        self.set_flag(FLAG_Z, result == 0)

    def set_add_flags(self, a: int, b: int, carry_in: int, result: int) -> None:
        a &= 0xFFFFFFFF
        b &= 0xFFFFFFFF
        unsigned_sum = a + b + carry_in
        result &= 0xFFFFFFFF
        self.set_nz(result)
        self.set_flag(FLAG_C, unsigned_sum > 0xFFFFFFFF)
        sa, sb, sr = (a >> 31) & 1, (b >> 31) & 1, (result >> 31) & 1
        self.set_flag(FLAG_V, sa == sb and sa != sr)

    def set_sub_flags(self, a: int, b: int, borrow: int, result: int) -> None:
        a &= 0xFFFFFFFF
        b &= 0xFFFFFFFF
        subtrahend = b + borrow
        result &= 0xFFFFFFFF
        self.set_nz(result)
        self.set_flag(FLAG_C, a >= subtrahend)
        sa, sb, sr = (a >> 31) & 1, (b >> 31) & 1, (result >> 31) & 1
        self.set_flag(FLAG_V, sa != sb and sa != sr)

    def cond_passed(self, cond: int) -> bool:
        n, z, c, v = self.flag_n(), self.flag_z(), self.flag_c(), self.flag_v()
        return [
            z,
            not z,
            c,
            not c,
            n,
            not n,
            v,
            not v,
            c and not z,
            (not c) or z,
            n == v,
            n != v,
            (not z) and (n == v),
            z or (n != v),
            True,
            False,
        ][cond & 0xF]

    def arm_read_reg(self, reg: int, pc: int) -> int:
        if reg == 15:
            return (pc + 8) & 0xFFFFFFFF
        return self.state.r[reg] & 0xFFFFFFFF

    def thumb_read_reg(self, reg: int, pc: int) -> int:
        if reg == 15:
            return (pc + 4) & 0xFFFFFFFC
        return self.state.r[reg] & 0xFFFFFFFF

    def write_reg_arm(self, reg: int, value: int) -> None:
        value &= 0xFFFFFFFF
        if reg == 15:
            self.state.r[15] = value & 0xFFFFFFFC
        else:
            self.state.r[reg] = value

    def write_reg_thumb(self, reg: int, value: int) -> None:
        value &= 0xFFFFFFFF
        if reg == 15:
            self.state.r[15] = value & 0xFFFFFFFE
        else:
            self.state.r[reg] = value

    def arm_shift(self, value: int, shift_type: int, amount: int, immediate_amount: bool) -> Tuple[int, int]:
        value &= 0xFFFFFFFF
        old_c = self.carry_bit()
        if shift_type == 0:  # LSL
            if amount == 0:
                return value, old_c
            if amount < 32:
                return (value << amount) & 0xFFFFFFFF, (value >> (32 - amount)) & 1
            if amount == 32:
                return 0, value & 1
            return 0, 0
        if shift_type == 1:  # LSR
            if amount == 0 and immediate_amount:
                amount = 32
            if amount == 0:
                return value, old_c
            if amount < 32:
                return value >> amount, (value >> (amount - 1)) & 1
            if amount == 32:
                return 0, (value >> 31) & 1
            return 0, 0
        if shift_type == 2:  # ASR
            if amount == 0 and immediate_amount:
                amount = 32
            if amount == 0:
                return value, old_c
            sign = value & 0x80000000
            if amount >= 32:
                return (0xFFFFFFFF if sign else 0), (1 if sign else 0)
            result = value >> amount
            if sign:
                result |= (0xFFFFFFFF << (32 - amount)) & 0xFFFFFFFF
            return result & 0xFFFFFFFF, (value >> (amount - 1)) & 1
        # ROR / RRX
        if amount == 0 and immediate_amount:
            return ((old_c << 31) | (value >> 1)) & 0xFFFFFFFF, value & 1
        amount &= 31
        if amount == 0:
            return value, old_c
        result = ror32(value, amount)
        return result, (result >> 31) & 1

    def arm_operand2(self, opcode: int, pc: int) -> Tuple[int, int]:
        if opcode & (1 << 25):
            imm = opcode & 0xFF
            rot = ((opcode >> 8) & 0xF) * 2
            result = ror32(imm, rot)
            carry = self.carry_bit() if rot == 0 else (result >> 31) & 1
            return result, carry
        rm = opcode & 0xF
        value = self.arm_read_reg(rm, pc)
        shift_type = (opcode >> 5) & 0x3
        if opcode & (1 << 4):
            rs = (opcode >> 8) & 0xF
            amount = self.arm_read_reg(rs, pc) & 0xFF
            return self.arm_shift(value, shift_type, amount, False)
        amount = (opcode >> 7) & 0x1F
        return self.arm_shift(value, shift_type, amount, True)

    def arm_transfer_offset(self, opcode: int, pc: int) -> int:
        if not (opcode & (1 << 25)):
            return opcode & 0xFFF
        rm = opcode & 0xF
        value = self.arm_read_reg(rm, pc)
        shift_type = (opcode >> 5) & 0x3
        amount = (opcode >> 7) & 0x1F
        shifted, _carry = self.arm_shift(value, shift_type, amount, True)
        return shifted

    def step(self) -> str:
        if self.state.halted:
            self.state.last_disasm = "HALTED"
            return self.state.last_disasm
        if self.state.waiting:
            self.state.cycles += 4
            self.state.waiting = False
            self.state.last_disasm = "WAIT/HALT satisfied by frame tick"
            return self.state.last_disasm
        if self.state.cpsr & CPSR_T:
            return self.step_thumb()
        return self.step_arm()

    def step_arm(self) -> str:
        s = self.state
        pc = s.r[15] & 0xFFFFFFFF
        opcode = self.mem.read32(pc)
        cond = (opcode >> 28) & 0xF
        next_pc = (pc + 4) & 0xFFFFFFFF
        dis = disassemble_arm(opcode, pc)

        if not self.cond_passed(cond):
            s.r[15] = next_pc
            s.cycles += 1
            s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis} ; condition false"
            return s.last_disasm

        # Branch exchange / branch link exchange variants.
        if (opcode & 0x0FFFFFF0) == 0x012FFF10:
            rm = opcode & 0xF
            target = self.arm_read_reg(rm, pc)
            self.branch_exchange(target)
            s.cycles += 3
            s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis}"
            return s.last_disasm

        # Multiply and multiply-accumulate.
        if (opcode & 0x0FC000F0) == 0x00000090:
            accumulate = bool(opcode & (1 << 21))
            set_flags = bool(opcode & (1 << 20))
            rd = (opcode >> 16) & 0xF
            rn = (opcode >> 12) & 0xF
            rs = (opcode >> 8) & 0xF
            rm = opcode & 0xF
            result = (self.arm_read_reg(rm, pc) * self.arm_read_reg(rs, pc)) & 0xFFFFFFFF
            if accumulate:
                result = (result + self.arm_read_reg(rn, pc)) & 0xFFFFFFFF
            self.write_reg_arm(rd, result)
            if set_flags:
                self.set_nz(result)
            s.r[15] = next_pc if rd != 15 else s.r[15]
            s.cycles += 3
            s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis}"
            return s.last_disasm

        # Halfword/signed data transfer.
        if (opcode & 0x0E000090) == 0x00000090:
            self.execute_arm_half_transfer(opcode, pc, next_pc)
            s.cycles += 2
            s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis}"
            return s.last_disasm

        # Branch / branch with link.
        if (opcode & 0x0E000000) == 0x0A000000:
            link = bool(opcode & (1 << 24))
            offset24 = opcode & 0x00FFFFFF
            signed = sign_extend(offset24, 24) << 2
            target = (pc + 8 + signed) & 0xFFFFFFFF
            if link:
                s.r[14] = (pc + 4) & 0xFFFFFFFF
            s.r[15] = target & 0xFFFFFFFC
            s.cycles += 3
            s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis}"
            return s.last_disasm

        # SWI HLE.
        if (opcode & 0x0F000000) == 0x0F000000:
            s.r[15] = next_pc
            self.handle_swi(opcode & 0xFF, pc, thumb=False)
            s.cycles += 8
            s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis} ; HLE"
            return s.last_disasm

        # Block data transfer LDM/STM.
        if (opcode & 0x0E000000) == 0x08000000:
            self.execute_arm_block_transfer(opcode, pc, next_pc)
            s.cycles += 3
            s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis}"
            return s.last_disasm

        # Single data transfer LDR/STR.
        if (opcode & 0x0C000000) == 0x04000000:
            self.execute_arm_single_transfer(opcode, pc, next_pc)
            s.cycles += 2
            s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis}"
            return s.last_disasm

        # PSR transfers, compact CPSR-only support.
        if (opcode & 0x0FBF0FFF) == 0x010F0000:  # MRS Rd, CPSR
            rd = (opcode >> 12) & 0xF
            self.write_reg_arm(rd, s.cpsr)
            s.r[15] = next_pc
            s.cycles += 1
            s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis}"
            return s.last_disasm

        if (opcode & 0x0DB0F000) == 0x0120F000:  # MSR CPSR_flg, op2-ish
            value, _carry = self.arm_operand2(opcode, pc)
            # Workbench-safe: update flags and T bit only, preserve mode bits.
            s.cpsr = (s.cpsr & 0x0FFFFFFF) | (value & 0xF0000000) | (value & CPSR_T)
            s.r[15] = next_pc
            s.cycles += 1
            s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis}"
            return s.last_disasm

        # Data processing.
        if (opcode & 0x0C000000) == 0x00000000:
            self.execute_arm_data_processing(opcode, pc, next_pc)
            s.cycles += 1
            s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis}"
            return s.last_disasm

        s.r[15] = next_pc
        s.cycles += 1
        s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis} ; skipped"
        return s.last_disasm

    def execute_arm_data_processing(self, opcode: int, pc: int, next_pc: int) -> None:
        s = self.state
        op = (opcode >> 21) & 0xF
        set_flags = bool(opcode & (1 << 20))
        rn = (opcode >> 16) & 0xF
        rd = (opcode >> 12) & 0xF
        a = self.arm_read_reg(rn, pc)
        b, sh_carry = self.arm_operand2(opcode, pc)
        write_result = True
        result = 0

        if op == 0x0:       # AND
            result = a & b
        elif op == 0x1:     # EOR
            result = a ^ b
        elif op == 0x2:     # SUB
            result = (a - b) & 0xFFFFFFFF
            if set_flags:
                self.set_sub_flags(a, b, 0, result)
        elif op == 0x3:     # RSB
            result = (b - a) & 0xFFFFFFFF
            if set_flags:
                self.set_sub_flags(b, a, 0, result)
        elif op == 0x4:     # ADD
            result = (a + b) & 0xFFFFFFFF
            if set_flags:
                self.set_add_flags(a, b, 0, result)
        elif op == 0x5:     # ADC
            c = self.carry_bit()
            result = (a + b + c) & 0xFFFFFFFF
            if set_flags:
                self.set_add_flags(a, b, c, result)
        elif op == 0x6:     # SBC
            borrow = 0 if self.flag_c() else 1
            result = (a - b - borrow) & 0xFFFFFFFF
            if set_flags:
                self.set_sub_flags(a, b, borrow, result)
        elif op == 0x7:     # RSC
            borrow = 0 if self.flag_c() else 1
            result = (b - a - borrow) & 0xFFFFFFFF
            if set_flags:
                self.set_sub_flags(b, a, borrow, result)
        elif op == 0x8:     # TST
            result = a & b
            self.set_nz(result)
            self.set_flag(FLAG_C, bool(sh_carry))
            write_result = False
        elif op == 0x9:     # TEQ
            result = a ^ b
            self.set_nz(result)
            self.set_flag(FLAG_C, bool(sh_carry))
            write_result = False
        elif op == 0xA:     # CMP
            result = (a - b) & 0xFFFFFFFF
            self.set_sub_flags(a, b, 0, result)
            write_result = False
        elif op == 0xB:     # CMN
            result = (a + b) & 0xFFFFFFFF
            self.set_add_flags(a, b, 0, result)
            write_result = False
        elif op == 0xC:     # ORR
            result = a | b
        elif op == 0xD:     # MOV
            result = b
        elif op == 0xE:     # BIC
            result = a & (~b & 0xFFFFFFFF)
        elif op == 0xF:     # MVN
            result = (~b) & 0xFFFFFFFF
        else:
            write_result = False

        if write_result:
            if set_flags and op not in (0x2, 0x3, 0x4, 0x5, 0x6, 0x7):
                self.set_nz(result)
                self.set_flag(FLAG_C, bool(sh_carry))
            self.write_reg_arm(rd, result)
            if rd != 15:
                s.r[15] = next_pc
        else:
            s.r[15] = next_pc

    def execute_arm_single_transfer(self, opcode: int, pc: int, next_pc: int) -> None:
        s = self.state
        p_bit = bool(opcode & (1 << 24))
        u_bit = bool(opcode & (1 << 23))
        b_bit = bool(opcode & (1 << 22))
        w_bit = bool(opcode & (1 << 21))
        l_bit = bool(opcode & (1 << 20))
        rn = (opcode >> 16) & 0xF
        rd = (opcode >> 12) & 0xF
        base = self.arm_read_reg(rn, pc)
        offset = self.arm_transfer_offset(opcode, pc)
        signed_off = offset if u_bit else -offset
        addr = (base + signed_off) & 0xFFFFFFFF if p_bit else base

        if l_bit:
            value = self.mem.read8(addr) if b_bit else self.mem.read32_rotate(addr)
            self.write_reg_arm(rd, value)
        else:
            value = self.arm_read_reg(rd, pc + 4) if rd == 15 else self.state.r[rd]
            if b_bit:
                self.mem.write8(addr, value)
            else:
                self.mem.write32(addr, value)

        if w_bit or not p_bit:
            self.write_reg_arm(rn, (base + signed_off) & 0xFFFFFFFF)
        if not (l_bit and rd == 15):
            s.r[15] = next_pc

    def execute_arm_half_transfer(self, opcode: int, pc: int, next_pc: int) -> None:
        s = self.state
        p_bit = bool(opcode & (1 << 24))
        u_bit = bool(opcode & (1 << 23))
        imm_off = bool(opcode & (1 << 22))
        w_bit = bool(opcode & (1 << 21))
        l_bit = bool(opcode & (1 << 20))
        rn = (opcode >> 16) & 0xF
        rd = (opcode >> 12) & 0xF
        mode = (opcode >> 5) & 0x3
        if imm_off:
            offset = ((opcode >> 4) & 0xF0) | (opcode & 0xF)
        else:
            offset = self.arm_read_reg(opcode & 0xF, pc)
        base = self.arm_read_reg(rn, pc)
        signed_off = offset if u_bit else -offset
        addr = (base + signed_off) & 0xFFFFFFFF if p_bit else base

        if l_bit:
            if mode == 1:
                value = self.mem.read16(addr)
            elif mode == 2:
                value = self.mem.read8_signed(addr)
            elif mode == 3:
                value = self.mem.read16_signed(addr)
            else:
                value = 0
            self.write_reg_arm(rd, value)
        else:
            if mode == 1:
                self.mem.write16(addr, self.arm_read_reg(rd, pc + 4))
            else:
                # Undefined store encodings are safely ignored.
                pass
        if w_bit or not p_bit:
            self.write_reg_arm(rn, (base + signed_off) & 0xFFFFFFFF)
        if not (l_bit and rd == 15):
            s.r[15] = next_pc

    def execute_arm_block_transfer(self, opcode: int, pc: int, next_pc: int) -> None:
        s = self.state
        p_bit = bool(opcode & (1 << 24))
        u_bit = bool(opcode & (1 << 23))
        w_bit = bool(opcode & (1 << 21))
        l_bit = bool(opcode & (1 << 20))
        rn = (opcode >> 16) & 0xF
        reglist = opcode & 0xFFFF
        regs = [i for i in range(16) if reglist & (1 << i)]
        count = len(regs) or 16
        base = self.arm_read_reg(rn, pc)

        if not regs:
            regs = list(range(16))

        if u_bit:
            addr = base + (4 if p_bit else 0)
            wb = base + 4 * count
        else:
            addr = base - (4 * count if p_bit else 4 * (count - 1))
            wb = base - 4 * count
        addr &= 0xFFFFFFFF

        loaded_pc = False
        for reg in regs:
            if l_bit:
                value = self.mem.read32(addr)
                self.write_reg_arm(reg, value)
                loaded_pc = loaded_pc or reg == 15
            else:
                value = self.arm_read_reg(reg, pc)
                if reg == 15:
                    value = (pc + 12) & 0xFFFFFFFF
                self.mem.write32(addr, value)
            addr = (addr + 4) & 0xFFFFFFFF

        if w_bit and not (l_bit and rn in regs):
            self.write_reg_arm(rn, wb)
        if not loaded_pc:
            s.r[15] = next_pc

    def step_thumb(self) -> str:
        s = self.state
        pc = s.r[15] & 0xFFFFFFFE
        op = self.mem.read16(pc)
        next_pc = (pc + 2) & 0xFFFFFFFF
        dis = disassemble_thumb(op, pc)

        # Format 1: move shifted register.
        if (op & 0xE000) == 0x0000 and (op & 0x1800) != 0x1800:
            shift_type = (op >> 11) & 0x3
            amount = (op >> 6) & 0x1F
            rs = (op >> 3) & 0x7
            rd = op & 0x7
            value, carry = self.arm_shift(s.r[rs], shift_type, amount, True)
            s.r[rd] = value
            self.set_nz(value)
            if not (shift_type == 0 and amount == 0):
                self.set_flag(FLAG_C, bool(carry))
            s.r[15] = next_pc
            s.cycles += 1
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 2: add/subtract register/immediate.
        if (op & 0xF800) == 0x1800:
            immediate = bool(op & 0x0400)
            subtract = bool(op & 0x0200)
            rn_or_imm = (op >> 6) & 0x7
            rs = (op >> 3) & 0x7
            rd = op & 0x7
            left = s.r[rs]
            right = rn_or_imm if immediate else s.r[rn_or_imm]
            if subtract:
                result = (left - right) & 0xFFFFFFFF
                self.set_sub_flags(left, right, 0, result)
            else:
                result = (left + right) & 0xFFFFFFFF
                self.set_add_flags(left, right, 0, result)
            s.r[rd] = result
            s.r[15] = next_pc
            s.cycles += 1
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 3: MOV/CMP/ADD/SUB immediate.
        if (op & 0xE000) == 0x2000:
            kind = (op >> 11) & 0x3
            rd = (op >> 8) & 0x7
            imm = op & 0xFF
            if kind == 0:
                s.r[rd] = imm
                self.set_nz(imm)
            elif kind == 1:
                res = (s.r[rd] - imm) & 0xFFFFFFFF
                self.set_sub_flags(s.r[rd], imm, 0, res)
            elif kind == 2:
                res = (s.r[rd] + imm) & 0xFFFFFFFF
                self.set_add_flags(s.r[rd], imm, 0, res)
                s.r[rd] = res
            else:
                res = (s.r[rd] - imm) & 0xFFFFFFFF
                self.set_sub_flags(s.r[rd], imm, 0, res)
                s.r[rd] = res
            s.r[15] = next_pc
            s.cycles += 1
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 4: ALU operations.
        if (op & 0xFC00) == 0x4000:
            self.execute_thumb_alu(op)
            s.r[15] = next_pc
            s.cycles += 1
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 5: high register operations / BX.
        if (op & 0xFC00) == 0x4400:
            h1 = (op >> 7) & 1
            h2 = (op >> 6) & 1
            op_kind = (op >> 8) & 0x3
            rs = ((op >> 3) & 0x7) | (h2 << 3)
            rd = (op & 0x7) | (h1 << 3)
            rs_val = self.thumb_read_reg(rs, pc)
            rd_val = self.thumb_read_reg(rd, pc)
            if op_kind == 0:  # ADD
                self.write_reg_thumb(rd, (rd_val + rs_val) & 0xFFFFFFFF)
            elif op_kind == 1:  # CMP
                result = (rd_val - rs_val) & 0xFFFFFFFF
                self.set_sub_flags(rd_val, rs_val, 0, result)
                s.r[15] = next_pc
            elif op_kind == 2:  # MOV
                self.write_reg_thumb(rd, rs_val)
            else:  # BX
                self.branch_exchange(rs_val)
            if op_kind != 3 and not (op_kind in (0, 2) and rd == 15):
                s.r[15] = next_pc
            s.cycles += 2 if op_kind == 3 else 1
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 6: PC-relative load.
        if (op & 0xF800) == 0x4800:
            rd = (op >> 8) & 0x7
            addr = ((pc + 4) & 0xFFFFFFFC) + ((op & 0xFF) << 2)
            s.r[rd] = self.mem.read32(addr)
            s.r[15] = next_pc
            s.cycles += 2
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 7: load/store register offset.
        if (op & 0xF200) == 0x5000:
            l_bit = bool(op & 0x0800)
            b_bit = bool(op & 0x0400)
            ro = (op >> 6) & 0x7
            rb = (op >> 3) & 0x7
            rd = op & 0x7
            addr = (s.r[rb] + s.r[ro]) & 0xFFFFFFFF
            if l_bit:
                s.r[rd] = self.mem.read8(addr) if b_bit else self.mem.read32_rotate(addr)
            else:
                if b_bit:
                    self.mem.write8(addr, s.r[rd])
                else:
                    self.mem.write32(addr, s.r[rd])
            s.r[15] = next_pc
            s.cycles += 2
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 8: load/store sign-extended byte/halfword.
        if (op & 0xF200) == 0x5200:
            kind = (op >> 10) & 0x3
            ro = (op >> 6) & 0x7
            rb = (op >> 3) & 0x7
            rd = op & 0x7
            addr = (s.r[rb] + s.r[ro]) & 0xFFFFFFFF
            if kind == 0:
                self.mem.write16(addr, s.r[rd])
            elif kind == 1:
                s.r[rd] = self.mem.read8_signed(addr)
            elif kind == 2:
                s.r[rd] = self.mem.read16(addr)
            else:
                s.r[rd] = self.mem.read16_signed(addr)
            s.r[15] = next_pc
            s.cycles += 2
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 9: immediate word/byte load/store.
        if (op & 0xE000) == 0x6000:
            b_bit = bool(op & 0x1000)
            l_bit = bool(op & 0x0800)
            imm5 = (op >> 6) & 0x1F
            rb = (op >> 3) & 0x7
            rd = op & 0x7
            addr = s.r[rb] + (imm5 if b_bit else imm5 << 2)
            if l_bit:
                s.r[rd] = self.mem.read8(addr) if b_bit else self.mem.read32_rotate(addr)
            else:
                if b_bit:
                    self.mem.write8(addr, s.r[rd])
                else:
                    self.mem.write32(addr, s.r[rd])
            s.r[15] = next_pc
            s.cycles += 2
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 10: immediate halfword load/store.
        if (op & 0xF000) == 0x8000:
            l_bit = bool(op & 0x0800)
            imm5 = (op >> 6) & 0x1F
            rb = (op >> 3) & 0x7
            rd = op & 0x7
            addr = s.r[rb] + (imm5 << 1)
            if l_bit:
                s.r[rd] = self.mem.read16(addr)
            else:
                self.mem.write16(addr, s.r[rd])
            s.r[15] = next_pc
            s.cycles += 2
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 11: SP-relative load/store.
        if (op & 0xF000) == 0x9000:
            l_bit = bool(op & 0x0800)
            rd = (op >> 8) & 0x7
            addr = s.r[13] + ((op & 0xFF) << 2)
            if l_bit:
                s.r[rd] = self.mem.read32(addr)
            else:
                self.mem.write32(addr, s.r[rd])
            s.r[15] = next_pc
            s.cycles += 2
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 12: load address.
        if (op & 0xF000) == 0xA000:
            rd = (op >> 8) & 0x7
            base = s.r[13] if (op & 0x0800) else ((pc + 4) & 0xFFFFFFFC)
            s.r[rd] = (base + ((op & 0xFF) << 2)) & 0xFFFFFFFF
            s.r[15] = next_pc
            s.cycles += 1
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 13: add/subtract offset to SP.
        if (op & 0xFF00) == 0xB000:
            offset = (op & 0x7F) << 2
            if op & 0x80:
                s.r[13] = (s.r[13] - offset) & 0xFFFFFFFF
            else:
                s.r[13] = (s.r[13] + offset) & 0xFFFFFFFF
            s.r[15] = next_pc
            s.cycles += 1
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 14: PUSH/POP.
        if (op & 0xF600) == 0xB400:
            pop = bool(op & 0x0800)
            extra = bool(op & 0x0100)
            reglist = op & 0xFF
            regs = [i for i in range(8) if reglist & (1 << i)]
            if pop:
                if extra:
                    regs.append(15)
                addr = s.r[13]
                for reg in regs:
                    value = self.mem.read32(addr)
                    if reg == 15:
                        self.write_reg_thumb(15, value)
                    else:
                        s.r[reg] = value
                    addr += 4
                s.r[13] = addr & 0xFFFFFFFF
                if 15 not in regs:
                    s.r[15] = next_pc
            else:
                if extra:
                    regs.append(14)
                addr = s.r[13] - 4 * len(regs)
                s.r[13] = addr & 0xFFFFFFFF
                for reg in regs:
                    self.mem.write32(addr, s.r[reg])
                    addr += 4
                s.r[15] = next_pc
            s.cycles += 2 + len(regs)
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 15: multiple load/store.
        if (op & 0xF000) == 0xC000:
            l_bit = bool(op & 0x0800)
            rb = (op >> 8) & 0x7
            reglist = op & 0xFF
            regs = [i for i in range(8) if reglist & (1 << i)]
            addr = s.r[rb]
            for reg in regs:
                if l_bit:
                    s.r[reg] = self.mem.read32(addr)
                else:
                    self.mem.write32(addr, s.r[reg])
                addr += 4
            if regs:
                s.r[rb] = addr & 0xFFFFFFFF
            s.r[15] = next_pc
            s.cycles += 2 + len(regs)
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 16/17: conditional branch and SWI.
        if (op & 0xF000) == 0xD000:
            cond = (op >> 8) & 0xF
            if cond == 0xF:
                s.r[15] = next_pc
                self.handle_swi(op & 0xFF, pc, thumb=True)
                s.cycles += 8
            elif cond != 0xE and self.cond_passed(cond):
                offset = sign_extend(op & 0xFF, 8) << 1
                s.r[15] = (pc + 4 + offset) & 0xFFFFFFFE
                s.cycles += 3
            else:
                s.r[15] = next_pc
                s.cycles += 1
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 18: unconditional branch.
        if (op & 0xF800) == 0xE000:
            off = sign_extend(op & 0x7FF, 11) << 1
            s.r[15] = (pc + 4 + off) & 0xFFFFFFFE
            s.cycles += 3
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # Format 19: long branch with link.
        if (op & 0xF800) in (0xF000, 0xF800):
            if (op & 0xF800) == 0xF000:
                offset = sign_extend(op & 0x7FF, 11) << 12
                s.r[14] = (pc + 4 + offset) & 0xFFFFFFFF
                s.r[15] = next_pc
            else:
                target = (s.r[14] + ((op & 0x7FF) << 1)) & 0xFFFFFFFF
                s.r[14] = (pc + 2) | 1
                s.r[15] = target & 0xFFFFFFFE
            s.cycles += 3
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        s.r[15] = next_pc
        s.cycles += 1
        s.last_disasm = f"{pc:08X}: {op:04X}      {dis} ; skipped"
        return s.last_disasm

    def execute_thumb_alu(self, op: int) -> None:
        s = self.state
        kind = (op >> 6) & 0xF
        rs = (op >> 3) & 0x7
        rd = op & 0x7
        a = s.r[rd]
        b = s.r[rs]
        result = a
        if kind == 0x0:  # AND
            result = a & b
            s.r[rd] = result
            self.set_nz(result)
        elif kind == 0x1:  # EOR
            result = a ^ b
            s.r[rd] = result
            self.set_nz(result)
        elif kind == 0x2:  # LSL
            result, carry = self.arm_shift(a, 0, b & 0xFF, False)
            s.r[rd] = result
            self.set_nz(result)
            self.set_flag(FLAG_C, bool(carry))
        elif kind == 0x3:  # LSR
            result, carry = self.arm_shift(a, 1, b & 0xFF, False)
            s.r[rd] = result
            self.set_nz(result)
            self.set_flag(FLAG_C, bool(carry))
        elif kind == 0x4:  # ASR
            result, carry = self.arm_shift(a, 2, b & 0xFF, False)
            s.r[rd] = result
            self.set_nz(result)
            self.set_flag(FLAG_C, bool(carry))
        elif kind == 0x5:  # ADC
            c = self.carry_bit()
            result = (a + b + c) & 0xFFFFFFFF
            s.r[rd] = result
            self.set_add_flags(a, b, c, result)
        elif kind == 0x6:  # SBC
            borrow = 0 if self.flag_c() else 1
            result = (a - b - borrow) & 0xFFFFFFFF
            s.r[rd] = result
            self.set_sub_flags(a, b, borrow, result)
        elif kind == 0x7:  # ROR
            result, carry = self.arm_shift(a, 3, b & 0xFF, False)
            s.r[rd] = result
            self.set_nz(result)
            self.set_flag(FLAG_C, bool(carry))
        elif kind == 0x8:  # TST
            result = a & b
            self.set_nz(result)
        elif kind == 0x9:  # NEG
            result = (-b) & 0xFFFFFFFF
            s.r[rd] = result
            self.set_sub_flags(0, b, 0, result)
        elif kind == 0xA:  # CMP
            result = (a - b) & 0xFFFFFFFF
            self.set_sub_flags(a, b, 0, result)
        elif kind == 0xB:  # CMN
            result = (a + b) & 0xFFFFFFFF
            self.set_add_flags(a, b, 0, result)
        elif kind == 0xC:  # ORR
            result = a | b
            s.r[rd] = result
            self.set_nz(result)
        elif kind == 0xD:  # MUL
            result = (a * b) & 0xFFFFFFFF
            s.r[rd] = result
            self.set_nz(result)
        elif kind == 0xE:  # BIC
            result = a & (~b & 0xFFFFFFFF)
            s.r[rd] = result
            self.set_nz(result)
        elif kind == 0xF:  # MVN
            result = (~b) & 0xFFFFFFFF
            s.r[rd] = result
            self.set_nz(result)

    def branch_exchange(self, target: int) -> None:
        if target & 1:
            self.state.cpsr |= CPSR_T
            self.state.r[15] = target & 0xFFFFFFFE
        else:
            self.state.cpsr &= ~CPSR_T
            self.state.r[15] = target & 0xFFFFFFFC

    def handle_swi(self, number: int, pc: int, thumb: bool) -> None:
        s = self.state
        n = number & 0xFF
        if n == 0x00:  # SoftReset
            old_rom = self.mem.rom
            self.reset()
            self.mem.rom = old_rom
            return
        if n == 0x01:  # RegisterRamReset
            mask = s.r[0]
            if mask & 0x01:
                self.mem.ewram[:] = b"\x00" * len(self.mem.ewram)
            if mask & 0x02:
                self.mem.iwram[:] = b"\x00" * len(self.mem.iwram)
            if mask & 0x04:
                self.mem.palette[:] = b"\x00" * len(self.mem.palette)
            if mask & 0x08:
                self.mem.vram[:] = b"\x00" * len(self.mem.vram)
            if mask & 0x10:
                self.mem.oam[:] = b"\x00" * len(self.mem.oam)
            if mask & 0x80:
                self.mem.reset_io()
            return
        if n in (0x02, 0x03, 0x04, 0x05):  # Halt/Stop/IntrWait/VBlankIntrWait.
            s.waiting = True
            return
        if n == 0x06:  # Div: r0 / r1
            self._bios_div(s.r[0], s.r[1])
            return
        if n == 0x07:  # DivArm: r1 / r0
            self._bios_div(s.r[1], s.r[0])
            return
        if n == 0x08:  # Sqrt
            s.r[0] = int(math.isqrt(s.r[0] & 0xFFFFFFFF)) & 0xFFFFFFFF
            return
        if n == 0x0B:  # CpuSet
            self._bios_cpuset(fast=False)
            return
        if n == 0x0C:  # CpuFastSet
            self._bios_cpuset(fast=True)
            return
        if n in (0x11, 0x12):  # LZ77UnCompWRAM/VRAM
            self._bios_lz77(vram=(n == 0x12))
            return
        if n in (0x14, 0x15):  # RLUnCompWRAM/VRAM
            self._bios_rl(vram=(n == 0x15))
            return
        # Unknown BIOS calls are non-fatal in the workbench.

    def _bios_div(self, numerator: int, denominator: int) -> None:
        s = self.state
        num = s32(numerator)
        den = s32(denominator)
        if den == 0:
            q = -1 if num >= 0 else 1
            r = num
        else:
            q = int(num / den)  # truncate toward zero, like ARM BIOS.
            r = num - q * den
        s.r[0] = q & 0xFFFFFFFF
        s.r[1] = r & 0xFFFFFFFF
        s.r[3] = abs(q) & 0xFFFFFFFF

    def _bios_cpuset(self, fast: bool) -> None:
        s = self.state
        src = s.r[0]
        dst = s.r[1]
        mode = s.r[2]
        fill = bool(mode & (1 << 24))
        word = fast or bool(mode & (1 << 26))
        unit = 4 if word else 2
        count = mode & 0x1FFFFF
        if fast:
            count *= 8
            unit = 4
        if count == 0:
            return
        if fill:
            value = self.mem.read32(src) if word else self.mem.read16(src)
        else:
            value = 0
        max_units = min(count, 0x40000)
        for _ in range(max_units):
            current = value if fill else (self.mem.read32(src) if word else self.mem.read16(src))
            if word:
                self.mem.write32(dst, current)
            else:
                self.mem.write16(dst, current)
            if not fill:
                src = (src + unit) & 0xFFFFFFFF
            dst = (dst + unit) & 0xFFFFFFFF

    def _bios_lz77(self, vram: bool) -> None:
        src = self.state.r[0]
        dst = self.state.r[1]
        header = self.mem.read32(src)
        if (header & 0xFF) != 0x10:
            return
        size = header >> 8
        src += 4
        written = 0
        out_byte = 0
        half_accum = 0
        half_pending = False

        def put_byte(value: int) -> None:
            nonlocal dst, written, half_accum, half_pending
            value &= 0xFF
            if vram:
                if not half_pending:
                    half_accum = value
                    half_pending = True
                else:
                    self.mem.write16(dst, half_accum | (value << 8))
                    dst += 2
                    half_pending = False
            else:
                self.mem.write8(dst, value)
                dst += 1
            written += 1

        while written < size:
            flags = self.mem.read8(src)
            src += 1
            for bit in range(7, -1, -1):
                if written >= size:
                    break
                if flags & (1 << bit):
                    b1 = self.mem.read8(src)
                    b2 = self.mem.read8(src + 1)
                    src += 2
                    length = (b1 >> 4) + 3
                    disp = (((b1 & 0xF) << 8) | b2) + 1
                    copy_src = (dst - disp) if not vram else (dst - disp)
                    for _ in range(length):
                        if written >= size:
                            break
                        out_byte = self.mem.read8(copy_src)
                        copy_src += 1
                        put_byte(out_byte)
                else:
                    put_byte(self.mem.read8(src))
                    src += 1
        if vram and half_pending:
            self.mem.write16(dst, half_accum)

    def _bios_rl(self, vram: bool) -> None:
        src = self.state.r[0]
        dst = self.state.r[1]
        header = self.mem.read32(src)
        if (header & 0xFF) != 0x30:
            return
        size = header >> 8
        src += 4
        written = 0
        half_accum = 0
        half_pending = False

        def put_byte(value: int) -> None:
            nonlocal dst, written, half_accum, half_pending
            value &= 0xFF
            if vram:
                if not half_pending:
                    half_accum = value
                    half_pending = True
                else:
                    self.mem.write16(dst, half_accum | (value << 8))
                    dst += 2
                    half_pending = False
            else:
                self.mem.write8(dst, value)
                dst += 1
            written += 1

        while written < size:
            flag = self.mem.read8(src)
            src += 1
            if flag & 0x80:
                count = (flag & 0x7F) + 3
                value = self.mem.read8(src)
                src += 1
                for _ in range(count):
                    if written >= size:
                        break
                    put_byte(value)
            else:
                count = (flag & 0x7F) + 1
                for _ in range(count):
                    if written >= size:
                        break
                    put_byte(self.mem.read8(src))
                    src += 1
        if vram and half_pending:
            self.mem.write16(dst, half_accum)

    def register_text(self) -> str:
        s = self.state
        lines = []
        for i in range(0, 16, 2):
            lines.append(f"r{i:<2}={s.r[i] & 0xFFFFFFFF:08X}  r{i+1:<2}={s.r[i+1] & 0xFFFFFFFF:08X}")
        flags = "".join([
            "N" if self.flag_n() else "n",
            "Z" if self.flag_z() else "z",
            "C" if self.flag_c() else "c",
            "V" if self.flag_v() else "v",
            "T" if (s.cpsr & CPSR_T) else "A",
        ])
        lines.append(f"CPSR={s.cpsr:08X}  FLAGS={flags}  CYCLES={s.cycles}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Disassemblers
# -----------------------------------------------------------------------------


def arm_operand2_text(opcode: int) -> str:
    if opcode & (1 << 25):
        imm = opcode & 0xFF
        rot = ((opcode >> 8) & 0xF) * 2
        return f"#0x{ror32(imm, rot):X}"
    rm = opcode & 0xF
    if opcode & (1 << 4):
        rs = (opcode >> 8) & 0xF
        typ = ["LSL", "LSR", "ASR", "ROR"][(opcode >> 5) & 3]
        return f"r{rm}, {typ} r{rs}"
    amount = (opcode >> 7) & 0x1F
    typ = ["LSL", "LSR", "ASR", "ROR"][(opcode >> 5) & 3]
    if amount == 0:
        return f"r{rm}"
    return f"r{rm}, {typ} #{amount}"


def disassemble_arm(opcode: int, pc: int) -> str:
    cond = cond_name((opcode >> 28) & 0xF)

    if (opcode & 0x0FFFFFF0) == 0x012FFF10:
        return f"BX{cond} r{opcode & 0xF}"

    if (opcode & 0x0FC000F0) == 0x00000090:
        acc = "MLA" if opcode & (1 << 21) else "MUL"
        s = "S" if opcode & (1 << 20) else ""
        rd = (opcode >> 16) & 0xF
        rn = (opcode >> 12) & 0xF
        rs = (opcode >> 8) & 0xF
        rm = opcode & 0xF
        return f"{acc}{s}{cond} r{rd}, r{rm}, r{rs}" + (f", r{rn}" if acc == "MLA" else "")

    if (opcode & 0x0E000000) == 0x0A000000:
        link = "L" if opcode & (1 << 24) else ""
        off = sign_extend(opcode & 0x00FFFFFF, 24) << 2
        target = (pc + 8 + off) & 0xFFFFFFFF
        return f"B{link}{cond} 0x{target:08X}"

    if (opcode & 0x0F000000) == 0x0F000000:
        return f"SWI{cond} #{opcode & 0xFF}"

    if (opcode & 0x0E000090) == 0x00000090:
        p_bit = bool(opcode & (1 << 24))
        u_bit = bool(opcode & (1 << 23))
        imm = bool(opcode & (1 << 22))
        w_bit = bool(opcode & (1 << 21))
        l_bit = bool(opcode & (1 << 20))
        rn = (opcode >> 16) & 0xF
        rd = (opcode >> 12) & 0xF
        mode = (opcode >> 5) & 3
        names = {1: "LDRH" if l_bit else "STRH", 2: "LDRSB", 3: "LDRSH"}
        name = names.get(mode, "HDT")
        sign = "+" if u_bit else "-"
        off = (((opcode >> 4) & 0xF0) | (opcode & 0xF)) if imm else (opcode & 0xF)
        off_text = f"#{sign}{off}" if imm else f"{sign}r{off}"
        bang = "!" if w_bit else ""
        return f"{name}{cond} r{rd}, [r{rn}, {off_text}]{bang}" + (" ; post" if not p_bit else "")

    if (opcode & 0x0E000000) == 0x08000000:
        p = "B" if opcode & (1 << 24) else "A"
        u = "I" if opcode & (1 << 23) else "D"
        l = "LDM" if opcode & (1 << 20) else "STM"
        rn = (opcode >> 16) & 0xF
        regs = [f"r{i}" for i in range(16) if opcode & (1 << i)]
        bang = "!" if opcode & (1 << 21) else ""
        return f"{l}{u}{p}{cond} r{rn}{bang}, {{{', '.join(regs)}}}"

    if (opcode & 0x0C000000) == 0x04000000:
        i_bit = bool(opcode & (1 << 25))
        p_bit = bool(opcode & (1 << 24))
        u_bit = bool(opcode & (1 << 23))
        b_bit = bool(opcode & (1 << 22))
        w_bit = bool(opcode & (1 << 21))
        l_bit = bool(opcode & (1 << 20))
        rn = (opcode >> 16) & 0xF
        rd = (opcode >> 12) & 0xF
        offset = opcode & 0xFFF
        name = "LDR" if l_bit else "STR"
        suffix = "B" if b_bit else ""
        sign = "+" if u_bit else "-"
        mode = "pre" if p_bit else "post"
        bang = "!" if w_bit else ""
        off_text = arm_operand2_text(opcode) if i_bit else f"#{sign}{offset}"
        return f"{name}{suffix}{cond} r{rd}, [r{rn}, {off_text}]{bang} ; {mode}"

    if (opcode & 0x0FBF0FFF) == 0x010F0000:
        return f"MRS{cond} r{(opcode >> 12) & 0xF}, CPSR"
    if (opcode & 0x0DB0F000) == 0x0120F000:
        return f"MSR{cond} CPSR_f, {arm_operand2_text(opcode)}"

    if (opcode & 0x0C000000) == 0x00000000:
        names = [
            "AND", "EOR", "SUB", "RSB", "ADD", "ADC", "SBC", "RSC",
            "TST", "TEQ", "CMP", "CMN", "ORR", "MOV", "BIC", "MVN",
        ]
        op = (opcode >> 21) & 0xF
        s = "S" if opcode & (1 << 20) else ""
        rn = (opcode >> 16) & 0xF
        rd = (opcode >> 12) & 0xF
        op2 = arm_operand2_text(opcode)
        if op in (0xD, 0xF):
            return f"{names[op]}{s}{cond} r{rd}, {op2}"
        if op in (0x8, 0x9, 0xA, 0xB):
            return f"{names[op]}{cond} r{rn}, {op2}"
        return f"{names[op]}{s}{cond} r{rd}, r{rn}, {op2}"

    return f"ARM{cond} 0x{opcode:08X}"


def disassemble_thumb(op: int, pc: int) -> str:
    if (op & 0xE000) == 0x0000 and (op & 0x1800) != 0x1800:
        names = ["LSL", "LSR", "ASR", "ROR"]
        kind = (op >> 11) & 0x3
        imm = (op >> 6) & 0x1F
        rs = (op >> 3) & 0x7
        rd = op & 0x7
        return f"{names[kind]} r{rd}, r{rs}, #{imm}"
    if (op & 0xF800) == 0x1800:
        name = "SUB" if op & 0x0200 else "ADD"
        imm = bool(op & 0x0400)
        val = (op >> 6) & 0x7
        rs = (op >> 3) & 0x7
        rd = op & 0x7
        return f"{name} r{rd}, r{rs}, " + (f"#{val}" if imm else f"r{val}")
    if (op & 0xE000) == 0x2000:
        names = ["MOV", "CMP", "ADD", "SUB"]
        kind = (op >> 11) & 0x3
        rd = (op >> 8) & 0x7
        imm = op & 0xFF
        return f"{names[kind]} r{rd}, #{imm}"
    if (op & 0xFC00) == 0x4000:
        names = ["AND", "EOR", "LSL", "LSR", "ASR", "ADC", "SBC", "ROR", "TST", "NEG", "CMP", "CMN", "ORR", "MUL", "BIC", "MVN"]
        return f"{names[(op >> 6) & 0xF]} r{op & 7}, r{(op >> 3) & 7}"
    if (op & 0xFC00) == 0x4400:
        names = ["ADD", "CMP", "MOV", "BX"]
        h1 = (op >> 7) & 1
        h2 = (op >> 6) & 1
        rs = ((op >> 3) & 0x7) | (h2 << 3)
        rd = (op & 0x7) | (h1 << 3)
        kind = (op >> 8) & 0x3
        return f"BX r{rs}" if kind == 3 else f"{names[kind]} r{rd}, r{rs}"
    if (op & 0xF800) == 0x4800:
        return f"LDR r{(op >> 8) & 7}, [PC, #{(op & 0xFF) << 2}]"
    if (op & 0xF200) == 0x5000:
        name = "LDR" if op & 0x0800 else "STR"
        suffix = "B" if op & 0x0400 else ""
        return f"{name}{suffix} r{op & 7}, [r{(op >> 3) & 7}, r{(op >> 6) & 7}]"
    if (op & 0xF200) == 0x5200:
        names = ["STRH", "LDRSB", "LDRH", "LDRSH"]
        return f"{names[(op >> 10) & 3]} r{op & 7}, [r{(op >> 3) & 7}, r{(op >> 6) & 7}]"
    if (op & 0xE000) == 0x6000:
        name = "LDR" if op & 0x0800 else "STR"
        suffix = "B" if op & 0x1000 else ""
        imm = (op >> 6) & 0x1F
        if not suffix:
            imm <<= 2
        return f"{name}{suffix} r{op & 7}, [r{(op >> 3) & 7}, #{imm}]"
    if (op & 0xF000) == 0x8000:
        name = "LDRH" if op & 0x0800 else "STRH"
        return f"{name} r{op & 7}, [r{(op >> 3) & 7}, #{((op >> 6) & 0x1F) << 1}]"
    if (op & 0xF000) == 0x9000:
        return f"{'LDR' if op & 0x0800 else 'STR'} r{(op >> 8) & 7}, [SP, #{(op & 0xFF) << 2}]"
    if (op & 0xF000) == 0xA000:
        return f"ADD r{(op >> 8) & 7}, {'SP' if op & 0x0800 else 'PC'}, #{(op & 0xFF) << 2}"
    if (op & 0xFF00) == 0xB000:
        return f"{'SUB' if op & 0x80 else 'ADD'} SP, #{(op & 0x7F) << 2}"
    if (op & 0xF600) == 0xB400:
        pop = bool(op & 0x0800)
        regs = [f"r{i}" for i in range(8) if op & (1 << i)]
        if op & 0x0100:
            regs.append("PC" if pop else "LR")
        return f"{'POP' if pop else 'PUSH'} {{{', '.join(regs)}}}"
    if (op & 0xF000) == 0xC000:
        regs = [f"r{i}" for i in range(8) if op & (1 << i)]
        return f"{'LDMIA' if op & 0x0800 else 'STMIA'} r{(op >> 8) & 7}!, {{{', '.join(regs)}}}"
    if (op & 0xF000) == 0xD000:
        cond = (op >> 8) & 0xF
        if cond == 0xF:
            return f"SWI #{op & 0xFF}"
        if cond == 0xE:
            return "UNDEFINED"
        target = (pc + 4 + (sign_extend(op & 0xFF, 8) << 1)) & 0xFFFFFFFF
        return f"B{cond_name(cond)} 0x{target:08X}"
    if (op & 0xF800) == 0xE000:
        target = (pc + 4 + (sign_extend(op & 0x7FF, 11) << 1)) & 0xFFFFFFFF
        return f"B 0x{target:08X}"
    if (op & 0xF800) == 0xF000:
        return f"BL prefix #{op & 0x7FF}"
    if (op & 0xF800) == 0xF800:
        return f"BL suffix #{op & 0x7FF}"
    return f"THUMB 0x{op:04X}"


# -----------------------------------------------------------------------------
# PPU/frame rendering
# -----------------------------------------------------------------------------


class GBAPPU:
    def __init__(self, memory: GBAMemory, accelerator: Accelerator) -> None:
        self.mem = memory
        self.acc = accelerator

    def render(self, frame: int, pc: int, keys_mask: int) -> bytes:
        dispcnt = self.mem.read16(IO_START + 0x000)
        mode = dispcnt & 0x7
        frame_select = 1 if (dispcnt & 0x0010) else 0
        bg2_enabled = bool(dispcnt & 0x0400)
        any_bg_enabled = bool(dispcnt & 0x0F00)

        # If no display registers have been touched yet, keep a lively ROM visualization.
        if not self.mem.display_touched and self.mem.rom:
            return self.acc.render_viz(self.mem.rom, frame, pc, keys_mask)

        if mode == 3 and bg2_enabled:
            return self.acc.render_mode3(self.mem.vram)
        if mode == 4 and bg2_enabled:
            return self.acc.render_mode4(self.mem.vram, self.mem.palette, frame_select)
        if mode == 5 and bg2_enabled:
            return self.acc.render_mode5(self.mem.vram, frame_select)
        if mode == 0 and any_bg_enabled:
            return self.render_text_bg(dispcnt)
        # Fallback visualization avoids a dead-black canvas while stepping code.
        return self.acc.render_viz(self.mem.rom, frame, pc, keys_mask)

    def render_text_bg(self, dispcnt: int) -> bytes:
        # Draw the first enabled regular text background.  This is intentionally
        # compact: no windowing, blending, OBJ rendering, affine modes, or mosaic.
        bg_index = None
        for i in range(4):
            if dispcnt & (0x0100 << i):
                bg_index = i
                break
        if bg_index is None:
            return bytes(SCREEN_W * SCREEN_H * 3)

        bgcnt = self.mem.read16(IO_START + 0x008 + bg_index * 2)
        scroll_x = self.mem.read16(IO_START + 0x010 + bg_index * 4) & 0x1FF
        scroll_y = self.mem.read16(IO_START + 0x012 + bg_index * 4) & 0x1FF
        char_base = ((bgcnt >> 2) & 0x3) * 0x4000
        color_256 = bool(bgcnt & 0x0080)
        screen_base = ((bgcnt >> 8) & 0x1F) * 0x800
        size_code = (bgcnt >> 14) & 0x3
        width_tiles = 32 if size_code in (0, 2) else 64
        height_tiles = 32 if size_code in (0, 1) else 64
        out = bytearray(SCREEN_W * SCREEN_H * 3)
        oi = 0

        for y in range(SCREEN_H):
            sy = (y + scroll_y) % (height_tiles * 8)
            tile_y = sy // 8
            fine_y = sy & 7
            for x in range(SCREEN_W):
                sx = (x + scroll_x) % (width_tiles * 8)
                tile_x = sx // 8
                fine_x = sx & 7
                map_block = 0
                local_x = tile_x
                local_y = tile_y
                if width_tiles == 64 and tile_x >= 32:
                    map_block += 1
                    local_x -= 32
                if height_tiles == 64 and tile_y >= 32:
                    map_block += 2 if width_tiles == 64 else 1
                    local_y -= 32
                entry_addr = screen_base + map_block * 0x800 + (local_y * 32 + local_x) * 2
                entry = self._vram16(entry_addr)
                tile = entry & 0x3FF
                hflip = bool(entry & 0x0400)
                vflip = bool(entry & 0x0800)
                pal_bank = (entry >> 12) & 0xF
                px = 7 - fine_x if hflip else fine_x
                py = 7 - fine_y if vflip else fine_y
                if color_256:
                    tile_addr = char_base + tile * 64 + py * 8 + px
                    idx = self._vram8(tile_addr)
                else:
                    tile_addr = char_base + tile * 32 + py * 4 + (px >> 1)
                    b = self._vram8(tile_addr)
                    idx = (b >> 4) if (px & 1) else (b & 0xF)
                    if idx:
                        idx += pal_bank * 16
                if idx == 0:
                    color = 0
                else:
                    color = self._pal16(idx * 2)
                out[oi], out[oi + 1], out[oi + 2] = rgb555_to_tuple(color)
                oi += 3
        return bytes(out)

    def _vram8(self, off: int) -> int:
        return self.mem.vram[off % len(self.mem.vram)]

    def _vram16(self, off: int) -> int:
        off %= len(self.mem.vram)
        return self.mem.vram[off] | (self.mem.vram[(off + 1) % len(self.mem.vram)] << 8)

    def _pal16(self, off: int) -> int:
        off &= 0x3FE
        return self.mem.palette[off] | (self.mem.palette[off + 1] << 8)


# -----------------------------------------------------------------------------
# Emulator container
# -----------------------------------------------------------------------------


@dataclass
class GBAEmulator:
    accelerator: Accelerator
    memory: GBAMemory = field(default_factory=GBAMemory)
    header: GBAHeader = field(default_factory=GBAHeader)
    rom_path: Optional[Path] = None
    frame: int = 0
    running: bool = False
    keys: Dict[str, bool] = field(default_factory=dict)
    turbo: bool = False

    def __post_init__(self) -> None:
        self.cpu = ARM7TDMI(self.memory)
        self.ppu = GBAPPU(self.memory, self.accelerator)

    def load_rom_file(self, path: Path) -> None:
        data = path.read_bytes()
        self.memory.load_rom(data)
        self.header = GBAHeader.from_rom(data, self.accelerator.header_checksum)
        self.rom_path = path
        self.frame = 0
        self.running = False
        self.memory.reset_ram()
        self.cpu.reset()

    def reset(self) -> None:
        old_rom = self.memory.rom
        old_path = self.rom_path
        self.running = False
        self.frame = 0
        self.memory.reset_ram()
        self.memory.rom = old_rom
        self.rom_path = old_path
        self.cpu.reset()

    def run_slice(self, max_instructions: int = 4096, max_cycles: int = GBA_CYCLES_PER_FRAME // 5) -> List[str]:
        lines: List[str] = []
        if not self.memory.rom:
            return ["No ROM loaded."]
        self.memory.update_key_input(self.keys)
        start_cycles = self.cpu.state.cycles
        for i in range(max_instructions):
            line = self.cpu.step()
            if len(lines) < 14:
                lines.append(line)
            if self.cpu.state.halted:
                self.running = False
                break
            if self.cpu.state.cycles - start_cycles >= max_cycles:
                break
        return lines

    def run_frame(self) -> List[str]:
        # Tkinter/Python cannot guarantee full ARM cycle accuracy, so this uses a
        # bounded instruction budget per visual frame.  Cython speeds rendering,
        # while this budget keeps the UI responsive at a 60 Hz cadence.
        budget = 24_000 if self.turbo else 7_000
        cycle_budget = GBA_CYCLES_PER_FRAME if self.turbo else GBA_CYCLES_PER_FRAME // 2
        trace = self.run_slice(budget, cycle_budget)
        self.frame += 1
        self.memory.set_vcount(0)
        return trace

    def render_frame(self) -> bytes:
        pc = self.cpu.state.r[15] & 0xFFFFFFFF
        keys_mask = self.key_mask()
        return self.ppu.render(self.frame, pc, keys_mask)

    def key_mask(self) -> int:
        value = 0
        for name, bit in KEY_BITS.items():
            if self.keys.get(name, False):
                value |= 1 << bit
        return value

    def render_seed(self) -> int:
        limit = min(len(self.memory.rom), 8192)
        return self.accelerator.rom_fold(self.memory.rom, self.frame, limit)


# -----------------------------------------------------------------------------
# Tkinter UI
# -----------------------------------------------------------------------------


class ChatGPTGBAApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.configure(bg=BLACK)
        self.geometry("1160x760")
        self.minsize(980, 660)

        self.accelerator = load_cython_accelerator()
        self.emu = GBAEmulator(self.accelerator)
        self.trace_lines: List[str] = []
        self.frame_times: List[float] = []
        self.fps = 0.0
        self.next_frame_deadline = time.perf_counter()
        self.raw_photo: Optional[tk.PhotoImage] = None
        self.scaled_photo: Optional[tk.PhotoImage] = None
        self.screen_item: Optional[int] = None
        self.last_panel_update = 0.0

        self._build_widgets()
        self._bind_keys()
        self._log("chtptsgbaemu ready.")
        if self.accelerator.available:
            self._log("Cython accelerator: ON")
        else:
            self._log("Cython accelerator: OFF, pure Python fallback active.")
            if self.accelerator.error:
                self._log(f"Cython note: {self.accelerator.error}")
        self._present_frame()
        self._update_panels(force=True)
        self.after(1, self._tick)

    def blue_button(self, parent: tk.Widget, text: str, command: Callable[[], None]) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=DEEP_BLUE,
            fg=TEXT_BLUE,
            activebackground=MID_BLUE,
            activeforeground=BRIGHT_BLUE,
            disabledforeground=DIM_BLUE,
            highlightbackground=BLACK,
            highlightcolor=BRIGHT_BLUE,
            relief=tk.RIDGE,
            bd=2,
            padx=8,
            pady=4,
            font=FONT_MONO,
        )

    def blue_label(self, parent: tk.Widget, text: str, big: bool = False) -> tk.Label:
        return tk.Label(
            parent,
            text=text,
            bg=BLACK,
            fg=TEXT_BLUE,
            font=FONT_MONO_BIG if big else FONT_MONO,
            anchor="w",
        )

    def _build_widgets(self) -> None:
        root = tk.Frame(self, bg=BLACK)
        root.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        top = tk.Frame(root, bg=BLACK)
        top.pack(fill=tk.X)
        self.blue_label(top, "chtptsgbaemu", big=True).pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            top,
            textvariable=self.status_var,
            bg=BLACK,
            fg=BRIGHT_BLUE,
            font=FONT_MONO,
            anchor="e",
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True)

        body = tk.Frame(root, bg=BLACK)
        body.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        left = tk.Frame(body, bg=BLACK)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        self.canvas = tk.Canvas(
            left,
            width=CANVAS_W,
            height=CANVAS_H,
            bg=BLACK,
            highlightthickness=2,
            highlightbackground=MID_BLUE,
        )
        self.canvas.pack(side=tk.TOP)
        self.screen_item = self.canvas.create_image(0, 0, anchor="nw")

        controls = tk.Frame(left, bg=BLACK)
        controls.pack(fill=tk.X, pady=8)
        for text, cmd in [
            ("Open ROM", self.open_rom),
            ("Start/Pause", self.toggle_run),
            ("Step", self.step_once),
            ("Reset", self.reset_emu),
            ("Turbo", self.toggle_turbo),
            ("Cython", self.show_cython_status),
            ("Clear Log", self.clear_log),
        ]:
            self.blue_button(controls, text, cmd).pack(side=tk.LEFT, padx=3)

        pad = tk.Frame(left, bg=BLACK)
        pad.pack(fill=tk.X, pady=(2, 8))
        self.key_var = tk.StringVar(value="Keys: arrows, Z=A, X=B, Enter=Start, Backspace=Select, A=L, S=R")
        tk.Label(pad, textvariable=self.key_var, bg=BLACK, fg=TEXT_BLUE, font=FONT_MONO_SMALL).pack(anchor="w")

        self.log = tk.Text(
            left,
            height=9,
            width=88,
            bg=PANEL_BLACK,
            fg=TEXT_BLUE,
            insertbackground=BRIGHT_BLUE,
            selectbackground=MID_BLUE,
            selectforeground=BRIGHT_BLUE,
            relief=tk.RIDGE,
            bd=2,
            font=FONT_MONO_SMALL,
            wrap=tk.WORD,
        )
        self.log.pack(fill=tk.BOTH, expand=True)

        right = tk.Frame(body, bg=BLACK)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        self.header_text = tk.Text(
            right,
            height=12,
            bg=PANEL_BLACK,
            fg=TEXT_BLUE,
            insertbackground=BRIGHT_BLUE,
            selectbackground=MID_BLUE,
            relief=tk.RIDGE,
            bd=2,
            font=FONT_MONO,
            wrap=tk.NONE,
        )
        self.header_text.pack(fill=tk.X)

        self.reg_text = tk.Text(
            right,
            height=12,
            bg=PANEL_BLACK,
            fg=TEXT_BLUE,
            insertbackground=BRIGHT_BLUE,
            selectbackground=MID_BLUE,
            relief=tk.RIDGE,
            bd=2,
            font=FONT_MONO,
            wrap=tk.NONE,
        )
        self.reg_text.pack(fill=tk.X, pady=(8, 0))

        self.disasm_text = tk.Text(
            right,
            bg=PANEL_BLACK,
            fg=TEXT_BLUE,
            insertbackground=BRIGHT_BLUE,
            selectbackground=MID_BLUE,
            relief=tk.RIDGE,
            bd=2,
            font=FONT_MONO_SMALL,
            wrap=tk.NONE,
        )
        self.disasm_text.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    def _bind_keys(self) -> None:
        mapping = {
            "Up": "UP", "Down": "DOWN", "Left": "LEFT", "Right": "RIGHT",
            "z": "A", "Z": "A", "x": "B", "X": "B",
            "Return": "START", "BackSpace": "SELECT",
            "a": "L", "A": "L", "s": "R", "S": "R",
        }

        def press(event: tk.Event) -> None:
            key = mapping.get(event.keysym)
            if key:
                self.emu.keys[key] = True
                self.emu.memory.update_key_input(self.emu.keys)
                self._update_key_text()

        def release(event: tk.Event) -> None:
            key = mapping.get(event.keysym)
            if key:
                self.emu.keys[key] = False
                self.emu.memory.update_key_input(self.emu.keys)
                self._update_key_text()

        self.bind("<KeyPress>", press)
        self.bind("<KeyRelease>", release)

    def _update_key_text(self) -> None:
        pressed = [k for k, down in sorted(self.emu.keys.items()) if down]
        if pressed:
            self.key_var.set("Pressed: " + ", ".join(pressed))
        else:
            self.key_var.set("Keys: arrows, Z=A, X=B, Enter=Start, Backspace=Select, A=L, S=R")

    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, f"[{stamp}] {text}\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.NORMAL)

    def clear_log(self) -> None:
        self.log.delete("1.0", tk.END)

    def open_rom(self) -> None:
        path_str = filedialog.askopenfilename(
            title="Open GBA ROM",
            filetypes=[
                ("Game Boy Advance ROM", "*.gba *.agb *.bin"),
                ("All files", "*.*"),
            ],
        )
        if not path_str:
            return
        try:
            path = Path(path_str)
            self.emu.load_rom_file(path)
            self.trace_lines.clear()
            self.next_frame_deadline = time.perf_counter()
            self._log(f"Loaded ROM: {path.name}")
            self._log(f"Title: {self.emu.header.title}")
            self._log(f"Save type: {self.emu.header.save_type}")
            if not self.emu.header.checksum_ok:
                self._log("Header checksum warning: ROM may be homebrew, patched, or invalid.")
            self._present_frame()
            self._update_panels(force=True)
        except Exception as exc:
            messagebox.showerror("Open ROM failed", f"{type(exc).__name__}: {exc}")
            self._log(traceback.format_exc())

    def toggle_run(self) -> None:
        if not self.emu.memory.rom:
            messagebox.showinfo("No ROM", "Open a .gba ROM first.")
            return
        self.emu.running = not self.emu.running
        self.next_frame_deadline = time.perf_counter()
        self._log("Running at 60 Hz target." if self.emu.running else "Paused.")

    def toggle_turbo(self) -> None:
        self.emu.turbo = not self.emu.turbo
        self._log("Turbo instruction budget ON." if self.emu.turbo else "Turbo instruction budget OFF.")
        self._update_panels(force=True)

    def step_once(self) -> None:
        if not self.emu.memory.rom:
            messagebox.showinfo("No ROM", "Open a .gba ROM first.")
            return
        lines = self.emu.run_slice(1, 16)
        self.trace_lines.extend(lines)
        self.trace_lines = self.trace_lines[-500:]
        self.emu.frame += 1
        self._present_frame()
        self._update_panels(force=True)

    def reset_emu(self) -> None:
        self.emu.reset()
        self.trace_lines.clear()
        self.next_frame_deadline = time.perf_counter()
        self._log("Reset.")
        self._present_frame()
        self._update_panels(force=True)

    def show_cython_status(self) -> None:
        if self.accelerator.available:
            text = "Cython accelerator is active for header checks, ROM folding, and bitmap/viz framebuffer rendering."
        else:
            text = "Cython accelerator is not active. Pure Python fallback is being used.\n\n" + self.accelerator.error
        messagebox.showinfo("Cython status", text)
        self._log(text.replace("\n", " "))

    def _tick(self) -> None:
        try:
            now = time.perf_counter()
            if self.emu.running and now >= self.next_frame_deadline:
                frame_start = time.perf_counter()
                lines = self.emu.run_frame()
                self.trace_lines.extend(lines)
                self.trace_lines = self.trace_lines[-500:]
                self._present_frame()
                frame_end = time.perf_counter()
                self.frame_times.append(frame_end - frame_start)
                self.frame_times = self.frame_times[-60:]
                elapsed = frame_end - now
                self.fps = 1.0 / max(TARGET_FRAME_TIME, elapsed)
                # Deadline-based scheduling limits drift while still yielding to Tk.
                self.next_frame_deadline += TARGET_FRAME_TIME
                if self.next_frame_deadline < frame_end - TARGET_FRAME_TIME:
                    self.next_frame_deadline = frame_end + TARGET_FRAME_TIME
                self._update_panels(force=False)
        finally:
            delay = max(1, int((self.next_frame_deadline - time.perf_counter()) * 1000))
            self.after(min(delay, 16), self._tick)

    def _present_frame(self) -> None:
        rgb = self.emu.render_frame()
        try:
            self.raw_photo = tk.PhotoImage(data=make_ppm(rgb), format="PPM")
            self.scaled_photo = self.raw_photo.zoom(SCREEN_SCALE, SCREEN_SCALE)
            if self.screen_item is not None:
                self.canvas.itemconfig(self.screen_item, image=self.scaled_photo)
        except tk.TclError:
            # Ultra-safe fallback: keep canvas responsive even if a Tk build lacks PPM data support.
            self.canvas.delete("fallback")
            self.canvas.create_rectangle(0, 0, CANVAS_W, CANVAS_H, fill=BLACK, outline=MID_BLUE, width=2, tags="fallback")

        self._draw_hud()

    def _draw_hud(self) -> None:
        c = self.canvas
        c.delete("hud")
        c.create_rectangle(0, 0, CANVAS_W, CANVAS_H, outline=MID_BLUE, width=2, tags="hud")
        if not self.emu.memory.rom:
            t = self.emu.frame
            c.create_text(CANVAS_W // 2, 88, text="chtptsgbaemu", fill=BRIGHT_BLUE, font=("Consolas", 28, "bold"), tags="hud")
            c.create_text(CANVAS_W // 2, 134, text="Open a .gba ROM to inspect and step code", fill=TEXT_BLUE, font=FONT_MONO, tags="hud")
            c.create_text(CANVAS_W // 2, 162, text="60 FPS target | optional Cython | single-file", fill=TEXT_BLUE, font=FONT_MONO, tags="hud")
            for i in range(48):
                angle = (i / 48.0) * math.tau + t * 0.05
                radius = 80 + 30 * math.sin(t * 0.04 + i)
                x = CANVAS_W // 2 + int(math.cos(angle) * radius)
                y = CANVAS_H // 2 + int(math.sin(angle) * radius)
                c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=BLUE, outline="", tags="hud")
            return

        pc = self.emu.cpu.state.r[15] & 0xFFFFFFFF
        mode = "THUMB" if self.emu.cpu.state.cpsr & CPSR_T else "ARM"
        title = self.emu.header.title[:18]
        c.create_rectangle(12, 12, CANVAS_W - 12, 92, fill="#000713", outline=BRIGHT_BLUE, width=2, tags="hud")
        c.create_text(26, 28, text="chtptsgbaemu", anchor="nw", fill=BRIGHT_BLUE, font=FONT_MONO_BIG, tags="hud")
        c.create_text(26, 55, text=f"ROM: {title}", anchor="nw", fill=TEXT_BLUE, font=FONT_MONO, tags="hud")
        c.create_text(26, 75, text=f"PC: {pc:08X}  MODE: {mode}  FRAME: {self.emu.frame}", anchor="nw", fill=TEXT_BLUE, font=FONT_MONO, tags="hud")

        pressed = [k for k, down in sorted(self.emu.keys.items()) if down]
        key_text = "INPUT: " + (", ".join(pressed) if pressed else "none")
        c.create_rectangle(12, CANVAS_H - 44, CANVAS_W - 12, CANVAS_H - 12, fill="#000713", outline=DIM_BLUE, width=1, tags="hud")
        c.create_text(26, CANVAS_H - 36, text=key_text, anchor="nw", fill=TEXT_BLUE, font=FONT_MONO, tags="hud")

    def _update_panels(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.last_panel_update < 0.10:
            return
        self.last_panel_update = now

        h = self.emu.header
        self.header_text.configure(state=tk.NORMAL)
        self.header_text.delete("1.0", tk.END)
        self.header_text.insert(tk.END, "ROM HEADER\n")
        self.header_text.insert(tk.END, "-" * 52 + "\n")
        for line in h.lines():
            self.header_text.insert(tk.END, line + "\n")
        self.header_text.insert(tk.END, f"CYTHON     : {'ON' if self.accelerator.available else 'OFF'}\n")
        self.header_text.insert(tk.END, f"TARGET FPS : {TARGET_FPS:.0f}\n")
        self.header_text.insert(tk.END, f"TURBO      : {'ON' if self.emu.turbo else 'OFF'}\n")
        self.header_text.configure(state=tk.DISABLED)

        self.reg_text.configure(state=tk.NORMAL)
        self.reg_text.delete("1.0", tk.END)
        self.reg_text.insert(tk.END, "CPU REGISTERS\n")
        self.reg_text.insert(tk.END, "-" * 52 + "\n")
        self.reg_text.insert(tk.END, self.emu.cpu.register_text())
        self.reg_text.configure(state=tk.DISABLED)

        self.disasm_text.configure(state=tk.NORMAL)
        self.disasm_text.delete("1.0", tk.END)
        self.disasm_text.insert(tk.END, "RECENT TRACE\n")
        self.disasm_text.insert(tk.END, "-" * 76 + "\n")
        if self.trace_lines:
            self.disasm_text.insert(tk.END, "\n".join(self.trace_lines[-180:]))
        else:
            self.disasm_text.insert(tk.END, self.emu.cpu.state.last_disasm)
        self.disasm_text.configure(state=tk.DISABLED)

        rom_name = self.emu.rom_path.name if self.emu.rom_path else "No ROM"
        run = "RUN" if self.emu.running else "PAUSE"
        avg_ms = (sum(self.frame_times) / len(self.frame_times) * 1000.0) if self.frame_times else 0.0
        self.status_var.set(
            f"{rom_name} | {run} | frame {self.emu.frame} | target 60 fps | work {avg_ms:4.1f} ms | "
            f"Cython {'ON' if self.accelerator.available else 'OFF'} | Turbo {'ON' if self.emu.turbo else 'OFF'}"
        )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> int:
    app = ChatGPTGBAApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
