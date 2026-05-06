#!/usr/bin/env python3
"""
programchatgptgba.py

A blue-on-black Tkinter Game Boy Advance emulator workbench.

Run:
    python programchatgptgba.py

Optional Cython acceleration:
    pip install cython
    python programchatgptgba.py

What this file includes:
    - A Tkinter UI styled with a black background and blue text/buttons.
    - ROM loading and GBA header parsing/checksum validation.
    - A GBA memory map scaffold.
    - A small ARM7TDMI interpreter skeleton with partial ARM/THUMB support.
    - Optional Cython helpers generated and compiled at runtime when Cython is available.

Important:
    This is a compact emulator scaffold/workbench, not a complete commercial-game-compatible
    GBA emulator. Full GBA compatibility requires a much larger implementation: complete
    ARM7TDMI timing, PPU modes, DMA, timers, interrupts, APU, save hardware, wait states,
    BIOS behavior, and many cartridge edge cases.
"""

from __future__ import annotations

import os
import sys
import math
import time
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox


# -----------------------------------------------------------------------------
# Blue-on-black theme
# -----------------------------------------------------------------------------

APP_TITLE = "ChatGPT GBA - Blue/Black Tkinter Workbench"
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

ROM_START = 0x08000000
EWRAM_START = 0x02000000
IWRAM_START = 0x03000000
IO_START = 0x04000000
PAL_START = 0x05000000
VRAM_START = 0x06000000
OAM_START = 0x07000000

CPSR_T = 1 << 5
FLAG_N = 1 << 31
FLAG_Z = 1 << 30
FLAG_C = 1 << 29
FLAG_V = 1 << 28


# -----------------------------------------------------------------------------
# Optional Cython acceleration
# -----------------------------------------------------------------------------

CYTHON_SOURCE = r'''
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True

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
'''


def python_header_checksum(data: bytes) -> int:
    if len(data) < 0xBE:
        return -1
    s = sum(data[0xA0:0xBD])
    return (-(s + 0x19)) & 0xFF


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


@dataclass
class Accelerator:
    available: bool = False
    error: str = ""
    header_checksum: Callable[[bytes], int] = python_header_checksum
    rom_fold: Callable[[bytes, int, int], int] = python_rom_fold


def load_cython_accelerator() -> Accelerator:
    """Build and import a tiny Cython helper module when possible."""
    try:
        import pyximport  # type: ignore

        cache_root = Path(tempfile.gettempdir()) / "programchatgptgba_cython"
        cache_root.mkdir(parents=True, exist_ok=True)
        pyx_path = cache_root / "gba_fast_blue.pyx"
        if not pyx_path.exists() or pyx_path.read_text(encoding="utf-8") != CYTHON_SOURCE:
            pyx_path.write_text(CYTHON_SOURCE, encoding="utf-8")

        if str(cache_root) not in sys.path:
            sys.path.insert(0, str(cache_root))

        pyximport.install(
            build_dir=str(cache_root / "build"),
            language_level=3,
            inplace=False,
        )
        import gba_fast_blue  # type: ignore

        return Accelerator(
            available=True,
            error="",
            header_checksum=gba_fast_blue.header_checksum,
            rom_fold=gba_fast_blue.rom_fold,
        )
    except Exception as exc:  # Cython/compiler missing is normal.
        return Accelerator(
            available=False,
            error=f"{type(exc).__name__}: {exc}",
            header_checksum=python_header_checksum,
            rom_fold=python_rom_fold,
        )


# -----------------------------------------------------------------------------
# GBA ROM/header helpers
# -----------------------------------------------------------------------------


def clean_ascii(raw: bytes) -> str:
    text = raw.decode("ascii", errors="replace")
    text = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in text)
    return text.rstrip(" \x00") or "<blank>"


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

    @classmethod
    def from_rom(cls, data: bytes, checksum_func: Callable[[bytes], int]) -> "GBAHeader":
        if len(data) < 0xBE:
            return cls(title="<file too small>", rom_size=len(data))
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
        )

    def lines(self) -> List[str]:
        mib = self.rom_size / (1024 * 1024) if self.rom_size else 0
        return [
            f"TITLE       : {self.title}",
            f"GAME CODE   : {self.game_code}",
            f"MAKER CODE  : {self.maker_code}",
            f"ROM SIZE    : {self.rom_size:,} bytes ({mib:.2f} MiB)",
            f"FIXED 0xB2  : 0x{self.fixed_value:02X} {'OK' if self.fixed_value == 0x96 else 'WARN'}",
            f"VERSION     : {self.version}",
            f"CHECKSUM    : 0x{self.checksum:02X} expected 0x{self.expected_checksum:02X} {'OK' if self.checksum_ok else 'WARN'}",
        ]


# -----------------------------------------------------------------------------
# GBA memory scaffold
# -----------------------------------------------------------------------------


class GBAMemory:
    """A compact model of the GBA address map.

    This provides enough structure for ROM inspection and CPU stepping. It is not yet
    a cycle-accurate memory bus with wait states or every mirrored region implemented.
    """

    def __init__(self) -> None:
        self.rom: bytes = b""
        self.ewram = bytearray(256 * 1024)
        self.iwram = bytearray(32 * 1024)
        self.io = bytearray(1024)
        self.palette = bytearray(1024)
        self.vram = bytearray(96 * 1024)
        self.oam = bytearray(1024)

    def load_rom(self, data: bytes) -> None:
        self.rom = data

    def reset_ram(self) -> None:
        self.ewram[:] = b"\x00" * len(self.ewram)
        self.iwram[:] = b"\x00" * len(self.iwram)
        self.io[:] = b"\x00" * len(self.io)
        self.palette[:] = b"\x00" * len(self.palette)
        self.vram[:] = b"\x00" * len(self.vram)
        self.oam[:] = b"\x00" * len(self.oam)

    def _region(self, addr: int) -> Tuple[Optional[bytearray], int]:
        addr &= 0xFFFFFFFF
        top = addr & 0xFF000000
        if top == EWRAM_START:
            return self.ewram, addr & (len(self.ewram) - 1)
        if top == IWRAM_START:
            return self.iwram, addr & (len(self.iwram) - 1)
        if top == IO_START:
            return self.io, addr & (len(self.io) - 1)
        if top == PAL_START:
            return self.palette, addr & (len(self.palette) - 1)
        if top == VRAM_START:
            return self.vram, addr % len(self.vram)
        if top == OAM_START:
            return self.oam, addr & (len(self.oam) - 1)
        return None, 0

    def read8(self, addr: int) -> int:
        addr &= 0xFFFFFFFF
        if 0x08000000 <= addr <= 0x0DFFFFFF:
            if not self.rom:
                return 0xFF
            idx = (addr - ROM_START) % len(self.rom)
            return self.rom[idx]
        region, off = self._region(addr)
        if region is None:
            return 0
        return region[off]

    def read16(self, addr: int) -> int:
        addr &= 0xFFFFFFFE
        return self.read8(addr) | (self.read8(addr + 1) << 8)

    def read32(self, addr: int) -> int:
        addr &= 0xFFFFFFFC
        return (
            self.read8(addr)
            | (self.read8(addr + 1) << 8)
            | (self.read8(addr + 2) << 16)
            | (self.read8(addr + 3) << 24)
        )

    def write8(self, addr: int, value: int) -> None:
        region, off = self._region(addr)
        if region is not None:
            region[off] = value & 0xFF

    def write16(self, addr: int, value: int) -> None:
        addr &= 0xFFFFFFFE
        self.write8(addr, value)
        self.write8(addr + 1, value >> 8)

    def write32(self, addr: int, value: int) -> None:
        addr &= 0xFFFFFFFC
        self.write8(addr, value)
        self.write8(addr + 1, value >> 8)
        self.write8(addr + 2, value >> 16)
        self.write8(addr + 3, value >> 24)


# -----------------------------------------------------------------------------
# Small ARM7TDMI interpreter skeleton
# -----------------------------------------------------------------------------


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


@dataclass
class CPUState:
    r: List[int] = field(default_factory=lambda: [0] * 16)
    cpsr: int = 0x1F
    cycles: int = 0
    last_disasm: str = ""
    halted: bool = False


class ARM7TDMI:
    """Partial ARM7TDMI model for stepping through ROM entry code.

    Implemented here: condition checks, ARM branch/branch-link, branch exchange,
    a small subset of data-processing and immediate single-data-transfer operations,
    plus a tiny THUMB subset. Unknown instructions are safely skipped and shown in
    the disassembly panel.
    """

    def __init__(self, memory: GBAMemory) -> None:
        self.mem = memory
        self.state = CPUState()
        self.reset()

    def reset(self) -> None:
        self.state = CPUState()
        s = self.state
        s.cpsr = 0x1F  # System mode, ARM state.
        s.r[13] = 0x03007F00
        s.r[14] = 0x00000000
        s.r[15] = ROM_START
        s.cycles = 0
        s.last_disasm = "RESET -> PC=08000000"
        s.halted = False

    def flag_n(self) -> bool:
        return bool(self.state.cpsr & FLAG_N)

    def flag_z(self) -> bool:
        return bool(self.state.cpsr & FLAG_Z)

    def flag_c(self) -> bool:
        return bool(self.state.cpsr & FLAG_C)

    def flag_v(self) -> bool:
        return bool(self.state.cpsr & FLAG_V)

    def set_nz(self, result: int) -> None:
        self.state.cpsr &= ~(FLAG_N | FLAG_Z)
        result &= 0xFFFFFFFF
        if result & 0x80000000:
            self.state.cpsr |= FLAG_N
        if result == 0:
            self.state.cpsr |= FLAG_Z

    def set_add_flags(self, a: int, b: int, result: int) -> None:
        self.set_nz(result)
        self.state.cpsr &= ~(FLAG_C | FLAG_V)
        if (a + b) > 0xFFFFFFFF:
            self.state.cpsr |= FLAG_C
        sa, sb, sr = (a >> 31) & 1, (b >> 31) & 1, (result >> 31) & 1
        if sa == sb and sa != sr:
            self.state.cpsr |= FLAG_V

    def set_sub_flags(self, a: int, b: int, result: int) -> None:
        self.set_nz(result)
        self.state.cpsr &= ~(FLAG_C | FLAG_V)
        if a >= b:
            self.state.cpsr |= FLAG_C
        sa, sb, sr = (a >> 31) & 1, (b >> 31) & 1, (result >> 31) & 1
        if sa != sb and sa != sr:
            self.state.cpsr |= FLAG_V

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

    def arm_operand2(self, op: int) -> int:
        if op & (1 << 25):
            imm = op & 0xFF
            rot = ((op >> 8) & 0xF) * 2
            return ror32(imm, rot)
        rm = op & 0xF
        # Simplified: register value without full ARM barrel-shifter semantics.
        return self.state.r[rm] & 0xFFFFFFFF

    def step(self) -> str:
        if self.state.halted:
            self.state.last_disasm = "HALTED"
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

        # Branch exchange: BX Rm
        if (opcode & 0x0FFFFFF0) == 0x012FFF10:
            rm = opcode & 0xF
            target = s.r[rm] & 0xFFFFFFFF
            if target & 1:
                s.cpsr |= CPSR_T
                target &= 0xFFFFFFFE
            else:
                s.cpsr &= ~CPSR_T
                target &= 0xFFFFFFFC
            s.r[15] = target
            s.cycles += 3
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
            s.r[15] = target
            s.cycles += 3
            s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis}"
            return s.last_disasm

        # SWI: mark halted so this scaffold does not wander into unmapped BIOS behavior.
        if (opcode & 0x0F000000) == 0x0F000000:
            s.halted = True
            s.r[15] = next_pc
            s.cycles += 3
            s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis} ; BIOS call placeholder"
            return s.last_disasm

        # Single data transfer, simplified immediate offset only.
        if (opcode & 0x0C000000) == 0x04000000:
            i_bit = bool(opcode & (1 << 25))
            p_bit = bool(opcode & (1 << 24))
            u_bit = bool(opcode & (1 << 23))
            b_bit = bool(opcode & (1 << 22))
            w_bit = bool(opcode & (1 << 21))
            l_bit = bool(opcode & (1 << 20))
            rn = (opcode >> 16) & 0xF
            rd = (opcode >> 12) & 0xF
            if not i_bit:
                offset = opcode & 0xFFF
                base = s.r[rn] & 0xFFFFFFFF
                signed_off = offset if u_bit else -offset
                addr = (base + signed_off) & 0xFFFFFFFF if p_bit else base
                if l_bit:
                    value = self.mem.read8(addr) if b_bit else self.mem.read32(addr)
                    s.r[rd] = value & 0xFFFFFFFF
                    if rd == 15:
                        next_pc = value & 0xFFFFFFFC
                else:
                    value = s.r[rd] & 0xFFFFFFFF
                    if b_bit:
                        self.mem.write8(addr, value)
                    else:
                        self.mem.write32(addr, value)
                if w_bit or not p_bit:
                    s.r[rn] = (base + signed_off) & 0xFFFFFFFF
            s.r[15] = next_pc
            s.cycles += 2
            s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis}"
            return s.last_disasm

        # Data processing subset.
        if (opcode & 0x0C000000) == 0x00000000:
            op = (opcode >> 21) & 0xF
            set_flags = bool(opcode & (1 << 20))
            rn = (opcode >> 16) & 0xF
            rd = (opcode >> 12) & 0xF
            a = s.r[rn] & 0xFFFFFFFF
            b = self.arm_operand2(opcode)
            write_result = True
            result = s.r[rd]

            if op == 0x0:       # AND
                result = a & b
            elif op == 0x2:     # SUB
                result = (a - b) & 0xFFFFFFFF
                if set_flags:
                    self.set_sub_flags(a, b, result)
            elif op == 0x4:     # ADD
                result = (a + b) & 0xFFFFFFFF
                if set_flags:
                    self.set_add_flags(a, b, result)
            elif op == 0xA:     # CMP
                result = (a - b) & 0xFFFFFFFF
                self.set_sub_flags(a, b, result)
                write_result = False
            elif op == 0xC:     # ORR
                result = a | b
            elif op == 0xD:     # MOV
                result = b
            elif op == 0xF:     # MVN
                result = (~b) & 0xFFFFFFFF
            else:
                write_result = False

            if write_result:
                if set_flags and op not in (0x2, 0x4):
                    self.set_nz(result)
                if rd == 15:
                    next_pc = result & 0xFFFFFFFC
                else:
                    s.r[rd] = result & 0xFFFFFFFF
            s.r[15] = next_pc
            s.cycles += 1
            s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis}"
            return s.last_disasm

        s.r[15] = next_pc
        s.cycles += 1
        s.last_disasm = f"{pc:08X}: {opcode:08X}  {dis} ; skipped"
        return s.last_disasm

    def step_thumb(self) -> str:
        s = self.state
        pc = s.r[15] & 0xFFFFFFFE
        op = self.mem.read16(pc)
        next_pc = (pc + 2) & 0xFFFFFFFF
        dis = disassemble_thumb(op, pc)

        # THUMB unconditional branch.
        if (op & 0xF800) == 0xE000:
            off = sign_extend(op & 0x7FF, 11) << 1
            s.r[15] = (pc + 4 + off) & 0xFFFFFFFE
            s.cycles += 3
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # THUMB MOV/CMP/ADD/SUB immediate.
        if (op & 0xE000) == 0x2000:
            kind = (op >> 11) & 0x3
            rd = (op >> 8) & 0x7
            imm = op & 0xFF
            if kind == 0:      # MOV
                s.r[rd] = imm
                self.set_nz(imm)
            elif kind == 1:    # CMP
                res = (s.r[rd] - imm) & 0xFFFFFFFF
                self.set_sub_flags(s.r[rd], imm, res)
            elif kind == 2:    # ADD
                res = (s.r[rd] + imm) & 0xFFFFFFFF
                self.set_add_flags(s.r[rd], imm, res)
                s.r[rd] = res
            elif kind == 3:    # SUB
                res = (s.r[rd] - imm) & 0xFFFFFFFF
                self.set_sub_flags(s.r[rd], imm, res)
                s.r[rd] = res
            s.r[15] = next_pc
            s.cycles += 1
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        # THUMB BX.
        if (op & 0xFF87) == 0x4700:
            rm = ((op >> 3) & 0xF)
            target = s.r[rm] & 0xFFFFFFFF
            if target & 1:
                s.cpsr |= CPSR_T
                s.r[15] = target & 0xFFFFFFFE
            else:
                s.cpsr &= ~CPSR_T
                s.r[15] = target & 0xFFFFFFFC
            s.cycles += 3
            s.last_disasm = f"{pc:08X}: {op:04X}      {dis}"
            return s.last_disasm

        s.r[15] = next_pc
        s.cycles += 1
        s.last_disasm = f"{pc:08X}: {op:04X}      {dis} ; skipped"
        return s.last_disasm

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


def disassemble_arm(opcode: int, pc: int) -> str:
    cond = cond_name((opcode >> 28) & 0xF)

    if (opcode & 0x0FFFFFF0) == 0x012FFF10:
        return f"BX{cond} r{opcode & 0xF}"

    if (opcode & 0x0E000000) == 0x0A000000:
        link = "L" if opcode & (1 << 24) else ""
        off = sign_extend(opcode & 0x00FFFFFF, 24) << 2
        target = (pc + 8 + off) & 0xFFFFFFFF
        return f"B{link}{cond} 0x{target:08X}"

    if (opcode & 0x0F000000) == 0x0F000000:
        return f"SWI{cond} #{opcode & 0x00FFFFFF}"

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
        off_text = "reg" if i_bit else f"#{sign}{offset}"
        return f"{name}{suffix}{cond} r{rd}, [r{rn}, {off_text}]{bang} ; {mode}"

    if (opcode & 0x0C000000) == 0x00000000:
        names = [
            "AND", "EOR", "SUB", "RSB", "ADD", "ADC", "SBC", "RSC",
            "TST", "TEQ", "CMP", "CMN", "ORR", "MOV", "BIC", "MVN",
        ]
        op = (opcode >> 21) & 0xF
        s = "S" if opcode & (1 << 20) else ""
        rn = (opcode >> 16) & 0xF
        rd = (opcode >> 12) & 0xF
        if opcode & (1 << 25):
            imm = opcode & 0xFF
            rot = ((opcode >> 8) & 0xF) * 2
            op2 = f"#0x{ror32(imm, rot):X}"
        else:
            op2 = f"r{opcode & 0xF}"
        if op in (0xD, 0xF):
            return f"{names[op]}{s}{cond} r{rd}, {op2}"
        if op in (0x8, 0x9, 0xA, 0xB):
            return f"{names[op]}{cond} r{rn}, {op2}"
        return f"{names[op]}{s}{cond} r{rd}, r{rn}, {op2}"

    return f"ARM{cond} 0x{opcode:08X}"


def disassemble_thumb(op: int, pc: int) -> str:
    if (op & 0xF800) == 0xE000:
        target = (pc + 4 + (sign_extend(op & 0x7FF, 11) << 1)) & 0xFFFFFFFF
        return f"B 0x{target:08X}"
    if (op & 0xE000) == 0x2000:
        names = ["MOV", "CMP", "ADD", "SUB"]
        kind = (op >> 11) & 0x3
        rd = (op >> 8) & 0x7
        imm = op & 0xFF
        return f"{names[kind]} r{rd}, #{imm}"
    if (op & 0xFF87) == 0x4700:
        rm = (op >> 3) & 0xF
        return f"BX r{rm}"
    return f"THUMB 0x{op:04X}"


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

    def __post_init__(self) -> None:
        self.cpu = ARM7TDMI(self.memory)

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
        self.running = False
        self.frame = 0
        self.memory.reset_ram()
        self.cpu.reset()

    def run_slice(self, instructions: int = 256) -> List[str]:
        lines = []
        if not self.memory.rom:
            return ["No ROM loaded."]
        for _ in range(instructions):
            line = self.cpu.step()
            if len(lines) < 10:
                lines.append(line)
            if self.cpu.state.halted:
                self.running = False
                break
        return lines

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
        self.geometry("1120x740")
        self.minsize(960, 650)

        self.accelerator = load_cython_accelerator()
        self.emu = GBAEmulator(self.accelerator)
        self.trace_lines: List[str] = []
        self.last_frame_time = time.perf_counter()
        self.fps = 0.0

        self._build_widgets()
        self._bind_keys()
        self._log("ChatGPTGBA ready.")
        if self.accelerator.available:
            self._log("Cython accelerator: ON")
        else:
            self._log("Cython accelerator: OFF, pure Python fallback active.")
            if self.accelerator.error:
                self._log(f"Cython note: {self.accelerator.error}")
        self._draw_screen()
        self._update_panels()
        self.after(16, self._tick)

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
        self.blue_label(top, "ChatGPTGBA", big=True).pack(side=tk.LEFT)
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

        controls = tk.Frame(left, bg=BLACK)
        controls.pack(fill=tk.X, pady=8)
        for text, cmd in [
            ("Open ROM", self.open_rom),
            ("Start/Pause", self.toggle_run),
            ("Step", self.step_once),
            ("Reset", self.reset_emu),
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
            height=8,
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
            height=10,
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
        self.header_text.insert(tk.END, "ROM header will appear here.\n")
        self.header_text.configure(state=tk.DISABLED)

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
                self._update_key_text()

        def release(event: tk.Event) -> None:
            key = mapping.get(event.keysym)
            if key:
                self.emu.keys[key] = False
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
            self._log(f"Loaded ROM: {path.name}")
            self._log(f"Title: {self.emu.header.title}")
            if not self.emu.header.checksum_ok:
                self._log("Header checksum warning: ROM may be homebrew, patched, or invalid.")
            self._draw_screen()
            self._update_panels()
        except Exception as exc:
            messagebox.showerror("Open ROM failed", f"{type(exc).__name__}: {exc}")
            self._log(traceback.format_exc())

    def toggle_run(self) -> None:
        if not self.emu.memory.rom:
            messagebox.showinfo("No ROM", "Open a .gba ROM first.")
            return
        self.emu.running = not self.emu.running
        self._log("Running." if self.emu.running else "Paused.")

    def step_once(self) -> None:
        if not self.emu.memory.rom:
            messagebox.showinfo("No ROM", "Open a .gba ROM first.")
            return
        lines = self.emu.run_slice(1)
        self.trace_lines.extend(lines)
        self.trace_lines = self.trace_lines[-400:]
        self.emu.frame += 1
        self._draw_screen()
        self._update_panels()

    def reset_emu(self) -> None:
        self.emu.reset()
        self.trace_lines.clear()
        self._log("Reset.")
        self._draw_screen()
        self._update_panels()

    def show_cython_status(self) -> None:
        if self.accelerator.available:
            text = "Cython accelerator is active."
        else:
            text = "Cython accelerator is not active. Pure Python fallback is being used.\n\n" + self.accelerator.error
        messagebox.showinfo("Cython status", text)
        self._log(text.replace("\n", " "))

    def _tick(self) -> None:
        try:
            if self.emu.running:
                lines = self.emu.run_slice(300)
                self.trace_lines.extend(lines)
                self.trace_lines = self.trace_lines[-400:]
                self.emu.frame += 1
                now = time.perf_counter()
                dt = max(now - self.last_frame_time, 1e-6)
                self.last_frame_time = now
                self.fps = 1.0 / dt
                self._draw_screen()
                self._update_panels()
        finally:
            self.after(16, self._tick)

    def _update_panels(self) -> None:
        h = self.emu.header
        self.header_text.configure(state=tk.NORMAL)
        self.header_text.delete("1.0", tk.END)
        self.header_text.insert(tk.END, "ROM HEADER\n")
        self.header_text.insert(tk.END, "-" * 48 + "\n")
        for line in h.lines():
            self.header_text.insert(tk.END, line + "\n")
        self.header_text.insert(tk.END, f"CYTHON     : {'ON' if self.accelerator.available else 'OFF'}\n")
        self.header_text.configure(state=tk.DISABLED)

        self.reg_text.configure(state=tk.NORMAL)
        self.reg_text.delete("1.0", tk.END)
        self.reg_text.insert(tk.END, "CPU REGISTERS\n")
        self.reg_text.insert(tk.END, "-" * 48 + "\n")
        self.reg_text.insert(tk.END, self.emu.cpu.register_text())
        self.reg_text.configure(state=tk.DISABLED)

        self.disasm_text.configure(state=tk.NORMAL)
        self.disasm_text.delete("1.0", tk.END)
        self.disasm_text.insert(tk.END, "RECENT TRACE\n")
        self.disasm_text.insert(tk.END, "-" * 72 + "\n")
        if self.trace_lines:
            self.disasm_text.insert(tk.END, "\n".join(self.trace_lines[-160:]))
        else:
            self.disasm_text.insert(tk.END, self.emu.cpu.state.last_disasm)
        self.disasm_text.configure(state=tk.DISABLED)

        rom_name = self.emu.rom_path.name if self.emu.rom_path else "No ROM"
        run = "RUN" if self.emu.running else "PAUSE"
        self.status_var.set(
            f"{rom_name} | {run} | frame {self.emu.frame} | fps {self.fps:5.1f} | "
            f"Cython {'ON' if self.accelerator.available else 'OFF'}"
        )

    def _draw_screen(self) -> None:
        c = self.canvas
        c.delete("all")
        c.create_rectangle(0, 0, CANVAS_W, CANVAS_H, fill=BLACK, outline=MID_BLUE, width=2)

        # Scanline glow grid.
        for y in range(0, CANVAS_H, 12):
            c.create_line(0, y, CANVAS_W, y, fill="#001020")
        for x in range(0, CANVAS_W, 24):
            c.create_line(x, 0, x, CANVAS_H, fill="#000812")

        if not self.emu.memory.rom:
            self._draw_boot_screen()
            return

        seed = self.emu.render_seed()
        title = self.emu.header.title[:18]
        pc = self.emu.cpu.state.r[15] & 0xFFFFFFFF
        mode = "THUMB" if self.emu.cpu.state.cpsr & CPSR_T else "ARM"

        # ROM-reactive animated blue tile field. This is a placeholder PPU visualization.
        cols, rows = 30, 20
        tile_w = CANVAS_W / cols
        tile_h = CANVAS_H / rows
        data = self.emu.memory.rom
        n = len(data) or 1
        for row in range(rows):
            for col in range(cols):
                idx = (row * cols + col + self.emu.frame * 3) % n
                b = data[idx]
                pulse = ((seed >> ((row + col) % 24)) & 0xFF)
                level = (b ^ pulse) & 0x7F
                if level < 36:
                    continue
                blue_level = 40 + min(180, level * 2)
                color = f"#0000{blue_level:02x}"
                x0 = int(col * tile_w)
                y0 = int(row * tile_h)
                x1 = int((col + 1) * tile_w) - 1
                y1 = int((row + 1) * tile_h) - 1
                c.create_rectangle(x0, y0, x1, y1, fill=color, outline="")

        # HUD overlay.
        c.create_rectangle(14, 14, CANVAS_W - 14, 96, fill="#000713", outline=BRIGHT_BLUE, width=2)
        c.create_text(28, 30, text="CHATGPTGBA", anchor="nw", fill=BRIGHT_BLUE, font=FONT_MONO_BIG)
        c.create_text(28, 58, text=f"ROM: {title}", anchor="nw", fill=TEXT_BLUE, font=FONT_MONO)
        c.create_text(28, 78, text=f"PC: {pc:08X}  MODE: {mode}  FRAME: {self.emu.frame}", anchor="nw", fill=TEXT_BLUE, font=FONT_MONO)

        if self.emu.keys:
            pressed = [k for k, down in sorted(self.emu.keys.items()) if down]
        else:
            pressed = []
        key_text = "INPUT: " + (", ".join(pressed) if pressed else "none")
        c.create_rectangle(14, CANVAS_H - 44, CANVAS_W - 14, CANVAS_H - 14, fill="#000713", outline=DIM_BLUE, width=1)
        c.create_text(28, CANVAS_H - 36, text=key_text, anchor="nw", fill=TEXT_BLUE, font=FONT_MONO)

        if self.emu.cpu.state.halted:
            c.create_text(
                CANVAS_W // 2,
                CANVAS_H // 2,
                text="HALTED / BIOS CALL PLACEHOLDER",
                fill=WARNING_BLUE,
                font=FONT_MONO_BIG,
            )

    def _draw_boot_screen(self) -> None:
        c = self.canvas
        t = self.emu.frame
        c.create_text(CANVAS_W // 2, 88, text="CHATGPTGBA", fill=BRIGHT_BLUE, font=("Consolas", 28, "bold"))
        c.create_text(
            CANVAS_W // 2,
            134,
            text="Open a .gba ROM to inspect and step its entry code",
            fill=TEXT_BLUE,
            font=FONT_MONO,
        )
        c.create_text(
            CANVAS_W // 2,
            162,
            text="Blue buttons + blue text + black background",
            fill=TEXT_BLUE,
            font=FONT_MONO,
        )
        for i in range(48):
            angle = (i / 48.0) * math.tau + t * 0.05
            radius = 80 + 30 * math.sin(t * 0.04 + i)
            x = CANVAS_W // 2 + int(math.cos(angle) * radius)
            y = CANVAS_H // 2 + int(math.sin(angle) * radius)
            c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=BLUE, outline="")


def main() -> int:
    app = ChatGPTGBAApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
