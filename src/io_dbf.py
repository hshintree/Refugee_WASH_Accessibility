"""Minimal dBASE III/V (.dbf) reader for shapefile attribute tables.

We use this to pull per-latrine capacity columns like `LT`, `LT_all_gen`,
`LT_Male`, `LT_Female` from `WASH_Latrine_20220531.dbf`. No heavy deps.
"""

from __future__ import annotations

import struct
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _read_dbf_bytes(data: bytes) -> pd.DataFrame:
    # Header
    n_records = struct.unpack("<I", data[4:8])[0]
    header_len = struct.unpack("<H", data[8:10])[0]
    record_len = struct.unpack("<H", data[10:12])[0]

    fields: list[tuple[str, str, int, int]] = []  # (name, type, length, decimals)
    off = 32
    while data[off] != 0x0D:
        name = data[off : off + 11].split(b"\x00", 1)[0].decode("latin-1").strip()
        ftype = data[off + 11 : off + 12].decode("latin-1")
        flen = data[off + 16]
        fdec = data[off + 17]
        fields.append((name, ftype, flen, fdec))
        off += 32
    rec_start = header_len

    rows: list[dict] = []
    for i in range(n_records):
        base = rec_start + i * record_len
        if base + record_len > len(data):
            break
        if data[base : base + 1] == b"*":  # deleted
            continue
        row: dict = {}
        col_off = base + 1
        for name, ftype, flen, fdec in fields:
            raw = data[col_off : col_off + flen]
            col_off += flen
            s = raw.decode("latin-1", errors="replace").strip()
            if ftype in ("N", "F"):  # numeric / float
                if s == "" or s == "*":
                    row[name] = np.nan
                else:
                    try:
                        row[name] = float(s) if (fdec or "." in s) else int(s)
                    except ValueError:
                        row[name] = np.nan
            elif ftype == "L":
                row[name] = s.upper() in ("T", "Y", "1")
            else:
                row[name] = s
        rows.append(row)
    return pd.DataFrame(rows)


def read_dbf(path: Path) -> pd.DataFrame:
    return _read_dbf_bytes(path.read_bytes())


def read_dbf_from_zip(zip_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        return _read_dbf_bytes(zf.read(member))
