"""Minimal TFRecord reader (no TensorFlow dependency).

Format (TensorFlow `io/record_writer.h`):
    uint64 length (little endian)
    uint32 masked_crc32c(length)
    bytes  data[length]
    uint32 masked_crc32c(data)
masked_crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8  (mod 2**32)
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterator

_CRC_TABLE = None


def _crc32c_table():
    global _CRC_TABLE
    if _CRC_TABLE is None:
        poly = 0x82F63B78
        tbl = []
        for i in range(256):
            c = i
            for _ in range(8):
                c = (c >> 1) ^ poly if c & 1 else c >> 1
            tbl.append(c)
        _CRC_TABLE = tbl
    return _CRC_TABLE


def crc32c(data: bytes) -> int:
    try:  # fast path if available
        import google_crc32c  # type: ignore

        return int.from_bytes(google_crc32c.Checksum(data).digest(), "big")
    except Exception:
        pass
    tbl = _crc32c_table()
    c = 0xFFFFFFFF
    for b in data:
        c = tbl[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def masked_crc32c(data: bytes) -> int:
    c = crc32c(data)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def iter_tfrecord(path: str | Path, check_crc: bool = True) -> Iterator[bytes]:
    """Yield raw record payloads. CRC of the length header is always checked (cheap);
    the data CRC is checked when `check_crc` is True."""
    with open(path, "rb") as f:
        while True:
            hdr = f.read(12)
            if not hdr:
                return
            if len(hdr) < 12:
                raise IOError(f"truncated tfrecord header in {path}")
            (length,) = struct.unpack("<Q", hdr[:8])
            (len_crc,) = struct.unpack("<I", hdr[8:12])
            if masked_crc32c(hdr[:8]) != len_crc:
                raise IOError(f"length CRC mismatch in {path}")
            data = f.read(length)
            tail = f.read(4)
            if len(data) < length or len(tail) < 4:
                raise IOError(f"truncated tfrecord payload in {path}")
            if check_crc:
                (data_crc,) = struct.unpack("<I", tail)
                if masked_crc32c(data) != data_crc:
                    raise IOError(f"data CRC mismatch in {path}")
            yield data


def write_tfrecord(path: str | Path, records: list[bytes]) -> None:
    """Writer used for synthetic fixtures and tests."""
    with open(path, "wb") as f:
        for data in records:
            hdr = struct.pack("<Q", len(data))
            f.write(hdr)
            f.write(struct.pack("<I", masked_crc32c(hdr)))
            f.write(data)
            f.write(struct.pack("<I", masked_crc32c(data)))
