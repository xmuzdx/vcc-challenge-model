#!/usr/bin/env python3
"""Resume the in-flight GCS upload with IPv4 + 32 MiB chunks."""

from __future__ import annotations

import socket
import time
from pathlib import Path

import httpx
from vcc import auth

_orig = socket.getaddrinfo


def _ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
    infos = _orig(host, port, socket.AF_INET, type, proto, flags)
    if not infos:
        raise socket.gaierror(socket.EAI_NONAME, "no IPv4")
    return infos


socket.getaddrinfo = _ipv4_only

CHUNK = 8 * 1024 * 1024
MAX_RETRIES = 12


def committed(url: str, total: int, client: httpx.Client) -> int:
    r = client.put(url, headers={"Content-Range": f"bytes */{total}"}, content=b"", timeout=120)
    if r.status_code in (200, 201):
        return total
    if r.status_code == 308:
        rng = r.headers.get("Range")
        if not rng:
            return 0
        return int(rng.split("-")[-1]) + 1
    raise RuntimeError(f"offset query HTTP {r.status_code}: {r.text[:200]}")


def main() -> None:
    pend = auth.list_pending_uploads("default")
    if not pend:
        raise SystemExit("no pending upload")
    rec = next(iter(pend.values()))
    url = rec["upload_url"]
    path = Path(rec["local_path"])
    total = path.stat().st_size
    t0 = time.time()
    # Local HTTP proxy drops large GCS PUTs; talk to storage directly.
    with httpx.Client(timeout=300.0, http2=False, trust_env=False) as client:
        offset = committed(url, total, client)
        sent0 = offset
        print(f"[fast] resume at {offset}/{total} ({offset / total:.1%})", flush=True)
        if offset >= total:
            print("[fast] already complete", flush=True)
            return
        attempts = 0
        with path.open("rb") as fh:
            while offset < total:
                fh.seek(offset)
                chunk = fh.read(min(CHUNK, total - offset))
                end = offset + len(chunk) - 1
                headers = {
                    "Content-Range": f"bytes {offset}-{end}/{total}",
                    "Content-Length": str(len(chunk)),
                }
                t1 = time.time()
                try:
                    r = client.put(url, headers=headers, content=chunk, timeout=300.0)
                except httpx.HTTPError as exc:
                    attempts += 1
                    if attempts > MAX_RETRIES:
                        raise RuntimeError(f"PUT failed after {MAX_RETRIES} retries: {exc}") from exc
                    time.sleep(min(2 ** attempts, 30))
                    offset = committed(url, total, client)
                    print(f"[fast] retry {attempts} after {exc!r}; offset={offset}", flush=True)
                    continue
                dt = max(time.time() - t1, 1e-6)
                if r.status_code in (200, 201):
                    offset = total
                    print(f"[fast] done {total} in {time.time() - t0:.1f}s", flush=True)
                    break
                if r.status_code == 308:
                    attempts = 0
                    rng = r.headers.get("Range")
                    offset = int(rng.split("-")[-1]) + 1 if rng else end + 1
                    inst = len(chunk) / dt / 1e6
                    avg = (offset - sent0) / max(time.time() - t0, 1e-6) / 1e6
                    eta = (total - offset) / max(avg * 1e6, 1)
                    print(
                        f"[fast] {offset / total:.1%} +{len(chunk) / 1e6:.1f}MB "
                        f"{inst:.1f} MB/s avg={avg:.1f} MB/s eta={eta / 60:.1f}m",
                        flush=True,
                    )
                    continue
                raise RuntimeError(f"PUT HTTP {r.status_code}: {r.text[:300]}")


if __name__ == "__main__":
    main()
