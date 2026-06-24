#!/usr/bin/env python3
"""Inject a backend marker into a wheel's *build tag* so the three backend
variants (cpu / cuda / rocm) of the same version don't collide as Release
assets, and so `pip install <url>` lands the right native build.

A wheel filename is:
    {distribution}-{version}[-{build tag}]-{python}-{abi}-{platform}.whl
PEP 427 requires the build tag to start with a digit, so we use a tag like
    0_cpu / 0_cuda12x / 0_rocm6x
which sorts/parses cleanly and is visible in the filename, e.g.
    torch_strumpack-0.0.1.dev0-0_cpu-cp312-cp312-manylinux_2_28_x86_64.whl

Usage:
    python tag_wheel.py <backend-tag> <wheel-dir> [<out-dir>]
"""
from __future__ import annotations

import re
import shutil
import sys
import zipfile
from pathlib import Path

WHEEL_RE = re.compile(
    r"^(?P<dist>.+?)-(?P<ver>.+?)(?:-(?P<build>\d[^-]*))?"
    r"-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>.+)\.whl$"
)


def retag(whl: Path, backend: str, out_dir: Path) -> Path:
    m = WHEEL_RE.match(whl.name)
    if not m:
        raise SystemExit(f"unrecognised wheel name: {whl.name}")
    g = m.groupdict()
    build = f"0_{backend}"  # PEP 427: build tag must start with a digit
    new_name = f"{g['dist']}-{g['ver']}-{build}-{g['py']}-{g['abi']}-{g['plat']}.whl"
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / new_name
    shutil.copy2(whl, dst)

    # Update the build tag inside the wheel's WHEEL metadata too, so
    # `pip` and `wheel unpack` stay consistent with the filename.
    _rewrite_build_in_wheel(dst, build)
    print(f"{whl.name}  ->  {dst.name}")
    return dst


def _rewrite_build_in_wheel(whl: Path, build: str) -> None:
    tmp = whl.with_suffix(".whl.tmp")
    with zipfile.ZipFile(whl) as zin, zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED
    ) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith(".dist-info/WHEEL"):
                text = data.decode("utf-8")
                if "Build:" in text:
                    text = re.sub(r"(?m)^Build:.*$", f"Build: {build}", text)
                else:
                    text = text.rstrip("\n") + f"\nBuild: {build}\n"
                data = text.encode("utf-8")
            zout.writestr(item, data)
    tmp.replace(whl)


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    backend = argv[1]
    wheel_dir = Path(argv[2])
    out_dir = Path(argv[3]) if len(argv) > 3 else wheel_dir
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        raise SystemExit(f"no wheels found in {wheel_dir}")
    for whl in wheels:
        retag(whl, backend, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
