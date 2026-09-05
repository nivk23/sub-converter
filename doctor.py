#!/usr/bin/env python3
"""Print a short environment verdict for installing sub-converter.

Run:  python3 doctor.py
Output fits on one phone screen; paste all of it when asking for help.
"""
import os
import platform
import shutil
import subprocess
import sys
import tempfile


def row(k, v):
    print(f"{k:<12} {v}")


def main():
    prefix = os.environ.get("PREFIX", "")
    termux = "com.termux" in prefix or "com.termux" in sys.executable
    osrel = ""
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    osrel = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    libc = " ".join(x for x in platform.libc_ver() if x) or "unknown"

    row("env", "Termux (plain, no proot)" if termux else (osrel or platform.platform()))
    row("arch", platform.machine())
    row("libc", libc)
    row("python", f"{platform.python_version()}  ({sys.executable})")
    row("venv", "yes" if sys.prefix != sys.base_prefix else "no")
    row("ffmpeg", shutil.which("ffmpeg") or "MISSING")

    try:
        import pip  # noqa: F401
        pipv = subprocess.run([sys.executable, "-m", "pip", "--version"],
                              capture_output=True, text=True).stdout.split()[1]
    except Exception:
        pipv = "MISSING"
    row("pip", pipv)

    wheel_ok = None
    if pipv != "MISSING":
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run(
                [sys.executable, "-m", "pip", "download", "--only-binary=:all:",
                 "--no-deps", "-q", "-d", d, "av"],
                capture_output=True, text=True)
            wheel_ok = r.returncode == 0
            tail = (r.stderr.strip().splitlines() or [""])[-1][:110]
        row("av wheel", "available" if wheel_ok else f"NONE  ({tail})")

    print()
    if termux:
        print("VERDICT: plain Termux. No PyPI wheels exist for this libc.")
        print("  Run:  proot-distro login ubuntu   (install it first if needed)")
        print("  then clone the repo again INSIDE ubuntu and rerun doctor.py there.")
    elif wheel_ok is False and sys.version_info >= (3, 14):
        print("VERDICT: Python too new for PyAV wheels.")
        print("  Run:  apt install -y python3.12 python3.12-venv")
        print("        python3.12 -m venv .venv && . .venv/bin/activate")
        print("        pip install -r requirements.txt")
    elif wheel_ok is False:
        print("VERDICT: no PyAV wheel for this platform/pip. Try: pip install -U pip")
    elif shutil.which("ffmpeg") is None:
        print("VERDICT: deps installable, but ffmpeg is missing (apt install ffmpeg).")
    else:
        print("VERDICT: looks good. Run: pip install -r requirements.txt")


if __name__ == "__main__":
    main()
