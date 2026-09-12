#!/usr/bin/env python3
"""Installer for xg-project."""

import shutil
import subprocess
import sys
from pathlib import Path


def _has_pipx() -> bool:
    return shutil.which("pipx") is not None


def _install_pipx() -> None:
    print("pipx not found. Installing via pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pipx"], check=True)
    print("pipx installed.")


def install_global() -> None:
    """Install xg user-wide via pipx."""
    root = Path(__file__).parent.resolve()

    print("xg installer (user-wide via pipx)")
    print("=" * 40)

    if not _has_pipx():
        _install_pipx()

    print("\nInstalling xg-project...")
    subprocess.run(
        ["pipx", "install", "-e", str(root)],
        check=True,
    )

    print("\n" + "=" * 40)
    print("Installation complete!")
    print()
    print("xg is now available globally:")
    print("  xg          # show help")
    print("  xg --start  # start with PTY")
    print()
    print("To uninstall:")
    print("  pipx uninstall xg-project")


def install_local() -> None:
    """Install into a local .venv (dev mode)."""
    root = Path(__file__).parent.resolve()
    venv_dir = root / ".venv"

    print("xg installer (local .venv)")
    print("=" * 40)

    # 1. Create venv if missing
    if not venv_dir.exists():
        print("\n[1/2] Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    else:
        print("\n[1/2] Virtual environment already exists, skipping.")

    # 2. Determine Python inside venv. Some Python distributions do not
    # include pip/ensurepip, so prefer uv when it is available.
    if sys.platform == "win32":
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    print("\n[2/2] Installing xg-project and dependencies...")
    uv = shutil.which("uv")
    if uv:
        subprocess.run(
            [uv, "pip", "install", "--python", str(venv_python), "-e", str(root)],
            check=True,
        )
    else:
        # Fall back to pip where the Python distribution provides it.
        pip_check = subprocess.run(
            [str(venv_python), "-m", "pip", "--version"],
            capture_output=True,
        )
        if pip_check.returncode != 0:
            raise RuntimeError(
                "pip is unavailable in .venv and uv is not installed. "
                "Install uv or recreate the environment with a Python "
                "distribution that includes pip."
            )
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-e", str(root)],
            check=True,
        )

    # Done
    if sys.platform == "win32":
        bin_dir = venv_dir / "Scripts"
    else:
        bin_dir = venv_dir / "bin"

    print("\n" + "=" * 40)
    print("Installation complete!")
    print()
    print("To use xg, run:")
    print(f"  {bin_dir / 'xg'}")
    print()
    print("Or activate the venv first:")
    if sys.platform == "win32":
        print(f"  {venv_dir / 'Scripts' / 'activate'}")
    else:
        print(f"  source {venv_dir / 'bin' / 'activate'}")
    print("  xg")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Installer for xg-project")
    parser.add_argument(
        "--global", dest="global_install", action="store_true",
        help="Install user-wide via pipx (default)",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Install into local .venv (dev mode)",
    )
    args = parser.parse_args()

    if args.local:
        install_local()
    else:
        install_global()


if __name__ == "__main__":
    main()
