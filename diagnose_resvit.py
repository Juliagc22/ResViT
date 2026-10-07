#!/usr/bin/env python3
"""External diagnostics for ResViT training.

This script does not import or modify ResViT source files.  It can:
  * check every image that the aligned dataset may load; and
  * run any training command while copying terminal output and errors to a log.
"""

import argparse
from datetime import datetime
from pathlib import Path
import os
import platform
import subprocess
import sys
import traceback


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".ppm", ".tif", ".tiff"}


def check_images(dataroot, phases):
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is required. Activate the same virtual environment used for training.", file=sys.stderr)
        return 2

    root = Path(dataroot)
    failures = []
    image_count = 0

    for phase in phases:
        directory = root / phase
        if not directory.is_dir():
            print("[SKIP] Directory does not exist: %s" % directory)
            continue

        files = sorted(path for path in directory.rglob("*")
                       if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        print("[%s] Checking %d images in %s" % (phase, len(files), directory))

        for index, path in enumerate(files):
            try:
                # verify() checks file structure; reopening and load() decodes pixels.
                with Image.open(str(path)) as image:
                    image.verify()
                with Image.open(str(path)) as image:
                    image.load()
                    image.convert("RGB")
                image_count += 1
            except Exception as error:
                failures.append((phase, index, path, error))
                print("[ERROR] phase=%s index=%d file=%s" % (phase, index, path))
                print("        %s: %s" % (type(error).__name__, error))

    print("\nChecked images: %d" % image_count)
    print("Unreadable images: %d" % len(failures))
    return 1 if failures else 0


def write_header(log_file, command):
    header = [
        "\n========== ResViT diagnostic session ==========" ,
        "Started: %s" % datetime.now().isoformat(timespec="seconds"),
        "Working directory: %s" % os.getcwd(),
        "Python: %s" % sys.version.replace("\n", " "),
        "Platform: %s" % platform.platform(),
        "Command: %s" % " ".join(command),
        "================================================\n",
    ]
    log_file.write("\n".join(header) + "\n")
    log_file.flush()


def run_and_log(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8", errors="backslashreplace") as log_file:
        write_header(log_file, command)
        print("Diagnostic log: %s" % log_path.resolve())
        print("Running: %s\n" % " ".join(command))

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
            )
            for line in iter(process.stdout.readline, ""):
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
            process.stdout.close()
            return_code = process.wait()
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            log_file.write("\nInterrupted by user.\n")
            return 130
        except Exception:
            traceback.print_exc()
            traceback.print_exc(file=log_file)
            return 1

        footer = "\nFinished: %s | Exit code: %d\n" % (
            datetime.now().isoformat(timespec="seconds"), return_code)
        print(footer, end="")
        log_file.write(footer)
        return return_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)

    checker = commands.add_parser("check-images", help="find unreadable images before training")
    checker.add_argument("--dataroot", required=True, help="same dataroot passed to train.py")
    checker.add_argument("--phases", nargs="+", default=["train", "val"],
                         help="dataset subdirectories to check (default: train val)")

    runner = commands.add_parser("run", help="run a command and save all terminal output")
    runner.add_argument("--log-file", default=None,
                        help="output text file (default: diagnostics/train_YYYYMMDD_HHMMSS.txt)")
    runner.add_argument("command", nargs=argparse.REMAINDER,
                        help="command to execute; write -- before it")

    args = parser.parse_args()
    if args.action == "check-images":
        return check_images(args.dataroot, args.phases)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("run requires a command, e.g. run -- python train.py --dataroot ...")

    if args.log_file:
        log_path = Path(args.log_file)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path("diagnostics") / ("train_%s.txt" % stamp)
    return run_and_log(command, log_path)


if __name__ == "__main__":
    sys.exit(main())
