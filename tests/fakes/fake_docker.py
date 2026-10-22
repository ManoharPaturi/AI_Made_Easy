#!/usr/bin/env python3
"""Stands in for ``docker`` in tests: ``run`` executes the command on this machine
with the -v mount's host folder in place of the container path."""
import os
import subprocess
import sys

args = sys.argv[1:]
log = os.environ.get("FAKE_DOCKER_LOG")
if log:
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(" ".join(args) + "\n")
if args[0] == "version":
    print("27.0-fake")
    sys.exit(0)
if args[:2] == ["image", "inspect"]:
    sys.exit(1)  # never cached: every test sees a build
if args[0] in ("build", "stop"):
    sys.stdin.read() if args[0] == "build" else None
    sys.exit(0)
assert args[0] == "run", args
args = args[1:]
mounts, image_and_command = {}, []
while args:
    flag = args.pop(0)
    if flag == "--rm":
        continue
    if flag in ("--name", "--gpus", "-w"):
        args.pop(0)
        continue
    if flag == "-v":
        host, container = args.pop(0).rsplit(":", 1)
        mounts[container] = host
        continue
    image_and_command = [flag, *args]
    break
command = image_and_command[1:]
command = [sys.executable if c == "python" else c for c in command]
for container, host in mounts.items():
    command = [c.replace(container, host) for c in command]
workdir = next(iter(mounts.values()))
sys.exit(subprocess.call(command, cwd=workdir))
