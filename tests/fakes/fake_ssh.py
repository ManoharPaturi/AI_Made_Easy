#!/usr/bin/env python3
"""Stands in for ``ssh`` in tests: runs the "remote" command with sh in $FAKE_SSH_HOME.

``FAKE_SSH_DROP=<marker file>``: the first follower connection dies (exit 255) after
two lines, like a dropped network link; later connections work.
"""
import os
import subprocess
import sys

args = sys.argv[1:]
while args and args[0] == "-o":
    args = args[2:]
host, command = args[0], args[1]
if host == "unreachable":
    print(f"ssh: connect to host {host} port 22: Connection refused", file=sys.stderr)
    sys.exit(255)
home = os.environ["FAKE_SSH_HOME"]
os.makedirs(home, exist_ok=True)
env = dict(os.environ, HOME=home)
drop = os.environ.get("FAKE_SSH_DROP")
if drop and " - " in command and not os.path.exists(drop):
    open(drop, "w", encoding="utf-8").close()
    proc = subprocess.Popen(["sh", "-c", command], cwd=home, env=env, stdin=sys.stdin,
                            stdout=subprocess.PIPE, text=True)
    for number, line in enumerate(proc.stdout):
        sys.stdout.write(line)
        sys.stdout.flush()
        if number >= 1:
            proc.kill()
            sys.exit(255)
    sys.exit(255)
os.chdir(home)
os.execvpe("sh", ["sh", "-c", command], env)
