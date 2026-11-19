"""Run a command with its output to a file; record its peak resident size (LARES-932 tests/run.sh).

usage: python3 -I peak.py <out> <command...>   writes the command's stdout and stderr to <out> and
`PEAK <KB>` to <out>.peak; exits with the command's exit status (a signal is 128 + its number).
"""
import resource
import subprocess
import sys

with open(sys.argv[1], "w") as out:
    r = subprocess.run(sys.argv[2:], stdout=out, stderr=subprocess.STDOUT)
with open(sys.argv[1] + ".peak", "w") as p:
    p.write(f"PEAK {resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}\n")
sys.exit(r.returncode if r.returncode >= 0 else 128 - r.returncode)
