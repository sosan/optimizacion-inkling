# launch_bench.ps1 - run the cross-platform Python benchmark with the repo defaults.
# Equivalent to: python tests/bench_perf.py cli --ctx 344064
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $root
$py = 'python'
if (-not (Get-Command $py -ErrorAction SilentlyContinue)) { $py = 'py' }
& $py tests/bench_perf.py cli --ctx 344064 @args
