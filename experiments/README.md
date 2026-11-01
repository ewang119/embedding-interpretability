# Experiments

Checks on the probe pipeline (`probe/`). Each script writes its raw output
to `experiments/data/` (not committed) and its write-up to
`experiments/findings/`.

| Script | Question | Write-up |
|---|---|---|
| `build_reuters_lsa.py` + `probe.run` | Does the pipeline work on real text? | `findings/real_text_check.md` |
| `sample_size.py` | How many labeled chunks does each concept need? | `findings/sample_size.md` |
| `rotation_control.py` | Do individual positions mean anything, or would any basis do? | `findings/rotation_control.md` |
