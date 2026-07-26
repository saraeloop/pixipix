# Post-M3 parity baseline

`baseline/post-m3.json` freezes selected observable behavior from the exact clean
M3 merge commit `aace7d9ac5fd4ba43c3315afd2f8eceb582d9020`.

The baseline covers complete Pixi and neutral robot lineages, exact CLI process
results, every published artifact, and structural direct-Python publisher
results. Artifact entries contain relative paths, byte lengths, and SHA-256
digests. Stage-tree hashes are derived from each sorted relative path encoded as
UTF-8, one NUL byte, and the unmodified file bytes.

Run the verifier with:

```bash
uv run python -B -m pytest tests/parity/test_parity.py -p no:cacheprovider
```

Normal tests never regenerate the baseline. To reproduce it for inspection,
choose a new destination that does not already exist:

```bash
uv run python -B -m tests.parity.generate_baseline /tmp/post-m3-reproduced.json
```

The generator creates a temporary detached Git worktree, verifies that its HEAD
and clean status match the authority commit, installs that checkout non-editably
into a disposable locked environment, runs with caller import paths and user
site-packages disabled, and removes the worktree. Dependency installation is
offline: run the repository's standard locked synchronization first so the
required wheels are present in the uv cache. It refuses to overwrite any
destination and has no update, bless, approve, or accept mode.

Architecture-refactor slices may not change the tracked baseline. An intentional
behavior change requires separate authority and review of old and new artifact
evidence. This baseline is authoritative only for its recorded Python,
dependency, and platform environment; it does not claim cross-platform byte
equality or cover unexercised behavior. The Python patch version is part of that
exact environment contract; a different patch version fails comparison rather
than silently weakening or regenerating the oracle.

The compatibility matrix also records private symbols consumed by current tests
and smoke tooling. Those entries are behavior-preservation obligations for the
architecture refactor, not declarations of supported public Python API.
