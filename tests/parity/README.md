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
exact environment contract. The exact behavior comparison skips before capture
when the Python implementation, patch version, or platform differs, with an
explicit expected-versus-actual reason; it never weakens or regenerates the
oracle. On the canonical runtime, dependency-version drift remains a failure.

Stable releases add explicit authorities without replacing history. `v0.1.0.json`
points to the immutable post-M3 baseline, while the active `v0.1.1.json` authority
points to the immutable `v0.1.0.json` authority and its exact SHA-256. Both use the
same behavioral, artifact-identity, release-identity, and repository/toolchain
classification. The v0.1.1 transition structurally parses each `stage.json`, changes
only exact `pixipixVersion` fields from `0.1.1` to `0.1.0`, reserializes with the
canonical stage metadata grammar, and requires the v0.1.0 metadata and stage-tree
hashes exactly. Arbitrary byte replacement, regex normalization, and normalization of
unrelated version-looking values are prohibited. PNG bytes and behavioral leaves must
already match without normalization.

Release gates run a non-skipping canonical-runtime precheck before the active
authority comparison. Ordinary noncanonical test runs may retain the diagnostic skip,
but a release is not eligible unless the precheck and canonical comparison both
execute successfully.

The compatibility matrix also records private symbols consumed by current tests
and smoke tooling. Those entries are behavior-preservation obligations for the
architecture refactor, not declarations of supported public Python API.
