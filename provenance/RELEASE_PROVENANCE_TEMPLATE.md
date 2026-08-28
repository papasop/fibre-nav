# Release provenance completion

The evidence content is frozen for the following release tag:

```text
evidence-v1.2.1-moving-fibre-f16
```

The source commit SHA cannot be truthfully filled before the repository commit
exists. It must also not be embedded into the same commit whose SHA it claims
to identify, because changing the tracked file changes that SHA.

After copying this archive into the repository, commit it and record:

```bash
git add .
git commit -m "Add v3.2c SI run record and prospective confirmation"
git rev-parse HEAD
git tag -a evidence-v1.2.1-moving-fibre-f16 -m "Moving-Fibre F16 v3.2c evidence"
git push origin HEAD
git push origin evidence-v1.2.1-moving-fibre-f16
```

Fill the manuscript/SI metadata after `git rev-parse HEAD` returns:

```text
Repository commit: <40-character SHA returned by git rev-parse HEAD>
Release tag: evidence-v1.2.1-moving-fibre-f16
```

The release tag should point directly to that commit. Verify with:

```bash
git rev-list -n 1 evidence-v1.2.1-moving-fibre-f16
git rev-parse HEAD
```

The two outputs must match. Put the returned SHA in the paper or release notes,
not by rewriting this tracked template after tagging.
