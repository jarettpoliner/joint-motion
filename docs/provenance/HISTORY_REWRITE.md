# Commit history rewrite (2026-11-04)

The published `main` history was rewritten once, only to remove AI co-author trailers
(`Co-Authored-By: Claude … <noreply@anthropic.com>`) from 27 commit messages.

- **Unchanged:** every commit's source tree, author, committer, timestamps and merge structure.
  Every file in the repository, including code, configs, manifests, registry entries, tables,
  figures and reports, is byte-identical. No experiment was re-run, and no result changed.
- **Changed:** commit messages lost only those trailer lines. Because the first commit had one,
  52 of the 56 commit IDs changed. The 4 LICENSE-only commits created by GitHub kept their IDs.
- **Before and after:** `main` moved from `327529dcdc1a2397f7fda4e54d0c264f372932d9` to
  `3219efa7eb19e43c8b0180c018a7efe053a36025`. Both tips have the identical tree, `4eac51f`.
  The maintainer keeps a complete offline bundle of the history before the rewrite.

## Original commit references are kept

Experimental records cite the commit that was checked out when they were made. These references
were left as written, because changing them would alter the records:

- `registry/experiments.jsonl`
- `registry/runs/*.json`
- `reports/followup_initial_motion/freeze.json`, in the `registration_commit` and
  `selection_commit` fields
- the commit lists in the reports
- `slides/content.json`

Those IDs are pre-rewrite IDs. To find the published commit with the same tree, look the ID up
in [`commit_sha_map_20261104.tsv`](commit_sha_map_20261104.tsv). The map has these columns:
`old_sha`, `new_sha`, `trailers_removed`, `rewritten`. For example:

```bash
grep '^d4d65b8' docs/provenance/commit_sha_map_20261104.tsv | cut -f2   # old short ID -> new full ID
```
