# Publishing this repository to GitHub

Two options. Option A is what you want unless something has gone wrong.

## Option A — git init from the files in this folder

This works directly from the files you can see in this folder. The `.git`
that Cowork created is incomplete (the sandbox could not finalise the
initial commit on the mounted filesystem), so the first step is to wipe
it and start fresh.

From a regular terminal on your machine, in this folder:

```bash
# Clean out the Cowork-side .git stub
rm -rf .git

# Initialise, stage, commit
git init -b main
git config user.email "timmaeus@gmail.com"
git config user.name  "Timothy Graham"
git add .
git commit -m "Initial commit: replication package for the Trump/Truth Social paper"

# Publish to a new GitHub repo (requires the gh CLI to be authenticated)
gh repo create trump-truth-social-replication \
    --public \
    --source . \
    --description "Reproducible replication package for 'Trump, Truth Social, and Moving the Market'" \
    --push
```

That's it. The repo will appear at
`https://github.com/<your-username>/trump-truth-social-replication`.

If you don't have `gh` installed:

```bash
# Same first six commands as above, then:
git remote add origin git@github.com:<your-username>/trump-truth-social-replication.git
git push -u origin main
```

after creating the empty repo on github.com manually.

## Option B — clone from the supplied git bundle

A complete `.git` history is shipped alongside this folder as
`trump-truth-social-replication-v4.bundle`. It is a single squashed
commit covering the full replication package: 17 numbered pipeline
scripts (data collection through the cross-asset BTC/ETH placebo
around oil events), all data and results, the compiled 23-page
report, and all docs. Compared to v3, v4 adds the signal-overlay
timeline (Fig 7), the per-event P&L concentration / burst-structure
chart (Fig 8), the cross-asset crypto placebo (Fig 9), the supporting
Coinbase data and analysis scripts (16, 17), and a new §5.4
"Cross-asset placebo" subsection.

(Three older bundles — `.bundle`, `-v2.bundle`, `-v3.bundle` — are
also present and can be deleted; use the v4 file.)

To use the bundle:

```bash
# From the folder *containing* trump-truth-social-replication/
git clone trump-truth-social-replication-v4.bundle trump-truth-social-replication-fresh
cd trump-truth-social-replication-fresh

# Confirm the commit is there
git log --oneline

# Then push as in Option A above
gh repo create trump-truth-social-replication --public --source . --push
```

The bundle is a single-file packaging of the entire git repo (history,
objects, refs) — handy if Option A has any issues.
