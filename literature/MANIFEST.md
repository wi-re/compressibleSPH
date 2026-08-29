# Literature

The papers this codebase is built against. **The documents themselves are not
in this repository and must not be added to it** — they are third-party
copyrighted material. What is tracked here is metadata: this manifest and
`references.bib`.

`.gitignore` enforces it two ways: everything in `literature/` is ignored
except `*.md` and `*.bib`, and `*.pdf` is ignored repo-wide. Neither is a
substitute for care, but both survive a careless `git add -A`.

## How to use it

Drop the PDFs into this directory, named as the **file** column below. Nothing
else needs doing for them to be readable — `literature/` is inside the working
tree, so relative paths work and no per-file approval is needed. If you would
rather keep them outside the repo entirely, put them anywhere and record the
directory under "Location" below; absolute paths work too, at the cost of an
approval prompt per file.

Location: `literature/` (this directory).

## What is here

`have?` is maintained by the sync step below, not by hand.

| plan | bib key | file | have? | what it is |
|---|---|---|---|---|
| `[C]` | `cornelis2019` | `cornelis2019-optimized-source-term.pdf` | no | **The paper this scheme implements** (VD+PS). |
| `[BK]` | `bender2015` | `bender2015-divergence-free-sph.pdf` | no | DFSPH proper. The published CFL constant. |
| `[I]` | `ihmsen2014` | `ihmsen2014-implicit-incompressible-sph.pdf` | no | IISPH — the solver the Jacobi loop discretises. |
| `[B]` | `band2018` | `band2018-mls-pressure-boundaries.pdf` | no | MLS pressure boundaries. |
| `[BWJ23]` | `bender2023` | `bender2023-consistent-rigid-fluid-coupling.pdf` | no | The derivation behind `staticBoundary`. |
| — | `adami2012` | `adami2012-generalized-wall-bc.pdf` | no | The wall BC `[B]` Eq. 3 extrapolates from. |
| — | `adami2013` | `adami2013-transport-velocity.pdf` | no | Transport velocity. |
| — | `akinci2012` | `akinci2012-versatile-rigid-fluid-coupling.pdf` | no | The boundary volume correction. |
| — | `ihmsen2010` | `ihmsen2010-pcisph-boundary-timestep.pdf` | no | The adaptive timestep `[BK]`'s CFL descends from. |
| — | `schechter2012` | `schechter2012-ghost-sph.pdf` | no | Ghost particles for free-surface density loss. |

The first five are recorded in `DFSPH_IMPROVEMENT_PLAN.md` §5 as read in full
(in earlier sessions, from copies that were never in this repo). The rest are
listed there as unavailable.

## What each unobtained paper actually unblocks

Worth knowing which ones are worth the effort:

- **`cornelis2019`** — the largest. Its Fig. 3 (sinus amplitude) and Fig. 4
  (max density) are the shear-wave reference curves, and grading `shearWave`
  against them is the one remaining step of §4 item 8. **Also wanted: the
  setup** — domain size, resolution, `u0`, `nu`, wavenumber, time range,
  kernel and support radius — since a curve comparison is meaningless if the
  configuration differs.
- **`adami2013`** — closes §5 Q7 (background pressure), open since Part 7.
- **`akinci2012`** — would settle whether `akinciBoundaryVolume`'s divergence
  under `minShift` is this codebase misapplying a one-layer correction to a
  five-layer band, which is the current hypothesis (§2) but is inferred rather
  than read.
- **`bender2015` / `ihmsen2014` / `band2018` / `bender2023`** — already read;
  re-obtaining them mainly makes the *specific* claims re-checkable, which
  matters because this document has retracted several readings of them.
- **`schechter2012` / `adami2012` / `ihmsen2010`** — background. Nothing is
  blocked on them.

## Syncing after adding PDFs

Yes — dropping the files in and running one prompt is enough. Ask for:

> Sync `literature/`: check which expected PDFs are present, verify the BibTeX
> against them, and update the manifest.

which should do, in order:

1. **Reconcile the table.** List the directory, set `have?` per row, and report
   anything present that no row claims (unexpected name → either add a row or
   rename the file).
2. **Verify `references.bib` against the documents.** Its header carries a
   provenance warning: the entries were written from memory before any PDF
   existed here, so volume/issue/pages/DOI are unverified. Correct them from
   the actual front matter and drop the warning for entries that have been
   checked.
3. **Report what became answerable.** Cross the newly-present files against the
   list above and against `DFSPH_IMPROVEMENT_PLAN.md`'s open items, and say
   which ones are now unblocked.

Two limits worth knowing before relying on it:

- **Values read off a plotted figure are good to about two significant
  figures.** This project's standard is exact reproduction, so a
  figure-derived number should be recorded as approximate and labelled as
  such — never presented like a measured row. If a paper states the same
  values in text or a table, those are worth far more than the plot.
- **A scanned PDF with no text layer cannot be read reliably.** Prefer
  publisher or arXiv copies over scans.
