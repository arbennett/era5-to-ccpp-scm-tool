# JOSS manuscript

| File | Contents |
|---|---|
| `paper.md` | The manuscript |
| `paper.bib` | References — all DOIs verified against Crossref |
| `make_workflow_figure.py` | Regenerates `workflow.png` (Figure 1) |
| `make_casestudy_figure.py` | Regenerates `case_study.png` (Figure 2) |

## Regenerating the figures

Figure 1 is self-contained:

```bash
uv run --extra paper python paper/make_workflow_figure.py
```

Figure 2 reads two DEPHY drivers and two CCPP-SCM output files, so it needs an
SCM checkout in which the `era5land_US-Whs` and `era5land_US-MMS` cases have
been installed and run under `SCM_GFS_v16`:

```bash
uv run --extra paper python paper/make_casestudy_figure.py --scm-root /path/to/ccpp-scm
```

It reads:

- `<scm-root>/scm/data/processed_case_input/era5land_{US-Whs,US-MMS}_SCM_driver.nc`
- `<scm-root>/scm/run/output_era5land_{US-Whs,US-MMS}_SCM_GFS_v16/output.nc`

and prints the flux statistics quoted in the case-study section. The flux
comparison is taken over the first simulated day with the initialisation and
cold-start steps dropped; the evaporative fraction is restricted to daylight
hours, since over a multi-day run the nocturnal sensible heat flux drives an
all-hours ratio through zero. A second block reports the five-day column drift.

Panel (c) also draws the GABLS3 template soil profile, which it reads from
`fluxnet_US-Whs_SCM_driver.nc` if that case is still installed. The manuscript
compares against the template-soil runs, whose numbers come from the same
script with `--case-prefix fluxnet`; that writes `case_study_fluxnet.png` rather
than overwriting the manuscript figure.

To rebuild those cases from scratch:

```bash
era5-scm-tool run_full_pipeline \
  --start_date 2019-01-01 --end_date 2019-01-05 \
  --lat 31.7438 --lon -110.0522 \
  --case_name era5land_US-Whs --output_dir . \
  --scm_cases_dir  "$SCM_ROOT/scm/data/processed_case_input" \
  --scm_config_dir "$SCM_ROOT/scm/etc/case_config"

era5-scm-tool run_full_pipeline \
  --start_date 2019-06-15 --end_date 2019-06-19 \
  --lat 39.3232 --lon -86.4131 \
  --case_name era5land_US-MMS --output_dir . \
  --scm_cases_dir  "$SCM_ROOT/scm/data/processed_case_input" \
  --scm_config_dir "$SCM_ROOT/scm/etc/case_config"
```

The template-soil comparison cases are the same commands with `--no_land`
added, written under the `fluxnet_` prefix.

Then run each with `./run_scm.py -c <case_name> -s SCM_GFS_v16` from
`$SCM_ROOT/scm/bin`. Each case runs five days, which is 720 timesteps at the
default 600 s timestep, taken from the length of the forcing. On a login node
`run_scm.py` sets the run up and then fails at `mpirun`; run `./scm` directly
from `$SCM_ROOT/scm/run` afterwards.

## Proofing the manuscript

```bash
pandoc paper/paper.md --citeproc --bibliography=paper/paper.bib -o /tmp/paper.html
```

JOSS itself builds the PDF with
[`openjournals/inara`](https://github.com/openjournals/inara).

## Colors

Both figures use slots 1 and 2 (blue, orange) of a colorblind-safe categorical
palette, assigned to sites in fixed order.
