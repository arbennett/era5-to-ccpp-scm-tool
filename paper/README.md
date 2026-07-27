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
SCM checkout in which the `fluxnet_US-Whs` and `fluxnet_US-MMS` cases have been
installed and run under `SCM_GFS_v16`:

```bash
uv run --extra paper python paper/make_casestudy_figure.py --scm-root /path/to/ccpp-scm
```

It reads:

- `<scm-root>/scm/data/processed_case_input/fluxnet_{US-Whs,US-MMS}_SCM_driver.nc`
- `<scm-root>/scm/run/output_fluxnet_{US-Whs,US-MMS}_SCM_GFS_v16/output.nc`

and prints the flux statistics quoted in the case-study section.

To rebuild those cases from scratch:

```bash
era5-scm-tool run_full_pipeline \
  --start_date 2019-01-01 --end_date 2019-01-02 \
  --lat 31.7438 --lon -110.0522 \
  --case_name fluxnet_US-Whs --output_dir . \
  --scm_cases_dir  "$SCM_ROOT/scm/data/processed_case_input" \
  --scm_config_dir "$SCM_ROOT/scm/etc/case_config"

era5-scm-tool run_full_pipeline \
  --start_date 2019-06-15 --end_date 2019-06-16 \
  --lat 39.3232 --lon -86.4131 \
  --case_name fluxnet_US-MMS --output_dir . \
  --scm_cases_dir  "$SCM_ROOT/scm/data/processed_case_input" \
  --scm_config_dir "$SCM_ROOT/scm/etc/case_config"
```

then run each with `./run_scm.py -c <case_name> -s SCM_GFS_v16` from
`$SCM_ROOT/scm/bin`.

## Proofing the manuscript

```bash
pandoc paper/paper.md --citeproc --bibliography=paper/paper.bib -o /tmp/paper.html
```

JOSS itself builds the PDF with
[`openjournals/inara`](https://github.com/openjournals/inara).

## Colors

Both figures use slots 1 and 2 (blue, orange) of a colorblind-safe categorical
palette, assigned to sites in fixed order.
