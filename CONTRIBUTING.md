# Contributing

Contributions are welcome, whether that is a bug report, a new template, or a
change to how the forcings are derived.

## Reporting a problem

Open an issue at
<https://github.com/arbennett/era5-to-ccpp-scm-tool/issues>. For anything that
produces a wrong number rather than an exception, please include:

- the exact command or Python call,
- the site coordinates and date range,
- `--source` (`aws` or `glade`), since the two backends read different copies
  of the archive,
- the versions of this package, `xarray`, and your NetCDF engine.

If a generated case fails inside CCPP-SCM rather than during generation, the
SCM version and physics suite matter too.

## Asking a question

Use the issue tracker and label it a question. Questions about the SCM itself —
building it, physics suites, namelist options — belong with the
[CCPP-SCM project](https://github.com/NCAR/ccpp-scm).

## Development setup

```bash
git clone https://github.com/arbennett/era5-to-ccpp-scm-tool.git
cd era5-to-ccpp-scm-tool
uv sync --extra test
uv run pytest
```

The test suite is network-free: it runs against a small ERA5 extraction
committed under `casegen_walnut_gulch/data`, so it works offline and in CI.

## Submitting a change

1. Open an issue first for anything that changes the science — a new forcing
   term, a different discretisation, a change to the DEPHY output. These
   deserve discussion before code.
2. Branch off `main`.
3. Add a test. For a bug fix, add the test that fails before your change; for a
   new derivation, an analytic case with a known answer is worth more than a
   regression against current output.
4. Run `uv run pytest` and make sure the suite passes.
5. Open a pull request describing what changed and why.

## Things worth knowing

- **The archive is chunked one global field per time step.** Reading a 3x3
  stencil costs the same as reading the whole file, so anything that loops over
  sites should instead batch them into a single pass. See
  `download_era5_sites`.
- **HDF5 is not thread-safe.** Concurrency in the downloader is across
  processes, deliberately. Threads will segfault the interpreter.
- **Static land-surface fields come from the GABLS3 template.** Soil state,
  vegetation and albedo are not site-specific. Replacing this with real
  site data is the most valuable open contribution.

## Code of conduct

Be civil and constructive. Harassment of any kind is not acceptable.
