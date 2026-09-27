# E06-E08 historical sweep audit

Run:

```bash
/home/peter/miniconda3/envs/fb-env/bin/python -m paper_reproduction.audit_historical_sweep_logs
```

The audit inventories and hashes every surviving `priors_*` stdout log,
parses its match rows, records first-200 metrics, and extracts the first
match's decision-minute pattern.

The logs do not form the exact paper sweeps:

- E06: one- and five-minute traces exist, but no authenticated ten-minute run.
- E07: filenames suggest 1/3/5/7 search settings, not required 1/5/10/30/50.
- E08: suffixes suggest 125/250/500-era configurations, not 0/40/200/400.
- The single checked-in `priors.py` was edited in place and therefore cannot
  authenticate the configuration that generated each older log.

Consequently these files are preserved as historical audit evidence only.
The explicit-config modern sweep remains necessary.
