# Myriad launchers

These are the final launchers retained from the audited Myriad snapshot.

| Path | Purpose |
| --- | --- |
| `celery/relnet_cpu.sh` | CPU resources for the shared Celery job |
| `celery/relnet_gpu.sh` | GPU resources for the shared Celery job |
| `celery/run_celery_job.sh` | Shared manager, worker and sweep logic |
| `preference/submit_preference_conditioned.sh` | Standard conditioned sweep |
| `preference/submit_preference_density_sweep.sh` | Density-conditioned sweep |
| `preference/relnet_cpu_preference_conditioned.sh` | One conditioned model |
| `preference/relnet_cpu_density_calibration.sh` | One density calibration |
| `single/submit_one_model.sh` | Submit one specialist model |
| `single/relnet_cpu_single_task.sh` | Run one model without Celery |
| `single/verify_single_task.py` | Validate a completed single-model job |

The scripts preserve the Myriad defaults used for the dissertation.  Set
`RELNET_SCRATCH` to use another scratch root.  The default is
`/home/ucabggs/Scratch`.

Create an untracked `relnet.env` from `relnet_example.env` before
submitting jobs.  Never commit that file.
