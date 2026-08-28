# Turbofan Engine Maintenance & Prognostics Guide (Sample)

## RUL Interpretation
Remaining Useful Life (RUL) is the estimated number of operating cycles before an
engine reaches a failure threshold. The model reports RUL with a health index (0-100%).
- Health index above 40%: normal operation, continue routine monitoring.
- Health index 15-40% (WARNING): schedule maintenance in the next planned window.
- Health index below 15% (CRITICAL): remove from service and inspect before next cycle.

## Common Failure Modes
HPC Degradation: rising HPC outlet temperature and fuel flow, falling HPC outlet
pressure. Mitigation: water-wash for fouling; blade inspection for erosion.
Fan Module Degradation: fan speed drift and increased vibration. Mitigation: balance
check, blade dressing or replacement.
Bearing Wear: rising vibration and secondary temperature rise. Mitigation: oil-debris
monitoring, bearing replacement at threshold.
Seal Leakage: declining static pressures with compensating fuel-flow increase.
Mitigation: seal inspection and replacement.

## Maintenance Actions by Alert Level
OK: continue scheduled line maintenance, log sensor trends.
WARNING: review the RUL trend, localize the failure mode from the dominant sensor
signature, schedule a borescope inspection, increase monitoring cadence.
CRITICAL: remove the engine from service, perform borescope and oil-debris analysis,
replace indicated components, do not return to service until RUL recovers.

## Preventing Unplanned Downtime
Unplanned in-service failure is the costliest outcome. Condition-based maintenance
driven by RUL prediction converts these into planned events. The model penalizes late
predictions more heavily than early ones, reflecting the asymmetric cost of downtime.