# Reproducibility boundary

The repository exposes the study's analytical logic, synthetic input schemas,
tested components and portable notebooks for the privacy safe source data
release. It is not the restricted production environment used for the 82-city
analysis.

| Component | Public implementation | Shareable result regeneration | Restricted production input |
|---|---:|---:|---:|
| Stay, OD and home inference | Yes, synthetic demo | No | Raw phone pings |
| Transport mode inference | Yes, train/apply API | No | Full trip features and AMap queries |
| Route reconstruction/validation | Yes, cached demo | No | Full cached AMap routes |
| Car and heavy truck exposure | Formula and processing functions | Daily derived indices only | Five-minute routes/vehicle fields |
| SWI, DWI and DCI | Yes, including city ECDF normalization | Street and neighborhood daily summaries | Full segment time inputs |
| Neighborhood measures | Yes, both representation rules | Released neighborhood summaries | Home coordinates/catchment joins |
| Route level PE | Yes, synthetic paths evaluated by time | Released aggregate resident curves | Reconstructed individual routes |
| Policy scenarios | Yes, residential catchments only | Released summaries for four cities | Full networks/trips/calibration fields |
| Main result figures | Figure 1 through 4 notebooks | Where released fields suffice | Some cartographic and production assets |

The raw phone records and heavy truck GPS trajectories cannot be redistributed
because of privacy, provider agreements and regulatory restrictions. The
released DWI files are daily summaries, not the complete five-minute
segment time exposure field. Therefore a third party can inspect and exercise
the method and regenerate shareable summaries, but cannot independently
recreate all 1.50 billion route experiences or every journal panel from raw
inputs. The portable Figure 1 city panel uses released coordinates instead of
the unavailable production boundary layers.

Policy perturbations are restricted to explicitly defined residential catchments.
