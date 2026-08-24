# PM metric routing and interpretation

Query metrics on the target device. Names and supported sampling suffixes vary
by architecture and NCU release.

For GB110/CC 10.3, the NCU 2026.1 `PmSampling` section uses domains such as:

- `TPC.TriageCompute`: SM active, instruction pipelines, Tensor/HMMA.
- `SM_A.TriageCompute`: Tensor Memory, L1 data pipe, shared/LGDS wavefronts.
- `SM_B.TriageCompute`: conventional L1 sector hit/lookup activity.
- `LTS.TriageCompute`: L2 hit and throughput.
- `FBSP.TriageCompute`: DRAM read/write/total throughput.

Interpretation constraints:

- PM values are interval statistics at correlation timestamps, not
  transaction start/end events or instantaneous telemetry.
- Conventional L1 hit can be zero/unavailable for TMA-heavy kernels. Require
  nonzero lookup hit/miss activity before interpreting the ratio.
- L2 and DRAM are shared GPU domains. Their time series cannot identify an
  owning SM or CTA.
- Shared-memory PM data may be wavefront/data-pipe percent-of-peak rather than
  byte bandwidth. Do not rename it bandwidth in GB/s.
- DRAM percent-of-peak can become estimated GB/s only with an explicitly
  supplied sustained peak matching the metric definition.
- Metrics collected in different replay passes must be aligned separately and
  keep pass provenance. Do not imply same-execution simultaneity.
