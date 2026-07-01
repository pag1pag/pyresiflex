# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - yyyy-mm-dd

### TODO

- Propagate uncertainty.

### Added

### Changed

### Fixed

## [0.2.0] - 2026-07-07

### Added

- Introduce Shannon-Nyquist criterion when using complex steady solver, replacing hard coded frequency/time-step. (#13)
- Take care of infinite impedance when omega=0 for a capacitance (#14)
- Add a lot more tests, with docstrings to explain what they are doing => Coverage is now >99% (#14)

### Changed

- Examples have been optimized to take less time to run (#11)
- For complex steady solver, coefficients are precomputed to accelerate the computation (#11)
- Update README, adding a small example and more links (#12)
- Update docstring of class, adding example and minigallery (#12)

### Fixed

..

## [0.1.0] - 2026-07-07

### Added

- **Voltage and current computations from knowledge of load resistance.**

  With the following assumptions:

  - the transmission line is perfect (no loss nor dispersion),
  - the load is a time-varying resistance, with Ohm's law being valid: $V(L, t) = R_l(t) I(L, t)$,
  - the generator is composed of a pulser with an internal resistance: $V(0, t) = V_g(t) - R_g I(0, t)$

  and with the following information:

  - generator resistance $R_g$ and pulser waveform $V_g$,
  - cable length $L$, impedance $Z_c$ and wave velocity $c$,
  - load impedance evolution in time $R_l(t)$,

  the following can be computed:

  - voltage at any time at any point of the transmission line: $V(x, t)$
  - current at any time at any point of the transmission line: $I(x, t)$
  - energy at any time at any point of the transmission line: $E(x, t)$

- **Resistance reconstruction from knowledge of measured voltage OR measured current.**

  With the following assumptions:

  - the transmission line is perfect (no loss nor dispersion),
  - the load is a time-varying resistance, with Ohm's law being valid: $V(L, t) = R_l(t) I(L, t)$,
  - the generator is composed of a pulser with an internal resistance: $V(0, t) = V_g(t) - R_g I(0, t)$

  and with the following information:

  - generator resistance $R_g$ and **pulser waveform $V_g$**,
  - cable length $L$, impedance $Z_c$ and wave velocity $c$,
  - a measured voltage $V\_\\text{meas}(t)$ **OR** a measured current $I\_\\text{meas}(t)$,

  the following can be computed:

  - load resistance evolution with time.

- **Resistance reconstruction from knowledge of measured voltage and measured current.**

  With the following assumptions:

  - the transmission line is perfect (no loss nor dispersion),
  - the load is a time-varying resistance, with Ohm's law being valid: $V(L, t) = R_l(t) I(L, t)$,
  - the generator is composed of a pulser with an internal resistance: $V(0, t) = V_g(t) - R_g I(0, t)$

  and with the following information:

  - generator resistance $R_g$,
  - cable length $L$, impedance $Z_c$ and wave velocity $c$,
  - a measured voltage $V\_\\text{meas}(t)$ **AND** a measured current $I\_\\text{meas}(t)$,

  the following can be computed:

  - load resistance evolution with time.

- **Validation and verification in tests to reproduce exact expected waveforms.**

- **Easy project installation with `uv`, `just` and `GitHub Codespaces`.**

- **Add gallery examples, including:**

  - Comparison with real and numerical experiments, including time-varying resistance and steady complex impedance.
  - Sensitivity analysis with input parameters,
  - Video to see electrical waves propagating along a cable.

- **Add documentation with `sphinx` (accessible online at `pag1pag.github.io/pyresiflex/`).**
