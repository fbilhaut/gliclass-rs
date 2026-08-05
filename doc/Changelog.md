# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-08-05

### Added

* Add `GLiClass::new_with_runtime()`, a convenience constructor that accepts RuntimeParameters so callers can specify an execution provider, typically for GPU execution provider support (see [PR#4](https://github.com/fbilhaut/gliclass-rs/pull/4)).

### Changed

- Switch to `orp` version `1.0.0`.
- Switch to `composable` version `1.0.0`.


## [0.9.1] - 2026-07-21

### Added

* Add the ability to load the `Parameters` from a `config.json` file as provided with the models.
* Add transitive feature flags in `Cargo.toml` to allow for `orp`/`ort` GPU settings to be set.

### Changed

* Switch to `orp` snapshot


## [0.9.0] - 2025-03-30

Initial release.
