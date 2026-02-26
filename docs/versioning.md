# Versioning Policy

`tldecpy` follows Semantic Versioning (`MAJOR.MINOR.PATCH`) and PEP 440-compliant packaging metadata.

## Public API stability

The package-level API exposed in `tldecpy/__init__.py` is considered public and stable within the same major series.

## Release semantics

- `MAJOR` (`2.0.0`): backward-incompatible API changes.
- `MINOR` (`1.1.0`): backward-compatible feature additions.
- `PATCH` (`1.0.1`): backward-compatible bug fixes.

## Deprecation policy

Deprecated features should emit `DeprecationWarning` and remain available for at least one minor release before removal in a future major release.
