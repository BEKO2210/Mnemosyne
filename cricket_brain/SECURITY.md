# Security Policy

## Supported Versions

We currently support security updates for the latest `main` branch and the most recent crates.io release.

## Reporting a Vulnerability

If you discover a security vulnerability:

1. **Do not** open a public issue with exploit details.
2. Email **belkis.aslani@gmail.com** with:
   - reproduction steps
   - affected commit/release
   - impact assessment
3. We will acknowledge receipt within 72 hours and provide a remediation timeline.

## Dependency and Supply-Chain Controls

- CI runs `cargo audit` on every pull request and push.
- Known vulnerable advisories fail CI.
- We prefer minimal dependencies in `crates/core` for reduced attack surface.

## Disclosure Process

- Fix prepared in a private branch when needed.
- Coordinated disclosure with reporter after patch release.
- Changelog and advisory references included in release notes.
