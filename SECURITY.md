# Security Policy

## Supported Versions

| Component | Version | Supported          |
| --------- | ------- | ------------------ |
| Arcanum   | 0.1.x   | :white_check_mark: |
| Moloch    | 0.1.x   | :white_check_mark: |
| Haagenti  | 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please report it responsibly.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please email security vulnerabilities to:
- **Email:** security@daemoniorum.com

### What to Include

Please include the following in your report:

1. **Description** - A clear description of the vulnerability
2. **Impact** - What could an attacker accomplish?
3. **Reproduction Steps** - Step-by-step instructions to reproduce the issue
4. **Affected Component** - Which crate(s) are affected (Arcanum, Moloch, Haagenti)
5. **Version** - What version(s) are affected
6. **Suggested Fix** - If you have one (optional)

### Response Timeline

- **Initial Response:** Within 48 hours
- **Triage Complete:** Within 7 days
- **Fix Timeline:** Depends on severity
  - Critical: 7 days
  - High: 14 days
  - Medium: 30 days
  - Low: 60 days

### What to Expect

1. **Acknowledgment** - We'll confirm receipt of your report
2. **Investigation** - We'll investigate and determine the impact
3. **Fix Development** - We'll develop and test a fix
4. **Coordinated Disclosure** - We'll work with you on disclosure timing
5. **Credit** - We'll credit you in the security advisory (if desired)

## Security Best Practices

### For Arcanum (Cryptography)

- Always use the high-level APIs rather than low-level primitives
- Never reuse nonces/IVs
- Use constant-time comparison for secrets
- Zeroize sensitive data after use
- Use hybrid post-quantum schemes for long-term security

### For Moloch (Audit Chain)

- Validate all event signatures before processing
- Use authenticated anchoring endpoints only
- Implement rate limiting on API endpoints
- Monitor for unusual event patterns
- Back up Merkle Mountain Range proofs regularly

### For Haagenti (Compression)

- Validate input sizes before decompression (zip bomb protection)
- Use sandboxed environments for untrusted model loading
- Verify HCT file integrity before use
- Limit memory allocation during decompression

## Known Security Considerations

### Arcanum

- **Side-channel attacks:** Timing attacks are mitigated through constant-time implementations, but physical side-channel attacks (power analysis, EM) are not addressed
- **Post-quantum algorithms:** ML-KEM and ML-DSA are based on NIST standardized algorithms but are relatively new

### Moloch

- **Anchoring trust:** Security depends on the integrity of anchor chains (Bitcoin, Ethereum)
- **Consensus:** Byzantine fault tolerance requires 2/3 honest nodes

### Haagenti

- **Model integrity:** Compressed models should be integrity-verified before use
- **Memory safety:** WebGPU operations run in sandboxed GPU contexts

## Dependency Security

We regularly audit dependencies for known vulnerabilities using:
- `cargo audit`
- `cargo deny`
- Dependabot alerts

## Contact

For non-vulnerability security questions, you can reach us at:
- **General:** hello@daemoniorum.com
- **Security:** security@daemoniorum.com
