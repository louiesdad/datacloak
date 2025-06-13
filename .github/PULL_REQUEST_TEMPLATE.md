# Pull Request

## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Security enhancement

## Testing
- [ ] I have run the local CI checks (`./scripts/ci-local.sh`)
- [ ] All tests pass locally
- [ ] I have added tests for new functionality
- [ ] Coverage is maintained at 80%+

## Security Checklist
- [ ] No new security vulnerabilities introduced (`cargo audit` passes)
- [ ] ReDoS protection maintained for any new regex patterns
- [ ] Rate limiting considerations addressed
- [ ] Input validation implemented where needed

## Code Quality
- [ ] Code follows Rust formatting standards (`cargo fmt`)
- [ ] All clippy warnings resolved (`cargo clippy`)
- [ ] Documentation updated for public APIs
- [ ] Performance impact considered

## CI Pipeline
The following checks must pass:
- [ ] 🔒 Security Audit (`cargo audit`)
- [ ] 📐 Format Check (`cargo fmt --check`)
- [ ] 📋 Clippy Lints (`cargo clippy -- -D warnings`)
- [ ] 🧪 Test Coverage (≥80%)
- [ ] ⚡ Benchmarks (performance thresholds met)
- [ ] 🌐 Cross-Platform Compatibility
- [ ] 🔧 CLI Integration Tests

## Additional Notes
Any additional information, deployment notes, or breaking changes.

---

By submitting this pull request, I confirm that:
- [ ] I have read the [Contributing Guidelines](CONTRIBUTING.md)
- [ ] I have reviewed the [Security Best Practices](SECURITY.md)
- [ ] I understand the [CI/CD requirements](CI_HARDENING.md)