# Contributing

Contributions are welcome. Please follow these guidelines.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a feature branch: `git checkout -b feature/your-feature`
4. Make your changes
5. Run tests: `cargo test`
6. Run clippy: `cargo clippy -- -D warnings`
7. Format code: `cargo fmt`
8. Commit your changes
9. Push to your fork
10. Open a pull request

## Pull Request Process

1. Update documentation if you're changing public APIs
2. Add tests for new functionality
3. Ensure CI passes
4. Request review from maintainers

## Code Style

- Follow Rust standard formatting (`cargo fmt`)
- No clippy warnings (`cargo clippy -- -D warnings`)
- Write tests for new code
- Document public APIs with doc comments

## Commit Messages

Use conventional commits:
- `feat:` new features
- `fix:` bug fixes
- `docs:` documentation changes
- `test:` adding or updating tests
- `refactor:` code changes that neither fix bugs nor add features
- `perf:` performance improvements
- `chore:` maintenance tasks

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
