---
title: Contributing
description: "Guidelines for contributing to the Neuro project."
---

## How to Contribute

We welcome contributions from the community! Whether you're fixing a bug, adding a new feature, or improving the documentation, your help is appreciated.

### Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/YOUR_USERNAME/neuro.git
    ```
3.  **Create a new branch** for your changes:
    ```bash
    git checkout -b my-feature-branch
    ```
4.  **Install the development dependencies**:
    ```bash
    pip install -r requirements-dev.txt
    ```

### Making Changes

-   **Follow the coding style**: We use Black for code formatting and Ruff for linting. Please run `black .` and `ruff .` before committing your changes.
-   **Write tests**: If you add a new feature, please include tests that cover it. Tests are located in the `src/tests/` directory.
-   **Update the documentation**: If your changes affect the API or the internal architecture, please update the relevant documentation in the `docs/` directory.

### Submitting a Pull Request

1.  **Commit your changes** with a clear and descriptive commit message.
2.  **Push your branch** to your fork:
    ```bash
    git push origin my-feature-branch
    ```
3.  **Open a pull request** from your fork to the `main` branch of the official Neuro repository.
4.  **Provide a detailed description** of your changes in the pull request.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for everyone. Please read and follow our [Code of Conduct](https://github.com/neuro-ai/neuro/blob/main/CODE_OF_CONDUCT.md).

## Development Roadmap

Here are some of the features we are planning to build next:

-   **Support for more model providers** (e.g., Google Gemini).
-   **Advanced tool use** integration for the `ReasoningAgent`.
-   **A web-based dashboard** for monitoring and evaluation.
-   **Real-time collaboration** features for building and sharing reasoning chains.

If you are interested in working on any of these, please open an issue to discuss it with the maintainers.
