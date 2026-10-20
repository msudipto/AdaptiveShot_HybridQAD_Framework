# Accessibility Statement

We want the Adaptive-Shot Hybrid QAD Framework to be accessible to everyone, including researchers, students, and practitioners with disabilities. This document outlines our current accessibility focus and how you can report barriers.

## Priorities

Our priority is ensuring that our research outputs, documentation, and terminal scripts are usable by all developers. We focus on:
* **Text Readability**: Writing clear, jargon-free explanations in our documentation and maximizing readability for screen readers.
* **Vector Graphics**: Exporting evaluation figures in stable vector formats like SVG and PDF so they can be infinitely scaled without losing text clarity.
* **Terminal Diagnostics**: Structuring execution text output to be compatible with command-line screen readers.

## Contributor expectations

When contributing user-facing modifications, documentation, or charts:
* Use high-contrast color palettes for code execution figures and charts to assist color-blind reviewers.
* Ensure all newly added mathematical definitions or scripts are fully documented in plain text inside the `README.md` or code comments.

## Reporting accessibility issues

If you encounter an accessibility barrier while running our scripts, reading the framework logs, or inspecting the publication charts, please report it privately via email:

* **Mubassir Serneabat Sudipto** — msudipto@iastate.edu

Please describe the barrier you encountered, the relevant script or file path, and any assistive technology you were using.

### How we respond

We will acknowledge your report within 5 business days and work to implement a clear text workaround or graphical update in the primary development branch.

## Supported environments

This framework is tested on standard terminal environments (Windows PowerShell and Unix Bash). We recommend using high-contrast terminal themes and vector-capable document viewers to inspect output charts under the `artifacts/` folder.
