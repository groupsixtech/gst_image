# GST Image Agent Memory

This directory is a technical reference for agents working on GST Image. It is
deliberately more detailed than the repository guide, but is not a replacement
for it. Start with [Repository Guidelines](../../AGENTS.md) for working rules,
commands, style, and pull-request expectations; then read only the note relevant
to the change.

## Notes

- [Architecture and persistence](architecture.md) — layers, runtime data flow,
  model ownership, project format, coordinate rules, and entry points.
- [Capabilities and algorithms](capabilities-and-algorithms.md) — current user
  features, scientific/technical algorithms, libraries, outputs, and limits.
- [Modification playbook](modification-playbook.md) — safe change paths,
  test strategy, performance guardrails, and common failure modes.

## Scope and maintenance

These notes describe the checked-in implementation, not product aspirations.
Update the affected note in the same change when altering an entry point,
persisted schema, algorithm, dependency, export, or user-visible workflow.
Prefer links to source files and tests over duplicating implementation details
that will drift.
