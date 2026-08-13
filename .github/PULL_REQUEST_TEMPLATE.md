## What this changes

<!-- One paragraph. What behaviour is different after this merges? -->

## Why

<!-- The problem, not the patch. If it fixes an issue, link it. -->

## Invariants

The project holds five invariants (see
[CONTRIBUTING.md](../blob/main/CONTRIBUTING.md)). Tick the ones this PR touches
and say below how it keeps them intact.

- [ ] An over-merge is worse than a duplicate — ambiguity resolves to "keep both"
- [ ] Never fabricate a vector — embedders raise rather than substitute
- [ ] Nothing is deleted — invalidate, never `DELETE`
- [ ] Every scoped statement filters `tenant_id` **and** `graph_id`
- [ ] The gate computes risk; the caller only proposes it
- [ ] None of the above

<!-- If you ticked any, explain here. -->

## Testing

- [ ] Unit tests pass (`pytest`)
- [ ] Integration tests pass against a real Postgres (`pytest -m integration`)
- [ ] `ruff check .` and `mypy src` are clean
- [ ] New behaviour has a test; if it touches SQL, that test runs against the
      real database rather than a mock

## If this changes a threshold

- [ ] The measurement behind the new value is recorded in the comment next to it

## Checklist

- [ ] `CHANGELOG.md` updated under "Unreleased"
- [ ] Public API changes are reflected in the docstrings and `docs/`
