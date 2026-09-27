# Changelog

## 0.1.0

Initial package version. The package audit adds:

- Input validation before native sampling, including positive sampling intervals and valid word counts.
- Reusable document/settings objects and quiet `verbose=False` fitting.
- Normalized posterior output for empty topics and keyword distributions restricted to each topic's dictionary.
- Regression tests across all six model specifications and four weighting rules.

The keyword-support correction changes some topic-word probabilities from earlier development snapshots. Historical manuscript topic-word metrics must be regenerated before being attributed to this corrected implementation.
