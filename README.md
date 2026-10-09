# DirectorBench

Adversarial benchmark tasks for AI models doing a video director's work.

Three tasks: PromptCraft (write a generation-ready prompt), ShotList (plan
shots from a brief with hard constraints), QC Judge (accept or reject a
frame; some frames are decoys that are correct by design and must be passed).

This repo holds the lean pilot: 24 adversarial items, answer keys, and the
deterministic scorer (regex checks plus word-overlap checks, no human
judgment).

## Files

- `items.json`: the 24 benchmark items (6 PromptCraft, 6 ShotList, 12 QC Judge)
- `answer_key.json`: expected answers for each item
- `scorer.py`: the deterministic scoring script

## Pilot results

Two models, Medium effort via Notion AI:

| Task | Opus 5.5 | Grok 4.6 |
|---|---|---|
| PromptCraft (6) | 1.000 | 0.931 |
| ShotList (6) | 0.810 | 0.780 |
| QC Judge (12) | 1.000 | 0.917 |

Full writeup: [DEV post](https://dev.to/ratnesh_saxena)

## License

MIT
