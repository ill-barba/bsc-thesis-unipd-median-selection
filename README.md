# Multi-pass median selection with limited memory

This is the code for my bachelor's thesis at the University of Padua (DEI).
It implements the multi-pass selection algorithm by Munro and Paterson (1980):
find the k-th largest of N distinct values that can only be read sequentially,
like a tape, using just S cells of working memory. The less memory you have,
the more passes over the input you need. The experiments measure exactly that
trade-off.

## Thesis

The bachelor's thesis (in Italian) is available as a [PDF](thesis-alessandro-barbacetto.pdf).

## Layout

```text
.
├── src/
│   └── multipass.py              # the algorithm (MP-SELECT)
├── tests/
│   └── test_multipass.py         # pytest suite for the algorithm
├── experiments/
│   ├── metrics_multipass.py      # helpers: counts passes and candidates
│   ├── experiment_passes.py      # experiment A: passes vs memory
│   ├── experiment_candidates.py  # experiment B: candidates vs bound
│   └── results/                  # CSV data and PDF figures
├── thesis-alessandro-barbacetto.pdf  # bachelor's thesis (Italian)
├── README.md
├── LICENSE                       # Apache License 2.0
└── NOTICE                        # attribution notice
```

## Requirements

- Python 3.10 or newer (developed with 3.13)
- matplotlib (developed with 3.11) for the figures
- pytest (developed with 9.1) for the tests

The algorithm itself uses only the standard library.

```bash
pip install matplotlib pytest
```

All commands are run from the repository root:

```bash
python -m pytest                              # run the tests
python experiments/experiment_passes.py       # experiment A, short computation
python experiments/experiment_candidates.py   # experiment B, quite long computation
```

Each experiment writes its CSV files and its PDF figure to `experiments/results/`.

To redraw a figure from the existing CSV files, without running the
experiment again (to save some time), add `--plot-only`:

```bash
python experiments/experiment_passes.py --plot-only
python experiments/experiment_candidates.py --plot-only
```

## Assumptions and measurements

**Input.** A sequence of N distinct integers, one per line, read
sequentially from a rewindable text stream. The algorithm does not check that
the values are distinct.

**Rank.** `k = 1` is the largest element; the median is `k = ceil(N/2)`.

**Memory.** S is the number of logical memory cells available to the
algorithm, as in the model of the thesis. It is not the RAM used by Python:
the experiment scripts keep extra data, such as the whole input, only to
measure and check the results.

**What is measured.**

- `P`: number of complete scans of the input, counted every time the tape is
  rewound. It is compared with `P_max`, the number of passes predicted by the
  recurrence on the candidate bound.
- `n_p`: actual number of candidates left after sampling pass `p`, counted
  directly on the input.
- `n̄_p = (2r_p − 1)·2^r_p`: the theoretical upper bound on the candidates
  after the same pass.

**How the data are generated.** Every input is a random permutation of
1..N with a fixed seed, so every run can be reproduced exactly.

- Experiment A: N = 10^3, 10^4, 10^5, 10^6; median; seed 2026; S on a
  geometric grid (factor 2^(1/4)) from the smallest S for which the
  algorithm terminates up to N + 2. Smaller inputs are checked again with
  other ranks and seeds.
- Experiment B: N = 10^4 and 10^5; 30 seeds per configuration. The figure
  uses the median over a finer grid of S (factor 2^(1/8)); the CSV files also
  keep four ranks and sorted inputs for four fixed memory sizes.

## Reference

J. I. Munro and M. S. Paterson, *Selection and Sorting with Limited Storage*,
**Theoretical Computer Science**, 12 (1980), 315–323.

## Author and license

**Alessandro Barbacetto** — [GitHub profile](https://github.com/ill-barba).

This software is distributed under the **Apache License, Version 2.0**.
See [`LICENSE`](LICENSE) for the license terms and [`NOTICE`](NOTICE) for the
project attribution notice. Redistribution is subject to the conditions of
that license, including the retention of applicable notices.
