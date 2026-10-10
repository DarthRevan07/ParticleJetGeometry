You are working in the JetTopoBench repository: a benchmark-first study of whether
persistent homology and simplicial message passing add anything to jet tagging
(top vs QCD, quark vs gluon). It supports a research report I will cite in PhD
applications, so correctness and honest reporting matter more than speed or SOTA numbers.

Before doing anything:
1. Read specs/00_overview.md completely, then the spec(s) for the task I give you.
2. Run `pip install -e ".[dev,physics,viz]"` and `pytest -q`. The baseline is
   64 passed, 37 xfailed, 0 failed. Report the result before changing code.

How to work:
- One spec at a time. Keep the public names and signatures in the stub files.
- VERIFY TESTS FIRST. The xfail tests for stub modules were written from the spec's math
  but never run against an implementation. Before implementing a stub, check every
  expected value in its test file against the spec's definitions, by hand or with a
  throwaway script outside the repo. If a test is wrong, fix the test and log the reason
  in that spec's Changelog section. Never loosen a tolerance, delete an assertion, or
  change a mathematical definition just to get a pass. If the spec itself looks wrong,
  stop and ask me.
- After each module: `pytest -q` must show no failures. That module's tests must now
  pass, not xfail.
- All geometry uses (dy, dphi) relative to the jet axis. Never build distances or
  complexes on momentum components.
- Every random operation takes a seed. Every evaluation uses the FULL official test set.
  Only training data is subsampled (configs/tiers.yaml, data/*/subsets_*.npz).
- Before any job longer than ~10 minutes, time 1,000 jets or one epoch on a small subset
  and tell me the projected runtime. Do not start long or GPU jobs yourself. For Colab
  runs, give me a ready-to-run notebook or command block that checkpoints to Google Drive.
- Never commit data, checkpoints, API keys or credentials.
- Fill in the "record here" items in the specs (dataset row counts, timings, chosen
  Rips scale, library versions) as you measure them.

When you finish a task, reply with:
(a) what you implemented, as files changed;
(b) any test you corrected and why;
(c) the pytest summary line;
(d) measured numbers or timings;
(e) open questions for me;
(f) for each new module, 3–5 sentences on the key design decision and the math it
    relies on. I need to explain this work in interviews.