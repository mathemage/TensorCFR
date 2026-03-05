# TensorCFR — tensor-based Counterfactual Regret Minimization

TL;DR
- TensorCFR is a tensor-backed implementation of Counterfactual Regret Minimization (CFR) for solving imperfect-information games (e.g., Kuhn/Leduc poker variants). Designed for experiments, fast batch updates, and reproducible evaluations. Includes training scripts, evaluation utilities, and example experiments.

What I built
- A CFR engine implemented with tensor operations to exploit GPU/CPU acceleration.
- Utilities for game definition, batched regret updates, and exploitability evaluation.
- Example experiments for standard benchmark games (Kuhn, Leduc) with scripts to train, evaluate, and plot convergence.

Highlights / Why this is interview-ready
- Non-trivial algorithmic project implementing a research algorithm end-to-end.
- Mix of algorithm design, numerical stability considerations, vectorized engineering, and experiment tracking.
- Results and convergence plots provide concrete evidence of performance you can demo in interviews.

Results
- Reproducible experiments available in `experiments/` (training logs and plots).
- Typical demonstration: show exploitability vs iterations plot and final strategy evaluation on a held-out opponent or exploitability metric.

Quick start (local)
1. Create env and install:
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
2. Run an example training:
   python examples/run_cfr.py --game kuhn --iters 20000 --batch-size 1024 --out-dir results/kuhn
3. Evaluate:
   python examples/evaluate.py --exp results/kuhn --metric exploitability --plot

Quick start (Docker)
- Build: docker build -t tensorcfr .
- Run: docker run --rm -v $(pwd):/work tensorcfr python examples/run_cfr.py --game kuhn --iters 5000

Files to show in interview
- examples/run_cfr.py — entrypoint for experiments
- core/cfr.py — algorithm logic (regret update, strategy update)
- experiments/* — logs, plots, and notebooks

Implementation notes (one-liner)
- Core implemented with vectorized tensor operations for batched regrets; numerical stability handled via smoothing and learning-rate scheduling. Designed so backend can be NumPy, PyTorch or TensorFlow with minimal adapter glue.

Talking points for interviews
- Why CFR and where it's used (imperfect information games, poker).
- Design choices: batching, tensorization, numerical stability.
- How you validated results: exploitability measure, baseline comparisons, convergence plots.
- Engineering trade-offs and what you'd improve (parallel training, saved checkpoints, Dockerized demo).

Repro / Demo checklist
- Ensure `requirements.txt` present and `examples/run_cfr.py` matches commands above.
- Point to `results/` for plots and a short demo script that loads a trained policy and runs a short head-to-head match.

License & contact
- License: (keep existing repo license)
- Contact: @mathemage (GitHub)
