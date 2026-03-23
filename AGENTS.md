---
noteId: "373a1a40245911f19cfafb6bb050f0f4"
tags: []

---

## Image Captioning COCO — Operator Playbook

1. **Mission** — Provide reliable guidance for agentic contributors touching any part of this repo. Follow every rule below even if not explicitly restated in user prompts.
2. **Repo health** — PyTorch project, no pre-commit hooks, no lint automation. Tests are pure Python and synthetic; data downloads happen only on demand.
3. **Cursor / Copilot rules** — None found (.cursor/ or .cursorrules absent, .github/copilot-instructions.md absent). This file therefore defines the full contract.
4. **Languages** — Source comments and docs are mostly French; keep that tone when expanding documentation or CLI help. Code identifiers remain English.
5. **Python version** — Target 3.8+. Assume CUDA when available but never require it for tests.

## Environment & Dependencies

6. **Virtualenv bootstrap**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
7. **CPU-only installs** — requirements.txt is CPU-safe; Torch will pick CUDA if the wheel matches the host. Do not pin extra GPU packages inside the repo.
8. **Dataset assets** — `./getCOCO.sh` downloads COCO 2017. Run it manually; never assume CI availability. Scripts look for `data/coco/...` and `data/coco_vocab.pkl`.
9. **Vocabulary build** — Run `python prepare_data.py` after downloading COCO to create `data/coco_vocab.pkl` and serialized pairs.
10. **Checkpoints/logs** — Stored under `checkpoints/<model>/<scheduler>/` and `logs/<model>/<scheduler>/`. Respect existing folder layout when saving artifacts.

## Core Commands (run from repo root)

11. **Full test suite** — `python test.py`
12. **Verbose tests** — `python test.py -v`
13. **Single test group** — `python test.py TestEncoder` (replace with any suite name listed inside `test.py`).
14. **Multiple groups** — `python test.py TestDecoder TestTrainer`
15. **Coverage** — `coverage run test.py && coverage report -m`
16. **Train (standard)** — `python train.py --model densenet --scheduler cosine`
17. **Train (fast debug)** — `python train.py --model cnn --scheduler cosine --fast`
18. **Resume training** — `python train.py --model densenet --scheduler cosine --resume checkpoints/densenet/cosine/best_model.pth`
19. **Evaluate checkpoints** — `python evaluate.py --checkpoint checkpoints/densenet/cosine/best_model.pth`
20. **Evaluate by architecture** — `python evaluate.py --model densenet resnet cnn --scheduler cosine`
21. **Demo single image** — `python demo.py --checkpoint checkpoints/densenet/cosine/best_model.pth --image ImagesTest/dog.jpg`
22. **Demo folder** — `python demo.py --checkpoint ... --image_dir ImagesTest/ --method beam_search`
23. **Attention viz** — `python visualize_attention.py --checkpoint checkpoints/densenet/cosine/best_model.pth --image ImagesTest/dog.jpg`
24. **Scripted pipeline** — `bash train.sh` exists but contains git commands; only run sections you actually need.

## Execution Notes

25. Tests are synthetic; they do not require COCO files. Prefer them for quick validation.
26. Training/eval scripts expect COCO assets. Guard code that reads disk; fail fast with clear FileNotFoundError messages.
27. Demo and visualize scripts auto-detect sources in this order: explicit args → `data/coco/test2017` → `ImagesTest/`.
28. BLEU/METEOR rely on NLTK corpora. Install-time downloads happen lazily inside scripts; ensure outbound network access or vendor the data.
29. GPU autodetection is ubiquitous (`torch.cuda.is_available()`). Keep code path agnostic: guard `.to(self.device)` calls and avoid `.cuda()` literals.

## Coding Style & Conventions

30. **General layout** — PEP 8 spacing, 4-space indents, <= 100–110 char lines when possible. Favor readability over golfed expressions.
31. **Imports** — Group stdlib, third-party, then local modules with a blank line between groups. Use absolute imports (`from utils.preprocessing import ...`).
32. **Constants** — Upper snake case (e.g., `VOCAB_SIZE`). Place near module top.
33. **Classes** — PascalCase, docstring immediately below the class line, describing role + compatibility notes. Methods use present tense.
34. **Functions** — snake_case, include docstrings if the function is non-trivial or public. Use `Args:` / `Returns:` style blocks like existing files.
35. **Typing** — Light-touch. Add hints when signatures would otherwise be ambiguous; do not blanket annotate everything if it hurts readability.
36. **Docstrings** — Multi-line triple-double-quoted strings. Keep them informative and, when relevant, bilingual clarity (FR explanations + EN identifiers).
37. **Logging & prints** — Preference for `print()` with human-friendly French sentences. Prefix auto-detected cases with `[Auto]` or `[Forcé]` as seen in demo scripts.
38. **CLI interfaces** — Use `argparse.ArgumentParser` with `RawDescriptionHelpFormatter`. Provide epilog examples; keep flag names consistent (`--image_dir`, `--save_dir`).
39. **Configuration** — Centralize defaults in `config.py`. New hyperparameters belong there with explicit comments. Always expose them via CLI flags before hardcoding.
40. **Data paths** — Never hardcode user-specific directories. Use `os.path.join` and respect `config['checkpoint_dir']`, `log_dir`, etc.
41. **Randomness** — Follow `config['random_seed']` when adding stochastic processes. Surface a CLI flag to override if necessary.
42. **Attention models** — Keep `forward_with_alphas` available whenever attention regularization is needed. If you add new architectures, mirror the API used in `Trainer`.
43. **Beam search** — Maintain consistent signatures: `generate_beam_search(features, beam_width, max_length, start_token, end_token)` for decoder-level, and `generate_caption` for end-to-end models.
44. **Error handling** — Raise `ValueError` for invalid CLI choices, `FileNotFoundError` for missing paths, and propagate helpful instructions. Do not silently skip failures.
45. **Warnings** — Use `warnings.filterwarnings('ignore')` sparingly; tests already do this globally. Prefer explicit handling.
46. **GPU transfers** — Always call `.to(self.device)` on tensors/buffers derived from loaders. Avoid `.cuda()` shorthand.
47. **Gradients** — Clip using `torch.nn.utils.clip_grad_norm_(..., max_norm=5.0)` when training loops change. Maintain parity with Trainer conventions.
48. **Optimization** — Stick with Adam + scheduler combos defined in `Trainer`. New schedulers must integrate with warmup logic and early stopping semantics.
49. **Metrics** — CIDEr is the authoritative checkpoint metric; BLEU/METEOR are informative. When you log metrics, clearly state which ones gate checkpoint saves.
50. **Plots** — Use Matplotlib in Agg mode by default, then attempt interactive backends (see demo/visualize). Reuse that pattern for new visualization tools.
51. **File outputs** — Ensure target directories exist via `os.makedirs(path, exist_ok=True)` before writing.
52. **Internationalization** — When printing captions or strings, avoid accent stripping; keep UTF-8 intact. Code files stay ASCII except for literal French text already present.

## Testing Expectations

53. Unit tests are modular (11 groups). When adding modules, create matching `Test...` suites in `test.py` using `unittest`.
54. Tests fabricate inputs via helper factories (e.g., `make_images`). Reuse them for consistency rather than inventing ad-hoc tensors.
55. When adding CLI flags, extend `TestConfig` to ensure `get_config()` exposes the new keys across architectures.
56. Use `coverage run test.py` before large refactors. Generated reports belong in `htmlcov/` (already gitignored).

## Data & Checkpoint Guidance

57. `prepare_data.py` builds the vocabulary once; do not auto-run it inside other scripts. Instead, detect missing `data/coco_vocab.pkl` and print actionable instructions.
58. `train.py` saves three notable files per combo: `best_model.pth` (val loss), `best_model_cider.pth` (CIDEr), and periodic `checkpoint_epoch_N.pth`. Preserve naming when adding new save points.
59. Cosine scheduler uses cycle checkpoints plus SWA (`averaged_model.pth`). New training logic must keep cycle bookkeeping in sync with `Trainer._cycle_checkpoints`.
60. Logs record histories under `logs/<model>/<scheduler>/`; extend JSON schema deliberately and document any new keys in this file.

## Workflow & Git Hygiene

61. Never run `git add/commit/push` from `train.sh` during automated workflows; the script’s git lines are historical references only.
62. Keep changes atomic. When in doubt, stage only the files required for a given feature or fix.
63. Do not store large checkpoints or datasets in git. Ensure `.gitignore` covers any new artifact paths.
64. Before committing, run `python test.py` (and targeted groups if time-limited). Mention coverage runs when relevant.
65. Summaries in PRs/commits should emphasize *why* a change exists (e.g., “stabilize attention regularization under mixed precision”).

## Handy Reference Table

66. | Task | Command | Notes |
67. | --- | --- | --- |
68. | Install deps | `pip install -r requirements.txt` | Run inside activated venv |
69. | Download COCO | `./getCOCO.sh` | Creates `data/coco/` tree |
70. | Build vocab | `python prepare_data.py` | Uses JSON annotations |
71. | Full tests | `python test.py` | Synthetic tensors only |
72. | Single test | `python test.py TestTrainer -v` | Fastest targeted loop |
73. | Coverage | `coverage run test.py` | Report via `coverage report -m` |
74. | Train (best) | `python train.py --model densenet --scheduler cosine` | GPU recommended |
75. | Evaluate | `python evaluate.py --checkpoint ...` | Accepts `--num_samples 500` |
76. | Demo | `python demo.py --checkpoint ... --image_dir ImagesTest/` | Saves PNG captions |
77. | Attention viz | `python visualize_attention.py --checkpoint ...` | Outputs overlays |

## Closing Checklist for Agents

78. Activate venv and install deps before running scripts.
79. Confirm dataset + vocab existence; if absent, mention the exact commands users must run.
80. Use synthetic tests for quick feedback; mention long-running trains separately.
81. Keep messaging bilingual where user-facing, but code identifiers stay English.
82. Respect all directory conventions for checkpoints, logs, and outputs.
83. When uncertain, prefer explicit instructions (raise ValueError) over implicit defaults.
84. Update this file whenever you add tooling, commands, or style rules that future agents must know.
85. Remember: clarity beats cleverness. Favor maintainability for the next agent coming after you.

## Test Suite Map

86. `TestConfig` — validates `config.py` keys, fast/slow modes, attention switches.
87. `TestVocabulary` — checks token indices, numericalize/denumericalize, persistence.
88. `TestEncoder` — enforces tensor shapes and parameter counts across EncoderCNN/Spatial/DenseNet.
89. `TestDecoder` — covers DecoderLSTM and DecoderWithAttention forward + generation modes.
90. `TestCaptionModel` — ensures factory, save/load roundtrips (vocab embedded in checkpoint).
91. `TestDataLoader` — verifies padding, sorting, and collate logic in synthetic loader fixtures.
92. `TestTrainer` — smoke-tests both schedulers with warmup, attention regularization, gradient clipping.
93. `TestDemo` — checks CLI defaults, auto path selection, and caption rendering.
94. `TestEvaluate` — validates CIDEr/BLEU computation and checkpoint loading pathways.
95. `TestVisualizeAttention` — ensures attention grids reshape to 7×7 and stop-word filtering logic.
96. `TestIntegration` — builds a full encoder+decoder pipeline on toy data to catch wiring regressions.

## Logging & Telemetry

97. Training histories land in `logs/<model>/<scheduler>/history_*.json`; keep schema stable so notebooks remain compatible.
98. Loss/metric PNGs live alongside history JSON; reuse `plot_learning_curves()` helpers when adding plots.
99. Console prints should stay bilingual-friendly: English identifiers in code, natural French sentences in user-facing logs.
100. When adding new metrics, extend both the history writer and `evaluate.py`, and describe them here.

## Support Scripts & Datasets

101. `getCOCO.sh` downloads train/val2017 + annotations; script assumes `unzip` availability.
102. `prepare_data.py` expects COCO JSON structure unchanged; guard for missing files with actionable `print()` instructions.
103. Mini datasets for docs/tests live under `ImagesTest/` and synthetic factories in `test.py`; keep them lightweight.
104. Do not auto-download data inside tests; rely on generated tensors.

## PR / Review Checklist

105. Ensure new CLI arguments propagate to `config.py`, parser(s), Trainer/evaluation, and `AGENTS.md` when user-facing.
106. Verify checkpoints/log folders exist before writing; tests may mock them via `tempfile`.
107. Document any change that alters file formats (history JSON, saved checkpoints) in this playbook.
108. Re-run the specific `python test.py Test...` groups that touch your change plus the full suite when time permits.
109. Mention long-running commands you skipped (e.g., training) so reviewers know what still needs verification.
