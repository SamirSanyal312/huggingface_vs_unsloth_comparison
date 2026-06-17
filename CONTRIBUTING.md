# Contributing

Contributions are welcome.

## Suggested Contribution Areas

- Add more benchmark results on different GPUs
- Add larger dataset runs
- Add evaluation prompts for quality comparison
- Add a script-based benchmark in addition to notebooks
- Improve documentation and reproducibility

## Development Guidelines

1. Keep Hugging Face and Unsloth experiments as comparable as possible.
2. Document any change in model, dataset, batch size, sequence length, or LoRA configuration.
3. Do not commit large model checkpoints or generated artifacts.
4. Add benchmark results clearly with hardware details.
5. Keep notebooks readable and organized for learning purposes.

## Pull Request Checklist

Before submitting a pull request:

- The notebook runs from top to bottom.
- Results are clearly printed or documented.
- Large model files are not committed.
- The README is updated if the experiment setup changes.
