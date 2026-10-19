# Fixed-ratio CoT ablation

This experiment creates four datasets whose compressed reasoning has exactly
20%, 40%, 60%, or 80% of the original CoT token count under the Qwen3-8B
tokenizer. The original target action is retained. Each ratio uses the same
source examples and the same Qwen3-8B LoRA hyperparameters.

The API key is read only from `DEEPSEEK_API_KEY`; it is never stored in code.
For unattended SSH runs, the pipeline can source it from the mode-600 file
`/home/liuheng/.config/edge_cloud/deepseek.env`.

Run the stages in order:

```bash
cd /home/liuheng/edge_cloude/fixed_ratio_experiments
export DEEPSEEK_API_KEY='set-this-in-your-shell'
PY=/home/liuheng/miniconda3/envs/llama_factory/bin/python
$PY generate_fixed_ratio_cot.py
$PY validate_fixed_ratio.py --data_dir data
$PY prepare_fixed_ratio_sft.py

cd /home/liuheng/LLaMA-Factory
bash train_fixed_ratios.sh
NUM_GAMES=50 bash evaluate_fixed_ratios.sh
```

Or run all stages with `bash run_fixed_ratio_pipeline.sh`. The last stage writes
`fixed_ratio_results.csv` with train loss, success rate, average steps, output
tokens, and runtime for all four ratios.

Generation is resumable. A record is skipped only after an exact-length row
has been written. Do not combine `failures.jsonl` with a training dataset;
rerun generation until validation reports `"exact": true` for all four sets.

The training script uses LoRA rank 8, batch size 1, gradient accumulation 4,
learning rate 1e-4, three epochs, cosine scheduling, 10% warmup, BF16, and the
`qwen3_nothink` template. Outputs are isolated under
`saves/qwen3-8b-fixed_ratio_*/lora/sft`.
