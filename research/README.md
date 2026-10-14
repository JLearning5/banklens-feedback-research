# Review benchmark

Inputs are `labelled_reviews.parquet` and an exported review JSON file with `schema_version: 1` and a `decisions` array. CSV input may instead supply `review_id` and `reviewer_sentiment` columns. The configuration requires at least 20 distinct reviewed text groups in each of the four sentiment classes.

Run on Kaggle with internet access and a free GPU accelerator:

```text
python run_experiments.py --data /kaggle/input/your-private-data/labelled_reviews.parquet --reviews /kaggle/input/your-private-data/review_decisions.json --output /kaggle/working/banklens_experiments
```

The five-fold group assignment reserves fold 0 for test and fold 1 for validation. The remaining folds form the training set. The seed, review-file hash and split-manifest hash are saved. All candidates use the same validation rows. Model selection uses validation macro F1, with runtime as a tie-breaker. The selective threshold is chosen on validation data. The winner is reloaded in a separate process before the final test evaluation.

Artifacts include model objects, tokenizer and checkpoint files, training history, retrieval examples, predictions, reports, comparison curves, resource measurements, source audits and a checksum manifest. Generative models record their exact hub revision and reload that revision.

To use a saved experiment directly, import `load_selected` from `run_experiments.py`, load `run_config.json` and `selection.json`, and pass the selected model folder. Input rows require `review_id`, `review_text` and `duplicate_group`.

The runnable notebook is in `notebooks/review_benchmark.ipynb`. Review procedures and interpretation guidance accompany the separate reviewer report.
