# BankLens

Banking customer feedback analysis, exploration and review, with a browser application and reproducible Kaggle research workflows.

[Open BankLens](https://jlearning5.github.io/banklens-feedback-research/)

## Application

Open the deployed application and use the four workspaces:

- **Overview:** filter banks, years, sentiment and service aspects; inspect totals and recurring issues.
- **Feedback:** search the complete collection, open a source record and export the current selection.
- **Analysis:** ask collection questions immediately, or analyse individual feedback with Standard or Enhanced detail. Results include counts, linked feedback and the tool sequence.
- **Review:** open a queued record, record your judgement, save and continue. Export decisions to JSON or CSV and import them on another browser.

Collection questions use catalog tools and linked source excerpts immediately. Standard feedback analysis starts the compact sentiment encoder on demand and batches its text fragments; it uses the explicit aspect patterns and priority rules. Enhanced feedback analysis additionally loads the compact Qwen model for language-based choices. The workspace shows each first-use download size, keeps prepared resources for later requests, and supports cancellation.

## Project structure

| Folder | Purpose |
| --- | --- |
| `UI-fixed` | Responsive dashboard |
| `Original` | Functional comparison interface |
| `shared` | Data worker, model worker, bounded agent, validation and review storage |
| `data` | Complete precomputed browser data and record shards |
| `processing` | Corpus processing source and external configuration |
| `research` | Review-based model experiments and saved-model inference |
| `notebooks` | Kaggle workflows |
| `tests` | Application regression checks |
| `scripts` | Preview and publishing utilities |

## Development

Use Node.js 20 or later. The interface has no build step or production npm dependencies.

```text
npm test
npm run preview
```

The preview prints the addresses for both interfaces. The production entry point opens `UI-fixed`.

## Free deployment

Publish this directory to a GitHub repository. In **Settings → Pages**, choose **Deploy from a branch**, select `main` and `/ (root)`, then save. The `.nojekyll` file preserves static assets. The application also works on other static hosts that serve HTTPS, JavaScript modules and `.gz` files.

Research processing runs in private Kaggle notebooks. Hosted interaction runs in the visitor's browser. The project requires no hosted inference key, database subscription or application server.

## Research workflow

The corpus notebook downloads the pinned source files, verifies their hashes, processes every row, generates labels, saves exact evidence spans, exports review worksheets and produces the application data. Upload the `processing` assets privately to Kaggle and attach them to the corpus notebook.

Run `notebooks/baseline_preparation.ipynb` to prepare the conventional classifiers. For evaluation with the existing data, attach the completed corpus and baseline-preparation notebooks as private Kaggle inputs and run `notebooks/existing_data_evaluation.ipynb`. Its source bundle includes the fixed pilot families. The workflow maps ratings to three sentiment classes, keeps duplicate families together, reserves pilot examples for training, compares the feedback pipeline and sentiment encoder with TF-IDF logistic regression and calibrated linear SVM, and evaluates the selected candidate on the reserved test split. Reuse checks the saved classifiers, split record and ordered validation IDs.

The separate `notebooks/review_benchmark.ipynb` supports an independent later study using completed review decisions. It compares conventional models, DistilBERT, Qwen and a bounded Qwen agent. This workflow is separate from the existing-data evaluation.

Detailed processing records and the reviewer report are supplied separately with the research delivery.

## Third-party notices

Bundled DM Sans font files are distributed under the SIL Open Font License in `shared/fonts/OFL.txt`. Model identifiers and pinned revisions are in `shared/browser_models.json`. The [Federal University Lokoja Bank-Review dataset](https://huggingface.co/datasets/Federal-University-Lokoja/Bank-Review) declares Apache-2.0; its license text is in `processing/DATASET_LICENSE.txt`, and the pinned file manifest is in `processing/expected_source_manifest.json`.
