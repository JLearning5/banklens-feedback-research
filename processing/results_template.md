# Banking feedback results

Processed records: **{{TOTAL_ROWS}}**  
Bank files: **{{BANKS}}**  
Distinct non-empty review texts: **{{UNIQUE_TEXTS}}**  
Records with sentiment labels: **{{LABELLED_ROWS}}**  
Records in the review queue: **{{REVIEW_ROWS}}**

## Sentiment

{{SENTIMENT_TABLE}}

## Bank distribution

{{BANK_TABLE}}

## Service aspects

{{ASPECT_TABLE}}

## Files

- `labelled_reviews.csv` and `labelled_reviews.parquet`: complete review table.
- `by_bank/`: individual bank tables.
- `aspect_evidence.csv` and `aspect_evidence.parquet`: aspect polarity and supporting text.
- `review_queue.csv`: records selected for review.
- `review_worksheet.csv`: review worksheet with blank reviewer fields.
- `summaries/`: grouped results.
- `figures/`: charts.
- `reviewer_only/reviewer_notes.md`: reviewer document.
