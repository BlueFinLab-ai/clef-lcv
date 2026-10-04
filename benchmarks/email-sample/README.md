# Public email benchmark sample

100 **synthetic adaptations** of the scenarios in the private email benchmark.
These are newly authored fixtures, not redacted copies of real messages. All
senders and account, purchase, event and household details are fictional. The
private emails and their identifiers are not included.

The fixture preserves the original category counts and broad body-length classes,
including distinctions between market news and financial records, a sales pitch
and an operational alert, and a public social post and a private household update.
The distributed fixture contains new wording, senders, subjects and shuffled
sample IDs, with no mapping back to the original messages. No raw message/thread IDs, source timestamps,
real addresses, phone numbers, account numbers, card suffixes, tracking links,
attachments or original label explanations are exported. Household learning
reports are rewritten completely and contain no actual student information.

## Files

| File | Purpose |
|---|---|
| [dataset.json](dataset.json) | Emails, new sample IDs and reference categories from GPT Sol 6.1 |
| [categories.json](categories.json) | Twelve-category classification policy |
| [category-guide.txt](category-guide.txt) | Reusable context for the model |
| [manifest.json](manifest.json) | Checksums, category counts and length classes |
| [validation.json](validation.json) | Publication checks and initial live validation |

Each record contains an `email` with `from`, `subject` and `body`; its reference
category and reason are separate. The runner sends **only the email, guide and
category definitions** to the service. Reference labels do not enter the prompt.

There are 41 marketing, 26 newsletter, 10 community, 6 event, 6 finance, 4 personal,
3 order, 2 security, 1 recruiting and 1 service cases. The policy also defines
`work` and `other`, but this fixture has no positive examples for those categories.
It is an imbalanced workload sample, not a balanced classification evaluation.

## Workload and interpretation

Bodies use coarse target classes of approximately 512, 1,024, 2,048, 4,096 and
8,192 characters. Freshly authored, repeated navigation/help sections extend the
short scenario descriptions to representative workload sizes. Bodies finish on
paragraph boundaries. Character length is not model token length; actual encoded
token counts are reported by the server.

The public bodies total about 278,000 characters. This broadly preserves workload
size, **not exact token lengths, source wording or semantic difficulty**. The
synthetic main messages are often clearer than their private counterparts, and
repeated boilerplate can affect tokenization and classification. Treat this as a
portable timing/cache/concurrency fixture and a new label-agreement baseline.
Do not compare its times or agreement percentages directly with historical runs
of the private email set.

## Reference labels

The reference categories were produced by GPT Sol 6.1, the hosted model used as
the quality baseline, classifying each email under the same category policy.
Agreement figures therefore measure how closely a model matches GPT Sol 6.1.
They are not independently adjudicated human ground truth.

## Run

From the project root, validate the fixture without a GPU or network:

```sh
python3 scripts/benchmark_emails.py --validate-only
```

An uncached serial baseline:

```sh
python3 scripts/benchmark_emails.py --url http://localhost:8080 \
  --concurrencies 1 --passes 1 --cache off \
  --output local/benchmarks/email-uncached.json
```

Then compare client concurrency and repeated passes with caching enabled:

```sh
python3 scripts/benchmark_emails.py --url http://localhost:8080 \
  --concurrencies 1,4,8 --passes 2 --cache on \
  --output local/benchmarks/email-cached.json
```

The model ID is discovered from `/v1/models`, so the same runner works with Flash
and Full. Set `--model` explicitly if needed. Use `--limit 10` for a smoke test.
Results include wall time, emails/second, client median/p95, server processing,
queue wait, input tokens, cache hits/reused tokens, reference agreement and a
confusion matrix. Error responses are counted and make the command exit nonzero.
Output files cannot overwrite an earlier run.

The runner does not clear server caches. Its first enabled pass is an **observed
first pass**, not guaranteed cold. Later concurrency trials inherit cache state.
For isolated cold comparisons, restart a dedicated benchmark service between
trials; record compilation warmup and competing GPU activity separately.
Concurrency means outstanding client requests; the current service processes
them through its single-GPU FIFO worker. This is not a claim of GPU batching.

## Initial live validation

Clef Flash on one RTX 3090 completed two 100-request passes with four outstanding
client calls and caching enabled. Wall times were 53.41 and 54.58 seconds, with
zero request errors and 96/100 agreement with the GPT Sol 6.1 reference labels on
both passes. All 100 choices were unchanged between passes. Existing caches were
not cleared; ten-case smoke tests preceded these runs. This is an observed serving
snapshot, not a controlled cold-cache comparison or a performance guarantee.

The four disagreements were the recipe promotion, streaming documentary launch,
property-price advertisements and vessel sales catalog: the model selected
`newsletters` instead of the reference `marketing`. Their reference labels remain
unchanged because the main purpose is commercial engagement or shopping. These
cases preserve useful editorial-versus-promotional boundary tests.

Full supports the same API and runner, but this fixture's initial live validation
used Flash only. Details and aggregate privacy checks are in the validation file.

## Privacy review

Preparation checked the fixture for real source email addresses, original
subjects/identifiers, long verbatim passages and source student-name fields.
The distributed validator checks the schema, checksums, fictional sender domains,
and unexpected URLs, long numeric identifiers and phone-like strings. These
guards help detect accidental fixture edits; they are not a general-purpose PII
detector for arbitrary new email exports. Do not replace this dataset with a raw
mailbox export when preparing a public release.
