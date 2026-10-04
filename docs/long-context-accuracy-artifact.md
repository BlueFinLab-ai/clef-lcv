# Long-context accuracy artifact

Recorded October 3, 2026. Status: observed, cause unresolved; further investigation
deferred at the user's request. This note does not change serving limits or runtime
behavior.

## Observation

Clef Flash sometimes selected an incorrect answer on a synthetic long-context
fact-retrieval test even when inference completed with finite outputs. This is
separate from the out-of-memory failures encountered while measuring capacity.

The test placed `START ALERT COLOR: RED.` at the beginning, repeated `word` as
filler, and placed `END ALERT COLOR: BLUE. Checkout is down; act immediately.` at
the end. Four native questions asked about the start color, end color, immediate
attention, and urgency. Total token counts included the question schema and
template. This repetitive input is a stress test, not a representative document
benchmark or an established general accuracy limit.

The same compact NF4 Flash checkpoint with FP16 compute was tested on a single
RTX 3070 Ti using chunked prefill and on a spare single RTX 3090 using both
unchunked and chunked prefill. Persistent prefix and image-feature caches were
disabled. The native decision head was unchanged.

| Reference input | Unchunked end-color result | Maximum probability difference, 1K chunks | Maximum probability difference, 4K chunks |
|---|---|---:|---:|
| 32,768 tokens | Incorrect: red 54.09%, blue 45.91% | 0.097 percentage points | 0.049 percentage points |
| 49,152 tokens | Correct: blue 51.56%, red 48.44% | 0.030 percentage points | 0.131 percentage points |

Both chunk sizes selected the same answers as the unchunked reference on the
3090. On the 3070 Ti, accuracy also varied nonmonotonically: the 8K-first/1K
continuation path failed the original end-color check at 32K and 36K, passed at
40K, and failed at 41,984 tokens. Reversed/equal-color controls at its largest
completed input passed. These option probabilities are model outputs, not an
established measure of calibrated confidence.

The evidence points away from chunking as the main cause of these particular
errors. It does not isolate NF4 quantization, FP16 compute, the language backbone,
or the decision head. The Full 27B model was not tested for this artifact.

## Decision-head observations and possible explanations

Inspection of `vendor/cloudflare/joint_schema_model.py` shows that the head
projects every retained token representation into its attention memory. It
averages only question and option spans, and uses the final token representation
as a global signal. Evidence-routing attention and subsequent decoder layers
operate over the token memory. Final option scores combine a lexical similarity
prior with learned joint evidence scores.

Possible explanations, none established by the current measurements:

- Evidence selection may become unstable among many repetitive or competing
  token representations. The memory includes context and question/schema tokens.
- Question/option span summaries or the final-token global signal may not retain
  the distinctions needed by this long input.
- The lexical prior may influence a close decision when routed evidence is weak;
  its actual contribution on the failing input has not been measured.
- Backbone representations, precision, or the model's training distribution may
  contribute. The configured position limit does not establish retrieval accuracy.

The head adds no explicit positional embedding in this implementation, but its
inputs are contextual backbone representations; this does not establish that it
is position-blind. Attention weights are not currently exported. Head configuration
is explicit in code; the uncertainty concerns the failure mechanism.

## Deferred diagnostics

If investigation resumes, compare varied filler and fact positions, one question
versus four, and dense higher-precision Flash against the compact NF4 checkpoint.
Instrument separate prior/joint score contributions and evidence routing, then
validate suspected evidence with controlled input changes. Use representative
long documents before deriving a production accuracy limit.

## Evidence record

The raw measurements and the experimental source are kept outside this
repository. The experiment was not deployed as a runtime change.
