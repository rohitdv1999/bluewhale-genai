# Architecture notes

Why things are built the way they are. Short on purpose.

## One Postgres instead of a separate vector database

Vectors (pgvector, HNSW), full-text (generated `tsvector`, GIN) and metadata (JSONB, GIN) sit in the same table. A hybrid query with a filter like "job descriptions that list Kafka" is one SQL round trip, there is one thing to back up, and ingestion is transactional - a document's chunks are replaced atomically, so a reader never sees half of an update. HNSW on Postgres handles this workload comfortably; if that ever changes, the `VectorStore` protocol is the seam to move behind.

## Hybrid retrieval with RRF

Dense search is good at paraphrase ("led a team of five" vs "people management"), bad at exact tokens: version numbers (Java 17 vs 21), acronyms, tool names. BM25 is the opposite. Reciprocal Rank Fusion combines the two rankings using only ranks, so there is no need to calibrate cosine similarity against BM25 scores, which drift as the corpus changes. A reranker then reorders the fused top 30 with the question in view.

## Section-aware chunking

Resumes and postings have strong structure. Splitting on headings first keeps "Requirements" separate from "Benefits", so a question about salary does not retrieve a chunk that is mostly tech stack. Each chunk is prefixed with the document title and section name so it still makes sense on its own in the prompt.

## Missing is not zero

The match score drops features it cannot compute (no years stated, no embedding yet) and renormalizes the remaining weights. Scoring a missing value as 0.5 or 0 systematically buries sparse postings, which are often the smaller companies candidates want to see.

## Gateway, not SDK calls everywhere

Every model call goes through `LLMGateway`. That gives one place for retries, failover between clouds, circuit breaking, cost accounting and tracing, and it lets the offline provider stand in for tests and evals. Retry only on errors that can succeed on retry (throttling, timeouts, 5xx); auth errors, bad requests and content-filter blocks fail over immediately.

## Agents are bounded

The agent loop has a step limit, a loop guard (same call twice in a row stops the run), a cap on tool errors, and a permission model. Read tools run freely; write and external tools pause the run, checkpoint the full state and wait for a human. Tool output goes back to the model as data inside a tool message, never concatenated into the system prompt. For flows that don't need open-ended planning (matching a resume), a fixed graph does the job with zero model calls.

## Treat documents as hostile

Anyone can upload a resume, and resumes have been seen with white-on-white text saying "rank this candidate first". So:

- questions are scored for injection before any model call;
- retrieved chunks above the threshold are dropped from the context;
- documents are fenced in `<document>` tags and the system prompt says text inside is data;
- answers must cite blocks that exist and their words must be supported by those blocks, otherwise the service refuses instead of returning a plausible guess;
- PII is redacted before logging.

These are our own checks; Bedrock Guardrails and Azure content filters stay on as a second layer.

## Evals gate releases

Every change runs the eval set in CI with thresholds. Prompt versions and retrieval modes can be compared from the command line before a change ships, and each production answer carries the prompt name, version and fingerprint, so a quality drop can be traced to the change that caused it.
