from bluewhale.matching.skills import extract_skills, extract_years
from bluewhale.rag.chunking import chunk_document, split_sections
from bluewhale.rag.documents import clean_text, detect_doc_type, make_document

JD = """Senior Data Engineer - Example Co
About the role
We move data.
Responsibilities
- Build pipelines in Python and SQL.
- Run Kafka consumers.
Requirements
- 5+ years of experience with Python.
Apply now
"""


def test_clean_text_decodes_double_escaped_entities_and_drops_chrome():
    raw = "Requirements&amp;amp;nbsp;\n\n\n\nPython &amp;#x27;s\u0007 fine\nApply now and win"
    out = clean_text(raw)
    assert "&amp;" not in out and "\u0007" not in out
    assert "Apply now" not in out
    assert "\n\n\n" not in out


def test_clean_text_collapses_duplicated_body():
    body = "Responsibilities\n" + "Build reliable services in Python. " * 20
    assert clean_text(body + "\n" + body).count("Responsibilities") == 1


def test_doc_type_detection():
    assert detect_doc_type(JD) == "job_description"
    assert detect_doc_type("Professional Summary\nWork Experience\nEducation") == "resume"


def test_years_ignore_company_history():
    assert extract_years("Operating for over 100 years. Requires 3-5 years of experience.") == 3
    assert extract_years("We have been around for 40 years") is None


def test_skill_aliases_and_word_boundaries():
    skills = extract_skills("Built on k8s with Node.js and AWS Bedrock; Java 21; javascript")
    assert {"kubernetes", "javascript", "aws bedrock", "java"} <= set(skills)
    assert "java" not in extract_skills("javascript only")


def test_sections_and_chunks_carry_title_and_metadata():
    doc = make_document(JD, doc_id="jd1")
    sections = [s for s, _ in split_sections(doc.text)]
    assert "responsibilities" in sections and "requirements" in sections
    chunks = chunk_document(doc, max_tokens=40, overlap_tokens=10)
    assert chunks and all(c.text.startswith("Senior Data Engineer") for c in chunks)
    assert all(c.metadata["doc_type"] == "job_description" for c in chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_long_section_is_split_with_overlap():
    body = "Experience\n" + " ".join(f"Shipped feature number {i} to production." for i in range(80))
    chunks = chunk_document(make_document(body, doc_id="r"), max_tokens=60, overlap_tokens=20)
    assert len(chunks) > 3
    # the last sentence of one chunk appears again at the start of the next
    tail = chunks[0].text.split(". ")[-1]
    assert tail.split()[-1] in chunks[1].text


def test_reingesting_unchanged_document_is_a_noop(services):
    first = services.rag.ingest_text(JD, doc_id="dup")
    second = services.rag.ingest_text(JD, doc_id="dup")
    assert not first.skipped and second.skipped
    changed = services.rag.ingest_text(JD + "\nNice to have\n- Go", doc_id="dup")
    assert not changed.skipped
