import pytest

from bluewhale.evals.run import GATES, run_eval
from bluewhale.llm.prompts import get_prompt, list_prompts


def test_prompt_versions_and_rendering():
    assert list_prompts()["grounded_answer"] == [1, 2]
    v1, latest = get_prompt("grounded_answer", 1), get_prompt("grounded_answer")
    assert latest.version == 2 and v1.fingerprint != latest.fingerprint
    system, user = latest.render(context="[1] x", question="why?")
    assert "untrusted" in system and user.endswith("Question: why?")
    with pytest.raises(KeyError):
        latest.render(context="only one value")
    with pytest.raises(KeyError):
        get_prompt("grounded_answer", 9)


def test_eval_suite_meets_regression_gates():
    report = run_eval()
    for metric, threshold in GATES.items():
        assert getattr(report, metric) >= threshold, (metric, report.failures)


@pytest.mark.parametrize("version", [1, 2])
def test_both_prompt_versions_stay_grounded(version):
    assert run_eval(prompt_version=version).refusal == 1.0
