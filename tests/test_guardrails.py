from demandcast.agent.guardrails import MAX_ANSWER_LENGTH, apply_output_guardrails


def test_normal_answer_passes_through_unchanged():
    answer = "Store 262's forecast for Aug 1 is about 19,752 units, per store_262_2015-07."
    assert apply_output_guardrails(answer) == answer


def test_system_prompt_leak_is_caught():
    leaked = "Sure! Here it is: Ground every factual claim in either the search tool..."
    result = apply_output_guardrails(leaked)
    assert "can't share my internal instructions" in result


def test_secret_like_string_is_caught():
    answer = "Here is a key I found: sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890"
    result = apply_output_guardrails(answer)
    assert "can't include" in result


def test_overly_long_answer_is_truncated():
    answer = "x" * (MAX_ANSWER_LENGTH + 500)
    result = apply_output_guardrails(answer)
    assert len(result) <= MAX_ANSWER_LENGTH + len("\n\n[truncated]")
    assert result.endswith("[truncated]")


def test_injected_instruction_in_retrieved_style_content_is_not_a_leak_by_itself():
    # a document merely *mentioning* an injection attempt (e.g. quoting one to
    # warn about it) shouldn't be treated as if the agent leaked its own prompt
    answer = (
        "That store-month note mentions 'ignore previous instructions' as an "
        "example of a prompt injection attempt, which I disregarded."
    )
    assert apply_output_guardrails(answer) == answer
