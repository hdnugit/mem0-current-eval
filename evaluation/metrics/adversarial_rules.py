"""Label rules for the adversarial (LOCOMO category 5) judge.

Pure functions, no API calls: the judge model answers three yes/no checks and
these rules turn them into a label. Kept apart from llm_judge.py so stored
checks can be re-labelled (eval_adversarial.py --report_only) without a key.
"""

import re

ADVERSARIAL_LABELS = ("DECLINED", "REATTRIBUTED", "TRAP", "OTHER")
ADVERSARIAL_CORRECT = ("DECLINED", "REATTRIBUTED")

# Words that make a statement a refusal or a denial.
_REFUSAL_WORDS = re.compile(
    r"\b(no|not|never|none|nothing|nobody|unknown|unclear|unspecified|cannot|unable|without|neither|nor|"
    r"lacks?|absent|insufficient)\b|n't",
    re.IGNORECASE,
)


def _normalise(text):
    text = str(text).lower().replace("’", "'")
    text = re.sub(r"[\"'`“”‘]", "", text)
    return re.sub(r"\s+", " ", text).strip(" .")


def valid_refusal_quote(quote, answer):
    """A refusal counts only if the judge's quote is really in the answer and refuses something.

    gpt-4o-mini often sets explicit_refusal for an answer that merely differs
    from the trap, "quoting" the whole answer or words that are not in it.
    """
    quote_n, answer_n = _normalise(quote), _normalise(answer)
    return bool(quote_n) and quote_n in answer_n and bool(_REFUSAL_WORDS.search(str(quote).replace("’", "'")))


def adversarial_label(gives_trap, explicit_refusal, reassigns):
    """Turn the three checks into one label (first match wins).

    TRAP          the answer asserts the trap for the person asked about (even if it hedges)
    DECLINED      it explicitly says the information is missing or the premise is false
    REATTRIBUTED  it moves the fact to the right person without saying the question is wrong
    OTHER         it accepts the premise and gives a different answer
    """
    if gives_trap:
        return "TRAP"
    if explicit_refusal:
        return "DECLINED"
    if reassigns:
        return "REATTRIBUTED"
    return "OTHER"


def label_from_checks(answer, checks):
    """Label for an answer given the judge's stored checks (applies the quote rule)."""
    refusal = bool(checks.get("explicit_refusal")) and valid_refusal_quote(checks.get("refusal_quote", ""), answer)
    return adversarial_label(bool(checks.get("gives_trap")), refusal, bool(checks.get("reassigns")))
