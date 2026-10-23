
import pytest

REQUIRED_SECTIONS = [
    "OBSERVATIONS:",
    "CALCULATIONS:",
    "EVIDENCE:",
    "ASSESSMENT:",
    "RECOMMENDATION:",
    "SAVINGS:",
    "MISSING_EVIDENCE:",
    "CONFIDENCE:",
]


def validate_output_format(response: str) -> list[str]:
    """Return required report sections missing from the response."""
    if not isinstance(response, str) or not response.strip():
        return REQUIRED_SECTIONS.copy()

    response_upper = response.upper()

    return [
        section
        for section in REQUIRED_SECTIONS
        if section not in response_upper
    ]


def test_valid_report_contains_all_sections():
    response = """
    OBSERVATIONS:
    CPU utilization is low.

    CALCULATIONS:
    $600 * 50% = $300 savings.

    EVIDENCE:
    Current monthly cost is $600.

    ASSESSMENT:
    Rightsizing may be worth investigating.

    RECOMMENDATION:
    Validate peak workload performance.

    SAVINGS:
    $300 per month.

    MISSING_EVIDENCE:
    Historical peak utilization.

    CONFIDENCE:
    Moderate.
    """

    assert validate_output_format(response) == []


def test_missing_confidence_section():
    response = """
    OBSERVATIONS:
    Example.

    CALCULATIONS:
    Example.

    EVIDENCE:
    Example.

    ASSESSMENT:
    Example.

    RECOMMENDATION:
    Example.

    SAVINGS:
    Not estimable.

    MISSING_EVIDENCE:
    Proposed pricing.
    """

    missing = validate_output_format(response)

    assert missing == ["CONFIDENCE:"]


def test_empty_response_fails_validation():
    assert validate_output_format("") == REQUIRED_SECTIONS


def test_none_response_fails_validation():
    assert validate_output_format(None) == REQUIRED_SECTIONS


def test_section_matching_is_case_insensitive():
    response = "\n".join(
        section.lower() for section in REQUIRED_SECTIONS
    )

    assert validate_output_format(response) == []
