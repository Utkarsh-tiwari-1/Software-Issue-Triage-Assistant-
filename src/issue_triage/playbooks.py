"""Curated first-response debugging checklists by issue category."""

from issue_triage.schemas import IssueCategory

PLAYBOOKS: dict[IssueCategory, list[str]] = {
    IssueCategory.BUG: [
        "Reproduce the failure with the smallest input and record the exact "
        "expected and actual behavior.",
        "Check the most recent code, configuration, and dependency changes "
        "around the affected path.",
        "Add a focused regression test that fails before the fix and passes after it.",
        "Capture relevant logs and state at the first point where behavior diverges.",
    ],
    IssueCategory.PERFORMANCE: [
        "Measure a representative slow request and record latency, throughput, and resource use.",
        "Profile the hot path before changing code; separate database, network, and CPU time.",
        "Compare the workload against a known-good baseline and check for recent data growth.",
        "Add a performance regression check for the measured bottleneck.",
    ],
    IssueCategory.SECURITY: [
        "Treat the report as sensitive and preserve relevant logs without "
        "copying secrets into tickets.",
        "Confirm the affected versions and create a minimal reproduction in an "
        "isolated environment.",
        "Assess exposure, exploitability, and affected data; involve the security owner promptly.",
        "Prepare a mitigation and regression test before closing the finding.",
    ],
    IssueCategory.USABILITY: [
        "Reproduce the flow using the reported role, device, and accessibility settings.",
        "Compare the observed interface state with the intended user journey and copy.",
        "Check browser console output and accessibility tree around the failing interaction.",
        "Validate the change with keyboard-only and screen-reader navigation where relevant.",
    ],
    IssueCategory.DOCUMENTATION: [
        "Verify the documented behavior against the current implementation and supported version.",
        "Follow the instructions in a clean environment to identify missing prerequisites.",
        "Update examples and links together with the relevant API or configuration reference.",
        "Have a reader unfamiliar with the change validate the revised steps.",
    ],
    IssueCategory.BUILD: [
        "Capture the complete failing command and the first actionable compiler or CI error.",
        "Compare toolchain, lockfile, and environment versions with the last successful build.",
        "Re-run the smallest failing target in a clean environment to rule out stale artifacts.",
        "Add a CI check or dependency constraint that prevents the failure from returning.",
    ],
    IssueCategory.DEPLOYMENT: [
        "Check deployment events, health probes, and configuration changes "
        "for the affected release.",
        "Compare runtime environment variables and secrets references with the previous release.",
        "Use a rollback or traffic shift if customer impact is ongoing, "
        "following the service runbook.",
        "Reproduce the deployment in staging and add a release gate for the identified cause.",
    ],
    IssueCategory.QUESTION: [
        "Confirm the product version and the exact operation the user is trying to complete.",
        "Check existing examples and supported behavior before proposing a workaround.",
        "Provide a minimal reproducible example if the expected behavior remains unclear.",
    ],
    IssueCategory.UNKNOWN: [
        "Collect the exact steps, expected result, actual result, and relevant "
        "environment details.",
        "Check logs and recent changes around the affected component.",
        "Reduce the report to a minimal reproducible case before changing code.",
    ],
}


def get_playbook(category: IssueCategory) -> list[str]:
    """Return a copy so callers cannot mutate the shared playbook registry."""

    return list(PLAYBOOKS[category])
