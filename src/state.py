from typing import TypedDict, List, Optional


class Finding(TypedDict):
    query: str
    content: str
    source: Optional[str]


class ResearchState(TypedDict):
    topic: str
    findings: List[Finding]
    summary: Optional[str]
    report: Optional[str]
    human_feedback: Optional[str]
    is_approved: bool
    revision_count: int
    max_revisions: int
    final_report: Optional[str]
