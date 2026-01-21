from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

from langgraph.graph import add_messages
from typing_extensions import Annotated, NotRequired


import operator


class OverallState(TypedDict):
    messages: Annotated[list, add_messages]
    search_query: Annotated[list, operator.add]
    web_research_result: Annotated[list, operator.add]
    sources_gathered: Annotated[list, operator.add]
    initial_search_query_count: int
    max_research_loops: int
    research_loop_count: int
    reasoning_model: str
    # Optional: Required for multi-tenant deployments, but safe to omit for CLI/Studio
    tenant_id: NotRequired[str]
    # URL-only mode: canonicalized URL fields (set by validate_input node)
    canonical_url: NotRequired[str]  # Full URL with scheme
    website_domain: NotRequired[str]  # Just the host (no scheme)
    input_valid: NotRequired[bool]  # Whether input passed validation
    # Run tracking: unique ID for correlating logs across nodes
    enrichment_run_id: NotRequired[str]  # Generated in validate_input
    # Strict failure mode: set when a node encounters a non-throttle error
    _fatal_error: NotRequired[bool]  # True if fatal error occurred
    _fatal_error_node: NotRequired[str]  # Node where error occurred
    _fatal_error_message: NotRequired[str]  # Error message
    # Throttle error flag (routes to END early)
    _throttle_error: NotRequired[bool]


class ReflectionState(TypedDict):
    is_sufficient: bool
    knowledge_gap: str
    follow_up_queries: Annotated[list, operator.add]
    research_loop_count: int
    number_of_ran_queries: int
    # Strict failure mode: propagate fatal error from upstream or set by this node
    _fatal_error: NotRequired[bool]
    _fatal_error_node: NotRequired[str]
    _fatal_error_message: NotRequired[str]


class Query(TypedDict):
    query: str
    rationale: str


class QueryGenerationState(TypedDict):
    search_query: list[Query]


class WebSearchState(TypedDict):
    search_query: str
    id: str
    enrichment_run_id: NotRequired[str]  # Passed from continue_to_web_research


@dataclass(kw_only=True)
class SearchStateOutput:
    running_summary: str = field(default=None)  # Final report
