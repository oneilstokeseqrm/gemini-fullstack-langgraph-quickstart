import json
import logging
import os
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any
from urllib.parse import urlparse

from agent.tools_and_schemas import SearchQueryList, Reflection, AccountProfile, Source, CRMField
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client
from google.genai.errors import ClientError

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    web_searcher_instructions,
    reflection_instructions,
    account_enrichment_query_instructions,
    account_enrichment_answer_instructions,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

load_dotenv()

# =============================================================================
# Logging Setup
# =============================================================================
logger = logging.getLogger("enrichment")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)

# =============================================================================
# Rate Limiting & Concurrency Control
# =============================================================================

# Global semaphore to limit concurrent enrichment runs
# Using BoundedSemaphore to detect over-release bugs (raises ValueError on over-release)
MAX_CONCURRENT_RUNS = int(os.getenv("MAX_CONCURRENT_RUNS", "1"))
_run_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_RUNS)

# Global semaphore to limit concurrent web research calls (within a run)
# Using BoundedSemaphore for safety
MAX_PARALLEL_SEARCHES = int(os.getenv("MAX_PARALLEL_SEARCHES", "1"))
_search_semaphore = threading.BoundedSemaphore(MAX_PARALLEL_SEARCHES)

# Track active runs for logging
_active_runs: Dict[str, Dict[str, Any]] = {}
_active_runs_lock = threading.Lock()

# Backoff configuration
INITIAL_BACKOFF_SECONDS = float(os.getenv("INITIAL_BACKOFF_SECONDS", "2.0"))
MAX_BACKOFF_SECONDS = float(os.getenv("MAX_BACKOFF_SECONDS", "60.0"))
BACKOFF_MULTIPLIER = float(os.getenv("BACKOFF_MULTIPLIER", "2.0"))
MAX_RETRIES = int(os.getenv("MAX_API_RETRIES", "5"))

# Production-safe defaults
DEFAULT_INITIAL_QUERIES = int(os.getenv("DEFAULT_INITIAL_QUERIES", "1"))
DEFAULT_MAX_LOOPS = int(os.getenv("DEFAULT_MAX_LOOPS", "1"))

# API endpoint info (for logging)
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/"

# =============================================================================
# Forced Error Testing (dev/test only)
# =============================================================================
# Set FORCE_THROTTLE_NODE to one of: generate_query, web_research, reflection, finalize_answer
# When set, the specified node will behave as if it hit a 429 throttle error.
# This is for deterministic testing of the throttle path.
FORCE_THROTTLE_NODE = os.getenv("FORCE_THROTTLE_NODE", "").strip().lower()

# Set FORCE_ERROR_NODE to one of: web_research, reflection
# When set, the specified node will raise a non-throttle exception.
# This is for deterministic testing of the ENRICHMENT_FAILED path.
FORCE_ERROR_NODE = os.getenv("FORCE_ERROR_NODE", "").strip().lower()

# Set FORCE_NO_SOURCES=1 to simulate zero sources in finalize_answer.
# This tests the guard that prevents "success" when no evidence was gathered.
FORCE_NO_SOURCES = os.getenv("FORCE_NO_SOURCES", "").strip().lower() in ("1", "true", "yes")


def should_force_throttle(node_name: str) -> bool:
    """Check if a node should simulate a throttle error (dev/test only)."""
    return FORCE_THROTTLE_NODE == node_name.lower()


def should_force_error(node_name: str) -> bool:
    """Check if a node should simulate a non-throttle error (dev/test only)."""
    return FORCE_ERROR_NODE == node_name.lower()


class ForcedThrottleError(Exception):
    """Simulated throttle error for testing."""
    def __str__(self):
        return "429 RESOURCE_EXHAUSTED: Forced throttle for testing (FORCE_THROTTLE_NODE)"


class ForcedError(Exception):
    """Simulated non-throttle error for testing ENRICHMENT_FAILED path."""
    def __init__(self, node_name: str):
        self.node_name = node_name

    def __str__(self):
        return f"Forced non-throttle error in {self.node_name} for testing (FORCE_ERROR_NODE)"


def generate_run_id() -> str:
    """Generate a unique run ID for tracking."""
    return str(uuid.uuid4())


def acquire_run_slot(run_id: str) -> bool:
    """Acquire a slot in the run semaphore. Returns True if acquired."""
    acquired = _run_semaphore.acquire(blocking=True, timeout=300)  # 5 min max wait
    if acquired:
        logger.info(f"[{run_id[:8]}] RUN_ACQUIRED semaphore_slots={MAX_CONCURRENT_RUNS}")
    else:
        logger.warning(f"[{run_id[:8]}] RUN_TIMEOUT failed to acquire semaphore after 300s")
    return acquired


def release_run_slot(run_id: str):
    """Release a slot in the run semaphore.

    With BoundedSemaphore, over-release raises ValueError - we log this loudly
    as it indicates a bug in semaphore lifecycle management.
    """
    try:
        _run_semaphore.release()
        logger.info(f"[{run_id[:8]}] RUN_RELEASED")
    except ValueError:
        # BoundedSemaphore raises ValueError on over-release - this is a bug!
        logger.error(
            f"[{run_id[:8]}] RUN_RELEASE_ERROR BoundedSemaphore over-release detected! "
            f"This indicates a bug in semaphore lifecycle management."
        )


@dataclass
class CallMetrics:
    """Tracks API call metrics for a single enrichment run."""
    run_id: str
    start_time: float = field(default_factory=time.time)
    generate_query_calls: int = 0
    web_research_calls: int = 0
    reflection_calls: int = 0
    finalize_calls: int = 0
    total_retries: int = 0
    throttle_events: int = 0


def get_run_metrics(run_id: str) -> CallMetrics:
    """Get or create metrics for a run."""
    with _active_runs_lock:
        if run_id not in _active_runs:
            _active_runs[run_id] = {"metrics": CallMetrics(run_id=run_id)}
        return _active_runs[run_id]["metrics"]


def log_call_start(node: str, model: str, run_id: str, attempt: int = 1):
    """Log the start of an API call."""
    metrics = get_run_metrics(run_id)
    logger.info(
        f"[{run_id[:8]}] API_CALL node={node} model={model} "
        f"endpoint={GEMINI_API_BASE} attempt={attempt}"
    )
    # Update metrics
    if node == "generate_query":
        metrics.generate_query_calls += 1
    elif node == "web_research":
        metrics.web_research_calls += 1
    elif node == "reflection":
        metrics.reflection_calls += 1
    elif node == "finalize_answer":
        metrics.finalize_calls += 1


def log_call_success(node: str, run_id: str, duration_ms: int):
    """Log a successful API call."""
    logger.info(f"[{run_id[:8]}] API_SUCCESS node={node} duration_ms={duration_ms}")


def log_throttle_event(node: str, run_id: str, attempt: int, error: Exception, backoff_seconds: float):
    """Log a 429/throttle event with full context."""
    metrics = get_run_metrics(run_id)
    metrics.throttle_events += 1
    metrics.total_retries += 1

    # Extract error details
    error_str = str(error)
    retry_after = None

    # Try to extract retry-after if present
    if hasattr(error, 'response') and error.response:
        retry_after = error.response.headers.get('Retry-After')

    logger.warning(
        f"[{run_id[:8]}] THROTTLE_429 node={node} attempt={attempt} "
        f"backoff_seconds={backoff_seconds:.1f} retry_after={retry_after} "
        f"error={error_str[:200]}"
    )


def log_run_summary(run_id: str):
    """Log a summary of API calls for a run."""
    with _active_runs_lock:
        if run_id not in _active_runs:
            return
        metrics = _active_runs[run_id]["metrics"]

    duration = time.time() - metrics.start_time
    total_calls = (
        metrics.generate_query_calls +
        metrics.web_research_calls +
        metrics.reflection_calls +
        metrics.finalize_calls
    )
    logger.info(
        f"[{run_id[:8]}] RUN_SUMMARY duration_s={duration:.1f} "
        f"total_calls={total_calls} "
        f"generate_query={metrics.generate_query_calls} "
        f"web_research={metrics.web_research_calls} "
        f"reflection={metrics.reflection_calls} "
        f"finalize={metrics.finalize_calls} "
        f"retries={metrics.total_retries} "
        f"throttle_events={metrics.throttle_events}"
    )


def exponential_backoff_with_jitter(attempt: int) -> float:
    """Calculate backoff time with exponential increase and jitter."""
    base_delay = INITIAL_BACKOFF_SECONDS * (BACKOFF_MULTIPLIER ** (attempt - 1))
    # Add jitter: 0.5x to 1.5x the base delay
    jitter = random.uniform(0.5, 1.5)
    delay = min(base_delay * jitter, MAX_BACKOFF_SECONDS)
    return delay


def is_throttle_error(error: Exception) -> bool:
    """Check if an error is a throttle/rate limit error."""
    # Check for forced throttle error (dev/test)
    if isinstance(error, ForcedThrottleError):
        return True
    error_str = str(error).lower()
    return (
        "429" in error_str or
        "resource_exhausted" in error_str or
        "rate limit" in error_str or
        "quota" in error_str
    )


def create_throttle_error_response(
    input_received: str,
    tenant_id: str,
    retry_after_seconds: Optional[float] = None
) -> dict:
    """Create a standardized throttle error response."""
    response = {
        "error": "UPSTREAM_THROTTLED",
        "message": "Gemini API temporarily throttled; please retry",
        "input_received": input_received,
        "tenant_id": tenant_id,
        "schema_version": "error-1.0.0"
    }
    if retry_after_seconds:
        response["retry_after_seconds"] = retry_after_seconds
    return response


if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# =============================================================================
# URL Canonicalization and Validation (URL-only mode)
# =============================================================================

@dataclass
class CanonicalizedURL:
    """Result of URL canonicalization."""
    canonical_url: str  # Full URL with scheme (e.g., https://example.com/path)
    website_domain: str  # Just the host (e.g., example.com)


def canonicalize_url(input_text: str) -> Tuple[Optional[CanonicalizedURL], Optional[str]]:
    """Canonicalize and validate a URL or domain input.

    Accepts:
    - Full URLs: https://example.com, http://example.com/path
    - Bare domains: example.com, www.example.com
    - Domains with paths: example.com/about

    Returns:
    - (CanonicalizedURL, None) on success
    - (None, error_message) on failure
    """
    text = input_text.strip()

    if not text:
        return None, "Input is empty"

    # Reject obvious non-URLs (emails, random text with spaces, etc.)
    if " " in text:
        return None, "Input contains spaces - not a valid URL or domain"
    if "@" in text and not text.startswith("http"):
        return None, "Input looks like an email address, not a URL"

    # Add scheme if missing
    if not text.startswith("http://") and not text.startswith("https://"):
        text = "https://" + text

    try:
        parsed = urlparse(text)

        # Validate: must have scheme and netloc (host)
        if not parsed.scheme or not parsed.netloc:
            return None, "Could not parse URL - missing scheme or host"

        # Validate: netloc must look like a domain (contains at least one dot or is localhost)
        netloc = parsed.netloc
        # Remove port if present for domain validation
        host = netloc.split(":")[0]

        if host != "localhost" and "." not in host:
            return None, f"Invalid domain: '{host}' - must contain at least one dot (e.g., example.com)"

        # Validate: domain parts should be reasonable
        # Basic check: no consecutive dots, no leading/trailing dots
        if ".." in host or host.startswith(".") or host.endswith("."):
            return None, f"Invalid domain format: '{host}'"

        # Build canonical URL (keep path, query, fragment if present)
        canonical_url = f"{parsed.scheme}://{parsed.netloc}"
        if parsed.path and parsed.path != "/":
            canonical_url += parsed.path
        if parsed.query:
            canonical_url += "?" + parsed.query
        if parsed.fragment:
            canonical_url += "#" + parsed.fragment

        # Extract domain (host without port, strip www. prefix for root domain)
        website_domain = host
        if website_domain.startswith("www."):
            website_domain = website_domain[4:]

        return CanonicalizedURL(
            canonical_url=canonical_url,
            website_domain=website_domain
        ), None

    except Exception as e:
        return None, f"URL parsing error: {str(e)}"


def create_error_response(
    error_code: str,
    message: str,
    input_received: str,
    tenant_id: str = "unknown"
) -> dict:
    """Create a standardized error response JSON."""
    return {
        "error": error_code,
        "message": message,
        "input_received": input_received,
        "tenant_id": tenant_id,
        "schema_version": "error-1.0.0"
    }


# =============================================================================
# Graph Nodes
# =============================================================================

def validate_input(state: OverallState, config: RunnableConfig) -> OverallState:
    """LangGraph node that validates and canonicalizes the URL input.

    This is the first node in the graph. It:
    - Generates a unique run_id for log correlation
    - Acquires the run semaphore (serializes concurrent runs)
    - Extracts the user input from messages
    - Canonicalizes the URL (adds https:// if missing, validates format)
    - Returns early with error JSON if input is invalid
    - Sets canonical_url, website_domain, and enrichment_run_id in state

    Args:
        state: Current graph state containing messages
        config: Configuration for the runnable

    Returns:
        Dictionary with state update including canonical_url, website_domain, input_valid, enrichment_run_id
    """
    # Generate unique run ID for this enrichment
    run_id = generate_run_id()

    # Acquire run semaphore to serialize concurrent enrichment runs
    if not acquire_run_slot(run_id):
        # Timeout waiting for semaphore - return error
        tenant_id = state.get("tenant_id", "unknown")
        user_input = get_research_topic(state["messages"])
        error_response = create_error_response(
            error_code="SYSTEM_BUSY",
            message="System is currently processing other requests. Please retry.",
            input_received=user_input,
            tenant_id=tenant_id
        )
        return {
            "messages": [AIMessage(content=json.dumps(error_response, indent=2))],
            "input_valid": False,
            "canonical_url": "",
            "website_domain": "",
            "enrichment_run_id": run_id,
        }

    # Initialize metrics for this run
    get_run_metrics(run_id)

    # Extract user input from messages
    user_input = get_research_topic(state["messages"])
    tenant_id = state.get("tenant_id", "unknown")

    # Canonicalize and validate
    result, error = canonicalize_url(user_input)

    if error:
        # Invalid input - release semaphore and return error response
        release_run_slot(run_id)
        error_response = create_error_response(
            error_code="INVALID_URL",
            message=f"Please provide a valid company URL or domain (e.g., https://example.com or example.com). {error}",
            input_received=user_input,
            tenant_id=tenant_id
        )
        return {
            "messages": [AIMessage(content=json.dumps(error_response, indent=2))],
            "input_valid": False,
            "canonical_url": "",
            "website_domain": "",
            "enrichment_run_id": run_id,
        }

    # Valid input - store canonicalized values (semaphore stays acquired)
    logger.info(f"[{run_id[:8]}] INPUT_VALID url={result.canonical_url} domain={result.website_domain}")
    return {
        "input_valid": True,
        "canonical_url": result.canonical_url,
        "website_domain": result.website_domain,
        "enrichment_run_id": run_id,
    }


def route_after_validation(state: OverallState) -> str:
    """Route to either continue processing or end early if input was invalid."""
    if state.get("input_valid", False):
        return "generate_query"
    else:
        return END


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries for company URL enrichment.

    Uses Gemini 2.0 Flash to create optimized search queries to gather company
    information from the provided URL. Includes exponential backoff on throttling.

    Args:
        state: Current graph state containing the canonical URL
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)
    run_id = state.get("enrichment_run_id", "unknown")
    model_name = configurable.query_generator_model
    canonical_url = state.get("canonical_url", get_research_topic(state["messages"]))
    tenant_id = state.get("tenant_id", "unknown")

    try:
        # Use production-safe default for initial queries
        if state.get("initial_search_query_count") is None:
            state["initial_search_query_count"] = min(
                configurable.number_of_initial_queries,
                DEFAULT_INITIAL_QUERIES
            )

        # URL-only mode: always use enrichment prompt with canonical URL
        current_date = get_current_date()

        formatted_prompt = account_enrichment_query_instructions.format(
            current_date=current_date,
            company_url=canonical_url,
            number_queries=state["initial_search_query_count"],
        )

        # Retry loop with exponential backoff
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # Check for forced throttle (dev/test only) - inside loop to go through throttle path
                if should_force_throttle("generate_query"):
                    logger.info(f"[{run_id[:8]}] FORCED_THROTTLE node=generate_query (FORCE_THROTTLE_NODE set)")
                    raise ForcedThrottleError()

                log_call_start("generate_query", model_name, run_id, attempt)
                start_time = time.time()

                # init Gemini 2.0 Flash (no SDK retries - we handle them)
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=1.0,
                    max_retries=0,  # We handle retries with backoff
                    api_key=os.getenv("GEMINI_API_KEY"),
                )
                structured_llm = llm.with_structured_output(SearchQueryList)
                result = structured_llm.invoke(formatted_prompt)

                duration_ms = int((time.time() - start_time) * 1000)
                log_call_success("generate_query", run_id, duration_ms)
                return {"search_query": result.query, "enrichment_run_id": run_id}

            except (ChatGoogleGenerativeAIError, ClientError, ForcedThrottleError, Exception) as e:
                last_error = e
                # For ForcedThrottleError, fail immediately (no retries)
                is_forced = isinstance(e, ForcedThrottleError)
                if is_throttle_error(e) and attempt < MAX_RETRIES and not is_forced:
                    backoff = exponential_backoff_with_jitter(attempt)
                    log_throttle_event("generate_query", run_id, attempt, e, backoff)
                    time.sleep(backoff)
                elif is_throttle_error(e):
                    # Final throttle failure (or forced throttle) - release semaphore and return structured error
                    log_throttle_event("generate_query", run_id, attempt, e, 0)
                    log_run_summary(run_id)
                    release_run_slot(run_id)
                    error_response = create_throttle_error_response(
                        input_received=canonical_url,
                        tenant_id=tenant_id,
                        retry_after_seconds=exponential_backoff_with_jitter(attempt)
                    )
                    return {
                        "search_query": [],
                        "messages": [AIMessage(content=json.dumps(error_response, indent=2))],
                        "_throttle_error": True,
                        "enrichment_run_id": run_id
                    }
                else:
                    # Non-throttle error - re-raise (will be caught by outer try)
                    raise

        # Should not reach here, but handle just in case
        raise last_error

    except Exception as e:
        # Unexpected exception - release semaphore to prevent leak
        logger.error(f"[{run_id[:8]}] UNEXPECTED_ERROR node=generate_query error={str(e)[:200]}")
        log_run_summary(run_id)
        release_run_slot(run_id)
        error_response = create_error_response(
            error_code="ENRICHMENT_FAILED",
            message=f"Unexpected error in generate_query: {str(e)[:200]}",
            input_received=canonical_url,
            tenant_id=tenant_id
        )
        return {
            "search_query": [],
            "messages": [AIMessage(content=json.dumps(error_response, indent=2))],
            "_throttle_error": True,  # Reuse flag to route to END
            "enrichment_run_id": run_id
        }


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    Limits fanout to MAX_PARALLEL_SEARCHES for production safety.
    Routes to END if throttle error occurred (error message already in state).
    """
    queries = state.get("search_query", [])
    run_id = state.get("enrichment_run_id", "unknown")

    # Check for throttle error from generate_query - route to END (message already set)
    if state.get("_throttle_error"):
        logger.info(f"[{run_id[:8]}] THROTTLE_EXIT skipping web_research due to upstream throttle")
        return END

    # No queries generated - shouldn't happen, but handle gracefully
    if not queries:
        logger.warning(f"[{run_id[:8]}] NO_QUERIES generate_query returned empty list")
        return END

    # Limit fanout to prevent burst
    queries_to_run = queries[:MAX_PARALLEL_SEARCHES]

    logger.info(f"[{run_id[:8]}] FANOUT web_research queries={len(queries_to_run)} (of {len(queries)} generated)")

    return [
        Send("web_research", {"search_query": search_query, "id": int(idx), "enrichment_run_id": run_id})
        for idx, search_query in enumerate(queries_to_run)
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using the native Google Search API tool.

    Executes a web search using the native Google Search API tool in combination with Gemini 2.0 Flash.
    Uses semaphore to limit concurrent searches and exponential backoff on throttling.

    STRICT FAILURE MODE: Any non-throttle error sets _fatal_error=True and propagates
    ENRICHMENT_FAILED to terminate the graph cleanly.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    configurable = Configuration.from_runnable_config(config)
    run_id = state.get("enrichment_run_id", "unknown")
    model_name = configurable.query_generator_model

    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Check for forced throttle (dev/test only) - not a fatal error
    if should_force_throttle("web_research"):
        logger.info(f"[{run_id[:8]}] FORCED_THROTTLE node=web_research (FORCE_THROTTLE_NODE set)")
        return {
            "sources_gathered": [],
            "search_query": [state["search_query"]],
            "web_research_result": ["[Search throttled - no results]"],
        }

    # Acquire search semaphore to limit concurrent API calls
    with _search_semaphore:
        # Check for forced error (dev/test only) - inside try block so it gets handled
        if should_force_error("web_research"):
            logger.info(f"[{run_id[:8]}] FORCED_ERROR node=web_research (FORCE_ERROR_NODE set)")
            # Non-throttle error - STRICT FAILURE: set _fatal_error flag
            logger.error(f"[{run_id[:8]}] FATAL_ERROR node=web_research error=ForcedError (testing)")
            return {
                "sources_gathered": [],
                "search_query": [state["search_query"]],
                "web_research_result": [],
                "_fatal_error": True,
                "_fatal_error_node": "web_research",
                "_fatal_error_message": "Forced non-throttle error in web_research for testing (FORCE_ERROR_NODE)",
            }

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                log_call_start("web_research", model_name, run_id, attempt)
                start_time = time.time()

                # Uses the google genai client for grounding metadata
                response = genai_client.models.generate_content(
                    model=model_name,
                    contents=formatted_prompt,
                    config={
                        "tools": [{"google_search": {}}],
                        "temperature": 0,
                    },
                )

                duration_ms = int((time.time() - start_time) * 1000)
                log_call_success("web_research", run_id, duration_ms)

                # Process response
                resolved_urls = resolve_urls(
                    response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
                )
                citations = get_citations(response, resolved_urls)
                modified_text = insert_citation_markers(response.text, citations)
                sources_gathered = [item for citation in citations for item in citation["segments"]]

                return {
                    "sources_gathered": sources_gathered,
                    "search_query": [state["search_query"]],
                    "web_research_result": [modified_text],
                }

            except (ClientError, ForcedError, Exception) as e:
                last_error = e
                if is_throttle_error(e) and attempt < MAX_RETRIES:
                    backoff = exponential_backoff_with_jitter(attempt)
                    log_throttle_event("web_research", run_id, attempt, e, backoff)
                    time.sleep(backoff)
                elif is_throttle_error(e):
                    # Final throttle failure - return empty result (not fatal, just no data)
                    log_throttle_event("web_research", run_id, attempt, e, 0)
                    return {
                        "sources_gathered": [],
                        "search_query": [state["search_query"]],
                        "web_research_result": ["[Search throttled - no results]"],
                    }
                else:
                    # Non-throttle error - STRICT FAILURE: set _fatal_error flag
                    logger.error(f"[{run_id[:8]}] FATAL_ERROR node=web_research error={str(e)[:200]}")
                    return {
                        "sources_gathered": [],
                        "search_query": [state["search_query"]],
                        "web_research_result": [],
                        "_fatal_error": True,
                        "_fatal_error_node": "web_research",
                        "_fatal_error_message": str(e)[:200],
                    }

        raise last_error


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output with exponential backoff.

    STRICT FAILURE MODE:
    - If _fatal_error is already set (from upstream node), skip work and propagate
    - Any non-throttle error in this node sets _fatal_error=True

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    run_id = state.get("enrichment_run_id", "unknown")

    # Check if upstream node already set _fatal_error - propagate without doing work
    if state.get("_fatal_error"):
        logger.info(f"[{run_id[:8]}] SKIP_REFLECTION upstream _fatal_error set, propagating")
        return {
            "is_sufficient": True,  # Force route to finalize_answer
            "knowledge_gap": "",
            "follow_up_queries": [],
            "research_loop_count": state.get("research_loop_count", 0),
            "number_of_ran_queries": len(state.get("search_query", [])),
            "_fatal_error": True,
            "_fatal_error_node": state.get("_fatal_error_node", "unknown"),
            "_fatal_error_message": state.get("_fatal_error_message", ""),
        }

    # Increment the research loop count and get the reasoning model
    research_loop_count = state.get("research_loop_count", 0) + 1
    model_name = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state.get("web_research_result", [])),
    )

    # Check for forced error (dev/test only) - handle directly without raising
    if should_force_error("reflection"):
        logger.info(f"[{run_id[:8]}] FORCED_ERROR node=reflection (FORCE_ERROR_NODE set)")
        logger.error(f"[{run_id[:8]}] FATAL_ERROR node=reflection error=ForcedError (testing)")
        return {
            "is_sufficient": True,  # Force route to finalize_answer
            "knowledge_gap": "",
            "follow_up_queries": [],
            "research_loop_count": research_loop_count,
            "number_of_ran_queries": len(state.get("search_query", [])),
            "_fatal_error": True,
            "_fatal_error_node": "reflection",
            "_fatal_error_message": "Forced non-throttle error in reflection for testing (FORCE_ERROR_NODE)",
        }

    # Check for forced throttle (dev/test only)
    if should_force_throttle("reflection"):
        logger.info(f"[{run_id[:8]}] FORCED_THROTTLE node=reflection (FORCE_THROTTLE_NODE set)")
        return {
            "is_sufficient": True,  # Skip further loops
            "knowledge_gap": "Throttled - proceeding with available data",
            "follow_up_queries": [],
            "research_loop_count": research_loop_count,
            "number_of_ran_queries": len(state.get("search_query", [])),
        }

    # Retry loop with exponential backoff
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log_call_start("reflection", model_name, run_id, attempt)
            start_time = time.time()

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=1.0,
                max_retries=0,  # We handle retries with backoff
                api_key=os.getenv("GEMINI_API_KEY"),
            )
            result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

            duration_ms = int((time.time() - start_time) * 1000)
            log_call_success("reflection", run_id, duration_ms)

            return {
                "is_sufficient": result.is_sufficient,
                "knowledge_gap": result.knowledge_gap,
                "follow_up_queries": result.follow_up_queries[:1],  # Limit follow-ups
                "research_loop_count": research_loop_count,
                "number_of_ran_queries": len(state.get("search_query", [])),
            }

        except (ChatGoogleGenerativeAIError, ClientError, ForcedError, Exception) as e:
            last_error = e
            if is_throttle_error(e) and attempt < MAX_RETRIES:
                backoff = exponential_backoff_with_jitter(attempt)
                log_throttle_event("reflection", run_id, attempt, e, backoff)
                time.sleep(backoff)
            elif is_throttle_error(e):
                # Final throttle failure - mark as sufficient to proceed (not fatal)
                log_throttle_event("reflection", run_id, attempt, e, 0)
                return {
                    "is_sufficient": True,  # Skip further loops on throttle
                    "knowledge_gap": "Throttled - proceeding with available data",
                    "follow_up_queries": [],
                    "research_loop_count": research_loop_count,
                    "number_of_ran_queries": len(state.get("search_query", [])),
                }
            else:
                # Non-throttle error - STRICT FAILURE: set _fatal_error flag
                logger.error(f"[{run_id[:8]}] FATAL_ERROR node=reflection error={str(e)[:200]}")
                return {
                    "is_sufficient": True,  # Force route to finalize_answer
                    "knowledge_gap": "",
                    "follow_up_queries": [],
                    "research_loop_count": research_loop_count,
                    "number_of_ran_queries": len(state.get("search_query", [])),
                    "_fatal_error": True,
                    "_fatal_error_node": "reflection",
                    "_fatal_error_message": str(e)[:200],
                }

    raise last_error


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.
    Uses production-safe defaults to limit loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)

    # Use production-safe default for max loops
    max_research_loops = min(
        state.get("max_research_loops") if state.get("max_research_loops") is not None
        else configurable.max_research_loops,
        DEFAULT_MAX_LOOPS
    )

    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        # Limit follow-up queries to prevent burst
        follow_ups = state["follow_up_queries"][:MAX_PARALLEL_SEARCHES]
        logger.info(f"FOLLOW_UP_RESEARCH queries={len(follow_ups)} loop={state['research_loop_count']}")

        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(follow_ups)
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that produces the final CRM-friendly AccountProfile JSON.

    URL-only mode: Always produces structured AccountProfile JSON output.
    Uses the canonical_url and website_domain from state (set by validate_input).
    Includes exponential backoff on throttling.
    Releases the run semaphore on all exit paths.

    STRICT FAILURE MODE: If _fatal_error is set by upstream node (web_research or
    reflection), returns ENRICHMENT_FAILED immediately without calling the LLM.

    Args:
        state: Current graph state containing research results and canonical URL

    Returns:
        Dictionary with state update containing the AccountProfile JSON as AIMessage
    """
    configurable = Configuration.from_runnable_config(config)
    run_id = state.get("enrichment_run_id", "unknown")
    model_name = state.get("reasoning_model") or configurable.answer_model
    current_date = get_current_date()

    # URL-only mode: use canonical URL from state
    canonical_url = state.get("canonical_url", get_research_topic(state["messages"]))
    website_domain = state.get("website_domain", "")
    tenant_id = state.get("tenant_id", "unknown")

    # STRICT FAILURE: Check if upstream node set _fatal_error
    if state.get("_fatal_error"):
        error_node = state.get("_fatal_error_node", "unknown")
        error_message = state.get("_fatal_error_message", "Unknown error")
        logger.error(f"[{run_id[:8]}] FATAL_ERROR_FINALIZE upstream error from {error_node}: {error_message}")
        log_run_summary(run_id)
        release_run_slot(run_id)
        error_response = create_error_response(
            error_code="ENRICHMENT_FAILED",
            message=f"Enrichment failed in {error_node}: {error_message}",
            input_received=canonical_url,
            tenant_id=tenant_id
        )
        return {
            "messages": [AIMessage(content=json.dumps(error_response, indent=2))],
            "sources_gathered": state.get("sources_gathered", []),
        }

    try:
        # Always produce structured AccountProfile output
        base_prompt = account_enrichment_answer_instructions.format(
            current_date=current_date,
            input_url=canonical_url,
            summaries="\n---\n\n".join(state["web_research_result"]),
        )

        # Check for forced throttle (dev/test only)
        if should_force_throttle("finalize_answer"):
            logger.info(f"[{run_id[:8]}] FORCED_THROTTLE node=finalize_answer (FORCE_THROTTLE_NODE set)")
            raise ForcedThrottleError()

        # Retry loop with exponential backoff for throttle errors
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                log_call_start("finalize_answer", model_name, run_id, attempt)
                start_time = time.time()

                # init Reasoning Model (no SDK retries - we handle them)
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=0,
                    max_retries=0,
                    api_key=os.getenv("GEMINI_API_KEY"),
                )
                structured_llm = llm.with_structured_output(AccountProfile)

                # On retry attempts for non-throttle errors, add repair instructions
                if attempt > 1 and last_error and not is_throttle_error(last_error):
                    repair_instruction = f"""

IMPORTANT - Previous attempt failed with error:
{str(last_error)}

Please fix this by:
1. Output valid JSON that strictly conforms to the AccountProfile schema
2. Use null for any optional fields where information is not available
3. Ensure company_name is always provided (required field)
4. Use empty arrays [] for list fields with no data
5. Do not include any text outside the JSON object"""
                    prompt_to_use = base_prompt + repair_instruction
                else:
                    prompt_to_use = base_prompt

                # Get the structured output
                partial_result = structured_llm.invoke(prompt_to_use)

                duration_ms = int((time.time() - start_time) * 1000)
                log_call_success("finalize_answer", run_id, duration_ms)

                # Build sources list from gathered sources (deduplicated, capped at 10)
                # NOTE: Must happen BEFORE releasing semaphore so guard can return error
                seen_urls = set()
                sources = []
                max_sources = 10
                for source in state["sources_gathered"]:
                    url = source.get("value", source.get("short_url", ""))
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    title = source.get("label", "").strip() or "Source"
                    sources.append(Source(url=url, title=title))
                    if len(sources) >= max_sources:
                        break

                # GUARD: Force zero sources for testing (dev/test only)
                if FORCE_NO_SOURCES:
                    logger.info(f"[{run_id[:8]}] FORCED_NO_SOURCES clearing sources list (FORCE_NO_SOURCES set)")
                    sources = []

                # GUARD: Zero sources = no evidence gathered = cannot return success
                # This likely indicates upstream throttling or search failures
                if len(sources) == 0:
                    logger.warning(f"[{run_id[:8]}] THROTTLED_NO_SOURCES web research returned no valid sources")
                    log_run_summary(run_id)
                    release_run_slot(run_id)
                    error_response = create_throttle_error_response(
                        input_received=canonical_url,
                        tenant_id=tenant_id,
                        retry_after_seconds=exponential_backoff_with_jitter(1)
                    )
                    error_response["message"] = "Web research returned no sources; likely upstream throttling. Please retry."
                    return {
                        "messages": [AIMessage(content=json.dumps(error_response, indent=2))],
                        "sources_gathered": state["sources_gathered"],
                    }

                # Log run summary and release semaphore
                log_run_summary(run_id)
                release_run_slot(run_id)

                # Use website_domain from state (already computed by validate_input)
                final_website_domain = website_domain or partial_result.website_domain or ""

                # Build crm_fields deterministically from core fields (fixed order)
                crm_fields = [
                    CRMField(key="Company Name", value=partial_result.company_name),
                    CRMField(key="Website", value=final_website_domain),
                    CRMField(key="Industry", value=partial_result.industry),
                    CRMField(key="Headquarters", value=partial_result.headquarters),
                    CRMField(key="Founded", value=str(partial_result.founded_year) if partial_result.founded_year else None),
                    CRMField(key="Employees", value=partial_result.employee_count_range),
                    CRMField(key="Company Type", value=partial_result.company_type),
                ]

                # Create the full AccountProfile with metadata
                account_profile = AccountProfile(
                    tenant_id=tenant_id,
                    input_url=canonical_url,
                    company_name=partial_result.company_name,
                    website_domain=final_website_domain,
                    headquarters=partial_result.headquarters,
                    industry=partial_result.industry,
                    founded_year=partial_result.founded_year,
                    employee_count_range=partial_result.employee_count_range,
                    company_type=partial_result.company_type,
                    primary_products=partial_result.primary_products or [],
                    customer_segments=partial_result.customer_segments or [],
                    data_delivery_channels=partial_result.data_delivery_channels or [],
                    one_line_description=partial_result.one_line_description,
                    crm_summary=partial_result.crm_summary,
                    differentiators=partial_result.differentiators or [],
                    recent_notable_updates=partial_result.recent_notable_updates or [],
                    crm_fields=crm_fields,
                    sources=sources,
                    enriched_at=datetime.now(timezone.utc).isoformat(),
                    schema_version="2.0.0",
                )

                return {
                    "messages": [
                        AIMessage(content=account_profile.model_dump_json(indent=2))
                    ],
                    "sources_gathered": state["sources_gathered"],
                }

            except (ChatGoogleGenerativeAIError, ClientError, ForcedThrottleError, Exception) as e:
                last_error = e
                if is_throttle_error(e) and attempt < MAX_RETRIES:
                    backoff = exponential_backoff_with_jitter(attempt)
                    log_throttle_event("finalize_answer", run_id, attempt, e, backoff)
                    time.sleep(backoff)
                elif is_throttle_error(e):
                    # Final throttle failure - release semaphore
                    log_throttle_event("finalize_answer", run_id, attempt, e, 0)
                    log_run_summary(run_id)
                    release_run_slot(run_id)
                    error_response = create_throttle_error_response(
                        input_received=canonical_url,
                        tenant_id=tenant_id,
                        retry_after_seconds=exponential_backoff_with_jitter(attempt)
                    )
                    return {
                        "messages": [AIMessage(content=json.dumps(error_response, indent=2))],
                        "sources_gathered": state["sources_gathered"],
                    }
                elif attempt < MAX_RETRIES:
                    # Non-throttle error - retry with repair prompt
                    continue
                else:
                    # All retries failed - release semaphore
                    log_run_summary(run_id)
                    release_run_slot(run_id)
                    error_response = create_error_response(
                        error_code="ENRICHMENT_FAILED",
                        message=f"Failed to generate structured output after {MAX_RETRIES} attempts: {str(last_error)}",
                        input_received=canonical_url,
                        tenant_id=tenant_id
                    )
                    return {
                        "messages": [AIMessage(content=json.dumps(error_response, indent=2))],
                        "sources_gathered": state["sources_gathered"],
                    }

        # Should not reach here
        raise last_error

    except Exception as e:
        # Unexpected exception not caught by inner try - release semaphore
        logger.error(f"[{run_id[:8]}] UNEXPECTED_ERROR node=finalize_answer error={str(e)[:200]}")
        log_run_summary(run_id)
        release_run_slot(run_id)
        error_response = create_error_response(
            error_code="ENRICHMENT_FAILED",
            message=f"Unexpected error in finalize_answer: {str(e)[:200]}",
            input_received=canonical_url,
            tenant_id=tenant_id
        )
        return {
            "messages": [AIMessage(content=json.dumps(error_response, indent=2))],
            "sources_gathered": state.get("sources_gathered", []),
        }


# =============================================================================
# Graph Builder (URL-only mode)
# =============================================================================

builder = StateGraph(OverallState, config_schema=Configuration)

# Define all nodes
builder.add_node("validate_input", validate_input)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Start with input validation
builder.add_edge(START, "validate_input")

# Route based on validation: continue or end early with error
builder.add_conditional_edges(
    "validate_input",
    route_after_validation,
    ["generate_query", END]
)

# Generate queries then fan out to parallel web research (or END on throttle error)
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research", END]
)

# Reflect on the web research
builder.add_edge("web_research", "reflection")

# Evaluate research: continue searching or finalize
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)

# Finalize produces the AccountProfile JSON
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="url-enrichment-agent")
