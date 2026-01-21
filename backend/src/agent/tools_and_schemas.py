from typing import List, Optional
from pydantic import BaseModel, Field


class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )


class Reflection(BaseModel):
    is_sufficient: bool = Field(
        description="Whether the provided summaries are sufficient to answer the user's question."
    )
    knowledge_gap: str = Field(
        description="A description of what information is missing or needs clarification."
    )
    follow_up_queries: List[str] = Field(
        description="A list of follow-up queries to address the knowledge gap."
    )


class Source(BaseModel):
    """A source URL with its title, used for citation tracking."""

    url: str = Field(description="The URL of the source")
    title: str = Field(description="The title or label of the source")


class CRMField(BaseModel):
    """A key-value pair for CRM display."""

    key: str = Field(description="The field name/label")
    value: Optional[str] = Field(description="The field value (null if unknown)")


class AccountProfile(BaseModel):
    """CRM-friendly structured output schema for company/account enrichment from URL.

    Designed for CRM integration with uniform core fields and optional long-text fields.
    """

    # === Required Identifiers ===
    tenant_id: str = Field(description="The tenant ID for multi-tenant tracking")
    input_url: str = Field(description="The original URL that was enriched")

    # === Core CRM Fields (uniform across accounts) ===
    company_name: str = Field(
        description="The official name of the company (required; best-effort if not found)"
    )
    website_domain: str = Field(
        description="The normalized root domain (e.g., 'stripe.com')"
    )
    headquarters: Optional[str] = Field(
        default=None, description="City, State/Country of headquarters"
    )
    industry: Optional[str] = Field(
        default=None, description="Primary industry classification"
    )
    founded_year: Optional[int] = Field(
        default=None, description="Year the company was founded"
    )
    employee_count_range: Optional[str] = Field(
        default=None,
        description="Standard bucket: '1-10', '11-50', '51-200', '201-500', '501-1000', '1001-5000', '5001-10000', '10000+'"
    )
    company_type: Optional[str] = Field(
        default=None,
        description="One of: 'public', 'private', 'subsidiary', 'nonprofit', 'government', or null if unknown"
    )
    primary_products: List[str] = Field(
        default_factory=list,
        description="Main products or services offered (short names, max 5)"
    )
    customer_segments: List[str] = Field(
        default_factory=list,
        description="Target customer segments (e.g., 'Enterprise', 'SMB', 'Consumer')"
    )
    data_delivery_channels: List[str] = Field(
        default_factory=list,
        description="How the company delivers data/services (e.g., 'API', 'SaaS', 'On-premise')"
    )

    # === Long Text Fields (optional) ===
    one_line_description: Optional[str] = Field(
        default=None,
        description="One sentence description (max 20 words)"
    )
    crm_summary: Optional[str] = Field(
        default=None,
        description="2-4 sentence summary suitable for CRM notes"
    )
    differentiators: List[str] = Field(
        default_factory=list,
        description="Key competitive differentiators (short bullets, max 5)"
    )
    recent_notable_updates: List[str] = Field(
        default_factory=list,
        description="Recent news, funding, or product updates (max 3)"
    )

    # === CRM Display Format ===
    crm_fields: List[CRMField] = Field(
        default_factory=list,
        description="Ordered list of core fields for CRM UI display"
    )

    # === Evidence ===
    sources: List[Source] = Field(
        default_factory=list,
        description="Deduplicated list of sources (max 10)"
    )

    # === Metadata ===
    enriched_at: str = Field(
        description="ISO 8601 timestamp of when enrichment occurred"
    )
    schema_version: str = Field(
        default="2.0.0",
        description="Schema version for backwards compatibility"
    )
