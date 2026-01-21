from datetime import datetime


# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


query_writer_instructions = """Your goal is to generate sophisticated and diverse web search queries. These queries are intended for an advanced automated web research tool capable of analyzing complex results, following links, and synthesizing information.

Instructions:
- Always prefer a single search query, only add another query if the original question requests multiple aspects or elements and one query is not enough.
- Each query should focus on one specific aspect of the original question.
- Don't produce more than {number_queries} queries.
- Queries should be diverse, if the topic is broad, generate more than 1 query.
- Don't generate multiple similar queries, 1 is enough.
- Query should ensure that the most current information is gathered. The current date is {current_date}.

Format: 
- Format your response as a JSON object with ALL two of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries

Example:

Topic: What revenue grew more last year apple stock or the number of people buying an iphone
```json
{{
    "rationale": "To answer this comparative growth question accurately, we need specific data points on Apple's stock performance and iPhone sales metrics. These queries target the precise financial information needed: company revenue trends, product-specific unit sales figures, and stock price movement over the same fiscal period for direct comparison.",
    "query": ["Apple total revenue growth fiscal year 2024", "iPhone unit sales growth fiscal year 2024", "Apple stock price growth fiscal year 2024"],
}}
```

Context: {research_topic}"""


web_searcher_instructions = """Conduct targeted Google Searches to gather the most recent, credible information on "{research_topic}" and synthesize it into a verifiable text artifact.

Instructions:
- Query should ensure that the most current information is gathered. The current date is {current_date}.
- Conduct multiple, diverse searches to gather comprehensive information.
- Consolidate key findings while meticulously tracking the source(s) for each specific piece of information.
- The output should be a well-written summary or report based on your search findings. 
- Only include the information found in the search results, don't make up any information.

Research Topic:
{research_topic}
"""

reflection_instructions = """You are an expert research assistant analyzing summaries about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration and generate a follow-up query. (1 or multiple).
- If provided summaries are sufficient to answer the user's question, don't generate a follow-up query.
- If there is a knowledge gap, generate a follow-up query that would help expand your understanding.
- Focus on technical details, implementation specifics, or emerging trends that weren't fully covered.

Requirements:
- Ensure the follow-up query is self-contained and includes necessary context for web search.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification
   - "follow_up_queries": Write a specific question to address this gap

Example:
```json
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
    "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"] // [] if is_sufficient is true
}}
```

Reflect carefully on the Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:

Summaries:
{summaries}
"""

answer_instructions = """Generate a high-quality answer to the user's question based on the provided summaries.

Instructions:
- The current date is {current_date}.
- You are the final step of a multi-step research process, don't mention that you are the final step.
- You have access to all the information gathered from the previous steps.
- You have access to the user's question.
- Generate a high-quality answer to the user's question based on the provided summaries and the user's question.
- Include the sources you used from the Summaries in the answer correctly, use markdown format (e.g. [apnews](https://vertexaisearch.cloud.google.com/id/1-0)). THIS IS A MUST.

User Context:
- {research_topic}

Summaries:
{summaries}"""


# Account Enrichment Prompts (for URL input mode)

account_enrichment_query_instructions = """Your goal is to generate search queries to gather comprehensive information about a company from its URL.

Instructions:
- The user has provided a company URL: {company_url}
- Generate search queries that will help gather the following information:
  1. Company name and official description
  2. Industry and business model
  3. Headquarters location
  4. Year founded
  5. Employee count or company size
  6. Key products or services
- Focus on authoritative sources like the company's own website, LinkedIn, Crunchbase, Wikipedia, and news articles.
- Generate {number_queries} targeted queries.
- The current date is {current_date}.

Format:
- Format your response as a JSON object with ALL two of these exact keys:
   - "rationale": Brief explanation of why these queries are relevant
   - "query": A list of search queries

Example:
```json
{{
    "rationale": "To build a comprehensive company profile, we need to search for official company information, business details, and recent news coverage.",
    "query": ["Stripe company official website about", "Stripe headquarters location founded year", "Stripe employee count company size"],
}}
```

Company URL: {company_url}"""


account_enrichment_answer_instructions = """Extract CRM-ready structured company profile data from research summaries.

CRITICAL: Output must be CRM-friendly - short, structured fields for database storage. NO long narratives.

Company URL: {input_url}
Current date: {current_date}

=== REQUIRED FIELDS ===
- company_name: Official company name (REQUIRED - use best effort from sources)

=== CORE CRM FIELDS (use null if unknown) ===
- headquarters: "City, State" or "City, Country" format only
- industry: Single primary industry classification
- founded_year: Integer year only (e.g., 2010)
- employee_count_range: MUST be one of: "1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000", "5001-10000", "10000+"
- company_type: One of: "public", "private", "subsidiary", "nonprofit", "government", or null
- primary_products: Array of product/service names (max 5 items, short names only)
- customer_segments: Array like ["Enterprise", "SMB", "Consumer", "Government"] (max 4)
- data_delivery_channels: Array like ["API", "SaaS", "Platform", "On-premise"] (max 4)

=== LONG TEXT FIELDS (strict length limits) ===
- one_line_description: ONE sentence, max 20 words. Example: "Stripe provides payment processing infrastructure for internet businesses."
- crm_summary: 2-4 sentences MAX. No fluff. Just facts about what the company does, who they serve, and key differentiators.
- differentiators: Array of short bullets (max 5), each under 10 words
- recent_notable_updates: Array of recent news/updates (max 3), each under 15 words

=== STRICT RULES ===
1. If information is not in the summaries, use null or empty array []
2. Do NOT write essays or long descriptions
3. Do NOT start with "As of [date]..." or similar preambles
4. Do NOT include marketing fluff or vague statements
5. Arrays should contain SHORT strings, not sentences

Summaries:
{summaries}"""
