"""Format research synthesis into conversational answers with risk assessment.

Converts raw research_and_summarize output into structured, trust-calibrated responses.
"""

from typing import Dict, Any, List
from collections import Counter


def _assess_risk_level(contested_claims: List[Dict], stats: Dict[str, int]) -> str:
    """
    Assess answer riskiness based on contested claims and source quality.

    Returns:
        "low": No contested claims, can answer assertively
        "medium": Some contested claims but main narrative stable
        "high": Central contested claims, early story, or thin sourcing
    """
    contested_count = len(contested_claims)
    sources = stats.get("sources_investigated", 0)

    # High risk: Thinly sourced or many contested claims
    if sources < 3:
        return "high"
    if contested_count > 3:
        return "high"

    # Check if contested claims are central (mentioned in narrative)
    # For now, treat any contestation as medium risk
    if contested_count > 0:
        return "medium"

    # Low risk: Multiple sources, no contestation
    return "low"


def _cluster_domains(docs: List[Dict]) -> Dict[str, List[str]]:
    """
    Cluster source URLs by domain type for provenance descriptions.

    Returns:
        Dict mapping domain type to list of domains
        Example: {"news_wire": ["apnews.com", "reuters.com"], "local": [...]}
    """
    # Simple heuristic: parse domains and categorize
    wire_services = {"apnews.com", "reuters.com", "ap.org", "upi.com"}
    major_news = {"nytimes.com", "washingtonpost.com", "wsj.com", "theguardian.com", "bbc.com"}
    academic = {"arxiv.org", "nature.com", "science.org", "pubmed.ncbi.nlm.nih.gov"}
    court = {"courtlistener.com", "supremecourt.gov", "pacer.gov"}

    clusters = {
        "wire": [],
        "major_news": [],
        "academic": [],
        "court": [],
        "other": []
    }

    for doc in docs:
        url = doc.get("url", "")
        if not url:
            continue

        try:
            domain = url.split("/")[2]
        except IndexError:
            continue

        if domain in wire_services:
            clusters["wire"].append(domain)
        elif domain in major_news:
            clusters["major_news"].append(domain)
        elif domain in academic:
            clusters["academic"].append(domain)
        elif domain in court:
            clusters["court"].append(domain)
        else:
            clusters["other"].append(domain)

    return clusters


def _format_provenance_summary(domain_clusters: Dict[str, List[str]]) -> str:
    """
    Format domain clusters into natural language provenance description.

    Returns:
        Human-readable string like "major wire services, court documents, and academic sources"
    """
    labels = []

    if domain_clusters.get("wire"):
        labels.append("major wire services")
    if domain_clusters.get("major_news"):
        labels.append("major news outlets")
    if domain_clusters.get("academic"):
        labels.append("academic sources")
    if domain_clusters.get("court"):
        labels.append("court documents")
    if domain_clusters.get("other"):
        count = len(set(domain_clusters["other"]))
        labels.append(f"{count} independent source{'s' if count > 1 else ''}")

    if not labels:
        return "various sources"

    if len(labels) == 1:
        return labels[0]
    elif len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    else:
        return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def format_research_answer(
    summary_obj: Dict[str, Any],
    source_docs: List[Dict] = None,
    include_open_questions: bool = True,
    verbose: bool = False
) -> str:
    """
    Format research synthesis into conversational answer with trust calibration.

    Args:
        summary_obj: Output from research_and_summarize()
        source_docs: Optional source documents for provenance clustering
        include_open_questions: Whether to include "What's still unclear" section
        verbose: If True, include full details (for debugging)

    Returns:
        Formatted conversational answer string

    Structure:
        1. Risk-calibrated opening (2-4 sentences)
        2. Key events (bullet list)
        3. Points of disagreement (if contested_claims exist)
        4. What's still unclear (if include_open_questions and open_questions exist)
        5. Source note (provenance summary)
    """
    narrative = summary_obj.get("narrative_summary", "")
    key_events = summary_obj.get("key_events", [])
    contested_claims = summary_obj.get("contested_claims", [])
    open_questions = summary_obj.get("open_questions", [])
    stats = summary_obj.get("coverage_stats", {})

    # Assess risk level
    risk = _assess_risk_level(contested_claims, stats)

    # Build answer
    answer = ""

    # 1. Risk-calibrated opening
    if risk == "high":
        answer += "**Note**: This is an early or contested story with limited sourcing. Here's what's known so far:\n\n"
    elif risk == "medium":
        answer += "**Note**: Some details of this story are disputed. Here's the main narrative:\n\n"

    # Add narrative summary
    if narrative:
        answer += f"{narrative}\n\n"

    # 2. Key events
    if key_events:
        # Limit to top 5 for readability
        events_to_show = key_events[:5]
        answer += "**Key Events**:\n"
        for event in events_to_show:
            answer += f"• {event}\n"
        answer += "\n"

    # 3. Points of disagreement
    if contested_claims:
        answer += "**Points of Disagreement**:\n"
        for claim in contested_claims[:5]:  # Limit to 5
            claim_text = claim.get("claim", "")
            reason = claim.get("reason", "")
            answer += f"• {claim_text}\n"
            if reason:
                answer += f"  *Why disputed*: {reason}\n"
        answer += "\n"

    # 4. What's still unclear
    if include_open_questions and open_questions:
        answer += "**What's Still Unclear**:\n"
        for q in open_questions[:5]:  # Limit to 5
            answer += f"• {q}\n"
        answer += "\n"

    # 5. Source note
    if source_docs:
        domain_clusters = _cluster_domains(source_docs)
        provenance = _format_provenance_summary(domain_clusters)
        source_count = stats.get("sources_investigated", len(source_docs))
        answer += f"*Based on {source_count} source{'s' if source_count != 1 else ''} including {provenance}*\n"
    else:
        source_count = stats.get("sources_investigated", 0)
        if source_count > 0:
            answer += f"*Based on {source_count} source{'s' if source_count != 1 else ''}*\n"

    # Verbose mode: add stats
    if verbose:
        answer += f"\n---\n"
        answer += f"**Research Stats**:\n"
        answer += f"• Sources: {stats.get('sources_investigated', 0)}\n"
        answer += f"• Claims: {stats.get('claims_extracted', 0)}\n"
        answer += f"• Tasks: {stats.get('tasks_executed', 0)}\n"
        answer += f"• Depth: {stats.get('depth_reached', 0)}\n"
        answer += f"• Risk Level: {risk}\n"

    return answer.strip()


def get_risk_hedging_guidance(risk_level: str) -> str:
    """
    Get conversational guidance for how to speak given risk level.

    Args:
        risk_level: "low", "medium", or "high"

    Returns:
        Natural language guidance for tone/hedging
    """
    if risk_level == "low":
        return "You can answer assertively. Multiple independent sources agree on the main facts."

    elif risk_level == "medium":
        return "Answer with light hedging. The main narrative is stable, but explicitly mention there is dispute on some details. Use phrases like 'according to most sources' or 'the majority view is'."

    else:  # high
        return "Present this as 'emerging' or 'developing'. Clearly label speculation vs fact. Use heavy hedging: 'early reports suggest', 'some sources claim', 'it's unclear whether'. Lean on 'here is what is known so far vs what is claimed'."
