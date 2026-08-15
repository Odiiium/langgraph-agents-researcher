import os
from typing import Literal
from .models import SearchResult
from .logging_config import get_logger
from langchain_core.tools import tool
from sympy import sympify
from tavily.tavily import TavilyClient
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

env = load_dotenv()
tavily_client = TavilyClient()

@tool
def calculator_tool(expression: str):
    """
    Calculate a mathematical expression.

    Use this tool for arithmetic calculations, percentages,
    powers, roots, trigonometric functions and other mathematical expressions.

    Examples:
    - "2 + 2"
    - "15 * 0.2"
    - "(100 - 20) / 4"
    - "2 ** 10"
    - "sqrt(144)"
    """
    if not expression or not expression.strip():
        return "ERROR: empty expression"
    get_logger().info("tool=calculator expression=%r", expression)
    try:
        result = str(sympify(expression).evalf())
        get_logger().info("tool=calculator observation=%s", result)
        return result
    except Exception as e:
        get_logger().warning("tool=calculator error=%s", e)
        return f"ERROR: could not calculate expression: {e}"

@tool
def search_tool(query : str,
                max_results : int = 5,
                topic : Literal["general", "news", "finance"] = "general",
                search_depth : Literal["basic", "advanced", "fast", "ultra-fast"] = "basic"):
    """
    Search the web for information relevant to the research task.

    Use this tool when external or up-to-date information is required.

    Args:
        query:
            Precise search query.

        max_results:
            Maximum number of results to return. Maximum is 10.

        topic:
            Search category.
            Use "news" for recent events and current news.
            Use "finance" for financial and market information.
            Use "general" for all other research.

        search_depth:
            Search depth.
            Use "basic" for simple searches.
            Use "advanced" for complex or comprehensive research.
            Use deeper modes when additional search quality is required.

    Returns:
        Search results containing title, URL and extracted content.
    """
    
    if not query or not query.strip():
        return "ERROR: empty search query"

    max_results = max(1, min(max_results, 10))
    get_logger().info("tool=search_tool query=%r topic=%s", query, topic)
    
    try:
        response = tavily_client.search(query=query,
                                        search_depth=search_depth,
                                        topic=topic,
                                        max_results=max_results)
        
        results = response.get("results", [])
        get_logger().info("tool=search_tool observation=%d results", len(results))
        
        if not results: 
            return []
        
        return [SearchResult(title=result.get("title", ""),
                                url = result.get("url", ""),
                                content = result.get("content", ""),
                                published_date=result.get("published_date")) for result in results]

    except Exception as e:
        get_logger().warning("tool=search_tool error=%s", e)
        return f"ERROR: search failed for query {query}: {e}"

@tool
def get_current_datetime(timezone: str = "UTC") -> str:
    """
    Get the current date and time for a specific IANA timezone.

    Use this tool when the research requires knowing the current date or time,
    especially when evaluating recent events, news, deadlines, schedules,
    or time-sensitive information.

    Args:
        timezone:
            IANA timezone name, for example:
            - "UTC"
            - "Europe/Kyiv"
            - "Europe/London"
            - "America/New_York"
            - "Asia/Tokyo"

    Returns:
        Current date and time in ISO 8601 format.
    """
    try:
        now = datetime.now(ZoneInfo(timezone))
        return now.isoformat()
    except Exception as e:
        return f"Could not get current time for timezone '{timezone}': {e}"