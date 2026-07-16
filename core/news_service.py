from __future__ import annotations

from urllib.parse import quote_plus

import feedparser

CATEGORY_QUERIES = {
    "Malaysia Headlines": "Malaysia latest news",
    "Politics": "Malaysia politics",
    "World": "world breaking news",
    "Business": "Malaysia business economy",
    "Technology": "technology artificial intelligence",
    "Health": "Malaysia health news",
    "Weather & Disasters": "Malaysia weather flood disaster",
}


def get_news(category: str = "Malaysia Headlines", limit: int = 12) -> list[dict]:
    query = CATEGORY_QUERIES.get(category, category)
    url = (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=en-MY&gl=MY&ceid=MY:en"
    )
    feed = feedparser.parse(url)
    items: list[dict] = []
    for entry in feed.entries[:limit]:
        source = ""
        if hasattr(entry, "source"):
            source = getattr(entry.source, "title", "")
        items.append(
            {
                "title": getattr(entry, "title", "Untitled"),
                "link": getattr(entry, "link", ""),
                "published": getattr(entry, "published", ""),
                "source": source,
                "summary": getattr(entry, "summary", ""),
            }
        )
    return items
