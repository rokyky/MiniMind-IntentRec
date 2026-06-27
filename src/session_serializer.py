"""
session_serializer.py - Serialize session data to human-readable text.

Formats a session into:
  "User recently viewed/bought: [title1] (category1) at [time1],
   [title2] (category2) at [time2], ..."

Supports configurable max items, timestamp formatting, and optional user ids.
"""

import datetime
from typing import List, Dict, Optional


def format_timestamp(ts: int) -> str:
    """Convert unix timestamp to human-readable date string."""
    if ts <= 0:
        return "unknown time"
    try:
        dt = datetime.datetime.fromtimestamp(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (OSError, ValueError, OverflowError):
        return f"timestamp({ts})"


def serialize_session(
    titles: List[str],
    categories: List[str],
    timestamps: Optional[List[int]] = None,
    max_items: int = 10,
    include_user: bool = False,
    user_id: Optional[str] = None,
    prefix: str = "User recently viewed/bought",
) -> str:
    """
    Serialize a session to a text string.

    Args:
        titles: List of item titles in chronological order.
        categories: List of item categories (same length as titles).
        timestamps: Optional list of unix timestamps (same length).
        max_items: Max number of items to include in the text.
        include_user: If True, prepend user id info.
        user_id: User identifier (used when include_user is True).
        prefix: Beginning phrase for the session text.

    Returns:
        Human-readable session text string.
    """
    if not titles or not categories:
        return ""

    # Truncate to max_items
    titles = titles[:max_items]
    categories = categories[:max_items]
    if timestamps:
        timestamps = timestamps[:max_items]
    else:
        timestamps = [0] * len(titles)

    parts = []
    for i, (title, category, ts) in enumerate(zip(titles, categories, timestamps)):
        item_text = f"{title} ({category})"
        ts_str = format_timestamp(ts)
        item_text += f" at [{ts_str}]"
        parts.append(item_text)

    if include_user and user_id:
        prefix = f"User [{user_id}] recently viewed/bought"

    return prefix + ": " + ", ".join(parts) + "."


def serialize_session_with_target(
    titles: List[str],
    categories: List[str],
    timestamps: Optional[List[int]] = None,
    target_title: Optional[str] = None,
    target_category: Optional[str] = None,
    max_items: int = 10,
) -> str:
    """
    Serialize session and target item for teacher labeling prompt.

    Returns (session_text, prompt_text) where prompt_text is the full
    text asking the teacher to predict the intent of this user.
    """
    session_text = serialize_session(
        titles, categories, timestamps, max_items=max_items
    )
    target_text = ""
    if target_title:
        target_text = f" Target item: {target_title}"
        if target_category:
            target_text += f" ({target_category})"
    return session_text, target_text
