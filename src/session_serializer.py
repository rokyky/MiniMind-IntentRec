"""
session_serializer.py - 将会话数据序列化为人类可读文本。

将会话格式化为：
  "User recently viewed/bought: [title1] (category1) at [time1],
   [title2] (category2) at [time2], ..."

支持可配置的最大物品数、时间戳格式和可选的用户 ID。
"""

import datetime
from typing import List, Dict, Optional


def format_timestamp(ts: int) -> str:
    """将 Unix 时间戳转换为人类可读的日期字符串。"""
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
    将会话序列化为文本字符串。

    Args:
        titles: 按时间顺序排列的物品标题列表。
        categories: 物品类别列表（长度与 titles 相同）。
        timestamps: 可选的 Unix 时间戳列表（长度相同）。
        max_items: 文本中包含的最大物品数。
        include_user: 如果为 True，则前置用户 ID 信息。
        user_id: 用户标识符（当 include_user 为 True 时使用）。
        prefix: 会话文本的开头短语。

    Returns:
        人类可读的会话文本字符串。
    """
    if not titles or not categories:
        return ""

    # 截断至 max_items
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
    为教师标注提示序列化会话和目标物品。

    返回 (session_text, prompt_text)，其中 prompt_text 是完整的
    要求教师预测该用户意图的文本。
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
