"""
teacher_label_cache.py - 离线教师标签缓存。

基于 JSON 的缓存格式：
  {session_id: {intent_json, model_version, created_at}}

支持加载/保存、按 session_id 查找、强制刷新标志和
分类体系变更的版本追踪。
"""

import json
import os
import datetime
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class TeacherLabelCache:
    """
    教师生成的意图标签的缓存。

    将标签存储在 JSON 文件中，支持按 session_id 查找。
    通过 model_version 和 taxonomy_version 跟踪缓存失效。
    """

    def __init__(
        self,
        cache_path: str,
        model_version: str = "1.0",
        taxonomy_version: str = "1.0",
    ):
        """
        初始化缓存。

        Args:
            cache_path: JSON 缓存文件的路径。
            model_version: 教师模型的版本标识符。
            taxonomy_version: 分类体系模式的版本标识符。
        """
        self.cache_path = cache_path
        self.model_version = model_version
        self.taxonomy_version = taxonomy_version
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._dirty: bool = False
        self._load()

    def _load(self):
        """从磁盘加载缓存。"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._cache = data.get("labels", {})
                logger.info(
                    f"Loaded cache with {len(self._cache)} entries from "
                    f"{self.cache_path}"
                )
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache from {self.cache_path}: {e}")
                self._cache = {}
        else:
            logger.info(f"No existing cache at {self.cache_path}, starting fresh.")
            self._cache = {}

    def save(self):
        """如果缓存有变更则保存到磁盘。"""
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        data = {
            "metadata": {
                "model_version": self.model_version,
                "taxonomy_version": self.taxonomy_version,
                "updated_at": datetime.datetime.now().isoformat(),
                "num_entries": len(self._cache),
            },
            "labels": self._cache,
        }
        tmp_path = self.cache_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.cache_path)
        self._dirty = False
        logger.info(f"Saved {len(self._cache)} cache entries to {self.cache_path}")

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        按 session_id 查找缓存的标签。

        如果找到且有效则返回 intent_json 字典，否则返回 None。
        如果缓存条目的版本与当前版本不匹配，也返回 None。
        """
        entry = self._cache.get(session_id)
        if entry is None:
            return None

        # 版本检查：如果模型或分类体系版本更改则失效
        if entry.get("model_version") != self.model_version:
            logger.debug(
                f"Cache entry for {session_id} has stale model_version "
                f"({entry.get('model_version')} vs {self.model_version})"
            )
            return None
        if entry.get("taxonomy_version") != self.taxonomy_version:
            logger.debug(
                f"Cache entry for {session_id} has stale taxonomy_version "
                f"({entry.get('taxonomy_version')} vs {self.taxonomy_version})"
            )
            return None

        return entry.get("intent_json")

    def set(
        self,
        session_id: str,
        intent_json: Dict[str, Any],
    ):
        """
        为一个 session_id 设置缓存标签。

        Args:
            session_id: 唯一的会话标识符。
            intent_json: 符合分类体系模式的意图标签字典。
        """
        self._cache[session_id] = {
            "intent_json": intent_json,
            "model_version": self.model_version,
            "taxonomy_version": self.taxonomy_version,
            "created_at": datetime.datetime.now().isoformat(),
        }
        self._dirty = True

    def has(self, session_id: str) -> bool:
        """检查 session_id 是否在缓存中且版本有效。"""
        return self.get(session_id) is not None

    def contains(self, session_id: str) -> bool:
        """检查 session_id 是否在缓存中（无论版本如何）。"""
        return session_id in self._cache

    def force_refresh(self, session_id: str) -> bool:
        """
        移除缓存的条目，以强制下次访问时重新标注。

        如果条目被移除返回 True，如果未找到返回 False。
        """
        if session_id in self._cache:
            del self._cache[session_id]
            self._dirty = True
            return True
        return False

    def force_refresh_all(self):
        """清除所有缓存的标签。"""
        self._cache.clear()
        self._dirty = True
        logger.info("Cleared all cached labels.")

    def get_all_labels(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有缓存的标签（仅 intent_json），以 session_id 为键。

        仅返回版本匹配的条目。
        """
        result = {}
        for sid, entry in self._cache.items():
            if entry.get("model_version") == self.model_version and \
               entry.get("taxonomy_version") == self.taxonomy_version:
                result[sid] = entry.get("intent_json", {})
        return result

    def get_stats(self) -> Dict[str, int]:
        """返回缓存统计信息。"""
        total = len(self._cache)
        valid = sum(
            1 for e in self._cache.values()
            if e.get("model_version") == self.model_version
            and e.get("taxonomy_version") == self.taxonomy_version
        )
        stale = total - valid
        return {
            "total_entries": total,
            "valid_entries": valid,
            "stale_entries": stale,
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, session_id: str) -> bool:
        return self.contains(session_id)
