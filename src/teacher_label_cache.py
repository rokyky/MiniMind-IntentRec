"""
teacher_label_cache.py - Offline teacher label cache.

JSON-based cache format:
  {session_id: {intent_json, model_version, created_at}}

Supports load/save, lookup by session_id, force-refresh flag, and
version tracking for taxonomy changes.
"""

import json
import os
import datetime
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class TeacherLabelCache:
    """
    Cache for teacher-generated intent labels.

    Stores labels in a JSON file and supports lookup by session_id.
    Tracks model_version and taxonomy_version for cache invalidation.
    """

    def __init__(
        self,
        cache_path: str,
        model_version: str = "1.0",
        taxonomy_version: str = "1.0",
    ):
        """
        Initialize the cache.

        Args:
            cache_path: Path to the JSON cache file.
            model_version: Version identifier of the teacher model.
            taxonomy_version: Version identifier of the taxonomy schema.
        """
        self.cache_path = cache_path
        self.model_version = model_version
        self.taxonomy_version = taxonomy_version
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._dirty: bool = False
        self._load()

    def _load(self):
        """Load cache from disk."""
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
        """Save cache to disk if dirty."""
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
        Lookup a cached label by session_id.

        Returns the intent_json dict if found and valid, None otherwise.
        Also returns None if the cache entry's version doesn't match current.
        """
        entry = self._cache.get(session_id)
        if entry is None:
            return None

        # Version check: invalidate if model or taxonomy version changed
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
        Set a cached label for a session_id.

        Args:
            session_id: Unique session identifier.
            intent_json: The intent label dict conforming to the taxonomy schema.
        """
        self._cache[session_id] = {
            "intent_json": intent_json,
            "model_version": self.model_version,
            "taxonomy_version": self.taxonomy_version,
            "created_at": datetime.datetime.now().isoformat(),
        }
        self._dirty = True

    def has(self, session_id: str) -> bool:
        """Check if a session_id exists in cache with valid version."""
        return self.get(session_id) is not None

    def contains(self, session_id: str) -> bool:
        """Check if session_id exists in cache regardless of version."""
        return session_id in self._cache

    def force_refresh(self, session_id: str) -> bool:
        """
        Remove a cached entry to force re-labeling on next access.

        Returns True if an entry was removed, False if not found.
        """
        if session_id in self._cache:
            del self._cache[session_id]
            self._dirty = True
            return True
        return False

    def force_refresh_all(self):
        """Clear all cached labels."""
        self._cache.clear()
        self._dirty = True
        logger.info("Cleared all cached labels.")

    def get_all_labels(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all cached labels (intent_json only), keyed by session_id.

        Only returns entries with matching versions.
        """
        result = {}
        for sid, entry in self._cache.items():
            if entry.get("model_version") == self.model_version and \
               entry.get("taxonomy_version") == self.taxonomy_version:
                result[sid] = entry.get("intent_json", {})
        return result

    def get_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
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
