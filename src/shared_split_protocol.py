"""
shared_split_protocol.py - Shared split protocol for fair intent variant comparison.

Ensures the same train/val/test split is used across all intent variants,
enabling fair comparison. Compatible with RoTE-TimeRec split protocols.
"""

import json
import hashlib
from typing import Dict, List, Optional, Tuple


class SplitProtocol:
    """
    Manages a consistent train/val/test split across intent variants.

    Splits are assigned per user (to prevent leakage) and can be
    saved/loaded from a JSON file for reproducibility.
    """

    def __init__(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self._user_splits: Dict[str, str] = {}

    def assign_splits(
        self,
        sessions: List[Dict],
        overwrite: bool = False,
    ) -> List[Dict]:
        """
        Assign split_ids to sessions based on user-level splits.

        Sessions belonging to the same user will get the same split.

        Args:
            sessions: List of session dicts.
            overwrite: If True, reassign splits even if already present.

        Returns:
            Sessions with 'split_id' field added/updated.
        """
        # Collect unique users
        users = set()
        for s in sessions:
            uid = s.get("user_id", "")
            if uid:
                users.add(uid)

        # Assign splits to users deterministically
        import random
        rng = random.Random(self.seed)
        sorted_users = sorted(users)
        rng.shuffle(sorted_users)

        n_users = len(sorted_users)
        n_test = max(1, int(n_users * self.test_ratio))
        n_val = max(1, int(n_users * self.val_ratio))

        for i, uid in enumerate(sorted_users):
            if i < n_test:
                self._user_splits[uid] = "test"
            elif i < n_test + n_val:
                self._user_splits[uid] = "val"
            else:
                self._user_splits[uid] = "train"

        # Apply splits to sessions
        for s in sessions:
            uid = s.get("user_id", "")
            if uid and (overwrite or "split_id" not in s):
                s["split_id"] = self._user_splits.get(uid, "train")

        return sessions

    def get_split(self, session: Dict) -> str:
        """Get the assigned split for a session."""
        uid = session.get("user_id", "")
        return self._user_splits.get(uid, session.get("split_id", "train"))

    def save(self, path: str):
        """Save user splits to JSON."""
        data = {
            "version": "1.0",
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "user_splits": self._user_splits,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SplitProtocol":
        """Load user splits from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        protocol = cls(
            val_ratio=data.get("val_ratio", 0.1),
            test_ratio=data.get("test_ratio", 0.1),
            seed=data.get("seed", 42),
        )
        protocol._user_splits = data.get("user_splits", {})
        return protocol

    def get_user_split(self, user_id: str) -> Optional[str]:
        """Get the split for a specific user."""
        return self._user_splits.get(user_id)

    @property
    def stats(self) -> Dict[str, int]:
        """Return split statistics."""
        counts = {"train": 0, "val": 0, "test": 0}
        for split in self._user_splits.values():
            counts[split] = counts.get(split, 0) + 1
        return counts


def split_by_timestamp(
    sessions: List[Dict],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> List[Dict]:
    """
    Alternative split method: split each user's sessions by timestamp.

    For each user, the most recent sessions go to test,
    the next recent to val, and the rest to train.
    """
    from collections import defaultdict

    user_sessions = defaultdict(list)
    for s in sessions:
        user_sessions[s.get("user_id", "")].append(s)

    result = []
    for uid, us in user_sessions.items():
        # Sort by last timestamp
        us.sort(key=lambda x: x.get("timestamps", [0])[-1] if x.get("timestamps") else 0)
        n = len(us)
        n_test = max(1, int(n * test_ratio))
        n_val = max(1, int(n * val_ratio))

        for i, s in enumerate(us):
            if i >= n - n_test:
                s["split_id"] = "test"
            elif i >= n - n_test - n_val:
                s["split_id"] = "val"
            else:
                s["split_id"] = "train"
            result.append(s)

    return result
