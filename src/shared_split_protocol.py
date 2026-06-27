"""
shared_split_protocol.py - 用于公平意图变体比较的共享划分协议。

确保所有意图变体使用相同的训练/验证/测试划分，
实现公平比较。与 RoTE-TimeRec 划分协议兼容。
"""

import json
import hashlib
from typing import Dict, List, Optional, Tuple


class SplitProtocol:
    """
    管理跨意图变体的一致训练/验证/测试划分。

    按用户分配划分（防止数据泄露），并且可以
    保存/加载到 JSON 文件以确保可复现性。
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
        根据用户级别的划分为会话分配 split_id。

        同一用户的会话将获得相同的划分。

        Args:
            sessions: 会话字典列表。
            overwrite: 如果为 True，即使已存在也重新分配划分。

        Returns:
            已添加/更新 'split_id' 字段的会话。
        """
        # 收集不重复的用户
        users = set()
        for s in sessions:
            uid = s.get("user_id", "")
            if uid:
                users.add(uid)

        # 确定性地为用户分配划分
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

        # 将会话应用划分
        for s in sessions:
            uid = s.get("user_id", "")
            if uid and (overwrite or "split_id" not in s):
                s["split_id"] = self._user_splits.get(uid, "train")

        return sessions

    def get_split(self, session: Dict) -> str:
        """获取会话的分配划分。"""
        uid = session.get("user_id", "")
        return self._user_splits.get(uid, session.get("split_id", "train"))

    def save(self, path: str):
        """将用户划分保存到 JSON。"""
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
        """从 JSON 加载用户划分。"""
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
        """获取特定用户的划分。"""
        return self._user_splits.get(user_id)

    @property
    def stats(self) -> Dict[str, int]:
        """返回划分统计信息。"""
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
    备选划分方法：按时间戳划分每个用户的会话。

    对于每个用户，最近的会话归测试集，
    次近的归验证集，其余归训练集。
    """
    from collections import defaultdict

    user_sessions = defaultdict(list)
    for s in sessions:
        user_sessions[s.get("user_id", "")].append(s)

    result = []
    for uid, us in user_sessions.items():
        # 按最后时间戳排序
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
