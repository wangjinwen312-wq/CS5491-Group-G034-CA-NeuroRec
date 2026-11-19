"""用可手算的小数据核对时间划分、评价范围及指标公式，无需下载数据。"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np

from neurorec.data import _extract_safely, load_data, summarize
from neurorec.metrics import evaluate_rating, prepare_ranking, ranking_metrics


class MetricTests(unittest.TestCase):
    def test_rating_hand_calculated(self):
        result = evaluate_rating(np.array([1, 3, 5]), np.array([2, 3, 3]))
        self.assertAlmostEqual(result["rmse"], np.sqrt(5 / 3))
        self.assertAlmostEqual(result["mae"], 1.0)

    def test_binary_ndcg_and_short_list(self):
        result = ranking_metrics([10, 20, 30], {20, 40}, (2, 5))
        self.assertEqual(result["precision@2"], .5)
        self.assertEqual(result["recall@2"], .5)
        self.assertAlmostEqual(result["ndcg@2"], (1 / np.log2(3)) / (1 + 1 / np.log2(3)))
        self.assertEqual(result["precision@5"], .2)

    def test_bad_metric_inputs_fail(self):
        with self.assertRaises(ValueError):
            ranking_metrics([1, 1], {1})
        with self.assertRaises(ValueError):
            ranking_metrics([1], set())
        with self.assertRaises(ValueError):
            evaluate_rating([], [])
        with self.assertRaises(ValueError):
            evaluate_rating([1], [np.nan])


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # 用户电影对互不重复；在目标划分边界设置相同时间戳，检查是否整体划分。
        rows = [
            (1, 1, 5, 1), (1, 2, 4, 2), (2, 1, 3, 3), (2, 3, 5, 4),
            (1, 3, 2, 5), (2, 2, 4, 6), (3, 1, 5, 7), (3, 2, 3, 8),
            (1, 4, 5, 9), (2, 4, 5, 9), (3, 3, 4, 10), (4, 4, 5, 11),
        ]
        (self.root / "ratings.dat").write_text("\n".join("::".join(map(str, r)) for r in rows), encoding="latin-1")
        (self.root / "movies.dat").write_text(
            "1::A::Action|Comedy\n2::B::Comedy\n3::C::Drama\n4::D::Action\n5::E::Drama", encoding="latin-1"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_global_chronology_and_training_only_statistics(self):
        data = load_data(self.root, .5, .25)
        self.assertLess(data.train.timestamp.max(), data.val.timestamp.min())
        self.assertLess(data.val.timestamp.max(), data.test.timestamp.min())
        self.assertEqual(int(data.train_user_counts.sum()), len(data.train))
        self.assertEqual(data.train_user_counts[3], 0)  # 仅在测试期出现的新用户。
        self.assertEqual(data.train_item_counts[3], 0)  # 训练结束后才出现评分的电影。
        self.assertEqual(data.train_item_counts[4], 0)  # 只有静态信息、没有评分的电影。
        self.assertEqual(data.genre_ids.shape[0], 5)
        self.assertEqual(data.n_genres, 4)
        report = summarize(data)
        self.assertEqual(report["partitions"]["test"]["cold_user_rows"], 2)

    def test_test_ranking_filters_history_cold_entities_and_no_positives(self):
        data = load_data(self.root, .7, .15)
        ranking = prepare_ranking(data, "test", threshold=4)
        # 唯一合格正例是原始用户3的电影3；用户4和电影4没有训练历史。
        self.assertEqual(ranking.users.tolist(), [2])
        self.assertEqual(ranking.relevant[2], {2})
        self.assertEqual(ranking.catalog.tolist(), [0, 1, 2])
        self.assertIn(0, ranking.seen[2])  # 测试推荐需要排除验证期已看电影。
        self.assertGreater(ranking.metadata["cold_users_excluded"], 0)

    def test_zip_traversal_rejected(self):
        archive = self.root / "unsafe.zip"
        with ZipFile(archive, "w") as z:
            z.writestr("../escape.txt", "bad")
        with self.assertRaises(ValueError):
            _extract_safely(archive, self.root / "extract")


class ArchiveTests(unittest.TestCase):
    def test_incomplete_manual_archive_is_rejected_without_network_or_extraction(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "ml-1m.zip").write_bytes(b"not a complete archive")
            with patch("neurorec.data.urlopen", side_effect=AssertionError("Unexpected network request")):
                with self.assertRaisesRegex(ValueError, "size mismatch"):
                    load_data(root)
            self.assertFalse((root / "ml-1m").exists())

    def test_same_size_corrupt_archive_is_rejected_by_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "ml-1m.zip").write_bytes(b"corrupt")
            # 用小文件验证哈希检查，不必在测试中构造真实大小的压缩包。
            with patch("neurorec.data.ARCHIVE_SIZE", 7), patch("neurorec.data.ARCHIVE_SHA256", "0" * 64):
                with patch("neurorec.data.urlopen", side_effect=AssertionError("Unexpected network request")):
                    with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                        load_data(root)
            self.assertFalse((root / "ml-1m").exists())


if __name__ == "__main__":
    unittest.main()
