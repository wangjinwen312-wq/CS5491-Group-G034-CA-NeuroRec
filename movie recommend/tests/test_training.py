"""检查训练均值回退，以及关闭类型后独立训练能否复现基线。"""
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from neurorec.data import DataBundle
from neurorec.training import fit_neural, make_model, rating_predict, set_seed, training_means


def tiny_data():
    """构造包含冷用户、冷电影的数据，单独检验均值回退。"""
    train = pd.DataFrame({
        "user": [0, 0, 1], "item": [0, 1, 0], "rating": [1.0, 2.0, 5.0],
        "timestamp": [1, 2, 3],
    })
    heldout = pd.DataFrame({"user": [2], "item": [2], "rating": [5.0], "timestamp": [4]})
    return SimpleNamespace(
        train=train, val=heldout.copy(), test=heldout.copy(), n_users=3, n_items=3,
        n_genres=3, genre_ids=np.array([[1, 0], [1, 2], [2, 0]], dtype=np.int64),
        train_user_counts=np.array([2, 1, 0]), train_item_counts=np.array([2, 1, 0]),
    )


def tiny_training_bundle():
    """构造互不重叠的三个分区，留出集用户和电影均有训练历史。"""
    def frame(rows, start):
        data = pd.DataFrame(rows, columns=["user", "item", "rating"])
        data["timestamp"] = np.arange(start, start + len(data))
        return data

    train = frame([(0, 0, 5.0), (0, 1, 1.0), (1, 1, 5.0),
                   (1, 2, 2.0), (2, 2, 4.0), (2, 3, 2.0)], 1)
    val = frame([(0, 2, 4.0), (1, 3, 1.0), (2, 0, 3.0)], 7)
    test = frame([(0, 3, 1.0), (1, 0, 4.0), (2, 1, 5.0)], 10)
    return DataBundle(
        train=train, val=val, test=test, n_users=3, n_items=4,
        genre_ids=np.array([[1, 0], [1, 2], [2, 0], [2, 0]], dtype=np.int64),
        n_genres=3,
        movies=pd.DataFrame({"item": np.arange(4), "movie_id": np.arange(10, 14),
                             "title": ["A", "B", "C", "D"],
                             "genres": ["Action", "Action|Comedy", "Comedy", "Comedy"]}),
        user_ids=np.arange(1, 4), item_ids=np.arange(10, 14),
        train_user_counts=np.bincount(train.user, minlength=3),
        train_item_counts=np.bincount(train.item, minlength=4),
        genre_names=["<PAD>", "Action", "Comedy"], metadata={"synthetic": True},
    )


class RecordingPredictor:
    def __init__(self):
        self.calls = []

    def predict(self, users, items):
        self.calls.append((users.copy(), items.copy()))
        # 故意返回超出评分范围的值，检验是否统一截断到 1–5 分。
        return np.where(users == 0, 9.0, -2.0)


class TrainingTests(unittest.TestCase):
    def test_means_ignore_validation_and_test_ratings(self):
        data = tiny_data()
        before = training_means(data)
        data.val["rating"] = -1000.0
        data.test["rating"] = 1000.0
        after = training_means(data)
        self.assertAlmostEqual(before[0], 8 / 3)
        np.testing.assert_allclose(before[1], [1.5, 5.0, 8 / 3])
        np.testing.assert_allclose(before[2], [3.0, 2.0, 8 / 3])
        for left, right in zip(before, after):
            np.testing.assert_array_equal(left, right)

    def test_cold_cases_use_only_training_means_and_warm_cases_are_clipped(self):
        data = tiny_data()
        model = RecordingPredictor()
        users = np.array([0, 0, 2, 2, 1])
        items = np.array([0, 2, 1, 2, 1])
        prediction = rating_predict(model, users, items, data, training_means(data))
        np.testing.assert_allclose(prediction, [5.0, 1.5, 2.0, 8 / 3, 1.0])
        self.assertEqual(len(model.calls), 1)
        np.testing.assert_array_equal(model.calls[0][0], [0, 1])
        np.testing.assert_array_equal(model.calls[0][1], [0, 1])

    def test_all_cold_cases_never_query_untrained_model_parameters(self):
        data = tiny_data()
        model = RecordingPredictor()
        prediction = rating_predict(model, [2, 0], [2, 2], data, training_means(data))
        np.testing.assert_allclose(prediction, [8 / 3, 1.5])
        self.assertEqual(model.calls, [])

    def test_disabled_genres_reproduce_neuralcf_initialisation_and_update(self):
        data = tiny_data()
        config = {"embedding_dim": 4, "hidden_dims": [8, 4]}
        set_seed(123)
        baseline = make_model("NeuralCF", data, config)
        set_seed(123)
        disabled = make_model("ContentAware", data, config, use_genres=False)
        self.assertEqual(set(baseline.state_dict()), set(disabled.state_dict()))
        for key, tensor in baseline.state_dict().items():
            torch.testing.assert_close(tensor, disabled.state_dict()[key], rtol=0, atol=0)
        users = torch.tensor(data.train.user.to_numpy(), dtype=torch.long)
        items = torch.tensor(data.train.item.to_numpy(), dtype=torch.long)
        ratings = torch.tensor(data.train.rating.to_numpy(), dtype=torch.float32)
        # 不仅比较初始预测，还比较一次更新，排除隐藏的类型分支影响。
        for model in (baseline, disabled):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(users, items), ratings)
            loss.backward()
            optimizer.step()
        for key, tensor in baseline.state_dict().items():
            torch.testing.assert_close(tensor, disabled.state_dict()[key], rtol=0, atol=0)

    def test_independent_genre_off_training_reproduces_every_epoch_and_best_weights(self):
        data = tiny_training_bundle()
        config = {"embedding_dim": 4, "hidden_dims": [8, 4], "epochs": 4,
                  "patience": 5, "batch_size": 2, "learning_rate": 0.01,
                  "weight_decay": 0.00001}
        means = training_means(data)
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            set_seed(123)
            initial = make_model("NeuralCF", data, config)
            initial_embedding = initial.user_embedding.weight.detach().clone()
            # 两次调用分别新建模型和 Adam 优化器；
            # GenreOff 不接收基线训练好的权重，验证真正的独立重训。
            with redirect_stdout(StringIO()):
                baseline, baseline_history, baseline_epoch, _ = fit_neural(
                    "NeuralCF", data, config, 123, means)
                disabled, disabled_history, disabled_epoch, _ = fit_neural(
                    "GenreOff", data, config, 123, means, use_genres=False)
            self.assertIsNot(baseline, disabled)
            self.assertFalse(disabled.use_genres)
            self.assertEqual(len(baseline_history), config["epochs"])
            self.assertEqual(len(disabled_history), config["epochs"])
            self.assertEqual(baseline_epoch, disabled_epoch)
            for left, right in zip(baseline_history, disabled_history):
                self.assertEqual(left["model"], "NeuralCF")
                self.assertEqual(right["model"], "GenreOff")
                for field in ["seed", "epoch", "train_mse", "val_rmse", "val_mae"]:
                    self.assertEqual(left[field], right[field], field)
            self.assertFalse(torch.equal(baseline.user_embedding.weight.detach(), initial_embedding))
            self.assertEqual(set(baseline.state_dict()), set(disabled.state_dict()))
            for key, tensor in baseline.state_dict().items():
                self.assertTrue(torch.equal(tensor, disabled.state_dict()[key]), key)
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
