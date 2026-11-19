"""检查近邻预测、无邻居回退及电影类型分支的行为。"""

import unittest

import numpy as np
import pandas as pd
import torch

from neurorec.models import ItemCF, MatrixFactorization, NeuralCF, UserCF


class NeighbourhoodTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(
            [(0, 0, 5.0), (0, 1, 1.0), (1, 0, 5.0), (1, 1, 1.0), (1, 2, 4.0),
             (2, 0, 1.0), (2, 1, 5.0), (2, 2, 2.0)],
            columns=["user", "item", "rating"],
        )

    def test_user_neighbour_learns_preference_and_excludes_self(self):
        model = UserCF(4, 4, neighbors=2, shrinkage=0).fit(self.train)
        self.assertEqual(model.similarity.diagonal().sum(), 0)
        self.assertGreater(float(model.predict([0], [2])[0]), model.user_means[0])
        self.assertAlmostEqual(float(model.predict([3], [0])[0]), model.global_mean, places=5)

    def test_item_neighbour_returns_finite_scores_with_missing_entities(self):
        model = ItemCF(4, 4, neighbors=2, shrinkage=2).fit(self.train)
        values = model.predict([0, 3, 3], [2, 0, 3])
        self.assertTrue(np.isfinite(values).all())
        self.assertAlmostEqual(float(values[1]), float(model.item_means[0]), places=5)
        self.assertAlmostEqual(float(values[2]), model.global_mean, places=5)
        self.assertEqual(model.similarity.diagonal().sum(), 0)

    def test_duplicate_observations_are_not_silently_summed(self):
        duplicated = pd.concat([self.train, self.train.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            UserCF(4, 4).fit(duplicated)


class NeuralContentTests(unittest.TestCase):
    genres = np.array([[1, 2, 0], [2, 0, 0], [0, 0, 0]], dtype=np.int64)

    def build(self, use_genres=False):
        return NeuralCF(3, 3, 3, self.genres, embedding_dim=4, hidden_dims=(8, 4), use_genres=use_genres)

    def test_disabled_content_is_exactly_the_baseline_under_same_seed(self):
        torch.manual_seed(73)
        baseline = self.build()
        torch.manual_seed(73)
        ablation = self.build(use_genres=False)
        users, items = torch.tensor([0, 1, 2]), torch.tensor([2, 0, 1])
        self.assertEqual(list(baseline.state_dict()), list(ablation.state_dict()))
        self.assertTrue(torch.equal(baseline(users, items), ablation(users, items)))
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, ablation.state_dict()[name]), name)

    def test_content_switch_preserves_base_initialisation_and_mlp_width(self):
        torch.manual_seed(19)
        baseline = self.build()
        torch.manual_seed(19)
        content = self.build(use_genres=True)
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, content.state_dict()[name]), name)
        self.assertEqual(baseline.mlp[0].in_features, content.mlp[0].in_features)

    def test_genre_average_masks_padding_and_empty_genres(self):
        model = self.build(use_genres=True)
        with torch.no_grad():
            model.genre_embedding.weight[0].fill_(999)  # 即使补齐向量异常，掩码也应将其排除。
            model.genre_embedding.weight[1].fill_(2)
            model.genre_embedding.weight[2].fill_(4)
        result = model.mean_genre_embedding(torch.tensor([0, 1, 2]))
        self.assertTrue(torch.equal(result[0], torch.full((4,), 3.0)))
        self.assertTrue(torch.equal(result[1], torch.full((4,), 4.0)))
        self.assertTrue(torch.equal(result[2], torch.zeros(4)))

    def test_genre_branch_receives_gradients_but_padding_does_not(self):
        torch.manual_seed(1)
        model = self.build(use_genres=True)
        predictions = model(torch.tensor([0, 1]), torch.tensor([0, 1]))
        ((predictions - torch.tensor([1.0, 5.0])) ** 2).mean().backward()
        self.assertGreater(float(model.genre_embedding.weight.grad[1:].abs().sum()), 0)
        self.assertEqual(float(model.genre_embedding.weight.grad[0].abs().sum()), 0)

    def test_neural_models_can_fit_a_tiny_rating_problem(self):
        users = torch.tensor([0, 0, 1, 1])
        items = torch.tensor([0, 1, 0, 1])
        target = torch.tensor([5.0, 1.0, 1.0, 5.0])
        for make_model in [lambda: MatrixFactorization(2, 2, embedding_dim=4),
                           lambda: NeuralCF(2, 2, 3, self.genres[:2], embedding_dim=4, hidden_dims=(8,), use_genres=True)]:
            torch.manual_seed(3)
            model = make_model()
            optimiser = torch.optim.Adam(model.parameters(), lr=0.03)
            initial = float(((model(users, items) - target) ** 2).mean().detach())
            for _ in range(120):
                optimiser.zero_grad()
                loss = ((model(users, items) - target) ** 2).mean()
                loss.backward()
                optimiser.step()
            final = float(((model(users, items) - target) ** 2).mean().detach())
            self.assertLess(final, initial / 20)


if __name__ == "__main__":
    unittest.main()
