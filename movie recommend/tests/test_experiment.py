"""检查记录缺失、指标无效或消融不一致时，实验能否正确拒绝完成。"""
import unittest

import numpy as np
import pandas as pd

from neurorec.experiment import required_metrics, validate_results


def complete_records(seeds=(11, 22, 33, 44, 55), mode="full"):
    """生成仅供校验器测试的模拟记录，不作为实验数据或展示结果。"""
    config = {"seeds": list(seeds), "ks": [5, 10, 20], "mode": mode}
    models = ["UserCF", "ItemCF", "MF", "NeuralCF", "ContentAware"]
    metrics = required_metrics(config)
    rows, ablations, checks, execution = [], [], [], []
    for seed in seeds:
        by_model = {}
        for index, model in enumerate(models):
            scores = {metric: 0.1 + index * .001 + seed * .00001 for metric in metrics}
            best_epoch = 0 if model in {"UserCF", "ItemCF"} else 2
            rows.append(dict(model=model, seed=seed, fit_seconds=.1, best_epoch=best_epoch, **scores))
            by_model[model] = scores
            execution.append(dict(
                run_id=f"main-{model}-{seed}", model=model, seed=seed, role="main",
                fit_seconds=.1, epochs=0 if best_epoch == 0 else 3, best_epoch=best_epoch,
                independent_fit=True,
            ))
        ablations.append(dict(variant="genre_off", seed=seed, max_abs_difference=0.0,
                              **by_model["NeuralCF"]))
        ablations.append(dict(variant="genre_on", seed=seed, max_abs_difference=.01,
                              **by_model["ContentAware"]))
        checks.append(dict(
            seed=seed, best_epoch=2, baseline_best_epoch=2, independently_trained=True,
            max_abs_prediction_difference=0.0, max_abs_parameter_difference=0.0,
            max_abs_train_loss_difference=0.0, max_abs_val_rmse_difference=0.0,
            max_abs_metric_difference=0.0,
        ))
        execution.append(dict(
            run_id=f"ablation-GenreOff-{seed}", model="GenreOff", seed=seed,
            role="ablation_off", fit_seconds=.1, epochs=3, best_epoch=2, independent_fit=True,
        ))
    return pd.DataFrame(rows), pd.DataFrame(ablations), pd.DataFrame(checks), pd.DataFrame(execution), config


class CompletionGateTests(unittest.TestCase):
    def setUp(self):
        self.runs, self.ablations, self.checks, self.execution, self.config = complete_records()

    def validate(self):
        return validate_results(self.runs, self.ablations, self.checks, self.execution, self.config)

    def test_complete_formal_records_pass(self):
        report = self.validate()
        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["formal"])
        self.assertEqual(report["model_seed_rows"], 25)
        self.assertEqual(report["independent_fits"], 30)
        self.assertEqual(report["ablation_rows"], 10)

    def test_quick_records_pass_without_claiming_formal_completion(self):
        report = validate_results(*complete_records(seeds=(11,), mode="quick"))
        self.assertEqual(report["status"], "passed")
        self.assertFalse(report["formal"])
        self.assertEqual(report["model_seed_rows"], 5)
        self.assertEqual(report["independent_fits"], 6)

    def test_full_mode_rejects_fewer_than_five_distinct_seeds(self):
        with self.assertRaises(ValueError):
            validate_results(*complete_records(seeds=(11, 22, 33, 44)))

    def test_duplicate_config_seed_is_rejected(self):
        self.config["seeds"][-1] = 44
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_model_seed_is_rejected(self):
        self.runs = self.runs.iloc[:-1].copy()
        with self.assertRaises(ValueError):
            self.validate()

    def test_duplicate_seed_cannot_replace_a_missing_seed_even_with_25_rows(self):
        mask = (self.runs.model == "MF") & (self.runs.seed == 55)
        self.runs.loc[mask, "seed"] = 44
        with self.assertRaises(ValueError):
            self.validate()

    def test_nonfinite_rating_and_ranking_metrics_are_rejected(self):
        for metric in ("rmse", "precision@10", "sparse_item_mae"):
            for invalid in (np.nan, np.inf, -np.inf):
                with self.subTest(metric=metric, invalid=invalid):
                    self.runs, self.ablations, self.checks, self.execution, self.config = complete_records()
                    self.runs.loc[0, metric] = invalid
                    with self.assertRaises(ValueError):
                        self.validate()

    def test_missing_required_metric_column_is_rejected(self):
        self.runs = self.runs.drop(columns="ndcg@20")
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_ablation_seed_is_rejected(self):
        self.ablations = self.ablations.iloc[:-1].copy()
        with self.assertRaises(ValueError):
            self.validate()

    def test_duplicate_ablation_seed_is_rejected(self):
        mask = (self.ablations.variant == "genre_off") & (self.ablations.seed == 55)
        self.ablations.loc[mask, "seed"] = 44
        with self.assertRaises(ValueError):
            self.validate()

    def test_ablation_metrics_must_match_the_corresponding_main_model(self):
        for variant in ("genre_off", "genre_on"):
            with self.subTest(variant=variant):
                self.runs, self.ablations, self.checks, self.execution, self.config = complete_records()
                index = self.ablations.index[self.ablations.variant == variant][0]
                self.ablations.loc[index, "rmse"] += .01
                with self.assertRaises(ValueError):
                    self.validate()

    def test_nonfinite_ablation_differences_are_rejected(self):
        for invalid in (np.nan, np.inf):
            with self.subTest(invalid=invalid):
                self.ablations.loc[0, "max_abs_difference"] = invalid
                with self.assertRaises(ValueError):
                    self.validate()

    def test_missing_independent_ablation_check_is_rejected(self):
        self.checks = self.checks.iloc[:-1].copy()
        with self.assertRaises(ValueError):
            self.validate()

    def test_duplicate_ablation_check_seed_is_rejected(self):
        self.checks.loc[self.checks.seed == 55, "seed"] = 44
        with self.assertRaises(ValueError):
            self.validate()

    def test_reused_baseline_cannot_claim_independent_training(self):
        self.checks.loc[0, "independently_trained"] = False
        with self.assertRaises(ValueError):
            self.validate()

    def test_independent_ablation_must_reproduce_all_five_kinds_of_evidence(self):
        columns = [name for name in self.checks if name.startswith("max_abs_")]
        for column in columns:
            for invalid in (1e-7, np.nan, np.inf):
                with self.subTest(column=column, invalid=invalid):
                    self.runs, self.ablations, self.checks, self.execution, self.config = complete_records()
                    self.checks.loc[0, column] = invalid
                    with self.assertRaises(ValueError):
                        self.validate()

    def test_ablation_best_epoch_must_match_baseline(self):
        self.checks.loc[0, "best_epoch"] = 1
        with self.assertRaises(ValueError):
            self.validate()

    def test_agreeing_ablation_epochs_must_match_the_actual_main_baseline(self):
        self.checks.loc[0, ["best_epoch", "baseline_best_epoch"]] = 999
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_independent_fit_record_is_rejected(self):
        # 保留全部25条主实验记录，仅删除独立关闭类型分支的拟合记录。
        self.execution = self.execution.iloc[:-1].copy()
        with self.assertRaises(ValueError):
            self.validate()

    def test_execution_record_must_state_actual_independent_fit(self):
        self.execution.loc[0, "independent_fit"] = False
        with self.assertRaises(ValueError):
            self.validate()

    def test_execution_ids_must_be_unique(self):
        self.execution.loc[1, "run_id"] = self.execution.loc[0, "run_id"]
        with self.assertRaises(ValueError):
            self.validate()

    def test_unique_run_ids_cannot_hide_a_duplicate_model_seed_role(self):
        for column in ("model", "seed", "role"):
            self.execution.loc[1, column] = self.execution.loc[0, column]
        with self.assertRaises(ValueError):
            self.validate()

    def test_neural_execution_epoch_must_match_its_main_result(self):
        index = self.execution.index[(self.execution.model == "NeuralCF") & (self.execution.seed == 11)][0]
        self.execution.loc[index, ["epochs", "best_epoch"]] = 0
        with self.assertRaises(ValueError):
            self.validate()

    def test_selected_epoch_cannot_exceed_executed_training_epochs(self):
        index = self.execution.index[(self.execution.model == "MF") & (self.execution.seed == 11)][0]
        self.execution.loc[index, ["epochs", "best_epoch"]] = [1, 5]
        with self.assertRaises(ValueError):
            self.validate()

    def test_negative_rating_error_is_invalid_even_when_tables_agree(self):
        self.runs["rmse"] = -1.0
        self.ablations["rmse"] = -1.0
        with self.assertRaises(ValueError):
            self.validate()

    def test_nonfinite_or_negative_execution_measurements_are_rejected(self):
        for column in ("fit_seconds", "epochs", "best_epoch"):
            for invalid in (np.nan, np.inf, -1):
                with self.subTest(column=column, invalid=invalid):
                    self.runs, self.ablations, self.checks, self.execution, self.config = complete_records()
                    self.execution[column] = self.execution[column].astype(float)
                    self.execution.loc[0, column] = invalid
                    with self.assertRaises(ValueError):
                        self.validate()


if __name__ == "__main__":
    unittest.main()
