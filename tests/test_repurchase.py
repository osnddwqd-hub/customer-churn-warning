"""Regression tests for leakage, temporal boundaries, and train-only fitting."""
import unittest
import numpy as np
import pandas as pd
from src.repurchase import (FEATURES, TARGET, build_features, build_samples,
                            clean_transactions, make_model, temporal_split)


def fixture():
    rows = [
        ("A", "a1", "2011-01-05", 1, 100),
        ("A", "a2", "2011-02-10", 1, 200),
        ("A", "a3", "2011-03-20", 1, 300),
        ("A", "a3", "2011-03-20", 2, 10),  # another line, same order
        ("B", "b1", "2011-01-03", 1, 100),
        ("B", "b2", "2011-01-20", 1, 80),
        ("A", "a4", "2011-04-01", 1, 50),  # exact cutoff is future
        ("B", "b3", "2011-05-01", 1, 60),  # exact label end excluded
        ("NEW", "n1", "2011-04-10", 1, 90),
        ("OLD", "o1", "2010-12-31", 1, 90),
    ]
    return clean_transactions(pd.DataFrame(rows, columns=["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]))


class PointInTimeTests(unittest.TestCase):
    def test_rfm_and_future_label(self):
        sample = build_samples(fixture(), ["2011-04-01"], coverage_start="2011-01-01", coverage_end="2011-05-01").set_index("CustomerID")
        self.assertEqual(set(sample.index), {"A", "B"})
        self.assertEqual(sample.loc["A", "Recency"], 12)
        self.assertEqual(sample.loc["A", "Frequency"], 3)
        self.assertEqual(sample.loc["A", "Monetary"], 620)
        self.assertEqual(sample.loc["A", TARGET], 0)
        self.assertEqual(sample.loc["B", TARGET], 1)

    def test_future_mutation_does_not_change_features(self):
        transactions = fixture()
        altered = transactions.copy()
        future = altered.InvoiceDate >= pd.Timestamp("2011-04-01")
        altered.loc[future, "TotalPrice"] = 999999
        altered = altered.loc[~(future & (altered.CustomerID == "A"))]
        pd.testing.assert_frame_equal(build_features(transactions, "2011-04-01"),
                                      build_features(altered, "2011-04-01"))
        sample = build_samples(altered, ["2011-04-01"], coverage_start="2011-01-01", coverage_end="2011-05-01")
        self.assertEqual(sample.loc[sample.CustomerID == "A", TARGET].item(), 1)

    def test_incomplete_label_rejected(self):
        with self.assertRaisesRegex(ValueError, "Incomplete future"):
            build_samples(fixture(), ["2011-04-01"], coverage_start="2011-01-01", coverage_end="2011-04-10")
        with self.assertRaisesRegex(ValueError, "Incomplete future"):
            build_samples(fixture(), ["2011-12-01"])
        with self.assertRaisesRegex(ValueError, "Incomplete observation"):
            build_samples(fixture(), ["2011-02-01"])

    def test_cleaning_removes_invalid_not_distinct_order_lines(self):
        raw = pd.DataFrame([
            (123.0, "1", "2011-01-01", 1, 5),
            (123.0, "1", "2011-01-01", 1, 5),
            (123.0, "1", "2011-01-01", 2, 5),
            (123.0, "C2", "2011-01-01", 1, 5),
            (123.0, "3", "2011-01-01", -1, 5),
            (123.0, "4", "2011-01-01", 1, 0),
            (None, "5", "2011-01-01", 1, 5),
            (123.0, "6", "invalid", 1, 5),
        ], columns=["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"])
        result = clean_transactions(raw)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result.CustomerID), {"123"})
        self.assertEqual(result.TotalPrice.sum(), 15)

    def test_repeated_user_and_mature_labels(self):
        rows = []
        for cutoff in pd.date_range("2011-03-01", "2011-11-01", freq="MS"):
            for customer, y in [("A", 0), ("B", 1)]:
                rows.append({"CustomerID": customer, "Cutoff": cutoff,
                             "LabelEnd": cutoff + pd.DateOffset(months=1), TARGET: y})
        samples = pd.DataFrame(rows)
        train, valid, test = temporal_split(samples)
        self.assertEqual(len(train), 10)
        self.assertEqual(len(valid), 2)
        self.assertEqual(len(test), 6)
        self.assertEqual(set(train.CustomerID), set(test.CustomerID))
        samples.loc[samples.Cutoff == pd.Timestamp("2011-07-01"), "LabelEnd"] = pd.Timestamp("2011-08-02")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            temporal_split(samples)

    def test_preprocessing_fit_uses_train_only(self):
        train = pd.DataFrame({"Recency": [1, 5, 20, 60], "Frequency": [10, 3, 2, 1], "Monetary": [500, 200, 90, 20]})
        model = make_model().fit(train[FEATURES], [0, 0, 1, 1])
        expected = np.column_stack([train.Recency, np.log1p(train.Frequency), np.log1p(train.Monetary)]).mean(axis=0)
        np.testing.assert_allclose(model.named_steps["scale"].mean_, expected)
        before = model.named_steps["scale"].mean_.copy()
        model.predict_proba(pd.DataFrame({"Recency": [100000], "Frequency": [100000], "Monetary": [1e9]}))
        np.testing.assert_array_equal(model.named_steps["scale"].mean_, before)


if __name__ == "__main__":
    unittest.main()
