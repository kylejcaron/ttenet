import numpy as np
import pandas as pd
import pytest

from ttenet.data import expand_covariates, prepare_history
from ttenet.dates import allowed_days, calendar_features, date_grid, elapsed_days, to_day

# ---------------------------------------------------------------------------
# dates.py
# ---------------------------------------------------------------------------


def test_to_day_normalizes_timezone_to_utc_calendar_day():
    # 2026-01-01 23:30 in US/Eastern (-05:00) is 2026-01-02 04:30 UTC.
    assert to_day("2026-01-01T23:30:00-05:00") == np.datetime64("2026-01-02")
    assert to_day("2026-01-01") == np.datetime64("2026-01-01")


def test_to_day_treats_missing_values_as_nat():
    assert np.isnat(to_day(None))
    assert np.isnat(to_day(pd.NaT))
    arr = to_day(["2026-01-01", None, ""])
    assert not np.isnat(arr[0])
    assert np.isnat(arr[1])
    assert np.isnat(arr[2])


def test_date_grid_is_inclusive_and_rejects_inverted_range():
    grid = date_grid("2026-01-01", "2026-01-03")
    np.testing.assert_array_equal(
        grid, np.array(["2026-01-01", "2026-01-02", "2026-01-03"], dtype="datetime64[D]")
    )
    with pytest.raises(ValueError):
        date_grid("2026-01-03", "2026-01-01")


def test_elapsed_days_broadcasts_and_propagates_missing():
    assert elapsed_days("2026-01-01", "2026-01-05") == 4
    result = elapsed_days("2026-01-01", [np.datetime64("NaT", "D"), "2026-01-05"])
    assert np.isnan(result[0])
    assert result[1] == 4


def test_calendar_features_weekday_and_trig_encoding():
    # 2026-01-01 is a Thursday (weekday=3).
    features = calendar_features(["2026-01-01"])
    assert features["weekday"].iloc[0] == 3
    assert features["weekday_sin"].iloc[0] == pytest.approx(np.sin(2 * np.pi * 3 / 7))
    assert features["weekday_cos"].iloc[0] == pytest.approx(np.cos(2 * np.pi * 3 / 7))


def test_allowed_days_masks_weekdays_and_closed_dates():
    grid = date_grid("2026-01-01", "2026-01-05")  # Thu, Fri, Sat, Sun, Mon
    mask = allowed_days(grid, weekdays=range(5), closed_dates=["2026-01-02"])
    np.testing.assert_array_equal(mask, [True, False, False, False, True])


# ---------------------------------------------------------------------------
# prepare_history: tabular layout
# ---------------------------------------------------------------------------


def test_tabular_hides_future_initiation_and_marks_eligible():
    history = prepare_history(
        pd.DataFrame(
            {
                "item_id": ["a", "b"],
                "sale_date": ["2026-01-01"] * 2,
                "initiation_date": ["2026-01-10", "2026-04-02"],
            }
        ),
        as_of="2026-01-05",
    )
    assert history.frame.initiation_date.isna().all()
    assert history.frame.eligible.all()


def test_refreshing_history_recomputes_ages_and_expired_eligibility():
    original = prepare_history(
        pd.DataFrame({"item_id": ["a"], "sale_date": ["2026-01-01"]}),
        as_of="2026-01-05",
    )
    refreshed = prepare_history(original.frame, as_of="2026-04-02")
    assert refreshed.frame.age_since_sale.iloc[0] == 91
    assert not refreshed.frame.eligible.iloc[0]


def test_tabular_day90_boundary_is_allowed():
    history = prepare_history(
        pd.DataFrame(
            {"item_id": ["a"], "sale_date": ["2026-01-01"], "initiation_date": ["2026-04-01"]}
        ),
        as_of="2026-04-05",
        policy_days=90,
    )
    assert not history.frame.eligible.iloc[0]
    assert history.frame.initiation_date.iloc[0] == np.datetime64("2026-04-01")


def test_tabular_day91_boundary_is_rejected_when_observed():
    with pytest.raises(ValueError, match="policy deadline"):
        prepare_history(
            pd.DataFrame(
                {"item_id": ["a"], "sale_date": ["2026-01-01"], "initiation_date": ["2026-04-02"]}
            ),
            as_of="2026-04-05",
            policy_days=90,
        )


def test_tabular_same_day_events_are_valid():
    history = prepare_history(
        pd.DataFrame(
            {
                "item_id": ["a"],
                "sale_date": ["2026-01-01"],
                "initiation_date": ["2026-01-01"],
                "receipt_date": ["2026-01-01"],
            }
        ),
        as_of="2026-01-01",
    )
    assert history.frame.age_since_sale.iloc[0] == 0
    assert history.frame.age_since_initiation.iloc[0] == 0
    assert not history.frame.eligible.iloc[0]


def test_tabular_sale_after_cutoff_is_excluded():
    history = prepare_history(
        pd.DataFrame({"item_id": ["a", "b"], "sale_date": ["2026-01-01", "2026-02-01"]}),
        as_of="2026-01-15",
    )
    assert history.frame.item_id.tolist() == ["a"]


def test_tabular_receipt_without_initiation_is_rejected():
    with pytest.raises(ValueError, match="receipt_date"):
        prepare_history(
            pd.DataFrame(
                {"item_id": ["a"], "sale_date": ["2026-01-01"], "receipt_date": ["2026-01-05"]}
            ),
            as_of="2026-01-10",
        )


def test_tabular_initiation_before_sale_is_rejected():
    with pytest.raises(ValueError, match="initiation_date"):
        prepare_history(
            pd.DataFrame(
                {"item_id": ["a"], "sale_date": ["2026-01-05"], "initiation_date": ["2026-01-01"]}
            ),
            as_of="2026-01-10",
        )


def test_tabular_receipt_before_initiation_is_rejected():
    with pytest.raises(ValueError, match="receipt_date"):
        prepare_history(
            pd.DataFrame(
                {
                    "item_id": ["a"],
                    "sale_date": ["2026-01-01"],
                    "initiation_date": ["2026-01-05"],
                    "receipt_date": ["2026-01-03"],
                }
            ),
            as_of="2026-01-10",
        )


def test_tabular_duplicate_item_id_is_rejected():
    with pytest.raises(ValueError, match="duplicate item_id"):
        prepare_history(
            pd.DataFrame({"item_id": ["a", "a"], "sale_date": ["2026-01-01", "2026-01-02"]}),
            as_of="2026-01-10",
        )


def test_tabular_missing_sale_date_is_rejected():
    with pytest.raises(ValueError, match="sale_date"):
        prepare_history(pd.DataFrame({"item_id": ["a"], "sale_date": [None]}), as_of="2026-01-10")


def test_tabular_timezone_normalization():
    history = prepare_history(
        pd.DataFrame({"item_id": ["a"], "sale_date": ["2026-01-01T23:30:00-05:00"]}),
        as_of="2026-01-02",
    )
    assert history.frame.sale_date.iloc[0] == np.datetime64("2026-01-02")


# ---------------------------------------------------------------------------
# prepare_history: longitudinal layout
# ---------------------------------------------------------------------------


def test_longitudinal_equivalent_to_tabular_example():
    history = prepare_history(
        pd.DataFrame(
            {
                "item_id": ["a", "a", "b", "b"],
                "recorded_date": ["2026-01-01", "2026-01-05", "2026-01-01", "2026-01-05"],
                "sale_date": ["2026-01-01"] * 4,
                "initiation_date": [None, None, None, None],
            }
        ),
        as_of="2026-01-05",
        layout="longitudinal",
    )
    assert history.frame.initiation_date.isna().all()
    assert history.frame.eligible.all()


def test_longitudinal_uses_latest_snapshot_known_by_as_of():
    history = prepare_history(
        pd.DataFrame(
            {
                "item_id": ["a", "a"],
                "recorded_date": ["2026-01-01", "2026-01-20"],
                "sale_date": ["2026-01-01", "2026-01-01"],
                "initiation_date": [None, "2026-01-10"],
            }
        ),
        as_of="2026-01-05",
        layout="longitudinal",
    )
    # The 2026-01-20 snapshot (which reveals the initiation) is not yet known.
    assert history.frame.initiation_date.isna().iloc[0]
    assert history.frame.eligible.iloc[0]


def test_longitudinal_excludes_items_without_known_snapshot():
    history = prepare_history(
        pd.DataFrame(
            {
                "item_id": ["a", "b"],
                "recorded_date": ["2026-02-01", "2026-01-01"],
                "sale_date": ["2026-01-01", "2026-01-01"],
            }
        ),
        as_of="2026-01-10",
        layout="longitudinal",
    )
    assert history.frame.item_id.tolist() == ["b"]


def test_longitudinal_concat_indices_do_not_duplicate_sold_units():
    snapshots = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "recorded_date": ["2026-01-02", "2026-01-02"],
            "sale_date": ["2026-01-01", "2026-01-01"],
        },
        index=[0, 0],
    )
    history = prepare_history(snapshots, as_of="2026-01-03", layout="longitudinal")
    assert history.frame.item_id.tolist() == ["a", "b"]
    assert history.frame.eligible.sum() == 2


def test_longitudinal_duplicate_snapshot_same_date_is_rejected():
    with pytest.raises(ValueError, match="more than one longitudinal snapshot"):
        prepare_history(
            pd.DataFrame(
                {
                    "item_id": ["a", "a"],
                    "recorded_date": ["2026-01-01", "2026-01-01"],
                    "sale_date": ["2026-01-01", "2026-01-01"],
                }
            ),
            as_of="2026-01-05",
            layout="longitudinal",
        )


def test_longitudinal_requires_recorded_date_column():
    with pytest.raises(ValueError, match="recorded_date"):
        prepare_history(pd.DataFrame({"item_id": ["a"]}), as_of="2026-01-01", layout="longitudinal")


# ---------------------------------------------------------------------------
# prepare_history: changes layout
# ---------------------------------------------------------------------------


def test_changes_equivalent_to_tabular_example():
    history = prepare_history(
        pd.DataFrame(
            {
                "item_id": ["a", "b", "a", "b"],
                "date": ["2026-01-01", "2026-01-01", "2026-01-10", "2026-04-02"],
                "event": ["sale", "sale", "initiation", "initiation"],
            }
        ),
        as_of="2026-01-05",
        layout="changes",
    )
    assert history.frame.initiation_date.isna().all()
    assert history.frame.eligible.all()


def test_changes_preserves_static_features_from_sale_row_only():
    history = prepare_history(
        pd.DataFrame(
            {
                "item_id": ["a", "a"],
                "date": ["2026-01-01", "2026-01-10"],
                "event": ["sale", "initiation"],
                "color": ["red", "blue"],
            }
        ),
        as_of="2026-01-15",
        layout="changes",
    )
    assert history.frame.color.tolist() == ["red"]


def test_changes_event_without_sale_is_rejected():
    with pytest.raises(ValueError, match="sale event"):
        prepare_history(
            pd.DataFrame({"item_id": ["a"], "date": ["2026-01-05"], "event": ["initiation"]}),
            as_of="2026-01-10",
            layout="changes",
        )


def test_changes_duplicate_sale_events_rejected():
    with pytest.raises(ValueError, match="duplicate 'sale'"):
        prepare_history(
            pd.DataFrame(
                {
                    "item_id": ["a", "a"],
                    "date": ["2026-01-01", "2026-01-02"],
                    "event": ["sale", "sale"],
                }
            ),
            as_of="2026-01-10",
            layout="changes",
        )


def test_changes_unknown_event_type_rejected():
    with pytest.raises(ValueError, match="unknown event types"):
        prepare_history(
            pd.DataFrame({"item_id": ["a"], "date": ["2026-01-01"], "event": ["shipped"]}),
            as_of="2026-01-10",
            layout="changes",
        )


def test_changes_future_only_sale_is_excluded_not_error():
    history = prepare_history(
        pd.DataFrame(
            {
                "item_id": ["a", "b"],
                "date": ["2026-02-01", "2026-01-01"],
                "event": ["sale", "sale"],
            }
        ),
        as_of="2026-01-10",
        layout="changes",
    )
    assert history.frame.item_id.tolist() == ["b"]


# ---------------------------------------------------------------------------
# expand_covariates
# ---------------------------------------------------------------------------


def test_expand_covariates_changes_layout_sparse_cells_unchanged():
    grid = date_grid("2026-01-01", "2026-01-10")
    records = pd.DataFrame(
        {
            "item_id": ["a", "a", "a", "a"],
            "date": ["2026-01-01", "2026-01-03", "2026-01-05", "2026-01-08"],
            "storm": [0.0, np.nan, 1.0, np.nan],
            "promo": [2.0, 5.0, np.nan, 9.0],
        }
    )
    arr = expand_covariates(records, ["a"], grid, ["storm", "promo"], layout="changes")
    # storm changes only on 2026-01-05; the NaN cell on 2026-01-03 is "no change".
    np.testing.assert_array_equal(arr[0, :, 0], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    # promo changes on 2026-01-01, 2026-01-03 (skipping the NaN 2026-01-05 cell), 2026-01-08.
    np.testing.assert_array_equal(arr[0, :, 1], [2, 2, 5, 5, 5, 5, 5, 9, 9, 9])


def test_conflicting_same_day_covariate_changes_are_rejected():
    changes = pd.DataFrame(
        {
            "item_id": ["a", "a"],
            "date": ["2026-01-01"] * 2,
            "storm": [0.0, 1.0],
        }
    )
    with pytest.raises(ValueError):
        expand_covariates(changes, ["a"], date_grid("2026-01-01", "2026-01-02"), ["storm"])


def test_expand_covariates_changes_rejects_dates_before_first_known_value():
    records = pd.DataFrame({"item_id": ["a"], "date": ["2026-01-05"], "storm": [1.0]})
    grid = date_grid("2026-01-01", "2026-01-10")
    with pytest.raises(ValueError, match="no known value"):
        expand_covariates(records, ["a"], grid, ["storm"], layout="changes")


def test_expand_covariates_changes_rejects_item_with_no_known_value():
    records = pd.DataFrame({"item_id": ["a"], "date": ["2026-01-01"], "storm": [1.0]})
    grid = date_grid("2026-01-01", "2026-01-02")
    with pytest.raises(ValueError, match="no known value"):
        expand_covariates(records, ["z"], grid, ["storm"], layout="changes")


def test_expand_covariates_longitudinal_exact_alignment():
    grid = date_grid("2026-01-01", "2026-01-03")
    records = pd.DataFrame(
        {
            "item_id": ["a", "a", "a"],
            "date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "temp": [1.0, 2.0, 3.0],
        }
    )
    arr = expand_covariates(records, ["a"], grid, ["temp"], layout="longitudinal")
    np.testing.assert_array_equal(arr[0, :, 0], [1.0, 2.0, 3.0])


def test_expand_covariates_longitudinal_rejects_missing_coverage():
    grid = date_grid("2026-01-01", "2026-01-03")
    records = pd.DataFrame(
        {"item_id": ["a", "a"], "date": ["2026-01-01", "2026-01-03"], "temp": [1.0, 3.0]}
    )
    with pytest.raises(ValueError, match="missing longitudinal covariate coverage"):
        expand_covariates(records, ["a"], grid, ["temp"], layout="longitudinal")


def test_expand_covariates_longitudinal_rejects_duplicate_rows():
    records = pd.DataFrame(
        {
            "item_id": ["a", "a", "a"],
            "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
            "temp": [1.0, 9.0, 2.0],
        }
    )
    grid = date_grid("2026-01-01", "2026-01-02")
    with pytest.raises(ValueError, match="duplicate longitudinal"):
        expand_covariates(records, ["a"], grid, ["temp"], layout="longitudinal")


def test_expand_covariates_rejects_non_numeric_values():
    records = pd.DataFrame({"item_id": ["a"], "date": ["2026-01-01"], "storm": ["not-a-number"]})
    grid = date_grid("2026-01-01", "2026-01-01")
    with pytest.raises(ValueError, match="non-numeric"):
        expand_covariates(records, ["a"], grid, ["storm"], layout="changes")


def test_expand_covariates_zero_features_returns_trivial_shape():
    grid = date_grid("2026-01-01", "2026-01-10")
    arr = expand_covariates(pd.DataFrame(), ["a", "b"], grid, [], layout="changes")
    assert arr.shape == (2, len(grid), 0)


def test_expand_covariates_preserves_requested_item_order():
    records = pd.DataFrame(
        {"item_id": ["b", "a"], "date": ["2026-01-01", "2026-01-01"], "v": [100.0, 1.0]}
    )
    grid = date_grid("2026-01-01", "2026-01-01")
    arr = expand_covariates(records, ["a", "b"], grid, ["v"], layout="changes")
    assert arr[0, 0, 0] == 1.0
    assert arr[1, 0, 0] == 100.0
