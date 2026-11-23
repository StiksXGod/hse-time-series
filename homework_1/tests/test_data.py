"""Business and calendar invariants independent of the real research results."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from retail_ts import data


def transaction(invoice: str = "1", code: str = "20728", quantity: float = 2,
                date: str = "2020-01-01", price: float = 1.5,
                description: str | None = "BAG", customer: float | None = 1) -> dict:
    """A small transaction fixture, not a replacement for the UCI dataset."""
    return dict(InvoiceNo=invoice, StockCode=code, Description=description,
                Quantity=quantity, InvoiceDate=date, UnitPrice=price,
                CustomerID=customer, Country="United Kingdom")


def test_only_exact_duplicates_are_removed():
    row = transaction()
    source = pd.DataFrame([row, row.copy(), transaction(quantity=3)])
    clean = data.clean_transactions(source)
    assert len(clean.sales) == 2
    assert clean.sales.Quantity.sum() == 5
    assert clean.audit.loc["duplicates_removed", "rows"] == 1
    assert len(data.clean_transactions(source, deduplicate=False).sales) == 3


def test_credits_adjustments_and_free_items_are_separate():
    source = pd.DataFrame([transaction(), transaction(invoice="C2", quantity=-1),
                           transaction(invoice="3", quantity=-2),
                           transaction(invoice="4", price=0),
                           transaction(invoice="5", quantity=0),
                           transaction(invoice="C6", quantity=1)])
    clean = data.clean_transactions(source)
    assert len(clean.sales) == 1 and clean.sales.Quantity.sum() == 2
    assert len(clean.credits) == 1 and clean.credits.Quantity.sum() == -1
    assert set(clean.records.category) == {"sale", "credit", "negative_adjustment",
                                           "nonpositive_price", "zero_quantity", "cancellation_other"}


def test_optional_missing_fields_do_not_delete_sales():
    clean = data.clean_transactions(pd.DataFrame([transaction(description=None, customer=None)]))
    assert len(clean.sales) == 1
    assert clean.sales.CustomerID.isna().all() and clean.sales.Description.isna().all()


@pytest.mark.parametrize("patch", [{"InvoiceDate": "bad date"}, {"Quantity": np.nan},
                                   {"UnitPrice": np.inf}, {"Quantity": 1.5}, {"StockCode": None}])
def test_invalid_required_values_are_quarantined(patch):
    row = transaction()
    row.update(patch)
    clean = data.clean_transactions(pd.DataFrame([row]))
    assert clean.sales.empty and clean.records.category.iloc[0] == "invalid_required"


def test_daily_zeros_and_unknown_coverage_are_not_conflated():
    source = pd.DataFrame([transaction(date="2020-01-01"),
                           transaction(invoice="2", code="20725", date="2020-01-03"),
                           transaction(invoice="C3", quantity=-1, date="2020-01-04")])
    daily = data.make_daily(data.clean_transactions(source), "20728")
    assert daily.units_recorded.tolist() == [2, 0, 0, 0]
    assert pd.isna(daily.loc["2020-01-02", "units_store_observed"])
    assert daily.loc["2020-01-03", "units_store_observed"] == 0
    assert daily.loc["2020-01-04", "net_recorded"] == -1
    assert daily.boundary_day.tolist() == [True, False, False, True]


def test_week_boundary_is_monday_and_totals_are_conserved():
    source = pd.DataFrame([transaction(date="2020-01-05", quantity=3),
                           transaction(invoice="2", date="2020-01-06", quantity=5)])
    daily = data.make_daily(data.clean_transactions(source), "20728")
    weekly = data.aggregate_calendar(daily, "W")
    assert weekly.units.tolist() == [3, 5]
    assert weekly.index.tolist() == [pd.Timestamp("2019-12-30"), pd.Timestamp("2020-01-06")]
    assert weekly.end_exclusive.tolist() == [pd.Timestamp("2020-01-06"), pd.Timestamp("2020-01-13")]
    assert weekly.units.sum() == daily.units_recorded.sum()
    assert not weekly.complete.any()


def test_month_boundary_and_leap_year_exposure():
    source = pd.DataFrame([transaction(date="2020-01-31", quantity=3),
                           transaction(invoice="2", date="2020-02-29", quantity=5),
                           transaction(invoice="3", date="2020-03-01", quantity=7)])
    monthly = data.aggregate_calendar(data.make_daily(data.clean_transactions(source), "20728"), "M")
    assert monthly.units.tolist() == [3, 5, 7]
    assert monthly.calendar_days.tolist() == [31, 29, 31]
    assert monthly.complete.tolist() == [False, True, False]
    assert monthly.days_in_extract.tolist() == [1, 29, 1]


def test_class_sku_is_explicitly_forbidden():
    clean = data.clean_transactions(pd.DataFrame([transaction(code="85123A")]))
    with pytest.raises(ValueError, match="Forbidden"):
        data.make_daily(clean, "85123A")


def test_longest_segment_keeps_calendar_spacing():
    s = pd.Series(range(10), index=pd.date_range("2020-01-01", periods=10))
    excluded = pd.Series(False, index=s.index)
    excluded.iloc[3] = True
    segment = data.longest_regular_segment(s, excluded)
    assert segment.index.equals(s.index[4:])
    assert len(segment) == 6


def test_download_archive_and_checksum_without_network(tmp_path, monkeypatch):
    content = b"xlsx fixture bytes"
    blob = BytesIO()
    with ZipFile(blob, "w") as archive:
        archive.writestr("Online Retail.xlsx", content)
    monkeypatch.setattr(data, "DATA_SHA256", data.sha256(content).hexdigest())
    monkeypatch.setattr(data, "urlopen", lambda *a, **k: BytesIO(blob.getvalue()))
    path = data.locate_dataset(tmp_path)
    assert path.read_bytes() == content
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA256"):
        data.locate_dataset(tmp_path)


def test_no_download_when_disabled(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.locate_dataset(tmp_path, download=False)
