"""Load original UCI transactions, audit them and form calendar aggregates."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"
DATA_SHA256 = "43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d"
FORBIDDEN_CODES = frozenset({"85123A"})
COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
           "UnitPrice", "CustomerID", "Country"]


def file_sha256(path: Path) -> str:
    """Hash a file without loading it entirely into memory."""
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def locate_dataset(root: Path, download: bool = True) -> Path:
    """Find the supplied XLSX; optionally download and verify the UCI original.

    Never overwrite an existing dataset. A different hash is an error, so that
    a changed input cannot silently invalidate the reported results.
    """
    for path in (root / "Online Retail.xlsx", root / "data/raw/Online Retail.xlsx"):
        if path.is_file():
            if file_sha256(path) != DATA_SHA256:
                raise ValueError(f"Unexpected dataset SHA256: {path}")
            return path
    if not download:
        raise FileNotFoundError("Online Retail.xlsx was not found")
    path = root / "data/raw/Online Retail.xlsx"
    try:
        with urlopen(DATA_URL, timeout=120) as response:
            archive = response.read()
        with ZipFile(BytesIO(archive)) as source:
            matches = [n for n in source.namelist() if Path(n).name == path.name]
            if len(matches) != 1:
                raise ValueError("The UCI archive must contain exactly one Online Retail.xlsx")
            content = source.read(matches[0])
        if sha256(content).hexdigest() != DATA_SHA256:
            raise ValueError("Downloaded dataset SHA256 differs from the verified input")
    except (OSError, ValueError) as exc:
        raise RuntimeError("Cannot load UCI data. Download Online Retail.xlsx from UCI "
                           "and place it in the project root.") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def load_transactions(path: Path) -> pd.DataFrame:
    """Read all eight original columns; preserve identifiers as strings."""
    frame = pd.read_excel(path, engine="openpyxl",
                          dtype={"InvoiceNo": "string", "StockCode": "string"})
    if set(COLUMNS) != set(frame.columns):
        raise ValueError(f"Unexpected columns: {frame.columns.tolist()}")
    return frame[COLUMNS]


@dataclass
class CleanData:
    """Separate audited records, valid gross sales and signed credit records."""

    records: pd.DataFrame
    sales: pd.DataFrame
    credits: pd.DataFrame
    audit: pd.DataFrame


def clean_transactions(raw: pd.DataFrame, *, deduplicate: bool = True) -> CleanData:
    """Classify every row without conflating credits, corrections and sales.

    Only exact repeats of all original columns are removed. Description and
    CustomerID are not prerequisites for SKU-level volume aggregation.
    Classification order makes the audit categories mutually exclusive.
    """
    if not set(COLUMNS).issubset(raw.columns):
        raise ValueError("Required transaction columns are absent")
    x = raw[COLUMNS].copy()
    duplicates = x.duplicated(keep="first")
    if deduplicate:
        x = x.loc[~duplicates].copy()
    for col in ("InvoiceNo", "StockCode", "Description", "Country"):
        x[col] = x[col].astype("string").str.strip().replace("", pd.NA)
    x["StockCode"] = x.StockCode.str.upper()
    x["InvoiceNo"] = x.InvoiceNo.str.upper()
    x["InvoiceDate"] = pd.to_datetime(x.InvoiceDate, errors="coerce")
    for col in ("Quantity", "UnitPrice"):
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x["date"] = x.InvoiceDate.dt.normalize()
    cancel = x.InvoiceNo.str.startswith("C", na=False)
    numeric_bad = ~np.isfinite(x.Quantity) | ~np.isfinite(x.UnitPrice)
    invalid = (x[["InvoiceNo", "StockCode", "InvoiceDate"]].isna().any(axis=1)
               | numeric_bad | x.Quantity.mod(1).ne(0))
    # A transaction line is not assumed to identify an individual retail unit:
    # one stock unit may itself be a pack (e.g. a pack of paper cases).
    x["category"] = np.select(
        [invalid.to_numpy(dtype=bool),
         (cancel & x.Quantity.lt(0) & x.UnitPrice.gt(0)).to_numpy(dtype=bool),
         cancel.to_numpy(dtype=bool),
         x.Quantity.lt(0).to_numpy(dtype=bool),
         x.Quantity.eq(0).to_numpy(dtype=bool),
         x.UnitPrice.le(0).to_numpy(dtype=bool)],
        ["invalid_required", "credit", "cancellation_other", "negative_adjustment",
         "zero_quantity", "nonpositive_price"], default="sale")
    counts = x.category.value_counts()
    audit = pd.DataFrame({"rows": {
        "raw": len(raw), "exact_duplicates_detected": int(duplicates.sum()),
        "duplicates_removed": int(duplicates.sum()) if deduplicate else 0,
        "retained_for_audit": len(x), **counts.to_dict()}})
    if counts.sum() != len(x):
        raise AssertionError("Audit categories do not reconcile")
    return CleanData(x, x.loc[x.category.eq("sale")].copy(),
                     x.loc[x.category.eq("credit")].copy(), audit)


def compare_products(sales: pd.DataFrame, codes: list[str],
                     start: pd.Timestamp, end: pd.Timestamp,
                     store_days: int) -> pd.DataFrame:
    """Compare history, zero share and concentration for a declared shortlist."""
    n_days = (end.normalize() - start.normalize()).days + 1
    rows = []
    for code in codes:
        if code in FORBIDDEN_CODES:
            raise ValueError(f"StockCode {code} was used in class and is forbidden")
        p = sales.loc[sales.StockCode.eq(code)]
        if p.empty:
            raise ValueError(f"No valid sales for {code}")
        daily = p.groupby("date").Quantity.sum()
        invoices = p.groupby("InvoiceNo").Quantity.sum()
        descriptions = p.Description.dropna().value_counts()
        rows.append({"StockCode": code, "Description": descriptions.index[0],
                     "first": daily.index.min(), "last": daily.index.max(),
                     "active_days": len(daily), "units": int(daily.sum()),
                     "zero_share": 1 - len(daily) / n_days,
                     "store_day_coverage": len(daily) / store_days,
                     "max_invoice_share": invoices.max() / daily.sum(),
                     "max_day_share": daily.max() / daily.sum()})
    return pd.DataFrame(rows).set_index("StockCode")


def make_daily(clean: CleanData, code: str) -> pd.DataFrame:
    """Build a complete daily calendar and explicit coverage flags.

    units_recorded is zero where the ledger contains no eligible SKU sales.
    units_store_observed leaves whole-store silent days unknown (NaN).
    Neither variable measures unconstrained customer demand.
    """
    if code in FORBIDDEN_CODES:
        raise ValueError(f"Forbidden class product: {code}")
    p = clean.sales.loc[clean.sales.StockCode.eq(code)]
    if p.empty:
        raise ValueError(f"No valid sales for {code}")
    start, end = clean.records.date.min(), clean.records.date.max()
    index = pd.date_range(start, end, freq="D", name="date")
    result = pd.DataFrame(index=index)
    result["store_rows"] = clean.records.groupby("date").size().reindex(index, fill_value=0)
    result["units_recorded"] = p.groupby("date").Quantity.sum().reindex(index, fill_value=0)
    credits = clean.credits.loc[clean.credits.StockCode.eq(code)]
    result["credit_units"] = (-credits.groupby("date").Quantity.sum()).reindex(index, fill_value=0)
    result["net_recorded"] = result.units_recorded - result.credit_units
    result["store_observed"] = result.store_rows.gt(0)
    result["units_store_observed"] = result.units_recorded.where(result.store_observed)
    result["is_saturday"] = index.dayofweek == 5
    result["silent_non_saturday"] = ~result.store_observed & ~result.is_saturday
    result["boundary_day"] = index.isin([start, end])
    if result.units_recorded.sum() != p.Quantity.sum():
        raise AssertionError("Daily aggregation lost units")
    return result


def aggregate_calendar(daily: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Aggregate [start, end) weeks (Monday start) or calendar months.

    Always expose partial boundary periods, no-store days and per-day rate.
    observed_units is an observed subtotal; no extrapolation is performed.
    """
    if frequency not in {"W", "M"}:
        raise ValueError("frequency must be W (Mon–Sun) or M (calendar month)")
    if not isinstance(daily.index, pd.DatetimeIndex) or not daily.index.is_unique:
        raise ValueError("Daily data must have a unique DatetimeIndex")
    if not daily.index.equals(pd.date_range(daily.index.min(), daily.index.max(), freq="D")):
        raise ValueError("Daily data must be sorted with no missing calendar dates")
    periods = daily.index.to_period("W-SUN" if frequency == "W" else "M")
    g = daily.groupby(periods)
    result = g.agg(units=("units_recorded", "sum"), credits=("credit_units", "sum"),
                   days_in_extract=("units_recorded", "size"),
                   store_days=("store_observed", "sum"),
                   uncertain_days=("silent_non_saturday", "sum"),
                   boundary_days=("boundary_day", "sum"))
    result["start"] = result.index.start_time
    result["end_exclusive"] = (result.index + 1).start_time
    result["calendar_days"] = (result.end_exclusive - result.start).dt.days
    result["complete"] = result.days_in_extract.eq(result.calendar_days) & result.boundary_days.eq(0)
    result["units_per_observed_day"] = result.units / result.days_in_extract
    result.index = pd.DatetimeIndex(result.start, name="period_start")
    if result.units.sum() != daily.units_recorded.sum():
        raise AssertionError("Calendar aggregation lost units")
    return result


def gap_intervals(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Summarize contiguous calendar runs for any flagged set of dates."""
    if len(index) == 0:
        return pd.DataFrame(columns=["start", "end", "days"])
    dates = pd.Series(index.sort_values().unique())
    groups = dates.diff().dt.days.ne(1).cumsum()
    return dates.groupby(groups).agg(start="min", end="max", days="size").reset_index(drop=True)


def longest_regular_segment(series: pd.Series, excluded: pd.Series) -> pd.Series:
    """Keep the longest uninterrupted daily interval with no excluded days.

    Avoid dropping interior dates then treating adjacent observations as daily.
    """
    valid = ~excluded.reindex(series.index).fillna(True).astype(bool)
    runs = valid.ne(valid.shift()).cumsum()
    candidates = [series.loc[g.index] for _, g in valid[valid].groupby(runs[valid])]
    if not candidates:
        raise ValueError("No valid contiguous segment")
    return max(candidates, key=len).asfreq("D")
