# =============================================================================
#  co.py  —  DataInspector + PlottingMethods  +  COVID-19 Global Data Demo
#  Dataset : Our World in Data (publicly hosted, no API key required)
#  Runtime : Google Colab  (Python 3.10+)
# =============================================================================
#
#  QUICK START
#  -----------
#  1. Open a new Google Colab notebook.
#  2. Upload this file or paste its contents into a single code cell.
#  3. Run the cell — all installs, class definitions, data loading, analysis,
#     and visualisations execute top-to-bottom automatically.
#
#  SECTION MAP
#  -----------
#  §0   System installs & imports
#  §1   DataInspector class
#  §2   PlottingMethods class
#  §3   COVID-19 dataset loading & feature engineering
#  §4   DataInspector pipeline  (cleaning → EDA → stats → PCA/FA → export)
#  §5   PlottingMethods showcase (8 chart types)
# =============================================================================


# ═════════════════════════════════════════════════════════════════════════════
# §0  SYSTEM INSTALLS & IMPORTS
# ═════════════════════════════════════════════════════════════════════════════

# Uncomment the two lines below the first time you run this in a fresh Colab
# session (graphviz is not pre-installed):
# !apt-get install -q graphviz
# !pip install -q graphviz

from __future__ import annotations

# ── Standard library ─────────────────────────────────────────────────────────
import io
import json
import time
import uuid
import copy
import base64
import inspect
import requests
from datetime import datetime
from typing import Optional, Sequence, Tuple, Dict, Any, List

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import scipy
from scipy.stats import (
    chi2_contingency,
    pointbiserialr,
    f_oneway,
    multivariate_normal,
)
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
)
from sklearn.decomposition import FactorAnalysis
from pydantic import BaseModel, ValidationError, field_validator

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import networkx as nx
import graphviz

# ── Colab-specific ────────────────────────────────────────────────────────────
try:
    from google.colab import files
    from IPython.display import display, HTML
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    # Graceful fallback so the file can be imported outside Colab too
    def display(x):   print(x)
    class _files:
        @staticmethod
        def upload():  return {}
        @staticmethod
        def download(path): print(f"[download] {path}")
    files = _files()
    class HTML:
        def __init__(self, html): self.html = html
        def __repr__(self): return f"<HTML ({len(self.html)} chars)>"


# ═════════════════════════════════════════════════════════════════════════════
# §1  DataInspector CLASS
# ═════════════════════════════════════════════════════════════════════════════

class DataInspector:
    """
    A comprehensive data cleaning and exploration tool for Google Colab.
    Provides interactive visualisations using Plotly and robust data
    sanitisation.
    """

    def __init__(self):
        self.df = None
        self.numeric_df = None
        self.categorical_df = None
        self.categorical_normalized_df = None
        self.normalized_data_df = None
        self.numeric_normalized_df = None

    # ── I/O ──────────────────────────────────────────────────────────────────

    def upload_data(self):
        """
        Prompts user to upload a CSV, handles common null strings,
        and attempts to auto-convert columns to their correct numeric types.
        """
        uploaded = files.upload()
        if not uploaded:
            return print("No file uploaded.")

        file_name = list(uploaded.keys())[0]
        self.df = pd.read_csv(
            io.BytesIO(uploaded[file_name]),
            na_values=["?", "n/a", "N/A", "NULL", "null", " "],
        )
        self.df["count"] = 1

        for col in self.df.columns:
            numeric_col = pd.to_numeric(self.df[col], errors="coerce")
            if not numeric_col.isna().all():
                self.df[col] = numeric_col

        print(f"\n✅ File '{file_name}' loaded and types sanitised!")

    def export_cleaned_data(self, filename="cleaned_data.csv"):
        """
        Converts the current DataFrame to CSV and triggers a browser download
        in the Google Colab environment.
        """
        if self.df is None:
            return
        self.df.to_csv(filename, index=False)
        files.download(filename)
        print(f"💾 '{filename}' has been generated and download triggered.")

    # ── Inspection & summary ─────────────────────────────────────────────────

    def get_summary(self):
        """Prints data dimensions and column type breakdown; shows first 20 rows."""
        if self.df is None:
            return print("Error: No data loaded.")
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        print("--- Data Summary ---")
        print(f"Rows: {self.df.shape[0]} | Columns: {self.df.shape[1]}")
        print(f"Numerical ({len(num_cols)}): {num_cols}")
        print(f"Categorical ({len(cat_cols)}): {cat_cols}")
        display(self.df.head(20))

    def show_missing_data(self):
        """Filters the DataFrame to show only rows containing at least one NaN."""
        if self.df is None:
            return
        missing_mask = self.df.isnull().any(axis=1) | (self.df == "").any(axis=1)
        missing_rows = self.df[missing_mask]
        if missing_rows.empty:
            print("✨ No missing data found!")
        else:
            print(f"🔍 Found {len(missing_rows)} rows with missing values:")
            display(missing_rows)

    def column_details(self):
        """Iterates through columns to show numeric ranges or categorical unique counts."""
        if self.df is None:
            return
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                print(
                    f"🔹 {col} (Numeric): Range [{self.df[col].min()} to {self.df[col].max()}]"
                )
            else:
                print(
                    f"🔸 {col} (Categorical): {self.df[col].nunique()} unique values"
                )

    def get_categorical_summary(self):
        """Generates a detailed summary for categorical columns (unique, mode, freq)."""
        if self.df is None:
            return
        cat_df = self.df.select_dtypes(exclude=[np.number])
        if cat_df.empty:
            return print("No categorical columns found.")
        summary = cat_df.describe().T[["unique", "top", "freq"]]
        print("--- Categorical Deep Dive ---")
        display(summary)

    # ── Data cleaning ────────────────────────────────────────────────────────

    def delete_rows(self):
        """Deletes rows by a comma-separated list of indices from user input."""
        if self.df is None:
            return
        try:
            user_input = input("Enter row indices to delete (e.g., 1, 3, 15): ")
            indices_to_drop = [
                int(i.strip()) for i in user_input.split(",") if i.strip().isdigit()
            ]
            existing_indices = [i for i in indices_to_drop if i in self.df.index]
            self.df = self.df.drop(index=existing_indices).reset_index(drop=True)
            print(
                f"🗑️ Deleted {len(existing_indices)} rows. New count: {len(self.df)}"
            )
        except Exception as e:
            print(f"❌ Error: {e}")

    def delete_columns(self):
        """Deletes columns by a comma-separated list of names from user input."""
        if self.df is None:
            return print("No data loaded.")
        try:
            print(f"Current columns: {', '.join(self.df.columns)}")
            user_input = input("Enter column names to delete (e.g., Column1, Column2): ")
            cols_to_drop = [c.strip() for c in user_input.split(",")]
            existing_cols = [c for c in cols_to_drop if c in self.df.columns]
            if not existing_cols:
                return print("⚠️ None of the provided column names were found.")
            self.df = self.df.drop(columns=existing_cols)
            print(
                f"🗑️ Deleted {len(existing_cols)} columns. Remaining: {len(self.df.columns)}"
            )
        except Exception as e:
            print(f"❌ Error: {e}")

    def handle_missing_values(self, columns=None, strategy="median", fill_value=None):
        """
        Imputes missing values in specified columns.

        Parameters
        ----------
        columns  : list of str, optional — defaults to all columns with NaNs
        strategy : 'mean' | 'median' | 'mode' | 'constant'
        fill_value : used only when strategy == 'constant'
        """
        if self.df is None:
            return
        target_cols = (
            columns
            if columns
            else self.df.columns[self.df.isnull().any()].tolist()
        )
        for col in target_cols:
            if strategy == "mean" and pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == "median" and pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == "mode":
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            elif strategy == "constant":
                self.df[col] = self.df[col].fillna(fill_value)
        print(
            f"🛠️ Imputation complete using '{strategy}' strategy for: {target_cols}"
        )

    def remove_duplicates(self):
        """Identifies and removes exact duplicate rows."""
        if self.df is None:
            return
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        dropped = initial_count - len(self.df)
        print(f"✨ Removed {dropped} duplicate rows. New row count: {len(self.df)}")

    def handle_outliers(self, columns=None, find_and_delete=False):
        """
        Flags outliers using IQR logic; optionally deletes flagged rows.
        """
        if self.df is None:
            return
        target_cols = (
            columns
            if columns
            else self.df.select_dtypes(include=[np.number]).columns.tolist()
        )
        all_outliers = set()
        for col in target_cols:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = self.df[
                (self.df[col] < (Q1 - 1.5 * IQR))
                | (self.df[col] > (Q3 + 1.5 * IQR))
            ]
            all_outliers.update(outliers.index.tolist())
            print(f"🚨 {col}: Found {len(outliers)} outliers.")
        if all_outliers:
            display(self.df.loc[list(all_outliers)])
            if find_and_delete:
                self.df = self.df.drop(index=list(all_outliers)).reset_index(
                    drop=True
                )
                print(f"🗑️ Deleted {len(all_outliers)} outlier rows.")

    # ── Feature extraction & normalisation ───────────────────────────────────

    def extract_numeric_data(self):
        """Returns a DataFrame of numeric columns only."""
        if self.df is None:
            return print("Error: No data loaded.")
        self.numeric_df = self.df.select_dtypes(include=[np.number])
        return self.numeric_df

    def extract_categorical_data(self):
        """Returns a DataFrame of categorical (non-numeric) columns only."""
        if self.df is None:
            return print("Error: No data loaded.")
        self.categorical_df = self.df.select_dtypes(exclude=[np.number])
        return self.categorical_df

    def extract_normalized_numeric_data(self, method="minmax"):
        """
        Scales numeric columns.

        Parameters
        ----------
        method : 'minmax' | 'standard' | 'robust'
        """
        if self.df is None:
            return print("Error: No data loaded.")
        num_df = self.df.select_dtypes(include=[np.number]).copy()
        if num_df.empty:
            print("⚠️ No numerical columns found to scale.")
            self.numeric_normalized_df = pd.DataFrame()
            return self.numeric_normalized_df
        if num_df.isnull().any().any():
            print("ℹ️ Imputing column medians before scaling…")
            num_df = num_df.fillna(num_df.median())
        method_lower = method.lower().strip()
        scaler_map = {
            "minmax": MinMaxScaler(),
            "standard": StandardScaler(),
            "robust": RobustScaler(),
        }
        if method_lower not in scaler_map:
            print(f"❌ Unknown method '{method}'. Defaulting to 'minmax'.")
            return self.extract_normalized_numeric_data(method="minmax")
        scaler = scaler_map[method_lower]
        scaled_data = scaler.fit_transform(num_df)
        self.numeric_normalized_df = pd.DataFrame(
            scaled_data, columns=num_df.columns, index=num_df.index
        )
        print(
            f"✨ Scaled numerical data using the '{method_lower}' method."
        )
        return self.numeric_normalized_df

    def extract_normalized_categorical_data(self, method="uniform"):
        """
        Encodes categorical columns.

        Parameters
        ----------
        method : 'uniform' | 'ordinal' | 'onehot' | 'minmax_ordinal'
        """
        if self.df is None:
            return print("Error: No data loaded.")
        cat_df = self.df.select_dtypes(exclude=[np.number]).copy()
        if cat_df.empty:
            print("⚠️ No categorical columns found.")
            self.categorical_normalized_df = pd.DataFrame()
            return self.categorical_normalized_df
        method_lower = method.lower().strip()
        if method_lower == "uniform":
            for col in cat_df.columns:
                codes = cat_df[col].astype("category").cat.codes
                max_code = codes.max()
                cat_df[col] = codes / max_code if max_code > 0 else 0.0
            self.categorical_normalized_df = cat_df
        elif method_lower == "ordinal":
            encoder = OrdinalEncoder()
            encoded = encoder.fit_transform(cat_df.fillna("Missing"))
            self.categorical_normalized_df = pd.DataFrame(
                encoded, columns=cat_df.columns, index=cat_df.index
            )
        elif method_lower == "onehot":
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = encoder.fit_transform(cat_df.fillna("Missing"))
            feature_names = encoder.get_feature_names_out(cat_df.columns)
            self.categorical_normalized_df = pd.DataFrame(
                encoded, columns=feature_names, index=cat_df.index
            )
        elif method_lower == "minmax_ordinal":
            encoder = OrdinalEncoder()
            scaler = MinMaxScaler()
            encoded = encoder.fit_transform(cat_df.fillna("Missing"))
            scaled = scaler.fit_transform(encoded)
            self.categorical_normalized_df = pd.DataFrame(
                scaled, columns=cat_df.columns, index=cat_df.index
            )
        else:
            print(f"❌ Unknown method '{method}'. Defaulting to 'uniform'.")
            return self.extract_normalized_categorical_data(method="uniform")
        print(
            f"✨ Encoded categorical data using the '{method_lower}' method."
        )
        return self.categorical_normalized_df

    def create_normalized_data_df(self):
        """Merges normalised numeric and normalised categorical DataFrames side-by-side."""
        if self.df is None:
            return print("Error: No data loaded.")
        num_df = self.extract_numeric_data()
        cat_norm_df = self.extract_normalized_categorical_data()
        if cat_norm_df is None or (
            isinstance(cat_norm_df, pd.DataFrame) and cat_norm_df.empty
        ):
            print("ℹ️ No categorical columns. Returning numeric DataFrame only.")
            self.normalized_data_df = num_df
            return self.normalized_data_df
        if num_df is None or (
            isinstance(num_df, pd.DataFrame) and num_df.empty
        ):
            print("ℹ️ No numeric columns. Returning categorical DataFrame only.")
            self.normalized_data_df = cat_norm_df
            return self.normalized_data_df
        self.normalized_data_df = pd.concat([num_df, cat_norm_df], axis=1)
        print(
            f"✅ Created merged DataFrame with {self.normalized_data_df.shape[1]} columns."
        )
        return self.normalized_data_df

    # ── Univariate visualisations ─────────────────────────────────────────────

    def plot_numerical(self, column_names):
        """Horizontal Violin + Scatter + Histogram for each numeric column."""
        if self.df is None:
            return
        if isinstance(column_names, str):
            column_names = [column_names]
        valid_cols = [
            c
            for c in column_names
            if c in self.df.columns
            and pd.api.types.is_numeric_dtype(self.df[c])
        ]
        for col in valid_cols:
            fig = make_subplots(
                rows=1,
                cols=3,
                subplot_titles=(
                    f"Horizontal Violin/Box: {col}",
                    f"Scatter Plot: {col}",
                    f"Distribution: {col}",
                ),
            )
            fig.add_trace(
                go.Violin(
                    x=self.df[col],
                    box_visible=True,
                    meanline_visible=True,
                    name=col,
                    orientation="h",
                    line_color="lightseagreen",
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    y=self.df[col],
                    mode="markers",
                    marker=dict(opacity=0.5, color="royalblue"),
                    name=col,
                ),
                row=1, col=2,
            )
            fig.add_trace(
                go.Histogram(x=self.df[col], name=col, marker_color="indianred"),
                row=1, col=3,
            )
            fig.update_layout(
                height=450,
                title_text=f"<b>Statistical Analysis: {col}</b>",
                showlegend=False,
                template="plotly_white",
            )
            fig.update_xaxes(title_text="Value", row=1, col=1)
            fig.update_yaxes(title_text="Value", row=1, col=2)
            fig.update_xaxes(title_text="Value", row=1, col=3)
            fig.show()

    def plot_categorical(self, column_names):
        """Interactive bar charts for categorical columns (counts + percentages)."""
        if self.df is None:
            return
        if isinstance(column_names, str):
            column_names = [column_names]
        for col in column_names:
            counts = self.df[col].value_counts().reset_index()
            counts.columns = [col, "count"]
            counts["percentage"] = (
                (counts["count"] / counts["count"].sum() * 100)
                .round(1)
                .astype(str)
                + "%"
            )
            fig = px.bar(
                counts,
                x=col,
                y="count",
                text="percentage",
                title=f"Frequency: {col}",
                color=col,
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.show()

    # ── Bivariate visualisations ──────────────────────────────────────────────

    def plot_relationship(self, col1, col2):
        """
        Selects the best plot based on column types:
          Num × Num  → scatter + OLS trendline
          Cat × Num  → box plot with data points
          Cat × Cat  → grouped bar chart
        """
        if self.df is None:
            return
        is_num1 = pd.api.types.is_numeric_dtype(self.df[col1])
        is_num2 = pd.api.types.is_numeric_dtype(self.df[col2])
        if is_num1 and is_num2:
            fig = px.scatter(
                self.df,
                x=col1,
                y=col2,
                trendline="ols",
                title=f"Correlation: {col1} vs {col2}",
            )
        elif is_num1 != is_num2:
            num, cat = (col1, col2) if is_num1 else (col2, col1)
            fig = px.box(
                self.df,
                x=cat,
                y=num,
                points="all",
                color=cat,
                title=f"Distribution of {num} by {cat}",
            )
        else:
            fig = px.histogram(
                self.df,
                x=col1,
                color=col2,
                barmode="group",
                title=f"Relationship: {col1} vs {col2}",
            )
        fig.show()

    # ── Correlation / association heatmaps ────────────────────────────────────

    def plot_numerical_correlation(self):
        """Pearson correlation heatmap for all numeric features."""
        if self.df is None:
            return
        numerical_df = self.df.select_dtypes(include=[np.number])
        corr = numerical_df.corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Pearson Correlation Heatmap",
        )
        fig.show()

    def plot_categorical_correlation(self):
        """Cramér's V association matrix for all categorical columns."""
        if self.df is None:
            return print("Error: No data loaded.")
        cat_df = self.df.select_dtypes(exclude=[np.number])
        if cat_df.empty:
            return print("⚠️ No categorical columns found.")
        cols = cat_df.columns
        n_cols = len(cols)
        corr_matrix = pd.DataFrame(
            np.zeros((n_cols, n_cols)), index=cols, columns=cols
        )
        for i in range(n_cols):
            for j in range(i, n_cols):
                col1, col2 = cols[i], cols[j]
                if i == j:
                    corr_matrix.loc[col1, col2] = 1.0
                    continue
                confusion_matrix = pd.crosstab(cat_df[col1], cat_df[col2])
                if confusion_matrix.size == 0 or min(confusion_matrix.shape) <= 1:
                    corr_matrix.loc[col1, col2] = 0.0
                    corr_matrix.loc[col2, col1] = 0.0
                    continue
                chi2 = chi2_contingency(confusion_matrix)[0]
                n = confusion_matrix.sum().sum()
                v = (
                    np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))
                    if n > 0
                    else 0.0
                )
                corr_matrix.loc[col1, col2] = v
                corr_matrix.loc[col2, col1] = v
        print("--- Cramér's V Association Matrix ---")
        display(corr_matrix.round(3))
        fig = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="<b>Cramér's V Categorical Association Heatmap</b>",
            labels=dict(color="Cramér's V"),
        )
        fig.update_layout(
            height=max(400, n_cols * 80),
            width=max(500, n_cols * 80),
            template="plotly_white",
        )
        fig.show()
        return corr_matrix

    def correlate_num_to_cat(self):
        """
        Point-Biserial r for binary categories; Eta (ANOVA) for multi-class.
        Returns a summary DataFrame.
        """
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        cat_cols = self.df.select_dtypes(exclude=[np.number]).columns
        if len(num_cols) == 0 or len(cat_cols) == 0:
            print("⚠️ Requires both numerical and categorical columns.")
            return pd.DataFrame()
        results = []
        for cat in cat_cols:
            for num in num_cols:
                valid_data = self.df[[cat, num]].dropna()
                if valid_data.empty:
                    continue
                categories = valid_data[cat].unique()
                if len(categories) < 2:
                    continue
                if len(categories) == 2:
                    binary_cat = pd.get_dummies(valid_data[cat], drop_first=True).iloc[:, 0]
                    corr, p_val = pointbiserialr(binary_cat, valid_data[num])
                    results.append({
                        "Categorical": cat,
                        "Numerical": num,
                        "Type": "Point-Biserial (Binary)",
                        "Correlation": round(corr, 3),
                        "P-Value": round(p_val, 4),
                    })
                else:
                    groups = [
                        valid_data[valid_data[cat] == val][num]
                        for val in categories
                    ]
                    groups = [g for g in groups if len(g) > 0]
                    if len(groups) > 1:
                        f_val, p_val = f_oneway(*groups)
                        grand_mean = valid_data[num].mean()
                        ss_total = ((valid_data[num] - grand_mean) ** 2).sum()
                        ss_between = sum(
                            len(g) * (g.mean() - grand_mean) ** 2 for g in groups
                        )
                        eta = np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0
                        results.append({
                            "Categorical": cat,
                            "Numerical": num,
                            "Type": "Eta (Multi-class ANOVA)",
                            "Correlation": round(eta, 3),
                            "P-Value": round(p_val, 4),
                        })
        return pd.DataFrame(results)

    def plot_all_associations_heatmap(self):
        """
        Unified association matrix for ALL column types (Pearson |r|,
        Cramér's V, Eta) displayed as a single Plotly heatmap.
        """
        if self.df is None:
            return print("Error: No data loaded.")
        cols = self.df.columns
        n_cols = len(cols)
        assoc_matrix = pd.DataFrame(
            np.zeros((n_cols, n_cols)), index=cols, columns=cols
        )
        for i in range(n_cols):
            for j in range(i, n_cols):
                col1, col2 = cols[i], cols[j]
                if i == j:
                    assoc_matrix.loc[col1, col2] = 1.0
                    continue
                valid_data = self.df[[col1, col2]].dropna()
                if valid_data.empty:
                    continue
                is_num1 = pd.api.types.is_numeric_dtype(valid_data[col1])
                is_num2 = pd.api.types.is_numeric_dtype(valid_data[col2])
                if is_num1 and is_num2:
                    val = abs(valid_data[col1].corr(valid_data[col2], method="pearson"))
                elif not is_num1 and not is_num2:
                    confusion_matrix = pd.crosstab(valid_data[col1], valid_data[col2])
                    if confusion_matrix.size > 0 and min(confusion_matrix.shape) > 1:
                        chi2 = chi2_contingency(confusion_matrix)[0]
                        n = confusion_matrix.sum().sum()
                        val = (
                            np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))
                            if n > 0
                            else 0.0
                        )
                    else:
                        val = 0.0
                else:
                    cat_col, num_col = (
                        (col1, col2) if not is_num1 else (col2, col1)
                    )
                    categories = valid_data[cat_col].unique()
                    if len(categories) > 1:
                        groups = [
                            valid_data[valid_data[cat_col] == c][num_col]
                            for c in categories
                        ]
                        groups = [g for g in groups if len(g) > 0]
                        grand_mean = valid_data[num_col].mean()
                        ss_total = ((valid_data[num_col] - grand_mean) ** 2).sum()
                        ss_between = sum(
                            len(g) * (g.mean() - grand_mean) ** 2 for g in groups
                        )
                        val = (
                            np.sqrt(ss_between / ss_total) if ss_total > 0 else 0.0
                        )
                    else:
                        val = 0.0
                assoc_matrix.loc[col1, col2] = round(val, 3)
                assoc_matrix.loc[col2, col1] = round(val, 3)
        print("--- Global Association Matrix ---")
        display(assoc_matrix)
        fig = px.imshow(
            assoc_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="viridis",
            title="<b>Unified Association Heatmap (Numeric & Categorical)</b>",
            labels=dict(color="Association Strength"),
        )
        fig.update_layout(
            height=max(500, n_cols * 45),
            width=max(600, n_cols * 45),
            template="plotly_white",
        )
        fig.show()
        return assoc_matrix

    # ── Statistical stationarity tests ────────────────────────────────────────

    def test_constant_mean(
        self,
        columns: Optional[Sequence[str]] = None,
        chunks: int = 10,
    ) -> Any:
        """
        MANOVA Wilks' Lambda test for first-moment homogeneity across
        sequential data blocks.
        """
        if self.df is None:
            raise ValueError("Error: No data loaded.")
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if "count" in target_cols:
                target_cols.remove("count")
            if not target_cols:
                raise ValueError("No numerical columns found.")
        else:
            if isinstance(columns, str):
                columns = [columns]
            non_numeric = [
                c
                for c in columns
                if c not in self.df.columns
                or not pd.api.types.is_numeric_dtype(self.df[c])
            ]
            if non_numeric:
                raise TypeError(
                    f"Not numerical or non-existent: {non_numeric}"
                )
            target_cols = list(columns)
        n = len(self.df)
        m = len(target_cols)
        chunk_size = n // chunks
        if chunk_size < m:
            raise ValueError(
                f"Sample per chunk ({chunk_size}) must be > features ({m}). Reduce chunks."
            )
        analysis_df = self.df[target_cols].copy().dropna()
        n = len(analysis_df)
        analysis_df["_chunk_label"] = np.minimum(
            np.arange(n) // chunk_size, chunks - 1
        )
        global_mean = analysis_df[target_cols].mean().values
        W = np.zeros((m, m))
        B = np.zeros((m, m))
        for label, group in analysis_df.groupby("_chunk_label"):
            X_chunk = group[target_cols].values
            chunk_mean = X_chunk.mean(axis=0)
            n_j = len(X_chunk)
            W += np.dot((X_chunk - chunk_mean).T, (X_chunk - chunk_mean))
            mean_diff = (chunk_mean - global_mean).reshape(-1, 1)
            B += n_j * np.dot(mean_diff, mean_diff.T)
        epsilon = 1e-6 * np.eye(m)
        W_stable = W + epsilon
        T_stable = W + B + epsilon
        sign_W, log_det_W = np.linalg.slogdet(W_stable)
        sign_T, log_det_T = np.linalg.slogdet(T_stable)
        if sign_W <= 0 or sign_T <= 0:
            raise np.linalg.LinAlgError("Matrices are poorly scaled.")
        log_wilks = log_det_W - log_det_T
        wilks_lambda = np.exp(log_wilks)
        df_stat = m * (chunks - 1)
        scale_factor = n - 1 - (m + chunks) / 2
        chi2_calc = max(0.0, -scale_factor * log_wilks)
        p_value = 1.0 - scipy.stats.chi2.cdf(chi2_calc, df_stat)
        print(
            f"\n--- MANOVA Mean Homogeneity Test (g={chunks} chunks, m={m} features) ---"
        )
        print(f"Wilks' Lambda (Λ): {wilks_lambda:.5f}")
        print(
            f"Chi-Square: {chi2_calc:.4f} | DF: {df_stat} | P-Value: {p_value:.6f}"
        )
        if p_value > 0.05:
            print(
                "✅ Fail to reject H0. No structural mean drift detected."
            )
        else:
            print(
                "🚨 Reject H0. Significant mean drift detected across rows."
            )
        return {
            "wilks_lambda": wilks_lambda,
            "chi2": chi2_calc,
            "p_value": p_value,
            "df": df_stat,
        }

    def test_constant_covariance(
        self,
        columns: Optional[Sequence[str]] = None,
        chunks: int = 5,
    ) -> Any:
        """
        Box's M-test for second-moment (covariance) homogeneity across
        sequential data blocks.
        """
        if self.df is None:
            raise ValueError("Error: No data loaded.")
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if "count" in target_cols:
                target_cols.remove("count")
            if not target_cols:
                raise ValueError("No numerical columns found.")
        else:
            if isinstance(columns, str):
                columns = [columns]
            non_numeric = [
                c
                for c in columns
                if c not in self.df.columns
                or not pd.api.types.is_numeric_dtype(self.df[c])
            ]
            if non_numeric:
                raise TypeError(f"Not numerical or non-existent: {non_numeric}")
            target_cols = list(columns)
        analysis_df = self.df[target_cols].copy().dropna()
        n = len(analysis_df)
        m = len(target_cols)
        chunk_size = n // chunks
        if chunk_size <= m:
            raise ValueError(
                f"DF per chunk ({chunk_size - 1}) must be > dimensions ({m})."
            )
        analysis_df["_chunk_label"] = np.minimum(
            np.arange(n) // chunk_size, chunks - 1
        )
        S_chunks, n_chunks = [], []
        log_det_S = 0.0
        pooled_S = np.zeros((m, m))
        total_df = 0
        epsilon = 1e-6 * np.eye(m)
        for label, group in analysis_df.groupby("_chunk_label"):
            X_chunk = group[target_cols].values
            n_j = len(X_chunk)
            S_j = np.cov(X_chunk, rowvar=False, ddof=1) + epsilon
            S_chunks.append(S_j)
            n_chunks.append(n_j)
            df_j = n_j - 1
            pooled_S += df_j * S_j
            total_df += df_j
            sign, logdet = np.linalg.slogdet(S_j)
            if sign <= 0:
                raise np.linalg.LinAlgError(
                    f"Covariance matrix for chunk {label} is non-positive definite."
                )
            log_det_S += df_j * logdet
        pooled_S /= total_df
        sign_p, log_det_Sp = np.linalg.slogdet(pooled_S)
        if sign_p <= 0:
            raise np.linalg.LinAlgError("Pooled covariance is non-positive definite.")
        M = total_df * log_det_Sp - log_det_S
        sum_inv_df = sum(1.0 / (nj - 1) for nj in n_chunks)
        inv_total_df = 1.0 / total_df
        numerator_C = 2.0 * m**2 + 3.0 * m - 1.0
        denominator_C = 6.0 * (m + 1.0) * (chunks - 1.0)
        C = (sum_inv_df - inv_total_df) * (numerator_C / denominator_C)
        chi2_calc = max(0.0, M * (1.0 - C))
        df_stat = (m * (m + 1) * (chunks - 1)) / 2.0
        p_value = 1.0 - scipy.stats.chi2.cdf(chi2_calc, df_stat)
        print(
            f"\n--- Box's M Covariance Test (g={chunks} chunks, m={m} features) ---"
        )
        print(f"Box's M: {M:.4f} | Chi²: {chi2_calc:.4f} | DF: {int(df_stat)} | P-Value: {p_value:.6f}")
        if p_value > 0.001:
            print(
                "✅ Fail to reject H0. Covariance structure is homoscedastic."
            )
        else:
            print("🚨 Reject H0. Multivariate heteroscedasticity detected.")
        return {
            "M": M,
            "chi2": chi2_calc,
            "p_value": p_value,
            "df": int(df_stat),
        }

    def test_row_independence(
        self,
        columns: Optional[Sequence[str]] = None,
        max_lag: Optional[int] = None,
    ) -> Any:
        """
        Multivariate Ljung-Box Portmanteau test for row-to-row independence.
        """
        if self.df is None:
            raise ValueError("Error: No data loaded.")
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if "count" in target_cols:
                target_cols.remove("count")
            if not target_cols:
                raise ValueError("No numerical columns found.")
        else:
            if isinstance(columns, str):
                columns = [columns]
            non_numeric = [
                c
                for c in columns
                if c not in self.df.columns
                or not pd.api.types.is_numeric_dtype(self.df[c])
            ]
            if non_numeric:
                raise TypeError(f"Not numerical or non-existent: {non_numeric}")
            target_cols = list(columns)
        analysis_df = self.df[target_cols].copy().dropna()
        n = len(analysis_df)
        m = len(target_cols)
        if max_lag is None:
            max_lag = int(np.ceil(np.log(n)))
        if max_lag >= n:
            raise ValueError(f"max_lag ({max_lag}) must be < sample size ({n}).")
        X = analysis_df[target_cols].values
        X_centered = X - X.mean(axis=0)
        epsilon = 1e-6 * np.eye(m)
        Gamma_0 = np.dot(X_centered.T, X_centered) / n + epsilon
        try:
            inv_Gamma_0 = np.linalg.inv(Gamma_0)
        except np.linalg.LinAlgError:
            inv_Gamma_0 = np.linalg.pinv(Gamma_0)
        Q_m = 0.0
        for k in range(1, max_lag + 1):
            Gamma_k = np.dot(X_centered[k:].T, X_centered[:-k]) / n
            M_k = np.dot(
                np.dot(np.dot(Gamma_k.T, inv_Gamma_0), Gamma_k), inv_Gamma_0
            )
            Q_m += np.trace(M_k) / (n - k)
        Q_m = max(0.0, Q_m * n**2)
        df_stat = m**2 * max_lag
        p_value = 1.0 - scipy.stats.chi2.cdf(Q_m, df_stat)
        print(
            f"\n--- Multivariate Ljung-Box Independence Test (lags={max_lag}) ---"
        )
        print(
            f"Q_m: {Q_m:.4f} | DF: {df_stat} | P-Value: {p_value:.6f}"
        )
        if p_value > 0.05:
            print(
                "✅ Fail to reject H0. No cross-autocorrelation detected. Rows are independent."
            )
        else:
            print(
                "🚨 Reject H0. Significant row-to-row serial dependency identified."
            )
        return {"Q_m": Q_m, "p_value": p_value, "df": df_stat}

    # ── Probabilistic modelling ───────────────────────────────────────────────

    def estimate_joint_normal(
        self, columns: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        """
        Fits a parametric Multivariate Normal distribution (MLE, Bessel's
        correction) to the verified IID baseline.
        Returns the distribution object and fit metrics (log-likelihood, AIC).
        """
        if self.df is None:
            raise ValueError("Error: No data loaded.")
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if "count" in target_cols:
                target_cols.remove("count")
        else:
            target_cols = list(columns)
        X = self.df[target_cols].copy().dropna().values
        n, m = X.shape
        if n <= m:
            raise ValueError("Sample size n must be > feature dimensions m.")
        mu_hat = np.mean(X, axis=0)
        epsilon = 1e-6 * np.eye(m)
        S_matrix = np.cov(X, rowvar=False, ddof=1) + epsilon
        joint_dist = multivariate_normal(
            mean=mu_hat, cov=S_matrix, allow_singular=True
        )
        log_likelihoods = joint_dist.logpdf(X)
        total_log_likelihood = np.sum(log_likelihoods)
        k_parameters = m + (m * (m + 1)) // 2
        aic = 2 * k_parameters - 2 * total_log_likelihood
        print(
            f"\n--- Micro-Scale MVN Model: X_i ~ N(μ_hat, S) ---"
        )
        print(f"Scale: m={m} features, n={n} samples")
        for col, val in zip(target_cols, mu_hat):
            print(f"  • {col}: {val:.4f}")
        print(
            f"Log-Likelihood: {total_log_likelihood:.4f} | AIC: {aic:.4f}"
        )
        return {
            "mean_vector": mu_hat,
            "covariance_matrix": S_matrix,
            "log_likelihood": total_log_likelihood,
            "aic": aic,
            "distribution_object": joint_dist,
            "features": target_cols,
        }

    def instantiate_macro_clt_distribution(
        self, columns: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        """
        Macro-scale CLT model: μ_hat_n ~ N(μ_hat_n, (1/n)·S).
        Models parameter uncertainty (sampling distribution of the mean vector).
        """
        if self.df is None:
            raise ValueError("Error: No data loaded.")
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if "count" in target_cols:
                target_cols.remove("count")
        else:
            target_cols = list(columns)
        X = self.df[target_cols].copy().dropna().values
        n, m = X.shape
        if n <= m:
            raise ValueError("Sample size n must be > feature dimensions m.")
        mu_hat = np.mean(X, axis=0)
        S_matrix = np.cov(X, rowvar=False, ddof=1)
        epsilon = 1e-10 * np.eye(m)
        clt_covariance = (1.0 / n) * S_matrix + epsilon
        macro_clt_dist = multivariate_normal(
            mean=mu_hat, cov=clt_covariance, allow_singular=True
        )
        total_parameter_variance = np.trace(clt_covariance)
        print(
            f"\n--- Macro-Scale CLT Model: μ_hat_n ~ N(μ_hat_n, (1/n)S) ---"
        )
        print(f"n={n} | Tr[(1/n)S]: {total_parameter_variance:.8f}")
        return {
            "mean_vector": mu_hat,
            "clt_covariance_matrix": clt_covariance,
            "total_parameter_variance": total_parameter_variance,
            "distribution_object": macro_clt_dist,
            "features": target_cols,
        }

    # ── Dimensionality reduction ──────────────────────────────────────────────

    def compute_empirical_pca(
        self,
        columns: Optional[Sequence[str]] = None,
        show_plot: bool = True,
    ) -> Dict[str, Any]:
        """
        Spectral decomposition of the empirical covariance matrix S.
        Computes Hotelling's T² and Q (SPE) statistics across all truncation
        boundaries k, and renders a 2×3 Plotly dashboard.
        """
        if self.df is None:
            raise ValueError("Error: No data loaded.")
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if "count" in target_cols:
                target_cols.remove("count")
        else:
            target_cols = list(columns)
        X = self.df[target_cols].copy().dropna().values
        n, m = X.shape
        if n <= m:
            raise ValueError("Sample size n must be > feature dimensions m.")
        mu_hat = np.mean(X, axis=0)
        X_centered = X - mu_hat
        S_matrix = np.cov(X, rowvar=False, ddof=1)
        eigenvalues, eigenvectors = np.linalg.eigh(S_matrix)
        idx = np.argsort(eigenvalues)[::-1]
        lambda_hat = np.clip(eigenvalues[idx], a_min=1e-15, a_max=None)
        P_hat = eigenvectors[:, idx]
        total_variance = np.sum(lambda_hat)
        if total_variance == 0:
            raise ValueError("Total variance is zero.")
        explained_variance_ratio = lambda_hat / total_variance
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        unexplained_variance_ratio = 1.0 - cumulative_variance_ratio
        Z_scores = np.dot(X_centered, P_hat)
        S_Z = np.cov(Z_scores, rowvar=False, ddof=1)
        k_range = np.arange(1, m)
        mean_T2_vs_k, mean_Q_vs_k = [], []
        T2_matrix = np.zeros((n, len(k_range)))
        Q_matrix = np.zeros((n, len(k_range)))
        for idx_k, k in enumerate(k_range):
            Z_k = Z_scores[:, :k]
            lambda_k = lambda_hat[:k]
            T2_samples = np.sum((Z_k**2) / lambda_k, axis=1)
            T2_matrix[:, idx_k] = T2_samples
            mean_T2_vs_k.append(np.mean(T2_samples))
            Z_residual = Z_scores[:, k:]
            Q_samples = np.sum(Z_residual**2, axis=1)
            Q_matrix[:, idx_k] = Q_samples
            mean_Q_vs_k.append(np.mean(Q_samples))
        print(
            f"\n--- PCA Decomposition: {m} features, {n} samples ---"
        )
        print(f"Total Variance (Tr[S]): {total_variance:.4f}")
        if show_plot:
            pc_labels = [f"PC {i+1}" for i in range(m)]
            k_labels = [f"k={k}" for k in k_range]
            fig = make_subplots(
                rows=2, cols=3,
                horizontal_spacing=0.18,
                vertical_spacing=0.28,
                subplot_titles=(
                    "Feature Loading Matrix |P_hat|",
                    "Component Values (Eigenvalues λ)",
                    "Information Profile (Explained Var.)",
                    "Residual Space (Unexplained Var.)",
                    "Mean Hotelling's T² vs Subspace Size k",
                    "Mean Q Statistic (SPE) vs Subspace Size k",
                ),
            )
            fig.add_trace(
                go.Heatmap(
                    z=np.abs(P_hat),
                    x=pc_labels,
                    y=target_cols,
                    colorscale="YlOrRd",
                    colorbar=dict(
                        title="Loading Weight",
                        x=-0.12, len=0.38, y=0.78,
                        yanchor="middle", xanchor="right", titleside="top",
                    ),
                    showscale=True, showlegend=False,
                ),
                row=1, col=1,
            )
            fig.update_xaxes(title_text="Principal Axes", row=1, col=1)
            fig.add_trace(
                go.Bar(
                    x=pc_labels, y=lambda_hat,
                    name="Eigenvalue (λ_j)",
                    marker=dict(color="#1f77b4", line=dict(color="black", width=0.5)),
                    legendgroup="eigen",
                ),
                row=1, col=2,
            )
            fig.update_yaxes(title_text="Variance Magnitude", row=1, col=2)
            fig.update_xaxes(title_text="Principal Axes", row=1, col=2)
            fig.add_trace(
                go.Bar(
                    x=pc_labels, y=explained_variance_ratio * 100,
                    name="Marginal Explained",
                    marker=dict(color="#ff7f0e", opacity=0.75),
                    legendgroup="expl",
                ),
                row=1, col=3,
            )
            fig.add_trace(
                go.Scatter(
                    x=pc_labels, y=cumulative_variance_ratio * 100,
                    mode="lines+markers",
                    name="Cumulative Captured",
                    line=dict(color="#d62728", width=2.5, dash="dash"),
                    legendgroup="expl",
                ),
                row=1, col=3,
            )
            fig.update_yaxes(title_text="Captured Structure (%)", range=[-2, 105], row=1, col=3)
            fig.update_xaxes(title_text="Principal Axes", row=1, col=3)
            fig.add_trace(
                go.Bar(
                    x=pc_labels, y=unexplained_variance_ratio * 100,
                    name="Remaining Noise",
                    marker=dict(color="#2ca02c", line=dict(color="black", width=0.5)),
                    legendgroup="noise",
                ),
                row=2, col=1,
            )
            fig.update_yaxes(title_text="Excluded Info (%)", range=[-2, 105], row=2, col=1)
            fig.update_xaxes(title_text="Principal Axes", row=2, col=1)
            fig.add_trace(
                go.Scatter(
                    x=k_labels, y=mean_T2_vs_k,
                    mode="lines+markers", name="Mean T²",
                    line=dict(color="#9467bd", width=2.5),
                    marker=dict(size=6, symbol="diamond"),
                    legendgroup="t2",
                ),
                row=2, col=2,
            )
            fig.update_yaxes(title_text="Average T² Metric", row=2, col=2)
            fig.update_xaxes(title_text="Truncation Cutoff (k)", row=2, col=2)
            fig.add_trace(
                go.Scatter(
                    x=k_labels, y=mean_Q_vs_k,
                    mode="lines+markers", name="Mean Q (SPE)",
                    line=dict(color="#e377c2", width=2.5),
                    marker=dict(size=6, symbol="square"),
                    legendgroup="q_stat",
                ),
                row=2, col=3,
            )
            fig.update_yaxes(title_text="Average Residual Energy", row=2, col=3)
            fig.update_xaxes(title_text="Truncation Cutoff (k)", row=2, col=3)
            fig.update_layout(
                title=dict(
                    text="PCA Optimisation & Feature Loading Dashboard",
                    x=0.5, y=0.97, xanchor="center", yanchor="top",
                ),
                template="plotly_white", showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(t=150, b=60, l=140, r=80),
                height=750, width=1250,
            )
            fig.show()
        return {
            "mean_vector": mu_hat,
            "covariance_matrix_S": S_matrix,
            "eigenvalues_lambda": lambda_hat,
            "eigenvectors_P": P_hat,
            "explained_variance_ratio": explained_variance_ratio,
            "cumulative_variance_ratio": cumulative_variance_ratio,
            "unexplained_variance_ratio": unexplained_variance_ratio,
            "transformed_scores_Z": Z_scores,
            "score_covariance_diagonal": np.diag(S_Z),
            "features": target_cols,
            "k_values": k_range,
            "T2_matrix_vs_k": T2_matrix,
            "Q_matrix_vs_k": Q_matrix,
            "mean_T2_profile": np.array(mean_T2_vs_k),
            "mean_Q_profile": np.array(mean_Q_vs_k),
        }

    def compute_empirical_fa(
        self,
        k: int,
        columns: Optional[Sequence[str]] = None,
        show_plot: bool = True,
    ) -> Dict[str, Any]:
        """
        Factor Analysis latent subspace framework (varimax rotation).
        Estimates latent factor scores via Thomson's MMSE regression method.
        Renders a 4-panel Plotly diagnostic dashboard.
        """
        if self.df is None:
            raise ValueError("Error: No data loaded.")
        if columns is None:
            target_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            if "count" in target_cols:
                target_cols.remove("count")
        else:
            target_cols = list(columns)
        X = self.df[target_cols].copy().dropna().values
        n, m = X.shape
        if n <= m:
            raise ValueError("Sample size n must be > feature dimensions m.")
        if k >= m:
            raise ValueError(f"k ({k}) must be < m ({m}).")
        mu_hat = np.mean(X, axis=0)
        std_hat = np.std(X, axis=0, ddof=1)
        std_hat[std_hat == 0] = 1e-15
        Z = (X - mu_hat) / std_hat
        R_matrix = np.corrcoef(X, rowvar=False)
        fa = FactorAnalysis(n_components=k, rotation="varimax", random_state=42)
        fa.fit(Z)
        lambda_matrix = fa.components_.T
        psi_diagonal = fa.noise_variance_
        communality = np.sum(lambda_matrix**2, axis=1)
        uniqueness = psi_diagonal
        F_scores = fa.transform(Z)
        print(
            f"\n--- Factor Analysis: {m} sensors → {k} latent factors ---"
        )
        print(
            f"Mean Communality: {np.mean(communality)*100:.2f}% | "
            f"Mean Uniqueness: {np.mean(uniqueness)*100:.2f}%"
        )
        if show_plot:
            factor_labels = [f"Factor {j+1}" for j in range(k)]
            fig = make_subplots(
                rows=2, cols=2,
                horizontal_spacing=0.24, vertical_spacing=0.28,
                subplot_titles=(
                    "Structural Loadings Matrix |λ_(j,r)|",
                    "Variance Partitioning (Communality vs Uniqueness)",
                    "Sensor Uniqueness Noise Floor (φ²)",
                    "Latent Factor Scores Empirical Variance",
                ),
            )
            fig.add_trace(
                go.Heatmap(
                    z=np.abs(lambda_matrix),
                    x=factor_labels, y=target_cols,
                    colorscale="YlOrRd",
                    colorbar=dict(
                        title="Sensitivity Score",
                        x=-0.15, len=0.38, y=0.78,
                        yanchor="middle", xanchor="right", titleside="top",
                    ),
                    showscale=True, name="Loadings",
                ),
                row=1, col=1,
            )
            fig.update_xaxes(title_text="Latent Structures", row=1, col=1)
            fig.add_trace(
                go.Bar(
                    y=target_cols, x=communality * 100,
                    name="Communality (h²)", orientation="h",
                    marker=dict(color="#1f77b4"),
                ),
                row=1, col=2,
            )
            fig.add_trace(
                go.Bar(
                    y=target_cols, x=uniqueness * 100,
                    name="Uniqueness (φ²)", orientation="h",
                    marker=dict(color="#ff7f0e"),
                ),
                row=1, col=2,
            )
            fig.update_layout(barmode="stack")
            fig.update_xaxes(title_text="Variance Allocation (%)", range=[0, 100], row=1, col=2)
            fig.add_trace(
                go.Scatter(
                    x=target_cols, y=uniqueness,
                    mode="lines+markers", name="Uniqueness Profile (φ²)",
                    line=dict(color="#d62728", width=2, dash="dot"),
                    marker=dict(size=8, symbol="x"),
                ),
                row=2, col=1,
            )
            fig.update_yaxes(range=[-0.05, 1.05], row=2, col=1)
            fig.update_xaxes(title_text="Monitored Channels", tickangle=25, row=2, col=1)
            factor_variances = np.var(F_scores, axis=0, ddof=1)
            fig.add_trace(
                go.Bar(
                    x=factor_labels, y=factor_variances,
                    name="Factor Empirical Variance",
                    marker=dict(color="#2ca02c", line=dict(color="black", width=0.5)),
                ),
                row=2, col=2,
            )
            fig.update_yaxes(title_text="Variance Level", row=2, col=2)
            fig.update_xaxes(title_text="Latent Vectors", row=2, col=2)
            fig.update_layout(
                title=dict(
                    text="Factor Analysis (FA) Latent Subspace Diagnostics Dashboard",
                    x=0.5, y=0.97, xanchor="center", yanchor="top",
                ),
                template="plotly_white", showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(t=150, b=60, l=140, r=80),
                height=750, width=1250,
            )
            fig.show()
        return {
            "mean_vector_mu": mu_hat,
            "std_vector_D": std_hat,
            "correlation_matrix_R": R_matrix,
            "factor_loadings_lambda": lambda_matrix,
            "uniqueness_psi": uniqueness,
            "communality_h2": communality,
            "latent_factor_scores_F": F_scores,
            "sensors": target_cols,
        }


# ═════════════════════════════════════════════════════════════════════════════
# §2  PlottingMethods CLASS
# ═════════════════════════════════════════════════════════════════════════════

class PlottingMethods:

    def get_methods_info(self, user_id=None):
        method_dicts = []
        methods = inspect.getmembers(self, inspect.ismethod)
        for name, method in methods:
            if name.startswith("_"):
                continue
            signature = inspect.signature(method)
            docstring = method.__doc__
            formatted_docstring = (
                docstring.strip() if docstring else "No description available"
            )
            method_dicts.append({
                "method": name,
                "signature": str(signature),
                "description": formatted_docstring,
            })
        return {"status": "success", "response": method_dicts}

    def _data_validate(self, data, message_dict):
        if data is None or (isinstance(data, str) and not data):
            message_dict.update({"message": "No data"})
            return {"status": "error", "message_dict": message_dict}
        if isinstance(data, pd.DataFrame):
            if data.empty:
                message_dict.update({"message": "No data"})
                return {"status": "error", "message_dict": message_dict}
            return {"status": "success", "data": data.to_dict(orient="records")}
        if isinstance(data, list):
            if not data:
                message_dict.update({"message": "No data"})
                return {"status": "error", "message_dict": message_dict}
            return {"status": "success", "data": data}
        try:
            parsed_data = json.loads(data)
            records = (
                parsed_data.get("records")
                if isinstance(parsed_data, dict)
                else parsed_data
            )
            if not records:
                message_dict.update({"message": "No data"})
                return {"status": "error", "message_dict": message_dict}
            return {"status": "success", "data": records}
        except (json.JSONDecodeError, TypeError):
            message_dict.update({"message": "Invalid data format"})
            return {"status": "error", "message_dict": message_dict}

    # ── Chart helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _to_html(fig, include_plotlyjs=True):
        plot_html = pio.to_html(
            fig,
            full_html=False,
            config={"displaylogo": False, "responsive": True},
            include_plotlyjs=include_plotlyjs,
        )
        fig_id = str(uuid.uuid4())[:8]
        return plot_html.replace("<div>", f'<div id="{fig_id}">')

    def display_image(self, result):
        if result["status"] == "success":
            response_data = json.loads(result["response"]["data"])
            plot_html = response_data["figure"]
            display(HTML(plot_html))
        else:
            print(f"Failed to plot: {result['response'].get('message', 'unknown error')}")

    # ── Public chart methods ─────────────────────────────────────────────────

    def plot_bar_chart(
        self,
        x="date", y="value", color=None, text=None,
        title="", barmode="stack", hover_data=None,
        data_id=None, data='{"records":[]}', meta_data={}, user_id=None,
    ):
        """Bar chart with optional colour grouping and hover data."""
        try:
            message_dict = {"message": meta_data}
            validated = self._data_validate(data, message_dict)
            if validated["status"] != "success":
                md = validated["message_dict"]
                return {"status": "error", "response": {"meta_data": md, "data": json.dumps({"figure": ""})}, "message": md.get("message")}
            data = validated["data"]
            if isinstance(hover_data, str):
                try:
                    parsed = json.loads(hover_data)
                    hover_data = parsed if isinstance(parsed, list) else None
                except json.JSONDecodeError:
                    hover_data = hover_data.split(",") if "," in hover_data else None
            df = pd.DataFrame(data)
            df[y] = pd.to_numeric(df[y])
            c_categories_labels = None
            if color is not None:
                df.dropna(subset=[color], inplace=True)
                c_categories_labels = df[color].unique()
                if not any(sub in color.lower() for sub in ["month", "week"]):
                    c_categories_labels = sorted(c_categories_labels)
                df[color] = pd.Categorical(df[color], categories=c_categories_labels, ordered=True)
            x_categories_labels = df[x].unique()
            df[x] = pd.Categorical(df[x], categories=x_categories_labels, ordered=True)
            if hover_data:
                hover_data = [col for col in hover_data if col in df.columns]
            cat_orders = {x: x_categories_labels}
            if color is not None and c_categories_labels is not None:
                cat_orders[color] = c_categories_labels
            fig = px.bar(
                df, x=x, y=y, color=color, title=title,
                text=text, hover_data=hover_data, category_orders=cat_orders,
            )
            fig.update_layout(
                xaxis_title=x, yaxis_title=y,
                uniformtext_minsize=8, uniformtext_mode="hide", barmode=barmode,
            )
            fig_return = self._to_html(fig)
            message_dict.update({"message": "Bar chart plotted"})
            return {"status": "success", "response": {"meta_data": message_dict, "data": json.dumps({"figure": fig_return}), "message": json.dumps(message_dict)}}
        except Exception as e:
            message_dict.update({"message": f"Error: {e}"})
            return {"status": "error", "response": {"meta_data": message_dict, "data": json.dumps({"figure": ""})}, "message": json.dumps(message_dict)}

    def plot_pie_chart(
        self,
        names="date", values="value", title="", hole=None,
        data_id=None, data='{"records":[]}', meta_data={}, user_id=None,
    ):
        """Responsive Plotly pie (or donut) chart."""
        try:
            message_dict = {"message": meta_data}
            validated = self._data_validate(data, message_dict)
            if validated["status"] != "success":
                md = validated["message_dict"]
                return {"status": "error", "response": {"meta_data": md, "data": json.dumps({"figure": ""})}, "message": md.get("message")}
            data = validated["data"]
            df = pd.DataFrame(data)
            fig = px.pie(df, names=names, values=values, title=title, hole=hole)
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(title=title, uniformtext_minsize=10, uniformtext_mode="hide")
            fig_return = self._to_html(fig)
            message_dict.update({"message": "Pie chart plotted"})
            return {"status": "success", "response": {"meta_data": message_dict, "data": json.dumps({"figure": fig_return}), "message": json.dumps(message_dict)}}
        except Exception as e:
            message_dict.update({"message": f"Error: {e}"})
            return {"status": "error", "response": {"meta_data": message_dict, "data": json.dumps({"figure": ""})}, "message": json.dumps(message_dict)}

    def plot_histogram(
        self,
        x="value", title="", bins=None,
        data='{"records":[]}', meta_data={}, user_id=None,
    ):
        """Plotly histogram with optional custom bin intervals."""
        message_dict = {"message": meta_data}
        validated = self._data_validate(data, message_dict)
        if validated["status"] != "success":
            md = validated["message_dict"]
            return {"status": "error", "response": {"meta_data": md, "data": json.dumps({"figure": ""})}, "message": md.get("message")}
        data = validated["data"]
        df = pd.DataFrame(data)
        if bins:
            if not isinstance(bins, list) or len(bins) < 2:
                return {"status": "error", "response": {"meta_data": "Invalid bins.", "data": {"figure": ""}}}
            df[x] = pd.cut(df[x], bins=bins, right=False).astype(str)
        fig = px.histogram(
            df, x=x, title=title,
            category_orders={x: [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins) - 1)]} if bins else None,
        )
        fig.update_layout(title=title, xaxis_title=x, yaxis_title="Count", bargap=0.2)
        fig_return = self._to_html(fig)
        message_dict.update({"message": "Histogram plotted"})
        return {"status": "success", "response": {"meta_data": json.dumps(message_dict), "data": json.dumps({"figure": fig_return})}}

    def plot_simple_sunburst_graph(
        self,
        path=["parent", "name"], values="marks", title="Hierarchy map",
        data_id=None, data='{"records":[]}', meta_data={}, user_id=None,
    ):
        """Sunburst chart for hierarchical data."""
        try:
            message_dict = {"message": meta_data}
            validated = self._data_validate(data, message_dict)
            if validated["status"] != "success":
                md = validated["message_dict"]
                return {"status": "error", "response": {"meta_data": md, "data": json.dumps({"figure": ""})}, "message": md.get("message")}
            data = validated["data"]
            df = pd.DataFrame(data).fillna("")
            fig = px.sunburst(df, path=path, values=values, title=title)
            fig_return = self._to_html(fig)
            message_dict.update({"message": "Sunburst plotted"})
            return {"status": "success", "response": {"meta_data": json.dumps(message_dict), "data": json.dumps({"figure": fig_return}), "message": json.dumps(message_dict)}}
        except Exception as e:
            return {"status": "error", "response": {"meta_data": json.dumps({"message": f"Error: {e}"}), "data": json.dumps({"figure": ""})}}

    def plot_tree_map(
        self,
        path=["parent", "name"], values="marks", title="Hierarchy map",
        data_id=None, data='{"records":[]}', meta_data={}, user_id=None,
    ):
        """Treemap for hierarchical data."""
        try:
            message_dict = {"message": meta_data}
            validated = self._data_validate(data, message_dict)
            if validated["status"] != "success":
                md = validated["message_dict"]
                return {"status": "error", "response": {"meta_data": md, "data": json.dumps({"figure": ""})}, "message": md.get("message")}
            data = validated["data"]
            df = pd.DataFrame(data)
            fig = px.treemap(df, path=path, values=values, title=title)
            fig_return = self._to_html(fig)
            message_dict.update({"message": "Treemap plotted"})
            return {"status": "success", "response": {"meta_data": json.dumps(message_dict), "data": json.dumps({"figure": fig_return}), "message": json.dumps(message_dict)}}
        except Exception as e:
            message_dict.update({"message": f"Error: {e}"})
            return {"status": "error", "response": {"meta_data": json.dumps(message_dict), "data": json.dumps({"figure": ""})}}

    def plot_sankey_diagram(
        self,
        source_column="parent", target_column="name", values="marks",
        title="Sankey Diagram",
        data_id=None, data='{"records":[]}', meta_data={}, user_id=None,
    ):
        """Sankey diagram from source / target / value columns."""
        try:
            message_dict = {"message": meta_data}
            validated = self._data_validate(data, message_dict)
            if validated["status"] != "success":
                md = validated["message_dict"]
                return {"status": "error", "response": {"meta_data": md, "data": json.dumps({"figure": ""})}, "message": md.get("message")}
            data = validated["data"]
            df = pd.DataFrame(data)
            grouped_df = df.groupby([source_column, target_column], as_index=False).agg({values: "sum"})
            unique_nodes = pd.concat([grouped_df[source_column], grouped_df[target_column]]).unique()
            node_map = {node: idx for idx, node in enumerate(unique_nodes)}
            sources = grouped_df[source_column].map(node_map).tolist()
            targets = grouped_df[target_column].map(node_map).tolist()
            vals = grouped_df[values].tolist()
            fig = go.Figure(
                data=[
                    go.Sankey(
                        node=dict(
                            pad=15, thickness=20,
                            line=dict(color="black", width=0.5),
                            label=list(node_map.keys()),
                        ),
                        link=dict(source=sources, target=targets, value=vals),
                    )
                ]
            )
            fig.update_layout(title_text=title, font_size=10)
            fig_return = self._to_html(fig)
            message_dict.update({"message": "Sankey plotted"})
            return {"status": "success", "response": {"meta_data": json.dumps(message_dict), "data": json.dumps({"figure": fig_return}), "message": json.dumps(message_dict)}}
        except Exception as e:
            message_dict.update({"message": f"Error: {e}"})
            return {"status": "error", "response": {"meta_data": message_dict, "data": json.dumps({"figure": ""})}}

    def plot_heat_map(
        self,
        values="Sales", index="Region", columns="Category",
        aggregade_method="sum", fill_value=0,
        title="Heatmap", width=None,
        data_id=None, data='{"records":[]}', meta_data={}, user_id=None,
    ):
        """Interactive pivot-table heatmap (imshow)."""
        try:
            message_dict = {"message": meta_data}
            validated = self._data_validate(data, message_dict)
            if validated["status"] != "success":
                md = validated["message_dict"]
                return {"status": "error", "response": {"meta_data": md, "data": json.dumps({"figure": ""})}, "message": md.get("message")}
            data = validated["data"]
            df = pd.DataFrame(data)
            col_labels = df[columns].unique()
            row_labels = df[index].unique()
            pivot_data = df.pivot_table(
                index=index, columns=columns, values=values,
                aggfunc=aggregade_method, fill_value=fill_value,
            )
            pivot_data = pivot_data.reindex(index=row_labels, columns=col_labels)
            fig = px.imshow(
                pivot_data,
                color_continuous_scale="Jet",
                labels=dict(y=index, x=columns, color=values),
                text_auto=True,
            )
            fig.update_layout(title=title, autosize=True, width=width)
            fig_return = self._to_html(fig)
            message_dict.update({"message": "Heatmap plotted"})
            return {"status": "success", "response": {"meta_data": message_dict, "data": json.dumps({"figure": fig_return}), "message": json.dumps(message_dict)}}
        except Exception as e:
            message_dict.update({"message": f"Error: {e}"})
            return {"status": "error", "response": {"meta_data": message_dict, "data": json.dumps({"figure": ""})}}

    def plot_multi_column_bar_graph(
        self,
        xLabel="Week",
        value_vars=["Total Slots Allocated by Region", "Containers Product Ordered by Region"],
        title="Slot Allocation", hover_data=[], barmode="group",
        data_id=None, data='{"records":[]}', meta_data={}, user_id=None,
    ):
        """Multi-series grouped/stacked bar graph (melted DataFrame)."""
        try:
            message_dict = {"message": meta_data}
            validated = self._data_validate(data, message_dict)
            if validated["status"] != "success":
                md = validated["message_dict"]
                return {"status": "error", "response": {"meta_data": md, "data": json.dumps({"figure": ""})}, "message": md.get("message")}
            records = validated["data"]
            df = pd.DataFrame(records)
            needed_columns = list(set([xLabel] + hover_data + value_vars))
            df = df[[c for c in needed_columns if c in df.columns]]
            df_melted = df.melt(
                id_vars=[xLabel] + [h for h in hover_data if h in df.columns],
                value_vars=[v for v in value_vars if v in df.columns],
                var_name="Group", value_name="Value",
            )
            fig = px.bar(
                df_melted, x=xLabel, y="Value", color="Group",
                barmode=barmode, title=title,
                hover_data=hover_data if hover_data else None,
            )
            fig.update_layout(
                title_font=dict(size=12), title_automargin=True,
                margin=dict(l=20, r=20, t=60, b=20),
                legend=dict(font=dict(size=10), orientation="h", x=0.5, xanchor="center", y=1, yanchor="bottom"),
                autosize=True, font=dict(size=9),
            )
            fig_return = self._to_html(fig)
            message_dict.update({"message": "Multi-column bar chart plotted"})
            return {"status": "success", "response": {"meta_data": message_dict, "data": json.dumps({"figure": fig_return}), "message": json.dumps(message_dict)}}
        except Exception as e:
            message_dict.update({"message": f"Error: {e}"})
            return {"status": "error", "response": {"meta_data": message_dict, "data": json.dumps({"figure": ""})}}

    def plot_flow_chart(
        self,
        data_id=None, data='{"records":[]}', meta_data={}, user_id=None,
    ):
        """
        Graphviz-rendered flowchart (PNG embedded as base64 HTML).

        Input JSON shape:
          {"records": {
              "nodes": [{"label":…, "shape":…, "style":…, "fillcolor":…, "fontcolor":…}, …],
              "edges": [{"start":…, "end":…, "label":…, "color":…, "penwidth":…}, …]
          }}
        """
        dot = graphviz.Digraph(format="png")
        G = nx.MultiDiGraph()
        message_dict = {"message": meta_data}
        if not data and not data_id:
            message_dict["message"] = "No data or data_id"
            return {"status": "error", "response": {"meta_data": message_dict, "data": json.dumps({"figure": ""})}}
        try:
            records = json.loads(data).get("records", {})
            if not records:
                message_dict["message"] = "No data"
                return {"status": "error", "response": {"meta_data": message_dict, "data": json.dumps({"figure": ""})}}
            states = records.get("edges", [])
            node_properties = records.get("nodes", [])
            for state in states:
                edge_label = str(state.get("label", ""))
                dot.edge(
                    state.get("start", ""), state.get("end", ""),
                    label=edge_label,
                    color=state.get("color", "black"),
                    penwidth=str(state.get("penwidth", 1)),
                )
                G.add_edge(
                    state.get("start", "").split(":")[0],
                    state.get("end", "").split(":")[0],
                    label=edge_label,
                )
            for node_p in node_properties:
                label = node_p.get("label", "")
                if label in G.nodes():
                    dot.node(
                        label,
                        shape=node_p.get("shape", "ellipse"),
                        style=node_p.get("style", "filled"),
                        fillcolor=node_p.get("fillcolor", "#bbbbbb"),
                        color=node_p.get("fontcolor", "black"),
                    )
            png_bytes = dot.pipe(format="png")
            encoded = base64.b64encode(png_bytes).decode("utf-8")
            plot_html = (
                f'<div><figure>'
                f'<img src="data:image/png;base64,{encoded}" alt="Flowchart">'
                f"</figure></div>"
            )
            fig_id = str(uuid.uuid4())[:8]
            fig_return = plot_html.replace("<div>", f'<div id="{fig_id}">')
            message_dict["message"] = "Flowchart plotted"
            return {"status": "success", "response": {"meta_data": message_dict, "data": json.dumps({"figure": fig_return}), "message": json.dumps(message_dict)}}
        except Exception as e:
            message_dict["message"] = f"Error: {e}"
            return {"status": "error", "response": {"meta_data": message_dict, "data": json.dumps({"figure": ""})}}

    def plot_flow_chart_plotly(
        self,
        data_id=None, data='{"records":[]}', meta_data={}, user_id=None,
    ):
        """
        Plotly/NetworkX-rendered flowchart (interactive HTML).
        Same input JSON shape as plot_flow_chart.
        """
        message_dict = {"message": meta_data}
        if not data and not data_id:
            message_dict["message"] = "No data or data_id"
            return {"status": "error", "response": {"meta_data": message_dict, "data": json.dumps({"figure": ""})}}
        try:
            records = json.loads(data).get("records", {})
            if not records:
                message_dict["message"] = "No data"
                return {"status": "error", "response": {"meta_data": message_dict, "data": json.dumps({"figure": ""})}}
            states = records.get("edges", [])
            node_properties = records.get("nodes", [])
            G = nx.MultiDiGraph()
            edge_labels = {}
            for state in states:
                start = state["start"].split(":")[0]
                end = state["end"].split(":")[0]
                label = state.get("label", "")
                G.add_edge(start, end)
                edge_labels[(start, end)] = label
            for node_p in node_properties:
                if node_p["label"] in G.nodes:
                    G.nodes[node_p["label"]].update(node_p)
            pos = nx.spring_layout(G, seed=42)
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color="black"),
                hoverinfo="none", mode="lines",
            )
            node_x, node_y, node_text, node_color = [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                info = G.nodes[node]
                node_text.append(str(info.get("label", node)))
                node_color.append(info.get("fillcolor", "#BBBBBB"))
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                marker=dict(size=40, color=node_color, line=dict(width=2, color="black")),
                text=node_text, textposition="middle center", hoverinfo="text",
            )
            elx, ely, elt = [], [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                elx.append((x0 + x1) / 2)
                ely.append((y0 + y1) / 2)
                elt.append(edge_labels.get((edge[0], edge[1]), ""))
            edge_label_trace = go.Scatter(
                x=elx, y=ely, mode="text", text=elt,
                textposition="top center", hoverinfo="none", showlegend=False,
            )
            fig = go.Figure(
                data=[edge_trace, node_trace, edge_label_trace],
                layout=go.Layout(
                    showlegend=False, hovermode="closest",
                    margin=dict(b=0, l=0, r=0, t=0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    plot_bgcolor="white",
                ),
            )
            fig_id = str(uuid.uuid4())[:8]
            from plotly.io import to_html as _to_html_plotly
            fig_html = _to_html_plotly(
                fig, include_plotlyjs="cdn", full_html=False, div_id=fig_id
            )
            message_dict["message"] = "Plotly flowchart plotted"
            return {"status": "success", "response": {"meta_data": message_dict, "data": json.dumps({"figure": fig_html}), "message": json.dumps(message_dict)}}
        except Exception as e:
            message_dict["message"] = f"Error: {e}"
            return {"status": "error", "response": {"meta_data": message_dict, "data": json.dumps({"figure": ""})}}


# ═════════════════════════════════════════════════════════════════════════════
# §3  COVID-19 DATASET — LOADING & FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

COVID_URL = (
    "https://raw.githubusercontent.com/owid/covid-19-data/master/"
    "public/data/owid-covid-data.csv"
)

print("=" * 70)
print("  COVID-19 Global Data  —  DataInspector + PlottingMethods Demo")
print("=" * 70)
print("\n⬇️  Downloading COVID-19 dataset from Our World in Data …")
response = requests.get(COVID_URL, timeout=60)
response.raise_for_status()

raw_df = pd.read_csv(
    io.StringIO(response.text),
    na_values=["", " ", "N/A", "NA", "NULL", "null", "?"],
    low_memory=False,
)
print(f"✅  Loaded {raw_df.shape[0]:,} rows × {raw_df.shape[1]} columns")

# ── Column selection ──────────────────────────────────────────────────────────
KEEP_COLS = [
    "iso_code", "continent", "location", "date",
    "total_cases", "new_cases",
    "total_deaths", "new_deaths",
    "total_vaccinations", "people_fully_vaccinated",
    "population", "population_density",
    "median_age", "gdp_per_capita",
    "hospital_beds_per_thousand", "life_expectancy",
    "human_development_index", "stringency_index",
]

# Drop continent/world aggregates
country_df = raw_df[
    ~raw_df["iso_code"].str.startswith("OWID_", na=True)
].copy()

# Latest snapshot per country
country_df["date"] = pd.to_datetime(country_df["date"])
latest_df = (
    country_df.sort_values("date")
    .groupby("location", as_index=False)
    .last()
)[KEEP_COLS]

# Derived ratio columns
latest_df["case_fatality_rate"] = (
    latest_df["total_deaths"] / latest_df["total_cases"]
).round(4)
latest_df["vaccination_rate"] = (
    latest_df["people_fully_vaccinated"] / latest_df["population"]
).round(4)
latest_df["cases_per_million"] = (
    (latest_df["total_cases"] / latest_df["population"]) * 1_000_000
).round(1)

print(
    f"\n📊 Analytical slice ready: {latest_df.shape[0]} countries, "
    f"{latest_df.shape[1]} columns"
)
display(latest_df.head())


# ═════════════════════════════════════════════════════════════════════════════
# §4  DataInspector PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

# ── 4.1  Initialise and inject DataFrame ─────────────────────────────────────
print("\n" + "─" * 60)
print("4.1  Initialising DataInspector")
print("─" * 60)
inspector = DataInspector()
inspector.df = latest_df.reset_index(drop=True)
inspector.df["count"] = 1
inspector.get_summary()

# ── 4.2  Missing data audit ───────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.2  Missing Data Audit")
print("─" * 60)
inspector.show_missing_data()

# ── 4.3  Imputation ───────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.3  Imputing Missing Values")
print("─" * 60)
inspector.handle_missing_values(strategy="median")
inspector.handle_missing_values(columns=["continent"], strategy="mode")

remaining_nulls = inspector.df.isnull().sum()
remaining_nulls = remaining_nulls[remaining_nulls > 0]
if remaining_nulls.empty:
    print("✅ No missing values remain.")
else:
    print("Remaining nulls:")
    print(remaining_nulls)

# ── 4.4  Deduplication ───────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.4  Removing Duplicates")
print("─" * 60)
inspector.remove_duplicates()

# ── 4.5  Column ranges ───────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.5  Column Ranges")
print("─" * 60)
inspector.column_details()

# ── 4.6  Categorical summary ─────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.6  Categorical Summary")
print("─" * 60)
inspector.get_categorical_summary()

# ── 4.7  Outlier detection ───────────────────────────────────────────────────
NUMERIC_FOCUS = [
    "total_cases", "total_deaths", "cases_per_million",
    "case_fatality_rate", "vaccination_rate",
    "gdp_per_capita", "median_age",
]
print("\n" + "─" * 60)
print("4.7  Outlier Detection (IQR)")
print("─" * 60)
inspector.handle_outliers(columns=NUMERIC_FOCUS, find_and_delete=False)

# ── 4.8  Univariate plots ─────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.8  Univariate Visualisations")
print("─" * 60)
inspector.plot_numerical(NUMERIC_FOCUS)
inspector.plot_categorical(["continent"])

# ── 4.9  Bivariate relationships ─────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.9  Bivariate Relationship Plots")
print("─" * 60)
inspector.plot_relationship("gdp_per_capita",     "vaccination_rate")
inspector.plot_relationship("median_age",         "case_fatality_rate")
inspector.plot_relationship("population_density", "cases_per_million")
inspector.plot_relationship("continent",          "vaccination_rate")
inspector.plot_relationship("continent",          "case_fatality_rate")
inspector.plot_relationship("human_development_index", "cases_per_million")
inspector.plot_relationship("life_expectancy",    "total_deaths")

# ── 4.10  Correlation heatmaps ───────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.10  Pearson Correlation Heatmap")
print("─" * 60)
inspector.plot_numerical_correlation()

print("\n" + "─" * 60)
print("4.11  Cramér's V Categorical Association Heatmap")
print("─" * 60)
inspector.plot_categorical_correlation()

print("\n" + "─" * 60)
print("4.12  Numeric ↔ Categorical Association Table")
print("─" * 60)
assoc_table = inspector.correlate_num_to_cat()
display(assoc_table.sort_values("Correlation", ascending=False))

print("\n" + "─" * 60)
print("4.13  Unified Association Heatmap (All Column Types)")
print("─" * 60)
inspector.plot_all_associations_heatmap()

# ── 4.14  Statistical stationarity tests ─────────────────────────────────────
print("\n" + "─" * 60)
print("4.14  MANOVA Mean Homogeneity Test")
print("─" * 60)
inspector.test_constant_mean(chunks=8)

print("\n" + "─" * 60)
print("4.15  Box's M Covariance Homogeneity Test")
print("─" * 60)
inspector.test_constant_covariance(chunks=5)

print("\n" + "─" * 60)
print("4.16  Multivariate Ljung-Box Row Independence Test")
print("─" * 60)
inspector.test_row_independence()

# ── 4.17  Probabilistic modelling ────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.17  Joint Multivariate Normal Model (Micro-Scale)")
print("─" * 60)
micro_model = inspector.estimate_joint_normal()

print("\n" + "─" * 60)
print("4.18  CLT Macro-Scale Sampling Distribution")
print("─" * 60)
macro_model = inspector.instantiate_macro_clt_distribution()

# ── 4.19  PCA dashboard ───────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.19  Principal Component Analysis (PCA) Dashboard")
print("─" * 60)
pca_results = inspector.compute_empirical_pca()
print("\nTop-3 PC explained variance:")
for i, v in enumerate(pca_results["explained_variance_ratio"][:3]):
    print(f"  PC{i+1}: {v*100:.1f}%")

# ── 4.20  Factor Analysis ─────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.20  Factor Analysis (k=3 latent factors)")
print("─" * 60)
fa_results = inspector.compute_empirical_fa(k=3)

# ── 4.21  Export ──────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("4.21  Export Cleaned Dataset")
print("─" * 60)
inspector.export_cleaned_data(filename="covid19_cleaned.csv")


# ═════════════════════════════════════════════════════════════════════════════
# §5  PlottingMethods SHOWCASE  (8 chart types)
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  §5  PlottingMethods Showcase")
print("=" * 70)

plotter = PlottingMethods()

# ── 5.1  Bar chart: top-20 countries by cases per million ────────────────────
print("\n5.1  Bar Chart — Top-20 Countries by Cases per Million")
top20 = (
    inspector.df.nlargest(20, "cases_per_million")[
        ["location", "cases_per_million", "continent"]
    ].rename(columns={"location": "country"})
)
bar_result = plotter.plot_bar_chart(
    x="country",
    y="cases_per_million",
    color="continent",
    title="Top 20 Countries — COVID-19 Cases per Million",
    barmode="group",
    data=top20.to_json(orient="records"),
)
plotter.display_image(bar_result)

# ── 5.2  Pie chart: continent share of total cases ───────────────────────────
print("\n5.2  Pie Chart — Continent Share of Total Cases")
continent_cases = inspector.df.groupby("continent", as_index=False)["total_cases"].sum()
pie_result = plotter.plot_pie_chart(
    names="continent",
    values="total_cases",
    title="Share of Total COVID-19 Cases by Continent",
    data=continent_cases.to_json(orient="records"),
)
plotter.display_image(pie_result)

# ── 5.3  Histogram: distribution of vaccination rates ────────────────────────
print("\n5.3  Histogram — Vaccination Rate Distribution")
hist_data = inspector.df[["vaccination_rate"]].dropna()
hist_result = plotter.plot_histogram(
    x="vaccination_rate",
    title="Distribution of Vaccination Rates Across Countries",
    data=hist_data.to_json(orient="records"),
)
plotter.display_image(hist_result)

# ── 5.4  Heat map: mean CFR by continent × income tier ───────────────────────
print("\n5.4  Heat Map — Mean CFR by Continent × Income Tier")
inspector.df["income_tier"] = pd.qcut(
    inspector.df["gdp_per_capita"],
    q=4,
    labels=["Low", "Lower-Mid", "Upper-Mid", "High"],
    duplicates="drop",
)
heatmap_data = inspector.df[["continent", "income_tier", "case_fatality_rate"]].dropna()
heat_result = plotter.plot_heat_map(
    values="case_fatality_rate",
    index="continent",
    columns="income_tier",
    aggregade_method="mean",
    title="Mean Case Fatality Rate — Continent × Income Tier",
    data=heatmap_data.to_json(orient="records"),
)
plotter.display_image(heat_result)

# ── 5.5  Sankey diagram: continent → income tier ─────────────────────────────
print("\n5.5  Sankey Diagram — Continent → Income Tier Flow")
sankey_data = (
    inspector.df[["continent", "income_tier", "count"]]
    .dropna()
    .groupby(["continent", "income_tier"], as_index=False)["count"].sum()
)
sankey_result = plotter.plot_sankey_diagram(
    source_column="continent",
    target_column="income_tier",
    values="count",
    title="Country Flow: Continent → Income Tier",
    data=sankey_data.to_json(orient="records"),
)
plotter.display_image(sankey_result)

# ── 5.6  Sunburst: top-40 countries by total cases ───────────────────────────
print("\n5.6  Sunburst — Top-40 Countries Hierarchy")
top40 = (
    inspector.df.nlargest(40, "total_cases")[
        ["continent", "income_tier", "location", "total_cases"]
    ]
    .dropna()
    .rename(columns={"location": "country"})
)
sunburst_result = plotter.plot_simple_sunburst_graph(
    path=["continent", "income_tier", "country"],
    values="total_cases",
    title="COVID-19 Total Cases: Continent → Income → Country (Top 40)",
    data=top40.to_json(orient="records"),
)
plotter.display_image(sunburst_result)

# ── 5.7  Treemap: same hierarchy, different geometry ─────────────────────────
print("\n5.7  Treemap — Top-40 Countries Hierarchy")
treemap_result = plotter.plot_tree_map(
    path=["continent", "income_tier", "country"],
    values="total_cases",
    title="COVID-19 Total Cases Treemap: Continent → Income → Country",
    data=top40.to_json(orient="records"),
)
plotter.display_image(treemap_result)

# ── 5.8  Plotly flow chart: analytical pipeline DAG ──────────────────────────
print("\n5.8  Plotly Flowchart — Analytical Pipeline DAG")
flow_data = {
    "records": {
        "nodes": [
            {"label": "Raw Data",          "shape": "box",     "style": "filled", "fillcolor": "#4A90D9", "fontcolor": "white"},
            {"label": "Data Cleaning",     "shape": "box",     "style": "filled", "fillcolor": "#7ED321", "fontcolor": "white"},
            {"label": "EDA",               "shape": "ellipse", "style": "filled", "fillcolor": "#F5A623", "fontcolor": "white"},
            {"label": "PCA / FA",          "shape": "ellipse", "style": "filled", "fillcolor": "#9B59B6", "fontcolor": "white"},
            {"label": "Statistical Tests", "shape": "diamond", "style": "filled", "fillcolor": "#E74C3C", "fontcolor": "white"},
            {"label": "MVN Model",         "shape": "box",     "style": "filled", "fillcolor": "#1ABC9C", "fontcolor": "white"},
            {"label": "Anomaly Scoring",   "shape": "ellipse", "style": "filled", "fillcolor": "#E67E22", "fontcolor": "white"},
        ],
        "edges": [
            {"start": "Raw Data",          "end": "Data Cleaning",     "label": "ingest",        "color": "#555555", "penwidth": 2},
            {"start": "Data Cleaning",     "end": "EDA",               "label": "sanitised",     "color": "#555555", "penwidth": 2},
            {"start": "EDA",               "end": "Statistical Tests", "label": "insights",      "color": "#F5A623", "penwidth": 1.5},
            {"start": "EDA",               "end": "PCA / FA",          "label": "dimension ↓",   "color": "#7ED321", "penwidth": 1.5},
            {"start": "Statistical Tests", "end": "MVN Model",         "label": "IID verified",  "color": "#E74C3C", "penwidth": 2},
            {"start": "PCA / FA",          "end": "MVN Model",         "label": "latent space",  "color": "#9B59B6", "penwidth": 2},
            {"start": "MVN Model",         "end": "Anomaly Scoring",   "label": "log-pdf score", "color": "#1ABC9C", "penwidth": 2.5},
        ],
    }
}
flow_result = plotter.plot_flow_chart_plotly(data=json.dumps(flow_data))
plotter.display_image(flow_result)

print("\n" + "=" * 70)
print("  🎉  Full COVID-19 demo complete!")
print("=" * 70)
