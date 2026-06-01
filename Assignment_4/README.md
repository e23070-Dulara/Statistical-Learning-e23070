# data-inspector-tool

A comprehensive Python toolkit for data cleaning, exploratory data analysis, statistical modelling, dimensionality reduction, and interactive visualisation — optimised for Google Colab. Built around two core classes: **DataInspector** and **PlottingMethods**, demonstrated on the Our World in Data COVID-19 global dataset.

---

## Features

**Intelligent Data Loading**
Automatically handles common null strings (`?`, `N/A`, `NULL`) and attempts auto-conversion of columns to their correct numeric types on upload.

**Comprehensive Inspection**
Quickly view data dimensions, column type breakdowns, numeric ranges, and statistical summaries for both numerical and categorical data.

**Automated Cleaning**
- Identify and impute missing values using `mean`, `median`, `mode`, or `constant` strategies.
- Remove exact duplicate rows.
- Detect and optionally delete outliers using IQR logic.
- Interactive row and column deletion by index or name.

**Advanced Scaling & Encoding**
- Numeric: Min-Max, Standard (Z-score), and Robust scaling.
- Categorical: One-Hot, Ordinal, Uniform, and MinMax-Ordinal encoding.
- Merged normalised DataFrame builder for ML-ready feature matrices.

**Interactive Visualisations (Plotly)**
- Horizontal violin + scatter + histogram panels per numeric column.
- Frequency bar charts for categorical columns.
- Auto-selected bivariate plots (scatter + OLS, box, grouped bar) based on column types.

**Deep Statistical Insights**
- Pearson correlation heatmap for numeric data.
- Cramér's V association heatmap for categorical data.
- Point-Biserial / Eta (ANOVA) numeric↔categorical association table.
- Unified association heatmap combining all column types.

**Statistical Stationarity & Independence Tests**
- MANOVA Wilks' Lambda test for mean homogeneity across sequential data blocks.
- Box's M-test for covariance homogeneity (homoscedasticity).
- Multivariate Ljung-Box Portmanteau test for row-to-row independence.

**Probabilistic Modelling**
- Micro-scale Joint Multivariate Normal (MVN) model via MLE with Bessel's correction.
- Macro-scale CLT sampling distribution of the mean vector.

**Dimensionality Reduction**
- Empirical PCA with Hotelling's T² and Q (SPE) statistics across all truncation boundaries — rendered as a 2×3 Plotly dashboard.
- Factor Analysis (varimax rotation) with communality / uniqueness diagnostics — rendered as a 4-panel Plotly dashboard.

**PlottingMethods Chart Suite**
Bar, Pie, Histogram, Heatmap, Sankey, Sunburst, Treemap, and interactive Plotly Flowchart (DAG).

---

## Installation

```bash
# Basic installation
pip install "git+https://github.com/dulara/data-inspector-tool.git"

# Install with plotting support (required for all visualisations)
pip install data-inspector-tool[plotting]

# Install with Google Colab utilities
pip install data-inspector-tool[colab]
```

---

## Quick Start (Use Cases)

The tool is optimised for use in **Google Colab**.

### 1. Data Loading & Cleaning

```python
from data_analysis import DataInspector

inspector = DataInspector()

# Upload a CSV interactively in Colab
inspector.upload_data()

# Impute missing numeric values with median; categorical with mode
inspector.handle_missing_values(strategy='median')
inspector.handle_missing_values(columns=['continent'], strategy='mode')

# Remove duplicate rows
inspector.remove_duplicates()

# Export cleaned data
inspector.export_cleaned_data(filename='cleaned_data.csv')
```

### 2. Exploratory Data Analysis

```python
# Summary of dimensions, column types, and first 20 rows
inspector.get_summary()

# Show only rows containing missing values
inspector.show_missing_data()

# Numeric ranges and categorical unique counts per column
inspector.column_details()

# Categorical deep-dive (unique, mode, frequency)
inspector.get_categorical_summary()
```

### 3. Univariate & Bivariate Visualisations

```python
# Violin + Scatter + Histogram panel for each numeric column
inspector.plot_numerical(['total_cases', 'vaccination_rate', 'gdp_per_capita'])

# Frequency bar chart for a categorical column
inspector.plot_categorical(['continent'])

# Auto-selected bivariate plot based on column types:
#   Num × Num  → scatter + OLS trendline
#   Cat × Num  → box plot with data points
#   Cat × Cat  → grouped bar chart
inspector.plot_relationship('gdp_per_capita', 'vaccination_rate')
inspector.plot_relationship('continent', 'case_fatality_rate')
```

### 4. Correlation & Association Mapping

```python
# Pearson heatmap (numeric only)
inspector.plot_numerical_correlation()

# Cramér's V heatmap (categorical only)
inspector.plot_categorical_correlation()

# Point-Biserial / Eta table (numeric ↔ categorical)
assoc_table = inspector.correlate_num_to_cat()

# Unified heatmap for all column types
inspector.plot_all_associations_heatmap()
```

### 5. Feature Engineering & Normalisation

```python
# Scale numeric columns (options: 'minmax', 'standard', 'robust')
normalized_numeric = inspector.extract_normalized_numeric_data(method='robust')

# Encode categorical columns (options: 'uniform', 'ordinal', 'onehot', 'minmax_ordinal')
encoded_cat = inspector.extract_normalized_categorical_data(method='onehot')

# Merge into a single ML-ready DataFrame
final_df = inspector.create_normalized_data_df()
```

### 6. Statistical Tests

```python
# MANOVA Wilks' Lambda — test for mean drift across sequential blocks
inspector.test_constant_mean(chunks=8)

# Box's M — test for covariance homoscedasticity
inspector.test_constant_covariance(chunks=5)

# Multivariate Ljung-Box — test for row-to-row serial dependency
inspector.test_row_independence()
```

### 7. Probabilistic Modelling & Dimensionality Reduction

```python
# Fit a joint Multivariate Normal model (micro-scale MLE)
micro_model = inspector.estimate_joint_normal()

# CLT macro-scale sampling distribution of the mean vector
macro_model = inspector.instantiate_macro_clt_distribution()

# PCA dashboard (loading matrix, eigenvalues, T², Q/SPE profiles)
pca_results = inspector.compute_empirical_pca()

# Factor Analysis dashboard — k latent factors, varimax rotation
fa_results = inspector.compute_empirical_fa(k=3)
```

### 8. Custom Chart Generation (PlottingMethods)

All `PlottingMethods` return a result dictionary containing Plotly HTML. Render it in Colab using `plotter.display_image(result)`.

```python
from data_analysis import PlottingMethods
plotter = PlottingMethods()

# Grouped bar chart
result = plotter.plot_bar_chart(
    x='location', y='cases_per_million',
    color='continent', barmode='group',
    title='Top 20 Countries — Cases per Million',
    data=top20_df.to_json(orient='records')
)
plotter.display_image(result)

# Donut pie chart
result = plotter.plot_pie_chart(
    names='continent', values='total_cases',
    hole=0.4, title='Share of Total Cases by Continent',
    data=continent_df.to_json(orient='records')
)
plotter.display_image(result)

# Histogram with custom bins
result = plotter.plot_histogram(
    x='vaccination_rate',
    bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    title='Vaccination Rate Distribution',
    data=vax_df.to_json(orient='records')
)
plotter.display_image(result)

# Pivot heatmap
result = plotter.plot_heat_map(
    values='case_fatality_rate', index='continent',
    columns='income_tier', aggregade_method='mean',
    title='Mean CFR — Continent × Income Tier',
    data=heatmap_df.to_json(orient='records')
)
plotter.display_image(result)

# Sankey diagram
result = plotter.plot_sankey_diagram(
    source_column='continent', target_column='income_tier',
    values='count', title='Country Flow: Continent → Income Tier',
    data=sankey_df.to_json(orient='records')
)
plotter.display_image(result)

# Sunburst chart
result = plotter.plot_simple_sunburst_graph(
    path=['continent', 'income_tier', 'country'],
    values='total_cases',
    title='COVID-19 Cases: Continent → Income → Country',
    data=top40_df.to_json(orient='records')
)
plotter.display_image(result)

# Treemap
result = plotter.plot_tree_map(
    path=['continent', 'income_tier', 'country'],
    values='total_cases',
    title='COVID-19 Cases Treemap',
    data=top40_df.to_json(orient='records')
)
plotter.display_image(result)

# Interactive Plotly flowchart (DAG)
result = plotter.plot_flow_chart_plotly(data=json.dumps(flow_data))
plotter.display_image(result)
```

---

## COVID-19 Demo (co.py)

The included `co.py` script runs a full end-to-end pipeline on the [Our World in Data COVID-19 dataset](https://github.com/owid/covid-19-data):

| Step | Description |
|------|-------------|
| §3 | Downloads dataset, filters country-level rows, takes latest snapshot per country, engineers `case_fatality_rate`, `vaccination_rate`, `cases_per_million` |
| §4 | Full DataInspector pipeline — cleaning → EDA → correlation → stationarity tests → MVN model → PCA → FA → export |
| §5 | PlottingMethods showcase across 8 chart types |

To run it, open a new Google Colab notebook and paste the contents of `co.py` into a single code cell.

---

## Project Structure

```
data-inspector-tool/
├── data_analysis/
│   ├── __init__.py
│   └── core.py          # DataInspector and PlottingMethods classes
├── co.py                # COVID-19 end-to-end demo script
├── pyproject.toml       # Project configuration and dependencies
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical computation, linear algebra |
| `pandas` | DataFrame operations |
| `scipy` | Statistical tests (chi², Ljung-Box, Box's M) |
| `scikit-learn` | Scalers, encoders, Factor Analysis |
| `pydantic` | Data validation |
| `networkx` | Graph construction for flowcharts |
| `graphviz` | Static flowchart rendering (PNG) |
| `requests` | Dataset download |
| `plotly` *(optional)* | All interactive visualisations |
| `google-colab` *(optional)* | File upload/download in Colab |

---

## License

MIT
