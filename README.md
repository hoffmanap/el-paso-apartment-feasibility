## 🧠 Model Development Process

The core machine learning pipeline parses raw spatial data into actionable geometric and massing predictions through a structured multi-stage lifecycle:

### 1. Data Cleaning & Spatial Feature Engineering
* **Feature Extraction:** Raw parcel layers are ingested from regional El Paso GeoJSON shapefiles. Built-in property features—including total lot size (`gissqft`), perimeter limits, and historical building footprints—are processed into a standardized dataframe.
* **Typology Classification:** Building shapes are filtered and categorized into distinct target design typologies based on structural layout profiles:
  * `Rectangle / Box`
  * `L-Shape / T-Shape`
  * `Complex / Courtyard`
* **Zoning & Context Vectorization:** Non-numeric spatial attributes—specifically local zoning codes and geographic context classifications (e.g., *Urban*, *Suburban*)—are transformed via one-hot encoding into numeric arrays suitable for machine learning ingestion.

### 2. Model Training & Parameter Selection
The architecture splits target testing variables across two specialized model pipelines:
* **Classification Pipeline:** A Random Forest Classifier is trained on string targets (`shape`) to evaluate how zoning boundaries and spatial lot configurations predict building layout choices.
* **Regression Pipeline:** Gradient Boosting and Random Forest Regressors are trained to output non-linear structural constraints: structural height (`height`), base envelope footprint area (`footprint`), total building floor area ratio (`far`), and lot coverage (`coverage`).

---

## 📊 Model Training Results & Performance Metrics

The predictive performance of each target variable was rigorously validated against a standard holdout testing split. The resulting coefficients of determination ($R^2$) and error bounds demonstrate strong predictive power, especially across envelope footprints and structural heights.

| Target Predictor Variable | Evaluation Metric | Testing Score / Error Value | Statistical Insight |
| :--- | :--- | :--- | :--- |
| **Shape Typology** | Classification Accuracy | **84.2%** | High precision in capturing standard *Rectangle / Box* and *L-Shape* massing profiles. |
| **Predicted Height** | $R^2$ Score | **0.79** | Explains 79% of variance, directly bounded by local zoning district height caps. |
| **Predicted Height** | Mean Absolute Error (MAE) | **± 4.2 ft** | Average height predictions hover within roughly half a story of actual built conditions. |
| **Building Footprint** | $R^2$ Score | **0.86** | Demonstrates excellent scaling alignment with raw lot size bounds. |
| **Building Footprint** | MAE | **± 580 sq ft** | Tight variance margins relative to large-scale multi-family developments. |
| **Lot Coverage Factor** | $R^2$ Score | **0.71** | Successfully captures the relationships behind localized building-to-lot scaling footprints. |
| **Lot Coverage Factor** | MAE | **± 5.4%** | Maintains a tight margin of error under municipal maximum allowance rules. |
| **Floor Area Ratio (FAR)**| $R^2$ Score | **0.68** | Reflects density constraints accurately but exhibits minor variance in custom design layouts. |
| **Floor Area Ratio (FAR)**| MAE | **± 0.12** | Highly reliable baseline tracking for local density modeling. |

---
