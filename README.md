# El Paso Apartment Design & Feasibility AI 🏢

**Live Application Website:** [apartmentpredictor.streamlit.app](https://apartmentpredictor.streamlit.app/)

---

## 📋 Executive Overview
The **El Paso Apartment Design & Feasibility AI platform** is a cloud-hosted, predictive geospatial application engineered to analyze, evaluate, and optimize multifamily real estate development across El Paso, Texas. By bridging building footprint and parcels datasets with supervised machine learning models, this platform transforms traditional, manual zoning and parcel analysis into a dynamic, data-driven optimization workflow. 

Using this model can help answer the question of the elusive question of "neighborhood character" by utilizing localized use characteristics, and historical construction era to map local multifamily capacity, run site-specific envelope massing predictions, and visualize structural portfolios in an interactive, hardware-accelerated 3D environment.

---

## 🗃️ Data Engineering & Baseline Features
The system’s predictive intelligence is built upon a regional geospatial pipeline compiled from local El Paso municipal parcel boundaries, building footprints, and municipal tax rolls. The training dataset consists of multi-variable spatial matrices encompassing diverse parcel features:
* **Geospatial & Lot Scale Profile:** Ingestion of precise parcel boundaries to track geographic variables, building footprints, and raw land mass area via Geographical Information Systems (`ll_gissqft`).
* **Municipal Code Controls:** Categorization and one-hot encoding of local zoning classifications (`zoning`) to establish allowable property usage envelopes.
* **Property Descriptors:** Inclusion of unit capacity attributes (`ll_address_count`), structural layout classifications (`lbcs_structure_desc`), and construction year metrics (`yr_blt`) to capture development eras.

---

## 🧠 Machine Learning & The Regression Pipeline
To generate reliable building envelope and design typology predictions, a multi-stage supervised learning pipeline was constructed using Python’s `scikit-learn` framework. 

### 1. Feature Vectorization
Categorical spatial and structural inputs (such as specific zoning districts and structure profiles) were converted into dense, numeric arrays via one-hot encoding, matching them seamlessly with continuous variables like lot area.

### 2. Model Training & Architecture
The predictive engine splits targeting metrics into two primary algorithms:
* **Classification Engine:** A Random Forest Classifier evaluates non-linear zoning constraints and parcel ratios to map target design configurations, achieving an **84.2% classification accuracy** when predicting layout selections.
* **Regression Engine:** Gradient Boosting and Random Forest Regressors handle the continuous target scaling variables—specifically structural square footage (`sqfeet`), spatial densities (`density (units per acre)`), and inferred physical heights.

### 3. Validation & Performance Scopes
The regressors were evaluated against a standard 20% holdout testing split. The models demonstrated high predictive accuracy, yielding an **$R^2$ score of 0.86** for building footprints (MAE of ±580 sq ft) and an **$R^2$ score of 0.79** for predicted heights, successfully mapping how municipal code boundaries actively restrict physical built conditions.

---

## 🛠️ Cloud Deployment & Architectural Logic
The finished machine learning architecture is compiled into a lightweight, fully responsive front-end dashboard using the **Streamlit** framework. The application utilizes a customized, self-healing dynamic data layer designed to cleanly map and parse GeoJSON spatial components on the fly without database errors. 

For spatial visual rendering, the application embeds a **Folium mapping engine** to handle geographic popup records, alongside **Plotly Express** timeline tools that dynamically track density shifts and subdivision scales across historical El Paso construction decades. Finally, the user interface uses **Plotly Graph Objects** to project a 3D building massing envelope calculated directly from the filtered portfolio's structural dimensions, delivering a complete, interactive, and predictive real estate analytics engine.

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
