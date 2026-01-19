
# Forecasting Water Levels in Hydropower Reservoirs 🌊⚡
### CS313 - Data Mining Project | Group 10

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)
![Framework](https://img.shields.io/badge/Framework-Scikit_Learn%20|%20XGBoost%20|%20TensorFlow-orange)

## 📖 Introduction

This project focuses on **forecasting upstream water levels** for critical hydropower reservoirs in Vietnam. Accurate water level prediction plays a pivotal role in supporting operational decision-making, ensuring dam safety, flood control, and optimizing power generation output.



We apply advanced **Data Mining** and **Machine Learning** techniques to process multivariate time-series data, combining hydrological parameters with meteorological conditions.

### 📍 Research Scope
The project targets three major reservoirs representing distinct climatic regions of Vietnam:
1.  **Hoa Binh Hydropower Plant** (Da River - Northern Region)
2.  **Song Ba Ha Hydropower Plant** (Ba River - Central Region)
3.  **Tri An Hydropower Plant** (Dong Nai River - Southern Region)

---

## 👥 Team & Instructor

**Supervisor:** Dr. Vo Le Nguyen Duy

**Team:** Group 10 - Class CS313.Q12

| No. | Student ID | Full Name | Role & Key Contributions |
|:---:|:---:|:---|:---|
| 1 | **23521440** | **Huỳnh Nhật Thanh** | Data Collection, EDA, Preprocessing, Feature Engineering, Full Model Training (ML + DL), Reporting & Slides (27%). |
| 2 | **23521441** | **Lê Đức Thanh** | Hydrological Data Crawling, EDA, Preprocessing, Model Evaluation (CNN+LSTM, XGBoost, Decision Tree), **Web Demo Development** (27%). |
| 3 | **23521443** | **Lê Ngọc Thanh** | Weather Data Crawling & Preprocessing, Deep Learning Model Evaluation, Presentation Slides (23%). |
| 4 | **23521454** | **Nguyễn Tiến Thanh** | Weather Data Crawling, Machine Learning Model Evaluation, Web Demo Development (XGBoost integration) (23%). |

---

## 🌐 Web Application & Project Structure

We have developed an interactive web application using **Streamlit** to visualize data and demonstrate real-time forecasting capabilities.

### Directory Structure
The codebase is organized into modular components to separate raw data, processed assets, and the application logic:

```text
hydro_streamlit_demo2
├── app.py                      # 🏠 Main Application (Forecasting Dashboard)
├── pages
│   └── 1_Raw_Data_Viewer.py    # 📊 Sub-page: Raw Data Exploration & Visualization
├── assets                      # 📦 Resources (Serialized Models, Processed Data)
│   ├── HoaBinh_merged.csv      # Processed data for Hoa Binh
│   ├── SongBaHa_merged.csv     # Processed data for Song Ba Ha
│   ├── TriAn_merged.csv        # Processed data for Tri An
│   ├── br_hoabinh.pkl          # 🧠 Bayesian Ridge Model (Hoa Binh)
│   ├── rf_songbaha.pkl         # 🧠 Random Forest Model (Song Ba Ha)
│   ├── dtree_trian.pkl         # 🧠 Decision Tree Model (Tri An)
│   └── ... (Feature Config JSONs & Raw CSVs)
└── requirements.txt            # ⚙️ Python Dependencies

```

### Deployment Details

Based on our experimental results (R² and RMSE scores), we deployed the most effective model for each specific dam in the demo:

* **Hoa Binh ➡️ Bayesian Ridge (`br_hoabinh.pkl`)**
* *Reason:* Best suited for the Northern region's complex seasonality with less noise.


* **Song Ba Ha ➡️ Random Forest (`rf_songbaha.pkl`)**
* *Reason:* Handles the high fluctuation and extreme peak flows of the Central region effectively.


* **Tri An ➡️ Decision Tree (`dtree_trian.pkl`)**
* *Reason:* Provides fast inference and high interpretability for the stable hydrological regime of the Southern region.



---

## 📊 Dataset Overview

The input dataset consists of historical time-series data with the following key variables:

* **Hydrological Data (Source: EVN):**
* *Features:* Upstream water level, Inflow to reservoir, Total discharge (via turbines & spillways).


* **Meteorological Data (Source: Open-Meteo API):**
* *Features:* Precipitation, Temperature, Relative Humidity, Cloud cover.


* **Modeling Approach:** A **28-day sliding window** is used to predict the water level at the target timestamp.

---

## 🛠️ Methodology

### 1. Preprocessing & Feature Engineering

* **Imputation Strategy:**
* *Water Level:* Month-wise statistical imputation to preserve seasonality.
* *Inflow:* **Event-Conditional Imputation**. We separate logic for "Normal Months" vs. "Flood/Event Months" to preserve critical peak signals.


* **Feature Engineering:**
* **`is_event_month`**: Flag created based on volatility (≥150%), climatological deviation (Ratio ≥1.5), and extreme value frequency.
* **Seasonality**: Explicit encoding for Rainy Season (May-Nov) vs. Dry Season (Dec-Apr).
* **Temporal Features**: Lags and Rolling statistics.



### 2. Modeling

We conducted extensive experiments comparing Machine Learning and Deep Learning approaches:

* **Machine Learning:** Linear Regression, Bayesian Ridge, Decision Tree, Random Forest, XGBoost.
* **Deep Learning:** CNN-LSTM and CNN-GRU (for spatial-temporal feature extraction).

---

## 📈 Key Results

Performance summary on the Test set:

| Reservoir | Best Performing Model | R² Score | RMSE | Insight |
| --- | --- | --- | --- | --- |
| **Hoa Binh** | **Bayesian Ridge / CNN+GRU** | > 0.94 | ~2.16 | Models effectively captured the complex seasonality of the North. |
| **Song Ba Ha** | **XGBoost / CNN+GRU** | > 0.96 | ~0.16 | High accuracy achieved despite the dam's volatile discharge regulation. |
| **Tri An** | **Random Forest / XGBoost** | > 0.99 | ~0.03 | The stable data distribution allowed Tree-based models to achieve near-perfect scores. |

> *Note: Detailed metrics, EDA charts, and Actual vs. Predicted graphs are available in the attached report.*

---

## 🚀 Installation & Usage

To run the Streamlit Demo locally:

1. **Clone the repository:**
```bash
git clone [https://github.com/leducthanh041/CS313-DoAn.git](https://github.com/leducthanh041/CS313-DoAn.git)
cd hydro_streamlit_demo2

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Launch the App:**
```bash
streamlit run app.py

```


*The app will open in your browser at `http://localhost:8501*`

---

<p align="center">
<i>Created with ❤️ by Group 10 - CS313.Q12</i>
</p>