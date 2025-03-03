# **Retail Sales Forecasting – End-to-End Project Documentation**

This documentation provides a step-by-step explanation of the **Retail Sales Forecasting** project, from **data preprocessing** to **model evaluation and business insights**. This project aims to predict daily sales for different product families across multiple stores in Ecuador, considering external factors like holidays, promotions, and oil prices.

---

## **1. Problem Statement 📌**  
Retail sales forecasting is crucial for **inventory management, marketing strategies, and financial planning**. This project focuses on **predicting future sales** based on historical data and external influences such as holidays, promotions, and economic conditions.

### **Goal 🎯**  
- Predict **daily sales for each product family** at each store for the next 15 days.  
- Compare multiple **time series forecasting models** to find the most accurate one.  
- Provide **business insights** to optimize sales forecasting strategies.  

---

## **2. Dataset Overview 📊**  
We used multiple datasets to build an accurate forecasting model:

| Dataset               | Description                                           |
|----------------------|---------------------------------------------------|
| **train.csv**        | Historical daily sales data per product family per store. |
| **test.csv**         | Future dates where sales need to be predicted.  |
| **stores.csv**       | Store metadata (location, type, cluster). |
| **oil.csv**          | Daily oil prices, affecting Ecuador’s economy.  |
| **holidays_events.csv** | List of national, regional, and local holidays/events. |

---

## **3. Data Preprocessing 🛠️**  

### **3.1 Handling Missing Values**  
✅ **Oil Prices (`oil.csv`)** had missing values → **Filled using linear interpolation** to maintain smooth trends.  
```python
oil['dcoilwtico'] = oil['dcoilwtico'].interpolate(method='linear')
```
✅ Checked for **missing or negative sales values** in `train.csv`.  
```python
print("Missing Sales:", train['sales'].isna().sum())
print("Negative Sales:", (train['sales'] < 0).sum())
```

### **3.2 Merging Datasets**  
We combined multiple datasets into a single DataFrame to create a comprehensive feature set.  
```python
train = train.merge(stores, on='store_nbr', how='left')
train = train.merge(oil, on='date', how='left')
train = train.merge(holidays_events, on='date', how='left')
```
✅ **Why?**  
- Store metadata helps **understand sales differences** between store types.  
- Oil prices provide insights into **economic conditions affecting sales**.  
- Holiday information captures **special events that influence buying behavior**.  

### **3.3 Feature Engineering 🏗️**  
To improve model performance, we extracted new features from the data.  

#### **📅 Time-Based Features**  
Extracted key components from the date:  
```python
train['day'] = train['date'].dt.day
train['week'] = train['date'].dt.isocalendar().week
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train['day_of_week'] = train['date'].dt.dayofweek
```
✅ **Why?**  
- Helps the model capture **seasonality trends** (e.g., higher sales in December).  
- Weekday vs. weekend sales patterns.  

#### **📢 Event-Based Features**  
Created binary flags for holidays, promotions, and economic events.  
```python
train['is_holiday'] = train['type'].notna().astype(int)
train['is_promotion'] = (train['onpromotion'] > 0).astype(int)
train['is_payday'] = ((train['date'].dt.day == 15) |
                      (train['date'] + pd.offsets.MonthEnd(0))).astype(int)
train['earthquake_impact'] = (train['date'] == '2016-04-16').astype(int)
```
✅ **Why?**  
- Holidays and promotions **significantly impact sales**.  
- Paydays influence customer spending behavior.  

#### **📈 Lagged Features (Past Sales Trends)**  
To help the model learn from past sales data:  
```python
train['sales_lag7'] = train.groupby(['store_nbr', 'family'])['sales'].shift(7)
train['sales_lag30'] = train.groupby(['store_nbr', 'family'])['sales'].shift(30)
```
✅ **Why?**  
- Weekly and monthly patterns help predict future sales.  

---

## **4. Model Training 🤖**  
We trained five different forecasting models:  

| Model                  | Description  |
|------------------------|-------------|
| **Naïve Forecasting**  | Assumes future sales = previous sales. Baseline model. |
| **ARIMA**              | Traditional time series model for trends & seasonality. |
| **Random Forest**      | Tree-based model that captures non-linear relationships. |
| **XGBoost**            | Gradient boosting model for accurate predictions. |
| **LSTM**               | Deep learning model for sequential data. |

---

## **5. Model Evaluation 📉**  
We compared models using **four metrics**:  

| Model               | RMSE (↓)  | MAPE (↓)  | R² (↑)  |
|--------------------|----------|----------|------|
| **ARIMA**            | **1411.32** | **Extremely High** | **-0.06** (Very Poor) |
| **Random Forest**    | **431.90** | **High** | **0.90** |
| **XGBoost**         | **307.43** | **Lower than RF** | **0.95** (Best) |

✅ **XGBoost performed the best** because it captured non-linear relationships and handled external factors effectively.  

---

## **6. Visualization 📊**  
### **📍 Actual vs. Predicted Sales**  
```python
plt.plot(y_val.index, y_val, label="Actual Sales", color='blue')
plt.plot(y_val.index, xgb_preds, label="XGBoost Predictions", color='purple', linestyle='dashed')
plt.legend()
plt.show()
```
✅ **Why?**  
- Helps visually analyze model accuracy.  
- Shows if the model captures **seasonality, trends, and demand fluctuations**.  

---

## **7. Business Insights & Recommendations 💡**  
### **How External Factors Affected Sales?**  
📌 **Holidays & Events** → Major sales spikes during Christmas & promotions.  
📌 **Oil Prices** → Correlation with consumer spending trends.  
📌 **Promotions** → Short-term increase in sales, but varied by product category.  

### **Business Strategies for Improved Forecasting**  
✅ **Inventory Optimization** – Stock high-demand products before peak seasons.  
✅ **Dynamic Pricing** – Adjust prices based on oil price fluctuations.  
✅ **Targeted Promotions** – Use historical data to optimize marketing campaigns.  

---

## **8. Conclusion 🏆**  
- **XGBoost was the best model** for sales forecasting (RMSE: **307.43**, R²: **0.95**).  
- **External factors like holidays & promotions significantly influenced sales**.  
- **Future work:** Hyperparameter tuning, adding deep learning models (LSTMs).  

🚀 **Next Steps:** Deploy XGBoost as a real-time forecasting solution!  

