-- Queries used by the SQL Explorer page. Table: customers (SQLite).
-- Format: "-- name: <title>" followed by one query.

-- name: Churn rate by contract type
SELECT Contract,
       COUNT(*) AS customers,
       SUM(Churn) AS churned,
       ROUND(AVG(Churn) * 100, 1) AS churn_pct
FROM customers
GROUP BY Contract
ORDER BY churn_pct DESC;

-- name: Churn rate by payment method
SELECT PaymentMethod,
       COUNT(*) AS customers,
       ROUND(AVG(Churn) * 100, 1) AS churn_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_pct DESC;

-- name: Churn rate by tenure group
SELECT CASE WHEN tenure <= 12 THEN '1) 0-12 months'
            WHEN tenure <= 24 THEN '2) 13-24 months'
            WHEN tenure <= 48 THEN '3) 25-48 months'
            ELSE '4) 49+ months' END AS tenure_group,
       COUNT(*) AS customers,
       ROUND(AVG(Churn) * 100, 1) AS churn_pct
FROM customers
GROUP BY tenure_group
ORDER BY tenure_group;

-- name: Average monthly charges - churned vs retained
SELECT CASE WHEN Churn = 1 THEN 'Churned' ELSE 'Retained' END AS status,
       COUNT(*) AS customers,
       ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges,
       ROUND(AVG(tenure), 1) AS avg_tenure_months
FROM customers
GROUP BY Churn;

-- name: Churn by internet service and tech support
SELECT InternetService, TechSupport,
       COUNT(*) AS customers,
       ROUND(AVG(Churn) * 100, 1) AS churn_pct
FROM customers
GROUP BY InternetService, TechSupport
HAVING customers > 50
ORDER BY churn_pct DESC;

-- name: High-risk segment (month-to-month, tenure under 12, fiber)
SELECT customerID, tenure, MonthlyCharges, PaymentMethod, Churn
FROM customers
WHERE Contract = 'Month-to-month'
  AND tenure < 12
  AND InternetService = 'Fiber optic'
ORDER BY MonthlyCharges DESC
LIMIT 50;

-- name: Monthly revenue lost to churn by contract
SELECT Contract,
       ROUND(SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END), 0) AS monthly_revenue_lost,
       ROUND(SUM(MonthlyCharges), 0) AS total_monthly_revenue,
       ROUND(100.0 * SUM(CASE WHEN Churn = 1 THEN MonthlyCharges ELSE 0 END) / SUM(MonthlyCharges), 1) AS pct_lost
FROM customers
GROUP BY Contract
ORDER BY monthly_revenue_lost DESC;

-- name: Senior citizens vs others
SELECT CASE WHEN SeniorCitizen = 1 THEN 'Senior' ELSE 'Non-senior' END AS segment,
       COUNT(*) AS customers,
       ROUND(AVG(Churn) * 100, 1) AS churn_pct,
       ROUND(AVG(MonthlyCharges), 2) AS avg_monthly_charges
FROM customers
GROUP BY SeniorCitizen;
