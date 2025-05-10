import streamlit as st
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import joblib

st.set_page_config(page_title="Financial Aid Estimator for Fulbright Students", layout="centered")

st.title("🎓 Fulbright University Vietnam Financial Aid Estimator")
st.markdown("This platform is exclusively designed for Fulbright applicants to estimate what **percentage of financial aid** you may receive based on your household data. Please fill in the form below to get your final result:")

TUITION_FEE = 500_000_000  # 500 million VND

# Define exchange rate
EXCHANGE_RATE = 468.54  # 1 PHP = 468.54 VND

# User inputs (in VND)
famsize_ip = st.slider("Family Size", 1, 15, 4)
income_total_ip = st.number_input("Total Monthly Household Income (VND)", value=18_000_000)
cashrep_ip = st.number_input("Monthly Cash Remittances (VND)", value=0)
rentals_nonagri_ip = st.number_input("Monthly Non-Agricultural Rental Income (VND)", value=11_000_000)
income_ea_ip = st.number_input("Monthly Earnings from Economic Activities (VND)", value=0)
interest_ip = st.number_input("Monthly Interest Income (VND)", value=0)
pension_ip = st.number_input("Monthly Pension Income (VND)", value=0)
dividends_ip = st.number_input("Monthly Dividends (VND)", value=34_000_000)

food_ip = st.number_input("Monthly Food Expenses (VND)", value=7_000_000)
clothing_ip = st.number_input("Monthly Clothing Expenses (VND)", value=3_000_000)
housing_ip = st.number_input("Monthly Housing Expenses (VND)", value=15_000_000)
health_ip = st.number_input("Monthly Health Expenses (VND)", value=3_000_000)
transport_ip = st.number_input("Monthly Transport Expenses (VND)", value=4_000_000)
communication_ip = st.number_input("Monthly Communication Expenses (VND)", value=1_000_000)
recreation_ip = st.number_input("Monthly Recreation Expenses (VND)", value=6_000_000)
education_ip = st.number_input("Monthly Education Expenses (VND)", value=10_000_000)
misc_ip = st.number_input("Monthly Miscellaneous Expenses (VND)", value=2_000_000)

dur_furniture_ip = st.number_input("Monthly Durable Furniture (VND)", value=1_000_000)
cash_loan_ip = st.number_input("Monthly Cash Loan Payments (VND)", value=0)
app_install_ip = st.number_input("Monthly Appliance Installments (VND)", value=0)
veh_install_ip = st.number_input("Monthly Vehicle Installments (VND)", value=0)
residence_ip = st.radio("Residence Type", ["Urban", "Rural"])
rural_ip = 1 if residence_ip == "Rural" else 0

# Load dataset (values in PHP)
df = pd.read_csv("ph_households_vF.csv")
df['cashrep'] = df['cashrep_abroad'] + df['cashrep_domestic']
df['rural'] = (df['residence'] == 'Rural').astype(int)
df = df.drop(['cashrep_abroad', 'cashrep_domestic', 'residence'], axis=1)
print(df.describe())
print(df.isnull().sum())

# Define variables
var_names = ['famsize', 'income_total', 'cashrep', 'rentals_nonagri', 
             'income_ea', 'interest', 'pension', 'dividends', 'food',
             'clothing', 'housing', 'health', 'transport', 'communication',
             'recreation', 'education', 'misc', 'dur_furniture', 'cash_loan',
             'app_install', 'veh_install', 'rural']

# Create binary rural column (Rural = 1, Urban = 0)
df = df.drop(columns=['ID', 'province', 'income_reg', 'income_ses'])

X = df[var_names]

# Verify dimensions
n, p = X.shape 
if p != len(var_names):
    raise ValueError(f"Mismatch: X has {p} columns, but var_names has {len(var_names)} elements")
print(f"Size of X: [{n}, {p}]")

# If X is a Pandas DataFrame:
nan_counts = X.isna().sum(axis=0)       # Series: number of NaNs per column
row_counts = X.notna().sum(axis=0)      # Series: number of non-NaNs per column

# Summary
total_nans = nan_counts.sum()
if total_nans == 0:
    print('No NaN values found in the dfset.')
else:
    print(f'Total NaN values found: {total_nans}')

n, p = X.shape  # n = number of observations, p = number of variables

# Standardize the data
X_mean = np.mean(X, axis=0)  # Mean of each column
X_std = np.std(X, axis=0)    # Standard deviation of each column

# Avoid division by zero
X_std[X_std == 0] = 1

# Standardize: (X - mean) / std
X_standardized = (X - X_mean) / X_std

# Compute the covariance matrix
cov_matrix = np.dot(X_standardized.T, X_standardized) / (n - 1)

# Eigen decomposition
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)

# Sort eigenvalues in descending order
idx = np.argsort(eig_vals)[::-1]
eig_vals_sorted = eig_vals[idx]
eig_vecs_sorted = eig_vecs[:, idx]

# Explained variance
explained = eig_vals_sorted / np.sum(eig_vals_sorted)

# Cumulative explained variance
cum_var = np.cumsum(explained)

# Target variance threshold
target_var = 0.90

# Find number of components that reach the target variance
num_components = np.argmax(cum_var >= target_var) + 1  # +1 because Python is 0-indexed

# Display result
print(f'Number of components explaining {target_var*100:.2f}% of variance: {num_components}')
for i in range(num_components):
    print(f'Component {i + 1}: {explained[i] * 100:.2f}% of variance')

# Select the top principal components
W = eig_vecs_sorted[:, :num_components]

# Project the standardized data onto the principal components
X_pca = np.dot(X_standardized, W)

# Compute scores for financial aid percentage
ix_income = [var_names.index(name) for name in ['income_total', 'cashrep', 'rentals_nonagri', 'income_ea']]
ix_demo = [var_names.index(name) for name in ['famsize', 'rural']]
ix_expenses = [var_names.index(name) for name in ['food', 'housing', 'health', 'education', 'clothing', 'misc', 'transport', 'communication', 'recreation']]
ix_debt = [var_names.index(name) for name in ['cash_loan', 'app_install', 'veh_install']]
ix_passive_income = [var_names.index(name) for name in ['pension', 'dividends', 'interest']]

# Initialize weight matrix
w_matrix = np.zeros(len(var_names))

# Assign weights to different variable groups
w_matrix[ix_income] = [0.12, 0.06, 0.04, 0.08]             # Income group (30%)
w_matrix[ix_demo] = [0.06, 0.04]                           # Demo group (10%)
w_matrix[ix_expenses] = [0.05, 0.05, 0.05, 0.04, 0.04, 0.02, 0.02, 0.01, 0.02]  # Expenses group (30%)
w_matrix[ix_debt] = [0.05, 0.05, 0.05]                     # Debt group (15%)
w_matrix[ix_passive_income] = [0.03, 0.07, 0.05]           # Passive income group (15%)

# Display total weight
print(f'Total weight: {np.sum(w_matrix):.2f}')

# Apply signs based on financial aid logic
w_matrix[ix_income] = -w_matrix[ix_income]                  # Higher income reduces need
w_matrix[ix_passive_income] = -w_matrix[ix_passive_income]  # Higher passive income reduces need

# Final score
aid_need_score = np.dot(X_standardized, w_matrix)

# Normalize the score to [0, 1]
aid_score_norm = (aid_need_score - np.min(aid_need_score)) / (np.max(aid_need_score) - np.min(aid_need_score))


# Summary statistics
print(pd.Series(aid_need_score).describe())

# Rescale
target_mean = 40
target_std = 20
current_mean = np.mean(aid_need_score)
current_std = np.std(aid_need_score)
FA_per = (aid_need_score - current_mean) / current_std  # z-score
FA_per = FA_per * target_std + target_mean  # rescale
FA_per = 100 * (FA_per - np.min(FA_per)) / (np.max(FA_per) - np.min(FA_per))
FA_per = np.clip(FA_per, 0, 100)  # Clip target variable before regression

# New Dataset
import numpy as np
import pandas as pd

# Combine PCA matrix with FA_per (assumed to be a 1D or 2D array of matching row length)
matrix_pca = np.hstack((X_pca, FA_per.reshape(-1, 1)))  # Ensure FA_per is a column vector

# Create column names
k = num_components
column_names = [f'PC{i+1}' for i in range(k)]
column_names.append('FA_Percentage')

# Create DataFrame
df_pca = pd.DataFrame(matrix_pca, columns=column_names)

# Display the DataFrame
print(df_pca)

#Least Square Solution
# Get dimensions
n, k = X_pca.shape

# Add intercept column (ones)
X1 = np.hstack((np.ones((n, 1)), X_pca))

# Normal equations
XTX = np.dot(X1.T, X1)
XTy = np.dot(X1.T, FA_per)

# Solve for coefficients
v = np.linalg.solve(XTX, XTy)  # Equivalent to MATLAB's backslash operator

# Extract intercept and coefficients
beta_0 = v[0]
beta = v[1:]

# Print results
print(f'Intercept (beta_0): {beta_0:.4f}')
for i in range(k):
    print(f'Coefficient for PC{i+1} (beta_{i+1}): {beta[i]:.4f}')

## Predicted FA
# Compute fitted values (FA_hat)
FA_hat = np.dot(X1, v)

# Compute residuals
residuals = FA_per - FA_hat

# Residual Sum of Squares (SSR)
SSR = np.sum(residuals ** 2)

# Print result
print(f'Residual Sum of Squares: {SSR:.4f}')

# Create input DataFrame (values in VND)
input_df = pd.DataFrame([[
    famsize_ip, income_total_ip, cashrep_ip, rentals_nonagri_ip, income_ea_ip, interest_ip,
    pension_ip, dividends_ip, food_ip, clothing_ip, housing_ip, health_ip, transport_ip,
    communication_ip, recreation_ip, education_ip, misc_ip, dur_furniture_ip, cash_loan_ip,
    app_install_ip, veh_install_ip, rural_ip
]], columns=var_names)

# Convert monetary features from VND to PHP
monetary_features = [
    'income_total', 'cashrep', 'rentals_nonagri', 'income_ea', 'interest', 'pension', 'dividends',
    'food', 'clothing', 'housing', 'health', 'transport', 'communication', 'recreation', 'education',
    'misc', 'dur_furniture', 'cash_loan', 'app_install', 'veh_install'
]
for feature in monetary_features:
    input_df[feature] = input_df[feature] / EXCHANGE_RATE

# Scale the new student's data
input_scaled = (input_df - X_mean) / X_std

# Project onto the principal components
input_pca = np.dot(input_scaled, W)

# Estimate the financial aid percentage
fa_percentage = beta_0 + np.dot(input_pca, beta)

# Clip the FA percentage to be between 0 and 100
fa_percentage = np.clip(fa_percentage, 0, 100)

# Print the estimated FA percentage
print(f'Estimated FA Percentage: {fa_percentage.item():.2f}%')

# Predict for training data
train_predictions = np.maximum(0, beta_0 + np.dot(X_pca, beta))
print("Training predictions range:", np.min(train_predictions), np.max(train_predictions))

# Predict FA percentage for input
raw_input_prediction = beta_0 + np.dot(input_pca, beta)
fa_percentage = np.maximum(0, raw_input_prediction)
fa_percentage = np.clip(fa_percentage.item(), 0, 100)

# Calculate VND amount (TUITION_FEE is already in VND)
final_aid_value = round(fa_percentage / 100 * TUITION_FEE)

# Debug
print("Raw input prediction:", raw_input_prediction)
print("input_scaled range:", np.min(input_scaled, axis=0), np.max(input_scaled, axis=0))
print("X_scaled range:", np.min(X_standardized, axis=0), np.max(X_standardized, axis=0))

# Display result
st.subheader("📊 Estimated Financial Aid Result")
st.success(f"Estimated Financial Aid: **{fa_percentage:.2f}%**")
st.markdown(f"🎯 This covers approximately **{final_aid_value:,.0f} VND** out of the tuition fee of {TUITION_FEE:,.0f} VND.")

st.caption("Note: This tool uses a PCA-based linear regression model. Actual financial aid decisions may consider additional factors.")
st.caption("Warning: Prediction is near bounds and may be less reliable.")