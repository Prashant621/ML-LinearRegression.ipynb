# 🏠 Housing Price Prediction using Linear Regression

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-013243.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-150458.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive implementation of **Simple and Multiple Linear Regression** from scratch using gradient descent to predict California housing prices. This project demonstrates fundamental machine learning concepts without relying on scikit-learn's built-in models.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Mathematical Foundation](#mathematical-foundation)
- [Results](#results)
- [Key Insights](#key-insights)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project implements linear regression algorithms from scratch to predict housing prices in California. It includes:

- **Task 1**: Simple Linear Regression using a single feature (housing median age)
- **Task 2**: Multiple Linear Regression using all available features

Both models are built using custom implementations of:
- Gradient Descent optimization
- Cost Function (Mean Squared Error)
- Feature Normalization

## ✨ Features

- 📊 **From-Scratch Implementation**: No use of sklearn's LinearRegression
- 🔢 **Gradient Descent**: Custom optimization algorithm
- 📈 **Feature Normalization**: Z-score standardization
- 🎨 **Visualizations**: Clear plots showing model fit
- 📝 **Detailed Documentation**: Comprehensive explanations of each step
- 🧪 **Reproducible Results**: Consistent random seeds and parameters

## 📊 Dataset

The project uses the **California Housing Dataset**, which contains information about housing districts in California.

### Features:

| Feature | Description | Type |
|---------|-------------|------|
| `longitude` | Longitude coordinate | Continuous |
| `latitude` | Latitude coordinate | Continuous |
| `housing_median_age` | Median age of houses in the district | Continuous |
| `total_rooms` | Total number of rooms in the district | Continuous |
| `total_bedrooms` | Total number of bedrooms in the district | Continuous |
| `population` | Total population in the district | Continuous |
| `households` | Total number of households in the district | Continuous |
| `median_income` | Median income of households (in $10,000s) | Continuous |
| `ocean_proximity` | Proximity to ocean | Categorical |

### Target Variable:
- `median_house_value`: Median house value in the district (USD)

### Dataset Statistics:
- **Total Samples**: ~20,640 districts
- **Missing Values**: Present in `total_bedrooms` column
- **Data Type**: Numerical (continuous) + 1 categorical feature

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

### Requirements

Create a `requirements.txt` file with:
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
jupyter>=1.0.0
```

## 📁 Project Structure

```
housing-price-prediction/
│
├── data/
│   └── housing.csv                 # California housing dataset
│
├── notebooks/
│   ├── task1_simple_regression.ipynb    # Single feature regression
│   └── task2_multiple_regression.ipynb  # Multiple feature regression
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py          # Data loading and cleaning
│   ├── gradient_descent.py         # Optimization algorithms
│   ├── cost_functions.py           # Loss calculation
│   └── visualization.py            # Plotting utilities
│
├── results/
│   ├── simple_regression_plot.png
│   └── coefficients.txt
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## 💻 Usage

### Quick Start

**Task 1: Simple Linear Regression**
```python
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('./housing.csv')

# Select single feature
X = data[['housing_median_age']].values.flatten()
Y = data['median_house_value'].values

# Add bias term
X_with_bias = np.column_stack((np.ones(len(X)), X))

# Run gradient descent
theta = gradient_descent(X_with_bias, Y, alpha=0.001, iterations=5000)

print(f"Intercept: {theta[0]:.2f}")
print(f"Coefficient: {theta[1]:.2f}")
```

**Task 2: Multiple Linear Regression**
```python
# Select multiple features
feature_columns = ['housing_median_age', 'total_rooms', 'total_bedrooms',
                   'population', 'households', 'median_income']
X = data[feature_columns].values
Y = data['median_house_value'].values

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Add bias term
X = np.column_stack((np.ones(X.shape[0]), X))

# Run gradient descent
theta = gradient_descent_multi(X, Y, iterations=10000, alpha=0.1)

print("Learned coefficients:", theta)
```

### Running the Jupyter Notebooks

```bash
jupyter notebook
# Navigate to notebooks/ and open task1_simple_regression.ipynb
```

## 🧮 Mathematical Foundation

### 1. Hypothesis Function

**Simple Linear Regression:**
```
h(x) = θ₀ + θ₁x
```

**Multiple Linear Regression:**
```
h(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
```

Or in matrix form:
```
h(X) = Xθ
```

### 2. Cost Function (Mean Squared Error)

```
J(θ) = (1/2m) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

Where:
- `m` = number of training examples
- `h(x⁽ⁱ⁾)` = predicted value
- `y⁽ⁱ⁾` = actual value

### 3. Gradient Descent Update Rule

```
θⱼ := θⱼ - α × (1/m) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾) × xⱼ⁽ⁱ⁾
```

Where:
- `α` = learning rate
- `θⱼ` = parameter j
- `xⱼ⁽ⁱ⁾` = feature j of example i

### 4. Feature Normalization (Z-score)

```
x_norm = (x - μ) / σ
```

Where:
- `μ` = mean of feature
- `σ` = standard deviation of feature

## 📈 Results

### Task 1: Simple Linear Regression

**Model Equation:**
```
Price = θ₀ + θ₁ × (Housing Median Age)
```

**Performance:**
- Uses only housing median age as predictor
- Provides baseline understanding of age-price relationship
- Limited predictive power due to single feature

**Interpretation:**
- **Intercept (θ₀)**: Expected price when age = 0 (baseline price)
- **Coefficient (θ₁)**: Change in price per year increase in housing age

### Task 2: Multiple Linear Regression

**Model Equation:**
```
Price = θ₀ + θ₁×age + θ₂×rooms + θ₃×bedrooms + θ₄×population + θ₅×households + θ₆×income
```

**Performance:**
- Significantly better than single-feature model
- Captures complex relationships between multiple factors
- Each coefficient represents the marginal effect of that feature

**Interpretation:**
- **Each θᵢ**: Change in price for one unit increase in feature i (holding others constant)
- **Normalized features**: Coefficients are comparable in magnitude

## 💡 Key Insights

### Model Comparison

| Aspect | Simple Regression | Multiple Regression |
|--------|------------------|---------------------|
| **Accuracy** | Lower | Higher ✓ |
| **Interpretability** | Very Easy ✓ | Moderate |
| **Visualization** | 2D Plot ✓ | High-dimensional |
| **Features Used** | 1 | 6 |
| **Use Case** | Quick insights | Production models |

### Why Multiple Features Help

1. **Captures Complexity**: Real-world prices depend on many factors
2. **Reduces Bias**: Single feature may miss important predictors
3. **Better Generalization**: More information leads to better predictions
4. **Accounts for Interactions**: Multiple features can work together

### When to Use Each Model

**Simple Linear Regression:**
- ✅ Need quick, interpretable insights
- ✅ Exploring single-variable relationships
- ✅ Educational purposes
- ✅ Data visualization for presentations

**Multiple Linear Regression:**
- ✅ Production prediction systems
- ✅ Need high accuracy
- ✅ Sufficient data available
- ✅ Understanding complex relationships

## 🔬 Implementation Details

### Data Preprocessing

```python
# 1. Handle missing values
data_cleaned = data.dropna(subset=['housing_median_age', 'median_house_value'])

# 2. Feature normalization
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)

# 3. Add bias term
X_with_bias = np.column_stack((np.ones(X.shape[0]), X_normalized))
```

### Hyperparameters

| Task | Learning Rate (α) | Iterations | Features |
|------|------------------|------------|----------|
| Task 1 | 0.001 | 5,000 | 1 |
| Task 2 | 0.1 | 10,000 | 6 |

### Convergence Criteria

The algorithm stops when:
- Maximum iterations reached, OR
- Gradient norm < 1e-6 (convergence threshold)

## 📊 Visualizations

### Simple Linear Regression Plot

The scatter plot shows:
- **Red X marks**: Actual data points
- **Blue line**: Fitted regression line
- Clear trend visualization

### Feature Importance

In multiple regression, coefficients indicate:
- **Positive coefficient**: Feature increases price
- **Negative coefficient**: Feature decreases price
- **Magnitude**: Strength of relationship

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Areas for Contribution

- [ ] Add polynomial regression implementation
- [ ] Implement regularization (Ridge, Lasso)
- [ ] Add cross-validation
- [ ] Create interactive visualizations
- [ ] Add unit tests
- [ ] Improve documentation
- [ ] Add more evaluation metrics (R², RMSE, MAE)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: California Housing Dataset (originally from StatLib repository)
- **Inspiration**: Andrew Ng's Machine Learning course
- **Mathematical Foundation**: Stanford CS229 course materials
- **Community**: Thanks to all contributors and users

## 📚 Further Reading

### Recommended Resources

1. **Books**
   - "Pattern Recognition and Machine Learning" by Christopher Bishop
   - "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman

2. **Courses**
   - [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)
   - [Stanford CS229: Machine Learning](http://cs229.stanford.edu/)

3. **Papers**
   - "Least Squares Optimization" - Classical papers on gradient descent
   - Feature normalization techniques in ML

### Related Projects

- [Scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [TensorFlow Linear Regression](https://www.tensorflow.org/tutorials)

## 📧 Contact

**Project Maintainer**: Your Name
- GitHub: [@yourusername](https://github.com/yourusername)
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

## 🌟 Show Your Support

If this project helped you learn or solve a problem, please consider:

- ⭐ **Starring** the repository
- 🍴 **Forking** for your own projects
- 📢 **Sharing** with others who might find it useful
- 💬 **Providing feedback** through issues

---

<div align="center">

**Made with ❤️ and Python**

[Report Bug](https://github.com/yourusername/housing-price-prediction/issues) • 
[Request Feature](https://github.com/yourusername/housing-price-prediction/issues) • 
[Documentation](https://github.com/yourusername/housing-price-prediction/wiki)

</div>
