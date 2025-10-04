import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load dataset (replace 'data.csv' with the actual file if needed)
data = pd.read_csv("data.csv")

# Extract X and Y columns
x = data["X"]
y = data["Y"]

# Pearson correlation
r, p_value = pearsonr(x, y)
print(f"Pearson correlation coefficient: r = {r:.4f}, p-value = {p_value:.4e}")

# Scatter plot
plt.scatter(x, y, color="skyblue", label="Data points")

# Regression line
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b, color="red", label="Best fit line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Scatter Plot with Pearson Correlation")
plt.legend()
plt.grid(True)
plt.show()
