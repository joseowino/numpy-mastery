# 🧠 NumPy Mastery

Welcome to the **NumPy Mastery** — a hands-on journey to mastering Python’s powerful numerical computing library, **NumPy**.
In this lab-style project, you’ll explore practical NumPy applications for data manipulation, numerical analysis, and array-based computation — skills that form the foundation of Data Science and Machine Learning.

---

## 🎯 Goal

To understand and practice **practical usage of NumPy** for efficient numerical computation, data manipulation, and statistical analysis.

By the end of this project, you’ll:

* Build, slice, and manipulate NumPy arrays.
* Use broadcasting for high-performance computation.
* Generate and process random data.
* Handle missing values (NaN).
* Perform statistical analysis on real datasets.

---

## 🧩 Learning Objectives

* Master fundamental NumPy operations.
* Efficiently manipulate, analyze, and extract insights from numerical data.
* Gain comfort with Jupyter Notebooks as a data science development environment.
* Understand how NumPy integrates with other tools in the Python ecosystem (e.g., Pandas, Scikit-learn).

---

## 🧰 Virtual Environment Setup

### Requirements

* **Python 3.x (3.8+ recommended)**
* **NumPy**
* **Jupyter Notebook / JupyterLab**

### Setup Instructions

```bash
# Create virtual environment
python3 -m venv ex00

# Activate environment (Linux/Mac)
source ex00/bin/activate

# Activate environment (Windows)
ex00\Scripts\activate

# Install required libraries
pip install numpy jupyter

# Save requirements
pip freeze > requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook --port=8891
```

Create a new notebook named **Notebook_ex00.ipynb**.

---

## 🚀 Exercises Overview

### **Exercise 0: Environment and Libraries**

Set up your Python environment and launch a Jupyter notebook.
Create headings and print the message:

```python
print("Buy the dip ?")
```

---

### **Exercise 1: Your First NumPy Array**

Create a NumPy array containing:

* integer, float, string, dictionary, list, tuple, set, boolean.

Then print the data types:

```python
for i in your_np_array:
    print(type(i))
```

---

### **Exercise 2: Zeros**

Create a 1D array of **300 zeros**, reshape it to **(3, 100)**.

---

### **Exercise 3: Slicing**

Work with arrays efficiently using slicing:

1. Create an array of integers from 1 to 100.
2. Extract all **odd integers**.
3. Extract all **even integers in reverse**.
4. Set every 3rd element (starting from the second) to **0**.

---

### **Exercise 4: Random**

Generate random data using NumPy’s random module.

* Set the seed to **888**.
* Generate:

  * 1D array (size 100) with a **normal distribution**.
  * 2D array (8×8) with random integers **[1–10]**.
  * 3D array (4×2×5) with random integers **[1–17]**.

---

### **Exercise 5: Split, Concatenate, Reshape**

* Create arrays `[1,...,50]` and `[51,...,100]`.
* Concatenate them → `[1,...,100]`.
* Reshape into a **10×10** array.
* Print the results.

---

### **Exercise 6: Broadcasting and Slicing**

1. Create a **9×9** array filled with `1`s (`dtype=int8`).
2. Use slicing to generate the pattern:

```
[[1 1 1 1 1 1 1 1 1]
 [1 0 0 0 0 0 0 0 1]
 [1 0 1 1 1 1 1 0 1]
 ...
 [1 1 1 1 1 1 1 1 1]]
```

3. Practice broadcasting:

```python
array_1 = np.array([1,2,3,4,5], dtype=np.int8)
array_2 = np.array([1,2,3], dtype=np.int8)
```

Expected:

```
[[ 1  2  3]
 [ 2  4  6]
 [ 3  6  9]
 [ 4  8 12]
 [ 5 10 15]]
```

---

### **Exercise 7: NaN Handling**

Simulate missing grades with `np.nan`:

```python
generator = np.random.default_rng(123)
grades = np.round(generator.uniform(low=0.0, high=10.0, size=(10, 2)))
grades[[1,2,5,7], [0,0,0,0]] = np.nan
```

Create a **third column** that uses:

* Grade 1 if available, otherwise Grade 2.
  Use `np.where` (no loops).

---

### **Exercise 8: Wine Dataset Analysis**

Analyze the **Red Wine dataset** using NumPy:

1. Load data via `np.genfromtxt('winequality-red.csv', delimiter=';', skip_header=1, dtype=np.float32)`.
2. Display the 2nd, 7th, and 12th rows.
3. Check if any wine has **alcohol > 20%**.
4. Compute the **average alcohol percentage**.
5. Compute statistics for **pH values** (min, max, 25%, 50%, 75%, mean).
6. Find the **average quality** of wines with the lowest 20% sulphates.
7. Compute the mean of all variables for:

   * **Best quality wines**
   * **Worst quality wines**

---

### **Exercise 9: Football Tournament**

You’re organizing a 10-team football tournament.

* Load the **model_forecasts.txt** matrix (predicted score differences).
* Each `(i, j)` entry = predicted score difference between Team i and Team j.
* Find pairs of teams that **minimize the sum of squared differences** (most exciting matches).

> Use **itertools** and **NumPy** (no for-loops).

Expected output format:

```
[[m1_t1 m2_t1 m3_t1 m4_t1 m5_t1]
 [m1_t2 m2_t2 m3_t2 m4_t2 m5_t2]]
```

---

## 🧠 Concepts Covered

* Array creation and manipulation
* Indexing and slicing
* Random number generation
* Broadcasting
* Handling NaN values
* Statistical computation
* Real dataset analysis
* Permutations and optimization

---

## 📦 Project Structure

```
numpy-training-room/
│
├── ex00/
│   ├── Notebook_ex00.ipynb
│   ├── requirements.txt
│   └── README.md
│
├── data/
│   ├── winequality-red.csv
│   └── model_forecasts.txt
│
└── exercises/
    ├── ex01_arrays.ipynb
    ├── ex02_zeros.ipynb
    ├── ex03_slicing.ipynb
    ├── ex04_random.ipynb
    ├── ex05_concat_reshape.ipynb
    ├── ex06_broadcasting.ipynb
    ├── ex07_nan.ipynb
    ├── ex08_wine.ipynb
    └── ex09_football.ipynb
```

---

## 🧾 Resources

* [NumPy Documentation](https://numpy.org/doc/stable/)
* [Jupyter Documentation](https://jupyter.org/)
* [Conda Documentation](https://docs.conda.io/)
* [Why You Should Use Jupyter Notebooks](https://realpython.com/jupyter-notebook-introduction/)
* [Computation on Arrays: Broadcasting (NumPy Guide)](https://numpy.org/doc/stable/user/basics.broadcasting.html)

---

## 💬 Author

**joseowino**