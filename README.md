# Portfolio Optimization — Markowitz Mean-Variance Model

> Master’s project in **Optimization and Quantitative Finance**, implemented in [![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/), based on **Markowitz’s Modern Portfolio Theory** and the **Frank–Wolfe algorithm** for constrained quadratic optimization.

---

## Executive Summary

This project explores the **Modern Portfolio Theory (MPT)** introduced by **Harry Markowitz (1952)** — a cornerstone of modern finance that formalizes the trade-off between **expected return** and **risk** through quantitative optimization.

The goal is to construct an **efficient portfolio** that maximizes expected return for a given level of risk (variance), or equivalently, minimizes variance for a given level of expected return.

### Key Concepts

- **Expected Return:**
  ```math
  E(R_p) = \sum_{i=1}^{n} w_i E(R_i)
  ```

- **Portfolio Variance (Risk):**
  ```math
  \sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \mathrm{Cov}(R_i, R_j)
  ```

- **Optimization Problem:**
  ```math
  \max_{w \in \mathbb{R}^n} \; E(R_p) - \frac{\lambda}{2} \sigma_p^2
  ```
  where $\lambda$ is the **risk aversion coefficient**.

This leads to the **efficient frontier**, representing optimal portfolios that achieve the best trade-offs between risk and return.

---

## Mathematical Framework

Starting from a Taylor expansion of a **utility function** $U(X)$ around its mean value $\bar{X}$ and introducing the **risk aversion coefficient**:
```math
\rho(X) = - \frac{U''(X)}{U'(X)}
```

We obtain the **certainty equivalent** of the portfolio:
```math
E_c(X) = \bar{X} - \frac{\rho(\bar{X})}{2} \mathrm{Var}(X)
```

Maximizing $E_c(X)$ leads to the Markowitz formulation:
```math
J(x) = \langle \bar{\mu}, x \rangle - \frac{\rho}{2} \langle K x, x \rangle
```

where:
- $\bar{\mu}$ = expected returns vector  
- $K$ = covariance matrix of returns  
- $\rho$ = risk aversion parameter  
- $x$ = portfolio weights  

The optimization problem becomes:
```math
\max_{x \in X_{ad}} \; J(\mu, x)
```

and admits a **saddle point** solution under convexity/concavity conditions:
```math
\min_{\mu \in U_{ad}} \max_{x \in X_{ad}} J(\mu, x)
= \max_{x \in X_{ad}} \min_{\mu \in U_{ad}} J(\mu, x)
```

---

## Multi-Scenario & Existence of Solution

Under the assumptions that $U_{ad}$ and $X_{ad}$ are **convex and compact**, and that $J(\mu, x)$ is continuous, convex in $\mu$, and concave in $x$, we can apply the **Sion’s Minimax Theorem**, ensuring that the saddle point exists and that:

```math
\min_{\mu} \max_{x} J(\mu, x)
= \max_{x} \min_{\mu} J(\mu, x)
```

Uniqueness is guaranteed when $J$ is **strictly concave in $x$** and **strictly convex in $\mu$**.

---

## Algorithmic Resolution — Frank–Wolfe Method

The **Frank–Wolfe algorithm** (also known as Conditional Gradient Method) is used to solve the constrained maximization problem in $x$:

### Algorithm Steps

1. **Initialization:**  
   Choose an initial point $x_0 \in X_{ad}$.

2. **Linear search (Simplex):**  
   Solve:
   ```math
   \max_{x \in X_{ad}} \langle \nabla J(x_k), x \rangle
   ```
   to obtain $\tilde{x}_{k+1}$.

3. **Update:**
   ```math
   x_{k+1} = (1 - \tau_{k+1}) x_k + \tau_{k+1} \tilde{x}_{k+1}, \quad \tau_{k+1} = \frac{2}{2 + k}
   ```

4. **Stopping condition:**
   ```math
   \|x_{k+1} - x_k\| < \varepsilon \|x_1 - x_0\|
   ```

This guarantees convergence because $J$ is differentiable and $X_{ad}$ is bounded and convex.

---

## Dual Formulation

Define the **dual function**:
```math
H(\mu) = \max_{x \in X_{ad}} \left( \langle \mu, x \rangle - \frac{\rho}{2} \langle Kx, x \rangle \right)
```

By **Cholesky factorization**, we can write $\rho K = L L^T$, where $L$ is lower-triangular.  
Then:
```math
H(\mu) = \frac{1}{2} \left( \langle K \mu, \mu \rangle - d^2(L^{-1} \mu, L^T X_{ad}) \right)
```

where $d^2(y, C) = \|y - \Pi_C(y)\|^2$ is the squared distance from $y$ to the convex set $C$.

---

## Gradient and Convexity of the Dual

The dual function $H$ is differentiable ($C^1$) with:
```math
\nabla H(\mu) = x(\mu)
```
where $x(\mu)$ is the maximizer of $J(\mu, x)$.  
Its Hessian is:
```math
\nabla^2 H(\mu) = (L L^T)^{-1}
```
which is **positive definite**, proving that $H$ is **strictly convex**.

---

## Numerical Implementation (Python)

The project includes a full implementation of the **Frank–Wolfe algorithm** for both the **primal** and **dual** problems:

```python
# Definition of the objective
def J(mu, x, rho, K):
    return np.dot(mu, x) - 0.5 * np.dot(np.dot(rho * K, x), x)

# Gradient of J wrt x
def gradient_J(mu, x, rho, K):
    return mu - rho * np.dot(K, x)

# Frank–Wolfe algorithm for maximizing J
def max_J_en_x(mu, rho, K, epsilon, N=1e6):
    x_k = np.random.rand(len(K))
    x_prev = np.zeros_like(x_k)
    k = 0
    while np.linalg.norm(x_k - x_prev) >= epsilon * np.linalg.norm(x_prev):
        x_tilde = np.zeros_like(x_k)
        x_tilde[np.argmin(gradient_J(mu, x_k, rho, K))] = 1
        step = 2 / (2 + k)
        x_prev = x_k.copy()
        x_k = (1 - step) * x_k + step * x_tilde
        k += 1
    return x_k
```

The same iterative structure is used to minimize the **dual function** $H(\mu)$, leveraging the same convergence rate and linear search logic.

---

## Results & Interpretation

- The **optimal portfolio weights** $x^*$ are obtained from the Frank–Wolfe algorithm.  
- The **dual variable** $\mu^*$ minimizes the dual function $H(\mu)$, corresponding to equilibrium in the saddle-point formulation.  
- The results illustrate the **duality and convergence** of the Markowitz optimization under convex constraints.

---

## Notes

- The **algorithm converges** under convexity and compactness assumptions on $X_{ad}$ and $U_{ad}$.  
- The **Frank–Wolfe step size** $\tau_k = 2 / (2 + k)$ ensures convergence without parameter tuning.  
- The **duality link** provides both theoretical and computational validation of Markowitz optimization.

---

**Author:** Shakil Rohimun  
**License:** Private Academic Use – Université Paris-Saclay
