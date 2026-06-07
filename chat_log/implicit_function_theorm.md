Here's the symbolic step-by-step:

## Step 1: The forward pass solves a root-finding problem

At convergence, the interior point method finds $(\mathbf{x}^*, \mathbf{z}^*, \mathbf{y}^*)$ that satisfy the (central-path-smoothed) KKT system:

$$\mathbf{F}(\mathbf{x}^*, \mathbf{z}^*, \mathbf{y}^*; \boldsymbol{\theta}) = 
\begin{bmatrix}
\mathbf{Q}\mathbf{x}^* + \mathbf{G}^T\mathbf{z}^* + \mathbf{A}^T\mathbf{y}^* + \mathbf{p} \\
\mathbf{G}\mathbf{x}^* - \mathbf{F}\mathbf{z}^* - \mathbf{h} \\
\mathbf{A}\mathbf{x}^* - \mathbf{b}
\end{bmatrix} = \mathbf{0}$$

where $\boldsymbol{\theta} = \{\mathbf{Q}, \mathbf{p}, \mathbf{G}, \mathbf{h}, \mathbf{A}, \mathbf{b}, \mathbf{F}\}$ are the parameters.

## Step 2: The Implicit Function Theorem

If $\mathbf{F}(\mathbf{x}^*; \boldsymbol{\theta}) = 0$ and the Jacobian $\frac{\partial \mathbf{F}}{\partial \mathbf{x}}$ is invertible, then $\mathbf{x}^*$ is a differentiable function of $\boldsymbol{\theta}$:

$$\frac{\partial \mathbf{x}^*}{\partial \boldsymbol{\theta}} = -\left[\frac{\partial \mathbf{F}}{\partial \mathbf{x}}\right]^{-1} \cdot \frac{\partial \mathbf{F}}{\partial \boldsymbol{\theta}}$$

The Jacobian $\frac{\partial \mathbf{F}}{\partial \mathbf{x}}$ is exactly the **KKT matrix**:

$$\mathbf{K} = \frac{\partial \mathbf{F}}{\partial(\mathbf{x}, \mathbf{z}, \mathbf{y})} = 
\begin{bmatrix}
\mathbf{Q} & \mathbf{G}^T & \mathbf{A}^T \\
\mathbf{G} & -\mathbf{F} & 0 \\
\mathbf{A} & 0 & 0
\end{bmatrix}$$

## Step 3: The backward pass avoids computing the inverse

We want $\frac{\partial L}{\partial \boldsymbol{\theta}}$ given $\frac{\partial L}{\partial \mathbf{x}^*}$ (the incoming gradient). By the chain rule:

$$\frac{\partial L}{\partial \boldsymbol{\theta}} = \frac{\partial L}{\partial \mathbf{x}^*} \cdot \frac{\partial \mathbf{x}^*}{\partial \boldsymbol{\theta}} = -\underbrace{\frac{\partial L}{\partial \mathbf{x}^*} \cdot \mathbf{K}^{-1}}_{\text{adjoint } \mathbf{v}} \cdot \frac{\partial \mathbf{F}}{\partial \boldsymbol{\theta}}$$

Define the **adjoint variable** $\mathbf{v}$:

$$\mathbf{v} = \left(\frac{\partial L}{\partial \mathbf{x}^*}\right) \mathbf{K}^{-1} \iff \mathbf{K}^T \mathbf{v}^T = \left(\frac{\partial L}{\partial \mathbf{x}^*}\right)^T$$

Since $\mathbf{K}$ is symmetric ($\mathbf{K}^T = \mathbf{K}$):

$$\mathbf{K} \cdot \mathbf{v}^T = \left(\frac{\partial L}{\partial \mathbf{x}^*}\right)^T$$

This is **one linear solve** with the already-factorized KKT matrix!

## Step 4: The code exactly implements this

```python
# lcp_solver.py backward() — line 297-301

# Solve: K · v = (∂L/∂x*)ᵀ
# K is [Q_LU, d, G, A, S_LU] — the factorized KKT matrix
dx, _, dlam, dnu = solve_kkt(
    Q_LU, d, G, A, S_LU,   # ← same factorization as forward pass!
    dl_dzhat,               # ← (∂L/∂x*)ᵀ
    zeros, zeros, zeros     # ← RHS for other rows is zero
)
# dx = ∂L/∂p, dlam = ∂L/∂h (negated), dnu = ∂L/∂b (negated)
```

## Step 5: Chain rule to each parameter

With $\mathbf{v} = (\mathbf{dx}, \mathbf{dlam}, \mathbf{dnu})$ computed, apply $\frac{\partial \mathbf{F}}{\partial \boldsymbol{\theta}}$:

| Parameter $\boldsymbol{\theta}$ | $\frac{\partial \mathbf{F}}{\partial \boldsymbol{\theta}}$ | Result $\frac{\partial L}{\partial \boldsymbol{\theta}}$ | Code |
|---|---|---|---|
| $\mathbf{Q}$ | $\mathbf{x}^*$ appearing in row 1 | $\frac{1}{2}(\mathbf{dx} \cdot \mathbf{x}^{*T} + \mathbf{x}^* \cdot \mathbf{dx}^T)$ | `dQs = 0.5 * (bger(dx, zhats) + bger(zhats, dx))` |
| $\mathbf{G}$ | $\mathbf{x}^*$ in row 1, $\mathbf{z}^*$ in row 2 | $\mathbf{dlam} \cdot \mathbf{x}^{*T} + \mathbf{z}^* \cdot \mathbf{dx}^T$ | `dGs = bger(dlam, zhats) + bger(lams, dx)` |
| $\mathbf{h}$ | $-1$ in row 2 | $-\mathbf{dlam}$ | `dhs = -dlam` |
| $\mathbf{p}$ | $1$ in row 1 | $\mathbf{dx}$ | `dps = dx` |
| $\mathbf{F}$ | $-\mathbf{z}^*$ in row 2 | $\mathbf{dlam} \cdot \mathbf{z}^{*T}$ | `dFs = bger(dlam, lams)` |

The key insight: **the backward pass is just one more KKT solve** with a different right-hand side, using the same factorization. No matrix inversion, no autograd tracing through the IPM iterations — just one linear system solve.

Updated todo list