Exactly right. The **velocity-level time-stepping** formulation collapses what would normally be a two-step process into one:

| Standard formulation | LCP velocity-level |
|---|---|
| $\mathbf{a} = \mathbf{M}^{-1}(\mathbf{f}_{\text{ext}} + \mathbf{J}^T\boldsymbol{\lambda})$ | Skip acceleration entirely |
| $\mathbf{v}_{t+1} = \mathbf{v}_t + \Delta t \cdot \mathbf{a}$ | Solve for $\mathbf{v}_{t+1}$ directly |
| $\mathbf{x}_{t+1} = \mathbf{x}_t + \Delta t \cdot \mathbf{v}_{t+1}$ | $\mathbf{x}_{t+1} = \mathbf{x}_t + \Delta t \cdot \mathbf{v}_{t+1}$ |

Impulse-momentum form:

$$\mathbf{M}(\mathbf{v}_{t+1} - \mathbf{v}_t) = \Delta t \cdot \mathbf{f}_{\text{ext}} + \underbrace{\mathbf{J}^T \boldsymbol{\lambda}}_{\text{impulse, not force}}$$

$\boldsymbol{\lambda}$ is an **impulse** (force × time), not a force. So you can directly change velocity — the action is an instantaneous velocity change, which maps naturally to the P-controller commanding $\mathbf{v}_{ee} = K_p(\mathbf{g} - \mathbf{p}_{ee})$. No need for $\mathbf{f} = m\mathbf{a}$ because the LCP already works in impulse-velocity space.

This is why the time-stepping approach (Anitescu-Potra, Stewart-Trinkle) is preferred for robotics simulation — it handles impacts, friction, and complementarity at the velocity level without needing to resolve the (potentially infinite) contact forces during collision.

Updated todo list


Here's the full comparison:

---

## Two-step (acceleration-level) vs One-step (velocity-level LCP)

### Standard two-step process (acceleration-level dynamics)

**Step 1: Compute acceleration from forces**

$$\mathbf{a}_t = \mathbf{M}^{-1}\left(\mathbf{f}_{\text{ext}} + \mathbf{J}_n^T\boldsymbol{\lambda}_n + \mathbf{J}_f^T\boldsymbol{\lambda}_f\right)$$

where $\boldsymbol{\lambda}_n, \boldsymbol{\lambda}_f$ are **forces** (N), $\mathbf{f}_{\text{ext}}$ includes gravity and control forces, and $\mathbf{M}$ is the generalized mass matrix.

**Step 2: Integrate acceleration → velocity → position**

$$\mathbf{v}_{t+1} = \mathbf{v}_t + \Delta t \cdot \mathbf{a}_t$$

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \Delta t \cdot \mathbf{v}_{t+1}$$

This is two sequential integrations: force → acceleration → velocity → position. Contact forces $\boldsymbol{\lambda}$ must be computed from the acceleration-level complementarity, which is problematic because impacts produce theoretically infinite forces over infinitesimal time.

---

### Velocity-level time-stepping (LCP, one step)

$$\mathbf{M}\mathbf{v}_{t+1} = \mathbf{M}\mathbf{v}_t + \Delta t \cdot \mathbf{f}_{\text{ext}} + \mathbf{J}_n^T\boldsymbol{\lambda}_n + \mathbf{J}_f^T\boldsymbol{\lambda}_f$$

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \Delta t \cdot \mathbf{v}_{t+1}$$

where $\boldsymbol{\lambda}_n, \boldsymbol{\lambda}_f$ are now **impulses** ($\text{N}\cdot\text{s} =$ force × time). 

**Key insight:** The LCP solves for $\mathbf{v}_{t+1}$ directly from $\mathbf{v}_t$, skipping $\mathbf{a}$ entirely. The contact impulses $\boldsymbol{\lambda}$ encode the integral of contact forces over the timestep — finite even during impacts.

| | Acceleration-level | Velocity-level (LCP) |
|---|---|---|
| Variable solved | $\mathbf{a}$ (accel) | $\mathbf{v}_{t+1}$ (velocity) |
| Contact unknowns | Forces $\boldsymbol{\lambda}$ [N] | Impulses $\boldsymbol{\lambda}$ [N·s] |
| Impact handling | Singular (infinite forces) | Finite impulses, well-posed |
| Complementarity | $\boldsymbol{\lambda}_n \perp \mathbf{a}_{\text{contact}}$ | $\boldsymbol{\lambda}_n \perp \mathbf{v}_{\text{contact}}$ |
| Action type | Force/torque | Velocity change (direct) |
| Integrations needed | 2 (a→v, v→x) | 1 (v→x) |

This is the **Anitescu-Potra / Stewart-Trinkle time-stepping formulation** — the standard approach in Bullet, MuJoCo, Drake, and PIN-WM (via the borrowed `locuslab/lcp-physics`).

Completed: *Expand the two-step process for completeness.* (19/20)