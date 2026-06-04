import numpy as np

#! In Isaac Sim, SimulationApp must start first so extensions like isaacsim.core are available.
#! then Isaac/Omniverse modules can be imported.


try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp


simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.physics_context import PhysicsContext


PhysicsContext()
GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]))
DynamicCuboid(
    prim_path="/World/cube",
    position=np.array([-0.5, -0.2, 1.0]),
    scale=np.array([0.5, 0.5, 0.5]),
    color=np.array([0.2, 0.3, 0.0]),
)

for _ in range(240):
    simulation_app.update()

simulation_app.close()
