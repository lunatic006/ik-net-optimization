import numpy as np
import torch, os, time
from tqdm import tqdm

from pydrake.all import (
    MultibodyPlant,
    StartMeshcat,
)
from src.utils import RepoDir, BuildEnv



class IKDataset():
    def __init__(self, diagram, ee_frame):
        self.diagram = diagram
        self.plant = diagram.GetSubsystemByName("plant")
        self.diagram_context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.diagram_context)
        self.ee_frame = ee_frame

    def create_data(self, samples, linear = False, manifold = None):
        '''Either sample joint angles linearly, or from a uniform distribution'''

        dimension = len(self.plant.GetDefaultPositions())
        if linear:
            n_per_dim = int(samples ** (1/dimension))
            grid_per_dim = []
            for i in range(dimension):
                if i in [1, 3, 5] and manifold != None:
                    grid_per_dim.append(manifold[(i-1)/2] * np.linspace(0, np.pi, n_per_dim + 2)[:-1])
                else:
                    grid_per_dim.append(np.linspace(-np.pi, np.pi, n_per_dim + 2)[:-1])

            grids = np.meshgrid(*grid_per_dim, indexing='ij')
            joint_angles = np.column_stack([grid.flatten() for grid in grids])
        else:
            joint_angles = np.zeros((samples, dimension))
            for i in range(dimension):
                if i in [1, 3, 5] and manifold != None:
                    # Sample from [0, manifold_value * π]
                    joint_angles[:, i] = np.random.uniform(0, manifold[(i-1)//2] * np.pi, size=samples)
                else:
                    # Sample from [-π, π]
                    joint_angles[:, i] = np.random.uniform(-np.pi, np.pi, size=samples)
    
        targets = np.zeros((samples, 6))
        for i, angles in tqdm(enumerate(joint_angles), total=samples):
            self.plant.SetPositions(self.plant_context, angles)
            pose = self.ee_frame.CalcPoseInWorld(self.plant_context)
            targets[i][:3] = pose.translation()
            targets[i][3:] = pose.rotation().ToRollPitchYaw().vector()

        return joint_angles, targets

class LinearModule(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layers = [128, 128, 128]):
        super(LinearModule, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_layers:
            layers.append(torch.nn.Linear(in_size, h))
            layers.append(torch.nn.ReLU())
            in_size = h
        layers.append(torch.nn.Linear(in_size, output_size))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    meshcat = StartMeshcat()
    diagram = BuildEnv(meshcat, directives_file = os.path.join(RepoDir(), "models/iiwa_collision.yaml"))
    ee_frame = diagram.GetSubsystemByName("plant").GetFrameByName("body")

    dataset = IKDataset(diagram, ee_frame)

    joint_angles, targets = dataset.create_data(7**7, manifold = [1, 1, 1])

    print(joint_angles[:10])
