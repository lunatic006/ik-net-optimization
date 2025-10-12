import numpy as np
import torch, os, time
from tqdm import tqdm

from pydrake.all import (
    MultibodyPlant,
    StartMeshcat,
)
from src.utils import RepoDir, BuildEnv, extract_xyzrpy



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

def train_test_split(X, y, test_size=0.2):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    split_idx = int(n_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test

def train_model(model, optimizer, loss_fn, X_train, y_train, epochs=1000):
    '''batch this'''
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train)

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = loss_fn(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

def evaluate_model(model, loss_fn, X_test, y_test, dataset):
    X_tensor = torch.FloatTensor(X_test)
    y_tensor = torch.FloatTensor(y_test)

    with torch.no_grad():
        predictions = model(X_tensor)
        loss = loss_fn(predictions, y_tensor)
    
    print(predictions.shape)
    backwards_loss = np.zeros(predictions.shape[0])
    for i, angles in enumerate(predictions):
        dataset.plant.SetPositions(dataset.plant_context, angles)
        pose = dataset.ee_frame.CalcPoseInWorld(dataset.plant_context)
        backwards_loss[i] = np.linalg.norm(extract_xyzrpy(pose) - X_tensor.numpy()[i][:6])
    


    print(f"Backwards Loss: {np.mean(backwards_loss):.6f}")
    print(f"Test Loss: {loss.item():.6f}")

if __name__ == "__main__":
    meshcat = StartMeshcat()
    diagram = BuildEnv(meshcat, directives_file = os.path.join(RepoDir(), "models/iiwa_collision.yaml"))
    ee_frame = diagram.GetSubsystemByName("plant").GetFrameByName("body")

    dataset = IKDataset(diagram, ee_frame)

    joint_angles, targets = dataset.create_data(5**7, manifold = [1, 1, 1])

    X = np.concatenate([targets, joint_angles[:, 0:1]], axis=1)
    y = joint_angles
    
    model = LinearModule(7, 7, hidden_layers = [256, 256, 256])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    train_model(model, optimizer, loss_fn, X_train, y_train, epochs=200)
    evaluate_model(model, loss_fn, X_test, y_test, dataset)