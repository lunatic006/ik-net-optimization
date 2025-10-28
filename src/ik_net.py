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

    def create_data(self, samples, linear = False, manifold = None, matrix = False):
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
    
        if matrix:
            targets = np.zeros((samples, 12))
        else:
            targets = np.zeros((samples, 6))
        for i, angles in tqdm(enumerate(joint_angles), total=samples):
            self.plant.SetPositions(self.plant_context, angles)
            pose = self.ee_frame.CalcPoseInWorld(self.plant_context)
            if matrix:
                targets[i][:3] = pose.translation()
                targets[i][3:] = pose.rotation().matrix().flatten()
            else:
                targets[i] = extract_xyzrpy(pose)

        return joint_angles, targets

class LinearModule(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_layers = [128, 128, 128], dropout_rate=0.1):
        super(LinearModule, self).__init__()
        layers = []
        in_size = input_size
        for h in hidden_layers:
            # Create layer
            linear = torch.nn.Linear(in_size, h)
            # Apply Kaiming init
            
            torch.nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
            torch.nn.init.zeros_(linear.bias)

            layers.append(linear)
            layers.append(torch.nn.ReLU())
            if dropout_rate > 0:
                layers.append(torch.nn.Dropout(dropout_rate))

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

def train_model(model, optimizer, loss_fn, X_train, y_train, epochs=1000, batch_size=None, device='cpu'):
    '''Train model with optional batching'''
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.FloatTensor(y_train).to(device)
    
    losses = []
    
    if batch_size is None:
        # No batching - use full dataset
        print("Training without batching (full dataset)")
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            predictions = model(X_tensor)
            loss = loss_fn(predictions, y_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    else:
        # Use batching
        print(f"Training with batching (batch_size={batch_size})")
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in tqdm(range(epochs)):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = loss_fn(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Average loss over all batches in the epoch
            losses.append(epoch_loss / len(dataloader))
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {losses[-1]:.6f}")
    
    return losses

def evaluate_model(model, loss_fn, X_test, y_test, dataset, matrix = False, device='cpu'):
    X_tensor = torch.FloatTensor(X_test).to(device)
    y_tensor = torch.FloatTensor(y_test).to(device)

    with torch.no_grad():
        predictions = model(X_tensor)
        loss = loss_fn(predictions, y_tensor)
    
    # Move predictions back to CPU for numpy operations
    predictions_cpu = predictions.cpu()
    X_tensor_cpu = X_tensor.cpu()
    
    backwards_loss = np.zeros(predictions.shape[0])
    position_loss = np.zeros(predictions.shape[0])
    for i, angles in enumerate(predictions_cpu):
        dataset.plant.SetPositions(dataset.plant_context, angles.numpy())
        pose = dataset.ee_frame.CalcPoseInWorld(dataset.plant_context)
        if matrix: pose_flatten = pose.translation().tolist() + pose.rotation().matrix().flatten().tolist()
        else: pose_flatten = extract_xyzrpy(pose)

        position_loss[i] = np.linalg.norm(pose.translation() - X_tensor_cpu.numpy()[i][:3])
        backwards_loss[i] = np.linalg.norm(pose_flatten - X_tensor_cpu.numpy()[i][:-1])

    print(f"Backwards Loss: {np.mean(backwards_loss):.6f}")
    print(f"Position Loss: {np.mean(position_loss):.6f}")
    print(f"Test Loss: {loss.item():.6f}")

def normalize_data(X, y):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0) + 1e-8
    
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std
    
    return X_norm, y_norm, X_mean, X_std, y_mean, y_std

def denormalize_output(y_norm, y_mean, y_std):
    return y_norm * y_std + y_mean


if __name__ == "__main__":
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    meshcat = StartMeshcat()
    diagram = BuildEnv(meshcat, directives_file = os.path.join(RepoDir(), "models/iiwa_collision.yaml"))
    ee_frame = diagram.GetSubsystemByName("plant").GetFrameByName("body")

    dataset = IKDataset(diagram, ee_frame)

    joint_angles, targets = dataset.create_data(5**7, manifold = [1, 1, 1], matrix=True)

    X = np.concatenate([targets, joint_angles[:, 0:1]], axis=1)
    y = joint_angles

    model = LinearModule(X.shape[1], 7, hidden_layers = [64, 64, 64], dropout_rate=0.0)
    model.to(device)  # Move model to GPU
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    losses = train_model(model, optimizer, loss_fn, X_train, y_train, epochs=1000, batch_size=2048, device=device)
    evaluate_model(model, loss_fn, X_test, y_test, dataset, matrix=True, device=device)