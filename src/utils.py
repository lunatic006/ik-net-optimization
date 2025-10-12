import os
import numpy as np
from pydrake.all import(
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    Parser,
    ProcessModelDirectives,
    LoadModelDirectives,
    ApplyVisualizationConfig,
    VisualizationConfig,
    RigidTransform,
    Rgba,
    Sphere,
    Cylinder,
    RotationMatrix,
    AutoDiffXd,
    RollPitchYaw_,
)

def RepoDir():
    return os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def BuildEnv(meshcat, directives_file=None):
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.1)

    # Load the model directives from the YAML file.
    parser = Parser(plant, scene_graph)
    package_xml_path = os.path.join(RepoDir(), "package.xml")
    parser.package_map().AddPackageXml(package_xml_path)
    ProcessModelDirectives(LoadModelDirectives(directives_file), plant, parser)

    plant.Finalize()
    vis_config = VisualizationConfig()
    vis_config.publish_illustration = True
    vis_config.publish_proximity = True
    vis_config.publish_inertia = True
    vis_config.delete_on_initialization_event = True   # Clear old visualizations
    ApplyVisualizationConfig(vis_config, builder, meshcat=meshcat)

    diagram = builder.Build()
    return diagram


def BuildEnvWithDynamicMugs(meshcat, mug_positions, directives_file=None):
    """
    Build environment with dynamically added collision geometry for mugs.

    Args:
        meshcat: Meshcat visualizer instance
        mug_positions: Array of shape (n, 2, 3) containing start and end positions for each mug
        directives_file: Path to the directives file for the base environment

    Returns:
        diagram: The built diagram with collision geometry for all mugs
    """
    if directives_file is None:
        directives_file = os.path.join(RepoDir(), 'models/iiwa_collision.yaml')

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.01)

    # Load the base environment
    parser = Parser(plant, scene_graph)
    package_xml_path = os.path.join(RepoDir(), "package.xml")
    parser.package_map().AddPackageXml(package_xml_path)
    ProcessModelDirectives(LoadModelDirectives(directives_file), plant, parser)

    # Add collision geometry for each dynamic mug
    mug_model_path = os.path.join(RepoDir(), "models/mug/mug_simple_red.urdf")

    for i, mug in enumerate(mug_positions):
        start_pos = mug[0]
        end_pos = mug[1]

        # Add mug at start position
        start_mug_name = f"dynamic_mug_start_{i}"
        start_instance = parser.AddModelFromFile(mug_model_path, start_mug_name)

        # Weld the start mug to world at the specified position
        start_frame = plant.GetFrameByName("mug_body_link", start_instance)
        plant.WeldFrames(
            plant.world_frame(),
            start_frame,
            RigidTransform(RotationMatrix(), start_pos)
        )

        # Add mug at end position
        end_mug_name = f"dynamic_mug_end_{i}"
        end_instance = parser.AddModelFromFile(mug_model_path, end_mug_name)

        # Weld the end mug to world at the specified position
        end_frame = plant.GetFrameByName("mug_body_link", end_instance)
        plant.WeldFrames(
            plant.world_frame(),
            end_frame,
            RigidTransform(RotationMatrix(), end_pos)
        )

    # Finalize the plant
    plant.Finalize()

    # Apply visualization configuration
    vis_config = VisualizationConfig()
    vis_config.publish_illustration = True
    vis_config.publish_proximity = True
    vis_config.publish_inertia = True
    vis_config.delete_on_initialization_event = True
    ApplyVisualizationConfig(vis_config, builder, meshcat=meshcat)

    diagram = builder.Build()
    return diagram


def DrawSphere(target, meshcat, name="/sphere", radius = 0.03, sphere_color = Rgba(1.0, 0.2, 0.2, 0.7)):
    sphere_position = target.translation()
    sphere_radius = radius
    meshcat.Delete(name)
    meshcat.SetObject(name, Sphere(sphere_radius), sphere_color)
    meshcat.SetTransform(name, RigidTransform(sphere_position))

def DrawAxes(pose, meshcat, name="/axes", length=0.1, radius=0.005, alpha=0.8):
    """
    Draw the xyz axes given a RigidTransform.
    Each axis is drawn as a colored cylinder:
      - X: red
      - Y: green
      - Z: blue
    """
    # Remove any previous axes with this name
    meshcat.Delete(name)
    # Get rotation and translation
    R = pose.rotation().matrix()
    origin = pose.translation()

    colors = [Rgba(1, 0, 0, alpha), Rgba(0, 1, 0, alpha), Rgba(0, 0, 1, alpha)]
    axes = [R[:, 0], R[:, 1], R[:, 2]]
    axis_names = ["x", "y", "z"]

    for i, (axis, color, axis_name) in enumerate(zip(axes, colors, axis_names)):
        # Cylinder is along z by default, so compute transform from z to axis
        # Compute rotation from z to axis
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, axis)
        c = np.dot(z_axis, axis)
        if np.allclose(v, 0) and c > 0.999:
            rot = np.eye(3)
        elif np.allclose(v, 0) and c < -0.999:
            rot = np.diag([1, -1, -1])
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
            rot = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))
        # Place cylinder so its base is at the origin, then translate along axis
        cyl_pose = RigidTransform(R=RotationMatrix(rot), p=origin + axis * length / 2)
        meshcat.SetObject(f"{name}/{axis_name}", Cylinder(radius, length), color)
        meshcat.SetTransform(f"{name}/{axis_name}", cyl_pose)

def DrawCylinder(A, B, meshcat, name="/cylinder", radius=0.03, color=Rgba(0.2, 0.2, 1.0, 0.7)):
    """
    Draw a cylinder from point A to point B in meshcat using Drake geometry.
    A, B: 3D numpy arrays or lists
    meshcat: MeshcatVisualizer instance
    name: path in meshcat
    radius: cylinder radius
    color: Rgba color
    """
    A = np.asarray(A).reshape(3)
    B = np.asarray(B).reshape(3)
    axis = B - A
    length = np.linalg.norm(axis)
    if length < 1e-8:
        # Degenerate case: don't draw
        return
    # Cylinder in Drake is aligned with z-axis by default
    z_axis = np.array([0, 0, 1])
    axis_dir = axis / length

    # Compute rotation matrix that aligns z_axis to axis_dir
    v = np.cross(z_axis, axis_dir)
    c = np.dot(z_axis, axis_dir)
    if np.allclose(v, 0) and c > 0.999:
        R = np.eye(3)
    elif np.allclose(v, 0) and c < -0.999:
        R = np.diag([1, -1, -1])
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))
    # The cylinder's center is at the midpoint between A and B
    center = (A + B) / 2
    pose = RigidTransform(RotationMatrix(R), center)
    meshcat.Delete(name)
    meshcat.SetObject(name, Cylinder(radius, length), color)
    meshcat.SetTransform(name, pose)

def extract_xyzrpy(pose):
    """pose is numpy array 4x4"""
    pose = pose.GetAsMatrix4()
    ad = isinstance(pose[0, 0], AutoDiffXd)
    type_used = AutoDiffXd if ad else float
    if ad:
        result = np.full(6, AutoDiffXd(0))
    else:
        result = np.zeros(6)
    result[:3] = pose[:3, 3]
    result[3:] = RollPitchYaw_[type_used](pose[:3, :3]).vector()
    return result
