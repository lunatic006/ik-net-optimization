from ikflow.model_loading import get_ik_solver
from ikflow.config import DEVICE
import torch
import numpy as np
from time import time, sleep
from dataclasses import dataclass, field
import sys, os
from src.utils import RepoDir, BuildEnv, DrawAxes
import numpy as np
from pydrake.all import (
    StartMeshcat, 
    RigidTransform,
    Quaternion,
    MathematicalProgram,
    AutoDiffXd, 
    Quaternion_, 
    IpoptSolver,
    SolverOptions,
    CommonSolverOption, 
    MinimumDistanceLowerBoundConstraint
)

@dataclass
class ProgramOptions:
    joint_centering_cost: float = field(default=0.0, metadata={"help": "Weight for joint centering cost"})
    collision_avoidance: bool = field(default=True, metadata={"help": "Add collision avoidance constraints"})
    joint_limits: bool = field(default=True, metadata={"help": "Enforce joint limits"})
    ik_constraint_tol: tuple = field(default=(1e-4, 0.01), metadata={"help": "Tolerance for IK constraints: tuple of (position tol, orientation tol)"})
    correction_cost_weight: float = field(default=0.0, metadata={"help": "Weight for correction cost to keep close to zero"})


    ## Solver options ##
    which_solver: str = field(default="ipopt", metadata={"help": "Which IKFlow solver to use"})
    acceptable_tol: float = field(default=1e-4, metadata={"help": "Acceptable tolerance for solver convergence"})
    acceptable_dual_inf_tol: float = field(default=1e-4, metadata={"help": "Acceptable dual infeasibility tolerance for solver convergence"})
    acceptable_compl_inf_tol: float = field(default=1e-4, metadata={"help": "Acceptable complementary infeasibility tolerance for solver convergence"})
    acceptable_constr_viol_tol: float = field(default=1e-4, metadata={"help": "Acceptable constraint violation tolerance for solver convergence"})
    file_print_level: int = field(default=5, metadata={"help": "File print level for the solver"})
    file_print_name: str = field(default="ikflow_solver_log.txt", metadata={"help": "File name for solver log"})
    max_wall_time: float = field(default=60, metadata={"help": "Maximum wall time for the solver in seconds"})





class IKFlowProgram:
    def __init__(self, diagram, frame, solver, options=ProgramOptions()):
        self.diagram = diagram
        self.plant = diagram.GetSubsystemByName("plant")
        self.autodiff_plant = self.plant.ToAutoDiffXd()
        self.diagram_context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.diagram_context)
        self.autodiff_context = self.autodiff_plant.CreateDefaultContext()
        self.diagram.ForcedPublish(self.diagram_context)

        self.frame = frame
        self.autodiff_frame = self.autodiff_plant.GetBodyByName(frame.name()).body_frame()

        self.ik_solver = solver
        self.ik_solver.nn_model.eval()
        self.options = options

    def fk(self, q):
        if isinstance(q[0], AutoDiffXd):
            self.autodiff_plant.SetPositions(self.autodiff_context, q)
            rigid_transform = self.autodiff_frame.CalcPoseInWorld(self.autodiff_context)

        else:
            self.plant.SetPositions(self.plant_context, q)
            rigid_transform = self.frame.CalcPoseInWorld(self.plant_context)
        return rigid_transform.translation(), rigid_transform.rotation().ToQuaternion().wxyz()


    ## These are Robot Specific need to be implemented in each file ##
    def ik_inference(self, vars):
        pass
    def VarsToQ(self, vars):
        pass
    
    def EvalPositionError(self, vars):
        q = self.VarsToQ(vars)
        position, _ = self.fk(q)
        pos_error = position - self.target_pose[:3]
        return pos_error
    def EvalOrientationError(self, vars):
        q = self.VarsToQ(vars)
        _, orientation = self.fk(q)
        orientation_error = 2 * np.arccos(np.abs(np.dot(orientation, self.target_pose[3:])))
        return np.array([orientation_error])
    
    def IKConstraint(self): ## extremely nonlinear equality constraint !!!
        pos_tol, ori_tol = self.options.ik_constraint_tol
        self.position_constraint = self.prog.AddConstraint(
            self.EvalPositionError, 
            lb=np.zeros(3) - pos_tol, 
            ub=np.zeros(3) + pos_tol,
            vars=self.lumped_vars
        )
        self.position_constraint.evaluator().set_description("PositionConstraint")
        self.orientation_constraint = self.prog.AddConstraint(
            self.EvalOrientationError,
            lb=np.array([0]),
            ub=np.array([ori_tol]), # allow extremely small error
            vars=self.lumped_vars
        )
        self.orientation_constraint.evaluator().set_description("OrientationConstraint")

    def CollisionFreeConstraint(self):
        self.collision_free_constraint = MinimumDistanceLowerBoundConstraint(
            plant=self.plant,
            bound=1e-3,
            influence_distance_offset=1e-1,
            plant_context=self.plant_context
        )
        self.collision_constraint = self.prog.AddConstraint(
            self.EvalCollisionFreeConstraint,
            lb=np.array([-np.inf]),
            ub = np.array([1]),
            vars=self.lumped_vars
        )
        self.collision_constraint.evaluator().set_description("CollisionFreeConstraint")
    
    def EvalCollisionFreeConstraint(self, vars):
        q = self.VarsToQ(vars)
        return self.collision_free_constraint.Eval(q)
    
    def JointLimitsConstraint(self):
        lower_limits = self.plant.GetPositionLowerLimits()
        upper_limits = self.plant.GetPositionUpperLimits()

        self.joint_limits_constraint = self.prog.AddConstraint(
            self.VarsToQ,
            lb=lower_limits,
            ub=upper_limits,
            vars=self.lumped_vars
        )
        self.joint_limits_constraint.evaluator().set_description("JointLimitsConstraint")
    
    def JointCenteringCost(self):
        self.joint_centering_cost = self.prog.AddCost(
            func = self.EvalJointCenteringCost,
            vars = self.lumped_vars
        )
        self.joint_centering_cost.evaluator().set_description("JointCenteringCost")
    
    def EvalJointCenteringCost(self, vars):
        q = self.VarsToQ(vars)
        diff = q - self.q_nominal
        return 0.5 * diff @ (self.options.joint_centering_cost * np.eye(9)) @ diff

    def BoundingBoxConstraint(self):
        z_lower_bound = -1.5
        z_upper_bound = 1.5
        self.bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            z_lower_bound * np.ones(self.ik_solver.network_width),
            z_upper_bound * np.ones(self.ik_solver.network_width),
            self.z
        )
        self.bounding_box_constraint.evaluator().set_description("ZBoundingBoxConstraint")
        c_lower_bound = self.target_pose - 5
        c_upper_bound = self.target_pose + 5
        self.c_bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            c_lower_bound,
            c_upper_bound,
            self.c
        )
        self.c_bounding_box_constraint.evaluator().set_description("CBoundingBoxConstraint")
        correction_lower_bound = -0.4 * np.ones(7)
        correction_upper_bound = 0.4 * np.ones(7)
        self.correction_bounding_box_constraint = self.prog.AddBoundingBoxConstraint(
            correction_lower_bound,
            correction_upper_bound,
            self.correction
        )
        self.correction_bounding_box_constraint.evaluator().set_description("CorrectionBoundingBoxConstraint")
    
    def CorrectionCost(self):
        self.correction_cost = self.prog.AddQuadraticCost(
            Q=self.options.correction_cost * np.eye(7),
            b=np.zeros(7),
            vars=self.correction
        )
        self.correction_cost.evaluator().set_description("CorrectionCost")
    

    def Solve(self):
        solver = IpoptSolver()
        solver_options = SolverOptions()
        solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_tol", 1e-4)
        solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_dual_inf_tol", 1e-4)
        solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_compl_inf_tol", 1e-4)
        solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_constr_viol_tol", 1e-6)
        solver_options.SetOption(IpoptSolver().solver_id(), "file_print_level", 5)
        solver_options.SetOption(IpoptSolver().solver_id(), "print_user_options", "yes")
        solver_options.SetOption(CommonSolverOption.kPrintFileName, "ipopt_output.txt")
        solver_options.SetOption(IpoptSolver().solver_id(), "max_wall_time", 60.0)
        
        return solver.Solve(self.prog, solver_options=solver_options)

