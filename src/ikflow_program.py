from ikflow.model_loading import get_ik_solver
from ikflow.config import DEVICE
import torch
import numpy as np
from time import time, sleep
import sys
import os
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
    CommonSolverOption
)

class PandaIKProgram:
    def __init__(self, diagram):
        self.diagram = diagram
        self.plant = diagram.GetSubsystemByName("plant")
        self.autodiff_plant = self.plant.ToAutoDiffXd()
        self.diagram_context = diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.diagram_context)
        self.autodiff_context = self.autodiff_plant.CreateDefaultContext()
        self.diagram.ForcedPublish(self.diagram_context) 

        self.frame = self.plant.GetBodyByName("panda_link8").body_frame()
        self.autodiff_frame = self.autodiff_plant.GetBodyByName("panda_link8").body_frame()

        model_name = "panda__full__lp191_5.25m"
        self.ik_solver, self.hyper_parameters = get_ik_solver(model_name)
        self.ik_solver.nn_model.eval()


    def create_prog(self, target_pose):
        self.prog = MathematicalProgram()
        self.c = self.prog.NewContinuousVariables(7) # x y z qw qx qy qz into nn model
        self.z = self.prog.NewContinuousVariables(self.ik_solver.network_width) # latent variables

        self.lumped_vars = np.hstack([self.c, self.z])

        ## TODO: Change the initial guess to something smarter

        self.target_pose = target_pose

        self.prog.SetInitialGuess(self.c, target_pose)
        self.prog.SetInitialGuess(self.z, np.random.randn(self.ik_solver.network_width))
        self.jacobian_gen = torch.func.jacrev(self.ik_inference) ## function that can compute jacobian dq/dvars

        ## Add Constraints
        self.IKConstraint()
        # self.IKCost()
        self.JointLimitsConstraint()
        self.BoundingBoxConstraint()



    def ik_inference(self, vars):
        '''Given a latent + target, returns corresponding joint angles
        vars can be either numpy array or torch tensor (for gradient computation)'''
        # Convert to tensor only if not already a tensor
        if not isinstance(vars, torch.Tensor):
            vars = torch.tensor(vars, device=DEVICE, dtype=torch.float32)
        
        c, z = (vars[:7], vars[7:])
        # Work directly with tensor slices - don't call torch.tensor() again!
        c_torch = torch.cat([c.unsqueeze(0), torch.zeros((1, 1), dtype=torch.float32, device=DEVICE)], dim=1)
        z_batch = z.unsqueeze(0)

        output, _ = self.ik_solver.nn_model(z_batch, c=c_torch, rev=True)
        q = output[:, :7].squeeze(0)
        return q
    
    def fk(self, q):
        if isinstance(q[0], AutoDiffXd):
            self.autodiff_plant.SetPositions(self.autodiff_context, q)
            rigid_transform = self.autodiff_frame.CalcPoseInWorld(self.autodiff_context)

        else:
            self.plant.SetPositions(self.plant_context, q)
            rigid_transform = self.frame.CalcPoseInWorld(self.plant_context)
        return rigid_transform.translation(), rigid_transform.rotation().ToQuaternion().wxyz()
        

    def VarsToQ(self, vars):
        ad = isinstance(vars[0], AutoDiffXd) ## The hard part is to make torch interact with AutoDiffXd necessary for drake.

        if not ad:
            q = np.zeros(9)
            q[:7] = self.ik_inference(vars).detach().cpu().numpy()
            return q
        
        else: # Compute AutoDiffXd with Jacobian_Gen
            # Extract values and gradients from AutoDiffXd
            vars_values = np.array([v.value() for v in vars])
            vars_gradients = np.array([v.derivatives() for v in vars])
            
            # Compute q values
            q_values = np.zeros(9)
            q_values[:7] = self.ik_inference(vars_values).detach().cpu().numpy()
            
            # Compute Jacobian dq/dvars
            vars_tensor = torch.tensor(vars_values, dtype=torch.float32, device=DEVICE, requires_grad=True)
            jacobian = self.jacobian_gen(vars_tensor)
            jacobian_np = jacobian.detach().cpu().numpy()
            
            # Chain rule: dq/dvars @ dvars = dq
            # For each element of q, compute gradient via chain rule
            q_gradients = np.zeros((9, len(vars)))
            q_gradients[:7, :] = jacobian_np @ vars_gradients
            
            # Create AutoDiffXd objects with value and gradient
            q_ad = np.array([AutoDiffXd(q_values[i], q_gradients[i]) for i in range(len(q_values))])
            
            return q_ad
    
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
    
    def IKConstraint(self):
        self.position_constraint = self.prog.AddConstraint(
            self.EvalPositionError, 
            lb=np.zeros(3),
            ub=np.zeros(3),
            vars=self.lumped_vars
        )
        self.position_constraint.evaluator().set_description("PositionConstraint")
        # self.orientation_constraint = self.prog.AddConstraint(
        #     self.EvalOrientationError,
        #     lb=np.array([0]),
        #     ub=np.array([0.3]), # allow small orientation error
        #     vars=self.lumped_vars
        # )
        # self.orientation_constraint.evaluator().set_description("OrientationConstraint")
    
    def EvalCost(self, vars):
        position, orientation = self.fk(self.VarsToQ(vars))
        # error = np.linalg.norm(np.concatenate([position, orientation], axis=0) - self.target_pose)
        error = np.linalg.norm(position - self.target_pose[:3])
        return error

    def IKCost(self):
        self.ik_cost = self.prog.AddCost(
            self.EvalCost,
            vars=self.lumped_vars
        )
        self.ik_cost.evaluator().set_description("IKCost")


    def JointLimitsConstraint(self):
        lower_limits = self.plant.GetPositionLowerLimits()[:-2]
        upper_limits = self.plant.GetPositionUpperLimits()[:-2]

        self.joint_limits_constraint = self.prog.AddConstraint(
            lambda vars: self.VarsToQ(vars)[:7],
            lb=lower_limits,
            ub=upper_limits,
            vars=self.lumped_vars
        )
        self.joint_limits_constraint.evaluator().set_description("JointLimitsConstraint")
    def BoundingBoxConstraint(self):
        z_lower_bound = -1.5
        z_upper_bound = 1.5
        self.bounding_box_constraint = self.prog.AddConstraint(
            lambda vars: vars[7:],  # Extract latent variables z
            lb=z_lower_bound * np.ones(self.ik_solver.network_width),
            ub=z_upper_bound * np.ones(self.ik_solver.network_width),
            vars=self.lumped_vars
        )
        self.bounding_box_constraint.evaluator().set_description("ZBoundingBoxConstraint")
        c_lower_bound = self.target_pose - 0.1
        c_upper_bound = self.target_pose + 0.1
        self.c_bounding_box_constraint = self.prog.AddConstraint(
            lambda vars: vars[:7],  # Extract c variables
            lb=c_lower_bound,
            ub=c_upper_bound,
            vars=self.lumped_vars
        )
        self.c_bounding_box_constraint.evaluator().set_description("CBoundingBoxConstraint")



    def Solve(self):
        solver = IpoptSolver()
        solver_options = SolverOptions()
        solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_tol", 1e-4)
        solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_dual_inf_tol", 1e-4)
        solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_compl_inf_tol", 1e-2)
        solver_options.SetOption(IpoptSolver().solver_id(), "acceptable_constr_viol_tol", 1e-6)
        solver_options.SetOption(IpoptSolver().solver_id(), "file_print_level", 5)
        solver_options.SetOption(IpoptSolver().solver_id(), "print_user_options", "yes")
        solver_options.SetOption(CommonSolverOption.kPrintFileName, "ipopt_output.txt")

        
        return solver.Solve(self.prog, solver_options=solver_options)




if __name__ == "__main__":
    a = AutoDiffXd(0, np.array([1.0, 2.0, 3.0]))