Copied from [RobotLocomotion/Hubo](https://github.com/robotLocomotion/Hubo) with slight modification to fix file/package paths.

One major change: the kinematics of the arms has been modified to match the idealized version considered in the following papers, so that we can use their analytic IK solution.

- *Closed-Form Inverse Kinematic Position Solution for Humanoid Robots,* Park, Ali, and Lee; International Journal of Humanoid Robotics; 2012.
- *Kinematics and Inverse Kinematics for the Humanoid Robot HUBO2+,* O'Flaherty, Vieira, Grey, Oh, Bobick, Egerstedt, and Stilman; Georgia Institute of Technology Technical Report; 2013.

This involves moving the fourth joint such that the axes of the third and fifth joints pass through its origin (using one-indexing).