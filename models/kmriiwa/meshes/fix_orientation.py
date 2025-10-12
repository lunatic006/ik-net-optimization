import sys
import numpy as np

def rotate_vertex(v):
    # 90 degrees about X-axis
    R = np.array([
        [1, 0,  0],
        [0, 0, -1],
        [0, 1,  0]
    ])
    return R @ v

def fix_obj_orientation(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.startswith('v '):
            parts = line.strip().split()
            v = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            v_rot = rotate_vertex(v)
            new_lines.append(f"v {v_rot[0]:.6f} {v_rot[1]:.6f} {v_rot[2]:.6f}\n")
        else:
            new_lines.append(line)

    with open(filename, 'w') as f:
        f.writelines(new_lines)

if __name__ == "__main__":
    fix_obj_orientation(sys.argv[1])
