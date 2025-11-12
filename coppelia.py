import csv
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np

np.set_printoptions(precision=3, suppress=True)

def main():
    # client = RemoteAPIClient()
    # sim = client.getObject('sim')


    revolute_joints = ["right_j0", "right_j1", "right_j2", "right_j3", "right_j4", "right_j5", "right_j6"]



    # Read joint trajectory from CSV (angles in radians)
    q_trajectory = np.load('q_opt.npy')

    print(len(q_trajectory))



    # For each timestep, set all joints and print end-effector position
    for step_idx, q in enumerate(q_trajectory):
        for joint, angle in zip(revolute_joints, q):
            print(step_idx, q)
            # joint_handle = sim.getObjectHandle(joint)
            # sim.setJointPosition(joint_handle, angle)

        time.sleep(0.05)  # adjust delay as needed

if __name__ == "__main__":
    main()
