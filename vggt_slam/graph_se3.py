import gtsam
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gtsam import Pose3, Rot3, Point3, NonlinearFactorGraph, Values, noiseModel, PriorFactorPose3
from gtsam.symbol_shorthand import X
# from gnss_processer import GNSSprocesser
class PoseGraph:
    def __init__(self):
        """Initialize a factor graph for Pose3 nodes with BetweenFactors."""
        self.graph = NonlinearFactorGraph()
        self.values = Values()
        self.relative_noise = noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
        self.anchor_noise = noiseModel.Diagonal.Sigmas([1e-6] * 6)
        self.gnss_noise = noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0, 1e6, 1e6, 1e6]))  # GNSS noise model
        self.initialized_nodes = set()
        self.num_loop_closures = 0 # Just used for debugging and analysis
        # self.gnss_processer = GNSSprocesser()
    def add_gnss_constraint(self, key, gnss_position, rotation=None):
        """Add a GNSS position constraint to the graph"""
        key = X(key)
        if key not in self.initialized_nodes:
            raise ValueError(f"Node {key} not in graph for GNSS constraint")
        
        # Get current pose to preserve rotation
        current_pose = self.values.atPose3(key)
        R = current_pose.rotation().matrix()
        t = gnss_position
        
        # Create pose with GNSS position but visual rotation
        gnss_pose = np.eye(4)
        gnss_pose[0:3, 0:3] = R
        gnss_pose[0:3, 3] = t
        
        # Add constraint
        self.graph.add(PriorFactorPose3(key, Pose3(gnss_pose), self.gnss_noise))
        print(f"Added GNSS constraint for node {key} at position {t}")
    def add_homography(self, key, pose):
        """Add a new pose node to the graph."""
        key = X(key)
        if key in self.initialized_nodes:
            print(f"Pose {key} already exists.")
            return
        self.values.insert(key, Pose3(pose))
        self.initialized_nodes.add(key)

    def add_between_factor(self, key1, key2, relative_pose, noise):
        """Add a relative Pose3 constraint between two nodes."""
        key1 = X(key1)
        key2 = X(key2)
        if key1 not in self.initialized_nodes or key2 not in self.initialized_nodes:
            raise ValueError(f"Both poses {key1} and {key2} must exist before adding a factor.")
        self.graph.add(gtsam.BetweenFactorPose3(key1, key2, Pose3(relative_pose), noise))
    
    def add_prior_factor(self, key, pose, noise):
        key = X(key)
        if key not in self.initialized_nodes:
            raise ValueError(f"Trying to add prior factor for key {key} but it is not in the graph.")
        self.graph.add(PriorFactorPose3(key, Pose3(pose), noise))

    def get_homography(self, node_id):
        """
        Get the optimized SO3 pose at a specific node.
        :param node_id: The ID of the node.
        :return: gtsam.Pose3 pose of the node.
        """
        node_id = X(node_id)
        return self.values.atPose3(node_id)

    def optimize(self):
        """Optimize the graph and update estimates."""
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values)
        result = optimizer.optimize()
        self.values = result  # Update values with optimized results
    def test_optimize(self):
        # 运行优化
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SUMMARY")  # 打印优化摘要
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.values, params)
        result = optimizer.optimize()
        
        # 更新优化结果
        self.values = result
    def print_estimates(self):
        """Print the optimized poses."""
        for key in sorted(self.initialized_nodes):
            print(f"Pose {key}:\n{self.values.atPose3(key)}\n")

    def increment_loop_closure(self):
        """Increment the loop closure count."""
        self.num_loop_closures += 1
    
    def get_num_loops(self):
        """Get the number of loop closures."""
        return self.num_loop_closures