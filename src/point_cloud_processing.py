#!/home/yimingf/.pyenv/versions/flowbot3d/bin/python
import rospy
import torch
print("torch version: ", torch.__version__)
import torch_geometric.data as tgd
from sensor_msgs.msg import PointCloud2
import open3d as o3d
from PyQt5 import QtWidgets
import ros_numpy
import numpy as np
from flowbot3d.models.flowbot3d import ArtFlowNet, ArtFlowNetParams
from scipy.spatial.transform import Rotation 
from torch_geometric.nn import PointConv, fps, global_max_pool, knn_interpolate, radius
# from open_anything_diffusion.models.modules.dit_models import DGDiT, DiT
# from open_anything_diffusion.models.flow_diffuser_dit import FlowTrajectoryDiffuserInferenceModule_DiT
# from open_anything_diffusion.models.flow_diffuser_dgdit import FlowTrajectoryDiffuserInferenceModule_DGDiT
import copy

print("Is CUDA available? ", torch.cuda.is_available())

# inference_module_class = {
#     "dit": FlowTrajectoryDiffuserInferenceModule_DiT,
#     "dgdit": FlowTrajectoryDiffuserInferenceModule_DGDiT,
# }
# networks = {
#     "dit": DiT(in_channels=6, depth=5, hidden_size=128, num_heads=4, learn_sigma=True),
#     "dgdit": DGDiT(in_channels=3, depth=5, hidden_size=128, patch_size=1, num_heads=4, n_points=1200),
# }

# class InferenceConfig:
#     def __init__(self):
#         self.batch_size = 1
#         self.trajectory_len = 1

# inference_config = InferenceConfig()

# class ModelConfig:
#     def __init__(self):
#         self.num_train_timesteps = 100

# model_config = ModelConfig()

# import os
# ckpt_dir = './pretrained'
# train_type = 'fullset_half_half'   # door_half_half, fullset_half_half - what dataset the model is trained on 
# model_type = 'dit'   # dit, dgdit - model structure
# ckpt_path = os.path.join(ckpt_dir, f'{train_type}_{model_type}.ckpt')

# diff_model = inference_module_class[model_type](
#     networks[model_type].cuda(), inference_cfg=inference_config, model_cfg=model_config
# ).cuda()
# diff_model.load_from_ckpt(ckpt_path)

# ckpt_file = '/home/yimingf/catkin_ws/src/flowbot3d/pretrained/model_nomask_vpa.ckpt'
# params = ArtFlowNetParams(mask_input_channel=False)
# model = ArtFlowNet.load_from_checkpoint(ckpt_file, p=params).cuda()

def transform_xyz_to_flowbot_frame(xyz):
    # Rotation matrix to rotate counter-clockwise 90 degrees around the z-axis
    R = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).float().cuda()
    xyz = xyz @ R.T

    # Run FPS to downsample the point cloud to 1200 points

    # Subtract the mean in x and y
    mean_x = xyz[:, 0].mean()
    mean_y = xyz[:, 1].mean()
    xyz[:, 0] -= mean_x
    xyz[:, 1] -= mean_y
    return xyz, mean_x, mean_y

def get_goal_point_and_orientation(contact_point, flow_vector):
    print("contact point device: ", contact_point.device)
    print("flow vector device: ", flow_vector.device)
    print("contact point dim: ", contact_point.dim())
    print("flow vector dim: ", flow_vector.dim())
    print("contact point", contact_point)
    print("flow vector", flow_vector)
    contact_point = contact_point.cuda()
    flow_vector = flow_vector.cuda()
    goal_point = contact_point + 0.2 * flow_vector
    e_z_init = torch.tensor([0, 0, 1.0]).float().cuda()
    e_y = -flow_vector
    e_x = torch.linalg.cross(e_y, e_z_init)
    e_x = e_x / e_x.norm(dim=-1)
    e_z = torch.linalg.cross(e_x, e_y)
    R_goal = torch.stack([e_x, e_y, e_z], dim=1).cuda()
    R_gripper = torch.as_tensor(
        [
            [1, 0, 0],
            [0, 0, 1.0],
            [0, -1.0, 0],
        ]
    ).cuda()

    goal_orientation = Rotation.from_matrix((R_goal @ R_gripper).cpu()).as_quat()
    return goal_point, goal_orientation

def transform_contact_point_goal_point_and_orientation_to_world(contact_point, goal_point, goal_orientation, mean_x, mean_y):
    # Formatting goal_point and goal_orientation to be in the same frame as the point cloud so that it can be visualized
    R = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).float().cuda()
    print("goal_point before transformation: ", goal_point)
    # Add the mean in x and y
    goal_point[0] += mean_x
    goal_point[1] += mean_y
    contact_point[0] += mean_x
    contact_point[1] += mean_y
    print("goal_point after translation: ", goal_point)
    goal_point = goal_point @ R.T
    contact_point = contact_point @ R.T
    
    print("goal_point after rotation: ", goal_point)
    goal_point = goal_point.cpu().numpy()
    goal_point = np.reshape(goal_point, (1, 3))
    contact_point = contact_point.cpu().numpy()
    contact_point = np.reshape(contact_point, (1, 3))

    goal_orientation = Rotation.from_quat(goal_orientation)
    goal_orientation = torch.from_numpy(goal_orientation.as_matrix()).float().cuda()
    goal_orientation = goal_orientation @ R.T
    goal_orientation = Rotation.from_matrix(goal_orientation.cpu()).as_quat()

    return contact_point, goal_point, goal_orientation

def get_contact_point_and_flow_vector(flow, xyz):
    print("flow shape: ", flow.shape)
    print("xyz shape: ", xyz.shape)
    magnitude = torch.norm(flow, dim=1)
    idx_of_max_flow = torch.argmax(magnitude.unsqueeze(1))
    contact_point = xyz[idx_of_max_flow]
    flow_vector = flow[idx_of_max_flow] 
    flow_vector_normalized = (flow_vector / flow_vector.norm(dim=-1)).float()
    return contact_point, flow_vector_normalized

rmatrix = [[-0.7343, -0.3504, 0.5814], 
           [-0.6787, 0.3962, -0.6184], 
           [-0.0137, -0.8487, -0.5287]]
tvector = [[0.0503], [0.4724], [0.5912]]
transformationMatrix = np.asarray([[-0.7343, -0.3504, 0.5814, 0.0503], 
                        [-0.6787, 0.3962, -0.6184, 0.4724], 
                        [-0.0137, -0.8487, -0.5287, 0.5912], 
                        [0, 0, 0, 1]])
# vol = o3d.visualization.read_selection_polygon_volume("cropped.json")

class azure_kinect_listener():
    def __init__(self):
        print('in init')
        self.pc = None
        self.n = 0
        self.listener()

    def callback(self, points):
        self.pc = points
        self.n = self.n + 1

    def listener(self):
        rospy.init_node('azure_kinect_listener')
        rospy.Subscriber('points2', PointCloud2, self.callback)

def to_world(pose, pc):
    # pose is 4x4, pc is nx3
    # pose = np.linalg.inv(pose)
    pc = np.hstack((pc, np.ones((pc.shape[0], 1))))
    pc = np.dot(pose, pc.T).T

    # Rotate the point cloud 5 degrees around the y axis
    rad = 2.5 * np.pi / 180
    R = np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]])
    pc[:, :3] = np.dot(R, pc[:, :3].T).T
    
    # Rotate the point cloud 1 degree around the x axis
    rad = 2 * np.pi / 180
    R = np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]])
    pc[:, :3] = np.dot(R, pc[:, :3].T).T

    pc = pc[:, :3]
    mask_x = np.logical_and(pc[:, 0] >= 0, pc[:, 0] <= 0.73)
    mask_y = np.logical_and(pc[:, 1] >= -0.57, pc[:, 1] <= 0.75)
    mask_z = np.logical_and(pc[:, 2] >= 0.00, pc[:, 2] <= 1)
    combined_mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
    pc = pc[combined_mask]
    np.save('pc_data_for_yishu/incorrect_toilet.npy', pc)
    return pc

class viewer(QtWidgets.QWidget):
    def __init__(self, subscriber, parent=None):
        self.subscriber = subscriber
        print('initialization of viewer')

        self.vis = o3d.visualization.Visualizer()
        self.point_cloud = None
        self.updater()

    def updater(self):
        print('starting viewer')
        # vol = o3d.visualization.read_selection_polygon_volume("cropped.json")

        if not isinstance(self.vis, o3d.visualization.Visualizer):
            self.vis = o3d.visualization.Visualizer()
        self.first = False
        while (self.subscriber.pc is None):
            rospy.sleep(2)
        self.point_cloud = o3d.geometry.PointCloud()
        # pc = np.load('pc_data_for_yishu/toilet_seat_closed.npy')
        self.point_cloud.points = o3d.utility.Vector3dVector(to_world(transformationMatrix, ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.subscriber.pc)))
        xyz = torch.from_numpy(to_world(transformationMatrix, ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.subscriber.pc))).float().cuda()
        xyz, mean_x, mean_y = transform_xyz_to_flowbot_frame(xyz)
        ixs = fps(xyz, ratio=1200/xyz.shape[0])
        xyz = xyz[ixs]
        numpy_xyz = xyz.cpu().numpy()
        print("xyz shape: ", xyz.shape)
        print("numpy_xyz shape: ", numpy_xyz.shape)
        
        flow = model.predict(xyz, torch.zeros(xyz.shape[0]).float().cuda()).cuda()
        diff_flow = diff_model.predict(numpy_xyz)[:, 0, :]

        batch = tgd.Batch(pos=xyz, mask=torch.zeros(xyz.shape[0]).float().cuda(), id="stfu", flow=flow).cpu()
        figs = ArtFlowNet.make_plots(flow.cpu(), batch)
        for fig in figs.values():
            fig.show()
        
        diff_batch = tgd.Batch(pos=xyz, mask=torch.zeros(xyz.shape[0]).float().cuda(), id="stfu", flow=diff_flow).cpu()
        figs = ArtFlowNet.make_plots(diff_flow.cpu(), diff_batch)
        for fig in figs.values():
            fig.show()

        contact_point, flow_vector_normalized = get_contact_point_and_flow_vector(flow, xyz)
        diff_contact_point, diff_flow_vector_normalized = get_contact_point_and_flow_vector(diff_flow, xyz)

        goal_point, goal_orientation = get_goal_point_and_orientation(contact_point, flow_vector_normalized)
        contact_point, goal_point, goal_orientation = transform_contact_point_goal_point_and_orientation_to_world(contact_point, goal_point, goal_orientation, mean_x, mean_y)
        diff_goal_point, diff_goal_orientation = get_goal_point_and_orientation(diff_contact_point, diff_flow_vector_normalized)
        diff_contact_point, diff_goal_point, diff_goal_orientation = transform_contact_point_goal_point_and_orientation_to_world(diff_contact_point, diff_goal_point, diff_goal_orientation, mean_x, mean_y)

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        sphere.translate(contact_point[0])
        diff_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        diff_sphere.translate(diff_contact_point[0])
        diff_sphere.paint_uniform_color([1, 0, 0])
    
        #make goal_orientation a coordinate frame with origin at goal_point

        goal_orientation_frame = (o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=goal_point[0]))
        goal_orientation_frame_r = copy.deepcopy(goal_orientation_frame)
        rot = goal_orientation_frame.get_rotation_matrix_from_quaternion(goal_orientation)
        goal_orientation_frame_r.rotate(rot, center=goal_point[0])
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

        diff_goal_orientation_frame = (o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=diff_goal_point[0]))
        diff_goal_orientation_frame_r = copy.deepcopy(diff_goal_orientation_frame)
        diff_rot = diff_goal_orientation_frame.get_rotation_matrix_from_quaternion(diff_goal_orientation)
        diff_goal_orientation_frame_r.rotate(diff_rot, center=diff_goal_point[0])

        #create visualization window and add the goal point and orientation vector to it
        self.vis.create_window()
        self.vis.add_geometry(goal_orientation_frame_r)
        self.vis.add_geometry(sphere)
        self.vis.add_geometry(self.point_cloud)
        self.vis.add_geometry(mesh_frame)
        self.vis.add_geometry(diff_sphere)
        self.vis.add_geometry(diff_goal_orientation_frame_r)

        self.vis.poll_events()
        self.vis.update_renderer()

        while not rospy.is_shutdown():
            # self.point_cloud.points = o3d.utility.Vector3dVector(to_world(transformationMatrix, ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.subscriber.pc)))
            # self.vis.add_geometry(self.point_cloud)
            self.vis.poll_events()
            self.vis.update_renderer()
            rospy.sleep(0.1)

if __name__ == '__main__':
    print('Starting Azure Kinect listener')
    listener = azure_kinect_listener()
    updater = viewer(listener)
    rospy.spin()
