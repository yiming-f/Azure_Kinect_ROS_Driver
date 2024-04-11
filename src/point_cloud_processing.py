import rospy
from sensor_msgs.msg import PointCloud2
import open3d as o3d
from PyQt5 import QtWidgets
import ros_numpy
import numpy as np

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
    pc = pc[:, :3]
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
        vol = o3d.visualization.read_selection_polygon_volume("cropped.json")

        if not isinstance(self.vis, o3d.visualization.Visualizer):
            self.vis = o3d.visualization.Visualizer()
        self.first = False
        while (self.subscriber.pc is None):
            rospy.sleep(2)
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(to_world(transformationMatrix, ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.subscriber.pc)))
        self.point_cloud = vol.crop_point_cloud(self.point_cloud)
        self.vis.create_window()
        print('get points')
        self.vis.add_geometry(self.point_cloud)

        # Add an xyz axis
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        self.vis.add_geometry(mesh_frame)

        print ('add points')
        self.vis.update_geometry(self.point_cloud)
        self.vis.poll_events()

        self.vis.update_renderer()

        while not rospy.is_shutdown():
            self.point_cloud.points = o3d.utility.Vector3dVector(to_world(transformationMatrix, ros_numpy.point_cloud2.pointcloud2_to_xyz_array(self.subscriber.pc)))
            vol = o3d.visualization.read_selection_polygon_volume("cropped.json")
            cropped_pc = vol.crop_point_cloud(self.point_cloud)
            self.point_cloud.points = cropped_pc.points
            self.vis.update_geometry(self.point_cloud)
            self.vis.poll_events()
            self.vis.update_renderer()

if __name__ == '__main__':
    print('Starting Azure Kinect listener')
    listener = azure_kinect_listener()
    updater = viewer(listener)
    rospy.spin()