#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import tf2_ros
import tf2_geometry_msgs

aruco_marker_frame = 'fiducial_11'
base_frame = 'camera_link'


class ArucoKF:
    def __init__(self):
        # Initialize the Kalman filter
        self.kf = KalmanFilter(dim_x=6, dim_z=6)
        self.kf.x = np.zeros(6)
        self.kf.F = np.array([[1, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 1]])
        self.kf.P  *= 1000.0
        self.kf.Q = np.eye(6) * 0.01
        self.kf.R = np.eye(6) * 0.1

        # Create a tf buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Advertise the transform broadcaster for the filtered pose of the ArUco marker
        self.br = tf2_ros.TransformBroadcaster()

    def callback(self, event):
        try:
            # Look up the transform of the ArUco marker
            transform = self.tf_buffer.lookup_transform(
                base_frame, aruco_marker_frame, rospy.Time.now(), rospy.Duration(0, 200000000))

            # Extract the position and orientation from the transform
            p = np.array([transform.transform.translation.x,
                         transform.transform.translation.y, transform.transform.translation.z])
            q = np.array([transform.transform.rotation.x, transform.transform.rotation.y,
                         transform.transform.rotation.z, transform.transform.rotation.w])
            roll, pitch, yaw = euler_from_quaternion(q)

            # Pack the state vector
            x = np.hstack((p, roll, pitch, yaw))

            # Predict the state
            self.kf.predict()

            # Update the state with the measurements
            self.kf.update(x)

            # Publish the filtered pose to the TF tree
            filtered_transform = TransformStamped()
            filtered_transform.header.stamp = rospy.Time.now()
            filtered_transform.header.frame_id = base_frame
            filtered_transform.child_frame_id = aruco_marker_frame + '_kf'
            filtered_transform.transform.translation.x = self.kf.x[0]
            filtered_transform.transform.translation.y = self.kf.x[1]
            filtered_transform.transform.translation.z = self.kf.x[2]
            q = quaternion_from_euler(self.kf.x[3], self.kf.x[4], self.kf.x[5])
            filtered_transform.transform.rotation.x = q[0]
            filtered_transform.transform.rotation.y = q[1]
            filtered_transform.transform.rotation.z = q[2]
            filtered_transform.transform.rotation.w = q[3]
            self.br.sendTransform(filtered_transform)
            print('published', transform)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            print('error')


if __name__ == '__main__':
    rospy.init_node('aruco_kf')
    node = ArucoKF()
    rospy.Timer(rospy.Duration(0.1), node.callback)
    rospy.spin()
