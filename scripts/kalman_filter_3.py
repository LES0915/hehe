import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class ArucoTracker:
    def __init__(self):
        # Initialize aruco marker detection
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_100)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        self.kf = KalmanFilter(dim_x=6, dim_z=6)
        self.kf.x = np.zeros(6)
        self.kf.P = np.eye(6) * 1.0
        self.kf.Q = np.eye(6) * 0.01
        self.kf.R = np.eye(6) * 0.1
        
        # Initialize camera matrix and distortion coefficients
        self.cameraMatrix = None
        self.distCoeffs = None
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Initialize ROS subscribers
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)

        self.pub = rospy.Publisher('/aruco', Image,queue_size=10)
        
    def camera_info_callback(self, data):
        # Extract camera matrix and distortion coefficients from CameraInfo message
        self.cameraMatrix = np.array(data.K).reshape(3,3)
        self.distCoeffs = np.array(data.D).reshape(1,5)
        
    def image_callback(self, data):
        if self.cameraMatrix is None or self.distCoeffs is None:
            return
        
        try:
            # Convert ROS image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return
        
        # Detect aruco marker
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(cv_image)
        
        # If a marker is detected, extract its pose and update Kalman filter
        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, self.cameraMatrix, self.distCoeffs)
            pose = np.concatenate((tvecs[0], rvecs[0])).reshape((6,))

            # Draw the filtered pose on the frame
            pose_filtered = np.hstack((pose[:3].T, pose[3:].T))
            cv2.drawFrameAxes(cv_image, self.cameraMatrix, self.distCoeffs, rvecs[0], tvecs[0], 0.1)
            # cv2.aruco.drawAxis(cv_image, self.cameraMatrix, self.distCoeffs, pose_filtered[3:], pose_filtered[:3], 0.1)
        
        # Display the resulting frame
        print('image')
        # cv2.imshow("Frame", cv_image)
        # cv2.waitKey
        self.pub.publish(self.bridge.cv2_to_imgmsg(cv_image))


if __name__ == '__main__':
    rospy.init_node('aruco_kf')
    node = ArucoTracker()
    # rospy.Timer(rospy.Duration(0.1), node.callback)
    rospy.spin()