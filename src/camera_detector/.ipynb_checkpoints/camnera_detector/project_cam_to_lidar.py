#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.2.1
Date    : Jan 20, 2019

Description:
Script to find the transformation between the Camera and the LiDAR

Example Usage:
1. To perform calibration using the GUI to pick correspondences:

    $ rosrun lidar_camera_calibration calibrate_camera_lidar.py --calibrate

    The point correspondences will be save as following:
    - PKG_PATH/calibration_data/lidar_camera_calibration/img_corners.npy
    - PKG_PATH/calibration_data/lidar_camera_calibration/pcl_corners.npy

    The calibrate extrinsic are saved as following:
    - PKG_PATH/calibration_data/lidar_camera_calibration/extrinsics.npz
    --> 'euler' : euler angles (3, )
    --> 'R'     : rotation matrix (3, 3)
    --> 'T'     : translation offsets (3, )

2. To display the LiDAR points projected on to the camera plane:

    $ roslaunch lidar_camera_calibration display_camera_lidar_calibration.launch

Notes:
Make sure this file has executable permissions:
$ chmod +x calibrate_camera_lidar.py

References: 
http://wiki.ros.org/message_filters
http://wiki.ros.org/cv_bridge/Tutorials/
http://docs.ros.org/api/image_geometry/html/python/
http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscribe
'''

# Python 2/3 compatibility
from __future__ import print_function

# Built-in modules
import os
import sys
import time
import threading
import multiprocessing

# External modules
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ROS modules
# PKG = 'lidar_camera_calibration'
# import roslib; roslib.load_manifest(PKG)
import rosbag
import rospy
import tf2_ros
import ros_numpy
import image_geometry
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from tf.transformations import euler_from_matrix
# from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import tf

# Global variables
PAUSE = False
FIRST_TIME = True
KEY_LOCK = threading.Lock()
TF_BUFFER = None
TF_LISTENER = None
CV_BRIDGE = CvBridge()
CAMERA_MODEL = image_geometry.PinholeCameraModel()


# Global paths
PKG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CALIB_PATH = 'calibration_data/lidar_camera_calibration'
global rotation_matrix_g
rotation_matrix_g = 0
global traslation_g
traslation_g = 0
global prestate
prestate =0


'''
Keyboard handler thread
Inputs: None
Outputs: None
'''
def handle_keyboard():
    global KEY_LOCK, PAUSE
    key = raw_input('Press [ENTER] to pause and pick points\n')
    with KEY_LOCK: PAUSE = True


'''
Start the keyboard handler thread
Inputs: None
Outputs: None
'''
def start_keyboard_handler():
    keyboard_t = threading.Thread(target=handle_keyboard)
    keyboard_t.daemon = True
    keyboard_t.start()


'''
Save the point correspondences and image data
Points data will be appended if file already exists

Inputs:
    data - [numpy array] - points or opencv image
    filename - [str] - filename to save
    folder - [str] - folder to save at
    is_image - [bool] - to specify whether points or image data

Outputs: None
'''
def save_data(data, filename, folder, is_image=False):
    # Empty data
    if not len(data): return

    # Handle filename
    filename = os.path.join(PKG_PATH, os.path.join(folder, filename))
    
    # Create folder
    try:
        os.makedirs(os.path.join(PKG_PATH, folder))
    except OSError:
        if not os.path.isdir(os.path.join(PKG_PATH, folder)): raise

    # Save image
    if is_image:
        cv2.imwrite(filename, data)
        return

    # Save points data
    if os.path.isfile(filename):
        rospy.logwarn('Updating file: %s' % filename)
        data = np.vstack((np.load(filename), data))
    np.save(filename, data)


'''
Runs the image point selection GUI process

Inputs:
    img_msg - [sensor_msgs/Image] - ROS sensor image message
    now - [int] - ROS bag time in seconds
    rectify - [bool] - to specify whether to rectify image or not

Outputs:
    Picked points saved in PKG_PATH/CALIB_PATH/img_corners.npy
'''
def extract_points_2D(img_msg, now, rectify=False):
    # Log PID
    rospy.loginfo('2D Picker PID: [%d]' % os.getpid())

    # Read image using CV bridge
    try:
        img = CV_BRIDGE.imgmsg_to_cv2(img_msg, 'bgr8')
    except CvBridgeError as e: 
        rospy.logerr(e)
        return

    # Rectify image
    if rectify: CAMERA_MODEL.rectifyImage(img, img)
    disp = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    # Setup matplotlib GUI
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Select 2D Image Points - %d' % now.secs)
    ax.set_axis_off()
    ax.imshow(disp)

    # Pick points
    picked, corners = [], []
    def onclick(event):
        x = event.xdata
        y = event.ydata
        if (x is None) or (y is None): return

        # Display the picked point
        picked.append((x, y))
        corners.append((x, y))
        rospy.loginfo('IMG: %s', str(picked[-1]))

        if len(picked) > 1:
            # Draw the line
            temp = np.array(picked)
            ax.plot(temp[:, 0], temp[:, 1])
            ax.figure.canvas.draw_idle()

            # Reset list for future pick events
            del picked[0]

    # Display GUI
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Save corner points and image
    rect = '_rect' if rectify else ''
    if len(corners) > 1: del corners[-1] # Remove last duplicate
    save_data(corners, 'img_corners%s.npy' % (rect), CALIB_PATH)
    save_data(img, 'image_color%s-%d.jpg' % (rect, now.secs), 
        os.path.join(CALIB_PATH, 'images'), True)


'''
Runs the LiDAR point selection GUI process

Inputs:
    velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
    now - [int] - ROS bag time in seconds

Outputs:
    Picked points saved in PKG_PATH/CALIB_PATH/pcl_corners.npy
'''
def extract_points_3D(velodyne, now):
    # Log PID
    rospy.loginfo('3D Picker PID: [%d]' % os.getpid())

    # Extract points data
    points = ros_numpy.point_cloud2.pointcloud2_to_array(velodyne)
    points = np.asarray(points.tolist())

    # Select points within chessboard range
    inrange = np.where((points[:, 0] > -2) &
                       (points[:, 0] < 2.2) &
                       (points[:, 1] < 0) &
                       (points[:, 1] > -10) &
                       (points[:, 2] < 2))
    points = points[inrange[0]]
    print(points.shape)
    if points.shape[0] > 5:
        rospy.loginfo('PCL points available: %d', points.shape[0])
    else:
        rospy.logwarn('Very few PCL points available in range')
        return

    # Color map for the points
    cmap = matplotlib.cm.get_cmap('hsv')
    colors = cmap(points[:, -1] / np.max(points[:, -1]))

    # Setup matplotlib GUI
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Select 3D LiDAR Points - %d' % now.secs, color='white')
    ax.set_axis_off()
    # ax.set_facecolor((0, 0, 0))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=2, picker=5)

    # Equalize display aspect ratio for all axes
    max_range = (np.array([points[:, 0].max() - points[:, 0].min(), 
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()]).max() / 2.0)
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Pick points
    picked, corners = [], []
    def onpick(event):
        ind = event.ind[0]
        x, y, z = event.artist._offsets3d

        # Ignore if same point selected again
        if picked and (x[ind] == picked[-1][0] and y[ind] == picked[-1][1] and z[ind] == picked[-1][2]):
            return
        
        # Display picked point
        picked.append((x[ind], y[ind], z[ind]))
        corners.append((x[ind], y[ind], z[ind]))
        rospy.loginfo('PCL: %s', str(picked[-1]))

        if len(picked) > 1:
            # Draw the line
            temp = np.array(picked)
            ax.plot(temp[:, 0], temp[:, 1], temp[:, 2])
            ax.figure.canvas.draw_idle()

            # Reset list for future pick events
            del picked[0]

    # Display GUI
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()

    # Save corner points
    if len(corners) > 1: del corners[-1] # Remove last duplicate
    save_data(corners, 'pcl_corners.npy', CALIB_PATH)


'''
Calibrate the LiDAR and image points using OpenCV PnP RANSAC
Requires minimum 5 point correspondences

Inputs:
    points2D - [numpy array] - (N, 2) array of image points
    points3D - [numpy array] - (N, 3) array of 3D points

Outputs:
    Extrinsics saved in PKG_PATH/CALIB_PATH/extrinsics.npz
'''
def calibrate(points2D=None, points3D=None):
    # Load corresponding points
    global rotation_matrix_g, traslation_g
    global prestate
    prestate =0

    folder = os.path.join(PKG_PATH, CALIB_PATH)
    if points2D is None: points2D = np.load(os.path.join(folder, 'img_corners.npy'))
    if points3D is None: points3D = np.load(os.path.join(folder, 'pcl_corners.npy'))
    
    # Check points shape
    assert(points2D.shape[0] == points3D.shape[0])
    if not (points2D.shape[0] >= 5):
        rospy.logwarn('PnP RANSAC Requires minimum 5 points')
        return

    # Obtain camera matrix and distortion coefficients
    camera_matrix = CAMERA_MODEL.intrinsicMatrix()
    dist_coeffs = CAMERA_MODEL.distortionCoeffs()

    if (prestate ==0 ):
        rmat = np.array([[-0.9876689, -0.1414207, -0.0671588],[0.0654959,  0.0163857, -0.9977183],[0.1421984, -0.9898140, -0.0069212]])
        rvec,jac = cv2.Rodrigues(rmat)
        tvec  =  np.array([-0.17319528, -0.94018986, -0.96836207])
    else:
        rmat = rotation_matrix_g
        rvec,jac = cv2.Rodrigues(rmat)
        tvec = traslation_g

        
     


    # Estimate extrinsics
    success, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(points3D, 
        points2D, camera_matrix, dist_coeffs,rvec=rvec,tvec=tvec,useExtrinsicGuess=True,iterationsCount=2000,reprojectionError=30, flags=cv2.SOLVEPNP_ITERATIVE)

    # success, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(points3D, 
        # points2D, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    print(success)

    # Refine estimate using LM
    if not success:
        rospy.logwarn('Initial estimation unsuccessful, skipping refinement')
    elif not hasattr(cv2, 'solvePnPRefineLM'):
        rospy.logwarn('solvePnPRefineLM requires OpenCV >= 4.1.1, skipping refinement')
    else:
        rotation_vector, translation_vector = cv2.solvePnPRefineLM(points3D,
            points2D, camera_matrix, dist_coeffs, rotation_vector, translation_vector)


    tot_error=0
    total_points=0
    for i in range(len(points3D)):
        reprojected_points, _ = cv2.projectPoints(points3D[i], rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        reprojected_points=reprojected_points.reshape(-1,2)
        tot_error+=np.sum(np.abs(points2D[i]-reprojected_points)**2)
        total_points+=1

    mean_error=np.sqrt(tot_error/total_points)
    print ("Mean reprojection error: ", mean_error)
    # Convert rotation vector
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    euler = euler_from_matrix(rotation_matrix)

    rotation_matrix_g = rotation_matrix
    traslation_g = translation_vector
    prestate =1
    
    # Save extrinsics
    np.savez(os.path.join(folder, 'extrinsics.npz'),
        euler=euler, R=rotation_matrix, T=translation_vector.T)

    # Display results
    print('Euler angles (RPY):', euler)
    print('Rotation Matrix:', rotation_matrix)
    print('Translation Offsets:', translation_vector.T)


'''
Projects the point cloud on to the image plane using the extrinsics

Inputs:
    img_msg - [sensor_msgs/Image] - ROS sensor image message
    velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
    image_pub - [sensor_msgs/Image] - ROS image publisher

Outputs:
    Projected points published on /sensors/camera/camera_lidar topic
'''
def project_point_cloud(velodyne, img_msg, image_pub):
    # Read image using CV bridge
    try:
        img = CV_BRIDGE.imgmsg_to_cv2(img_msg, 'bgr8')
    except CvBridgeError as e: 
        rospy.logerr(e)
        return

    # Transform the point cloud
    try:
        transform = TF_BUFFER.lookup_transform('os_sensor', 'world', rospy.Time())
        # trans,rot  = tf_listen.lookupTransform('world', 'os_sensor', rospy.Time())
        # print(trans)
        velodyne = do_transform_cloud(velodyne, transform)
    except tf2_ros.LookupException:
        pass

    # Extract points from message
    v = ros_numpy.numpify(velodyne)
    # print(v['reflectivity'])
    points3D = np.vstack([v['x'].flatten(),v['y'].flatten(),v['z'].flatten(),v['intensity'].flatten()]).T
    
    # Filter points in front of camera
    # inrange = np.where((points3D[:, 2] < 5) &
    #                    (np.abs(points3D[:, 0]) < 6) &
    #                    (points3D[:,1]<0))


    inrange = np.where( (points3D[:,1]<75) &   # Z axis
                        (points3D[:,1]>-5) & 
                        (points3D[:,2]>0) &  # X axis\
                        (points3D[:,2]<40) &
                          (points3D[:, 0] > -12) & # Y axis
                          (points3D[:, 0] < 12 ) 
                        )
    points3D[:,3] = np.where(points3D[:,3]>100.0,100.0,points3D[:,3])
    # points3D[:,3] = np.where(points3D[:,3]<1.0,1.0,points3D[:,3])
    max_intensity = np.max(points3D[:, -1])
    points3D = points3D[inrange[0]]

    # Color map for the points
    cmap = matplotlib.cm.get_cmap('jet')
    colors = cmap(points3D[:, -1] / max_intensity) * 255

    # Project to 2D and filter points within image boundaries
    points2D = [ CAMERA_MODEL.project3dToPixel(point) for point in points3D[:, :3] ]
    points2D = np.asarray(points2D)
    if points2D.size>0:
        inrange = np.where((points2D[:, 0] >= 0) &
                        (points2D[:, 1] >= 0) &
                        (points2D[:, 0] < img.shape[1]) &
                        (points2D[:, 1] < img.shape[0]))
        points2D = points2D[inrange[0]].round().astype('int')
        points2D = points2D.round().astype('int')

        # Draw the projected 2D points
        for i in range(len(points2D)):
            cv2.circle(img, tuple(points2D[i]), 3, tuple(colors[i]), -1)

        # Publish the projected points image
        try:
            image_pub.publish(CV_BRIDGE.cv2_to_imgmsg(img, "bgr8"))
        except CvBridgeError as e: 
            rospy.logerr(e)


'''
Callback function to publish project image and run calibration

Inputs:
    image - [sensor_msgs/Image] - ROS sensor image message
    camera_info - [sensor_msgs/CameraInfo] - ROS sensor camera info message
    velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
    image_pub - [sensor_msgs/Image] - ROS image publisher

Outputs: None
'''
def callback(image, camera_info, velodyne, image_pub=None):
    global CAMERA_MODEL, FIRST_TIME, PAUSE, TF_BUFFER, TF_LISTENER, tf_listen
    # Setup the pinhole camera model
    if FIRST_TIME:
        FIRST_TIME = False

        # Setup camera model
        rospy.loginfo('Setting up camera model')
        CAMERA_MODEL.fromCameraInfo(camera_info)

        # TF listener
        rospy.loginfo('Setting up static transform listener')
        TF_BUFFER = tf2_ros.Buffer()
        TF_LISTENER = tf2_ros.TransformListener(TF_BUFFER)
        tf_listen = tf.TransformListener()

    project_point_cloud(velodyne, image, image_pub)



'''
The main ROS node which handles the topics

Inputs:
    camera_info - [str] - ROS sensor camera info topic
    image_color - [str] - ROS sensor image topic
    velodyne - [str] - ROS velodyne PCL2 topic
    camera_lidar - [str] - ROS projected points image topic

Outputs: None
'''
def listener(camera_info, image_color, velodyne_points, camera_lidar=None):
    # Start node
    rospy.init_node('calibrate_camera_lidar', anonymous=True)
    rospy.loginfo('Current PID: [%d]' % os.getpid())
    rospy.loginfo('Projection mode: %s' % PROJECT_MODE)
    rospy.loginfo('CameraInfo topic: %s' % camera_info)
    rospy.loginfo('Image topic: %s' % image_color)
    rospy.loginfo('PointCloud2 topic: %s' % velodyne_points)
    rospy.loginfo('Output topic: %s' % camera_lidar)

    # Subscribe to topics
    info_sub = message_filters.Subscriber(camera_info, CameraInfo)
    image_sub = message_filters.Subscriber(image_color, Image)
    velodyne_sub = message_filters.Subscriber(velodyne_points, PointCloud2)

    # Publish output topic
    image_pub = None
    if camera_lidar: image_pub = rospy.Publisher(camera_lidar, Image, queue_size=5)

    # Synchronize the topics by time
    ats = message_filters.ApproximateTimeSynchronizer(
        [image_sub, info_sub, velodyne_sub], queue_size=5, slop=0.1)
    ats.registerCallback(callback, image_pub)

    # Keep python from exiting until this node is stopped
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo('Shutting down')


if __name__ == '__main__':

    camera_info = '/zed/zed_node/left_raw/camera_info'
    image_color = '/zed/zed_node/left_raw/image_rect_color'
    velodyne_points = '/os_cloud_node/points'
    camera_lidar = '/zed/zed_node/left/lidar_projection'
    PROJECT_MODE = False
    # Start keyboard handler thread
    if not PROJECT_MODE: start_keyboard_handler()

    # Start subscriber
    listener(camera_info, image_color, velodyne_points, camera_lidar)