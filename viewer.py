import cv2
import open3d
import numpy as np
import pyrealsense2 as rs

colour_image = None
depth_image = None

last_n_frames = []

def main():
    global colour_image
    global depth_image
    
    # open3d visualiser
    visualiser = open3d.visualization.Visualizer()
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(np.array([[i, i, i] for i in range(-5, 5)]))
    visualiser.create_window()
    visualiser.add_geometry(point_cloud)

    # Create windows
    cv2.namedWindow("Colour", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

    # Camera config
    config = rs.config()
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

    # Start camera pipeline
    pipeline = rs.pipeline()
    pipeline.start(config)

    # Depth and colour alignment
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Filters
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter(0.4, 20, 5)
    disparity = rs.disparity_transform(True)
    coloriser = rs.colorizer()

    # Get stream profile and camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    intrinsics = depth_profile.get_intrinsics()

    # Create camera intrinsics for open3d
    intrinsic = open3d.camera.PinholeCameraIntrinsic(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.width // 2, intrinsics.height // 2)

    while True:
        # Obtain and align frames
        current_frame = pipeline.wait_for_frames()
        current_frame = align.process(current_frame)
        
        depth_frame = temporal.process(current_frame.get_depth_frame())
        # Get colour and depth frames
        #depth_image = np.asanyarray(spatial.process(current_frame.get_depth_frame()).get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        colour_image = np.asanyarray(current_frame.get_color_frame().get_data())

        # Create rgbd image
        #rgbd = open3d.geometry.create_rgbd_image_from_color_and_depth(open3d.geometry.Image(cv2.cvtColor(colour_image, cv2.COLOR_BGR2RGB)), open3d.geometry.Image(depth_image), convert_rgb_to_intensity=False)

        # Create point cloud
        #pcd = open3d.geometry.create_point_cloud_from_rgbd_image(rgbd, intrinsic)
        
        # Update point cloud for visualiser
        #point_cloud.points = pcd.points
        #point_cloud.colors = pcd.colors

        # Update visualiser
        #visualiser.update_geometry()
        #visualiser.poll_events()
        #visualiser.update_renderer()

        depth_image = np.asanyarray(coloriser.colorize(depth_frame).get_data())

        cv2.imshow("Depth", depth_image)
        cv2.imshow("Colour", colour_image)
        cv2.waitKey(1)

main()
