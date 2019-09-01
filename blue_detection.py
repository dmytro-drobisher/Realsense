import cv2
import open3d
import numpy as np
import pyrealsense2 as rs

colour_image = None
depth_image = None

last_n_frames = []

#lower_threshold = np.array([75, 90, 90])
#upper_threshold = np.array([120, 255, 255])
lower_threshold = np.array([75, 100, 100])
upper_threshold = np.array([120, 255, 255])


def get_colour_threshold(event, x, y, flags, param):
    global lower_threshold
    global upper_threshold
    
    hsv_image = cv2.cvtColor(colour_image, cv2.COLOR_BGR2HSV)
    colour = hsv_image[y, x]
    lower_threshold = np.array([colour[0] - 10, 90, 90])
    upper_threshold = np.array([colour[0] + 10, 255, 255])
    print(upper_threshold)

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
    cv2.namedWindow("Colour")
    cv2.namedWindow("Depth")
    cv2.namedWindow("Filtered")
    #cv2.setMouseCallback("Colour", get_colour_threshold)

    # Camera config
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start camera pipeline
    pipeline = rs.pipeline()
    pipeline.start(config)

    # Depth and colour alignment
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Spatial filter
    spatial = rs.spatial_filter()
    
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

        # Get colour and depth frames
        depth_image = np.asanyarray(spatial.process(current_frame.get_depth_frame()).get_data())
        colour_image = np.asanyarray(current_frame.get_color_frame().get_data())

        # Create rgbd image
        rgbd = open3d.geometry.create_rgbd_image_from_color_and_depth(open3d.geometry.Image(cv2.cvtColor(colour_image, cv2.COLOR_BGR2RGB)), open3d.geometry.Image(depth_image), convert_rgb_to_intensity=False)

        # Create point cloud
        pcd = open3d.geometry.create_point_cloud_from_rgbd_image(rgbd, intrinsic)
        
        # Update point cloud for visualiser
        point_cloud.points = pcd.points
        point_cloud.colors = pcd.colors

        # Update visualiser
        visualiser.update_geometry()
        visualiser.poll_events()
        visualiser.update_renderer()

        # Segment the image
        if (lower_threshold is not None) and (upper_threshold is not None):
            # Gaussian blur
            smoothed = cv2.GaussianBlur(colour_image, (5, 5), 3)
            
            # Segment image and filter noise from mask
            mask = cv2.inRange(cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV), lower_threshold, upper_threshold)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (11, 11))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (11, 11))
            im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Iterate over contours
            for contour in contours:
                
                # remove small contours
                if cv2.contourArea(contour) > 200:
                    
                    # Get centroid
                    centre = np.mean(contour, axis=0)[0].astype(int)
                    
                    # Display contours
                    cv2.putText(colour_image, str(depth_image[centre[1], centre[0]]) + "mm", (centre[0], centre[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    cv2.drawContours(colour_image, np.array([contour]), -1, [0, 255, 0])
            
            cv2.imshow("Filtered", mask)

        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        cv2.imshow("Depth", depth_image)
        cv2.imshow("Colour", colour_image)
        cv2.waitKey(1)

main()
