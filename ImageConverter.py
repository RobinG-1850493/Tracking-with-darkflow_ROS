#! /usr/bin/env python
from darkflow.darkflow.net.build import TFNet
import cv2
import rospy
from deep_sort import generate_detections
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import argparse
import os
import sys

class ImageConverter:
    def __init__(self, args):
        self.tfnet = None
        self.image_publisher = rospy.Publisher(args.output, Image, queue_size=1)
        self.subscriber = rospy.Subscriber(args.input, Image, self.callback, queue_size = 1, buff_size=2**24)
        self.subscriber = rospy.Subscriber("/cctv_info", String, self.exit)

        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", 0.2, 100)
        self.tracker = Tracker(metric)
        self.encoder = generate_detections.create_box_encoder(
            os.path.abspath("deep_sort/resources/networks/mars-small128.ckpt-68577"))

        options = {"model": "darkflow/cfg/yolo.cfg", "load": "darkflow/bin/yolo.weights", "threshold": 0.1,
                   "track": True, "trackObj": ["person"], "BK_MOG": True, "tracker": "deep_sort", "csv": False}

        self.tfnet = TFNet(options)

    def callback(self, image_message):
        bridge = CvBridge()
        image = bridge.imgmsg_to_cv2(image_message, 'bgr8')
        if(self.tfnet != None):
            processedimg, fps = self.tfnet.image_return(image, self.encoder, self.tracker)
            print(fps)
            self.image_publisher.publish(bridge.cv2_to_imgmsg(processedimg, "bgr8"))
            cv2.waitKey(1)

    def exit(self, data):
        if data.data == "quit":
            if self.tfnet is not None:
                os.system("rosnode kill /image_converter_node")
                os.system("rosnode kill /openpose_ros_node")
                os.exit(0)

def main(args):
    try:
        rospy.init_node('image_converter_node', anonymous=False)
        ImageConverter(args)
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        help='Specify the topic you wish to use as input.', default='/stream_1')
    parser.add_argument('--output', type=str,
                        help='Specify the topic you wish to publish the output image to.', default='/tracking_1')
    args = parser.parse_args()
    main(args)
