#! /usr/bin/env python
from darkflow.darkflow.net.build import TFNet
import cv2
import rospy
from deep_sort import generate_detections
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import argparse
import os

class ImageConverter:
    def __init__(self, args):
        topicName = "/stream_1"
        self.image_publisher = rospy.Publisher(args.output, Image)
        self.subscriber = rospy.Subscriber(args.input, Image, self.callback, queue_size = 1, buff_size=2**24)

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
        processedimg = self.tfnet.image_return(image, self.encoder, self.tracker)
        self.image_publisher.publish(bridge.cv2_to_imgmsg(processedimg, "bgr8"))
        cv2.waitKey(1)

def main(args):
    ImageConverter(args)
    rospy.init_node('ImageConverter', anonymous=False)
    try:
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
