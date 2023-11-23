from calendar import c
from re import T
import rclpy
import rclpy.node
import math
import traceback
from argparse import ArgumentParser
from dataclasses import dataclass
import time
import cv2
import apriltag
import numpy as np
import threading

from frankx import *

@dataclass
class Pose:
    """Planar grasp with stroke d"""
    x: float  # [m]
    y: float  # [m]
    z: float  # [m]
    x_axis_rotation: float  # [m]
    y_axis_rotation: float  # [m]
    z_axis_rotation: float  # [m]

class AprilTagDetector():
    def __init__(self):
        self.stop_detection = False
        self.tag_postion = None
        self.base_frame = None
        self.initDetector()

    def initDetector(self):
        self.vid = cv2.VideoCapture(4)
        options = apriltag.DetectorOptions(families='tag36h11')
        self.detector = apriltag.Detector(options)
        self.process_thread = threading.Thread(target=self.processDetection)
        self.process_thread.start()
    
    def stopDetection(self):
        self.stop_detection = True
        self.process_thread.join()
        self.vid.release()

    def generateCoordinate(self, pose):
        camera_frame = np.array([[0],
                                [0],
                                [0],
                                [1]])
        return np.dot(pose, camera_frame)
    
    def processDetection(self):
        while True:
            if self.stop_detection:
                break
            ret, frame = self.vid.read() 
            result, overlay = apriltag.detect_tags(frame,
                                                    self.detector,      
                                                    camera_params=(322.282410, 322.282410, 320.818268, 178.779297), # camera_params=(3156.71852, 3129.52243, 359.097908, 239.736909)
                                                    tag_size= 0.06,
                                                    vizualization=3,
                                                    verbose=3,
                                                    annotation=True
                                                    )
            if len(result) > 1: # Detect tag or not
                self.tag_postion = self.generateCoordinate(result[1])
            # cv2.imshow('frame', overlay) 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break 
    
    def getTagPosition(self):
        return self.tag_postion

    def getTagMotion(self):
        tag_motion = [0, 0]
        if self.checkDetectEdge(self.tag_postion[0][0], self.tag_postion[1][0]) and self.base_frame is not None:
            tag_motion[0] = (self.base_frame[0] - self.tag_postion[0][0]) * 0.5 # 2D x axis moving distance 
            tag_motion[1] = (self.base_frame[1] - self.tag_postion[1][0]) * 0.5 # 2D y axis moving distance
        return tag_motion

    def checkDetectEdge(self, x, y):
        if x < 0.72 and x > -0.08 and y < 0.32 and y > -0.13:
            return True
        else:
            return False
    
    def setBaseFrame(self, x, y):
        if self.checkDetectEdge(x,y):
            self.base_frame = [0, 0]
            self.base_frame[0] = x
            self.base_frame[1] = y
        else:
            print("Set Base Frame Error")

class MotionPlaner(rclpy.node.Node):
    def __init__(self):
        super().__init__('motion_test_node')
        self.parser = ArgumentParser()
        self.parser.add_argument('--host', default='172.16.0.2', help='FCI IP of the robot')
        self.args = self.parser.parse_args()
        self.robot_state = None
        self.record_pose = []
        self.record_grasp_obj_pose = None

        ## Set Robot arm parameter ##
        self.robot = Robot(self.args.host)
        self.robot.set_default_behavior()
        self.robot.recover_from_errors()
        self.robot.set_dynamic_rel(0.2)

        ## Set Gripper parameter ##
        self.gripper = self.robot.get_gripper()
        self.gripper.gripper_speed = 0.05
        self.gripper.gripper_force = 40
        self.GRIPPER_MAX_WIDTH = 0.08 # m

        ## Test for April tag following ##
        self.apriltag_detector = AprilTagDetector()
        self.base_pos = None
        self.stop_following = False

    @staticmethod
    def get_base(x, y, z, roll = 0.0, pitch = 0.0, yaw = 0.0): 
        """Cartesian space coordinate"""
        return Affine(0.3 + x, y, 0.5 + z, roll, pitch, yaw)

    def recordPoseTrajectory(self):
        self.record_pose.append(self.robot.current_pose())
        print("Record pose: {}, {}, {}".format(self.robot.current_pose().x, self.robot.current_pose().y, self.robot.current_pose().z))

    def replayRecordTrajectory(self):
        """only support point to point trajectory"""
        if len(self.record_pose) != 0:
            for pose in self.record_pose:
                success = self.moveToTarget(pose, "Linear")
                if not success:
                    print("Fail to move target point: {}, {}, {}".format(pose.x, pose.y, pose.z))
                    break
            self.record_pose = [] # reset record list
        else:
            print("Record pose list is empty")

    def moveToTarget(self, pose: Affine, motion_type = "WayMotion"):
        """Target point coordinate should be based on end effector"""
        success = False
        if motion_type == "WayMotion":
            success = self.robot.move(WaypointMotion([
                Waypoint(pose)
            ]))
        elif motion_type == "Linear":
            success = self.robot.move(LinearMotion(
                pose
            ))
        if not success:
            self.robot.recover_from_errors()
            print("Fail to move target point: {}, {}, {}".format(pose.x, pose.y, pose.z))
            return False
        return True
    
    def moveToHome(self):
        # self.robot.move(JointMotion([0, -0.25 * math.pi, 0, -0.725 * math.pi, 0, 0.5 * math.pi, 0.25 * math.pi]))
        self.robot.move(JointMotion([0, -0.796, 0, -2.329, 0, 1.53, 0.785]))
        # self.robot.move(LinearMotion(self.get_base(0, 0, 0)))
        print("Return to Initial pose complete")

    def openGripper(self):
        self.gripper.move(self.GRIPPER_MAX_WIDTH)
        
    def closeGripper(self):
        is_grasp = self.gripper.clamp()
        if is_grasp:
            print("object grasp sucess")
        
    def getRobotState(self):
        self.robot_state = self.robot.read_once()
        print('\nPose: ', self.robot.current_pose())
        print('O_TT_E: ', self.robot_state.O_T_EE)
        print('Joints: ', self.robot_state.q)
        print('Elbow: ', self.robot_state.elbow)
        # return state
    
    def setGraspObjPos(self):
        self.record_grasp_obj_pose = self.robot.current_pose()
        print("obj pose: {}, {}, {}".format(self.robot.current_pose().x, self.robot.current_pose().y, self.robot.current_pose().z))

    def graspObject(self):
        if self.record_grasp_obj_pose is not None:
            self.openGripper()
            self.record_grasp_obj_pose.z = 0.3
            self.moveToTarget(self.record_grasp_obj_pose)
            motion_data = MotionData(0.6).with_reaction(Reaction(Measure.ForceZ < -5.0, LinearRelativeMotion(Affine(0, 0, 0))))
            pose = Affine(0, 0, -0.35)
            motion = LinearRelativeMotion(pose)
            self.robot.move(motion, motion_data)
            if motion_data.did_break:
                self.closeGripper()
                return True
            return False
    
    
    def test(self):
        motion = WaypointMotion([
            Waypoint(Affine(0.55676, 0.01219, 0.529)),
            Waypoint(Affine(0.54094, 0.01607, 0.529)),
            Waypoint(Affine(0.52512, 0.01996, 0.529)),
            Waypoint(Affine(0.50929, 0.02384, 0.529)),
            Waypoint(Affine(0.49347, 0.02773, 0.529)),
            Waypoint(Affine(0.47765, 0.03161, 0.529)),
            Waypoint(Affine(0.47078, 0.02072, 0.529)),
            Waypoint(Affine(0.4669, 0.00489, 0.529)),
            Waypoint(Affine(0.46301, -0.01093, 0.529)),
            Waypoint(Affine(0.45913, -0.02675, 0.529)),
            Waypoint(Affine(0.45524, -0.04, 0.529)),
            Waypoint(Affine(0.45136, -0.05839, 0.529)),
            Waypoint(Affine(0.45733, -0.06825, 0.529)),
            Waypoint(Affine(0.47315, -0.07213, 0.529)),
            Waypoint(Affine(0.48897, -0.07602, 0.529)),
            Waypoint(Affine(0.50479, -0.0799, 0.529)),
            Waypoint(Affine(0.52062, -0.08378, 0.529)),
            Waypoint(Affine(0.53644, -0.08767, 0.529)),
            Waypoint(Affine(0.54928, -0.08663, 0.529)),
            Waypoint(Affine(0.55316, -0.0708, 0.529)),
            Waypoint(Affine(0.55704, -0.05498, 0.529)),
            Waypoint(Affine(0.56093, -0.03916, 0.529)),
            Waypoint(Affine(0.56481, -0.02334, 0.529)),
            Waypoint(Affine(0.5687, -0.00752, 0.529)),
        ])

        self.robot.move(motion)

        # if (self.graspObject()):
        #     self.replayRecordTrajectory()
        #     self.openGripper()
        # self.moveToHome()

        # tag_pos = self.apriltag_detector.getTagPosition()
        # if tag_pos is not None and self.base_pos is not None:
        #     motion = self.apriltag_detector.getTagMotion()
        #     print(motion)
        #     target_pos = Affine(self.base_pos.x + motion[0], 0.0,self.base_pos.z + motion[1])
        #     self.moveToTarget(target_pos)

        # target_pos = Affine(0.35, 0.0 , 0.05)
        # self.moveToTarget(target_pos)

        # print("Test finish")
    def goToBaseFrame(self):
        tag_pos = self.apriltag_detector.getTagPosition()
        if tag_pos is not None:
            self.apriltag_detector.setBaseFrame(tag_pos[0][0], tag_pos[1][0])
            target_position = Affine(tag_pos[0][0], 0.0, tag_pos[1][0])
            print(target_position)
            self.moveToTarget(target_position)
            self.base_pos = self.robot.current_pose()

    def followAprilTag(self):
        while True:
            if self.stop_following:
                break
            tag_pos = self.apriltag_detector.getTagPosition()
            if tag_pos is not None and self.base_pos is not None:
                motion = self.apriltag_detector.getTagMotion()
                # print("Excute motion: {}".format(motion))
                target_pos = Affine(self.base_pos.x + motion[0], 0.0,self.base_pos.z + motion[1])
                self.moveToTarget(target_pos)
                time.sleep(0.1)

    def startFollow(self):
        self.stop_following = False
        self.follow_tag_thread = threading.Thread(target=self.followAprilTag)
        self.follow_tag_thread.start()

    def stopFollowing(self):
        self.stop_following = True
        self.follow_tag_thread.join()

def main():
    rclpy.init()
    controller = MotionPlaner()
    command_dict = {
        "home": controller.moveToHome,
        "test": controller.test,
        "check": controller.getRobotState,
        "open": controller.openGripper,
        "close": controller.closeGripper,
        "grasp": controller.setGraspObjPos,
        "replay": controller.replayRecordTrajectory,
        "set": controller.recordPoseTrajectory,
        "base": controller.goToBaseFrame,
        "stop": controller.stopFollowing,
        "follow": controller.startFollow,
    }
    while True:
        try:
            cmd = input("CMD : ")
            if cmd in command_dict:
                command_dict[cmd]()
            elif cmd == "exit":
                controller.apriltag_detector.stopDetection()
                break
        except Exception as e:
            traceback.print_exc()
            break
if __name__ == '__main__':
    main()

