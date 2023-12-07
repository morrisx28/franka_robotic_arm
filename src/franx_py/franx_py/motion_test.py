import rclpy
import rclpy.node
from sensor_msgs.msg import JointState
import traceback
from argparse import ArgumentParser
from dataclasses import dataclass
import time
import cv2
import apriltag
import numpy as np
import threading
import pickle

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

class CoffeeParameter:
    def __init__(self):
        self.ready_pos = []
        self.grasp_pos = []
        self.poll_pos = []
        self.put_pos = []

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
        """Set camera detect edge (FOV)"""
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

        ## For record motion setting
        self.is_recording = False

        ## For publish to Isaac sim ##
        self.joint_state_pub = self.create_publisher(JointState, "joint_command", 10)
        self.joint_state = JointState()
        self.joint_state.name = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]

        ## Set Robot arm parameter ##
        self.robot = Robot(self.args.host)
        self.robot.set_default_behavior()
        self.robot.recover_from_errors()
        self.robot.set_dynamic_rel(0.2)

        ## Set Gripper parameter ##
        self.gripper = self.robot.get_gripper()
        self.gripper.gripper_speed = 0.05
        self.gripper.gripper_force = 60
        self.GRIPPER_MAX_WIDTH = 0.08 # m

        ## Test for April tag following ##
        # self.apriltag_detector = AprilTagDetector()
        self.base_pos = None
        self.stop_following = False
    
    def pubToTopic(self, joints_pos):
        """input 7 axis joints position """ 
        self.joint_state.header.stamp = self.get_clock().now().to_msg()
        self.joint_state.position = joints_pos

    @staticmethod
    def get_base(x, y, z, roll = 0.0, pitch = 0.0, yaw = 0.0): 
        """Cartesian space coordinate"""
        return Affine(0.3 + x, y, 0.5 + z, roll, pitch, yaw)

    def recordPose(self):
        self.record_pose.append(self.robot.current_pose())
        print("Record pose: {}, {}, {}".format(self.robot.current_pose().x, self.robot.current_pose().y, self.robot.current_pose().z))

    def startRecordTraj(self):
        self.is_recording = True
        self.record_thread = threading.Thread(target=self.recordPoseTrajectory)
        self.record_thread.start()

    def stopRecordTraj(self):
        self.is_recording = False
        if self.record_thread is not None:
            self.record_thread.join()
            print("stop recording")
        else:
            print("record not start yet")

    def recordPoseTrajectory(self, collect_freq = 100):
        """Default collect frequency is 10hz"""
        while self.is_recording:
            self.record_pose.append(self.robot.current_pose())
            print("Record pose: {}, {}, {}".format(self.robot.current_pose().x, self.robot.current_pose().y, self.robot.current_pose().z))
            time.sleep(1 / collect_freq) 

    def replayRecordTrajectory(self, record_pos_list, motion_type = "Linear", dt = 0.5, cycle = 1):
        """support follow waypoints and impedence trajectory"""
        if len(record_pos_list) != 0:
            if motion_type == "Linear":
                for _ in range(cycle):
                    for pose in record_pos_list:
                        if not self.moveToTarget(pose, "Linear"):
                            print("Fail to move target point: {}, {}, {}".format(pose.x, pose.y, pose.z))
                            break
            elif motion_type == "Impedence":
                impedance_motion = ImpedanceMotion(400, 30) # translational stiffness, rotational stiffness
                execute_thread = self.robot.move_async(impedance_motion)
                time.sleep(0.5)
                for _ in range(cycle):
                    for pose in record_pos_list:
                        print('target: ', impedance_motion.target)
                        impedance_motion.target = pose
                        time.sleep(dt) # operation frequency 2hz
                impedance_motion.finish()
                execute_thread.join()
            # self.record_pose = [] # reset record list
        else:
            print("Record pose list is empty")
    
    def clearRecordTrajectory(self):
        self.record_pose = []

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
            return True
        return False
        
    def getRobotState(self):
        self.robot_state = self.robot.read_once()
        print('\nPose: ', self.robot.current_pose())
        print('O_TT_E: ', self.robot_state.O_T_EE)
        print('Joints: ', self.robot_state.q)
        print('Elbow: ', self.robot_state.elbow)
        return self.robot_state
    
    def setGraspObjPos(self):
        if len(self.record_pose) != 0:
            self.grasp_pos = self.record_pose
            self.clearRecordTrajectory()
            print("Set grasp pose sucess")
        else:
            print("Set grasp pose fail")
    #     self.record_grasp_obj_pose = self.robot.current_pose()
    #     print("obj pose: {}, {}, {}".format(self.robot.current_pose().x, self.robot.current_pose().y, self.robot.current_pose().z))

    def setReadyPos(self):
        if len(self.record_pose) != 0:
            self.ready_pos = self.record_pose
            self.clearRecordTrajectory()
            print("Set ready pose sucess")
        else:
            print("Set ready pose fail")

    def setPollPos(self):
        if len(self.record_pose) != 0:
            self.poll_pos = self.record_pose
            self.clearRecordTrajectory()
            print("Set poll pose sucess")
        else:
            print("Set poll pose fail")

    def setPutCup(self):
        if len(self.record_pose) != 0:
            self.put_pos = self.record_pose
            self.clearRecordTrajectory()
            print("Set put cup sucess")
        else:
            print("Set put cup pose fail")

    def getCup(self):
        if self.grasp_pos is not None:
            self.replayRecordTrajectory(self.grasp_pos)
            return self.closeGripper()
        print("Grasp trajectory not set yet")
        return False
    
    def putBackCup(self):
        if self.put_pos is not None:
            self.replayRecordTrajectory(self.put_pos)
            self.openGripper()

    def pollWater(self, cycle=1):
        if self.poll_pos is not None:
            self.replayRecordTrajectory(self.poll_pos, motion_type='Impedence', cycle=cycle)

    def readyPos(self):
        if self.ready_pos is not None:
            self.replayRecordTrajectory(self.ready_pos)
    
    def saveParameter(self):
        param = CoffeeParameter()
        try:
            param.ready_pos = self.AffineToPose(self.ready_pos)
            param.grasp_pos = self.AffineToPose(self.grasp_pos)
            param.poll_pos = self.AffineToPose(self.poll_pos)
            param.put_pos = self.AffineToPose(self.put_pos)
            file_name = "coffee_param"
            with open(file_name, 'wb+') as f:
                pickle.dump(param, f)
        except Exception as e:
            print("Fail to save parameter")
    
    def loadParameter(self):
        try:
            file_name = "coffee_param"
            with open(file_name, 'rb') as f:
                coffee_param = pickle.load(f)
            self.ready_pos = self.PoseToAffine(coffee_param.ready_pos)
            self.grasp_pos = self.PoseToAffine(coffee_param.grasp_pos)
            self.poll_pos = self.PoseToAffine(coffee_param.poll_pos)
            self.put_pos = self.PoseToAffine(coffee_param.put_pos)
        except Exception as e:
            print("Fail to load parameter")
    
    def AffineToPose(self, affine_list):
        pose_list = []
        for affine in affine_list:
            pose = [affine.x, affine.y, affine.z, affine.a, affine.b, affine.c]
            pose_list.append(pose)
        return pose_list
    
    def PoseToAffine(self, pose_list):
        affine_list = []
        for pose in pose_list:
            affine = Affine(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5])
            affine_list.append(affine)
        return affine_list

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
        
    def pollCoffee(self):
        if self.getCup():
            self.readyPos()
            ## Stage 1 ##
            self.pollWater(6)
            self.readyPos()
            time.sleep(15)
            ## Stage 2 ##
            self.pollWater(10)
            self.readyPos()
            time.sleep(6)
            ## Stage 3
            self.pollWater(12)
            self.readyPos()
            self.putBackCup()
            self.openGripper()
            self.moveToHome()
        else:
            print("Cup is not exist")
    
    def startPollCoffee(self):
        pass
    
    def test(self):
        if self.getCup():
            self.readyPos()
            # Stage 1 ##
            # self.pollWater(3)
            # self.readyPos()
            # time.sleep(15)
            # ## Stage 2 ##
            # self.pollWater(5)
            # self.readyPos()
            # time.sleep(6)
            # ## Stage 3
            # self.pollWater(6)
            # self.readyPos()
            # self.putBackCup()
            # self.moveToHome()
        

        # if (self.graspObject()):
        #     self.replayRecordTrajectory()
        #     self.openGripper()
        # self.moveToHome()
    
    def testPoll(self):
        self.pollWater()
        # if self.getCup():
        #     self.readyPos()
        #     self.pollWater(2)
        #     self.putBackCup()
        #     self.moveToHome()

    def goToBaseFrame(self):
        tag_pos = self.apriltag_detector.getTagPosition()
        if tag_pos is not None:
            self.apriltag_detector.setBaseFrame(tag_pos[0][0], tag_pos[1][0])
            target_position = Affine(tag_pos[0][0], 0.0, tag_pos[1][0])
            print(target_position)
            self.moveToTarget(target_position)
            self.base_pos = self.robot.current_pose()

    def followAprilTag(self):
        self.impedance_motion = ImpedanceMotion(300.0, 20.0)
        self.execute_thread = self.robot.move_async(self.impedance_motion)
        time.sleep(0.1)
        while True:
            if self.stop_following:
                break
            tag_pos = self.apriltag_detector.getTagPosition()
            if tag_pos is not None and self.base_pos is not None:
                motion = self.apriltag_detector.getTagMotion()
                # print("Excute motion: {}".format(motion))
                # target_pos = Affine(self.base_pos.x + motion[0], 0.0,self.base_pos.z + motion[1])
                self.impedance_motion.target = Affine(self.base_pos.x + motion[0], 0.0,self.base_pos.z + motion[1])
                # self.moveToTarget(target_pos)
                time.sleep(0.1)

    def startFollow(self):
        self.stop_following = False
        self.follow_tag_thread = threading.Thread(target=self.followAprilTag)
        self.follow_tag_thread.start()

    def stopFollowing(self):
        self.stop_following = True
        self.impedance_motion.finish()
        self.execute_thread.join()
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
        "poll": controller.setPollPos,
        "set": controller.recordPose,
        "r": controller.startRecordTraj,
        "s": controller.stopRecordTraj,
        "ready": controller.setReadyPos,
        "put": controller.setPutCup,
        "base": controller.goToBaseFrame,
        "stop": controller.stopFollowing,
        "follow": controller.startFollow,
        "clear": controller.clearRecordTrajectory,
        "save": controller.saveParameter,
        "load": controller.loadParameter,
        "coffee": controller.pollCoffee,
        "pt": controller.testPoll,
    }
    while True:
        try:
            cmd = input("CMD : ")
            if cmd in command_dict:
                command_dict[cmd]()
            elif cmd == "exit":
                # controller.apriltag_detector.stopDetection()
                break
        except Exception as e:
            traceback.print_exc()
            break
if __name__ == '__main__':
    main()

