import serial
import time
import logging
from threading import Thread
from nav import lane
from nav import sign
from nav import crosswalk
import cv2 as cv


class CarState:
    def __init__(self, task):
        self.frame = None
        self.lane_angle = 0 # TODO should this be None?
        self.action = "follow_lane"
        self.frame_counter = 0
        self.task = task
        self.stop_sign = None

    def is_ready(self):
        if self.frame is None: return False
        return True

    def update_frame(self, frame):
        self.frame = frame
        cv.imwrite(f'data/exp_debug_3/frame_{self.frame_counter}.png', self.frame)
        self.frame_counter += 1

    def update_lane_angle(self, lane_angle):
        self.lane_angle = lane_angle

    def update_stop_sign(self, stop_sign):
        self.stop_sign = stop_sign

    def read_frame(self):
        return self.frame

    def read_lane_angle(self):
        return self.lane_angle

    def read_task(self):
        return self.task

    def read_stop_sign(self):
        return self.stop_sign

def parallel_park(car_state):
    steer(0)
    set_speed(20)
    time.sleep(3.8)
    set_speed(0)
    steer(25)
    set_speed(-15)
    time.sleep(3)
    steer(-25)
    time.sleep(3)
    steer(0)
    set_speed(0)
    logging.info("stop")

def get_camera_lane(car_state):
    while True:
        if car_state.is_ready():
            try:
                lane_angle = lane.find_lanes(car_state.read_frame())[1]
                car_state.update_lane_angle(lane_angle)
                # logging.info(f'lane angle:{lane_angle}')
            except:
                pass

def get_stop_sign(car_state):
    while True:
        if car_state.is_ready():
            frame = car_state.read_frame()
            stop_sign = sign.stop_signs(frame)
            car_state.update_stop_sign(stop_sign)

def set_speed(speed):
    serial_port = serial.Serial('/dev/ttyACM0', baudrate=19200)
    serial_port.write(f'#1:{speed};;\r\n'.encode())
    return True

def steer(angle):
    serial_port = serial.Serial('/dev/ttyACM0', baudrate=19200)
    serial_port.write(f'#2:{angle};;\r\n'.encode())
    return True

def read_camera(car_state):
    cap = cv.VideoCapture(0)
    # cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 3)
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
    # x = cap.set(cv.CAP_PROP_EXPOSURE, -10)
    while True:
        ret, frame = cap.read()
        car_state.update_frame(frame)
        logging.info("Added frame")

def controller(car_state):
    steer(0)
    set_speed(0)
    # if car_state.read_task()==2: time.sleep(15)
    while not car_state.is_ready():
        pass
    start = time.time()
    while True:
        if car_state.read_task()==1:
            if car_state.is_ready():
                if car_state.action == "follow_lane":
                    set_speed(10)
                    steer_angle = 0.15*car_state.read_lane_angle()-8
                    steer(steer_angle)
                    # logging.info(f'STEER ANGLE: {steer_angle}')
        elif car_state.read_task()==2:
            # if car_state.is_ready():
            if True:
                logging.info('task: 2')
                if time.time()-start<5: set_speed(15)
                steer_angle = 0.15*car_state.read_lane_angle()-8
                steer(steer_angle)
                if time.time()-start>5:
                    print("should stop now!")
                    set_speed(0)
                '''
                frame = car_state.read_frame()
                cw_dist = crosswalk.find_crosswalk_dist(frame)
                logging.info(f'crosswalk distance: {cw_dist}')
                x = car_state.read_stop_sign()
                logging.info(f'stop sign: {x}')
                if cw_dist<100:
                    set_speed(0)
                '''
        elif car_state.read_task()==3:
            logging.info('task: 3')
            if time.time()-start<2: set_speed(25)
            elif time.time()-start<8: set_speed(10)
            elif time.time()-start<9.5: set_speed(25)
            else: set_speed(0)
            steer_angle = 0.15*car_state.read_lane_angle()-10
            steer(steer_angle)
        elif car_state.read_task()==4:
            parallel_park(None)
            steer(0)
            set_speed(0)
            break

        elif car_state.read_task()==5:
            logging.info('task: 5')
            if time.time()-start<5: 
                set_speed(15)
                steer_angle = 0.15*car_state.read_lane_angle()-8
                steer(steer_angle)
            elif time.time()-start<7.5:
                set_speed(0)
            elif time.time()-start<9:
                set_speed(20)
            elif time.time()-start<15:
                set_speed(23)
                steer(-25)
            elif time.time()-start<17:
                set_speed(15)
                steer(0)
            else:
                set_speed(0)
                steer(0)

        elif car_state.read_task()==6:
            logging.info('task: 6')
            if time.time()-start<3:
                set_speed(0)
                steer(0)
            elif time.time()-start<5:
                set_speed(25)
                steer(-25)
            elif time.time()-start<8:
                steer(25)
            elif time.time()-start<9:
                steer(0)
            elif time.time()-start<11:
                steer(-25)
            elif time.time()-start<12:
                steer(0)
                set_speed(25)
            else:
                set_speed(0)
                steer(0)


def main(task):
    car_state = CarState(task)

    t_controller = Thread(target=controller, args=(car_state,))
    t_read_camera = Thread(target=read_camera, args=(car_state,))
    t_get_camera_lane = Thread(target=get_camera_lane, args=(car_state,))
    t_get_stop_sign = Thread(target=get_stop_sign, args=(car_state,))

    # start camera first, because it takes some time to get first frame
    t_read_camera.start()
    t_controller.start()
    t_get_camera_lane.start()
    t_get_stop_sign.start()

'''
tasks:
1: lane following
2: stop at stop sign
3: slow down at crosswalk
4: parallel parking
5: intersection crossing
6: obstacle detection
'''

if __name__ == '__main__':
    log_format = "%(asctime)s.%(msecs)03d: %(message)s"
    logging.basicConfig(format=log_format, level=logging.INFO, datefmt="%H:%M:%S")
    logging.info("Running main.py")
    TASK = 1
    main(TASK)
    # parallel_park(None)
