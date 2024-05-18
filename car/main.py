import serial
import time
import logging
from threading import Thread
from nav import lane
# from nav import sign
# from nav import crosswalk
import cv2 as cv
import subprocess
import sys
sys.path.insert(0,'brain/src/data/TrafficCommunication')
sys.path.append('brain/src/data/CarsAndSemaphores')
sys.path.append('brain/src/data/TrafficCommunication/useful')
sys.path.append('brain')
from processTrafficCommunication import p_get_loc
from processCarsAndSemaphores import p_get_s1, p_get_s3
print(p_get_loc())
print(p_get_s1())
print(p_get_s3())


class CarState:
    def __init__(self, task):
        self.frame = None
        self.lane_angle = 0 # TODO should this be None?
        self.action = "follow_lane"
        self.frame_counter = 0
        self.task = task
        self.stop_sign = None
        self.semaphore_1 = 'green'
        self.semaphore_3 = 'green'
        self.loc_x = 1.0
        self.loc_y = 1000.0

    def is_ready(self):
        if self.frame is None: return False
        return True

    def update_frame(self, frame):
        self.frame = frame
        cv.imwrite(f'data/frame_{self.frame_counter}.png', self.frame)
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

    def update_semaphore_1(self, s1):
        self.semaphore_1 = s1

    def read_semaphore_1(self):
        return self.semaphore_1

    def update_semaphore_3(self, s3):
        self.semaphore_3 = s3

    def read_semaphore_3(self):
        return self.semaphore_3

    def update_loc(self, loc_x, loc_y):
        self.loc_x = loc_x
        self.loc_y = loc_y

    def read_loc(self):
        return [self.loc_x, self.loc_y]

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
    # TODO changed here
    subprocess.run('v4l2-ctl -c auto_exposure=1', shell=True)
    subprocess.run('v4l2-ctl -c exposure_time_absolute=100', shell=True)
    '''
    # cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 3)
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)
    # x = cap.set(cv.CAP_PROP_EXPOSURE, -10)
    '''
    while True:
        ret, frame = cap.read()
        car_state.update_frame(frame)
        # logging.info("Added frame")

def follow_lane(car_state):
    set_speed(15)
    steer_angle = 0.15*car_state.read_lane_angle()-8
    if steer_angle < -25: 
        steer_angle = -25
    if steer_angle > 25:
        steer_angle = 25
    steer(steer_angle)

def do_traffic_stop():
    set_speed(0)
    time.sleep(2)
    set_speed(15)

def do_turn_left():
    set_speed(15)
    steer(0)
    time.sleep(3)
    steer(-15)
    time.sleep(6)
    steer(0)

def do_turn_right():
    set_speed(15)
    steer(15)
    time.sleep(6)
    steer(0)

def do_roundabout():
    set_speed(15)
    steer(20)
    time.sleep(3)
    steer(-25)
    time.sleep(15)
    steer(25)
    time.sleep(2.5)
    steer(0)
    time.sleep(1)

def controller(car_state):
    steer(0)
    set_speed(0)
    while not car_state.is_ready():
        pass
    logging.info(car_state.read_semaphore_1())
    logging.info(car_state.read_semaphore_3())
    logging.info(car_state.read_loc())
    start = time.time()
    # >> semaphore start
    logging.info('state: semaphore start')
    while car_state.read_semaphore_1() != 'green':
        pass
    # up to node id=270
    while car_state.read_loc()[0] < 14.14:
        follow_lane(car_state)

    # >> slight turn right
    logging.info('state: slight turn right')
    steer(-10)
    time.sleep(5)
    steer(0)
    do_traffic_stop()
    # up to node id=252
    while car_state.read_loc()[1] < 2.15:
        follow_lane(car_state)
    do_traffic_stop()
    while car_state.read_loc()[0]>17.02:
        follow_lane(car_state)

    do_traffic_stop()
    
    # >> sharp turn right
    logging.info('state: sharp turn right')
    do_turn_right()
    # up to node id=316
    while car_state.read_loc()[1]<9.02:
        follow_lane(car_state)
    do_traffic_stop()

    # >> turn at roundabout
    logging.info('state: turn at roundabout')
    do_roundabout()
    # up to node id=443
    while car_state.read_loc()[0] > 0.33 and car_state.read_loc()[1] > 10.81:
        follow_lane(car_state)
    do_traffic_stop()
    # up to node id=483/92
    while car_state.read_loc()[1] > 7.48:
        follow_lane(car_state)
    do_traffic_stop()
    # up to node id=402
    while car_state.read_loc()[1] > 4.59:
        follow_lane(car_state)
    do_traffic_stop()

    # >> turn left for S3
    logging.info('state: turn left at semaphore 3')
    do_turn_left()
    # up to node id=26
    while car_state.read_loc()[1] < 1.93: 
        follow_lane(car_state)

    # >> stop for s3
    logging.info('state: stop for semaphore 3')
    do_traffic_stop()
    while car_state.read_semaphore_3() != 'green':
        set_speed(0)

    # >> turn right after s3
    logging.info('state: turn right after s3')
    do_turn_right()
    # up to node id=73
    while car_state.read_loc()[1] > 1.30:
        follow_lane(car_state)
    do_traffic_stop()

    # >> turn left
    logging.info('state: turn left')
    do_turn_left()
    # up to node id=185
    while car_state.read_loc()[0] < 4.21:
        follow_lane(car_state)
    do_traffic_stop()
    # up to node id=192
    while car_state.read_loc()[1] < 0.86:
        follow_lane(car_state)

    # >> slight right to parking
    logging.info('state: slight right to parking')
    steer(20)
    time.sleep(4.5)
    # up to node id=226
    while car_state.read_loc()[0] < 8.17:
        follow_lane(car_state)
    do_traffic_stop()
    # up to node id=240
    while car_state.read_loc()[0] < 13.47:
        follow_lane(car_state)
    do_traffic_stop()

    # >> park
    logging.info('state: park')
    parallel_park(car_state)

def old_controller(car_state):
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
                    set_speed(15)
                    steer_angle = 0.1*car_state.read_lane_angle()-8
                    steer(steer_angle)
                    logging.info(f'STEER ANGLE: {steer_angle}')
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


def get_s1(car_state):
    while True:
        car_state.update_semaphore_1(p_get_s1())
        time.sleep(0.05)

def get_s3(car_state):
    while True:
        car_state.update_semaphore_3(p_get_s3())
        time.sleep(0.05)

def get_loc(car_state):
    while True:
        tmp_list = p_get_loc()
        x_tmp = tmp_list[0]
        y_tmp = tmp_list[1]
        car_state.update_loc(float(x_tmp), float(y_tmp))
        time.sleep(0.05)

def main(task):
    car_state = CarState(task)

    t_controller = Thread(target=controller, args=(car_state,))
    t_read_camera = Thread(target=read_camera, args=(car_state,))
    t_get_camera_lane = Thread(target=get_camera_lane, args=(car_state,))
    t_get_stop_sign = Thread(target=get_stop_sign, args=(car_state,))
    t_get_s1 = Thread(target=get_s1, args=(car_state,))
    t_get_s3 = Thread(target=get_s3, args=(car_state,))
    t_get_loc = Thread(target=get_loc, args=(car_state,))

    # start camera first, because it takes some time to get first frame
    t_read_camera.start()
    t_controller.start()
    t_get_camera_lane.start()
    t_get_stop_sign.start()
    t_get_s1.start()
    t_get_s3.start()
    t_get_loc.start()

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
