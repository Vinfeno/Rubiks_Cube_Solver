#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile

SPEED = 1000
TRAY_FULL_ROTATION = 1080
SCAN_MID_POSITION = -500    
SCAN_EDGE_POSITION = -320
SCAN_CORNER_POSITION = -250
CORRECTION_ANGLE = 60
TRAY_SCAN_CORRECTION = 30


ev3 = EV3Brick()
motor_tilt = Motor(Port.A)
motor_tray = Motor(Port.B)
motor_tilt.run_until_stalled(-100, then=Stop.HOLD)
motor_tilt.reset_angle(0)

def safe_grab():
    motor_tilt.run_angle(2000, 40, then=Stop.HOLD)
    motor_tilt.run_angle(2000, -40, then=Stop.HOLD)
    motor_tray.run_angle(2000, 70, then=Stop.HOLD)
    motor_tray.run_angle(2000, -130, then=Stop.HOLD)
    motor_tray.run_angle(2000, 60, then=Stop.HOLD)
    motor_tilt.run_angle(2000, 40, then=Stop.HOLD)
    motor_tilt.run_angle(2000, -40, then=Stop.HOLD)

def tilt_cube(times=1):
    # motor_tilt.run_angle(400, 100, then=Stop.HOLD)
    grab_cube()
    # safe_grab()
    for i in range(times):
        motor_tilt.run_angle(700, 115, then=Stop.HOLD)
        motor_tilt.hold()
        wait(50)
        if i + 1 != times:
            motor_tilt.run_angle(1000, -145, then=Stop.HOLD)
            motor_tilt.run_angle(1000, 30, then=Stop.HOLD)
    motor_tilt.hold()
    wait(50)
    ungrab_cube()

def grab_cube():
    motor_tilt.run_angle(300, 110, then=Stop.HOLD)
    safe_grab()

def ungrab_cube():
    # motor_tilt.run_angle(500, -130, then=Stop.HOLD)
    motor_tilt.run_until_stalled(-400)
    motor_tilt.hold()
    motor_tilt.reset_angle(0)

def rotate_tray(direction, turns=1, with_cube=False):
    rotation_speed = SPEED if direction == "COUNTERCLOCKWISE" else -SPEED
    angle = 90 * 3 * turns + CORRECTION_ANGLE
    directed_angle = angle if direction == "COUNTERCLOCKWISE" else -angle
    if with_cube:
        motor_tray.run_angle(rotation_speed, directed_angle, then=Stop.HOLD)
        # motor_tray.brake()
        correct_tray_position(direction)
    else:
        motor_tray.run_angle(rotation_speed, 90 * 3 * turns, then=Stop.HOLD)
        # motor_tray.brake()
    motor_tray.hold()
    
def correct_tray_position(direction):
    if direction == "COUNTERCLOCKWISE":
        motor_tray.run_angle(SPEED, -CORRECTION_ANGLE, then=Stop.HOLD)
    else:
        motor_tray.run_angle(-SPEED, CORRECTION_ANGLE, then=Stop.HOLD)

def execute_instruction(instruction, current_face):
    if instruction[0] == "F":
        face = "FRONT"
    elif instruction[0] == "D": 
        face = "DOWN"
    elif instruction[0] == "U":
        face = "UP"
    elif instruction[0] == "L":
        face = "LEFT"
    elif instruction[0] == "R":
        face = "RIGHT"
    elif instruction[0] == "B":
        face = "BACK"
    if face != current_face:
        current_face = turn_to_target_face(face, current_face)

    if len(instruction) > 1 and instruction[1] == "'":
        direction = "COUNTERCLOCKWISE"
    else:
        direction = "CLOCKWISE"

    num_turns = 1
    if len(instruction) == 2 and instruction[1] == "2":
        num_turns = 2
    if len(instruction) == 3:
        num_turns = 2

    grab_cube()
    rotate_tray(direction, num_turns, True)
    ungrab_cube()
    return current_face

def turn_to_target_face(target, current_face):
    print(target,current_face)
    if current_face == "FRONT":
        if target == "UP":
            tilt_cube()
            current_face = "UP"
        elif target == "DOWN":
            tilt_cube(3)
            current_face = "DOWN"
        elif target == "LEFT":
            rotate_tray("COUNTERCLOCKWISE", 1)
            tilt_cube()
            rotate_tray("CLOCKWISE", 1)
            current_face = "LEFT"
        elif target == "RIGHT":
            rotate_tray("CLOCKWISE", 1)
            tilt_cube()
            rotate_tray("COUNTERCLOCKWISE", 1)
            current_face = "RIGHT"
        elif target == "BACK":
            rotate_tray("CLOCKWISE", 2)
            tilt_cube(2)
            current_face = "BACK"

    elif current_face == "UP":
        if target == "BACK":
            tilt_cube()
            current_face = "BACK"
        elif target == "FRONT":
            tilt_cube(3)
            current_face = "FRONT"
        elif target == "LEFT":
            rotate_tray("COUNTERCLOCKWISE", 1)
            tilt_cube()
            rotate_tray("CLOCKWISE", 2)
            current_face = "LEFT"
        elif target == "RIGHT":
            rotate_tray("CLOCKWISE", 1)
            tilt_cube()
            rotate_tray("COUNTERCLOCKWISE", 2)
            current_face = "RIGHT"
        elif target == "DOWN":
            tilt_cube(2)
            current_face = "DOWN"

    elif current_face == "DOWN":
        if target == "FRONT":
            tilt_cube()
            current_face = "FRONT"
        elif target == "UP":
            tilt_cube(2)
            current_face = "UP"
        elif target == "LEFT":
            rotate_tray("CLOCKCLOCKWISE", 1)
            tilt_cube()
            current_face = "LEFT"
        elif target == "RIGHT":
            rotate_tray("CLOCKWISE", 1)
            tilt_cube()
            current_face = "RIGHT"
        elif target == "BACK":
            rotate_tray("CLOCKWISE", 2)
            tilt_cube(3)
            current_face = "BACK"
        
    elif current_face == "LEFT":
        if target == "FRONT":
            rotate_tray("CLOCKWISE", 1)
            tilt_cube()
            rotate_tray("COUNTERCLOCKWISE", 1)
            current_face = "FRONT"
        elif target == "UP":
            tilt_cube(1)
            rotate_tray("COUNTERCLOCKWISE", 1)
            current_face = "UP"
        elif target == "DOWN":
            rotate_tray("CLOCKWISE", 2)
            tilt_cube()
            rotate_tray("COUNTERCLOCKWISE", 1)
            current_face = "DOWN"
        elif target == "RIGHT":
            tilt_cube(2)
            rotate_tray("CLOCKWISE", 2)
            current_face = "RIGHT"
        elif target == "BACK":
            rotate_tray("COUNTERCLOCKWISE", 1)
            tilt_cube()
            rotate_tray("CLOCKWISE", 1)
            current_face = "BACK"
        
    elif current_face == "RIGHT":
        if target == "FRONT":
            rotate_tray("COUNTERCLOCKWISE", 1)
            tilt_cube()
            rotate_tray("CLOCKWISE", 1)
            current_face = "FRONT"
        elif target == "UP":
            tilt_cube(1)
            rotate_tray("CLOCKWISE", 1)
            current_face = "UP"
        elif target == "DOWN":
            rotate_tray("COUNTERCLOCKWISE", 2)
            tilt_cube()
            rotate_tray("CLOCKWISE", 1)
            current_face = "DOWN"
        elif target == "LEFT":
            tilt_cube(2)
            rotate_tray("CLOCKWISE", 2)
            current_face = "LEFT"
        elif target == "BACK":
            rotate_tray("CLOCKWISE", 1)
            tilt_cube()
            rotate_tray("COUNTERCLOCKWISE", 1)
            current_face = "BACK"

    elif current_face == "BACK":
        if target == "UP":
            tilt_cube()
            rotate_tray("CLOCKWISE", 2)
            current_face = "UP"
        elif target == "DOWN":
            rotate_tray("CLOCKWISE", 2)
            tilt_cube()
            current_face = "DOWN"
        elif target == "LEFT":
            rotate_tray("CLOCKWISE", 1)
            tilt_cube()
            rotate_tray("COUNTERCLOCKWISE", 1)
            current_face = "LEFT"
        elif target == "RIGHT":
            rotate_tray("COUNTERCLOCKWISE", 1)
            tilt_cube()
            rotate_tray("CLOCKWISE", 1)
            current_face = "RIGHT"
        elif target == "FRONT":
            rotate_tray("CLOCKWISE", 2)
            tilt_cube(2)
            current_face = "FRONT"
    return current_face


with open("solution.txt", 'r') as file:
    solution = file.read()
    print(solution)

solution = solution.split(',')[:-1]
print(solution)

while Button.CENTER not in ev3.buttons.pressed():
    if len(ev3.buttons.pressed()) >= 0:
        if Button.LEFT in ev3.buttons.pressed():
            motor_tray.run(50)
        elif Button.RIGHT in ev3.buttons.pressed():
            motor_tray.run(-50)
        else:
            motor_tray.brake()   

current_face = "FRONT"
counter = 1
for instruction in solution:
    print(str(counter) + "/" + str(len(solution)) + " " + instruction)
    ev3.screen.print(str(counter) + "/" + str(len(solution)) + " " + instruction)
    current_face = execute_instruction(instruction, current_face)
    counter += 1
motor_tilt.run_until_stalled(-500)
motor_tilt.hold()
rotate_tray("CLOCKWISE", turns=8)