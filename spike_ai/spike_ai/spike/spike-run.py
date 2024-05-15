import time
import re
import hub

print(" -- serial init -- ")
ser = hub.port.D
ser.mode(hub.port.MODE_FULL_DUPLEX)
time.sleep(1)
ser.baud(115200)
while True:
    reply = ser.read(10000)
    print(reply)
    if reply == b'':
        break

print(" -- device init -- ")
while True:
    # motor init
    r_motor = hub.port.B.motor
    l_motor = hub.port.F.motor
    force_sensor = hub.port.A.device
    if r_motor == None or l_motor == None or force_sensor == None:
        continue
    print("Right Motor :", r_motor)
    print("Left  Motor :", l_motor)
    print("Force Sensor:", force_sensor)
    break

def move_steering(throttle, steer):
    r_motor.run_at_speed(speed = throttle - steer)
    l_motor.run_at_speed(speed = (throttle * -1) - steer)

def stop():
    r_motor.brake()
    l_motor.brake()

if __name__ == "__main__":

    print(" -- start")
    start_flag = True
    throttle = 0
    steer = 0

    while True:
        cmd = ""
        s_time = time.ticks_us()
        while True:
            reply = ser.read(8-len(cmd))
            reply = reply.decode("utf-8")
            cmd = cmd + reply

            if len(cmd) >= 8 or cmd[-1:] == "@":
                e_time = time.ticks_us() - s_time
                cmd = cmd.rstrip("@")
                cmd = cmd.split(",")
                if len(cmd) != 2:
                    cmd = ""
                    continue

                if cmd[0] == "end":
                    print(" -- end")
                    break
                else:
                    if start_flag == True:
                        throttle = int(cmd[0])
                        steer = int(cmd[1])
                        print("throttle:{:3d} steer:{:3d} - {:.f}[msec]".format(throttle, steer, e_time/1000))
                    break

        if cmd[0] == "end":
            break

        if force_sensor.get()[0] != 0:
            print(" -- push force sensor")
            stop()
            while force_sensor.get()[0] != 0:
                pass
            start_flag = not bool(start_flag)
            throttle = 0
            steer = 0

        move_steering(throttle, steer)

    stop()
    print(" -- end -- ")
