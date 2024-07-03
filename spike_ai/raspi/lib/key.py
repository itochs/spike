import os
import sys
import termios
import time
import fcntl


class Key:
    def key_get(self, non_blocking):
        if non_blocking:
            #標準入力のファイルディスクリプタを取得
            self.fd = sys.stdin.fileno()
            
            #fdの端末属性をゲットする
            #oldとnewには同じものが入る。
            #newに変更を加えて、適応する
            #oldは、後で元に戻すため
            self.old = termios.tcgetattr(self.fd)
            self.new = termios.tcgetattr(self.fd)
            
            #new[3]はlflags
            #ICANON(カノニカルモードのフラグ)を外す
            self.new[3] &= ~termios.ICANON
            #ECHO(入力された文字を表示するか否かのフラグ)を外す
            self.new[3] &= ~termios.ECHO
            
            try:
                self.fcntl_old = fcntl.fcntl(self.fd, fcntl.F_GETFL)
                fcntl.fcntl(self.fd, fcntl.F_SETFL, self.fcntl_old |
                            os.O_NONBLOCK)

                # 書き換えたnewをfdに適応する
                termios.tcsetattr(self.fd, termios.TCSANOW, self.new)
                # キーボードから入力を受ける。
                # lfalgsが書き換えられているので、エンターを押さなくても次に進む。echoもしない
                self.ch = sys.stdin.read(1)
            finally:
                fcntl.fcntl(self.fd, fcntl.F_SETFL, self.fcntl_old)
                # fdの属性を元に戻す
                # 具体的にはICANONとECHOが元に戻る
                termios.tcsetattr(self.fd, termios.TCSANOW, self.old)
            return self.ch

        else:
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            self.new = termios.tcgetattr(self.fd)
            self.new[3] &= ~termios.ICANON
            self.new[3] &= ~termios.ECHO
            try:
                termios.tcsetattr(self.fd, termios.TCSANOW, self.new)
                self.ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(self.fd, termios.TCSANOW, self.old)
            return self.ch

    def key_control(self, throttle, steer, non_blocking = True):
        self._key = self.key_get(non_blocking)
        if self._key == 'e':
            throttle = 0
        elif self._key == 'w' and throttle < 100:  # throttle up
            throttle += 5
        elif self._key == 's' and throttle > -100:  # throttle down
            throttle -= 5
        elif self._key == 'r':  # go
            steer = 0
        elif self._key == 'a' and steer > -50:  # right
            steer -= 2 #5
        elif self._key == 'd' and steer < 50:  # left
            steer += 2 #5
        elif self._key == 'q':  # end
            steer = 200
        else:
            pass

        return throttle, steer


class Mode:
    key = Key()

    def __init__(self):
        self.cnt = 0
        self._cnt_list = []
        self.log_flag = False
        self.throttle = 0
        self.steer = 0

    def get_mode_log(self):
        print('Please type mode number')
        print('0:Logging\n1:JustRun')
        self._log_mode = self.key.key_get(False)
        if self._log_mode == '0':
            self.log_flag = True
            print('MODE:LoggingMode\n')
            print('Please type mode number')
            print('0:Overwrite\n1:Adding')
            self._log_type_mode = self.key.key_get(False)
            if self._log_type_mode == '0':
                pass
            elif self._log_type_mode == '1':
                print('Please type start number of digits')
                self._n_digits = int(self.key.key_get(False))
                print(self._n_digits)
                print('Please type start number')
                for self._n_digit in range(self._n_digits):
                    self._cnt_list.append(self.key.key_get(False))
                    print('\r'+''.join(self._cnt_list), end='')
                print('\n')
                self.cnt = ''.join(self._cnt_list)
        elif self._log_mode == '1':
            print('MODE:JustRun\n')
        else:
            print('MODE:JustRun\n')
        time.sleep(1)
        return self.log_flag, int(self.cnt)

    def get_mode_control(self):
        print('Please type mode number')
        print('0:ImageProcessing\n1:KeyControl')
        self.cntl_mode = self.key.key_get(False)
        if self.cntl_mode == '0':
            print('MODE:ImageProcessing\n')
        elif self.cntl_mode == '1':
            print('MODE:KeyControl\n')
            print('key maping\n w:throttleUp\n s:throttleDown\n',
                  'a:left\n d:right\n',
                  'r:steerReset\n e:stop\n q:quit\n')
            time.sleep(2)
        else:
            self.cntl_mode = 100
            print('ERROR:Illegal Input')
        time.sleep(1)
        return int(self.cntl_mode)


if __name__ == '__main__':
    mode = Mode()
    non_blocking = False
    log_flag, cnt = mode.get_mode_log()
    print('log_flag = {}, cnt= {}'.format(log_flag, cnt))
    c_mode, throttle, steer = mode.get_mode_control()
    print('c_mode = {}, throttle = {}, steer = {}'.format(c_mode, throttle,
                                                          steer))
    key = Key()
    while not steer == 200:
        throttle, steer = key.key_control(throttle, steer, non_blocking)
        print('throttle = {}, steer = {}'.format(throttle, steer))
