import sys
import time
import signal


def term_sig_handler(signum, frame):
    print('catched singal: %d' % signum)
    sys.exit()


if __name__ == '__main__':
    # catch term signal
    signal.signal(signal.SIGTERM, term_sig_handler)
    signal.signal(signal.SIGINT, term_sig_handler)

    while True:
        print('hello')
        time.sleep(3)
