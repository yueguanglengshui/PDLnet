import time


def print_log(print_string, log):
    print("{}".format(print_string))
    if log is not None:
        log.write('{}\n'.format(print_string))
        log.flush()


def time_for_file():
    return '{}'.format(time.strftime(time.strftime("%Y-%m-%d-%H-%M-%S")))
