import signal

import utils


def _protect_log(tag, message):
    text = str(message)
    if tag:
        return f"[守护模块][{tag}] {text}"
    return f"[守护模块] {text}"


def _ignore_sigterm():
    try:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except ValueError:
        pass


def task(params):
    utils.log.write.info(_protect_log("任务", "整网链路"))
    utils.log.write.info(_protect_log("任务", f"任务参数: {params}"))
    _ignore_sigterm()
    from utils.task import fullnet

    return fullnet.main(params)
