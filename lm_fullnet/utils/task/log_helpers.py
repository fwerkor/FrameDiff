import utils

LOG_SCOPE = ""


def _format_log(tag, msg):
    text = str(msg)
    if tag:
        return f"[{LOG_SCOPE}][{tag}] {text}"
    return f"[{LOG_SCOPE}] {text}"


def log_info(msg):
    utils.log.write.info(_format_log(None, msg))


def log_warn(msg):
    utils.log.write.warn(_format_log(None, msg))


def log_error(msg):
    utils.log.write.error(_format_log(None, msg))


def log_step(msg):
    utils.log.write.info(_format_log("阶段", msg))


def log_kv(group, key, value):
    utils.log.write.info(_format_log(group, f"{key}: {value}"))

