import sys

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")
        self._closed = False

    def write(self, message):
        self.terminal.write(message)  # 输出到控制台
        self.log.write(message)       # 写入日志文件

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        if self._closed:
            return
        try:
            self.log.close()
        finally:
            self._closed = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()