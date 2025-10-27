import sys

class TeeLogger:
    def __init__(self, filepath):
        # Use line buffering: flushes on every newline
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Needed by some tools using sys.stdout (e.g., tqdm)
        self.terminal.flush()
        self.log.flush()
