class StdoutCsvTee:
    """Mirror stdout line-by-line into a CSV log while still printing to the console."""

    def __init__(self, stream, csv_path: Path):
        self.stream = stream
        self._buffer = ''
        self._file = open(csv_path, 'w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._file)
        self._writer.writerow(['timestamp', 'message'])
        self._file.flush()

    def write(self, data):
        self.stream.write(data)
        if not isinstance(data, str):
            data = str(data)
        self._buffer += data
        while '\n' in self._buffer:
            line, self._buffer = self._buffer.split('\n', 1)
            self._writer.writerow([datetime.now().isoformat(timespec='seconds'), line])
            self._file.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self._file.flush()

    def close(self):
        if self._buffer:
            self._writer.writerow([datetime.now().isoformat(timespec='seconds'), self._buffer])
            self._buffer = ''
            self._file.flush()
        self._file.close()

    def isatty(self):
        return self.stream.isatty()

    def __getattr__(self, name):
        return getattr(self.stream, name)