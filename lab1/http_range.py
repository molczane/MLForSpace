"""Seekable public HTTP reader with bounded parallel range transfers for COGs."""
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPException
from io import RawIOBase
from time import sleep
from urllib.request import Request, urlopen


class HTTPRangeReader(RawIOBase):
    def __init__(self, url, size):
        self.name, self.size, self.position = url, size, 0
        self._small_blocks = {}

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if position < 0:
            raise ValueError("Negative file offset")
        self.position = position
        return position

    def _range(self, bounds):
        start, end = bounds
        for attempt in range(3):
            try:
                request = Request(self.name, headers={"Range": f"bytes={start}-{end-1}"})
                with urlopen(request, timeout=60) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end-1}/{self.size}":
                        raise OSError("Archive did not return the requested byte range")
                    data = response.read()
                if len(data) != end-start:
                    raise OSError("Truncated HTTP byte range")
                return data
            except (OSError, HTTPException):
                if attempt == 2:
                    raise
                sleep(attempt+1)

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position+size)
        if self.position >= end:
            return b""
        if 0 <= size <= 16384 and self.position // 16384 == (end-1) // 16384:
            start = self.position // 16384 * 16384
            if start not in self._small_blocks:
                self._small_blocks[start] = self._range((start, min(start+16384, self.size)))
            data = self._small_blocks[start][self.position-start:end-start]
            self.position = end
            return data
        ranges = [(start, min(start+262144, end)) for start in range(self.position, end, 262144)]
        if not ranges:
            return b""
        if len(ranges) == 1:
            data = self._range(ranges[0])
        else:
            with ThreadPoolExecutor(max_workers=8) as pool:
                data = b"".join(pool.map(self._range, ranges))
        self.position = end
        return data


def open_asset(asset):
    """Use Rasterio's file opener protocol without loading the entire remote file."""
    import rasterio

    url = asset["href"]

    def opener(path, mode="rb"):
        if path != url:
            raise FileNotFoundError(path)
        return HTTPRangeReader(url, asset["file:size"])

    return rasterio.open(url, opener=opener)
