"""Ricevitore del feed stereo della testa del G1.

Stesso protocollo di G1GrokChatbot/robot/stereo_feed.py: hello UDP sulla
porta 5601, magic "G1ST", poi H.264 in RTP unicast sulla porta che
chiediamo. Qui teniamo il frame intero side-by-side, non solo l'occhio
sinistro, perché la distanza si stima dal disallineamento fra i due occhi.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from typing import Optional

import numpy as np

MAGIC = "G1ST"
DISCOVERY_PORT = 5601
HEARTBEAT_S = 2.0


def discover(timeout: float, video_port: int, discovery_port: int = DISCOVERY_PORT) -> dict:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(1.0)
    hello = json.dumps({"magic": MAGIC, "op": "hello", "port": video_port}).encode("utf-8")
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            sock.sendto(hello, ("255.255.255.255", discovery_port))
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            message = json.loads(data.decode("utf-8"))
            if message.get("magic") == MAGIC and message.get("op") == "welcome":
                message["server"] = addr[0]
                return message
    finally:
        sock.close()
    raise TimeoutError(
        "Streamer stereo non trovato. La testa deve essere accesa e il "
        "broadcast UDP %s deve arrivare." % discovery_port
    )


class StereoEyes:
    """Ultimo frame BGR affiancato (sinistra | destra)."""

    def __init__(self, video_port: int, eye_width: int = 640, eye_height: int = 480):
        self.video_port = video_port
        self.eye_width = eye_width
        self.eye_height = eye_height
        self.info = None
        self._stop = threading.Event()
        self._heartbeat = None
        self._pipeline = None
        self._sink = None
        self._gst = None

    def start(self, timeout: float = 8.0) -> dict:
        self.info = discover(timeout, self.video_port)
        self._stop.clear()
        self._heartbeat = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat.start()
        self._start_pipeline()
        return self.info

    def latest(self, timeout_s: float = 0.5) -> Optional[np.ndarray]:
        if self._sink is None or self._gst is None:
            return None
        sample = self._sink.emit("try-pull-sample", int(timeout_s * self._gst.SECOND))
        if sample is None:
            return None
        buf = sample.get_buffer()
        ok, mapped = buf.map(self._gst.MapFlags.READ)
        if not ok:
            return None
        try:
            caps = sample.get_caps().get_structure(0)
            width = caps.get_value("width")
            height = caps.get_value("height")
            frame = np.ndarray((height, width, 4), np.uint8, mapped.data).copy()
        finally:
            buf.unmap(mapped)
        return frame[:, :, :3]

    def close(self) -> None:
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=1.0)
            self._heartbeat = None
        if self._pipeline is not None:
            self._pipeline.set_state(self._gst.State.NULL)
            self._pipeline = None
            self._sink = None

    def _heartbeat_loop(self) -> None:
        server = (self.info or {}).get("server")
        if not server:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        hello = json.dumps({
            "magic": MAGIC, "op": "hello", "port": self.video_port,
        }).encode("utf-8")
        while not self._stop.wait(HEARTBEAT_S):
            try:
                sock.sendto(hello, (server, DISCOVERY_PORT))
            except OSError:
                pass
        sock.close()

    def _start_pipeline(self) -> None:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst
        Gst.init(None)
        self._gst = Gst

        codec = (self.info or {}).get("codec", "h264")
        if codec == "h265":
            encoding, depay, parse, decoder = "H265", "rtph265depay", "h265parse", "avdec_h265"
        else:
            encoding, depay, parse, decoder = "H264", "rtph264depay", "h264parse", "nvv4l2decoder"
        out_w = self.eye_width * 2
        out_h = self.eye_height

        def launch(decoder_name: str) -> None:
            desc = (
                "udpsrc port=%d buffer-size=4194304 "
                'caps="application/x-rtp,media=video,encoding-name=%s,payload=96,clock-rate=90000" '
                "! rtpjitterbuffer latency=60 "
                "! %s ! %s ! %s ! nvvidconv interpolation-method=1 "
                "! video/x-raw,format=BGRx,width=%d,height=%d "
                "! appsink name=sink max-buffers=1 drop=true sync=false"
                % (self.video_port, encoding, depay, parse, decoder_name, out_w, out_h)
            )
            self._pipeline = Gst.parse_launch(desc)
            self._sink = self._pipeline.get_by_name("sink")
            result = self._pipeline.set_state(Gst.State.PLAYING)
            if result == Gst.StateChangeReturn.FAILURE:
                self._pipeline.set_state(Gst.State.NULL)
                raise RuntimeError("pipeline non partita con %s" % decoder_name)

        try:
            launch(decoder)
        except Exception:
            if decoder == "nvv4l2decoder":
                launch("avdec_h264" if codec != "h265" else "avdec_h265")
            else:
                raise
