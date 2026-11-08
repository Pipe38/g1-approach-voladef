#!/usr/bin/env python3
"""Lancia VolaDef quando una persona entra nel metro davanti alla testa.

Il feed è lo stream stereo side-by-side della testa (come
G1GrokChatbot/robot/stereo_feed.py). La distanza è

    Z = focale_px * baseline_m / disparità_px

La focale si ricava dal campo visivo orizzontale di un occhio. Baseline e
campo visivo sono quelli tipici della camera 3D USB della testa; se il metro
reale non coincide, si correggono con --baseline e --hfov. In piedi a un
metro dalla testa, la distanza stampata deve leggere circa 1.0.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
import socket
import struct
import threading
import time

import cv2
import numpy as np

import unitree_sdk2py.core.channel as channel
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.g1.arm.g1_arm_action_client import G1ArmActionClient
from unitree_sdk2py.g1.loco.g1_loco_api import ROBOT_API_ID_LOCO_SET_FSM_ID
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

from stereo import StereoEyes

TEACH_API = 7108
TEACH_NAME = "VolaDef"
POLICY_FSM = 802
# Base misurata con una persona ferma a 1.00 m: disparità della testa 22 px
# con i due occhi scostati di 14 px in verticale. Focale 457 px.
DEFAULT_BASELINE_M = 0.04814
# Altezza del riquadro della persona a 1 m, nello stesso fotogramma di calibrazione.
BOX_PX_AT_1M = 310.0
# Una persona resta la stessa finché il riquadro non sparisce per un secondo.
TRACK_HOLD_S = 1.0
# Deve restare davvero oltre la soglia di uscita, non un fotogramma solo.
EXIT_HOLD_S = 0.5
# Mano destra del firmware: ultimo ottetto 201, porta 4210.
HAND_IP = "192.168.123.201"
HAND_PORT = 4210
GRAB_ON_S = 3.0
GRAB_OFF_S = 11.0
DEFAULT_HFOV_DEG = 70.0
PERSON_CLASS = 15
ROOT = os.path.dirname(os.path.abspath(__file__))


def log(msg: str) -> None:
    now = time.time()
    ms = int(now * 1000) % 1000
    print(time.strftime("%H:%M:%S") + ".%03d %s" % (ms, msg), flush=True)


def focal_px(eye_width: int, hfov_deg: float) -> float:
    half = math.radians(hfov_deg) / 2.0
    return (eye_width / 2.0) / math.tan(half)


class People:
    """MobileNet-SSD addestrato su VOC. La classe 15 e' la persona."""

    def __init__(self, proto: str, weights: str):
        self.net = cv2.dnn.readNetFromCaffe(proto, weights)

    def detect(self, image: np.ndarray, confidence: float = 0.5):
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, 0.007843, (300, 300), 127.5, swapRB=False, crop=False,
        )
        self.net.setInput(blob)
        det = self.net.forward()
        found = []
        for i in range(det.shape[2]):
            if int(det[0, 0, i, 1]) != PERSON_CLASS:
                continue
            score = float(det[0, 0, i, 2])
            if score < confidence:
                continue
            x1 = max(0, int(det[0, 0, i, 3] * width))
            y1 = max(0, int(det[0, 0, i, 4] * height))
            x2 = min(width - 1, int(det[0, 0, i, 5] * width))
            y2 = min(height - 1, int(det[0, 0, i, 6] * height))
            if x2 - x1 < 24 or y2 - y1 < 48:
                continue
            found.append((x1, y1, x2 - x1, y2 - y1, score))
        found.sort(key=lambda item: item[2] * item[3], reverse=True)
        return found


def disparity_px(left: np.ndarray, right: np.ndarray, box, max_disp: int):
    """Disparità sulla testa. Il busto nero non ha contrasto e i due occhi
    non sono allineati in verticale: si cerca anche uno scarto di qualche riga.
    """
    x, y, w, h = box[:4]
    y1 = y + int(h * 0.05)
    y2 = y + int(h * 0.40)
    x1 = x + int(w * 0.28)
    x2 = x + int(w * 0.72)
    if y2 - y1 < 16 or x2 - x1 < 16:
        return None
    left_g = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_g = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    template = left_g[y1:y2, x1:x2]
    best = None
    # Lo scarto verticale fra i due occhi è fisso, circa 14 px.
    for dy in range(8, 18, 2):
        yy1 = y1 + dy
        yy2 = y2 + dy
        if yy1 < 0 or yy2 > right_g.shape[0]:
            continue
        rx1 = max(0, x1 - max_disp)
        rx2 = min(right_g.shape[1], x2 + 24)
        if rx2 - rx1 <= template.shape[1] + 4:
            continue
        strip = right_g[yy1:yy2, rx1:rx2]
        score_map = cv2.matchTemplate(strip, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(score_map)
        disp = x1 - (rx1 + int(loc[0]))
        if disp < 4:
            continue
        if best is None or score > best[0]:
            best = (float(score), float(disp))
    if best is None or best[0] < 0.52:
        return None
    return best[1], best[0]


class HandPump:
    """Grab UDP della pompa. Se la mano non c'è, gli invii falliscono e basta.

    Il firmware spegne la pompa se non arriva nulla per 2 secondi, quindi
    fra i 3 e gli 11 secondi il grab alto viene ripetuto, e ogni comando
    parte più volte di fila.
    """

    def __init__(self, ip=HAND_IP, port=HAND_PORT):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._start = None
        self._warned = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def begin(self):
        with self._lock:
            self._start = time.monotonic()
            self._warned = False
        log("mano: grab al massimo a %.0f s, aperto a %.0f s" % (GRAB_ON_S, GRAB_OFF_S))

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.sock.close()
        except OSError:
            pass

    def _send(self, grab, times):
        payload = struct.pack("<fff", float(grab), 0.0, 0.0)
        for _ in range(times):
            if self._stop.is_set():
                return
            try:
                self.sock.sendto(payload, self.addr)
            except OSError as exc:
                if not self._warned:
                    self._warned = True
                    log("mano non raggiungibile (%s), continuo senza" % exc)
                return
            time.sleep(0.04)

    def _run(self):
        announced = None
        try:
            while not self._stop.is_set():
                with self._lock:
                    start = self._start
                if start is None:
                    announced = None
                    self._stop.wait(0.1)
                    continue
                elapsed = time.monotonic() - start
                if elapsed < GRAB_ON_S:
                    self._stop.wait(0.05)
                    continue
                if elapsed < GRAB_OFF_S:
                    if announced != "on":
                        announced = "on"
                        log("mano: grab al massimo")
                        self._send(1.0, 8)
                    else:
                        self._send(1.0, 3)
                    self._stop.wait(0.35)
                    continue
                if announced != "off":
                    announced = "off"
                    log("mano: grab a zero")
                    self._send(0.0, 8)
                with self._lock:
                    if self._start == start:
                        self._start = None
        except Exception as exc:
            log("mano: %s, continuo senza" % exc)


class VolaDef:
    def __init__(self):
        channel.ChannelConfigHasInterface = channel.ChannelConfigHasInterface.replace(
            "/tmp/cdds.LOG", "/tmp/cdds_approach.LOG"
        )
        ChannelFactoryInitialize(0, "eth0")
        self.loco = LocoClient()
        self.loco.SetTimeout(1.5)
        self.loco.Init()
        self.arm = G1ArmActionClient()
        self.arm.SetTimeout(3.0)
        self.arm.Init()
        self.arm._RegistApi(TEACH_API, 0)

    def launch(self) -> int:
        code, _ = self.arm._Call(TEACH_API, json.dumps({"action_name": TEACH_NAME}))
        if code == 0:
            log("VolaDef lanciato")
            return 0
        if code == 7404:
            set_code, _ = self.loco._Call(
                ROBOT_API_ID_LOCO_SET_FSM_ID, json.dumps({"data": POLICY_FSM})
            )
            log("teaching rifiutato in questo FSM, passo a %s (risposta %s)" % (POLICY_FSM, set_code))
            time.sleep(0.6)
            code, _ = self.arm._Call(TEACH_API, json.dumps({"action_name": TEACH_NAME}))
            if code == 0:
                log("VolaDef lanciato dopo il passaggio a %s" % POLICY_FSM)
                return 0
        if code == 7400:
            log("gesto ancora in corso")
            return code
        log("VolaDef non partito, codice %s" % code)
        return code


def _center(box):
    return box[0] + box[2] * 0.5, box[1] + box[3] * 0.5


class Track:
    """Una persona seguita nel tempo. Il gesto scatta una volta per ingresso."""

    def __init__(self, box, now):
        self.box = box
        self.seen = now
        self.history = deque(maxlen=5)
        self.hits = 0
        self.greeted = False
        self.outside_since = None

    def follows(self, box) -> bool:
        cx, cy = _center(box)
        tx, ty = _center(self.box)
        reach = 0.6 * max(box[2], box[3], self.box[2], self.box[3])
        return math.hypot(cx - tx, cy - ty) <= max(reach, 40.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="VolaDef quando qualcuno entra nel metro")
    parser.add_argument("--distance", type=float, default=1.5, help="soglia in metri")
    parser.add_argument("--leave", type=float, default=1.9, help="oltre questa distanza si riarma")
    parser.add_argument("--baseline", type=float, default=DEFAULT_BASELINE_M, help="base stereo in metri")
    parser.add_argument("--hfov", type=float, default=DEFAULT_HFOV_DEG, help="campo visivo orizzontale di un occhio, gradi")
    parser.add_argument("--port", type=int, default=5610, help="porta RTP locale, diversa da quella del chatbot")
    parser.add_argument("--hits", type=int, default=3, help="rilevamenti consecutivi sotto soglia")
    parser.add_argument("--dry-run", action="store_true", help="stima la distanza senza lanciare VolaDef")
    args = parser.parse_args()

    feed = StereoEyes(args.port)
    log("cerco lo streamer stereo")
    info = feed.start()
    log("stream da %s, %sx%s %s, porta locale %s" % (
        info.get("server"), info.get("width"), info.get("height"), info.get("codec"), args.port,
    ))
    people_net = People(
        os.path.join(ROOT, "models", "deploy.prototxt"),
        os.path.join(ROOT, "models", "mobilenet_iter_73000.caffemodel"),
    )
    trigger = None if args.dry_run else VolaDef()
    hand = HandPump()
    fx = None
    last_report = 0.0
    last_frame = time.time()
    next_launch = 0.0
    tracks = []

    try:
        while True:
            frame = feed.latest(0.5)
            if frame is None:
                if time.time() - last_frame > 5.0:
                    log("nessun frame, riprovo lo stream")
                    feed.close()
                    info = feed.start()
                    log("stream da %s" % info.get("server"))
                    last_frame = time.time()
                continue
            last_frame = time.time()
            eye_w = frame.shape[1] // 2
            left = frame[:, :eye_w]
            right = frame[:, eye_w:eye_w * 2]
            if fx is None:
                fx = focal_px(eye_w, args.hfov)
                log("focale %.0f px, base %.3f m, soglia %.2f m" % (fx, args.baseline, args.distance))
            people = people_net.detect(left)
            now = time.time()
            used = set()
            closest = None
            for box in people:
                track = next((t for t in tracks if id(t) not in used and t.follows(box)), None)
                if track is None:
                    track = Track(box, now)
                    tracks.append(track)
                used.add(id(track))
                track.box = box
                track.seen = now
                match = disparity_px(left, right, box, max_disp=min(200, eye_w // 3))
                if match is None:
                    continue
                disp, score = match
                raw = fx * args.baseline / disp
                # Pochi pixel con una persona ancora grande sono un falso aggancio,
                # non qualcuno che se ne va. Non aggiornano la distanza.
                size_m = BOX_PX_AT_1M / max(float(box[3]), 1.0)
                if raw > size_m * 1.8:
                    continue
                track.history.append(disp)
                if len(track.history) < 3:
                    continue
                disp = sorted(track.history)[len(track.history) // 2]
                distance = fx * args.baseline / disp
                if closest is None or distance < closest[0]:
                    closest = (distance, score, disp)
                if distance > args.leave:
                    track.hits = 0
                    if track.outside_since is None:
                        track.outside_since = now
                    elif now - track.outside_since >= EXIT_HOLD_S and track.greeted:
                        track.greeted = False
                        log("persona oltre %.2f m, può rifare il gesto" % args.leave)
                    continue
                track.outside_since = None
                if distance <= args.distance:
                    track.hits += 1
                else:
                    track.hits = 0
                if track.hits < args.hits or track.greeted or now < next_launch:
                    continue
                log("persona a %.2f m" % distance)
                if trigger is None:
                    log("dry-run, VolaDef non lanciato")
                    track.greeted = True
                else:
                    code = trigger.launch()
                    if code == 0:
                        track.greeted = True
                        hand.begin()
                    elif code == 7400:
                        next_launch = now + 2.0
                    else:
                        next_launch = now + 2.0
                if track.greeted:
                    next_launch = now + 1.0
            if closest is not None and now - last_report > 0.8:
                distance, score, disp = closest
                log("distanza %.2f m (disparità %.0f px, match %.2f, persone %d)" % (
                    distance, disp, score, len(people),
                ))
                last_report = now
            tracks = [t for t in tracks if now - t.seen <= TRACK_HOLD_S]
    finally:
        hand.close()
        feed.close()


if __name__ == "__main__":
    main()
