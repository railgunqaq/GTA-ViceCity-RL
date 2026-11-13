"""reVC ML protocol v7: verified mission telemetry and lockstep snapshots.

Diagnostic hold() runs freely. step(frames=N) applies one action for exactly N
10ms simulation ticks and freezes the completed observation while learning.
Game and client must both use protocol v7. Mission objectives have semantic
indices independent of radar-marker creation; unverified telemetry is invalid.
"""

import ctypes
import ctypes.wintypes as wt
from contextlib import contextmanager
import math
import os
import threading
import time
import weakref

FILE_MAP_ALL_ACCESS = 0xF001F
TH32CS_SNAPPROCESS = 0x2
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STILL_ACTIVE = 259

ML_MAGIC = 0x43564552  # 'REVC'
ML_VERSION = 7
SENSOR_SCHEMA = "collision-rays-v2"
_LOCAL_CONTROLLERS = weakref.WeakValueDictionary()
_LOCAL_CONTROLLERS_LOCK = threading.Lock()


class GameDeadError(RuntimeError):
    """The game process is gone."""


class ControllerBusyError(RuntimeError):
    """Another controller owns this game's bridge."""


class GeometryUnavailableError(RuntimeError):
    """A native query confirmed collision data is unavailable, never clear."""

    def __init__(self, result):
        self.query_status = result['status']
        self.result = result
        super().__init__('native path geometry unavailable, status=%d' % self.query_status)


class MlBlip(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float),
                ("color", ctypes.c_uint32), ("blipType", ctypes.c_uint32),
                ("sprite", ctypes.c_uint32), ("blipId", ctypes.c_uint32),
                ("objectiveKind", ctypes.c_int32), ("objectiveIndex", ctypes.c_int32)]


class MlObjective(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int32), ("index", ctypes.c_int32),
                ("valid", ctypes.c_int32), ("x", ctypes.c_float),
                ("y", ctypes.c_float), ("z", ctypes.c_float)]


class MlHit(ctypes.Structure):
    _fields_ = [("valid", ctypes.c_int32), ("hit", ctypes.c_int32),
                ("entityType", ctypes.c_int32), ("modelId", ctypes.c_int32),
                ("surface", ctypes.c_int32), ("distance", ctypes.c_float),
                ("point", ctypes.c_float * 3), ("normal", ctypes.c_float * 3)]


class MlQueryResult(ctypes.Structure):
    _fields_ = [("seq", ctypes.c_uint32), ("status", ctypes.c_int32),
                ("frame", ctypes.c_uint32), ("episodeId", ctypes.c_uint32),
                ("radius", ctypes.c_float), ("sampleSpacing", ctypes.c_float),
                ("length", ctypes.c_float), ("line", MlHit), ("sweep", MlHit)]


class MlState(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32), ("version", ctypes.c_uint32),
        ("size", ctypes.c_uint32), ("pid", ctypes.c_uint32),
        ("seq", ctypes.c_uint32), ("frame", ctypes.c_uint32),
        ("missionRunning", ctypes.c_int32), ("cutsceneProcessing", ctypes.c_int32),
        ("playerState", ctypes.c_int32),
        ("gameTimerMs", ctypes.c_uint32),
        ("clockHours", ctypes.c_float), ("clockMinutes", ctypes.c_float),
        ("playerX", ctypes.c_float), ("playerY", ctypes.c_float), ("playerZ", ctypes.c_float),
        ("heliExists", ctypes.c_int32), ("heliModelId", ctypes.c_int32),
        ("heliX", ctypes.c_float), ("heliY", ctypes.c_float), ("heliZ", ctypes.c_float),
        ("heliHeading", ctypes.c_float),
        ("heliVX", ctypes.c_float), ("heliVY", ctypes.c_float), ("heliVZ", ctypes.c_float),
        ("heliSpeed", ctypes.c_float), ("heliHealth", ctypes.c_float),
        ("rays", ctypes.c_float * 13),
        ("numBlips", ctypes.c_uint32),
        ("blips", MlBlip * 32),
        ("lastActionSeq", ctypes.c_uint32), ("lastCommandSeq", ctypes.c_uint32),
        ("commandResult", ctypes.c_int32),
        ("controllerPid", ctypes.c_uint32), ("injectActive", ctypes.c_int32),
        ("lastStepSeq", ctypes.c_uint32), ("stepFrames", ctypes.c_uint32),
        ("stepActive", ctypes.c_int32), ("lockstep", ctypes.c_int32),
        ("episodeId", ctypes.c_uint32), ("missionSuccessCount", ctypes.c_int32),
        ("simStepMs", ctypes.c_uint32), ("controlsMode", ctypes.c_int32),
        ("heliForward", ctypes.c_float * 3), ("heliUp", ctypes.c_float * 3),
        ("heliTurnVel", ctypes.c_float * 3), ("heartbeatAgeMs", ctypes.c_uint32),
        ("stepStartFrame", ctypes.c_uint32), ("stepStartGameMs", ctypes.c_uint32),
        ("heliRef", ctypes.c_int32), ("_pad", ctypes.c_uint8 * 4),
        ("missionTelemetryValid", ctypes.c_int32), ("missionId", ctypes.c_int32),
        ("missionEpoch", ctypes.c_uint32), ("bombState", ctypes.c_int32 * 4),
        ("targetCompleteMask", ctypes.c_uint32), ("timerState", ctypes.c_int32),
        ("missionTimerMs", ctypes.c_int32), ("objectives", MlObjective * 8),
        ("rayHits", MlHit * 13), ("sensorVersion", ctypes.c_uint32),
        ("clearanceRadius", ctypes.c_float), ("bodyBoundRadius", ctypes.c_float),
        ("bodyBoundCenter", ctypes.c_float * 3), ("query", MlQueryResult),
    ]


class MlAction(ctypes.Structure):
    _fields_ = [
        ("magic", ctypes.c_uint32), ("version", ctypes.c_uint32),
        ("seq", ctypes.c_uint32), ("commandSeq", ctypes.c_uint32),
        ("controllerPid", ctypes.c_uint32),
        ("active", ctypes.c_int32),
        ("LeftStickX", ctypes.c_int16), ("LeftStickY", ctypes.c_int16),
        ("RightStickX", ctypes.c_int16), ("RightStickY", ctypes.c_int16),
        ("LeftShoulder1", ctypes.c_int16), ("LeftShoulder2", ctypes.c_int16),
        ("RightShoulder1", ctypes.c_int16), ("RightShoulder2", ctypes.c_int16),
        ("DPadUp", ctypes.c_int16), ("DPadDown", ctypes.c_int16),
        ("DPadLeft", ctypes.c_int16), ("DPadRight", ctypes.c_int16),
        ("Start", ctypes.c_int16), ("Select", ctypes.c_int16),
        ("Square", ctypes.c_int16), ("Triangle", ctypes.c_int16),
        ("Cross", ctypes.c_int16), ("Circle", ctypes.c_int16),
        ("LeftShock", ctypes.c_int16), ("RightShock", ctypes.c_int16),
        ("command", ctypes.c_uint32), ("commandArg", ctypes.c_int32),
        ("actionSeq", ctypes.c_uint32), ("stepSeq", ctypes.c_uint32),
        ("stepFrames", ctypes.c_uint32), ("lockstep", ctypes.c_int32),
        ("heartbeatSeq", ctypes.c_uint32), ("_pad", ctypes.c_uint8 * 44),
        ("querySeq", ctypes.c_uint32), ("queryStart", ctypes.c_float * 3),
        ("queryEnd", ctypes.c_float * 3), ("queryRadius", ctypes.c_float),
    ]


class MlShared(ctypes.Structure):
    _fields_ = [("state", MlState), ("action", MlAction)]


def _kernel32():
    k32 = ctypes.windll.kernel32
    k32.OpenFileMappingA.restype = ctypes.c_void_p
    k32.OpenFileMappingA.argtypes = [ctypes.c_uint32, wt.BOOL, ctypes.c_char_p]
    k32.MapViewOfFile.restype = ctypes.c_void_p
    k32.MapViewOfFile.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                  ctypes.c_uint32, ctypes.c_uint32, ctypes.c_size_t]
    k32.UnmapViewOfFile.argtypes = [ctypes.c_void_p]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_uint32, wt.BOOL, ctypes.c_uint32]
    k32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
    k32.CreateMutexA.argtypes = [ctypes.c_void_p, wt.BOOL, ctypes.c_char_p]
    k32.CreateMutexA.restype = ctypes.c_void_p
    k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k32.WaitForSingleObject.restype = ctypes.c_uint32
    k32.ReleaseMutex.argtypes = [ctypes.c_void_p]
    k32.ReleaseMutex.restype = wt.BOOL
    return k32


def pid_alive(pid):
    k32 = _kernel32()
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return False
    code = ctypes.c_uint32(0)
    k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return code.value == STILL_ACTIVE


def _pids():
    """All reVC.exe process ids via ToolHelp32 (no external deps)."""
    k32 = ctypes.windll.kernel32
    k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    k32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]

    class PROCENTRY32(ctypes.Structure):
        _fields_ = [("dwSize", ctypes.c_uint32), ("cntUsage", ctypes.c_uint32),
                    ("th32ProcessID", ctypes.c_uint32),
                    ("th32DefaultHeapID", ctypes.c_void_p),
                    ("th32ModuleID", ctypes.c_uint32), ("cntThreads", ctypes.c_uint32),
                    ("th32ParentProcessID", ctypes.c_uint32), ("pcPriClassBase", ctypes.c_int32),
                    ("dwFlags", ctypes.c_uint32), ("szExeFile", ctypes.c_char * 260)]
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if not snap or snap == ctypes.c_void_p(-1).value:
        return []
    k32.Process32First.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCENTRY32)]
    k32.Process32Next.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCENTRY32)]
    e = PROCENTRY32(); e.dwSize = ctypes.sizeof(PROCENTRY32)
    out = []
    if k32.Process32First(snap, ctypes.byref(e)):
        while True:
            if e.szExeFile.decode(errors="ignore").lower() == "revc.exe":
                out.append(e.th32ProcessID)
            if not k32.Process32Next(snap, ctypes.byref(e)):
                break
    k32.CloseHandle(snap)
    return out


def discover():
    """Return pids of running games that expose the ML bridge."""
    k32 = _kernel32()
    live = []
    for pid in _pids():
        h = k32.OpenFileMappingA(FILE_MAP_ALL_ACCESS, False,
                                 ("ReVC_ML_Bridge_PID%d" % pid).encode())
        if h:
            k32.CloseHandle(h)
            live.append(pid)
    return live


assert ctypes.sizeof(MlBlip) == 36 and MlBlip.blipId.offset == 24
assert MlBlip.objectiveKind.offset == 28 and MlBlip.objectiveIndex.offset == 32
assert ctypes.sizeof(MlObjective) == 24
assert ctypes.sizeof(MlHit) == 48 and ctypes.sizeof(MlQueryResult) == 124
assert MlState.rayHits.offset == 1652 and MlState.query.offset == 2300
assert MlAction.querySeq.offset == 136
assert ctypes.sizeof(MlState) == 2424
assert MlState.lastActionSeq.offset == 1312
assert MlState.missionTelemetryValid.offset == 1420 and MlState.objectives.offset == 1460
assert ctypes.sizeof(MlAction) == 168
assert ctypes.sizeof(MlShared) == 2592 and MlShared.action.offset == 2424
assert MlState.blips.offset == 160 and MlState.lastActionSeq.offset == 1312
assert MlShared.action.offset == 2424
assert MlState.seq.offset == 16 and MlAction.seq.offset == 8


def _next_seq(seq):
    return (int(seq) + 1) % 0x100000000 or 1


def _axis(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("controller value must be finite")
    return round(max(-1.0, min(1.0, value)) * 128)


def _pressure(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("controller value must be finite")
    return round(max(0.0, min(1.0, value)) * 255)


class ReVCInstance:
    """Protocol-v7 controller; all mutation methods belong to one caller thread.

    Named mutexes provide interprocess ordering for complete snapshots. A
    third mutex serializes controller ownership; read-only observers never
    acquire it and close() never releases somebody else's controls.
    """

    def __init__(self, pid=None, timeout=10.0):
        self.k32 = _kernel32()
        self.pid = pid
        self._map = self._view = None
        self.shared = None
        self._state_mutex = self._action_mutex = self._controller_mutex = None
        self._owns_controller = False
        self._lock = threading.RLock()
        self._refresh_thread = None
        self._refresh_stop = threading.Event()
        self._action_seq = self._command_seq = self._step_seq = self._query_seq = 0
        self._pending = MlAction()
        self._last_step_state = None
        if pid is None:
            found = discover()
            if not found:
                raise RuntimeError("no bridge-enabled reVC running")
            self.pid = pid = found[0]
        deadline = time.perf_counter() + timeout
        name = ("ReVC_ML_Bridge_PID%d" % pid).encode()
        try:
            while True:
                self._map = self.k32.OpenFileMappingA(FILE_MAP_ALL_ACCESS, False, name)
                if self._map:
                    break
                self._require_alive()
                if time.perf_counter() > deadline:
                    raise TimeoutError("bridge not found for pid %d" % pid)
                time.sleep(0.05)
            # Read only the common header before accepting the larger v6 layout.
            self._view = self.k32.MapViewOfFile(self._map, FILE_MAP_ALL_ACCESS, 0, 0, 0)
            if not self._view:
                raise RuntimeError("MapViewOfFile failed")
            header = (ctypes.c_uint32 * 4).from_address(self._view)
            if header[0] != ML_MAGIC or header[1] != ML_VERSION or header[2] != ctypes.sizeof(MlState):
                raise RuntimeError("bridge layout/version mismatch (game v%d, client v%d); deploy matching build" % (header[1], ML_VERSION))
            self.shared = ctypes.cast(self._view, ctypes.POINTER(MlShared)).contents
            for attr, tag in (("_state_mutex", "State"), ("_action_mutex", "Action"), ("_controller_mutex", "Controller")):
                handle = self.k32.CreateMutexA(None, False, ("ReVC_ML_%s_PID%d" % (tag, pid)).encode())
                if not handle:
                    raise RuntimeError("cannot open bridge mutex")
                setattr(self, attr, handle)
            st = self.get_state()
            self._action_seq = st["lastActionSeq"]
            self._command_seq = st["lastCommandSeq"]
            self._step_seq = st["lastStepSeq"]
            self._query_seq = st["query"]["seq"]
            self._pending.magic = ML_MAGIC
            self._pending.version = ML_VERSION
            self._pending.controllerPid = os.getpid()
            self._pending.commandSeq = self._command_seq
            self._pending.stepSeq = self._step_seq
            self._pending.querySeq = self._query_seq
        except BaseException:
            self._close_handles()
            raise

    def alive(self):
        return pid_alive(self.pid)

    def _require_alive(self):
        if not self.alive():
            raise GameDeadError("reVC pid %d is gone" % self.pid)

    @contextmanager
    def _mutex(self, handle, timeout_ms=1000):
        result = self.k32.WaitForSingleObject(handle, timeout_ms)
        if result not in (0, 0x80):  # WAIT_OBJECT_0 / WAIT_ABANDONED
            self._require_alive()
            raise TimeoutError("bridge mutex timed out")
        try:
            yield
        finally:
            self.k32.ReleaseMutex(handle)

    def get_state(self, retries=100, stable_gap=0.002):
        """Return a complete frame, including a stationary lockstep snapshot."""
        self._require_alive()
        for _ in range(retries):
            with self._mutex(self._state_mutex):
                raw = bytes(self.shared.state)
            st = MlState.from_buffer_copy(raw)
            if st.seq and not st.seq & 1:
                if st.numBlips > 32 or st.magic != ML_MAGIC or st.version != ML_VERSION:
                    raise RuntimeError("invalid state layout")
                return self._decode_state(st)
            time.sleep(stable_gap)
        self._require_alive()
        raise TimeoutError("no complete bridge state published")

    @staticmethod
    def _decode_hit(hit):
        if hit.valid not in (0, 1) or hit.hit not in (0, 1) or (hit.hit and not hit.valid):
            raise RuntimeError("invalid collision hit flags")
        if hit.valid and (not math.isfinite(hit.distance) or hit.distance < 0 or
                          not all(math.isfinite(v) for v in (*hit.point, *hit.normal))):
            raise RuntimeError("invalid collision hit geometry")
        return dict(valid=bool(hit.valid), hit=bool(hit.hit), entity_type=hit.entityType,
                    model_id=hit.modelId, surface=hit.surface, distance=hit.distance,
                    point=tuple(hit.point), normal=tuple(hit.normal))

    @staticmethod
    def _decode_query(query):
        if query.status not in (0, 1, -1, -2, -3, -4):
            raise RuntimeError("invalid path-query status")
        line, sweep = ReVCInstance._decode_hit(query.line), ReVCInstance._decode_hit(query.sweep)
        valid = query.status == 1
        if valid and (not line['valid'] or not sweep['valid'] or
                      not all(math.isfinite(v) for v in (query.radius, query.sampleSpacing, query.length)) or
                      query.radius <= 0 or not 0 <= query.sampleSpacing <= .25001 or
                      not 0 <= query.length <= 200.0001):
            raise RuntimeError("invalid path-query geometry")
        return dict(seq=query.seq, status=query.status, valid=valid,
                    blocked=not valid or line['hit'] or sweep['hit'],
                    frame=query.frame, episode_id=query.episodeId, radius=query.radius,
                    sample_spacing=query.sampleSpacing, length=query.length, line=line, sweep=sweep)

    @staticmethod
    def _decode_state(st):
        valid = st.missionTelemetryValid == 1
        if st.missionTelemetryValid not in (0, 1):
            raise RuntimeError("invalid mission telemetry flag")
        if valid and (st.missionId != 19 or not st.missionEpoch or
                      any(value not in (0, 1, 2, 3) for value in st.bombState) or
                      sum(value == 1 for value in st.bombState) > 1 or
                      st.targetCompleteMask > 15 or st.timerState not in (0, 1, 2) or
                      not 0 <= st.missionTimerMs <= 420000 or
                      (st.timerState == 0 and st.missionTimerMs != 0)):
            raise RuntimeError("invalid mission telemetry values")
        # Fail closed even if a malformed invalid payload contains stale globals.
        mission = dict(valid=valid, id=19 if valid else -1, epoch=st.missionEpoch,
                       bomb_states=list(st.bombState) if valid else [-1] * 4,
                       completed_mask=st.targetCompleteMask if valid else 0,
                       timer_state=st.timerState if valid else -1,
                       remaining_ms=st.missionTimerMs if valid else -1)
        objectives = []
        for i, objective in enumerate(st.objectives):
            kind, index = 1 + i // 4, i % 4
            pos = (objective.x, objective.y, objective.z)
            if valid and (objective.kind != kind or objective.index != index or
                          objective.valid not in (0, 1) or
                          (objective.valid and not all(math.isfinite(v) for v in pos))):
                raise RuntimeError("invalid mission objective")
            available = valid and bool(objective.valid)
            objectives.append(dict(kind=kind, index=index, valid=available,
                                   pos=pos if available else (0.0, 0.0, 0.0)))
        blips = []
        for blip in st.blips[:st.numBlips]:
            kind, index = (blip.objectiveKind, blip.objectiveIndex) if valid else (0, -1)
            if (kind == 0 and index != -1) or (kind in (1, 2) and index not in range(4)) or kind not in (0, 1, 2):
                raise RuntimeError("invalid blip objective identity")
            blips.append(dict(blipId=blip.blipId, pos=(blip.x, blip.y, blip.z),
                              color=blip.color, blipType=blip.blipType, sprite=blip.sprite,
                              objectiveKind=kind, objectiveIndex=index))
        return {
            "pid": st.pid, "seq": st.seq, "frame": st.frame,
            "missionRunning": st.missionRunning, "cutsceneProcessing": st.cutsceneProcessing,
            "playerState": st.playerState, "gameTimerMs": st.gameTimerMs,
            "clock": (st.clockHours, st.clockMinutes),
            "player": (st.playerX, st.playerY, st.playerZ),
            "heli": None if not st.heliExists else {
                "pos": (st.heliX, st.heliY, st.heliZ), "heading": st.heliHeading,
                "vel": (st.heliVX, st.heliVY, st.heliVZ), "speed": st.heliSpeed,
                "modelId": st.heliModelId, "ref": st.heliRef, "health": st.heliHealth, "rays": list(st.rays),
                "forward": tuple(st.heliForward), "up": tuple(st.heliUp),
                "turnVel": tuple(st.heliTurnVel),
                "ray_hits": [ReVCInstance._decode_hit(hit) for hit in st.rayHits],
                "sensor_schema": SENSOR_SCHEMA if st.sensorVersion == 2 else "unknown",
                "clearance_radius": st.clearanceRadius, "body_bound_radius": st.bodyBoundRadius,
                "body_bound_center": tuple(st.bodyBoundCenter),
            },
            "numBlips": st.numBlips,
            "blips": blips,
            "mission": mission, "objectives": objectives,
            "query": ReVCInstance._decode_query(st.query),
            "lastActionSeq": st.lastActionSeq, "lastCommandSeq": st.lastCommandSeq,
            "lastCommandResult": st.commandResult, "controllerPid": st.controllerPid,
            "injectActive": st.injectActive, "lastStepSeq": st.lastStepSeq,
            "stepFrames": st.stepFrames, "stepActive": bool(st.stepActive),
            "lockstep": bool(st.lockstep), "episodeId": st.episodeId,
            "missionSuccessCount": st.missionSuccessCount, "simStepMs": st.simStepMs,
            "controlsMode": st.controlsMode, "heartbeatAgeMs": st.heartbeatAgeMs,
            "stepStartFrame": st.stepStartFrame, "stepStartGameMs": st.stepStartGameMs,
        }

    def query_path(self, start, end, radius=None, timeout=3.0):
        """Read current static collision geometry while an owned step is frozen.

        This does not move/load/teleport entities or advance simulation. Buildings,
        objects and dummies participate; people/vehicles are not covered. The
        sampled conservative sphere includes rotor and carried-bomb bounds.
        Invalid or unavailable geometry raises; blocked is a normal valid result.
        """
        def point(value):
            try:
                values = tuple(float(x) for x in value)
            except (TypeError, ValueError) as exc:
                raise ValueError('query point must contain three finite coordinates') from exc
            if (len(values) != 3 or not all(math.isfinite(x) for x in values) or
                    abs(values[0]) > 1900 or abs(values[1]) > 1900 or not -100 <= values[2] <= 200):
                raise ValueError('query point outside supported world bounds')
            return values
        start, end = point(start), point(end)
        if math.dist(start, end) > 200:
            raise ValueError('query segment exceeds 200 metres')
        value = 0. if radius is None else float(radius)
        if not math.isfinite(value) or (radius is not None and not 0 < value <= 5):
            raise ValueError('query radius must be finite, positive and at most 5 metres')
        self._claim(timeout)
        before = self.get_state()
        if (not before['lockstep'] or before['stepActive'] or
                before['controllerPid'] != os.getpid() or not self._pending.active or not self._pending.lockstep):
            raise RuntimeError('query_path requires this controller to own a frozen lockstep')
        minimum = (before.get('heli') or {}).get('clearance_radius', 0.)
        if not math.isfinite(minimum) or minimum <= 0:
            raise RuntimeError('query_path needs verified RC collision bounds')
        if radius is not None and value < minimum - .00001:
            raise ValueError('query radius is smaller than the current helicopter clearance radius')
        with self._lock:
            self._query_seq = _next_seq(max(getattr(self, '_query_seq', 0), before['query']['seq']))
            seq = self._query_seq
            a = self._pending
            a.queryStart[:], a.queryEnd[:], a.queryRadius = start, end, value
            a.querySeq = seq
            a.command = 0
            # Preserve actionSeq/stepSeq and all controls. Native handles this
            # separate request before returning to the same paused barrier.
            self._publish()
        self._start_refresher()
        after = self._wait(lambda st: st['query']['seq'] == seq, timeout, 'path query was not acknowledged')
        if any(after[key] != before[key] for key in ('frame', 'gameTimerMs', 'episodeId', 'pid')) or not after['lockstep'] or after['stepActive']:
            raise RuntimeError('simulation changed during read-only path query')
        result = after['query']
        if result['frame'] != before['frame'] or result['episode_id'] != before['episodeId']:
            raise RuntimeError('stale path-query result')
        if not result['valid']:
            if result['status'] == -3:
                raise GeometryUnavailableError(result)
            raise RuntimeError('native path query rejected, status=%d' % result['status'])
        return result

    def wait_frames(self, n=5, timeout=30.0):
        """For free-running diagnostics only; timeout is never a valid step."""
        st = self.get_state()
        if st["lockstep"] and not st["stepActive"]:
            raise RuntimeError("wait_frames cannot advance paused lockstep; use step()")
        f0 = st["frame"]
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if (self.get_state()["frame"] - f0) % 0x100000000 >= n:
                return True
            time.sleep(0.002)
        self._require_alive()
        raise TimeoutError("game did not advance %d frames" % n)

    def _claim(self, timeout=3.0):
        self._require_alive()
        if self._owns_controller:
            return
        with _LOCAL_CONTROLLERS_LOCK:
            existing = _LOCAL_CONTROLLERS.get(self.pid)
            if existing is not None and existing is not self:
                raise ControllerBusyError("another client object in this process controls the bridge")
            result = self.k32.WaitForSingleObject(self._controller_mutex, max(0, round(timeout * 1000)))
            if result not in (0, 0x80):
                raise ControllerBusyError("another controller holds the bridge")
            self._owns_controller = True
            _LOCAL_CONTROLLERS[self.pid] = self
        try:
            st = self.get_state()
            if st["controllerPid"] not in (0, os.getpid()):
                raise ControllerBusyError("bridge owned by pid %d" % st["controllerPid"])
            # Continue from the latest acknowledged AND published counters.
            # This also avoids reusing a pending previous controller's ID.
            with self._mutex(self._action_mutex):
                prev = MlAction.from_buffer_copy(bytes(self.shared.action))
            self._action_seq = max(self._action_seq, st["lastActionSeq"], prev.actionSeq)
            self._command_seq = max(self._command_seq, st["lastCommandSeq"], prev.commandSeq)
            self._step_seq = max(self._step_seq, st["lastStepSeq"], prev.stepSeq)
            self._query_seq = max(getattr(self, '_query_seq', 0), st['query']['seq'], prev.querySeq)
            self._pending.querySeq = self._query_seq
            self._pending.commandSeq = self._command_seq
        except BaseException:
            self._drop_claim()
            raise

    def _drop_claim(self):
        if self._owns_controller:
            self.k32.ReleaseMutex(self._controller_mutex)
            self._owns_controller = False
            with _LOCAL_CONTROLLERS_LOCK:
                if _LOCAL_CONTROLLERS.get(self.pid) is self:
                    del _LOCAL_CONTROLLERS[self.pid]

    def _publish(self):
        # Windows mutex release/acquire is the memory barrier on the Python
        # side (kernel32 does not export Interlocked* on all 64-bit builds).
        with self._lock, self._mutex(self._action_mutex):
            a = self._pending
            a.heartbeatSeq = _next_seq(a.heartbeatSeq)
            seq = ((self.shared.action.seq & ~1) + 2) % 0x100000000 or 2
            a.seq = seq
            self.shared.action.seq = seq - 1
            base = ctypes.addressof(self.shared.action)
            src = ctypes.addressof(a)
            ctypes.memmove(base, src, MlAction.seq.offset)
            offset = MlAction.commandSeq.offset
            ctypes.memmove(base + offset, src + offset, ctypes.sizeof(a) - offset)
            self.shared.action.seq = seq
        return a.actionSeq

    def _start_refresher(self):
        if self._refresh_thread is not None:
            return
        self._refresh_stop.clear()
        def refresh():
            while not self._refresh_stop.wait(0.7):
                try:
                    self._publish()
                except Exception:
                    return
        self._refresh_thread = threading.Thread(target=refresh, daemon=True)
        self._refresh_thread.start()

    def _stop_refresher(self):
        thread = self._refresh_thread
        if thread is not None:
            self._refresh_stop.set()
            thread.join(2.0)
            if thread.is_alive():
                raise RuntimeError("bridge refresher did not stop; mapping kept alive")
            self._refresh_thread = None

    def _controls(self, lx=0.0, ly=0.0, rx=0.0, ry=0.0, dpad=(0, 0, 0, 0),
                  buttons=(), shoulders=(0, 0, 0, 0), cross_val=None, brake_val=None,
                  lockstep=False, frames=0, timeout=3.0):
        if not isinstance(frames, int) or not 0 <= frames <= 1000:
            raise ValueError("frames must be an integer in [0, 1000]")
        # Validate everything before publishing any part of an action.
        sticks = tuple(_axis(v) for v in (lx, ly, rx, ry))
        analog_cross = None if cross_val is None else _pressure(cross_val)
        analog_brake = None if brake_val is None else _pressure(brake_val)
        if len(dpad) != 4 or len(shoulders) != 4:
            raise ValueError("dpad and shoulders need four values")
        self._claim(timeout)
        with self._lock:
            a = self._pending
            a.active = 1
            a.LeftStickX, a.LeftStickY, a.RightStickX, a.RightStickY = sticks
            a.DPadUp, a.DPadDown, a.DPadLeft, a.DPadRight = (255 if x else 0 for x in dpad)
            for name in ("Square", "Triangle", "Cross", "Circle", "Start", "Select", "LeftShock", "RightShock"):
                setattr(a, name, 255 if name in buttons else 0)
            if analog_cross is not None:
                a.Cross = analog_cross
            if analog_brake is not None:
                a.Square = analog_brake
            a.LeftShoulder1, a.LeftShoulder2, a.RightShoulder1, a.RightShoulder2 = (255 if x else 0 for x in shoulders)
            self._action_seq = _next_seq(self._action_seq)
            a.actionSeq = self._action_seq
            a.command = 0
            a.lockstep = int(lockstep)
            a.stepFrames = frames
            if lockstep:
                self._step_seq = _next_seq(self._step_seq)
                a.stepSeq = self._step_seq
            self._publish()
        self._start_refresher()
        return self._action_seq, self._step_seq

    def _wait(self, predicate, timeout, description):
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            st = self.get_state()
            if st["controllerPid"] not in (0, os.getpid()):
                raise ControllerBusyError("bridge owned by pid %d" % st["controllerPid"])
            if predicate(st):
                return st
            time.sleep(0.002)
        self._require_alive()
        raise TimeoutError(description)

    def hold(self, timeout=3.0, **controls):
        """Free-running diagnostic input: native sticks ±128, buttons 0..255."""
        self._last_step_state = None
        self._needs_barrier = False
        seq, _ = self._controls(timeout=timeout, **controls)
        self._wait(lambda st: st["lastActionSeq"] == seq and st["injectActive"] and
                   st["controllerPid"] == os.getpid() and not st["lockstep"],
                   timeout, "action was not acknowledged")
        return True

    def step(self, frames=10, timeout=30.0, **controls):
        """Run exactly N fixed 10ms frames and return their final frozen state."""
        if getattr(self, "_needs_barrier", False):
            raise RuntimeError("previous step failed; reset or pause before another step")
        try:
            return self._step(frames=frames, timeout=timeout, **controls)
        except (RuntimeError, TimeoutError):
            self._last_step_state = None
            self._needs_barrier = True
            raise

    def _step(self, frames=10, timeout=30.0, **controls):
        previous = getattr(self, "_last_step_state", None)
        seq, step_seq = self._controls(lockstep=True, frames=frames, timeout=timeout, **controls)
        st = self._wait(lambda st: st["lastActionSeq"] == seq and
                        st["lastStepSeq"] == step_seq and not st["stepActive"] and
                        st["lockstep"] and st["controllerPid"] == os.getpid(),
                        timeout, "lockstep action did not complete")
        if st["stepFrames"] != frames or (st["frame"] - st["stepStartFrame"]) % 0x100000000 != frames:
            raise RuntimeError("lockstep frame count mismatch")
        if (st["gameTimerMs"] - st["stepStartGameMs"]) % 0x100000000 != frames * st["simStepMs"]:
            raise RuntimeError("lockstep game time mismatch")
        if previous is not None:
            if (st["stepStartFrame"] != previous["frame"] or
                    st["stepStartGameMs"] != previous["gameTimerMs"] or
                    st.get("episodeId") != previous.get("episodeId") or
                    st.get("pid") != previous.get("pid")):
                self._last_step_state = None
                raise RuntimeError("simulation advanced between lockstep actions; discard transition and reset")
        self._last_step_state = st
        return st

    def pause(self, timeout=5.0):
        """Zero-frame barrier used after reset to establish a frozen initial state."""
        self._last_step_state = None
        self._needs_barrier = False
        return self.step(frames=0, timeout=timeout)

    def release(self, timeout=3.0):
        """Release only this instance's control and return to real-time simulation."""
        self._last_step_state = None
        self._needs_barrier = False
        self._stop_refresher()
        if not self._owns_controller or self._view is None:
            return
        try:
            self._require_alive()
            with self._lock:
                self._pending.active = 0
                self._pending.lockstep = 0
                self._pending.command = 0
                self._publish()
            self._wait(lambda st: st["controllerPid"] == 0 and not st["lockstep"],
                       timeout, "release was not acknowledged")
        finally:
            self._drop_claim()

    def send_mission(self, mission, wait_ack=2.0, force=False):
        """Switch mission 19 with a unique command ID and exact acknowledgement.

        `force=True` keeps the legacy command ID but never overrides an active
        mission: the native bridge rejects both IDs until cleanup is complete.
        Returns 1 on acceptance or -1 on rejection. Missing ack raises; old
        acknowledgements are never treated as acceptance of a new command.
        """
        self._last_step_state = None
        self._needs_barrier = False
        if mission != 19:
            raise ValueError("this bridge task supports mission 19 only")
        self._claim(max(wait_ack, 0.1))
        self._stop_refresher()
        try:
            with self._lock:
                self._command_seq = _next_seq(self._command_seq)
                a = self._pending
                a.active = 0
                a.lockstep = 0
                a.command = 2 if force else 1
                a.commandArg = mission
                a.commandSeq = self._command_seq
                self._publish()
            st = self._wait(lambda st: st["lastCommandSeq"] == self._command_seq,
                            wait_ack, "mission command was not acknowledged")
            return st["lastCommandResult"]
        finally:
            self._drop_claim()

    def _close_handles(self):
        if self._view:
            self.k32.UnmapViewOfFile(self._view)
            self._view = None
        for attr in ("_map", "_state_mutex", "_action_mutex", "_controller_mutex"):
            handle = getattr(self, attr, None)
            if handle:
                self.k32.CloseHandle(handle)
                setattr(self, attr, None)
        self.shared = None

    def close(self):
        try:
            self.release()
        finally:
            # release() joins the heartbeat thread before any memory is unmapped.
            if self._refresh_thread is None:
                self._close_handles()
