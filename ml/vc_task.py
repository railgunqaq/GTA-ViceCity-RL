"""Demolition Man (无人机炸楼) RL task pipeline — end-to-end.

Ties together everything built so far:

    launch game (headless, no-activate)
      -> ~12s auto: intro skipped, mission 19 started, cutscenes skipped,
         player warped into the Top Fun van, RC-helicopter phase active
      -> connect ML bridge (per-PID shared memory, multi-open safe)
      -> observe (heli pos/vel/heading, mission blips, clock, flags)
      -> act (CPad injection: sticks + buttons, hold semantics)
      -> reset (in-game mission switch, or hard process restart)

Usage:
    python vc_task.py                # run the built-in demo pipeline
    from vc_task import VcDemolitionTask
    task = VcDemolitionTask()        # attaches/launches automatically
    obs = task.observe()
    obs = task.step(lx=0.0, ly=-1.0, throttle=True)   # fly forward 1s
    task.reset()                     # mission restart (soft) / hard_reset()

Control mapping (empirically verified on this build):
    throttle=True  -> Cross (vehicle accelerate); brake=True -> Square
    lx/ly          -> left stick steer (DirectInput convention: ly<0 = forward)
Multi-open: pass game_dir (own copy per instance) and/or pid to attach to a
specific one; each instance's bridge is named by its PID.
"""

import ctypes
import os
import subprocess
import sys
import time

from ml_client import ReVCInstance, discover, GameDeadError, ControllerBusyError

# Set before starting Python; every entry point shares this default.
GAME_DIR = os.environ.get("REVC_GAME_DIR", r"C:\VC")
SW_SHOWNOACTIVATE = 4


def launch_game(game_dir=GAME_DIR):
    """Launch reVC without stealing focus. Returns the new pid."""
    exe = os.path.join(game_dir, "reVC.exe")

    class STARTUPINFO(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("lpReserved", ctypes.c_wchar_p),
                    ("lpDesktop", ctypes.c_wchar_p), ("lpTitle", ctypes.c_wchar_p),
                    ("dwX", ctypes.c_uint32), ("dwY", ctypes.c_uint32),
                    ("dwXSize", ctypes.c_uint32), ("dwYSize", ctypes.c_uint32),
                    ("dwXCountChars", ctypes.c_uint32), ("dwYCountChars", ctypes.c_uint32),
                    ("dwFillAttribute", ctypes.c_uint32), ("dwFlags", ctypes.c_uint32),
                    ("wShowWindow", ctypes.c_uint16), ("cbReserved2", ctypes.c_uint16),
                    ("lpReserved2", ctypes.c_void_p), ("hStdInput", ctypes.c_void_p),
                    ("hStdOutput", ctypes.c_void_p), ("hStdError", ctypes.c_void_p)]

    class PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = [("hProcess", ctypes.c_void_p), ("hThread", ctypes.c_void_p),
                    ("dwProcessId", ctypes.c_uint32), ("dwThreadId", ctypes.c_uint32)]

    si = STARTUPINFO()
    si.cb = ctypes.sizeof(STARTUPINFO)
    si.dwFlags = 0x01  # STARTF_USESHOWWINDOW
    si.wShowWindow = SW_SHOWNOACTIVATE
    pi = PROCESS_INFORMATION()
    ok = ctypes.windll.kernel32.CreateProcessW(
        exe, None, None, None, False, 0, None, game_dir,
        ctypes.byref(si), ctypes.byref(pi))
    if not ok:
        raise RuntimeError("CreateProcess failed: %d" %
                           ctypes.windll.kernel32.GetLastError())
    ctypes.windll.kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    ctypes.windll.kernel32.CloseHandle(pi.hThread)
    ctypes.windll.kernel32.CloseHandle(pi.hProcess)
    return pi.dwProcessId


class VcDemolitionTask:
    """RL-style wrapper around one reVC instance running mission 19."""

    def __init__(self, pid=None, game_dir=GAME_DIR, auto=True, verbose=True):
        self.game_dir = game_dir
        self.verbose = verbose
        self.game = None
        self.recovered = 0
        self.owned_pids = set()
        if auto:
            self.ensure_game(pid)

    def log(self, msg):
        if self.verbose:
            print("[%s] %s" % (time.strftime("%H:%M:%S"), msg))

    # ---------------- lifecycle ----------------

    def _attach(self, pid, startup_timeout=25.0):
        """Connect to a pid, tolerating the bridge appearing a bit late."""
        deadline = time.perf_counter() + startup_timeout
        while True:
            try:
                return ReVCInstance(pid, timeout=2.0)
            except (RuntimeError, GameDeadError, TimeoutError) as e:
                if time.perf_counter() > deadline or not _pid_alive_quiet(pid):
                    raise
                time.sleep(0.3)

    def _bridge_responds(self, pid):
        """True if the pid's bridge answers a state read — filters zombie
        pids that still appear in the process list while terminating."""
        inst = None
        try:
            inst = ReVCInstance(pid, timeout=1.0)
            st = inst.get_state()
            return st["controllerPid"] in (0, os.getpid())
        except Exception:
            return False
        finally:
            if inst is not None:
                try:
                    inst.close()
                except Exception:
                    pass

    def ensure_game(self, pid=None, attempts=4):
        """Attach to a running instance (or launch one) and wait until the
        RC-helicopter phase of mission 19 is active. Retries launch on
        startup failure (rare early-boot flake)."""
        explicit_pid = pid is not None
        last_err = None
        for attempt in range(1, attempts + 1):
            try:
                if pid is None:
                    found = [p for p in discover() if self._bridge_responds(p)]
                    pid = found[0] if found else self._launch()
                    if not found:
                        self.log("launched game pid=%d (attempt %d), "
                                 "waiting for RC phase..." % (pid, attempt))
                self.game = self._attach(pid)
                st = self.game.get_state()
                if st["controllerPid"] not in (0, os.getpid()):
                    raise ControllerBusyError("selected game is controlled by another process")
                if not self._in_rc_phase(st):
                    if st["missionRunning"] == 0:
                        self.log("mission not running — relaunching mission 19")
                        self.game.send_mission(19)
                    self.wait_until(self._in_rc_phase, timeout=90.0,
                                    what="RC phase ready")
                st = self.game.get_state()
                self.log("attached pid=%d frame=%d heli=%s" %
                         (pid, st["frame"], tuple(round(v, 1) for v in st["heli"]["pos"])))
                return pid
            except ControllerBusyError:
                if self.game is not None:
                    self.game.close()
                    self.game = None
                raise
            except (TimeoutError, RuntimeError, GameDeadError) as e:
                if explicit_pid:
                    # Never replace a user-selected PID with an unrelated game.
                    if self.game is not None:
                        self.game.close()
                        self.game = None
                    raise
                last_err = e
                self.log("attempt %d failed: %s" % (attempt, e))
                # clean up a half-attached/dead game and retry fresh
                if self.game is not None:
                    try:
                        self.game.close()
                    except Exception:
                        pass
                    self.game = None
                if pid not in self.owned_pids:
                    raise
                subprocess.run(["taskkill", "/PID", str(pid), "/F"],
                               capture_output=True)
                pid = None
                # This machine's iGPU/D3D needs the old process object fully
                # gone plus a settle gap, or the next launch exits silently
                # at engine init (no crash log — clean early quit)
                _wait_revc_gone(15.0)
                time.sleep(6.0)
        raise RuntimeError("ensure_game failed after %d attempts: %s" %
                           (attempts, last_err))

    def _launch(self):
        pid = launch_game(self.game_dir)
        self.owned_pids.add(pid)
        return pid

    @staticmethod
    def _in_rc_phase(st):
        return (st["missionRunning"] == 1 and st["playerState"] == 2
                and st["heli"] is not None)

    @staticmethod
    def _in_fresh_rc_phase(st):
        """RC phase with a freshly spawned heli (full health). Distinguishes
        a real restart from the dying heli of the previous mission, whose
        state can transiently look like RC phase during fail cleanup."""
        mission = st.get("mission", {})
        return (VcDemolitionTask._in_rc_phase(st)
                and st["heli"]["health"] >= 1.3
                and mission.get("valid") is True and mission.get("id") == 19)

    def wait_until(self, cond, timeout=30.0, what="", poll=0.25):
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            try:
                st = self.game.get_state()
            except GameDeadError:
                raise
            except RuntimeError:
                st = None
            if st is not None and cond(st):
                return True
            time.sleep(poll)
        raise TimeoutError("timeout waiting for: %s" % what)

    def _advance_native_tutorial(self, timeout=30.0):
        """Acknowledge tutorial prompts with minimum native Cross pressure.

        The SCM checks nonzero pressure. This is ordinary input, before the
        reset barrier; native mission/HUD validation and fresh-state checks
        remain mandatory. Never return ready merely because time elapsed.
        """
        deadline = time.perf_counter() + timeout
        self.wait_until(self._in_rc_phase, timeout=max(0., deadline - time.perf_counter()),
                        what="RC phase before tutorial acknowledgement")
        def ready(st):
            mission = st.get("mission", {})
            return (self._in_rc_phase(st) and mission.get("valid") is True
                    and mission.get("id") == 19)
        if ready(self.game.get_state()):
            return True
        try:
            self.game.hold(cross_val=1. / 255., timeout=min(3., max(0., deadline - time.perf_counter())))
            self.wait_until(ready, timeout=max(0., deadline - time.perf_counter()),
                            what="native mission telemetry after tutorial", poll=0.05)
        finally:
            # Also neutralize an acknowledged hold when the wait fails.
            self.game.hold()
        return True

    # ---------------- observation ----------------

    def observe(self):
        """Snapshot with relative vectors heli->blip added for convenience."""
        if self.game is None:
            raise GameDeadError("no game attached")
        st = self.game.get_state()
        heli = st["heli"]
        if heli:
            hx, hy, hz = heli["pos"]
            for b in st["blips"]:
                bx, by, bz = b["pos"]
                b["rel"] = (bx - hx, by - hy, bz - hz)
                b["dist2d"] = (bx - hx) ** 2 + (by - hy) ** 2
        return st

    def targets(self):
        """Mission blips sorted by distance to the heli (nearest first)."""
        st = self.observe()
        if not st["heli"]:
            return []
        return sorted(st["blips"], key=lambda b: b["dist2d"])

    @staticmethod
    def is_done(st):
        """Episode over: mission no longer running, or the controlled heli
        is gone (destroyed / phase left), while no cutscene is mid-flight."""
        if st["cutsceneProcessing"]:
            return False          # transition in progress, not done
        return st["missionRunning"] == 0 or st["heli"] is None

    def crash_heli(self, max_seconds=10.0):
        """Force the running mission to end now: brake (DPadDown = descend)
        plus full forward pitch slams the RC heli into the ground in well
        under a second (measured 0.5s). Used when the trainer ends an episode
        at its step cap / stall terminator with the heli still alive —
        without this the uncontrolled heli drifts for ~20s before the
        in-game mission fails and a reset becomes possible."""
        t0 = time.perf_counter()
        k = 0
        while time.perf_counter() - t0 < max_seconds:
            st = self.step(lx=0.0, ly=(-1.0 if k % 2 == 0 else 1.0),
                           brake=True, frames=30)
            k += 1
            if self.is_done(st):
                return True
        return False

    # ---------------- action ----------------

    def step(self, lx=0.0, ly=0.0, rx=0.0, ry=0.0, throttle=False,
             brake=False, frames=10, buttons=(), auto_recover=False, cross=None,
             brake_val=None, lockstep=True):
        """Apply native controls for exactly `frames` 10ms simulation ticks.

        lx controls roll (negative roll), ly pitch, rx yaw. Cross/Square
        control positive/negative collective. A completed step stays frozen
        until the next step or release(). Game loss never returns a reset
        observation as an ordinary transition, even with auto_recover=True.
        """
        btns = set(buttons) | ({"Cross"} if throttle else set())
        if brake:
            btns.add("Square")
        controls = dict(lx=lx, ly=ly, rx=rx, ry=ry, buttons=btns,
                        cross_val=cross, brake_val=brake_val)
        try:
            if lockstep:
                obs = self.game.step(frames=frames, **controls)
            else:
                self.game.hold(**controls)
                if not self.game.wait_frames(frames):
                    raise TimeoutError("frame wait did not complete")
                obs = self.observe()
        except GameDeadError as exc:
            if auto_recover:
                self.hard_reset()
                self.recovered += 1
                raise GameDeadError("game restarted; discard transition and reset episode") from exc
            raise
        obs["recovered"] = False
        return obs

    def release(self):
        try:
            self.game.release()
        except GameDeadError:
            pass

    # ---------------- reset ----------------

    def reset(self, timeout=60.0):
        """Soft reset: when the mission is over, relaunch mission 19 in-game
        (cutscenes auto-skipped, player re-warped; ~10s). Returns False if
        the mission is still running (use reset(force=True) then)."""
        st = self.game.get_state()
        if st["missionRunning"]:
            return False
        if self.game.send_mission(19) != 1:
            return False
        self.wait_until(self._in_rc_phase, timeout=timeout, what="mission restart")
        self.game.pause()
        self.log("reset: back in RC phase")
        return True

    def reset_force(self, natural_wait=30.0, timeout=60.0):
        """Restart mission 19, verify its new entity/epoch, then freeze it."""
        try:
            self.release()
            before = self.game.get_state()
            if before["missionRunning"]:
                return self.hard_reset(reset_mode="hard_active_mission")
            old_ref = before["heli"]["ref"] if before["heli"] else None
            old_epoch = before["episodeId"]
            result = self.game.send_mission(19, wait_ack=2.0)
            if result != 1:
                return self.hard_reset(reset_mode="hard_busy_mission_cleanup")
            def fresh(st):
                return (self._in_fresh_rc_phase(st) and st["episodeId"] != old_epoch
                        and (old_ref is None or st["heli"]["ref"] != old_ref))
            self._advance_native_tutorial(timeout=timeout)
            self.wait_until_stable(fresh, stable=0.4, timeout=timeout,
                                   what="new mission epoch and RC vehicle")
            initial = self.game.pause()
            if not fresh(initial):
                raise RuntimeError("RC phase changed before reset barrier")
            self.last_reset_mode = "soft_idle"
            self.log("reset_mode=soft_idle: fresh RC phase paused")
            return True
        except (TimeoutError, GameDeadError):
            return self.hard_reset(reset_mode="hard_transport_failure")

    def wait_until_stable(self, cond, stable=0.4, timeout=30.0, what="",
                          poll=0.1):
        """Like wait_until, but the condition must HOLD for `stable` seconds
        continuously — filters the mission script's transient first RC
        handoff (the heli appears briefly, control drops for the intro beat,
        then the real handoff lands)."""
        t0 = time.perf_counter()
        held_since = None
        while time.perf_counter() - t0 < timeout:
            try:
                st = self.game.get_state()
            except GameDeadError:
                raise
            except RuntimeError:
                st = None
            ok = st is not None and cond(st)
            if ok and held_since is None:
                held_since = time.perf_counter()
            elif not ok:
                held_since = None
            if held_since is not None and time.perf_counter() - held_since >= stable:
                return True
            time.sleep(poll)
        raise TimeoutError("timeout waiting for stable: %s" % what)

    def hard_reset(self, timeout=90.0, reset_mode="hard_requested"):
        """Restart this instance only, with an exclusive controller claim."""
        self.last_reset_mode = reset_mode
        self.log("reset_mode=%s: starting a clean game process" % reset_mode)
        if self.game is not None:
            game = self.game
            pid = game.pid
            # Refuse to kill an instance owned by another trainer, including
            # the interval between its mutex claim and game-side publication.
            if game.alive():
                game._claim(timeout=0.1)
                try:
                    game._stop_refresher()
                    result = subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
                    if result.returncode and game.alive():
                        raise RuntimeError("could not terminate selected game")
                finally:
                    try:
                        game.close()
                    except GameDeadError:
                        pass
            else:
                try:
                    game.close()
                except GameDeadError:
                    pass
            self.game = None
            _wait_revc_gone(15.0, pid=pid)
            time.sleep(6.0)
        self.ensure_game(self._launch())
        self._advance_native_tutorial(timeout=timeout)
        self.wait_until_stable(self._in_fresh_rc_phase, stable=0.4,
                               timeout=timeout, what="fresh RC phase after launch")
        initial = self.game.pause()
        if not self._in_fresh_rc_phase(initial):
            raise RuntimeError("RC phase changed before hard-reset barrier")
        return True


def _wait_revc_gone(timeout=15.0, pid=None):
    """Block until no reVC.exe process object remains in the system.
    Output is matched as raw bytes: tasklist text is codepage-encoded
    (GBK here) but the image name itself is ASCII."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if pid is not None:
            if not _pid_alive_quiet(pid):
                return True
            time.sleep(0.1)
            continue
        out = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq reVC.exe", "/FO", "CSV", "/NH"],
            capture_output=True).stdout or b""
        if b"reVC.exe" not in out:
            return True
        time.sleep(1.0)
    return False


def _pid_alive_quiet(pid):
    try:
        from ml_client import pid_alive
        return pid_alive(pid)
    except Exception:
        return True


def demo():
    print("=== VC Demolition Man task pipeline demo ===")
    task = VcDemolitionTask()
    obs = task.observe()
    print("mission=%d cutscene=%d playerState=%d clock=%02d:%02d frame=%d" % (
        obs["missionRunning"], obs["cutsceneProcessing"], obs["playerState"],
        obs["clock"][0], obs["clock"][1], obs["frame"]))

    print("\n-- nearest mission targets --")
    for b in task.targets()[:5]:
        print("  blip pos=(%7.1f, %7.1f, %5.1f) rel=(%6.1f, %6.1f, %5.1f) "
              "dist=%.1f color=%d type=%d" %
              (b["pos"] + b["rel"] + (b["dist2d"] ** 0.5, b["color"], b["blipType"])))

    print("\n-- hover test: full throttle 2s, then coast 2s --")
    h0 = task.observe()["heli"]
    obs = task.step(throttle=True, frames=60)
    h1 = obs["heli"]
    obs = task.step(frames=60)
    h2 = obs["heli"]
    for tag, h in (("start", h0), ("after throttle", h1), ("after coast", h2)):
        print("  %-15s pos=(%7.1f, %7.1f, %6.1f) speed=%.2f" %
              (tag, h["pos"][0], h["pos"][1], h["pos"][2], h["speed"]))

    task.release()
    print("\nDemo done. Controls released; game keeps running headless.")


if __name__ == "__main__":
    demo()
