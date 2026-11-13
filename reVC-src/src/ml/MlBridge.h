// ML protocol v7. State and action have separate named Win32 mutexes.
// A mutex acquire/release is the interprocess memory-ordering boundary;
// odd/even publication sequences additionally reject interrupted writes.
// RL steps execute exactly N fixed 10ms gameplay frames, then remain frozen.
// Free-running diagnostic hold() and mission reset explicitly leave lockstep.
#pragma once
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstddef>
#include "Pad.h"
#include "Clock.h"
#include "Timer.h"
#include "Radar.h"
#include "World.h"
#include "Pools.h"
#include "PlayerInfo.h"
#include "CutsceneMgr.h"
#include "Stats.h"
#include "User.h"
#include "Object.h"
#include "Automobile.h"
#include "ColStore.h"
#include "ModelIndices.h"
#include "control/Script.h"
#define REVC_ML_MAGIC 0x43564552u
#define REVC_ML_VERSION 7u
#define REVC_ML_STEP_MS 10u
#define REVC_ML_WATCHDOG_MS 5000u
namespace ML {
struct MlBlip {
	float x, y, z;
	uint32_t color, blipType, sprite;
	uint32_t blipId; // native radar slot | (generation << 16), independent of position
	int32_t objectiveKind, objectiveIndex; // 0/-1 unknown; 1=bomb, 2=drop target
};
struct MlObjective {
	int32_t kind, index, valid;
	float x, y, z;
};
// collision-rays-v2: metres from body origin, physical (not visibility) materials.
struct MlHit {
 int32_t valid, hit, entityType, modelId, surface;
 float distance, point[3], normal[3];
};
struct MlQueryResult {
 uint32_t seq;
 int32_t status; // 1=valid, -1=input, -2=not frozen, -3=unloaded, -4=no valid RC bounds
 uint32_t frame, episodeId;
 float radius, sampleSpacing, length;
 MlHit line, sweep;
};
struct MlState {
	uint32_t magic, version, size, pid;
	uint32_t seq, frame;
	int32_t missionRunning, cutsceneProcessing, playerState;
	uint32_t gameTimerMs;
	float clockHours, clockMinutes;
	float playerX, playerY, playerZ;
	int32_t heliExists, heliModelId;
	float heliX, heliY, heliZ, heliHeading;
	float heliVX, heliVY, heliVZ, heliSpeed, heliHealth;
	float rays[13];
	uint32_t numBlips;
	MlBlip blips[32];
	uint32_t lastActionSeq, lastCommandSeq;
	int32_t commandResult;
	uint32_t controllerPid;
	int32_t injectActive;
	uint32_t lastStepSeq, stepFrames;
	int32_t stepActive, lockstep;
	uint32_t episodeId;
	int32_t missionSuccessCount;
	uint32_t simStepMs;
	int32_t controlsMode;
	float heliForward[3], heliUp[3], heliTurnVel[3];
	uint32_t heartbeatAgeMs;
	uint32_t stepStartFrame, stepStartGameMs;
	int32_t heliRef;
	uint8_t _pad[4];
	int32_t missionTelemetryValid, missionId;
	uint32_t missionEpoch;
	int32_t bombState[4];
	uint32_t targetCompleteMask;
	int32_t timerState, missionTimerMs;
	MlObjective objectives[8];
	MlHit rayHits[13];
	uint32_t sensorVersion;
	float clearanceRadius, bodyBoundRadius, bodyBoundCenter[3];
	MlQueryResult query;
};
struct MlAction {
	uint32_t magic, version, seq, commandSeq, controllerPid;
	int32_t active;
	int16_t LeftStickX, LeftStickY, RightStickX, RightStickY;
	int16_t LeftShoulder1, LeftShoulder2, RightShoulder1, RightShoulder2;
	int16_t DPadUp, DPadDown, DPadLeft, DPadRight;
	int16_t Start, Select, Square, Triangle, Cross, Circle, LeftShock, RightShock;
	uint32_t command;
	int32_t commandArg;
	uint32_t actionSeq, stepSeq, stepFrames;
	int32_t lockstep;
	uint32_t heartbeatSeq;
	uint8_t _pad[44];
	uint32_t querySeq;
	float queryStart[3], queryEnd[3], queryRadius;
};
struct MlShared { MlState state; MlAction action; };
static_assert(sizeof(MlBlip) == 36, "ML blip ABI");
static_assert(offsetof(MlBlip, blipId) == 24, "ML blip ID offset");
static_assert(offsetof(MlBlip, objectiveKind) == 28, "ML objective kind offset");
static_assert(sizeof(MlObjective) == 24, "ML objective ABI");
static_assert(sizeof(MlState) == 2424, "ML state ABI");
static_assert(sizeof(MlAction) == 168, "ML action ABI");
static_assert(sizeof(MlShared) == 2592, "ML shared ABI");
static_assert(offsetof(MlState, blips) == 160, "ML blip array offset");
static_assert(offsetof(MlState, lastActionSeq) == 1312, "ML post-blip state offset");
static_assert(offsetof(MlState, missionTelemetryValid) == 1420, "ML mission offset");
static_assert(offsetof(MlState, objectives) == 1460, "ML objectives offset");
static_assert(offsetof(MlState, seq) == 16, "ML state seq alignment");
static_assert(offsetof(MlAction, seq) == 8, "ML action seq alignment");
static_assert(offsetof(MlShared, action) == 2424, "ML action offset");

static_assert(sizeof(MlHit) == 48, "ML hit ABI");
static_assert(sizeof(MlQueryResult) == 124, "ML query ABI");
static_assert(offsetof(MlState, rayHits) == 1652, "ML rays offset");
static_assert(offsetof(MlState, query) == 2300, "ML query offset");
static_assert(offsetof(MlAction, querySeq) == 136, "ML query request offset");

struct Bridge {
	MlShared *shared;
	HANDLE mapHandle, stateMutex, actionMutex;
	MlAction action;
	MlQueryResult query;
	uint32_t publication, owner, heartbeat, lastHeartbeatMs;
	uint32_t frame, episodeId, lastActionSeq, lastCommandSeq;
	int32_t commandResult;
	uint32_t requestedStep, completedStep, remaining, executed;
	uint32_t stepStartFrame, stepStartGameMs;
	bool lockstep, fixedTick, padSaved;
	int16_t oldPadMode;
	CRunningScript *missionScript;
	uint32_t missionEpoch, missionEpisode;
	bool missionFingerprint, missionInitialized;
	uint32_t missionHash, missionLoggedEpoch;
	int32_t missionReason;
#ifdef DETECT_PAD_INPUT_SWITCH
	bool oldControllerFlag;
#endif
	Bridge() : shared(nil), mapHandle(nil), stateMutex(nil), actionMutex(nil), publication(0),
		owner(0), heartbeat(0), lastHeartbeatMs(0), frame(0), episodeId(1),
		lastActionSeq(0), lastCommandSeq(0), commandResult(0), requestedStep(0),
		completedStep(0), remaining(0), executed(0), stepStartFrame(0),
		stepStartGameMs(0), lockstep(false),
		fixedTick(false), padSaved(false), oldPadMode(0), missionScript(nil),
		missionEpoch(0), missionEpisode(0), missionFingerprint(false), missionInitialized(false),
		missionHash(0), missionLoggedEpoch(0), missionReason(-1) {
		memset(&action, 0, sizeof(action));
		memset(&query, 0, sizeof(query));
		char name[80];
		sprintf(name, "ReVC_ML_State_PID%u", GetCurrentProcessId());
		stateMutex = CreateMutexA(nil, FALSE, name);
		sprintf(name, "ReVC_ML_Action_PID%u", GetCurrentProcessId());
		actionMutex = CreateMutexA(nil, FALSE, name);
		sprintf(name, "ReVC_ML_Bridge_PID%u", GetCurrentProcessId());
		mapHandle = CreateFileMappingA(INVALID_HANDLE_VALUE, nil, PAGE_READWRITE,
		                               0, sizeof(MlShared), name);
		if (mapHandle) {
			shared = (MlShared *)MapViewOfFile(mapHandle, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(MlShared));
			// Keep the named mapping handle alive for later client attachment.
		}
		if (shared) {
			memset(shared, 0, sizeof(*shared));
			shared->state.magic = REVC_ML_MAGIC;
			shared->state.version = REVC_ML_VERSION;
			shared->state.size = sizeof(MlState);
			shared->state.pid = GetCurrentProcessId();
		}
	}
};
inline Bridge &Runtime() { static Bridge b; return b; }
inline MlShared *GetShared() { return Runtime().shared; }
// Called after StartNewScript initializes its slot. Capturing the actual start
// prevents a slot reused between two published frames from inheriting validity.
inline void OnScriptStarted(CRunningScript *script, uint32_t ip) {
	Bridge &b = Runtime();
	if (ip != 0 && ip != SIZE_MAIN_SCRIPT && script != b.missionScript) return;
	if (++b.missionEpoch == 0) ++b.missionEpoch;
	b.missionScript = nil; b.missionEpisode = 0;
	b.missionFingerprint = false; b.missionInitialized = false; b.missionHash = 0;
	if (ip != SIZE_MAIN_SCRIPT) return;
	// Hash loaded bytes, not the potentially different current disk file.
	const uint32_t missionSize = 23780;
	static_assert(SIZE_MISSION_SCRIPT >= missionSize, "mission19 fingerprint range");
	uint32_t hash = 2166136261u;
	for (uint32_t i = 0; i < missionSize; ++i)
		hash = (hash ^ CTheScripts::ScriptSpace[SIZE_MAIN_SCRIPT + i]) * 16777619u;
	b.missionScript = script;
	b.missionHash = hash;
	b.missionFingerprint = hash == 0x007e5a02u;
}
// A bounded diagnostic: one line per gate transition, never per frame. No
// protocol expansion and no relaxed validity when initialization is pending.
inline void MissionGateReason(int32_t reason) {
	Bridge &b = Runtime();
	if (b.missionReason == reason && b.missionLoggedEpoch == b.missionEpoch) return;
	b.missionReason = reason; b.missionLoggedEpoch = b.missionEpoch;
	FILE *log = fopen("revc_ml_telemetry.log", "a");
	if (!log) return;
	CRunningScript *script = b.missionScript;
	fprintf(log, "pid=%u epoch=%u episode=%u frame=%u reason=%d hash=%08X ip=%d name=%.8s initialized=%d varspace=%d",
	        GetCurrentProcessId(), b.missionEpoch, b.episodeId, b.frame, reason, b.missionHash,
	        script ? int32_t(script->m_nIp) - int32_t(SIZE_MAIN_SCRIPT) : -1,
	        script ? script->m_abScriptName : "none", b.missionInitialized ? 1 : 0,
	        CTheScripts::GetSizeOfVariableSpace());
	for (int i = 0; i < NUMONSCREENCOUNTERS; ++i) {
		const COnscreenCounterEntry &c = CUserDisplay::OnscnTimer.m_sCounters[i];
		fprintf(log, " counter%d=%u:%.10s", i, c.m_nCounterOffset, c.m_aCounterText);
	}
	fprintf(log, "\n"); fclose(log);
}
inline int32_t MissionInt(uint32_t offset) {
	int32_t value;
	memcpy(&value, &CTheScripts::ScriptSpace[offset], sizeof(value));
	return value;
}
inline float MissionFloat(uint32_t offset) {
	float value;
	memcpy(&value, &CTheScripts::ScriptSpace[offset], sizeof(value));
	return value;
}
inline void ReadMissionTelemetry(MlState &s, int32_t (&bombRefs)[4], uint32_t (&targetRefs)[4]) {
	Bridge &b = Runtime();
	s.missionId = -1; s.missionEpoch = b.missionEpoch;
	s.timerState = s.missionTimerMs = -1;
	for (int i = 0; i < 4; ++i) {
		s.bombState[i] = bombRefs[i] = -1; targetRefs[i] = 0xffffffffu;
		s.objectives[i].kind = 1; s.objectives[i].index = i;
		s.objectives[4 + i].kind = 2; s.objectives[4 + i].index = i;
	}
	if (!b.missionScript) { MissionGateReason(1); return; }
	if (!b.missionFingerprint) { MissionGateReason(2); return; }
	bool active = false;
	for (CRunningScript *script = CTheScripts::pActiveScripts; script; script = script->GetNext()) {
		if (script == b.missionScript && script->m_bIsActive && script->m_bIsMissionScript)
			active = true;
	}
	if (!active || (b.missionEpisode && b.missionEpisode != b.episodeId)) {
		MissionGateReason(!active ? 3 : 4);
		b.missionScript = nil; b.missionInitialized = false; b.missionFingerprint = false;
		return;
	}
	b.missionEpisode = b.episodeId;
	// COMMAND_SCRIPT_NAME lowercases the eight-byte name in Script4.cpp.
	static const char scriptName[8] = "serg3";
	if (memcmp(b.missionScript->m_abScriptName, scriptName, sizeof(scriptName)) != 0) {
		MissionGateReason(5); return;
	}
	if (CTheScripts::GetSizeOfVariableSpace() < 13616) { MissionGateReason(6); return; }
	if (!b.missionInitialized) {
		// The first WAIT at +62 is before global initialization. Require both
		// execution beyond initialization and the mission's own HUD binding.
		uint32_t ip = b.missionScript->m_nIp;
		if (ip < SIZE_MAIN_SCRIPT + 698 || ip >= SIZE_MAIN_SCRIPT + 23780) { MissionGateReason(7); return; }
		for (int i = 0; i < NUMONSCREENCOUNTERS; ++i) {
			const COnscreenCounterEntry &counter = CUserDisplay::OnscnTimer.m_sCounters[i];
			if (counter.m_nCounterOffset == 13228 &&
			    strncmp(counter.m_aCounterText, "TEX3_11", 8) == 0)
				b.missionInitialized = true;
		}
		if (!b.missionInitialized) { MissionGateReason(8); return; }
	}
	int32_t states[4], timer = MissionInt(13612), remaining = 0, carrying = 0;
	uint32_t mask = 0;
	for (int i = 0; i < 4; ++i) {
		states[i] = MissionInt(13568 + i * 4);
		int32_t done = MissionInt(13584 + i * 4);
		if (states[i] < 0 || states[i] > 3 || done < 0 || done > 1) { MissionGateReason(9); return; }
		if (states[i] == 1) ++carrying;
		mask |= uint32_t(done) << i;
	}
	if (carrying > 1 || timer < 0 || timer > 2) { MissionGateReason(9); return; }
	if (timer) {
		remaining = MissionInt(13608);
		if (remaining < 0 || remaining > 420000) { MissionGateReason(9); return; }
	}
	s.missionTelemetryValid = 1; s.missionId = 19;
	MissionGateReason(0);
	s.targetCompleteMask = mask; s.timerState = timer; s.missionTimerMs = remaining;
	// CREATE_OBJECT's ARGUMENT_INT8 byte 0x84 is signed -124, which the
	// engine resolves through UsedObjectArray[124], not runtime model 132.
	int32_t bombModel = -1;
	if (CTheScripts::NumberOfUsedObjects > 124 &&
	    strncmp(CTheScripts::UsedObjectArray[124].name, "DYNAMITE", USED_OBJECT_NAME_LENGTH) == 0)
		bombModel = CTheScripts::UsedObjectArray[124].index;
	for (int i = 0; i < 4; ++i) {
		s.bombState[i] = states[i];
		int32_t ref = MissionInt(13340 + i * 4);
		CObjectPool *pool = CPools::GetObjectPool();
		CObject *object = pool && ref >= 0 && (ref >> 8) < pool->GetSize() ? pool->GetAt(ref) : nil;
		if (object && bombModel > 0 && object->GetModelIndex() == bombModel) {
			const CVector &pos = object->GetPosition();
			if (std::isfinite(pos.x) && std::isfinite(pos.y) && std::isfinite(pos.z)) {
				bombRefs[i] = ref;
				MlObjective &o = s.objectives[i];
				o.valid = 1; o.x = pos.x; o.y = pos.y; o.z = pos.z;
			}
		}
		// These are the same script floats used by LOCATE_OBJECT_3D and
		// ADD_BLIP_FOR_COORD; no geometric matching or inferred marker identity.
		float x = MissionFloat(13468 + i * 12), y = MissionFloat(13472 + i * 12),
		      z = MissionFloat(13476 + i * 12);
		if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
			MlObjective &o = s.objectives[4 + i];
			o.valid = 1; o.x = x; o.y = y; o.z = z;
		}
		targetRefs[i] = uint32_t(MissionInt(13516 + i * 4));
	}
}
inline void EmptyHit(MlHit &hit) {
 memset(&hit, 0, sizeof(hit));
 hit.entityType = hit.modelId = hit.surface = -1;
}
inline void SetHit(MlHit &hit, CEntity *entity, const CColPoint &col, float distance) {
 hit.valid = hit.hit = 1; hit.entityType = entity->GetType();
 hit.modelId = entity->GetModelIndex(); hit.surface = col.surfaceB; hit.distance = distance;
 hit.point[0] = col.point.x; hit.point[1] = col.point.y; hit.point[2] = col.point.z;
 hit.normal[0] = col.normal.x; hit.normal[1] = col.normal.y; hit.normal[2] = col.normal.z;
}
inline CEntity *CarriedBomb(const MlState &state, const int32_t (&refs)[4]) {
 if (!state.missionTelemetryValid) return nil;
 for (int i = 0; i < 4; ++i)
  if (state.bombState[i] == 1 && refs[i] >= 0) return CPools::GetObjectPool()->GetAt(refs[i]);
 return nil;
}
inline float ClearanceRadius(CVehicle *heli, CEntity *carried) {
 if (!heli || !heli->IsCar() || (heli->GetModelIndex() != MI_RCGOBLIN && heli->GetModelIndex() != MI_RCRAIDER)) return 0;
 CColModel *model = heli->GetColModel();
 RwFrame *rotor = ((CAutomobile *)heli)->m_aCarNodes[CAR_BONNET];
 if (!model || !rotor) return 0;
 float radius = model->boundingSphere.radius + model->boundingSphere.center.Magnitude();
 CMatrix rotorMatrix; rotorMatrix.Attach(RwFrameGetMatrix(rotor));
 float rotorRadius = rotorMatrix.GetPosition().Magnitude() + 0.72f;
 if (rotorRadius > radius) radius = rotorRadius;
 if (carried) {
  float bombRadius = (carried->GetBoundCentre() - heli->GetPosition()).Magnitude() + carried->GetBoundRadius();
  if (bombRadius > radius) radius = bombRadius;
 }
 return std::isfinite(radius) && radius > 0 ? radius + 0.15f : 0;
}
inline bool LoadedPoint(const CVector &point, float radius) {
 // HasCollisionLoaded uses the same streamed collision residency check as missions.
 // Corner checks cover the query footprint as well as its centre.
 for (int x = -1; x <= 1; ++x) for (int y = -1; y <= 1; ++y)
  if (!CColStore::HasCollisionLoaded(CVector2D(point.x + x*radius, point.y + y*radius))) return false;
 return true;
}
inline bool LoadedSegment(const CVector &from, const CVector &to, float radius) {
 CVector delta = to - from;
 int samples = (int)ceilf(delta.Magnitude() / 5.0f); if (samples < 1) samples = 1;
 for (int i = 0; i <= samples; ++i)
  if (!LoadedPoint(from + delta * (float(i) / samples), radius)) return false;
 return true;
}
inline void Raycast(const CVector &from, const CVector &to, CEntity *carried, MlHit &hit) {
 EmptyHit(hit); hit.valid = 1;
 if ((to-from).MagnitudeSqr() < 1.0e-12f) return; // point occupancy belongs to sphere sweep
 CEntity *previousIgnore = CWorld::pIgnoreEntity;
 CWorld::pIgnoreEntity = carried;
 CColPoint col = {}; CEntity *entity = nil;
 // Static geometry includes see-through *physical* surfaces (mesh/scaffold/glass).
 // All vehicles are excluded, hence the helicopter cannot hit itself. Peds are
 // not static map geometry; these sensors are not dynamic-agent avoidance.
 bool found = CWorld::ProcessLineOfSight(from, to, col, entity, true, false, false, true, true, false, false, false);
 CWorld::pIgnoreEntity = previousIgnore;
 if (found && entity) SetHit(hit, entity, col, (col.point-from).Magnitude());
 else hit.distance = (to-from).Magnitude();
}
inline bool QueryPointValid(const CVector &point) {
 return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z) &&
  fabsf(point.x) <= 1900 && fabsf(point.y) <= 1900 && point.z >= -100 && point.z <= 200;
}
inline void ProcessQuery(const MlAction &action) {
 Bridge &b = Runtime();
 MlQueryResult result = {}; result.seq = action.querySeq; result.status = -1;
 result.frame = b.frame; result.episodeId = b.episodeId;
 EmptyHit(result.line); EmptyHit(result.sweep);
 if (!b.lockstep || b.remaining || !b.owner || !action.active) result.status = -2;
 else {
  CVector from(action.queryStart[0], action.queryStart[1], action.queryStart[2]);
  CVector to(action.queryEnd[0], action.queryEnd[1], action.queryEnd[2]);
  CVector delta = to - from; float length = delta.Magnitude();
  MlState state = {}; int32_t refs[4]; uint32_t targetRefs[4];
  ReadMissionTelemetry(state, refs, targetRefs);
  CEntity *carried = CarriedBomb(state, refs);
  CVehicle *heli = CWorld::Players[CWorld::PlayerInFocus].m_pRemoteVehicle;
  float minimum = ClearanceRadius(heli, carried);
  float radius = action.queryRadius == 0 ? minimum : action.queryRadius;
  if (minimum <= 0) result.status = -4;
  else if (QueryPointValid(from) && QueryPointValid(to) && std::isfinite(radius) &&
           radius >= minimum - 0.00001f && radius <= 5.0f && length <= 200.0f) {
   result.radius = radius; result.length = length;
   int samples = (int)ceilf(length / 0.25f); if (samples < 1) samples = 1;
   result.sampleSpacing = length / samples;
   // Every swept-sphere point is at most half a sample interval from a tested
   // centre. Adding that amount makes the sampled union conservative, including
   // obstacles near an arbitrary normal; both segment endpoints are sampled.
   float testRadius = radius + result.sampleSpacing * 0.5f;
   if (!LoadedSegment(from, to, testRadius)) result.status = -3;
   else {
    result.status = 1;
    Raycast(from, to, carried, result.line);
    result.sweep.valid = 1; result.sweep.distance = length;
    for (int i = 0; i <= samples; ++i) {
     CVector centre = from + delta * (float(i) / samples);
     CEntity *entity = CWorld::TestSphereAgainstWorld(centre, testRadius, carried,
       true, false, false, true, true, false);
     if (entity) {
      SetHit(result.sweep, entity, gaTempSphereColPoints[0], length * float(i) / samples);
      break;
     }
    }
   }
  }
 }
 b.query = result;
}
inline bool Lock(HANDLE mutex, DWORD timeout) {
	if (!mutex) return false;
	DWORD result = WaitForSingleObject(mutex, timeout);
	return result == WAIT_OBJECT_0 || result == WAIT_ABANDONED;
}
inline bool ReadAction(MlAction &a) {
	Bridge &b = Runtime();
	if (!b.shared || !Lock(b.actionMutex, 0)) return false;
	uint32_t seq = b.shared->action.seq;
	bool valid = seq && !(seq & 1);
	if (valid) memcpy(&a, &b.shared->action, sizeof(a));
	ReleaseMutex(b.actionMutex);
	return valid && a.magic == REVC_ML_MAGIC && a.version == REVC_ML_VERSION;
}
inline void PublishState() {
	Bridge &b = Runtime();
	if (!b.shared) return;
	MlState s = {};
	s.magic = REVC_ML_MAGIC; s.version = REVC_ML_VERSION;
	s.size = sizeof(MlState); s.pid = GetCurrentProcessId(); s.frame = b.frame;
	s.missionRunning = CTheScripts::bAlreadyRunningAMissionScript ? 1 : 0;
	s.cutsceneProcessing = CCutsceneMgr::IsCutsceneProcessing() ? 1 : 0;
	s.gameTimerMs = CTimer::GetTimeInMilliseconds();
	s.clockHours = (float)CClock::GetHours();
	s.clockMinutes = (float)CClock::GetMinutes();

	s.playerState = 0;
	CPed *pPlayer = FindPlayerPed();
	if (pPlayer) {
		const CVector &v = pPlayer->GetPosition();
		s.playerX = v.x; s.playerY = v.y; s.playerZ = v.z;
		if (pPlayer->InVehicle())
			s.playerState = 1;
	}

	CVehicle *pHeli = CWorld::Players[CWorld::PlayerInFocus].m_pRemoteVehicle;
	if (pHeli) {
		s.playerState = 2;
	}
	if (pHeli) {
		s.heliExists = 1;
		s.heliModelId = pHeli->GetModelIndex();
		s.heliRef = CPools::GetVehicleRef(pHeli);
		const CVector &v = pHeli->GetPosition();
		s.heliX = v.x; s.heliY = v.y; s.heliZ = v.z;
		const CVector &fwd = pHeli->GetForward();
		s.heliHeading = atan2f(fwd.y, fwd.x);
		const CVector &up = pHeli->GetUp();
		const CVector &turn = pHeli->m_vecTurnSpeed;
		s.heliForward[0] = fwd.x; s.heliForward[1] = fwd.y; s.heliForward[2] = fwd.z;
		s.heliUp[0] = up.x; s.heliUp[1] = up.y; s.heliUp[2] = up.z;
		s.heliTurnVel[0] = turn.x; s.heliTurnVel[1] = turn.y; s.heliTurnVel[2] = turn.z;
		const CVector &vel = pHeli->m_vecMoveSpeed;
		s.heliVX = vel.x; s.heliVY = vel.y; s.heliVZ = vel.z;
		s.heliSpeed = vel.Magnitude();
		s.heliHealth = pHeli->m_fHealth / 1000.0f;


	} else {
		s.heliExists = 0;
	}
	int32_t bombRefs[4]; uint32_t targetRefs[4];
	ReadMissionTelemetry(s, bombRefs, targetRefs);
	CEntity *carried = CarriedBomb(s, bombRefs);
	s.sensorVersion = 2;
	if (pHeli) {
		CColModel *model = pHeli->GetColModel();
		if (model) {
			s.bodyBoundRadius = model->boundingSphere.radius;
			s.bodyBoundCenter[0] = model->boundingSphere.center.x;
			s.bodyBoundCenter[1] = model->boundingSphere.center.y;
			s.bodyBoundCenter[2] = model->boundingSphere.center.z;
		}
		s.clearanceRadius = ClearanceRadius(pHeli, carried);
		const float h = s.heliHeading;
		static const float yaw[8] = {0, 0.7853982f, 1.5707963f, 2.3561945f,
		                             3.1415927f, -2.3561945f, -1.5707963f, -0.7853982f};
		const CVector from = pHeli->GetPosition();
		for (int i = 0; i < 13; ++i) {
			CVector dir;
			if (i < 8) dir = CVector(cosf(h + yaw[i]), sinf(h + yaw[i]), 0);
			else if (i == 8) dir = CVector(0, 0, 1);
			else if (i == 9) dir = CVector(cosf(h)*0.8660254f, sinf(h)*0.8660254f, 0.5f);
			else if (i == 10) dir = CVector(0, 0, -1);
			else if (i == 11) dir = CVector(cosf(h)*0.8660254f, sinf(h)*0.8660254f, -0.5f);
			else dir = CVector(-cosf(h)*0.8660254f, -sinf(h)*0.8660254f, -0.5f);
			CVector to = from + dir * 20.0f;
			EmptyHit(s.rayHits[i]);
			// Unknown collision coverage is conservatively zero range, not clear.
			if (LoadedSegment(from, to, 0.0f)) {
				Raycast(from, to, carried, s.rayHits[i]);
				s.rays[i] = s.rayHits[i].hit ? s.rayHits[i].distance / 20.0f : 1.0f;
			}
		}
	}
	s.query = b.query;

	uint32_t n = 0;
	for (int i = 0; i < NUMRADARBLIPS && n < 32; i++) {
		sRadarTrace &t = CRadar::ms_RadarTrace[i];
		if (!t.m_bInUse)
			continue;
		MlBlip b = {};
		b.objectiveIndex = -1;
		b.blipId = uint32_t(i) | (uint32_t(t.m_BlipIndex) << 16);
		b.color = t.m_nColor;
		b.blipType = t.m_eBlipType;
		b.sprite = t.m_eRadarSprite;
		if (t.m_eBlipType == BLIP_COORD || t.m_eBlipType == BLIP_CONTACT_POINT) {
			b.x = t.m_vecPos.x; b.y = t.m_vecPos.y; b.z = t.m_vecPos.z;
		} else {
			// entity blip: resolve through the pools; drop dead entities
			CEntity *pEntity = nil;
			if (t.m_eBlipType == BLIP_CAR)
				pEntity = CPools::GetVehiclePool()->GetAt(t.m_nEntityHandle);
			else if (t.m_eBlipType == BLIP_CHAR)
				pEntity = CPools::GetPedPool()->GetAt(t.m_nEntityHandle);
			else if (t.m_eBlipType == BLIP_OBJECT)
				pEntity = CPools::GetObjectPool()->GetAt(t.m_nEntityHandle);
			if (!pEntity)
				continue;
			const CVector &v = pEntity->GetPosition();
			b.x = v.x; b.y = v.y; b.z = v.z;
		}
		if (s.missionTelemetryValid) {
			for (int objective = 0; objective < 4; ++objective) {
				if (t.m_eBlipType == BLIP_OBJECT && bombRefs[objective] >= 0 &&
				    t.m_nEntityHandle == bombRefs[objective]) {
					b.objectiveKind = 1; b.objectiveIndex = objective;
				} else if (t.m_eBlipType == BLIP_COORD && b.blipId == targetRefs[objective]) {
					b.objectiveKind = 2; b.objectiveIndex = objective;
				}
			}
		}
		s.blips[n++] = b;
	}
	s.numBlips = n;

	s.lastActionSeq = b.lastActionSeq; s.lastCommandSeq = b.lastCommandSeq;
	s.commandResult = b.commandResult; s.controllerPid = b.owner;
	s.injectActive = b.owner && b.action.active ? 1 : 0;
	s.lastStepSeq = b.completedStep; s.stepFrames = b.executed;
	s.stepStartFrame = b.stepStartFrame; s.stepStartGameMs = b.stepStartGameMs;
	s.stepActive = b.remaining ? 1 : 0; s.lockstep = b.lockstep ? 1 : 0;
	s.episodeId = b.episodeId; s.missionSuccessCount = CStats::MissionsPassed;
	s.simStepMs = REVC_ML_STEP_MS; s.controlsMode = CPad::GetPad(0)->Mode;
	s.heartbeatAgeMs = b.owner ? GetTickCount() - b.lastHeartbeatMs : 0;
	if (!Lock(b.stateMutex, 100)) return;
	uint32_t seq = (b.shared->state.seq + 2u) & ~1u;
	if (!seq) seq = 2;
	InterlockedExchange((volatile LONG *)&b.shared->state.seq, (LONG)(seq - 1));
	// Do not overwrite the publication word during the body copy.
	memcpy(&b.shared->state, &s, offsetof(MlState, seq));
	memcpy((char *)&b.shared->state + offsetof(MlState, frame),
	       (char *)&s + offsetof(MlState, frame), sizeof(s) - offsetof(MlState, frame));
	MemoryBarrier();
	InterlockedExchange((volatile LONG *)&b.shared->state.seq, (LONG)seq);
	ReleaseMutex(b.stateMutex);
}
inline void RestorePad() {
	Bridge &b = Runtime();
	if (b.padSaved) {
		CPad::GetPad(0)->Mode = b.oldPadMode;
#ifdef DETECT_PAD_INPUT_SWITCH
		CPad::IsAffectedByController = b.oldControllerFlag;
#endif
		b.padSaved = false;
	}
}
inline void ConfigurePad() {
	Bridge &b = Runtime();
	CPad *pad = CPad::GetPad(0);
	if (!b.padSaved) {
		b.oldPadMode = pad->Mode;
#ifdef DETECT_PAD_INPUT_SWITCH
		b.oldControllerFlag = CPad::IsAffectedByController;
#endif
		b.padSaved = true;
	}
	pad->Mode = 0;
#ifdef DETECT_PAD_INPUT_SWITCH
	CPad::IsAffectedByController = false;
#endif
}
inline void StopControl() {
	Bridge &b = Runtime();
	b.owner = 0; b.action.active = 0; b.lockstep = false; b.remaining = 0;
	RestorePad();
	CTimer::ResetWallClock();
}
inline bool HasActiveMissionScript() {
	for (CRunningScript *script = CTheScripts::pActiveScripts; script; script = script->GetNext())
		if (script->m_bIsMissionScript) return true;
	return false;
}
// Called from Idle BEFORE advancing the clock. Returning false yields to
// the Windows event loop, so a locked simulation never freezes the window.
inline bool BeforeTick() {
	Bridge &b = Runtime();
	if (!b.shared) return true;
	MlAction a;
	bool changed = false;
	if (ReadAction(a) && a.seq != b.publication) {
		b.publication = a.seq;
		if (a.controllerPid != 0 && (b.owner == 0 || b.owner == a.controllerPid)) {
			if (a.active) {
				if (!b.owner) {
					b.owner = a.controllerPid;
					b.lastHeartbeatMs = GetTickCount();
				}
				if (a.heartbeatSeq != b.heartbeat) {
					b.heartbeat = a.heartbeatSeq;
					b.lastHeartbeatMs = GetTickCount();
				}
				b.action = a;
				b.lastActionSeq = a.actionSeq;
				ConfigurePad();
				if (!a.lockstep) { b.lockstep = false; b.remaining = 0; }
				else if (a.stepSeq && a.stepSeq != b.requestedStep) {
					b.lockstep = true; b.requestedStep = a.stepSeq;
					b.remaining = a.stepFrames <= 1000 ? a.stepFrames : 0;
					b.executed = 0; b.stepStartFrame = b.frame;
					b.stepStartGameMs = CTimer::GetTimeInMilliseconds();
					if (!b.remaining) b.completedStep = a.stepSeq;
				}
			} else if (b.owner == a.controllerPid) StopControl();
			if (a.command != 0 && a.commandSeq && a.commandSeq != b.lastCommandSeq) {
				// A mission command explicitly returns to free-running mode.
				StopControl();
				b.lastCommandSeq = a.commandSeq;
				b.commandResult = -1;
				// Never overwrite mission bytecode while an old mission is still
				// active, including its asynchronous death/arrest cleanup. Legacy
				// command 2 is accepted only under the same safe idle conditions.
				if ((a.command == 1 || a.command == 2) && a.commandArg == 19 &&
				    !CTheScripts::bAlreadyRunningAMissionScript &&
				    !HasActiveMissionScript() && !CCutsceneMgr::IsCutsceneProcessing()) {
					CTheScripts::SwitchToMission(a.commandArg);
					++b.episodeId;
					b.commandResult = 1;
				}
			}
			if (a.querySeq && a.querySeq != b.query.seq) ProcessQuery(a);
			changed = true;
		}
	}
	if (b.owner && GetTickCount() - b.lastHeartbeatMs > REVC_ML_WATCHDOG_MS) {
		StopControl(); changed = true;
	}
	if (changed) PublishState();
	if (b.lockstep && !b.remaining) {
		CTimer::ResetWallClock();
		Sleep(1);
		return false;
	}
	b.fixedTick = b.lockstep && b.remaining != 0;
	return true;
}
inline bool IsFixedTick() { return Runtime().fixedTick; }
// Called after CPad::UpdatePads, before scripts and physics consume input.
inline void ProcessFrame() {
	Bridge &b = Runtime();
	if (!b.owner || !b.action.active) return;
	CPad *pad = CPad::GetPad(0);
	ConfigurePad();
	CControllerState &cs = pad->NewState;
	const MlAction &a = b.action;
#define ML_AXIS(name) cs.name = a.name < -128 ? -128 : (a.name > 128 ? 128 : a.name)
#define ML_BUTTON(name) cs.name = a.name < 0 ? 0 : (a.name > 255 ? 255 : a.name)
	ML_AXIS(LeftStickX); ML_AXIS(LeftStickY); ML_AXIS(RightStickX); ML_AXIS(RightStickY);
	ML_BUTTON(LeftShoulder1); ML_BUTTON(LeftShoulder2); ML_BUTTON(RightShoulder1); ML_BUTTON(RightShoulder2);
	ML_BUTTON(DPadUp); ML_BUTTON(DPadDown); ML_BUTTON(DPadLeft); ML_BUTTON(DPadRight);
	ML_BUTTON(Start); ML_BUTTON(Select); ML_BUTTON(Square); ML_BUTTON(Triangle);
	ML_BUTTON(Cross); ML_BUTTON(Circle); ML_BUTTON(LeftShock); ML_BUTTON(RightShock);
#undef ML_AXIS
#undef ML_BUTTON
}
// The observation belongs to the END of a complete physics/script frame.
inline void EndFrame() {
	Bridge &b = Runtime();
	if (!CTimer::GetIsPaused()) {
		++b.frame;
		if (b.fixedTick && b.remaining) {
			--b.remaining; ++b.executed;
			if (!b.remaining) b.completedStep = b.requestedStep;
		}
	}
	PublishState();
}
} // namespace ML
#endif // _WIN32
