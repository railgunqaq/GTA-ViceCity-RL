#define WITHWINDOWS
#include "common.h"
#include "crossplatform.h"

#include "DMAudio.h"
#include "Record.h"
#include "Timer.h"
#include "SpecialFX.h"

uint32 CTimer::m_snTimeInMilliseconds;
PauseModeTime CTimer::m_snTimeInMillisecondsPauseMode = 1;

uint32 CTimer::m_snTimeInMillisecondsNonClipped;
uint32 CTimer::m_snPreviousTimeInMilliseconds;
uint32 CTimer::m_FrameCounter;
float CTimer::ms_fTimeScale;
float CTimer::ms_fTimeStep;
float CTimer::ms_fTimeStepNonClipped;
bool  CTimer::m_UserPause;
bool  CTimer::m_CodePause;

uint32 _nCyclesPerMS = 1;

#ifdef _WIN32
LARGE_INTEGER _oldPerfCounter;
LARGE_INTEGER perfSuspendCounter;
#define RsTimerType uint32
#else
#define RsTimerType double
#endif

RsTimerType oldPcTimer;

RsTimerType suspendPcTimer;

uint32 suspendDepth;

#ifdef FIX_HIGH_FPS_BUGS_ON_FRONTEND
double frameTime;
#endif

// TEMP ML instrumentation state (see dump block at end of Update)
static int32 s_lastCyc = 0;
static double s_sumFrameTime = 0.0, s_sumRaw = 0.0;
static int32 s_minCyc = 0x7FFFFFFF, s_maxCyc = -0x7FFFFFFF;
static uint32 s_negCyc = 0, s_paused = 0, s_n = 0, s_sumAdvance = 0;
static double s_clockCarry = 0.0; // fractional-ms carry for the uint32 clock

void CTimer::Initialise(void)
{
	debug("Initialising CTimer...\n");
	
	ms_fTimeScale = 1.0f;
	ms_fTimeStep = 1.0f;
	suspendDepth = 0;
	m_UserPause = false;
	m_CodePause = false;
	m_snTimeInMillisecondsNonClipped = 0;
	m_snPreviousTimeInMilliseconds = 0;
	m_snTimeInMilliseconds = 1;
	
#ifdef _WIN32
	LARGE_INTEGER perfFreq;
	if ( QueryPerformanceFrequency(&perfFreq) )
	{
		OutputDebugString("Performance counter available\n");
		_nCyclesPerMS = uint32(perfFreq.QuadPart / 1000);
		QueryPerformanceCounter(&_oldPerfCounter);
	}
	else
#endif
	{
		OutputDebugString("Performance counter not available, using millesecond timer\n");
		_nCyclesPerMS = 0;
		oldPcTimer = RsTimer();
	}
	
	m_snTimeInMilliseconds = m_snPreviousTimeInMilliseconds;
	
	m_FrameCounter = 0;
	
	DMAudio.ResetTimers(m_snPreviousTimeInMilliseconds);
	
	debug("CTimer ready\n");
}

void CTimer::Shutdown(void)
{
	;
}

void CTimer::ResetWallClock(void)
{
#ifdef _WIN32
	if (_nCyclesPerMS != 0)
		QueryPerformanceCounter(&_oldPerfCounter);
	else
#endif
		oldPcTimer = RsTimer();
}

void CTimer::UpdateFixed(uint32 milliseconds)
{
	ResetWallClock();
	m_snPreviousTimeInMilliseconds = m_snTimeInMilliseconds;
	uint32 dt = GetIsPaused() ? 0 : milliseconds;
	m_snTimeInMilliseconds += dt;
	m_snTimeInMillisecondsNonClipped += dt;
	m_snTimeInMillisecondsPauseMode += milliseconds;
	ms_fTimeStep = dt * 0.05f;
	ms_fTimeStepNonClipped = ms_fTimeStep;
#ifdef FIX_HIGH_FPS_BUGS_ON_FRONTEND
	frameTime = milliseconds;
#endif
	++m_FrameCounter;
}

void CTimer::Update(void)
{
	m_snPreviousTimeInMilliseconds = m_snTimeInMilliseconds;
	
#ifdef _WIN32
	if ( (double)_nCyclesPerMS != 0.0 )
	{
		LARGE_INTEGER pc;
		QueryPerformanceCounter(&pc);
		
			int32 updInCycles = (pc.LowPart - _oldPerfCounter.LowPart); // & 0x7FFFFFFF; pointless

			_oldPerfCounter = pc;
			s_lastCyc = updInCycles; // TEMP ML instrumentation
		
		float updInCyclesScaled = GetIsPaused() ? updInCycles : updInCycles * ms_fTimeScale;
		
		// We need that real frame time to fix transparent menu bug.
#ifndef FIX_HIGH_FPS_BUGS_ON_FRONTEND
		double
#endif
		frameTime = updInCyclesScaled / (double)_nCyclesPerMS;

		m_snTimeInMillisecondsPauseMode = m_snTimeInMillisecondsPauseMode + frameTime;
		
		if ( GetIsPaused() )
			ms_fTimeStep = 0.0f;
		else
		{
			// ML fix: accumulate with a fractional carry. The stock code
			// stores the clock in uint32 and truncates the sub-millisecond
			// remainder on EVERY frame, which ate 20-40% of game time at
			// high frame rates (1.7ms frames lose ~0.4ms each).
			s_clockCarry += frameTime;
			uint32 nWholeMs = (uint32)s_clockCarry;
			s_clockCarry -= nWholeMs;
			m_snTimeInMilliseconds = m_snTimeInMilliseconds + nWholeMs;
			m_snTimeInMillisecondsNonClipped = m_snTimeInMillisecondsNonClipped + nWholeMs;
			ms_fTimeStep = frameTime / 1000.0f * 50.0f;
		}
	}
	else
#endif
	{
		RsTimerType timer = RsTimer();

		RsTimerType updInMs = timer - oldPcTimer;

		// We need that real frame time to fix transparent menu bug.
#ifndef FIX_HIGH_FPS_BUGS_ON_FRONTEND
		double
#endif
		frameTime = (double)updInMs * ms_fTimeScale;

		oldPcTimer = timer;

		m_snTimeInMillisecondsPauseMode = m_snTimeInMillisecondsPauseMode + frameTime;

		if ( GetIsPaused() )
			ms_fTimeStep = 0.0f;
		else
		{
			s_clockCarry += frameTime;
			uint32 nWholeMs2 = (uint32)s_clockCarry;
			s_clockCarry -= nWholeMs2;
			m_snTimeInMilliseconds = m_snTimeInMilliseconds + nWholeMs2;
			m_snTimeInMillisecondsNonClipped = m_snTimeInMillisecondsNonClipped + nWholeMs2;
			ms_fTimeStep = frameTime / 1000.0f * 50.0f;
		}
	}
	
	if ( ms_fTimeStep < 0.01f && !GetIsPaused() && !CSpecialFX::bSnapShotActive)
		ms_fTimeStep = 0.01f;

	ms_fTimeStepNonClipped = ms_fTimeStep;
	
	if ( !CRecordDataForGame::IsPlayingBack() )
	{
		ms_fTimeStep = Min(3.0f, ms_fTimeStep);
		// ML fix: the stock 60ms clock cap discarded real wall time during
		// streaming stalls (measured single frames of 249ms losing 70%+ of
		// their time). Let the clock run true to wall time; per-frame
		// physics remains bounded by the timestep cap above.
	}
  
	if ( CRecordDataForChase::IsRecording() )
	{
		ms_fTimeStep = 1.0f;
		m_snTimeInMilliseconds = m_snPreviousTimeInMilliseconds + 16;
	}
  
	m_FrameCounter++;

	// TEMP ML instrumentation: per-frame timing diagnostics to find why the
	// game clock lags wall time under render skipping. Remove after debug.
	{
		s_n++;
		s_sumRaw += s_lastCyc;
		if (s_lastCyc < s_minCyc) s_minCyc = s_lastCyc;
		if (s_lastCyc > s_maxCyc) s_maxCyc = s_lastCyc;
		if (s_lastCyc < 0) s_negCyc++;
		s_sumFrameTime += frameTime;
		if (GetIsPaused()) s_paused++;
		s_sumAdvance += m_snTimeInMilliseconds - m_snPreviousTimeInMilliseconds;
		if (s_n >= 300) {
			FILE* dbg = fopen("revc_timer_dbg.log", "a");
			if (dbg) {
				LARGE_INTEGER nowpc; QueryPerformanceCounter(&nowpc);
				fprintf(dbg, "n=%u cycPerMS=%u rawSum=%.0f min=%d max=%d neg=%u paused=%u frameTimeSum=%.1f advSum=%u pc=%llu ts=%.2f step=%.3f\n",
					s_n, _nCyclesPerMS, s_sumRaw, s_minCyc, s_maxCyc, s_negCyc, s_paused,
					s_sumFrameTime, s_sumAdvance, (unsigned long long)nowpc.QuadPart, ms_fTimeScale, ms_fTimeStep);
				fclose(dbg);
			}
			s_sumFrameTime = s_sumRaw = 0.0; s_minCyc = 0x7FFFFFFF; s_maxCyc = -0x7FFFFFFF;
			s_negCyc = s_paused = 0; s_n = 0; s_sumAdvance = 0;
		}
	}
}

void CTimer::Suspend(void)
{
	// ML build no-op: streaming/collision/script loaders suspend the clock
	// around every disk read (an HDD-era design). With render skipping the
	// per-frame load budget fires many times more often per real second, and
	// the accumulated suspension slowed the whole world to ~0.5x. Letting
	// loads count as game time is safe here: the 60ms/frame cap in Update()
	// bounds each jump to 60ms and this machine's SSD loads in ~1ms.
}

void CTimer::Resume(void)
{
	// no-op companion to the Suspend() no-op above (suspendDepth stays 0,
	// the reference counter is never shifted, so nothing to restore)
}

uint32 CTimer::GetCyclesPerMillisecond(void)
{
#ifdef _WIN32
	if (_nCyclesPerMS != 0)
		return _nCyclesPerMS;
	else 
#endif
		return 1;
}

uint32 CTimer::GetCurrentTimeInCycles(void)
{
#ifdef _WIN32
	if ( _nCyclesPerMS != 0 )
	{
		LARGE_INTEGER pc;
		QueryPerformanceCounter(&pc);
		return (pc.LowPart - _oldPerfCounter.LowPart); // & 0x7FFFFFFF; pointless
	}
	else
#endif
		return RsTimer() - oldPcTimer;
}

bool CTimer::GetIsSlowMotionActive(void)
{
	return ms_fTimeScale < 1.0f;
}

void CTimer::Stop(void)
{
	m_snPreviousTimeInMilliseconds = m_snTimeInMilliseconds;
}

void CTimer::StartUserPause(void)
{
	m_UserPause = true;
}

void CTimer::EndUserPause(void)
{
	m_UserPause = false;
}

uint32 CTimer::GetCyclesPerFrame()
{
	return 20;
}

