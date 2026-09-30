// ecoqos_run <command line...>
// Starts the command with process-level EcoQoS (power throttling) forced on,
// the same state Task Manager's "Efficiency mode" sets, and waits for it.
#define _WIN32_WINNT 0x0A00
#include <windows.h>
#include <stdio.h>
#include <wchar.h>

int wmain(void) {
    wchar_t * cl = GetCommandLineW();
    // skip our own program name
    if (*cl == L'"') { cl++; while (*cl && *cl != L'"') cl++; if (*cl) cl++; }
    else { while (*cl && *cl != L' ' && *cl != L'\t') cl++; }
    while (*cl == L' ' || *cl == L'\t') cl++;
    if (!*cl) { fwprintf(stderr, L"usage: ecoqos_run <command...>\n"); return 2; }

    STARTUPINFOW si = { sizeof(si) };
    PROCESS_INFORMATION pi;
    if (!CreateProcessW(NULL, cl, NULL, NULL, TRUE, CREATE_SUSPENDED, NULL, NULL, &si, &pi)) {
        fwprintf(stderr, L"CreateProcess failed: %lu\n", GetLastError());
        return 1;
    }

    PROCESS_POWER_THROTTLING_STATE s = { 0 };
    s.Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    s.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    s.StateMask   = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    if (!SetProcessInformation(pi.hProcess, ProcessPowerThrottling, &s, sizeof(s))) {
        fwprintf(stderr, L"SetProcessInformation failed: %lu\n", GetLastError());
        TerminateProcess(pi.hProcess, 1);
        return 1;
    }

    ResumeThread(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return (int) code;
}
