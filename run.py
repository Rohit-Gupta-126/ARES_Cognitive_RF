import subprocess
import sys
import os
import signal
import time
import threading

def log_stream(stream, prefix):
    """Reads a process stream line-by-line and prints with a prefix safely."""
    for line in iter(stream.readline, b''):
        # Decode stream
        decoded_line = line.decode('utf-8', errors='replace').strip()
        if decoded_line:
            msg = f"{prefix} {decoded_line}"
            try:
                print(msg, flush=True)
            except UnicodeEncodeError:
                # Bulletproof fallback for unsupported terminal characters
                encoding = sys.stdout.encoding or 'ascii'
                safe_bytes = msg.encode(encoding, errors='replace')
                print(safe_bytes.decode(encoding), flush=True)

def free_port(port):
    """Detects and terminates any process currently bound to the specified port."""
    try:
        # Check netstat for active port binding
        cmd = f"netstat -ano | findstr :{port}"
        output = subprocess.check_output(cmd, shell=True).decode('utf-8', errors='ignore')
        terminated_pids = set()
        for line in output.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 5 and 'LISTENING' in parts:
                pid = parts[-1]
                # Avoid self-termination
                if int(pid) != os.getpid() and int(pid) > 0 and pid not in terminated_pids:
                    print(f"[SYSTEM] Cleaning port {port} (Terminating PID {pid})...", flush=True)
                    subprocess.run(
                        f"taskkill /F /PID {pid}", 
                        shell=True, 
                        stdout=subprocess.DEVNULL, 
                        stderr=subprocess.DEVNULL
                    )
                    terminated_pids.add(pid)
                    time.sleep(0.5) # Allow OS to release port binding
    except Exception:
        pass

def run_system():
    # Enable Windows Virtual Terminal Processing for ANSI color support
    if os.name == 'nt':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            hStdOut = kernel32.GetStdHandle(-11) # STD_OUTPUT_HANDLE
            if hStdOut != -1:
                mode = ctypes.c_ulong()
                if kernel32.GetConsoleMode(hStdOut, ctypes.byref(mode)):
                    # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                    kernel32.SetConsoleMode(hStdOut, mode.value | 0x0004)
        except Exception:
            pass

    # Proactively clean ports 3000 (Next.js) and 8765 (WebSocket Server) to prevent conflicts
    free_port(3000)
    free_port(8765)

    # Detect executable names
    python_exe = sys.executable
    npm_cmd = "npm.cmd" if os.name == 'nt' else "npm"

    # Forward any arguments from the root script down to the backend simulation
    backend_args = sys.argv[1:]
    
    # Defaults for backend if no args provided (default to mock-zek for easy startup)
    if not any(arg.startswith("--mock-zek") or arg.startswith("--zek-port") for arg in backend_args):
        backend_args.append("--mock-zek")
    if not any(arg.startswith("--interval") for arg in backend_args):
        backend_args.extend(["--interval", "0.2"])

    banner = """
\033[96m   ╔█████╗      ╔██████╗     ╔██████╗     ╔██████╗ 
   ║██╔═██╗     ║██╔══██╗    ║██╔═══╝     ║██╔═══╝ 
   ║██████║     ║██████╔╝    ║█████╗      ╚█████╗  
   ║██╔═██║ \033[95m╔█╗\033[96m ║██╔══██╗\033[95m╔█╗\033[96m ║██╔══╝ \033[95m╔█╗\033[96m  ╔═══██║ 
   ║██║ ██║ \033[95m╚█╝\033[96m ║██║  ██║\033[95m╚█╝\033[96m ║██████╗\033[95m╚█╝\033[96m  ║██████║ 
   ╚══╝ ╚══╝    ╚══╝  ╚══╝   ╚═════╝      ╚═════╝\033[0m
\033[95m   =================================================\033[0m
\033[96m   [A.R.E.S.] - AUTONOMOUS RADIO EVASION SYSTEM HUD\033[0m
\033[95m   =================================================\033[0m
    """
    try:
        print(banner, flush=True)
    except UnicodeEncodeError:
        # Safe ASCII fallback banner if the terminal doesn't support Unicode characters
        ascii_fallback = """
\033[95m     A. R. E. S.
    [Autonomous Radio Evasion System]\033[0m
\033[96m    =================================\033[0m
        """
        print(ascii_fallback, flush=True)
    print(f"Backend Command: {python_exe} -m simulation.run_hil_loop " + " ".join(backend_args))
    print(f"Frontend Command: {npm_cmd} run dev (in ./dashboard)")
    print("Press Ctrl+C to terminate both servers safely.\n")

    # 1. Start Python Simulation Backend
    backend_proc = subprocess.Popen(
        [python_exe, "-m", "simulation.run_hil_loop"] + backend_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )

    # 2. Start Next.js Frontend Dev Server
    frontend_proc = subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd="dashboard",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0
    )

    # 3. Spin up threads to read stdout and stderr in real-time
    threads = [
        threading.Thread(target=log_stream, args=(backend_proc.stdout, "\033[96m[BACKEND]\033[0m"), daemon=True),
        threading.Thread(target=log_stream, args=(backend_proc.stderr, "\033[91m[BACKEND-ERR]\033[0m"), daemon=True),
        threading.Thread(target=log_stream, args=(frontend_proc.stdout, "\033[95m[FRONTEND]\033[0m"), daemon=True),
        threading.Thread(target=log_stream, args=(frontend_proc.stderr, "\033[91m[FRONTEND-ERR]\033[0m"), daemon=True),
    ]

    for t in threads:
        t.start()

    # 4. Watch processes and wait for exit/interrupt
    try:
        while True:
            # Check if either process died unexpectedly
            if backend_proc.poll() is not None:
                print(f"\n[SYSTEM] Backend process terminated with exit code {backend_proc.returncode}")
                break
            if frontend_proc.poll() is not None:
                print(f"\n[SYSTEM] Frontend process terminated with exit code {frontend_proc.returncode}")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n\n[SYSTEM] Shutdown signal (Ctrl+C) received. Cleaning up processes...")
    finally:
        # 5. Clean up both subprocesses
        print("[SYSTEM] Stopping Backend...")
        backend_proc.terminate()
        print("[SYSTEM] Stopping Frontend...")
        frontend_proc.terminate()
        
        # Give them a moment to terminate, force kill if necessary
        try:
            backend_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print("[SYSTEM] Force-killing Backend...")
            backend_proc.kill()
            
        try:
            frontend_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print("[SYSTEM] Force-killing Frontend...")
            frontend_proc.kill()

        print("=" * 70)
        print("ARES SYSTEM OFFLINE")
        print("=" * 70)

if __name__ == "__main__":
    run_system()
