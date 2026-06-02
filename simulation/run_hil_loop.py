import sys
import os

# Reconfigure console output to UTF-8 on Windows to handle emojis/Unicode logging
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import torch
import argparse
import time
import json
import asyncio
import websockets
import threading

# Adjust path to import from root workspace directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.ether import RFEther
from simulation.jammer import SweepJammer, BarrageJammer, FollowerJammer, RandomJammer
from hardware_bridge.hardware_bridge import ZenTropyBridge
from models.model import JammerPredictorGRU
from models.dqn_agent import DQNAgent

# Global WebSocket variables
connected_clients = set()
ws_loop = None
ws_thread = None

def start_ws_server(host="localhost", port=8765):
    """Runs the WebSockets event loop in a background thread."""
    global ws_loop
    
    async def ws_handler(websocket, *args):
        connected_clients.add(websocket)
        logging_print(f"WebSocket Client connected from {websocket.remote_address}")
        try:
            await websocket.wait_closed()
        finally:
            if websocket in connected_clients:
                connected_clients.remove(websocket)
            logging_print(f"WebSocket Client disconnected: {websocket.remote_address}")

    async def main():
        global ws_loop
        ws_loop = asyncio.get_running_loop()
        async with websockets.serve(ws_handler, host, port):
            logging_print(f"WebSocket Server running on ws://{host}:{port}")
            await asyncio.Future()  # run forever

    try:
        asyncio.run(main())
    except Exception as e:
        logging_print(f"WebSocket Server stopped or encountered error: {e}")

def broadcast_telemetry(data):
    """Sends telemetry data to all connected WebSocket clients."""
    if not connected_clients or ws_loop is None:
        return
    message = json.dumps(data)
    
    async def send_all():
        if connected_clients:
            await asyncio.gather(
                *[client.send(message) for client in connected_clients],
                return_exceptions=True
            )
            
    asyncio.run_coroutine_threadsafe(send_all(), ws_loop)

def logging_print(msg):
    """Thread-safe print helper with fallback for Windows Unicode encodings."""
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        safe_msg = msg.replace("✅", "[SUCCESS]").replace("❌", "[COLLISION]").replace("⚡", "[HIL]")
        print(safe_msg, flush=True)

def run_evasion_loop(args):
    global ws_thread, ws_loop
    
    # 1. Initialize WebSocket Server
    logging_print("Starting WebSocket Server thread...")
    ws_thread = threading.Thread(target=start_ws_server, args=(args.host, args.port), daemon=True)
    ws_thread.start()
    time.sleep(0.5)

    # 2. Select and load AI agent
    num_channels = 32
    seq_len = 10
    # CUDA auto-detection: use RTX 2050 / any GPU for inference if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging_print(f"[AI] Inference device: {device}")

    dqn_agent = None
    gru_model  = None

    # ── Primary: DQN brain (Stage 5) ──────────────────────────────────────
    if args.dqn_model and os.path.exists(args.dqn_model):
        logging_print(f"[AI] Loading DQN brain from '{args.dqn_model}'...")
        dqn_agent = DQNAgent(state_dim=seq_len * num_channels + 1, num_actions=num_channels)
        dqn_agent.load(args.dqn_model, inference_only=True)
        agent_mode = "DQN+ZEK"
        logging_print(f"[AI] DQN brain loaded — Hybrid RL-HIL engine active (top_k={args.top_k}).")
    elif args.dqn_model:
        logging_print(f"[AI] WARNING: DQN model not found at '{args.dqn_model}'. Train first: python -m models.train_rl")

    # ── Optional: GRU brain (Stage 3, kept for A/B comparison) ────────────
    if os.path.exists(args.model_path):
        logging_print(f"[AI] Loading GRU brain from '{args.model_path}'...")
        gru_model = JammerPredictorGRU(input_dim=num_channels, hidden_dim=64, num_layers=2, output_dim=num_channels)
        gru_model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        gru_model.eval()
        if dqn_agent is None:
            agent_mode = "GRU+ZEK"
        logging_print("[AI] GRU brain loaded (fallback / A-B comparison mode).")

    # Abort if neither model loaded
    if dqn_agent is None and gru_model is None:
        logging_print("ERROR: No AI model available. Provide --dqn-model or ensure models/best_brain.pth exists.")
        return

    if dqn_agent is None and gru_model is not None:
        agent_mode = "GRU+ZEK"

    # 3. Initialize HIL ZenTropy Bridge
    logging_print("Initializing ZenTropy Key Bridge...")
    bridge = ZenTropyBridge(port=args.zek_port, mock=args.mock_zek)

    # 4. Initialize Simulation Ether and Jammers
    ether = RFEther(num_channels=num_channels)
    jammers = {
        "sweep":    SweepJammer(num_channels, step_size=1, jam_width=1, randomness_prob=0.1),
        "barrage":  BarrageJammer(num_channels, block_size=5, shift_interval=20),
        "follower": FollowerJammer(num_channels),
        "random":   RandomJammer(num_channels, num_to_jam=2),
    }

    # 5. Initialize loop states
    history    = np.zeros((seq_len, num_channels), dtype=np.float32)
    tx_history = []
    tx_channel = 0           # current TX channel (initialised for DQN state on step 0)

    total_packets     = 0
    successful_packets = 0

    active_jammer_type = args.initial_jammer
    jammer = jammers[active_jammer_type]
    
    logging_print(f"\nEntering Cognitive RF Evasion Loop (Interval: {args.interval}s, Jammer: {active_jammer_type})...")
    logging_print("Press Ctrl+C to terminate.")
    
    step = 0
    try:
        while True:
            # A. Dynamic Jammer Cycling if requested
            if args.dynamic_jammer and step > 0 and step % args.cycle_interval == 0:
                available_types = list(jammers.keys())
                current_idx = available_types.index(active_jammer_type)
                active_jammer_type = available_types[(current_idx + 1) % len(available_types)]
                jammer = jammers[active_jammer_type]
                logging_print(f"\n[EVENT] Cycling EW Jammer strategy to: {active_jammer_type.upper()}")
            
            # B. Build flat DQN state: (10×32) history + normalised tx_channel
            dqn_state = np.concatenate([
                history.flatten(),
                [tx_channel / max(num_channels - 1, 1)]
            ]).astype(np.float32)

            # C. ── Hybrid RL-HIL Decision Engine ──────────────────────────
            q_values       = np.zeros(num_channels, dtype=np.float32)  # raw Q-values
            top_k_channels = list(range(args.top_k))                   # safe-zone default
            probs          = np.zeros(num_channels, dtype=np.float32)  # GRU probs (or Q norm)

            if dqn_agent is not None:
                # Stage 5: DQN outputs Q-values for all 32 channels
                q_values = dqn_agent.get_q_values(dqn_state)           # shape (32,)

                # Top-K Safe Zone: highest Q-value channels → feed to ZEK
                top_k_channels = dqn_agent.get_top_k_channels(dqn_state, k=args.top_k)

                # Normalise Q-values to [0,1] for dashboard display
                q_min, q_max = q_values.min(), q_values.max()
                q_range = q_max - q_min if q_max != q_min else 1.0
                probs = ((q_values - q_min) / q_range).astype(np.float32)

                safe_channels = top_k_channels  # ZEK picks from top-K

            elif gru_model is not None:
                # Stage 3 fallback: GRU predicts P(Jam) per channel
                input_tensor = torch.tensor(history, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = gru_model(input_tensor)
                    probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()

                safe_channels = np.where(probs < args.threshold)[0].tolist()
                if not safe_channels:
                    safe_channels = np.argsort(probs)[:3].tolist()
                top_k_channels = safe_channels

            # D. Get EW Jammer targets (1-step delay — correct causal order)
            jam_channels = jammer.get_jammed_channels(step, tx_history)

            # E. Query ZEK for cryptographic channel selection from the Safe Zone
            tx_channel = bridge.get_random_channel(safe_channels)
            tx_history.append(tx_channel)
            if len(tx_history) > 10:
                tx_history.pop(0)

            # F. Step the RF Ether environment
            result = ether.step(tx_channel, jam_channels)

            # G. Update Stats
            total_packets += 1
            if result['success']:
                successful_packets += 1
            pdr = (successful_packets / total_packets) * 100.0

            # H. Update Sliding History
            history = np.roll(history, -1, axis=0)
            history[-1] = result['jammed_vector']

            # I. Formulate Telemetry Payload
            telemetry = {
                "step":             step,
                "channel_states":   result['channel_states'].tolist(),
                "jammed_vector":    result['jammed_vector'].tolist(),
                "prediction_probs": probs.tolist(),       # normalised Q-values or GRU probs
                "q_values":         q_values.tolist(),    # raw DQN Q-values (zeros in GRU mode)
                "top_k_channels":   top_k_channels,       # Safe Zone indices
                "safe_channels":    safe_channels,
                "tx_channel":       tx_channel,
                "jam_channels":     jam_channels,
                "success":          bool(result['success']),
                "pdr":              round(pdr, 2),
                "entropy_source":   bridge.entropy_source,
                "is_zek_connected": bridge.is_connected,
                "active_jammer":    active_jammer_type,
                "agent_mode":       agent_mode,
            }
            
            # J. Broadcast and print telemetry
            broadcast_telemetry(telemetry)
            
            # Console logger (prints status summary)
            status_char = "✅" if result['success'] else "❌ COLLISION"
            entropy_tag = "[HW-ZEK]" if bridge.entropy_source == "HARDWARE_ENTROPY" else "[SW-OS]"
            logging_print(
                f"Step {step:04d} | TX: {tx_channel:2d} | Jammed: {str(jam_channels):15s} | "
                f"PDR: {pdr:5.1f}% | {status_char} | {entropy_tag} | [{agent_mode}] top-{args.top_k}: {top_k_channels}"
            )
            
            # K. Sleep
            time.sleep(args.interval)
            step += 1
            
    except KeyboardInterrupt:
        logging_print("\nTerminating HIL loop by user request.")
    finally:
        # Cleanup
        logging_print("Closing serial connections and stopping WebSocket server...")
        bridge.close()
        if ws_loop:
            ws_loop.call_soon_threadsafe(ws_loop.stop)
        logging_print("Cleanup completed. ARES system offline.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARES Cognitive RF Evasion HIL Simulation Loop")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket server port")
    parser.add_argument("--dqn-model",  type=str, default="models/dqn_brain.pth",
                        help="Path to trained DQN checkpoint (Stage 5 primary engine)")
    parser.add_argument("--top-k",       type=int, default=5,
                        help="Safe-Zone size: top-K Q-value channels fed to ZEK for cryptographic selection")
    parser.add_argument("--model-path",  type=str, default="models/best_brain.pth",
                        help="Path to trained GRU model (optional A/B comparison fallback)")
    parser.add_argument("--zek-port",    type=str, default=None,
                        help="Specific COM port for ZEK serial interface")
    parser.add_argument("--mock-zek",    action="store_true",
                        help="Explicitly force mock ZEK (software entropy only)")
    parser.add_argument("--interval",    type=float, default=0.2,
                        help="Simulation step interval in seconds")
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="GRU probability threshold for safe channels (GRU mode only)")
    parser.add_argument("--initial-jammer", type=str,
                        choices=["sweep", "barrage", "follower", "random"], default="sweep",
                        help="Initial EW Jammer type")
    parser.add_argument("--dynamic-jammer", action="store_true", default=True,
                        help="Automatically cycle through jammer strategies")
    parser.add_argument("--cycle-interval", type=int, default=50,
                        help="Steps between jammer strategy rotations")
    
    args = parser.parse_args()
    
    # If explicitly forcing mock mode, log it
    if args.mock_zek:
        logging_print("Forcing ZEK mock mode (Software Entropy fallback).")
        
    run_evasion_loop(args)
