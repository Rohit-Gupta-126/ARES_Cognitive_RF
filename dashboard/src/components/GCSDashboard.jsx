"use client";

import React, { useEffect, useState, useRef } from "react";

export default function GCSDashboard() {
  const [connected, setConnected] = useState(false);
  const [telemetry, setTelemetry] = useState(null);
  const [waterfallHistory, setWaterfallHistory] = useState([]);
  const [stats, setStats] = useState({
    totalPackets: 0,
    collisions: 0,
    successes: 0,
  });
  
  const wsRef = useRef(null);
  const maxHistoryLength = 15;

  useEffect(() => {
    // Connect to WebSocket server running in run_hil_loop.py
    const connectWS = () => {
      console.log("Connecting to ARES Telemetry WebSocket...");
      const ws = new WebSocket("ws://127.0.0.1:8765");
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        console.log("WebSocket Connection established!");
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setTelemetry(data);

          // Update statistics running totals
          setStats((prev) => {
            const nextSuccesses = prev.successes + (data.success ? 1 : 0);
            const nextCollisions = prev.collisions + (data.success ? 0 : 1);
            return {
              totalPackets: data.step + 1,
              successes: nextSuccesses,
              collisions: nextCollisions,
            };
          });

          // Add to waterfall history (prepend to scroll downwards)
          setWaterfallHistory((prev) => {
            const newRow = {
              step: data.step,
              tx_channel: data.tx_channel,
              jam_channels: data.jam_channels,
              jammed_vector: data.jammed_vector,
            };
            const updated = [newRow, ...prev];
            if (updated.length > maxHistoryLength) {
              return updated.slice(0, maxHistoryLength);
            }
            return updated;
          });
        } catch (err) {
          console.error("Error parsing WebSocket telemetry message:", err);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        console.log("WebSocket Connection closed. Retrying in 3 seconds...");
        setTimeout(connectWS, 3000);
      };

      ws.onerror = (err) => {
        console.error("WebSocket connection encountered error:", err);
        ws.close();
      };
    };

    connectWS();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Helpers for determining waterfall cell classes
  const getCellClass = (row, colIndex) => {
    const isTx = row.tx_channel === colIndex;
    const isJammed = row.jam_channels.includes(colIndex);

    if (isTx && isJammed) return "waterfall-cell collision";
    if (isTx) return "waterfall-cell tx";
    if (isJammed) return "waterfall-cell jammed";
    return "waterfall-cell idle";
  };

  const channelsList = Array.from({ length: 32 }, (_, i) => i);
  const pdr = telemetry ? telemetry.pdr : 100.0;
  const isZekConnected = telemetry ? telemetry.is_zek_connected : false;
  const entropySource = telemetry ? telemetry.entropy_source : "SOFTWARE_ENTROPY";
  const activeJammer = telemetry ? telemetry.active_jammer : "N/A";

  return (
    <>
      {/* 1. HeaderHUD */}
      <header className="gcs-header">
        <div className="gcs-title-group">
          <h1>PROJECT A.R.E.S.</h1>
          <p>Autonomous Radio Evasion System — Ground Control Station</p>
        </div>
        
        <div style={{ display: "flex", gap: "15px", alignItems: "center" }}>
          {/* WebSocket Connection Badge */}
          <span className={`status-badge ${connected ? "online" : "offline"}`}>
            <span style={{
              width: "8px",
              height: "8px",
              borderRadius: "50%",
              backgroundColor: connected ? "var(--color-safe)" : "var(--color-jammed)",
              display: "inline-block",
              boxShadow: connected ? "0 0 8px var(--color-safe)" : "0 0 8px var(--color-jammed)"
            }}></span>
            {connected ? "GCS Online" : "GCS Offline"}
          </span>

          {/* Cryptographic Entropy Trust Badge */}
          {connected && telemetry && (
            <span className={`status-badge ${entropySource === "HARDWARE_ENTROPY" ? "hil-hw" : "hil-sw"}`}>
              {entropySource === "HARDWARE_ENTROPY" ? "⚡ HW-ZEK Cryptographic Trust" : "⚠️ SW-OS Fallback Mode"}
            </span>
          )}
        </div>
      </header>

      {/* 2. Main Dashboard Layout */}
      <main className="dashboard-container">
        
        {/* Warning Alerts Banner */}
        {connected && telemetry && !telemetry.success && (
          <div className="col-span-12" style={{
            background: "rgba(244, 63, 94, 0.15)",
            border: "1px solid var(--color-jammed)",
            borderRadius: "8px",
            padding: "15px",
            textAlign: "center",
            boxShadow: "0 0 15px rgba(244, 63, 94, 0.2)",
            animation: "pulse-red 1.5s infinite"
          }}>
            <h3 style={{ color: "var(--color-jammed)", fontWeight: "bold", fontSize: "1.1rem" }}>
              ⚠️ EW SIGNAL JAMMING COLLISION DETECTED ON CHANNEL {telemetry.tx_channel}
            </h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "4px" }}>
              Adversarial follower jammer successfully intercepted frequency node. Executing adaptive hop.
            </p>
          </div>
        )}

        {connected && telemetry && entropySource === "SOFTWARE_ENTROPY" && (
          <div className="col-span-12" style={{
            background: "rgba(245, 158, 11, 0.15)",
            border: "1px solid var(--color-warning)",
            borderRadius: "8px",
            padding: "15px",
            textAlign: "center",
            boxShadow: "0 0 15px rgba(245, 158, 11, 0.2)",
            animation: "pulse-yellow 2s infinite"
          }}>
            <h3 style={{ color: "var(--color-warning)", fontWeight: "bold", fontSize: "1.1rem" }}>
              ⚠️ ZENTROPY KEY DISCONNECTED — OPERATIONAL RESILIENCY FALLBACK
            </h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "4px" }}>
              Hardware cryptographic root of trust un-available. Seeding frequency hopper with cryptographically secure software entropy (`os.urandom`).
            </p>
          </div>
        )}

        {/* Column 1: Left Stats and Spectrum Waterfall (Span 8) */}
        <section className="col-span-8" style={{ display: "flex", flexDirection: "column", gap: "30px" }}>
          
          {/* Metrics Cards row */}
          <div className="metrics-row">
            <div className="cyber-panel metric-card">
              <div className="metric-label">Packet Delivery Ratio</div>
              <div className="metric-value" style={{ 
                color: pdr >= 90 ? "var(--color-safe)" : pdr >= 75 ? "var(--color-warning)" : "var(--color-jammed)" 
              }}>
                {pdr.toFixed(1)}%
              </div>
              <div className="metric-subtext">HIL Evasion Success Rate</div>
            </div>

            <div className="cyber-panel metric-card">
              <div className="metric-label">Active EW Jammer</div>
              <div className="metric-value" style={{ 
                color: activeJammer === "sweep" ? "var(--color-tx)" : 
                       activeJammer === "barrage" ? "var(--color-collision)" : 
                       activeJammer === "follower" ? "var(--color-jammed)" : "var(--text-muted)",
                textTransform: "uppercase",
                fontSize: "1.5rem"
              }}>
                {activeJammer}
              </div>
              <div className="metric-subtext">Threat Jamming Strategy</div>
            </div>

            <div className="cyber-panel metric-card">
              <div className="metric-label">Link Statistics</div>
              <div className="metric-value">
                {stats.successes} / {stats.totalPackets}
              </div>
              <div className="metric-subtext">Successful Hops vs Attempts</div>
            </div>

            <div className="cyber-panel metric-card">
              <div className="metric-label">Collisions Avoided</div>
              <div className="metric-value" style={{ color: stats.collisions > 0 ? "var(--color-warning)" : "var(--color-safe)" }}>
                {stats.collisions}
              </div>
              <div className="metric-subtext">Total Intercepted Packets</div>
            </div>
          </div>

          {/* Real-time Spectrum Waterfall Card */}
          <div className="cyber-panel">
            <h2 style={{ fontSize: "1.2rem", fontWeight: "bold", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "10px" }}>
              ⚡ Scrolling 32-Channel RF Waterfall
            </h2>
            
            <div className="waterfall-grid-container">
              {/* Channel Numbers Header */}
              <div className="waterfall-row-container" style={{ marginBottom: "6px" }}>
                <div className="waterfall-row-label">Channel</div>
                <div className="waterfall-grid">
                  {channelsList.map((ch) => (
                    <div key={ch} className="waterfall-header-cell">{ch}</div>
                  ))}
                </div>
              </div>

              {/* Waterfall History Rows */}
              {waterfallHistory.length === 0 ? (
                <div style={{ padding: "40px 0", textAlign: "center", color: "var(--text-muted)", fontFamily: "Share Tech Mono, monospace" }}>
                  WAITING FOR SIMULATION STREAM...
                </div>
              ) : (
                waterfallHistory.map((row) => (
                  <div key={row.step} className="waterfall-row-container">
                    <div className="waterfall-row-label">T-{telemetry.step - row.step}</div>
                    <div className="waterfall-grid">
                      {channelsList.map((ch) => (
                        <div
                          key={ch}
                          className={getCellClass(row, ch)}
                          title={`Step: ${row.step}, Channel: ${ch}`}
                        />
                      ))}
                    </div>
                  </div>
                ))
              )}
            </div>

            <div style={{ display: "flex", gap: "25px", marginTop: "20px", fontSize: "0.8rem", color: "var(--text-muted)", justifyContent: "center" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ width: "12px", height: "12px", borderRadius: "2px", backgroundColor: "var(--color-tx)" }}></span>
                <span>Transmitter Active (TX)</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ width: "12px", height: "12px", borderRadius: "2px", backgroundColor: "var(--color-jammed)" }}></span>
                <span>EW Jammer Active</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ width: "12px", height: "12px", borderRadius: "2px", backgroundColor: "var(--color-collision)" }}></span>
                <span>Packet Collision</span>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ width: "12px", height: "12px", borderRadius: "2px", backgroundColor: "#0f172a", border: "1px solid rgba(255,255,255,0.05)" }}></span>
                <span>Idle Channel</span>
              </div>
            </div>
          </div>
        </section>

        {/* Column 2: Right AI Predictive Brain (Span 4) */}
        <section className="col-span-4" style={{ display: "flex", flexDirection: "column", gap: "30px" }}>
          
          <div className="cyber-panel" style={{ height: "100%" }}>
            <h2 style={{ fontSize: "1.2rem", fontWeight: "bold", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "10px" }}>
              🧠 GRU Channel Jamming Probabilities
            </h2>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "6px" }}>
              AI Model predictions $P(Jam)$ for the next step. Node targets with values $P \ge 0.5$ are avoided.
            </p>

            {telemetry ? (
              <div className="prob-bars-grid">
                {/* 50% Threshold line Overlay */}
                <div className="threshold-line-overlay" style={{ bottom: "50%" }}>
                  <div className="threshold-line-overlay-label">Evasion Threshold (0.50)</div>
                </div>

                {channelsList.map((ch) => {
                  const prob = telemetry.prediction_probs[ch];
                  const isJammed = prob >= 0.5;
                  const isTx = telemetry.tx_channel === ch;
                  
                  let barClass = "prob-bar-vertical safe";
                  if (isTx) barClass = "prob-bar-vertical tx-active";
                  else if (isJammed) barClass = "prob-bar-vertical jammed";

                  return (
                    <div key={ch} className="prob-bar-col">
                      <div
                        className={barClass}
                        style={{ height: `${prob * 100}%` }}
                        title={`Channel ${ch}: ${(prob * 100).toFixed(1)}% probability`}
                      />
                      <span className="prob-bar-axis-label">{ch}</span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div style={{ padding: "80px 0", textAlign: "center", color: "var(--text-muted)", fontFamily: "Share Tech Mono, monospace" }}>
                AWAITING METRICS...
              </div>
            )}

            <div style={{ marginTop: "25px", background: "rgba(2, 6, 23, 0.4)", borderRadius: "8px", padding: "12px", border: "1px solid rgba(255,255,255,0.03)" }}>
              <h4 style={{ fontSize: "0.85rem", fontWeight: "bold", fontFamily: "Share Tech Mono, monospace", color: "var(--color-tx)" }}>
                SYSTEM DECISION LOG:
              </h4>
              <div style={{ fontSize: "0.75rem", fontFamily: "Share Tech Mono, monospace", color: "var(--text-muted)", marginTop: "8px", display: "flex", flexDirection: "column", gap: "6px" }}>
                <div>&gt; Brain identified {telemetry ? telemetry.safe_channels.length : 32} safe frequency nodes.</div>
                <div>&gt; ZEK seed squeezed for channel selection.</div>
                <div>
                  &gt; Hopped to node: <span style={{ color: "var(--color-tx)", fontWeight: "bold" }}>{telemetry ? telemetry.tx_channel : "N/A"}</span>.
                </div>
                <div>&gt; Channel status: {telemetry ? (telemetry.success ? "CLEAR - TRANSMISSION OK" : "JAMMED - COLLISION") : "STANDBY"}</div>
              </div>
            </div>
          </div>
        </section>

      </main>
    </>
  );
}
