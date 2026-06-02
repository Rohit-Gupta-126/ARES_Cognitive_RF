"use client";

import React, { useEffect, useState, useRef, useCallback } from "react";

/* ─── Helpers ─────────────────────────────────────────────── */
const CHANNELS = Array.from({ length: 32 }, (_, i) => i);
const MAX_HISTORY = 18;

function PDRGauge({ pdr }) {
  const r = 24;
  const circ = 2 * Math.PI * r;
  const dash = (pdr / 100) * circ;
  const color = pdr >= 90 ? "#34d399" : pdr >= 70 ? "#fbbf24" : "#fb7185";

  return (
    <div className="pdr-gauge-ring">
      <svg width="60" height="60" viewBox="0 0 60 60">
        <circle cx="30" cy="30" r={r} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="5" />
        <circle
          cx="30" cy="30" r={r}
          fill="none"
          stroke={color}
          strokeWidth="5"
          strokeDasharray={`${dash} ${circ}`}
          strokeLinecap="round"
          style={{ filter: `drop-shadow(0 0 4px ${color})`, transition: "stroke-dasharray 0.4s ease" }}
        />
      </svg>
      <div className="pdr-gauge-value" style={{ color }}>{pdr.toFixed(0)}%</div>
    </div>
  );
}

function JammerBadge({ type }) {
  if (!type || type === "N/A") return <span style={{ color: "var(--text-tertiary)", fontFamily: "JetBrains Mono, monospace", fontSize: "0.8rem" }}>—</span>;
  return <span className={`jammer-badge jammer-${type}`}>{type.toUpperCase()}</span>;
}

function MetricCard({ label, value, subtext, tag, accentColor, accentBg, accentBorder }) {
  return (
    <div
      className="metric-card"
      style={{
        "--card-accent": accentColor,
        "--card-color": accentColor,
        "--card-tag-bg": accentBg,
      }}
    >
      <div className="metric-card-label">{label}</div>
      <div className="metric-card-value">{value}</div>
      <div className="metric-card-footer">
        <div className="metric-card-subtext">{subtext}</div>
        {tag && <div className="metric-card-tag" style={{ background: accentBg, color: accentColor, borderColor: accentBorder }}>{tag}</div>}
      </div>
    </div>
  );
}

/* ─── Main Component ─────────────────────────────────────── */
export default function GCSDashboard() {
  const [connected, setConnected] = useState(false);
  const [telemetry, setTelemetry] = useState(null);
  const [history, setHistory] = useState([]);
  const [stats, setStats] = useState({ total: 0, success: 0, collisions: 0 });
  const wsRef = useRef(null);
  const reconnectTimer = useRef(null);
  const connectWSRef = useRef(null);

  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket("ws://127.0.0.1:8765");
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        setTelemetry(data);

        setStats((prev) => ({
          total: data.step + 1,
          success: prev.success + (data.success ? 1 : 0),
          collisions: prev.collisions + (data.success ? 0 : 1),
        }));

        setHistory((prev) => {
          const next = [
            { step: data.step, tx: data.tx_channel, jam: data.jam_channels, success: data.success },
            ...prev,
          ];
          return next.slice(0, MAX_HISTORY);
        });
      } catch {}
    };

    // Use ref to avoid accessing connectWS before it is fully declared
    ws.onclose = () => {
      setConnected(false);
      reconnectTimer.current = setTimeout(() => connectWSRef.current?.(), 3000);
    };

    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    // Sync ref after render so onclose always calls the latest version
    connectWSRef.current = connectWS;
    connectWS();
    return () => {
      wsRef.current?.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
    };
  }, [connectWS]);

  /* ── Derived State ── */
  const pdr = telemetry?.pdr ?? 100.0;
  const isHW = telemetry?.entropy_source === "HARDWARE_ENTROPY";
  const activeJammer = telemetry?.active_jammer ?? "N/A";
  const safeChannelCount = telemetry?.safe_channels?.length ?? 32;
  const currentStep = telemetry?.step ?? 0;
  const txChannel = telemetry?.tx_channel;
  const lastSuccess = telemetry?.success;
  const agentMode = telemetry?.agent_mode ?? "GRU+ZEK";
  const isDQN = agentMode === "DQN+ZEK";
  const topKChannels = new Set(telemetry?.top_k_channels ?? []);
  const qValues = telemetry?.q_values ?? null;

  function getCellClass(row, ch) {
    const isTx = row.tx === ch;
    const isJam = row.jam.includes(ch);
    if (isTx && isJam) return "wf-cell wf-collision";
    if (isTx) return "wf-cell wf-tx";
    if (isJam) return "wf-cell wf-jammed";
    return "wf-cell wf-idle";
  }

  return (
    <div className="app-shell">

      {/* ── Top Nav ── */}
      <nav className="top-nav">
        <div className="nav-brand">
          <div className="nav-logo-mark">AR</div>
          <div>
            <div className="nav-title">Project A.R.E.S.</div>
            <div className="nav-subtitle">Autonomous Radio Evasion System — GCS</div>
          </div>
        </div>

        <div className="nav-center">
          <div className="nav-center-dot" />
          <span>LIVE TELEMETRY</span>
          {telemetry && (
            <>
              <span style={{ color: "var(--border)" }}>•</span>
              <span>STEP {String(currentStep).padStart(5, "0")}</span>
            </>
          )}
        </div>

        <div className="nav-badges">
          <span className={`badge ${connected ? "badge-online" : "badge-offline"}`}>
            <span className="badge-dot" />
            {connected ? "GCS Online" : "Disconnected"}
          </span>
          {connected && telemetry && (
            <span className={`badge ${isDQN ? "badge-hw" : "badge-sw"}`}>
              <span className="badge-dot" />
              {isDQN ? "DQN · RL Engine" : "GRU · Pattern Engine"}
            </span>
          )}
          {connected && telemetry && (
            <span className={`badge ${isHW ? "badge-hw" : "badge-sw"}`}>
              <span className="badge-dot" />
              {isHW ? "HW-ZEK Trust" : "SW-OS Fallback"}
            </span>
          )}
        </div>
      </nav>

      {/* ── Alert Banners ── */}
      {connected && telemetry && !telemetry.success && (
        <div className="alert-banner danger">
          <span className="alert-banner-icon">⚡</span>
          <div className="alert-banner-content">
            <strong>EW COLLISION DETECTED — Channel {txChannel}</strong>
            <p>Adversarial jammer intercepted transmission. Adaptive frequency hop initiated.</p>
          </div>
        </div>
      )}
      {connected && telemetry && !isHW && (
        <div className="alert-banner warning">
          <span className="alert-banner-icon">⚠</span>
          <div className="alert-banner-content">
            <strong>ZENTROPY KEY DISCONNECTED — Software Entropy Mode Active</strong>
            <p>Cryptographic root of trust unavailable. Seeding with os.urandom() fallback.</p>
          </div>
        </div>
      )}

      {/* ── Main ── */}
      <main className="main-content">

        {/* ── Metric Cards Row ── */}
        <div className="grid-metrics">
          <MetricCard
            label="Packet Delivery Ratio"
            value={`${pdr.toFixed(1)}%`}
            subtext="HIL evasion success rate"
            tag={pdr >= 90 ? "NOMINAL" : pdr >= 70 ? "DEGRADED" : "CRITICAL"}
            accentColor={pdr >= 90 ? "var(--safe)" : pdr >= 70 ? "var(--warning)" : "var(--danger)"}
            accentBg={pdr >= 90 ? "var(--safe-dim)" : pdr >= 70 ? "var(--warning-dim)" : "var(--danger-dim)"}
            accentBorder={pdr >= 90 ? "rgba(52,211,153,0.2)" : pdr >= 70 ? "rgba(251,191,36,0.2)" : "rgba(251,113,133,0.2)"}
          />

          <div
            className="metric-card"
            style={{ "--card-accent": "var(--tx)", "--card-color": "var(--tx)" }}
          >
            <div className="metric-card-label">Active EW Jammer</div>
            <div style={{ marginTop: 8 }}>
              <JammerBadge type={activeJammer} />
            </div>
            <div className="metric-card-footer">
              <div className="metric-card-subtext">Current threat strategy</div>
            </div>
          </div>

          <MetricCard
            label="Link Statistics"
            value={`${stats.success} / ${stats.total}`}
            subtext="Successful hops vs attempts"
            tag="LIVE"
            accentColor="var(--cyan)"
            accentBg="var(--cyan-dim)"
            accentBorder="rgba(34,211,238,0.2)"
          />

          <MetricCard
            label="Safe Channels"
            value={safeChannelCount}
            subtext={`${32 - safeChannelCount} channels blocked by AI`}
            tag={`${stats.collisions} COLLISIONS`}
            accentColor={stats.collisions > 5 ? "var(--warning)" : "var(--safe)"}
            accentBg={stats.collisions > 5 ? "var(--warning-dim)" : "var(--safe-dim)"}
            accentBorder={stats.collisions > 5 ? "rgba(251,191,36,0.2)" : "rgba(52,211,153,0.2)"}
          />
        </div>

        {/* ── Main Grid: Waterfall + Sidebar ── */}
        <div className="grid-main">

          {/* ── Left: Waterfall ── */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">
                <span className="panel-title-icon">▦</span>
                32-Channel RF Spectrum Waterfall
              </div>
              {telemetry && (
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span className="step-tag">T-0 = Step {currentStep}</span>
                  <span
                    style={{
                      width: 8, height: 8, borderRadius: "50%",
                      background: lastSuccess ? "var(--safe)" : "var(--danger)",
                      boxShadow: `0 0 8px ${lastSuccess ? "var(--safe)" : "var(--danger)"}`,
                    }}
                  />
                </div>
              )}
            </div>

            <div className="panel-body">
              {history.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-state-icon">📡</div>
                  Awaiting telemetry stream…
                </div>
              ) : (
                <div className="waterfall-wrapper">
                  <div className="waterfall-inner">
                    {/* Channel Header */}
                    <div className="waterfall-header">
                      <div className="waterfall-header-spacer" />
                      <div className="waterfall-header-cells">
                        {CHANNELS.map((ch) => (
                          <div key={ch} className="waterfall-header-cell">{ch}</div>
                        ))}
                      </div>
                    </div>

                    {/* Rows */}
                    <div className="waterfall-rows">
                      {history.map((row, idx) => (
                        <div key={row.step} className="waterfall-row">
                          <div className={`waterfall-row-label ${idx === 0 ? "current" : ""}`}>
                            {idx === 0 ? "NOW" : `T-${idx}`}
                          </div>
                          <div className="waterfall-cells">
                            {CHANNELS.map((ch) => (
                              <div
                                key={ch}
                                className={getCellClass(row, ch)}
                                title={`Step ${row.step} | Ch ${ch}${row.tx === ch ? " [TX]" : ""}${row.jam.includes(ch) ? " [JAM]" : ""}`}
                              />
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>

                    {/* Legend */}
                    <div className="waterfall-legend">
                      <div className="legend-item">
                        <div className="legend-swatch" style={{ background: "var(--tx)", boxShadow: "0 0 6px var(--tx-glow)" }} />
                        Transmitter Active (TX)
                      </div>
                      <div className="legend-item">
                        <div className="legend-swatch" style={{ background: "var(--danger)", boxShadow: "0 0 6px var(--danger-glow)" }} />
                        EW Jammer Active
                      </div>
                      <div className="legend-item">
                        <div className="legend-swatch" style={{ background: "var(--collision)", boxShadow: "0 0 6px var(--collision-glow)" }} />
                        Packet Collision
                      </div>
                      <div className="legend-item">
                        <div className="legend-swatch" style={{ background: "rgba(255,255,255,0.04)", border: "1px solid var(--border)" }} />
                        Idle Channel
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* ── Right Sidebar ── */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

            {/* PDR + Step Summary */}
            <div className="panel">
              <div className="panel-header">
                <div className="panel-title">
                  <span className="panel-title-icon">◎</span>
                  PDR Gauge
                </div>
              </div>
              <div className="panel-body" style={{ display: "flex", alignItems: "center", gap: 20 }}>
                <PDRGauge pdr={pdr} />
                <div>
                  <div style={{ fontSize: "1.5rem", fontWeight: 800, color: pdr >= 90 ? "var(--safe)" : pdr >= 70 ? "var(--warning)" : "var(--danger)", letterSpacing: "-0.5px" }}>
                    {pdr.toFixed(2)}%
                  </div>
                  <div style={{ fontSize: "0.72rem", color: "var(--text-secondary)", marginTop: 2 }}>Packet Delivery Ratio</div>
                  <div style={{ fontSize: "0.68rem", color: "var(--text-tertiary)", marginTop: 6, fontFamily: "JetBrains Mono, monospace" }}>
                    {stats.success} OK · {stats.collisions} FAIL · {stats.total} TOTAL
                  </div>
                </div>
              </div>
            </div>

            {/* AI Decision Engine Panel — DQN Q-Value heatmap OR GRU P(Jam) bars */}
            <div className="panel" style={{ flex: 1 }}>
              <div className="panel-header">
                <div className="panel-title">
                  <span className="panel-title-icon">◈</span>
                  {isDQN ? "DQN Q-Value Heatmap" : "GRU Jamming Probability"}
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{
                    fontFamily: "JetBrains Mono, monospace",
                    fontSize: "0.62rem",
                    padding: "2px 7px",
                    borderRadius: 4,
                    background: isDQN ? "rgba(34,211,238,0.1)" : "rgba(167,139,250,0.1)",
                    color: isDQN ? "var(--cyan)" : "var(--violet)",
                    border: `1px solid ${isDQN ? "rgba(34,211,238,0.2)" : "rgba(167,139,250,0.2)"}`,
                  }}>
                    {isDQN ? `TOP-${topKChannels.size} SAFE ZONE` : "P(JAM) / CH"}
                  </span>
                </div>
              </div>

              <div className="panel-body">
                {!telemetry ? (
                  <div className="empty-state">
                    <div className="empty-state-icon">🧠</div>
                    Awaiting AI predictions…
                  </div>
                ) : (
                  <>
                    <div className="prob-chart">
                      {/* Threshold / reference line */}
                      <div className="prob-threshold-line" style={{ bottom: "50%" }}>
                        <div className="prob-threshold-label">
                          {isDQN ? "Q-value (norm 0.5)" : "0.50 threshold"}
                        </div>
                      </div>
                      <div className="prob-chart-base" />

                      {CHANNELS.map((ch) => {
                        const dispVal = telemetry.prediction_probs[ch]; // normalised 0-1
                        const isTx      = ch === txChannel;
                        const isTopK    = topKChannels.has(ch);
                        const isDangerous = !isDQN && dispVal >= 0.5;

                        // DQN mode: cyan = Safe Zone, blue = TX, grey = out-of-zone
                        // GRU mode: green = safe, red = predicted jammed, blue = TX
                        let barColor, barShadow;
                        if (isTx) {
                          barColor = "var(--tx)";
                          barShadow = "0 0 8px var(--tx-glow)";
                        } else if (isDQN && isTopK) {
                          barColor = "var(--cyan)";
                          barShadow = "0 0 8px var(--cyan-glow)";
                        } else if (isDQN) {
                          barColor = "rgba(255,255,255,0.08)";
                          barShadow = "none";
                        } else if (isDangerous) {
                          barColor = "var(--danger)";
                          barShadow = "0 0 6px var(--danger-glow)";
                        } else {
                          barColor = "var(--safe)";
                          barShadow = "none";
                        }

                        const tooltipText = isDQN
                          ? `Ch ${ch}: Q=${qValues ? qValues[ch].toFixed(2) : "?"} ${isTopK ? "[SAFE ZONE]" : ""}${isTx ? " [TX]" : ""}`
                          : `Ch ${ch}: P(Jam)=${(dispVal * 100).toFixed(1)}%${isTx ? " [TX]" : ""}`;

                        return (
                          <div key={ch} className="prob-col" title={tooltipText}>
                            <div
                              style={{
                                width: "100%",
                                height: `${Math.max(dispVal * 100, 2)}%`,
                                borderRadius: "2px 2px 0 0",
                                transition: "height 0.18s ease, background 0.18s ease",
                                background: barColor,
                                boxShadow: barShadow,
                                minHeight: 2,
                              }}
                            />
                            {ch % 8 === 0 && <span className="prob-axis-label">{ch}</span>}
                          </div>
                        );
                      })}
                    </div>

                    {/* Legend — adapts to agent mode */}
                    <div style={{ display: "flex", gap: 12, marginTop: 10, flexWrap: "wrap" }}>
                      {isDQN ? [
                        { color: "var(--cyan)",   label: `Safe Zone (top-${topKChannels.size})` },
                        { color: "var(--tx)",     label: "TX Channel" },
                        { color: "rgba(255,255,255,0.08)", label: "Out of Zone", border: "1px solid rgba(255,255,255,0.06)" },
                      ] : [
                        { color: "var(--safe)",   label: "Safe" },
                        { color: "var(--danger)", label: "Jammed" },
                        { color: "var(--tx)",     label: "TX Channel" },
                      ].map(({ color, label, border }) => (
                        <div key={label} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: "0.68rem", color: "var(--text-secondary)" }}>
                          <div style={{ width: 8, height: 8, borderRadius: 2, background: color, border: border ?? "none", flexShrink: 0 }} />
                          {label}
                        </div>
                      ))}
                    </div>

                    {/* Decision Log */}
                    <div className="decision-log">
                      <div className="decision-log-title">System Decision Log</div>
                      <div className="log-line">
                        <span className="log-prompt">›</span>
                        <span>
                          {isDQN
                            ? <>{"Agent computed "}<span className="log-value">{topKChannels.size}-channel Safe Zone</span></>  
                            : <>{"Brain identified "}<span className="log-value">{safeChannelCount}</span>{" safe channels"}</>}
                        </span>
                      </div>
                      <div className="log-line">
                        <span className="log-prompt">›</span>
                        <span>ZEK SHAKE-128 selected from Safe Zone</span>
                      </div>
                      <div className="log-line">
                        <span className="log-prompt">›</span>
                        <span>Hopped to node <span className="log-value">CH-{txChannel ?? "—"}</span></span>
                      </div>
                      <div className={`log-line ${lastSuccess ? "log-success" : "log-error"}`}>
                        <span className="log-prompt">›</span>
                        <span>Status: <span className="log-value">{lastSuccess ? "CLEAR — TX OK" : "JAMMED — COLLISION"}</span></span>
                      </div>
                      <div className="log-line">
                        <span className="log-prompt">›</span>
                        <span>Engine: <span className="log-value">{agentMode}</span></span>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
