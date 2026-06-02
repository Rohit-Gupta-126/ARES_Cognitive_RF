import "./globals.css";

export const metadata = {
  title: "Project A.R.E.S. — Ground Control Station",
  description:
    "Autonomous Radio Evasion System — Cognitive RF Evasion HIL Simulation Telemetry Dashboard",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
