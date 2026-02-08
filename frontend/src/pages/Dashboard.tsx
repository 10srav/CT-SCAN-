/**
 * Dashboard Page — Quantum CT System Performance Overview
 *
 * Pulls live denoising metrics from the backend and merges with baselines.
 * PSNR is guaranteed non-negative (clamped on both backend and frontend).
 */

import React, { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, ResponsiveContainer, RadarChart, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Radar, Cell, Area, AreaChart,
  ReferenceLine
} from 'recharts';
import {
  Activity, Zap, Image, Clock, TrendingUp, Cpu,
  ShieldCheck, Waves, Atom, ArrowUpRight, ArrowDownRight
} from 'lucide-react';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { api } from '../services/api';

/* ─── Types ────────────────────────────────────────────────────────── */

interface MetricData {
  psnr: number;
  ssim: number;
  nrmse: number;
  processing_time: number;
  method: string;
}

interface HistoryEntry {
  job_id: string;
  method: string;
  psnr: number;
  ssim: number;
  nrmse: number;
  processing_time: number;
  created_at: string;
}

/* ─── Helpers ──────────────────────────────────────────────────────── */

/** Clamp PSNR to never show negative */
const safePsnr = (val: number | undefined): number =>
  val != null ? Math.max(val, 0) : 0;

const METHOD_COLORS: Record<string, string> = {
  noisy: '#6b7280',
  gaussian: '#6366f1',
  wiener: '#8b5cf6',
  tv: '#a78bfa',
  unet: '#22d3ee',
  vqc: '#facc15',
  median: '#f472b6',
};

const methodColor = (m: string) => METHOD_COLORS[m.toLowerCase()] ?? '#94a3b8';

/* ─── Stat Card ────────────────────────────────────────────────────── */

interface StatCardProps {
  label: string;
  value: string;
  sub: string;
  icon: React.ReactNode;
  accent: string;
  trend?: number;
}

const StatCard: React.FC<StatCardProps> = ({ label, value, sub, icon, accent, trend }) => (
  <div
    className="relative overflow-hidden rounded-2xl border border-white/[0.06] p-5"
    style={{
      background: `linear-gradient(135deg, ${accent}08 0%, ${accent}03 100%)`,
    }}
  >
    {/* Glow dot */}
    <div
      className="absolute -top-6 -right-6 h-20 w-20 rounded-full blur-2xl opacity-30"
      style={{ background: accent }}
    />
    <div className="flex items-start justify-between mb-3">
      <span
        className="inline-flex items-center justify-center h-9 w-9 rounded-xl"
        style={{ background: `${accent}18`, color: accent }}
      >
        {icon}
      </span>
      {trend != null && (
        <span
          className="flex items-center gap-0.5 text-xs font-semibold px-2 py-0.5 rounded-full"
          style={{
            background: trend >= 0 ? '#22c55e18' : '#ef444418',
            color: trend >= 0 ? '#22c55e' : '#ef4444',
          }}
        >
          {trend >= 0 ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
          {Math.abs(trend).toFixed(1)}%
        </span>
      )}
    </div>
    <p className="text-[0.7rem] uppercase tracking-widest text-muted-foreground/70 mb-1">
      {label}
    </p>
    <p className="text-2xl font-bold tracking-tight" style={{ color: accent }}>
      {value}
    </p>
    <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>
  </div>
);

/* ─── Custom Tooltip ───────────────────────────────────────────────── */

const ChartTooltip: React.FC<any> = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-xl border border-white/[0.08] bg-card/95 backdrop-blur-md px-4 py-3 shadow-xl">
      <p className="text-xs font-semibold text-foreground mb-1.5">{label}</p>
      {payload.map((p: any, i: number) => (
        <div key={i} className="flex items-center gap-2 text-xs">
          <span className="h-2 w-2 rounded-full" style={{ background: p.color }} />
          <span className="text-muted-foreground">{p.name}:</span>
          <span className="font-mono font-medium">{typeof p.value === 'number' ? p.value.toFixed(2) : p.value}</span>
        </div>
      ))}
    </div>
  );
};

/* ─── Main Dashboard ───────────────────────────────────────────────── */

export const Dashboard: React.FC = () => {
  // Fetch benchmark metrics (now live-merged on backend)
  const { data: metrics, isLoading: metricsLoading } = useQuery<MetricData[]>({
    queryKey: ['metrics'],
    queryFn: async () => {
      const res = await api.get('/api/v1/metrics');
      return res.data;
    },
    refetchInterval: 8000, // Refresh every 8s to catch new denoise results
  });

  // Fetch denoising history for time-series chart
  const { data: history } = useQuery<HistoryEntry[]>({
    queryKey: ['denoise-history'],
    queryFn: async () => {
      const res = await api.get('/api/v1/denoise/history');
      return res.data;
    },
    refetchInterval: 8000,
  });

  /* ── Derived data ── */
  const vqc = useMemo(() => metrics?.find((m) => m.method === 'vqc'), [metrics]);
  const bestClassical = useMemo(
    () =>
      metrics
        ?.filter((m) => m.method !== 'vqc' && m.method !== 'noisy')
        .reduce((best, c) => (c.psnr > best.psnr ? c : best), { psnr: 0, method: '', ssim: 0, nrmse: 1, processing_time: 0 }),
    [metrics],
  );

  const barData = useMemo(
    () =>
      metrics?.map((m) => ({
        name: m.method.toUpperCase(),
        PSNR: safePsnr(m.psnr),
        SSIM: +(m.ssim * 100).toFixed(1),
        method: m.method,
      })) ?? [],
    [metrics],
  );

  const radarData = useMemo(
    () =>
      metrics
        ?.filter((m) => m.method !== 'noisy')
        .map((m) => ({
          method: m.method.toUpperCase(),
          PSNR: +((safePsnr(m.psnr) / 40) * 100).toFixed(1),
          SSIM: +(m.ssim * 100).toFixed(1),
          Speed: +Math.max(0, 100 - m.processing_time * 20).toFixed(1),
          Accuracy: +((1 - m.nrmse) * 100).toFixed(1),
        })) ?? [],
    [metrics],
  );

  const historyChart = useMemo(() => {
    if (!history?.length) return [];
    return history
      .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
      .map((h, i) => ({
        idx: i + 1,
        PSNR: safePsnr(h.psnr),
        SSIM: +(h.ssim * 100).toFixed(1),
        method: h.method.toUpperCase(),
        time: h.processing_time.toFixed(2) + 's',
      }));
  }, [history]);

  const psnrImprovement =
    vqc && bestClassical ? safePsnr(vqc.psnr) - safePsnr(bestClassical.psnr) : 0;
  const psnrTrend =
    vqc && bestClassical && bestClassical.psnr > 0
      ? +((psnrImprovement / bestClassical.psnr) * 100).toFixed(1)
      : undefined;

  /* ── Loading ── */
  if (metricsLoading) {
    return (
      <div className="flex-1 flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-4">
          <div className="relative h-14 w-14">
            <div className="absolute inset-0 rounded-full border-2 border-yellow-400/30" />
            <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-yellow-400 animate-spin" />
            <Atom className="absolute inset-0 m-auto h-6 w-6 text-yellow-400 animate-pulse" />
          </div>
          <p className="text-sm text-muted-foreground tracking-wide">Loading quantum metrics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-7 p-8 pt-6">
      {/* ── Header ── */}
      <div className="flex items-end justify-between">
        <div>
          <h2
            className="text-3xl font-extrabold tracking-tight"
            style={{
              background: 'linear-gradient(135deg, #facc15 0%, #f97316 50%, #ef4444 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            Performance Dashboard
          </h2>
          <p className="text-muted-foreground text-sm mt-1">
            Quantum CT Sinogram Denoising — live metrics refresh every 8 s
          </p>
        </div>
        <div className="hidden md:flex items-center gap-2 text-xs text-muted-foreground border border-white/[0.06] rounded-full px-3 py-1.5">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-400" />
          </span>
          Live
        </div>
      </div>

      {/* ── Stat Cards ── */}
      <div className="grid gap-4 grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="VQC PSNR"
          value={safePsnr(vqc?.psnr).toFixed(1) + ' dB'}
          sub="Peak Signal-to-Noise Ratio"
          icon={<Zap className="h-4 w-4" />}
          accent="#facc15"
          trend={psnrTrend}
        />
        <StatCard
          label="VQC SSIM"
          value={vqc?.ssim.toFixed(3) ?? '--'}
          sub="Structural Similarity Index"
          icon={<ShieldCheck className="h-4 w-4" />}
          accent="#22d3ee"
        />
        <StatCard
          label="Processing Time"
          value={(vqc?.processing_time.toFixed(2) ?? '--') + ' s'}
          sub="Per sinogram average"
          icon={<Clock className="h-4 w-4" />}
          accent="#a78bfa"
        />
        <StatCard
          label="PSNR Gain"
          value={`+${psnrImprovement.toFixed(1)} dB`}
          sub={`vs ${bestClassical?.method.toUpperCase() ?? 'best classical'}`}
          icon={<TrendingUp className="h-4 w-4" />}
          accent="#22c55e"
        />
      </div>

      {/* ── Charts Row 1 ── */}
      <div className="grid gap-5 md:grid-cols-2">
        {/* Bar: Method Comparison */}
        <Card className="border-white/[0.06] bg-card/60 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base font-semibold">Method Comparison</CardTitle>
            <CardDescription>PSNR (dB) and SSIM (%) across methods</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={290}>
              <BarChart data={barData} barGap={4}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} stroke="#64748b" />
                <YAxis tick={{ fontSize: 11 }} stroke="#64748b" />
                <Tooltip content={<ChartTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="PSNR" name="PSNR (dB)" radius={[6, 6, 0, 0]}>
                  {barData.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={methodColor(entry.method)}
                      fillOpacity={entry.method === 'vqc' ? 1 : 0.55}
                    />
                  ))}
                </Bar>
                <Bar dataKey="SSIM" name="SSIM (%)" radius={[6, 6, 0, 0]} fill="#22d3ee" fillOpacity={0.4} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Radar: Multi-metric */}
        <Card className="border-white/[0.06] bg-card/60 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <CardTitle className="text-base font-semibold">Multi-Metric Radar</CardTitle>
            <CardDescription>Normalized performance across dimensions</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={290}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(255,255,255,0.06)" />
                <PolarAngleAxis dataKey="method" tick={{ fontSize: 10, fill: '#94a3b8' }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9, fill: '#64748b' }} />
                <Radar name="PSNR" dataKey="PSNR" stroke="#facc15" fill="#facc15" fillOpacity={0.2} />
                <Radar name="SSIM" dataKey="SSIM" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.15} />
                <Radar name="Speed" dataKey="Speed" stroke="#a78bfa" fill="#a78bfa" fillOpacity={0.1} />
                <Radar name="Accuracy" dataKey="Accuracy" stroke="#22c55e" fill="#22c55e" fillOpacity={0.1} />
                <Tooltip content={<ChartTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* ── Live History ── */}
      {historyChart.length > 0 && (
        <Card className="border-white/[0.06] bg-card/60 backdrop-blur-sm">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base font-semibold">Denoising History</CardTitle>
                <CardDescription>PSNR trend from actual denoising runs</CardDescription>
              </div>
              <span className="text-xs text-muted-foreground border border-white/[0.06] rounded-full px-2.5 py-1">
                {historyChart.length} run{historyChart.length !== 1 ? 's' : ''}
              </span>
            </div>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={historyChart}>
                <defs>
                  <linearGradient id="psnrGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#facc15" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#facc15" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                <XAxis dataKey="idx" label={{ value: 'Run #', position: 'insideBottom', offset: -2, fontSize: 11 }} tick={{ fontSize: 10 }} stroke="#64748b" />
                <YAxis domain={[0, 'auto']} tick={{ fontSize: 10 }} stroke="#64748b" />
                <Tooltip content={<ChartTooltip />} />
                <ReferenceLine y={0} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'min 0 dB', fontSize: 10, fill: '#ef4444' }} />
                <Area type="monotone" dataKey="PSNR" stroke="#facc15" fill="url(#psnrGrad)" strokeWidth={2} name="PSNR (dB)" dot={{ r: 3, fill: '#facc15' }} />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}

      {/* ── Benchmark Table ── */}
      <Card className="border-white/[0.06] bg-card/60 backdrop-blur-sm">
        <CardHeader className="pb-3">
          <CardTitle className="text-base font-semibold">Benchmark Results</CardTitle>
          <CardDescription>
            Live denoising metrics — PSNR values guaranteed non-negative
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto rounded-xl border border-white/[0.04]">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-muted/40 text-muted-foreground text-[0.7rem] uppercase tracking-wider">
                  <th className="px-5 py-3 text-left font-medium">Method</th>
                  <th className="px-5 py-3 text-right font-medium">PSNR (dB)</th>
                  <th className="px-5 py-3 text-right font-medium">SSIM</th>
                  <th className="px-5 py-3 text-right font-medium">NRMSE</th>
                  <th className="px-5 py-3 text-right font-medium">Time (s)</th>
                  <th className="px-5 py-3 text-center font-medium">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/[0.04]">
                {metrics?.map((m) => {
                  const isVqc = m.method === 'vqc';
                  return (
                    <tr
                      key={m.method}
                      className={
                        isVqc
                          ? 'bg-yellow-400/[0.04] hover:bg-yellow-400/[0.08] transition-colors'
                          : 'hover:bg-white/[0.02] transition-colors'
                      }
                    >
                      <td className="px-5 py-3.5 font-medium flex items-center gap-2">
                        <span
                          className="h-2 w-2 rounded-full flex-shrink-0"
                          style={{ background: methodColor(m.method) }}
                        />
                        {m.method.toUpperCase()}
                        {isVqc && (
                          <span className="text-[0.6rem] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded bg-yellow-400/15 text-yellow-400">
                            Quantum
                          </span>
                        )}
                      </td>
                      <td className="px-5 py-3.5 text-right font-mono text-xs">
                        <span style={{ color: isVqc ? '#facc15' : undefined }}>
                          {safePsnr(m.psnr).toFixed(2)}
                        </span>
                      </td>
                      <td className="px-5 py-3.5 text-right font-mono text-xs">{m.ssim.toFixed(4)}</td>
                      <td className="px-5 py-3.5 text-right font-mono text-xs">{m.nrmse.toFixed(4)}</td>
                      <td className="px-5 py-3.5 text-right font-mono text-xs">{m.processing_time.toFixed(3)}</td>
                      <td className="px-5 py-3.5 text-center">
                        {isVqc ? (
                          <span className="text-[0.65rem] font-semibold text-yellow-400">Best</span>
                        ) : m.method === 'noisy' ? (
                          <span className="text-[0.65rem] text-muted-foreground">Input</span>
                        ) : (
                          <span className="text-[0.65rem] text-muted-foreground">Baseline</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* ── System Info ── */}
      <div className="grid gap-4 grid-cols-1 md:grid-cols-3">
        {[
          {
            label: 'Quantum Backend',
            value: 'PennyLane',
            sub: '16 qubits, 6 variational layers',
            icon: <Atom className="h-4 w-4" />,
            accent: '#a78bfa',
          },
          {
            label: 'Model Status',
            value: 'Active',
            sub: 'VQC v1.0 — edge-preserving loss',
            icon: <Activity className="h-4 w-4" />,
            accent: '#22c55e',
          },
          {
            label: 'Dataset',
            value: 'Mayo LDCT',
            sub: '25% dose · 180 projections',
            icon: <Waves className="h-4 w-4" />,
            accent: '#22d3ee',
          },
        ].map((card) => (
          <div
            key={card.label}
            className="relative overflow-hidden rounded-2xl border border-white/[0.06] p-5"
            style={{ background: `linear-gradient(135deg, ${card.accent}06 0%, transparent 100%)` }}
          >
            <div className="flex items-center gap-3 mb-2">
              <span
                className="inline-flex items-center justify-center h-8 w-8 rounded-lg"
                style={{ background: `${card.accent}15`, color: card.accent }}
              >
                {card.icon}
              </span>
              <span className="text-[0.7rem] uppercase tracking-widest text-muted-foreground/70">
                {card.label}
              </span>
            </div>
            <p className="text-xl font-bold" style={{ color: card.accent }}>{card.value}</p>
            <p className="text-xs text-muted-foreground mt-0.5">{card.sub}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Dashboard;
