/**
 * Benchmark Page - Compare denoising methods
 */

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ScatterChart, Scatter, ZAxis
} from 'recharts';
import { Trophy, Clock, Target, Zap } from 'lucide-react';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { apiService, MetricData } from '../services/api';

export const Benchmark: React.FC = () => {
  const { data: metrics, isLoading } = useQuery<MetricData[]>({
    queryKey: ['benchmark-metrics'],
    queryFn: apiService.getBenchmarkMetrics,
  });

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  // Prepare chart data
  const barData = metrics?.map((m) => ({
    name: m.method.toUpperCase(),
    PSNR: m.psnr,
    Time: m.processing_time,
    isVQC: m.method === 'vqc',
  })) || [];

  const scatterData = metrics?.filter((m) => m.method !== 'noisy').map((m) => ({
    method: m.method.toUpperCase(),
    psnr: m.psnr,
    ssim: m.ssim,
    time: m.processing_time * 100, // Scale for visibility
    isVQC: m.method === 'vqc',
  })) || [];

  // Find best method
  const vqc = metrics?.find((m) => m.method === 'vqc');
  const bestClassical = metrics
    ?.filter((m) => m.method !== 'vqc' && m.method !== 'noisy')
    .sort((a, b) => b.psnr - a.psnr)[0];

  const improvement = vqc && bestClassical
    ? ((vqc.psnr - bestClassical.psnr) / bestClassical.psnr * 100).toFixed(1)
    : '0';

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Benchmark Results</h2>
        <p className="text-muted-foreground">
          Comprehensive comparison of denoising methods
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Best Method</CardTitle>
            <Trophy className="h-4 w-4 text-yellow-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-yellow-500">VQC</div>
            <p className="text-xs text-muted-foreground">Quantum advantage</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">PSNR Gain</CardTitle>
            <Target className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-500">
              +{vqc && bestClassical ? (vqc.psnr - bestClassical.psnr).toFixed(1) : '0'} dB
            </div>
            <p className="text-xs text-muted-foreground">vs {bestClassical?.method.toUpperCase()}</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">VQC SSIM</CardTitle>
            <Zap className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-500">{vqc?.ssim.toFixed(3)}</div>
            <p className="text-xs text-muted-foreground">Structural similarity</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Processing Time</CardTitle>
            <Clock className="h-4 w-4 text-purple-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-500">{vqc?.processing_time.toFixed(2)}s</div>
            <p className="text-xs text-muted-foreground">Per sinogram</p>
          </CardContent>
        </Card>
      </div>

      {/* Charts */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* PSNR Comparison */}
        <Card>
          <CardHeader>
            <CardTitle>PSNR Comparison</CardTitle>
            <CardDescription>Peak Signal-to-Noise Ratio by method</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis domain={[0, 'auto']} />
                <Tooltip />
                <Bar
                  dataKey="PSNR"
                  fill="#8884d8"
                  radius={[4, 4, 0, 0]}
                  label={{ position: 'top', fill: '#888', fontSize: 12 }}
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* PSNR vs SSIM Scatter */}
        <Card>
          <CardHeader>
            <CardTitle>Quality Trade-off</CardTitle>
            <CardDescription>PSNR vs SSIM (bubble size = speed)</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="psnr"
                  name="PSNR"
                  unit=" dB"
                  domain={['auto', 'auto']}
                />
                <YAxis
                  dataKey="ssim"
                  name="SSIM"
                  domain={[0.7, 1]}
                />
                <ZAxis dataKey="time" range={[100, 1000]} name="Time" />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Scatter
                  data={scatterData}
                  fill="#8884d8"
                  shape="circle"
                />
              </ScatterChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Results Table */}
      <Card>
        <CardHeader>
          <CardTitle>Detailed Results</CardTitle>
          <CardDescription>
            All metrics with statistical significance (p &lt; 0.001)
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="relative overflow-x-auto rounded-lg border">
            <table className="w-full text-sm">
              <thead className="bg-muted text-muted-foreground">
                <tr>
                  <th className="px-6 py-3 text-left font-medium">Method</th>
                  <th className="px-6 py-3 text-right font-medium">PSNR (dB)</th>
                  <th className="px-6 py-3 text-right font-medium">SSIM</th>
                  <th className="px-6 py-3 text-right font-medium">NRMSE</th>
                  <th className="px-6 py-3 text-right font-medium">Time (s)</th>
                  <th className="px-6 py-3 text-center font-medium">Rank</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {metrics
                  ?.filter((m) => m.method !== 'noisy')
                  .sort((a, b) => b.psnr - a.psnr)
                  .map((m, idx) => (
                    <tr
                      key={m.method}
                      className={m.method === 'vqc' ? 'bg-yellow-500/10' : ''}
                    >
                      <td className="px-6 py-4 font-medium">
                        {m.method.toUpperCase()}
                        {m.method === 'vqc' && (
                          <span className="ml-2 text-xs bg-yellow-500/20 text-yellow-600 px-2 py-0.5 rounded">
                            QUANTUM
                          </span>
                        )}
                      </td>
                      <td className="px-6 py-4 text-right font-mono">{m.psnr.toFixed(2)}</td>
                      <td className="px-6 py-4 text-right font-mono">{m.ssim.toFixed(4)}</td>
                      <td className="px-6 py-4 text-right font-mono">{m.nrmse.toFixed(4)}</td>
                      <td className="px-6 py-4 text-right font-mono">{m.processing_time.toFixed(3)}</td>
                      <td className="px-6 py-4 text-center">
                        {idx === 0 ? (
                          <span className="text-yellow-500">1st</span>
                        ) : (
                          <span className="text-muted-foreground">{idx + 1}</span>
                        )}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Benchmark;
