/**
 * Dashboard Page - Main overview of the Quantum CT system
 */

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  LineChart, Line, ResponsiveContainer, RadarChart, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import { Activity, Zap, Image, Clock, TrendingUp, Cpu } from 'lucide-react';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { api } from '../services/api';

// Types
interface MetricData {
  psnr: number;
  ssim: number;
  nrmse: number;
  processing_time: number;
  method: string;
}

interface StatsCardProps {
  title: string;
  value: string | number;
  description: string;
  icon: React.ReactNode;
  trend?: number;
}

// Stats Card Component
const StatsCard: React.FC<StatsCardProps> = ({ title, value, description, icon, trend }) => (
  <Card>
    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
      <CardTitle className="text-sm font-medium">{title}</CardTitle>
      {icon}
    </CardHeader>
    <CardContent>
      <div className="text-2xl font-bold">{value}</div>
      <p className="text-xs text-muted-foreground">
        {description}
        {trend && (
          <span className={trend > 0 ? 'text-green-500' : 'text-red-500'}>
            {' '}({trend > 0 ? '+' : ''}{trend}%)
          </span>
        )}
      </p>
    </CardContent>
  </Card>
);

export const Dashboard: React.FC = () => {
  // Fetch benchmark metrics
  const { data: metrics, isLoading } = useQuery<MetricData[]>({
    queryKey: ['metrics'],
    queryFn: async () => {
      const response = await api.get('/api/v1/metrics');
      return response.data;
    },
  });

  // Prepare chart data
  const barChartData = metrics?.map(m => ({
    name: m.method.toUpperCase(),
    PSNR: m.psnr,
    SSIM: m.ssim * 30, // Scale for visibility
  })) || [];

  const radarData = metrics?.filter(m => m.method !== 'noisy').map(m => ({
    method: m.method,
    PSNR: (m.psnr / 35) * 100,
    SSIM: m.ssim * 100,
    Speed: Math.max(0, 100 - m.processing_time * 20),
    Accuracy: (1 - m.nrmse) * 100,
  })) || [];

  const vqcMetric = metrics?.find(m => m.method === 'vqc');
  const bestClassical = metrics?.filter(m => m.method !== 'vqc' && m.method !== 'noisy')
    .reduce((best, curr) => curr.psnr > best.psnr ? curr : best, { psnr: 0, method: '' });

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Dashboard</h2>
          <p className="text-muted-foreground">
            Quantum CT Sinogram Denoising Performance Overview
          </p>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatsCard
          title="VQC PSNR"
          value={vqcMetric?.psnr.toFixed(1) + ' dB' || '--'}
          description="Peak Signal-to-Noise Ratio"
          icon={<Zap className="h-4 w-4 text-yellow-500" />}
          trend={vqcMetric && bestClassical ? Number(((vqcMetric.psnr - bestClassical.psnr) / bestClassical.psnr * 100).toFixed(1)) : undefined}
        />
        <StatsCard
          title="VQC SSIM"
          value={vqcMetric?.ssim.toFixed(3) || '--'}
          description="Structural Similarity Index"
          icon={<Image className="h-4 w-4 text-blue-500" />}
        />
        <StatsCard
          title="Processing Time"
          value={vqcMetric?.processing_time.toFixed(2) + 's' || '--'}
          description="Average per sinogram"
          icon={<Clock className="h-4 w-4 text-green-500" />}
        />
        <StatsCard
          title="PSNR Improvement"
          value={vqcMetric && bestClassical ? `+${(vqcMetric.psnr - bestClassical.psnr).toFixed(1)} dB` : '--'}
          description={`vs ${bestClassical?.method.toUpperCase() || 'best classical'}`}
          icon={<TrendingUp className="h-4 w-4 text-purple-500" />}
        />
      </div>

      {/* Charts */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Bar Chart - Method Comparison */}
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Method Comparison</CardTitle>
            <CardDescription>PSNR and SSIM scores by denoising method</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={barChartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="PSNR" fill="#8884d8" name="PSNR (dB)" />
                <Bar dataKey="SSIM" fill="#82ca9d" name="SSIM (×30)" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Radar Chart - Multi-metric Comparison */}
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle>Multi-Metric Analysis</CardTitle>
            <CardDescription>Comprehensive performance radar</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="method" />
                <PolarRadiusAxis angle={30} domain={[0, 100]} />
                <Radar name="PSNR" dataKey="PSNR" stroke="#8884d8" fill="#8884d8" fillOpacity={0.3} />
                <Radar name="SSIM" dataKey="SSIM" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.3} />
                <Radar name="Speed" dataKey="Speed" stroke="#ffc658" fill="#ffc658" fillOpacity={0.3} />
                <Tooltip />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Method Details Table */}
      <Card>
        <CardHeader>
          <CardTitle>Benchmark Results</CardTitle>
          <CardDescription>
            Detailed comparison of all denoising methods on the test dataset
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="relative overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs uppercase bg-muted">
                <tr>
                  <th className="px-6 py-3">Method</th>
                  <th className="px-6 py-3">PSNR (dB)</th>
                  <th className="px-6 py-3">SSIM</th>
                  <th className="px-6 py-3">NRMSE</th>
                  <th className="px-6 py-3">Time (s)</th>
                  <th className="px-6 py-3">Status</th>
                </tr>
              </thead>
              <tbody>
                {metrics?.map((m, idx) => (
                  <tr key={m.method} className={idx % 2 === 0 ? 'bg-background' : 'bg-muted/50'}>
                    <td className="px-6 py-4 font-medium">
                      {m.method.toUpperCase()}
                      {m.method === 'vqc' && (
                        <span className="ml-2 px-2 py-1 text-xs bg-yellow-500/20 text-yellow-500 rounded">
                          QUANTUM
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4">{m.psnr.toFixed(2)}</td>
                    <td className="px-6 py-4">{m.ssim.toFixed(4)}</td>
                    <td className="px-6 py-4">{m.nrmse.toFixed(4)}</td>
                    <td className="px-6 py-4">{m.processing_time.toFixed(3)}</td>
                    <td className="px-6 py-4">
                      {m.method === 'vqc' ? (
                        <span className="text-green-500">Best</span>
                      ) : (
                        <span className="text-muted-foreground">Baseline</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* System Info */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Quantum Backend</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold">PennyLane</div>
            <p className="text-xs text-muted-foreground">16 qubits, 6 layers</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Model Status</CardTitle>
            <Activity className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold text-green-500">Active</div>
            <p className="text-xs text-muted-foreground">VQC v1.0 loaded</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Dataset</CardTitle>
            <Image className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-xl font-bold">Mayo LDCT</div>
            <p className="text-xs text-muted-foreground">25% dose, 180 projections</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Dashboard;
