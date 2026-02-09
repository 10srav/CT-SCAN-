/**
 * Settings Page - Application configuration
 */

import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Save, RefreshCw, Server, Cpu, HardDrive } from 'lucide-react';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Switch } from '../components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { useToast } from '../components/ui/use-toast';
import { apiService, HealthResponse } from '../services/api';

export const Settings: React.FC = () => {
  const { toast } = useToast();

  // System health
  const { data: health, refetch: refetchHealth } = useQuery<HealthResponse>({
    queryKey: ['health'],
    queryFn: apiService.checkHealth,
    refetchInterval: 30000, // Refresh every 30s
  });

  // Settings state
  const [settings, setSettings] = useState({
    defaultMethod: 'vqc',
    patchSize: 16,
    useGpu: true,
    autoReconstruct: true,
    saveResults: true,
    darkMode: true,
  });

  const handleSave = () => {
    // In a real app, save to backend/localStorage
    localStorage.setItem('quantum-ct-settings', JSON.stringify(settings));
    toast({
      title: 'Settings saved',
      description: 'Your preferences have been updated.',
    });
  };

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Settings</h2>
          <p className="text-muted-foreground">
            Configure application preferences and view system status
          </p>
        </div>
        <Button onClick={handleSave}>
          <Save className="mr-2 h-4 w-4" />
          Save Changes
        </Button>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* System Status */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Server className="h-5 w-5" />
              System Status
            </CardTitle>
            <CardDescription>Backend service health and metrics</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Health Status */}
            <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
              <div className="flex items-center gap-3">
                <div className={`h-3 w-3 rounded-full ${health?.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'}`} />
                <span className="font-medium">API Status</span>
              </div>
              <span className="text-sm text-muted-foreground">
                {health?.status || 'Unknown'}
              </span>
            </div>

            {/* Version */}
            <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
              <span className="font-medium">Version</span>
              <span className="text-sm text-muted-foreground">
                {health?.version || '--'}
              </span>
            </div>

            {/* Quantum Backend */}
            <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-blue-500" />
                <span className="font-medium">Quantum Backend</span>
              </div>
              <span className="text-sm text-muted-foreground">
                {health?.quantum_backend || '--'}
              </span>
            </div>

            {/* GPU Status */}
            <div className="flex items-center justify-between p-4 bg-muted rounded-lg">
              <span className="font-medium">GPU Available</span>
              <span className={health?.gpu_available ? 'text-green-500' : 'text-muted-foreground'}>
                {health?.gpu_available ? 'Yes' : 'No'}
              </span>
            </div>

            {/* Refresh Button */}
            <Button variant="outline" className="w-full" onClick={() => refetchHealth()}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Refresh Status
            </Button>
          </CardContent>
        </Card>

        {/* Processing Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-5 w-5" />
              Processing Settings
            </CardTitle>
            <CardDescription>Configure denoising defaults</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Default Method */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Default Denoising Method</label>
              <Select
                value={settings.defaultMethod}
                onValueChange={(val) => setSettings({ ...settings, defaultMethod: val })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="vqc">VQC (Quantum)</SelectItem>
                  <SelectItem value="gaussian">Gaussian Filter</SelectItem>
                  <SelectItem value="wiener">Wiener Filter</SelectItem>
                  <SelectItem value="tv">Total Variation</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Patch Size */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Default Patch Size</label>
              <Select
                value={settings.patchSize.toString()}
                onValueChange={(val) => setSettings({ ...settings, patchSize: parseInt(val) })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="8">8x8</SelectItem>
                  <SelectItem value="16">16x16 (Recommended)</SelectItem>
                  <SelectItem value="32">32x32</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* GPU Toggle */}
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Use GPU Acceleration</p>
                <p className="text-sm text-muted-foreground">
                  Enable CUDA for faster processing
                </p>
              </div>
              <Switch
                checked={settings.useGpu}
                onCheckedChange={(val) => setSettings({ ...settings, useGpu: val })}
              />
            </div>

            {/* Auto Reconstruct */}
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Auto Reconstruct</p>
                <p className="text-sm text-muted-foreground">
                  Automatically run FBP after denoising
                </p>
              </div>
              <Switch
                checked={settings.autoReconstruct}
                onCheckedChange={(val) => setSettings({ ...settings, autoReconstruct: val })}
              />
            </div>
          </CardContent>
        </Card>

        {/* Storage Settings */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <HardDrive className="h-5 w-5" />
              Storage Settings
            </CardTitle>
            <CardDescription>Manage data and model storage</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Save Results */}
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium">Save Results</p>
                <p className="text-sm text-muted-foreground">
                  Persist denoised sinograms
                </p>
              </div>
              <Switch
                checked={settings.saveResults}
                onCheckedChange={(val) => setSettings({ ...settings, saveResults: val })}
              />
            </div>

            {/* Storage Info */}
            <div className="p-4 bg-muted rounded-lg space-y-2">
              <div className="flex justify-between text-sm">
                <span>Models Directory</span>
                <span className="text-muted-foreground">./data/models</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Results Directory</span>
                <span className="text-muted-foreground">./data/results</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* About */}
        <Card>
          <CardHeader>
            <CardTitle>About</CardTitle>
            <CardDescription>Project information</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-4 bg-muted rounded-lg space-y-2">
              <h4 className="font-semibold">Quantum CT Sinogram Denoising</h4>
              <p className="text-sm text-muted-foreground">
                VQC-based denoising for low-dose CT imaging
              </p>
            </div>

            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Team ID</span>
                <span>7A</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Guide</span>
                <span>Mrs. S. Anusha</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Academic Year</span>
                <span>2025-26</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">SDG Goals</span>
                <span>3 (Health), 9 (Innovation)</span>
              </div>
            </div>

            <div className="pt-4 border-t">
              <p className="text-xs text-muted-foreground text-center">
                Built with PennyLane, PyTorch, FastAPI, and React
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Settings;
