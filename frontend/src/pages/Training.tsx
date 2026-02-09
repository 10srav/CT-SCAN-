/**
 * Training Page - Train custom VQC models
 */

import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Play, Pause, RotateCcw, Settings2, Cpu, Layers } from 'lucide-react';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Progress } from '../components/ui/progress';
import { Slider } from '../components/ui/slider';
import { useToast } from '../components/ui/use-toast';
import { apiService, TrainRequest, ProgressWebSocket } from '../services/api';

export const Training: React.FC = () => {
  const { toast } = useToast();

  // Training configuration
  const [config, setConfig] = useState<TrainRequest>({
    epochs: 100,
    batchSize: 32,
    learningRate: 0.01,
    numQubits: 16,
    vqcLayers: 6,
    validationSplit: 0.1,
  });

  // Training state
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [trainingLoss, setTrainingLoss] = useState<number[]>([]);
  const [valLoss, setValLoss] = useState<number[]>([]);

  // Start training mutation
  const trainMutation = useMutation({
    mutationFn: async () => {
      const response = await apiService.startTraining(config);
      return response;
    },
    onSuccess: (data) => {
      setIsTraining(true);
      toast({
        title: 'Training started',
        description: `Job ID: ${data.job_id}`,
      });

      // Connect WebSocket for progress updates
      const ws = new ProgressWebSocket(data.job_id, {
        onProgress: (update) => {
          setProgress(update.progress || 0);
          setCurrentEpoch(update.epoch || 0);
          if (update.train_loss) {
            setTrainingLoss((prev) => [...prev, update.train_loss]);
          }
          if (update.val_loss) {
            setValLoss((prev) => [...prev, update.val_loss]);
          }
        },
        onComplete: (result) => {
          setIsTraining(false);
          setProgress(100);
          toast({
            title: 'Training complete!',
            description: `Model saved. Best val loss: ${result.best_val_loss?.toFixed(6) || 'N/A'}`,
          });
        },
        onError: (error) => {
          setIsTraining(false);
          toast({
            title: 'Training failed',
            description: error.message,
            variant: 'destructive',
          });
        },
      });

      ws.connect();
    },
    onError: (error: Error) => {
      toast({
        title: 'Failed to start training',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  const resetTraining = () => {
    setProgress(0);
    setCurrentEpoch(0);
    setTrainingLoss([]);
    setValLoss([]);
    setIsTraining(false);
  };

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Model Training</h2>
        <p className="text-muted-foreground">
          Train custom VQC models on your CT data
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Configuration Card */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Settings2 className="h-5 w-5" />
              Training Configuration
            </CardTitle>
            <CardDescription>
              Configure VQC architecture and training parameters
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Epochs */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium">Epochs</label>
                <span className="text-sm text-muted-foreground">{config.epochs}</span>
              </div>
              <Slider
                value={[config.epochs]}
                onValueChange={([val]) => setConfig({ ...config, epochs: val })}
                min={10}
                max={500}
                step={10}
                disabled={isTraining}
              />
            </div>

            {/* Batch Size */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium">Batch Size</label>
                <span className="text-sm text-muted-foreground">{config.batchSize}</span>
              </div>
              <Slider
                value={[config.batchSize]}
                onValueChange={([val]) => setConfig({ ...config, batchSize: val })}
                min={8}
                max={128}
                step={8}
                disabled={isTraining}
              />
            </div>

            {/* Learning Rate */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium">Learning Rate</label>
                <span className="text-sm text-muted-foreground">{config.learningRate}</span>
              </div>
              <Slider
                value={[config.learningRate * 1000]}
                onValueChange={([val]) => setConfig({ ...config, learningRate: val / 1000 })}
                min={1}
                max={100}
                step={1}
                disabled={isTraining}
              />
            </div>

            {/* Qubits */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Cpu className="h-4 w-4" />
                  Number of Qubits
                </label>
                <span className="text-sm text-muted-foreground">{config.numQubits}</span>
              </div>
              <Slider
                value={[config.numQubits]}
                onValueChange={([val]) => setConfig({ ...config, numQubits: val })}
                min={4}
                max={24}
                step={2}
                disabled={isTraining}
              />
            </div>

            {/* VQC Layers */}
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Layers className="h-4 w-4" />
                  VQC Layers
                </label>
                <span className="text-sm text-muted-foreground">{config.vqcLayers}</span>
              </div>
              <Slider
                value={[config.vqcLayers]}
                onValueChange={([val]) => setConfig({ ...config, vqcLayers: val })}
                min={2}
                max={12}
                step={1}
                disabled={isTraining}
              />
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4 pt-4">
              <Button
                onClick={() => trainMutation.mutate()}
                disabled={isTraining || trainMutation.isPending}
                className="flex-1"
              >
                {isTraining ? (
                  <>
                    <Pause className="mr-2 h-4 w-4" />
                    Training...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" />
                    Start Training
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                onClick={resetTraining}
                disabled={isTraining}
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Training Progress Card */}
        <Card>
          <CardHeader>
            <CardTitle>Training Progress</CardTitle>
            <CardDescription>
              {isTraining
                ? `Epoch ${currentEpoch} / ${config.epochs}`
                : 'Ready to train'}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Progress Bar */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progress</span>
                <span>{progress.toFixed(1)}%</span>
              </div>
              <Progress value={progress} className="h-3" />
            </div>

            {/* Stats */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-muted rounded-lg p-4 text-center">
                <p className="text-2xl font-bold text-blue-500">
                  {trainingLoss.length > 0
                    ? trainingLoss[trainingLoss.length - 1].toFixed(6)
                    : '--'}
                </p>
                <p className="text-sm text-muted-foreground">Training Loss</p>
              </div>
              <div className="bg-muted rounded-lg p-4 text-center">
                <p className="text-2xl font-bold text-green-500">
                  {valLoss.length > 0
                    ? valLoss[valLoss.length - 1].toFixed(6)
                    : '--'}
                </p>
                <p className="text-sm text-muted-foreground">Validation Loss</p>
              </div>
            </div>

            {/* Model Info */}
            <div className="bg-muted/50 rounded-lg p-4 space-y-2">
              <p className="text-sm font-medium">Model Architecture</p>
              <div className="text-xs text-muted-foreground space-y-1">
                <p>Qubits: {config.numQubits}</p>
                <p>VQC Layers: {config.vqcLayers}</p>
                <p>Gates: RY, RZ rotations + CZ entanglement</p>
                <p>Embedding: Angle encoding</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Training;
