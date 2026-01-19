/**
 * Denoise Page - Upload and process sinograms with VQC
 */

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useMutation } from '@tanstack/react-query';
import { Upload, Zap, RefreshCw, Download, Image as ImageIcon } from 'lucide-react';

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Progress } from '../components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Slider } from '../components/ui/slider';
import { useToast } from '../components/ui/use-toast';
import { api } from '../services/api';

interface DenoiseResult {
  job_id: string;
  status: string;
  message: string;
  metrics?: {
    psnr: number;
    ssim: number;
    nrmse: number;
  };
  processing_time?: number;
}

export const Denoise: React.FC = () => {
  const { toast } = useToast();
  const [method, setMethod] = useState<string>('vqc');
  const [patchSize, setPatchSize] = useState<number>(16);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<DenoiseResult | null>(null);

  // File upload handler
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setUploadedFile(file);
      // Create preview for image files
      if (file.type.startsWith('image/')) {
        setPreviewUrl(URL.createObjectURL(file));
      } else {
        setPreviewUrl(null);
      }
      toast({
        title: 'File uploaded',
        description: `${file.name} ready for processing`,
      });
    }
  }, [toast]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/octet-stream': ['.npy'],
      'application/dicom': ['.dcm', '.dicom'],
      'image/*': ['.png', '.jpg', '.jpeg'],
    },
    maxFiles: 1,
  });

  // Denoise mutation
  const denoiseMutation = useMutation({
    mutationFn: async () => {
      if (!uploadedFile) throw new Error('No file uploaded');

      const formData = new FormData();
      formData.append('file', uploadedFile);

      const response = await api.post('/api/v1/denoise', formData, {
        params: { method, patch_size: patchSize, return_metrics: true },
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120000, // 2 minutes for VQC
      });

      return response.data as DenoiseResult;
    },
    onSuccess: (data) => {
      setResult(data);
      toast({
        title: 'Denoising complete!',
        description: `Processed in ${data.processing_time?.toFixed(2)}s`,
      });
    },
    onError: (error: Error) => {
      toast({
        title: 'Denoising failed',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  // Download result
  const downloadResult = async () => {
    if (!result?.job_id) return;

    try {
      const response = await api.get(`/api/v1/denoise/${result.job_id}/result`, {
        responseType: 'blob',
      });

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `denoised_${result.job_id}.npy`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      toast({
        title: 'Download failed',
        description: 'Could not download result',
        variant: 'destructive',
      });
    }
  };

  return (
    <div className="flex-1 space-y-6 p-8 pt-6">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Denoise Sinogram</h2>
        <p className="text-muted-foreground">
          Upload a CT sinogram and apply quantum denoising
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle>Upload Sinogram</CardTitle>
            <CardDescription>
              Supports .npy, .dcm, and image files
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Dropzone */}
            <div
              {...getRootProps()}
              className={`
                border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
                transition-colors duration-200
                ${isDragActive ? 'border-primary bg-primary/10' : 'border-muted-foreground/25'}
                ${uploadedFile ? 'border-green-500 bg-green-500/10' : ''}
              `}
            >
              <input {...getInputProps()} />
              <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
              {isDragActive ? (
                <p>Drop the file here...</p>
              ) : uploadedFile ? (
                <div>
                  <p className="font-medium text-green-500">{uploadedFile.name}</p>
                  <p className="text-sm text-muted-foreground">
                    {(uploadedFile.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              ) : (
                <div>
                  <p>Drag & drop a sinogram file, or click to select</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    .npy, .dcm, .png, .jpg supported
                  </p>
                </div>
              )}
            </div>

            {/* Preview */}
            {previewUrl && (
              <div className="mt-4">
                <p className="text-sm font-medium mb-2">Preview:</p>
                <img
                  src={previewUrl}
                  alt="Sinogram preview"
                  className="max-h-48 rounded border"
                />
              </div>
            )}
          </CardContent>
        </Card>

        {/* Settings Section */}
        <Card>
          <CardHeader>
            <CardTitle>Denoising Settings</CardTitle>
            <CardDescription>
              Configure the denoising parameters
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Method Selection */}
            <div className="space-y-2">
              <label className="text-sm font-medium">Denoising Method</label>
              <Select value={method} onValueChange={setMethod}>
                <SelectTrigger>
                  <SelectValue placeholder="Select method" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="vqc">
                    <span className="flex items-center gap-2">
                      <Zap className="h-4 w-4 text-yellow-500" />
                      VQC (Quantum)
                    </span>
                  </SelectItem>
                  <SelectItem value="gaussian">Gaussian Filter</SelectItem>
                  <SelectItem value="wiener">Wiener Filter</SelectItem>
                  <SelectItem value="tv">Total Variation</SelectItem>
                  <SelectItem value="median">Median Filter</SelectItem>
                </SelectContent>
              </Select>
              {method === 'vqc' && (
                <p className="text-xs text-yellow-500">
                  Best quality - Uses 16-qubit VQC with edge preservation
                </p>
              )}
            </div>

            {/* Patch Size */}
            <div className="space-y-2">
              <label className="text-sm font-medium">
                Patch Size: {patchSize}×{patchSize}
              </label>
              <Slider
                value={[patchSize]}
                onValueChange={([val]) => setPatchSize(val)}
                min={4}
                max={64}
                step={4}
              />
              <p className="text-xs text-muted-foreground">
                Smaller patches = finer details, larger = faster processing
              </p>
            </div>

            {/* Process Button */}
            <Button
              onClick={() => denoiseMutation.mutate()}
              disabled={!uploadedFile || denoiseMutation.isPending}
              className="w-full"
              size="lg"
            >
              {denoiseMutation.isPending ? (
                <>
                  <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Zap className="mr-2 h-4 w-4" />
                  Denoise Sinogram
                </>
              )}
            </Button>

            {/* Progress */}
            {denoiseMutation.isPending && (
              <div className="space-y-2">
                <Progress value={66} className="h-2" />
                <p className="text-sm text-center text-muted-foreground">
                  Running quantum circuit...
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Results Section */}
      {result && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ImageIcon className="h-5 w-5" />
              Denoising Results
            </CardTitle>
            <CardDescription>
              Job ID: {result.job_id}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid gap-4 md:grid-cols-4">
              {/* Metrics */}
              <div className="text-center p-4 bg-muted rounded-lg">
                <p className="text-2xl font-bold text-blue-500">
                  {result.metrics?.psnr?.toFixed(2) || '--'} dB
                </p>
                <p className="text-sm text-muted-foreground">PSNR</p>
              </div>
              <div className="text-center p-4 bg-muted rounded-lg">
                <p className="text-2xl font-bold text-green-500">
                  {result.metrics?.ssim?.toFixed(4) || '--'}
                </p>
                <p className="text-sm text-muted-foreground">SSIM</p>
              </div>
              <div className="text-center p-4 bg-muted rounded-lg">
                <p className="text-2xl font-bold text-yellow-500">
                  {result.metrics?.nrmse?.toFixed(4) || '--'}
                </p>
                <p className="text-sm text-muted-foreground">NRMSE</p>
              </div>
              <div className="text-center p-4 bg-muted rounded-lg">
                <p className="text-2xl font-bold text-purple-500">
                  {result.processing_time?.toFixed(2) || '--'}s
                </p>
                <p className="text-sm text-muted-foreground">Time</p>
              </div>
            </div>

            {/* Download Button */}
            <div className="mt-6 flex justify-center">
              <Button onClick={downloadResult} variant="outline" size="lg">
                <Download className="mr-2 h-4 w-4" />
                Download Denoised Sinogram
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default Denoise;
