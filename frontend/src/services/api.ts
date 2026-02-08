/**
 * API Service for Quantum CT Denoising
 *
 * Provides typed API client for all backend endpoints
 */

import axios, { AxiosInstance, AxiosError, AxiosRequestConfig } from 'axios';

// Types
export interface DenoiseRequest {
  method: 'vqc' | 'gaussian' | 'wiener' | 'tv' | 'median';
  patchSize?: number;
  returnMetrics?: boolean;
}

export interface DenoiseResponse {
  job_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  message: string;
  metrics?: {
    psnr: number;
    ssim: number;
    nrmse: number;
  };
  processing_time?: number;
}

export interface TrainRequest {
  epochs: number;
  batchSize: number;
  learningRate: number;
  numQubits: number;
  vqcLayers: number;
  validationSplit?: number;
}

export interface TrainResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface MetricData {
  psnr: number;
  ssim: number;
  nrmse: number;
  processing_time: number;
  method: string;
}

export interface HealthResponse {
  status: string;
  timestamp: string;
  version: string;
  quantum_backend: string;
  gpu_available: boolean;
}

export interface JobStatus {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress?: number;
  result?: any;
  error?: string;
}

// API Configuration
// Use empty string for relative URLs so Vite's dev proxy handles /api/* routes
const API_BASE_URL = import.meta.env.VITE_API_URL || '';

// Create axios instance
export const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for auth
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.response?.status === 401) {
      // Handle unauthorized — clear token but don't redirect (no login page)
      localStorage.removeItem('auth_token');
    }
    return Promise.reject(error);
  }
);

// API Methods
export const apiService = {
  // Health
  async checkHealth(): Promise<HealthResponse> {
    const response = await api.get<HealthResponse>('/api/v1/health');
    return response.data;
  },

  // Denoising
  async denoiseSinogram(
    file: File,
    options: DenoiseRequest
  ): Promise<DenoiseResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const config: AxiosRequestConfig = {
      headers: { 'Content-Type': 'multipart/form-data' },
      params: {
        method: options.method,
        patch_size: options.patchSize || 16,
        return_metrics: options.returnMetrics ?? true,
      },
    };

    const response = await api.post<DenoiseResponse>(
      '/api/v1/denoise',
      formData,
      config
    );
    return response.data;
  },

  async getDenoiseResult(jobId: string): Promise<Blob> {
    const response = await api.get(`/api/v1/denoise/${jobId}/result`, {
      responseType: 'blob',
    });
    return response.data;
  },

  // Reconstruction
  async reconstructImage(
    file: File,
    filterName: string = 'ramp'
  ): Promise<{ job_id: string; status: string }> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/api/v1/reconstruct', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      params: { filter_name: filterName },
    });
    return response.data;
  },

  // Training
  async startTraining(request: TrainRequest): Promise<TrainResponse> {
    const response = await api.post<TrainResponse>('/api/v1/train', {
      epochs: request.epochs,
      batch_size: request.batchSize,
      learning_rate: request.learningRate,
      num_qubits: request.numQubits,
      vqc_layers: request.vqcLayers,
      validation_split: request.validationSplit || 0.1,
    });
    return response.data;
  },

  async getTrainingStatus(jobId: string): Promise<JobStatus> {
    const response = await api.get<JobStatus>(`/api/v1/train/${jobId}`);
    return response.data;
  },

  // Metrics
  async getBenchmarkMetrics(): Promise<MetricData[]> {
    const response = await api.get<MetricData[]>('/api/v1/metrics');
    return response.data;
  },

  // File Upload
  async uploadDicom(files: File[]): Promise<{ uploaded: number; files: any[] }> {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

    const response = await api.post('/api/v1/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  // Authentication
  async login(username: string, password: string): Promise<{ access_token: string }> {
    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);

    const response = await api.post('/token', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });

    const { access_token } = response.data;
    localStorage.setItem('auth_token', access_token);
    return response.data;
  },

  logout(): void {
    localStorage.removeItem('auth_token');
  },

  isAuthenticated(): boolean {
    return !!localStorage.getItem('auth_token');
  },
};

// WebSocket for real-time progress
export class ProgressWebSocket {
  private ws: WebSocket | null = null;
  private jobId: string;
  private onProgress: (data: any) => void;
  private onComplete: (data: any) => void;
  private onError: (error: Error) => void;

  constructor(
    jobId: string,
    callbacks: {
      onProgress: (data: any) => void;
      onComplete: (data: any) => void;
      onError: (error: Error) => void;
    }
  ) {
    this.jobId = jobId;
    this.onProgress = callbacks.onProgress;
    this.onComplete = callbacks.onComplete;
    this.onError = callbacks.onError;
  }

  connect(): void {
    const wsBase = API_BASE_URL || `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`;
    const wsUrl = wsBase.replace(/^http/, 'ws');
    this.ws = new WebSocket(`${wsUrl}/ws/progress/${this.jobId}`);

    this.ws.onopen = () => {
      console.log(`WebSocket connected for job ${this.jobId}`);
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.status === 'completed' || data.status === 'failed') {
        this.onComplete(data);
        this.disconnect();
      } else {
        this.onProgress(data);
      }
    };

    this.ws.onerror = (error) => {
      this.onError(new Error('WebSocket error'));
    };

    this.ws.onclose = () => {
      console.log(`WebSocket closed for job ${this.jobId}`);
    };
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

export default api;
