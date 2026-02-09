/**
 * Quantum CT Sinogram Denoising - Main Application
 *
 * A React-based dashboard for:
 * - Uploading and visualizing CT sinograms
 * - Running VQC denoising
 * - Comparing with classical methods
 * - Viewing reconstruction results
 */

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Components
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { Denoise } from './pages/Denoise';
import { Training } from './pages/Training';
import { Benchmark } from './pages/Benchmark';
import { Settings } from './pages/Settings';
import { Toaster } from './components/ui/toaster';

// Providers
import { ThemeProvider } from './components/ThemeProvider';

// Create Query Client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="dark" storageKey="quantum-ct-theme">
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Navigate to="/dashboard" replace />} />
              <Route path="dashboard" element={<Dashboard />} />
              <Route path="denoise" element={<Denoise />} />
              <Route path="training" element={<Training />} />
              <Route path="benchmark" element={<Benchmark />} />
              <Route path="settings" element={<Settings />} />
            </Route>
          </Routes>
          <Toaster />
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
