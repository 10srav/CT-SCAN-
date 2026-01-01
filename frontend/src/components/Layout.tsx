/**
 * Main Layout Component
 *
 * Provides the application shell with navigation sidebar
 */

import React from 'react';
import { Outlet, Link, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  Zap,
  GraduationCap,
  BarChart3,
  Settings,
  Github,
  Moon,
  Sun,
  Activity
} from 'lucide-react';
import { useTheme } from './ThemeProvider';
import { Button } from './ui/button';

interface NavItem {
  path: string;
  label: string;
  icon: React.ReactNode;
}

const navItems: NavItem[] = [
  { path: '/dashboard', label: 'Dashboard', icon: <LayoutDashboard className="h-5 w-5" /> },
  { path: '/denoise', label: 'Denoise', icon: <Zap className="h-5 w-5" /> },
  { path: '/training', label: 'Training', icon: <GraduationCap className="h-5 w-5" /> },
  { path: '/benchmark', label: 'Benchmark', icon: <BarChart3 className="h-5 w-5" /> },
  { path: '/settings', label: 'Settings', icon: <Settings className="h-5 w-5" /> },
];

export const Layout: React.FC = () => {
  const location = useLocation();
  const { theme, setTheme } = useTheme();

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  return (
    <div className="flex h-screen bg-background">
      {/* Sidebar */}
      <aside className="w-64 border-r bg-card flex flex-col">
        {/* Logo */}
        <div className="p-6 border-b">
          <Link to="/" className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <Activity className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-lg">Quantum CT</h1>
              <p className="text-xs text-muted-foreground">Sinogram Denoising</p>
            </div>
          </Link>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4">
          <ul className="space-y-2">
            {navItems.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <li key={item.path}>
                  <Link
                    to={item.path}
                    className={`
                      flex items-center gap-3 px-4 py-2.5 rounded-lg
                      transition-colors duration-200
                      ${isActive
                        ? 'bg-primary text-primary-foreground'
                        : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                      }
                    `}
                  >
                    {item.icon}
                    <span className="font-medium">{item.label}</span>
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Footer */}
        <div className="p-4 border-t space-y-4">
          {/* Theme Toggle */}
          <Button
            variant="outline"
            className="w-full justify-start gap-3"
            onClick={toggleTheme}
          >
            {theme === 'dark' ? (
              <>
                <Sun className="h-5 w-5" />
                Light Mode
              </>
            ) : (
              <>
                <Moon className="h-5 w-5" />
                Dark Mode
              </>
            )}
          </Button>

          {/* GitHub Link */}
          <a
            href="https://github.com/quantum-ct/quantum-ct-sinogram-denoise"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            <Github className="h-4 w-4" />
            View on GitHub
          </a>

          {/* Version Info */}
          <div className="text-xs text-muted-foreground">
            <p>Team 7A | 2025-26</p>
            <p>Guide: Mrs. S. Anusha</p>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;
