import React from 'react';

// Simple toast implementation
// In production, use @radix-ui/react-toast with full styling

export interface ToastProps {
  title?: string;
  description?: string;
  variant?: 'default' | 'destructive';
}

export const Toaster: React.FC = () => {
  return <div id="toaster" className="fixed bottom-4 right-4 z-50" />;
};

export const useToast = () => {
  const toast = (props: ToastProps) => {
    const container = document.getElementById('toaster');
    if (!container) return;

    const toastEl = document.createElement('div');
    toastEl.className = `
      p-4 rounded-lg shadow-lg mb-2 animate-in slide-in-from-right
      ${props.variant === 'destructive' ? 'bg-red-600 text-white' : 'bg-card border'}
    `;
    toastEl.innerHTML = `
      <div class="font-medium">${props.title || ''}</div>
      <div class="text-sm text-muted-foreground">${props.description || ''}</div>
    `;

    container.appendChild(toastEl);

    setTimeout(() => {
      toastEl.remove();
    }, 5000);
  };

  return { toast };
};
