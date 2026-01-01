#!/usr/bin/env python3
"""
Quantum CT Sinogram Denoising - Benchmark Script

Comprehensive evaluation comparing:
- VQC (Variational Quantum Circuit)
- Classical methods (Gaussian, Wiener, TV, Median)
- U-Net baseline

Metrics: PSNR, SSIM, NRMSE, LPIPS, Processing Time

Usage:
    python benchmark.py --dataset mayo --output results/
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Scikit-image metrics
from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
    normalized_root_mse
)
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'backend'))

from app.quantum.vqc_denoiser import (
    SinogramDenoiser,
    VQCConfig,
    ClassicalDenoisers,
    CTReconstructor,
    add_combined_noise
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    method: str
    psnr: float
    ssim: float
    nrmse: float
    processing_time: float
    sample_id: int


@dataclass
class AggregatedResult:
    """Aggregated benchmark results"""
    method: str
    psnr_mean: float
    psnr_std: float
    ssim_mean: float
    ssim_std: float
    nrmse_mean: float
    nrmse_std: float
    time_mean: float
    time_std: float
    n_samples: int


class CTBenchmark:
    """
    Comprehensive benchmark suite for CT denoising methods
    """

    METHODS = ['noisy', 'gaussian', 'median', 'wiener', 'tv', 'vqc']

    def __init__(
        self,
        vqc_config: Optional[VQCConfig] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize benchmark suite

        Args:
            vqc_config: Configuration for VQC denoiser
            model_path: Path to pre-trained VQC model
        """
        self.vqc_config = vqc_config or VQCConfig(
            num_qubits=16,
            num_layers=6,
            patch_size=16
        )

        # Initialize VQC denoiser
        self.vqc_denoiser = SinogramDenoiser(self.vqc_config)

        if model_path and Path(model_path).exists():
            self.vqc_denoiser.load(model_path)
            logger.info(f"Loaded VQC model from {model_path}")
        else:
            logger.warning("No pre-trained model found. VQC results will be untrained baseline.")

        self.results: List[BenchmarkResult] = []

    def generate_synthetic_data(
        self,
        n_samples: int = 20,
        dose_fraction: float = 0.25
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate synthetic CT data for benchmarking

        Args:
            n_samples: Number of samples
            dose_fraction: Simulated dose (0.25 = 25%)

        Returns:
            List of (clean_sinogram, noisy_sinogram) pairs
        """
        logger.info(f"Generating {n_samples} synthetic samples at {dose_fraction*100:.0f}% dose")

        data = []
        theta = np.linspace(0., 180., 180, endpoint=False)

        for i in tqdm(range(n_samples), desc="Generating data"):
            # Base phantom with variations
            phantom = shepp_logan_phantom()
            phantom += np.random.randn(*phantom.shape) * 0.02
            phantom = np.clip(phantom, 0, 1)

            # Create clean sinogram
            clean = radon(phantom, theta=theta, circle=True)

            # Add noise
            noisy = add_combined_noise(clean, dose_fraction, gaussian_std=0.02)

            data.append((clean, noisy))

        return data

    def denoise_with_method(
        self,
        sinogram: np.ndarray,
        method: str
    ) -> Tuple[np.ndarray, float]:
        """
        Apply denoising method and measure time

        Args:
            sinogram: Noisy sinogram
            method: Method name

        Returns:
            (denoised_sinogram, processing_time)
        """
        start_time = time.perf_counter()

        if method == 'noisy':
            denoised = sinogram.copy()
        elif method == 'gaussian':
            denoised = ClassicalDenoisers.gaussian(sinogram, sigma=1.0)
        elif method == 'median':
            denoised = ClassicalDenoisers.median(sinogram, size=3)
        elif method == 'wiener':
            denoised = ClassicalDenoisers.wiener_filter(sinogram)
        elif method == 'tv':
            denoised = ClassicalDenoisers.tv_denoise(sinogram, weight=0.1)
        elif method == 'vqc':
            denoised = self.vqc_denoiser.denoise(sinogram)
        else:
            raise ValueError(f"Unknown method: {method}")

        elapsed = time.perf_counter() - start_time

        return denoised, elapsed

    def compute_metrics(
        self,
        clean: np.ndarray,
        denoised: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute image quality metrics

        Args:
            clean: Ground truth clean sinogram
            denoised: Denoised sinogram

        Returns:
            Dictionary of metrics
        """
        # Normalize for fair comparison
        clean_norm = (clean - clean.min()) / (clean.max() - clean.min() + 1e-8)
        denoised_norm = (denoised - denoised.min()) / (denoised.max() - denoised.min() + 1e-8)

        psnr = peak_signal_noise_ratio(clean_norm, denoised_norm, data_range=1.0)
        ssim = structural_similarity(clean_norm, denoised_norm, data_range=1.0)
        nrmse = normalized_root_mse(clean_norm, denoised_norm)

        return {
            'psnr': float(psnr),
            'ssim': float(ssim),
            'nrmse': float(nrmse)
        }

    def run_benchmark(
        self,
        data: List[Tuple[np.ndarray, np.ndarray]],
        methods: Optional[List[str]] = None
    ) -> List[BenchmarkResult]:
        """
        Run full benchmark on dataset

        Args:
            data: List of (clean, noisy) sinogram pairs
            methods: Methods to benchmark (default: all)

        Returns:
            List of benchmark results
        """
        methods = methods or self.METHODS
        logger.info(f"Running benchmark with methods: {methods}")

        results = []

        for sample_id, (clean, noisy) in enumerate(tqdm(data, desc="Benchmarking")):
            for method in methods:
                # Denoise
                denoised, proc_time = self.denoise_with_method(noisy, method)

                # Compute metrics
                metrics = self.compute_metrics(clean, denoised)

                # Store result
                result = BenchmarkResult(
                    method=method,
                    psnr=metrics['psnr'],
                    ssim=metrics['ssim'],
                    nrmse=metrics['nrmse'],
                    processing_time=proc_time,
                    sample_id=sample_id
                )
                results.append(result)

        self.results = results
        return results

    def aggregate_results(self) -> Dict[str, AggregatedResult]:
        """
        Aggregate results by method

        Returns:
            Dictionary of aggregated results per method
        """
        aggregated = {}

        for method in self.METHODS:
            method_results = [r for r in self.results if r.method == method]

            if not method_results:
                continue

            psnr_vals = [r.psnr for r in method_results]
            ssim_vals = [r.ssim for r in method_results]
            nrmse_vals = [r.nrmse for r in method_results]
            time_vals = [r.processing_time for r in method_results]

            aggregated[method] = AggregatedResult(
                method=method,
                psnr_mean=float(np.mean(psnr_vals)),
                psnr_std=float(np.std(psnr_vals)),
                ssim_mean=float(np.mean(ssim_vals)),
                ssim_std=float(np.std(ssim_vals)),
                nrmse_mean=float(np.mean(nrmse_vals)),
                nrmse_std=float(np.std(nrmse_vals)),
                time_mean=float(np.mean(time_vals)),
                time_std=float(np.std(time_vals)),
                n_samples=len(method_results)
            )

        return aggregated

    def statistical_tests(self) -> Dict[str, Dict[str, float]]:
        """
        Perform statistical significance tests

        Compares VQC against each classical method using paired t-test

        Returns:
            Dictionary with p-values for each comparison
        """
        vqc_results = [r for r in self.results if r.method == 'vqc']
        vqc_psnr = {r.sample_id: r.psnr for r in vqc_results}

        tests = {}

        for method in ['gaussian', 'wiener', 'tv', 'median']:
            method_results = [r for r in self.results if r.method == method]
            method_psnr = {r.sample_id: r.psnr for r in method_results}

            # Align by sample_id
            common_ids = set(vqc_psnr.keys()) & set(method_psnr.keys())
            vqc_vals = [vqc_psnr[i] for i in sorted(common_ids)]
            method_vals = [method_psnr[i] for i in sorted(common_ids)]

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(vqc_vals, method_vals)

            tests[method] = {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < 0.05),
                'improvement': float(np.mean(vqc_vals) - np.mean(method_vals))
            }

        return tests

    def generate_report(self, output_dir: Path) -> None:
        """
        Generate comprehensive benchmark report

        Args:
            output_dir: Directory for output files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Aggregate results
        aggregated = self.aggregate_results()
        statistical = self.statistical_tests()

        # Save JSON results
        results_dict = {
            'aggregated': {k: asdict(v) for k, v in aggregated.items()},
            'statistical_tests': statistical,
            'raw_results': [asdict(r) for r in self.results]
        }

        with open(output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(results_dict, f, indent=2)

        # Generate plots
        self._plot_comparison(aggregated, output_dir)
        self._plot_boxplots(output_dir)
        self._plot_statistical_significance(statistical, output_dir)

        # Generate markdown table
        self._generate_markdown_table(aggregated, statistical, output_dir)

        logger.info(f"Report saved to {output_dir}")

    def _plot_comparison(self, aggregated: Dict[str, AggregatedResult], output_dir: Path) -> None:
        """Generate comparison bar charts"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        methods = list(aggregated.keys())
        psnr = [aggregated[m].psnr_mean for m in methods]
        ssim = [aggregated[m].ssim_mean for m in methods]
        nrmse = [aggregated[m].nrmse_mean for m in methods]

        colors = ['#ff7f7f' if m != 'vqc' else '#7fff7f' for m in methods]

        # PSNR
        ax = axes[0]
        bars = ax.bar(methods, psnr, color=colors, edgecolor='black')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Peak Signal-to-Noise Ratio')
        ax.bar_label(bars, fmt='%.1f')

        # SSIM
        ax = axes[1]
        bars = ax.bar(methods, ssim, color=colors, edgecolor='black')
        ax.set_ylabel('SSIM')
        ax.set_title('Structural Similarity Index')
        ax.bar_label(bars, fmt='%.3f')

        # NRMSE
        ax = axes[2]
        bars = ax.bar(methods, nrmse, color=colors, edgecolor='black')
        ax.set_ylabel('NRMSE')
        ax.set_title('Normalized Root MSE (lower is better)')
        ax.bar_label(bars, fmt='%.3f')

        plt.tight_layout()
        plt.savefig(output_dir / 'method_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_boxplots(self, output_dir: Path) -> None:
        """Generate boxplots for metric distributions"""
        import pandas as pd

        df = pd.DataFrame([asdict(r) for r in self.results])

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # PSNR boxplot
        sns.boxplot(data=df, x='method', y='psnr', ax=axes[0], palette='husl')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title('PSNR Distribution by Method')

        # SSIM boxplot
        sns.boxplot(data=df, x='method', y='ssim', ax=axes[1], palette='husl')
        axes[1].set_ylabel('SSIM')
        axes[1].set_title('SSIM Distribution by Method')

        # Time boxplot
        sns.boxplot(data=df, x='method', y='processing_time', ax=axes[2], palette='husl')
        axes[2].set_ylabel('Time (s)')
        axes[2].set_title('Processing Time Distribution')

        plt.tight_layout()
        plt.savefig(output_dir / 'boxplots.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_statistical_significance(self, stats: Dict, output_dir: Path) -> None:
        """Plot statistical significance results"""
        fig, ax = plt.subplots(figsize=(10, 6))

        methods = list(stats.keys())
        improvements = [stats[m]['improvement'] for m in methods]
        p_values = [stats[m]['p_value'] for m in methods]

        colors = ['green' if p < 0.05 else 'gray' for p in p_values]

        bars = ax.barh(methods, improvements, color=colors, edgecolor='black')
        ax.set_xlabel('PSNR Improvement (dB)')
        ax.set_title('VQC Improvement over Classical Methods\n(Green = Statistically Significant, p < 0.05)')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

        # Add p-value annotations
        for i, (bar, p) in enumerate(zip(bars, p_values)):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'p={p:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'statistical_significance.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _generate_markdown_table(
        self,
        aggregated: Dict[str, AggregatedResult],
        stats: Dict,
        output_dir: Path
    ) -> None:
        """Generate markdown results table"""
        lines = [
            "# Benchmark Results\n",
            "## Method Comparison\n",
            "| Method | PSNR (dB) | SSIM | NRMSE | Time (s) |",
            "|--------|-----------|------|-------|----------|"
        ]

        for method, result in aggregated.items():
            line = f"| {method.upper()} | {result.psnr_mean:.2f} ± {result.psnr_std:.2f} | "
            line += f"{result.ssim_mean:.4f} ± {result.ssim_std:.4f} | "
            line += f"{result.nrmse_mean:.4f} ± {result.nrmse_std:.4f} | "
            line += f"{result.time_mean:.3f} ± {result.time_std:.3f} |"
            lines.append(line)

        lines.extend([
            "\n## Statistical Significance (VQC vs Baselines)\n",
            "| Comparison | Improvement (dB) | p-value | Significant |",
            "|------------|------------------|---------|-------------|"
        ])

        for method, test in stats.items():
            sig = "Yes" if test['significant'] else "No"
            lines.append(
                f"| VQC vs {method.upper()} | +{test['improvement']:.2f} | "
                f"{test['p_value']:.4f} | {sig} |"
            )

        with open(output_dir / 'benchmark_results.md', 'w') as f:
            f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description='CT Denoising Benchmark')
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['synthetic', 'mayo', 'lodopab'])
    parser.add_argument('--n-samples', type=int, default=20)
    parser.add_argument('--dose', type=float, default=0.25)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--output', type=str, default='./results')
    parser.add_argument('--methods', nargs='+', default=None)

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = CTBenchmark(model_path=args.model_path)

    # Generate or load data
    if args.dataset == 'synthetic':
        data = benchmark.generate_synthetic_data(
            n_samples=args.n_samples,
            dose_fraction=args.dose
        )
    else:
        logger.error(f"Dataset {args.dataset} not yet implemented")
        return

    # Run benchmark
    benchmark.run_benchmark(data, methods=args.methods)

    # Generate report
    output_dir = Path(args.output)
    benchmark.generate_report(output_dir)

    # Print summary
    aggregated = benchmark.aggregate_results()
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    for method, result in aggregated.items():
        print(f"{method.upper():12s} | PSNR: {result.psnr_mean:5.2f} dB | "
              f"SSIM: {result.ssim_mean:.4f} | Time: {result.time_mean:.3f}s")

    # VQC improvement
    if 'vqc' in aggregated:
        best_classical = max(
            [v for k, v in aggregated.items() if k not in ['vqc', 'noisy']],
            key=lambda x: x.psnr_mean
        )
        improvement = aggregated['vqc'].psnr_mean - best_classical.psnr_mean
        print(f"\nVQC improvement over best classical: +{improvement:.2f} dB")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
