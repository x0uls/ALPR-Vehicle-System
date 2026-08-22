import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def generate_benchmark_charts(easy_metrics, tess_metrics, output_path="outputs/charts/benchmark_dashboard.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor='#00040f')
    fig.subplots_adjust(hspace=0.38, wspace=0.25)
    
    easy_color, tess_color = '#38bdf8', '#818cf8'
    width = 0.35

    def style_ax(ax, title):
        ax.set_facecolor('#121215')
        ax.set_title(title, color='#e4e4e7', fontsize=11, fontweight='bold', pad=10)
        ax.tick_params(colors='#a1a1aa', labelsize=9)
        for s in ['top', 'right']: ax.spines[s].set_visible(False)
        for s in ['left', 'bottom']: ax.spines[s].set_color('#3f3f46')
        ax.grid(True, linestyle='--', alpha=0.3, color='#27272a')

    # Chart 1: Accuracy Benchmarks
    ax1 = axes[0, 0]
    style_ax(ax1, "Chart 1: Accuracy Benchmarks (EMA, Precision, CRR %)")
    acc_labels = ['Exact Match (EMA)', 'Char Precision', 'Char Recog (CRR)']
    easy_acc = [(easy_metrics.get("exact_match_rate", 0) or 0)*100, (easy_metrics.get("char_precision", 0) or 0)*100, easy_metrics.get("crr", 0) or 0]
    tess_acc = [(tess_metrics.get("exact_match_rate", 0) or 0)*100, (tess_metrics.get("char_precision", 0) or 0)*100, tess_metrics.get("crr", 0) or 0]
    x1 = np.arange(len(acc_labels))
    b1 = ax1.bar(x1 - width/2, easy_acc, width, label='EasyOCR', color=easy_color, alpha=0.85)
    b2 = ax1.bar(x1 + width/2, tess_acc, width, label='PyTesseract', color=tess_color, alpha=0.85)
    ax1.set_xticks(x1); ax1.set_xticklabels(acc_labels, fontsize=8, color='#a1a1aa'); ax1.set_ylim(0, 110); ax1.set_ylabel('Percentage (%)', color='#a1a1aa', fontsize=9)
    ax1.legend(facecolor='#18181b', edgecolor='#3f3f46', fontsize=8, loc='upper right')

    for bar, col in [(b1, easy_color), (b2, tess_color)]:
        for b in bar:
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width()/2., h + 1, f"{h:.1f}%", ha='center', va='bottom', color=col, fontsize=7, fontweight='bold')

    # Chart 2: Inference Latency
    ax2 = axes[0, 1]
    style_ax(ax2, "Chart 2: Inference Latency Per Crop (ms)")
    lat_vals = [float(easy_metrics.get("latency_per_plate_ms") or 0.0), float(tess_metrics.get("latency_per_plate_ms") or 0.0)]
    bars2 = ax2.bar([0, 1], lat_vals, width=0.45, color=[easy_color, tess_color], alpha=0.85)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(['EasyOCR', 'PyTesseract'], fontsize=9, color='#a1a1aa')
    ax2.set_ylabel('Latency (ms / crop)', color='#a1a1aa', fontsize=9)
    max_lat = max(lat_vals) if max(lat_vals) > 0 else 100
    ax2.set_ylim(0, max_lat * 1.25)
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width()/2., h + (max_lat * 0.02), f"{h:.1f} ms", ha='center', va='bottom', color='#e4e4e7', fontsize=8, fontweight='bold')

    # Chart 3: Levenshtein Edit Error Breakdown
    ax3 = axes[1, 0]
    style_ax(ax3, "Chart 3: Levenshtein Edit Error Breakdown")
    easy_errs = [easy_metrics.get("substitutions", 0), easy_metrics.get("insertions", 0), easy_metrics.get("deletions", 0)]
    tess_errs = [tess_metrics.get("substitutions", 0), tess_metrics.get("insertions", 0), tess_metrics.get("deletions", 0)]

    x3 = np.arange(3)
    b3_1 = ax3.bar(x3 - width/2, easy_errs, width, label='EasyOCR', color=easy_color, alpha=0.85)
    b3_2 = ax3.bar(x3 + width/2, tess_errs, width, label='PyTesseract', color=tess_color, alpha=0.85)
    ax3.set_xticks(x3); ax3.set_xticklabels(['Substitutions', 'Insertions', 'Deletions'], fontsize=9)
    ax3.set_ylabel('Count', color='#a1a1aa', fontsize=9); ax3.legend(facecolor='#18181b', edgecolor='#3f3f46', fontsize=8, loc='upper right')

    tot_easy_errs = sum(easy_errs)
    tot_tess_errs = sum(tess_errs)
    all_err_vals = easy_errs + tess_errs
    max_err = max(all_err_vals) if all_err_vals and max(all_err_vals) > 0 else 10
    ax3.set_ylim(0, max_err * 1.25)

    for bar, col, tot in [(b3_1, easy_color, tot_easy_errs), (b3_2, tess_color, tot_tess_errs)]:
        for b in bar:
            h = b.get_height()
            pct = (h / tot * 100) if tot > 0 else 0.0
            ax3.text(b.get_x() + b.get_width()/2., h + (max_err * 0.02), f"{int(h)} ({pct:.1f}%)", ha='center', va='bottom', color=col, fontsize=7, fontweight='bold')

    # Chart 4: Character Precision, Recall & Confidence
    ax4 = axes[1, 1]
    style_ax(ax4, "Chart 4: Character Precision, Recall & Confidence")
    easy_macro = [(easy_metrics.get("char_precision") or 0)*100, (easy_metrics.get("char_recall") or 0)*100, (easy_metrics.get("average_confidence") or 0)*100]
    tess_macro = [(tess_metrics.get("char_precision") or 0)*100, (tess_metrics.get("char_recall") or 0)*100, (tess_metrics.get("average_confidence") or 0)*100]
    x4 = np.arange(3)
    b4_1 = ax4.bar(x4 - width/2, easy_macro, width, label='EasyOCR', color=easy_color, alpha=0.85)
    b4_2 = ax4.bar(x4 + width/2, tess_macro, width, label='PyTesseract', color=tess_color, alpha=0.85)
    ax4.set_xticks(x4); ax4.set_xticklabels(['Char Precision %', 'Char Recall %', 'Avg Confidence %'], rotation=5, ha='center', fontsize=8)
    ax4.set_ylim(0, 110); ax4.set_ylabel('Score (%)', color='#a1a1aa', fontsize=9); ax4.legend(facecolor='#18181b', edgecolor='#3f3f46', fontsize=8, loc='upper right')

    for bar, col in [(b4_1, easy_color), (b4_2, tess_color)]:
        for b in bar:
            h = b.get_height()
            ax4.text(b.get_x() + b.get_width()/2., h + 1, f"{h:.1f}%", ha='center', va='bottom', color=col, fontsize=7, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return "/" + output_path.replace("\\", "/")
