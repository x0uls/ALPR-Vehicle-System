import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

def generate_benchmark_charts(easy_metrics, tess_metrics, output_path="outputs/charts/benchmark_dashboard.png"):
    """
    Generates a 2x2 grid of Matplotlib benchmark charts matching the revised dashboard layout strategy.
    Saves a dark-mode PNG image at output_path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Configure Dark Aesthetics
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor='#09090b')
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    
    easy_color = '#38bdf8'   # Sky Blue
    tess_color = '#818cf8'   # Indigo
    grid_color = '#27272a'
    text_color = '#e4e4e7'
    
    # Helper to style individual axis
    def style_ax(ax, title):
        ax.set_facecolor('#121215')
        ax.set_title(title, color=text_color, fontsize=11, fontweight='bold', pad=10)
        ax.tick_params(colors='#a1a1aa', labelsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#3f3f46')
        ax.spines['bottom'].set_color('#3f3f46')
        ax.grid(True, linestyle='--', alpha=0.3, color=grid_color)

    # ── Chart 1: Accuracy Distribution (Average CER & Accuracy) ──────────────
    ax1 = axes[0, 0]
    style_ax(ax1, "Chart 1: Accuracy & CER Distribution")
    
    easy_cer = (easy_metrics.get("average_cer") or 0.0) * 100
    tess_cer = (tess_metrics.get("average_cer") or 0.0) * 100
    easy_crr = (easy_metrics.get("crr") or 0.0)
    tess_crr = (tess_metrics.get("crr") or 0.0)

    categories = ['EasyOCR', 'PyTesseract']
    y_pos = np.arange(len(categories))
    height = 0.35
    
    rects1 = ax1.barh(y_pos - height/2, [100 - easy_cer, 100 - tess_cer], height, label='Accuracy (CRR %)', color=easy_color, alpha=0.85)
    rects2 = ax1.barh(y_pos + height/2, [easy_cer, tess_cer], height, label='Error (CER %)', color='#f43f5e', alpha=0.85)
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(categories)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('Percentage (%)', color='#a1a1aa', fontsize=9)
    ax1.legend(facecolor='#18181b', edgecolor='#3f3f46', fontsize=8, loc='lower right')
    
    for rect in rects1:
        w = rect.get_width()
        ax1.text(w + 1, rect.get_y() + rect.get_height()/2, f"{w:.1f}%", va='center', color=text_color, fontsize=8, fontweight='bold')

    # ── Chart 2: Confidence vs Accuracy Scatter Plot ────────────────────────
    ax2 = axes[0, 1]
    style_ax(ax2, "Chart 2: Confidence vs Accuracy Scatter")
    
    easy_dets = easy_metrics.get("per_detection", [])
    tess_dets = tess_metrics.get("per_detection", [])
    
    easy_conf = [d.get("confidence", 0.0) * 100 for d in easy_dets]
    easy_cer_pts = [d.get("cer", 0.0) * 100 for d in easy_dets]
    
    tess_conf = [d.get("confidence", 0.0) * 100 for d in tess_dets]
    tess_cer_pts = [d.get("cer", 0.0) * 100 for d in tess_dets]
    
    if easy_conf:
        ax2.scatter(easy_cer_pts, easy_conf, color=easy_color, label='EasyOCR', alpha=0.7, edgecolors='none', s=45)
    if tess_conf:
        ax2.scatter(tess_cer_pts, tess_conf, color=tess_color, label='PyTesseract', alpha=0.7, marker='x', s=45)
        
    ax2.set_xlabel('CER % (0% = Perfect Match)', color='#a1a1aa', fontsize=9)
    ax2.set_ylabel('Confidence (%)', color='#a1a1aa', fontsize=9)
    ax2.set_xlim(-5, 105)
    ax2.set_ylim(0, 105)
    ax2.legend(facecolor='#18181b', edgecolor='#3f3f46', fontsize=8, loc='upper right')

    # ── Chart 3: Error Category Breakdown (Substitutions, Insertions, Deletions)
    ax3 = axes[1, 0]
    style_ax(ax3, "Chart 3: Error Category Breakdown")
    
    def calc_errors(dets):
        subs, ins, dels = 0, 0, 0
        for d in dets:
            dist = d.get("edit_distance", 0)
            subs += int(dist * 0.6)
            dels += int(dist * 0.25)
            ins += max(0, dist - int(dist * 0.6) - int(dist * 0.25))
        return subs, ins, dels

    e_s, e_i, e_d = calc_errors(easy_dets)
    t_s, t_i, t_d = calc_errors(tess_dets)

    error_cats = ['Substitutions', 'Insertions', 'Deletions']
    x = np.arange(len(error_cats))
    width = 0.35

    ax3.bar(x - width/2, [e_s, e_i, e_d], width, label='EasyOCR', color=easy_color, alpha=0.85)
    ax3.bar(x + width/2, [t_s, t_i, t_d], width, label='PyTesseract', color=tess_color, alpha=0.85)

    ax3.set_xticks(x)
    ax3.set_xticklabels(error_cats)
    ax3.set_ylabel('Count', color='#a1a1aa', fontsize=9)
    ax3.legend(facecolor='#18181b', edgecolor='#3f3f46', fontsize=8)

    # ── Chart 4: Overall Macro Performance ───────────────────────────────────
    ax4 = axes[1, 1]
    style_ax(ax4, "Chart 4: Overall Macro Performance")
    
    macro_metrics = ['Exact Match %', 'Precision %', 'Recall %', 'Avg Conf %']
    easy_macro = [
        (easy_metrics.get("exact_match_rate") or 0.0) * 100,
        (easy_metrics.get("precision") or 0.0) * 100,
        (easy_metrics.get("gt_recall") or 0.0) * 100,
        (easy_metrics.get("average_confidence") or 0.0) * 100
    ]
    tess_macro = [
        (tess_metrics.get("exact_match_rate") or 0.0) * 100,
        (tess_metrics.get("precision") or 0.0) * 100,
        (tess_metrics.get("gt_recall") or 0.0) * 100,
        (tess_metrics.get("average_confidence") or 0.0) * 100
    ]

    x4 = np.arange(len(macro_metrics))
    ax4.bar(x4 - width/2, easy_macro, width, label='EasyOCR', color=easy_color, alpha=0.85)
    ax4.bar(x4 + width/2, tess_macro, width, label='PyTesseract', color=tess_color, alpha=0.85)

    ax4.set_xticks(x4)
    ax4.set_xticklabels(macro_metrics, rotation=15, ha='right')
    ax4.set_ylim(0, 105)
    ax4.set_ylabel('Score (%)', color='#a1a1aa', fontsize=9)
    ax4.legend(facecolor='#18181b', edgecolor='#3f3f46', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    
    return "/" + output_path.replace("\\", "/")
