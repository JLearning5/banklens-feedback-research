import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize


def save_figure(fig, axes, path, figure_id, dpi):
    for ax in np.asarray(axes).ravel():
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print('FIGURE_ID', figure_id, flush=True)
    plt.show()
    plt.close(fig)


def report_model(y, scores, classes, model_id, output):
    predicted = scores.argmax(1)
    text = classification_report(y, predicted, labels=np.arange(len(classes)), target_names=classes, zero_division=0)
    report = classification_report(y, predicted, labels=np.arange(len(classes)), target_names=classes, zero_division=0, output_dict=True)
    print('CLASSIFICATION_REPORT', model_id, flush=True)
    print(text, flush=True)
    (output / f'{model_id}_classification_report.txt').write_text(text)
    pd.DataFrame(report).T.to_csv(output / f'{model_id}_classification_report.csv')
    return {'macro_f1': f1_score(y, predicted, average='macro', zero_division=0), 'accuracy': float(np.mean(predicted == y)), 'report': report}


def confusion_figure(y, scores, classes, model_id, output, dpi):
    matrix = confusion_matrix(y, scores.argmax(1), labels=np.arange(len(classes)))
    image = np.ones((*matrix.shape, 3))
    blue = np.array([47, 107, 255]) / 255
    orange = np.array([242, 142, 43]) / 255
    for i in range(len(classes)):
        for j in range(len(classes)):
            if matrix[i, j]:
                image[i, j] = blue if i == j else orange
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(image)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if i == j and matrix[i, j] else 'black')
    ax.set_xticks(range(len(classes)), classes, rotation=25, ha='right')
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel('Predicted sentiment')
    ax.set_ylabel('Actual sentiment')
    ax.set_title(model_id + ' validation counts')
    save_figure(fig, [ax], output / f'{model_id}_confusion.png', model_id + '_confusion', dpi)


def comparison_curves(y, predictions, classes, output, dpi):
    binary = label_binarize(y, classes=np.arange(len(classes)))
    panels = classes + ['micro', 'macro']
    for kind in ['roc', 'pr']:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for panel, ax in zip(panels, axes.ravel()):
            for model_id, scores in predictions.items():
                if panel in classes:
                    index = classes.index(panel)
                    actual, values = binary[:, index], scores[:, index]
                elif panel == 'micro':
                    actual, values = binary.ravel(), scores.ravel()
                else:
                    grid = np.linspace(0, 1, 1001)
                    curves, areas = [], []
                    for index in range(len(classes)):
                        if kind == 'roc':
                            x, yy, _ = roc_curve(binary[:, index], scores[:, index])
                            areas.append(roc_auc_score(binary[:, index], scores[:, index]))
                        else:
                            precision, recall, _ = precision_recall_curve(binary[:, index], scores[:, index])
                            x, yy = recall[::-1], precision[::-1]
                            areas.append(average_precision_score(binary[:, index], scores[:, index]))
                        curves.append(np.interp(grid, x, yy))
                    ax.plot(grid, np.mean(curves, axis=0), label=f'{model_id} {"AUC" if kind == "roc" else "AP"}={np.mean(areas):.3f}')
                    continue
                if kind == 'roc':
                    x, yy, _ = roc_curve(actual, values)
                    area = roc_auc_score(actual, values)
                else:
                    yy, x, _ = precision_recall_curve(actual, values)
                    area = average_precision_score(actual, values)
                ax.plot(x, yy, label=f'{model_id} {"AUC" if kind == "roc" else "AP"}={area:.3f}')
            if kind == 'roc':
                ax.plot([0, 1], [0, 1], '--', color='#A0A0A0', label='Chance AUC=0.500')
                ax.set_xlabel('False-positive rate')
                ax.set_ylabel('True-positive rate')
            else:
                prevalence = float(binary[:, classes.index(panel)].mean()) if panel in classes else 1 / len(classes)
                ax.axhline(prevalence, linestyle='--', color='#A0A0A0', label=f'Chance AP={prevalence:.3f}')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
            ax.set_title(panel)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.legend(fontsize=7)
        if len(panels) < len(axes.ravel()):
            ax = axes.ravel()[-1]
            for model_id, scores in predictions.items():
                threshold_grid = np.linspace(0, 1, 51)
                coverage = [float(((scores.max(1) > 0) & (scores.max(1) >= threshold)).mean()) for threshold in threshold_grid]
                ax.plot(threshold_grid, coverage, label=model_id)
            ax.set_title('Decision coverage')
            ax.set_xlabel('Maximum class score threshold')
            ax.set_ylabel('Coverage')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.legend(fontsize=7)
        save_figure(fig, axes, output / f'combined_{kind}.png', 'combined_' + kind, dpi)
    grid = np.linspace(0, 1, 51)
    curve_rows = []
    for model_id, scores in predictions.items():
        for threshold in grid:
            accepted = (scores.max(1) > 0) & (scores.max(1) >= threshold)
            predicted = scores.argmax(1)
            predicted[~accepted] = -1
            precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, labels=np.arange(len(classes)), average='macro', zero_division=0)
            curve_rows.append({'model': model_id, 'threshold': float(threshold), 'precision': precision, 'recall': recall, 'f1': f1, 'coverage': float(accepted.mean())})
    frame = pd.DataFrame(curve_rows)
    frame.to_csv(output / 'confidence_curves.csv', index=False)
    for metric in ['precision', 'recall', 'f1']:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for model_id, group in frame.groupby('model', sort=False):
            ax.plot(group['threshold'], group[metric], label=model_id)
        ax.set_xlabel('Maximum class score threshold')
        ax.set_ylabel('Macro ' + metric)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_title('Validation ' + metric + ' with abstentions')
        ax.legend(fontsize=8)
        save_figure(fig, [ax], output / f'combined_{metric}_confidence.png', 'combined_' + metric + '_confidence', dpi)
    return frame


def choose_threshold(y, scores, target):
    candidates = []
    for threshold in np.linspace(0, 1, 101):
        mask = (scores.max(1) > 0) & (scores.max(1) >= threshold)
        precision = float(np.mean(scores.argmax(1)[mask] == y[mask])) if mask.any() else None
        if precision is not None and precision >= target:
            candidates.append({'threshold': float(threshold), 'coverage': float(mask.mean()), 'accepted_accuracy': precision})
    return max(candidates, key=lambda x: (x['coverage'], -x['threshold'])) if candidates else {'threshold': 1.000001, 'coverage': 0.0, 'accepted_accuracy': None}


def resources_figure(samples, output, dpi):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for model_id, rows in samples.items():
        frame = pd.DataFrame(rows)
        axes[0].plot(frame['seconds'], frame['rss_mb'], marker='.', markersize=2, label=model_id)
        axes[1].plot(frame['seconds'], frame['cpu_percent'], marker='.', markersize=2, label=model_id)
    axes[0].set_ylabel('Resident memory (MB)')
    axes[1].set_ylabel('CPU allocation (%)')
    axes[1].set_xlabel('Elapsed time (seconds)')
    axes[0].set_title('Benchmark resource measurements')
    axes[0].legend(fontsize=8)
    save_figure(fig, axes, output / 'resources.png', 'resources', dpi)
