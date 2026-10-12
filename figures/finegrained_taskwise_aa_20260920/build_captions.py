"""Generate self-contained LaTeX captions for the five standalone figures."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = 'figures/finegrained_taskwise_aa_20260920/output'
DATASETS = [
    ('Aircraft', 'aircraft',
     'All runs use 40 epochs per task and seed 0; SDLoRA uses learning rate 0.02, '
     'whereas the other methods use 0.05.'),
    ('Cars', 'cars',
     'All runs use 40 epochs per task. EWCLoRA uses seed 1; the other methods use seed 0.'),
    ('CUB-200', 'cub200', 'All runs use 40 epochs per task and seed 0.'),
    ('Flowers', 'flowers',
     'All runs use 40 epochs per task. EWCLoRA uses seed 1; the other methods use seed 0.'),
    ('5Datasets', '5datasets',
     r'The stream order is CUB-200 $\rightarrow$ FGVC-Aircraft $\rightarrow$ '
     r'Stanford Cars $\rightarrow$ Oxford Flowers-102 $\rightarrow$ Oxford-IIIT Pet. '
     'InfLoRA, OLoRA, and SDLoRA use 40 epochs per task; the other methods use 30. '
     'EWCLoRA uses seed 2; the other methods use seed 0.'),
]
COMMON = (
    r'At stage $t$, $\mathrm{AA}_t=\frac{1}{t}\sum_{j=1}^{t} a_{t,j}$ '
    'averages test accuracies on all seen tasks, including the current task, with equal task weights. '
    r'Each curve shows one archived run ($n=1$ per method); no uncertainty band is estimated. '
    'The endpoint is FAA, and the mean over all stage points is stage-averaged AAA. '
    'Runs share ViT-B/16, LoRA rank 16, and batch size 128, while retaining documented '
    'method-specific configurations. EWCLoRA uses the eligible archived run with the highest '
    'FAA for this dataset, selected retrospectively using final test performance. '
)


def main():
    out = ['% Requires \\usepackage{graphicx}. Paths are relative to the O-LoRA workspace.\n'
           '% Standalone figures use figure* to preserve legend legibility in a two-column paper.\n']
    for name, stem, detail in DATASETS:
        caption = f'{name}: task-macro average accuracy after each learned task. ' + COMMON + detail
        out.append(
            '\\begin{figure*}[t]\n'
            '    \\centering\n'
            f'    \\includegraphics[width=0.78\\textwidth]{{{ROOT}/{stem}_taskwise_aa.pdf}}\n'
            f'    \\caption{{{caption}}}\n'
            f'    \\label{{fig:{stem}_taskwise_aa}}\n'
            '\\end{figure*}\n')
    out.append(
        '% === Updated horizontal overview requested by the user ===\n'
        '\\begin{figure*}[t]\n'
        '    \\centering\n'
        f'    \\includegraphics[width=\\textwidth]{{{ROOT}/all_datasets_taskwise_aa.pdf}}\n'
        '    \\caption{Taskwise average accuracy on Aircraft, Cars, CUB-200, Flowers, and '
        '5Datasets (left to right, matching the result table). At each stage, accuracy is '
        'averaged equally over all seen tasks, including the current task. Every curve '
        'shows one archived run ($n=1$). Shading is a fixed $\\pm0.2$ percentage-point '
        'illustrative band following the reference visualization; it is not a standard '
        'deviation, standard error, or confidence interval. Center curves are unchanged '
        'from the selected table runs. Runs retain documented method-specific configurations; '
        'EWCLoRA uses the highest-FAA eligible archived run per dataset, retrospectively '
        'selected using final test performance. SDLoRA/Aircraft uses learning rate 0.02 '
        '(other Aircraft runs: 0.05). On 5Datasets, InfLoRA/OLoRA/SDLoRA use 40 epochs per '
        'task and the other methods use 30; the other four datasets use 40. EWCLoRA uses '
        'seed 1 on Cars/Flowers and seed 2 on 5Datasets; other plotted runs use seed 0. '
        'The 5Datasets order is CUB-200 $\\rightarrow$ FGVC-Aircraft $\\rightarrow$ '
        'Stanford Cars $\\rightarrow$ Oxford Flowers-102 $\\rightarrow$ Oxford-IIIT Pet.}\n'
        '    \\label{fig:finegrained_taskwise_aa_overview}\n'
        '\\end{figure*}\n')
    (HERE / 'latex_includes.tex').write_text('\n'.join(out))


if __name__ == '__main__':
    main()
