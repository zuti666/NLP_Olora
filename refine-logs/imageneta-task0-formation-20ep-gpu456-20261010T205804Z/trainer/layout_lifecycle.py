"""Class partitions and metrics shared by the one-task and CL experiments.

The single200 condition changes the dataset partition and classifier to 200
classes; it is not a max_train_tasks=1 truncation of the 20-class CL run.
"""
from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class TaskLayout:
    name: str
    increments: tuple

    @classmethod
    def from_name(cls, name):
        if name == 'single200':
            return cls(name, (200,))
        if name == 'cl10':
            return cls(name, (20,) * 10)
        raise ValueError('response_layout must be single200 or cl10')

    @property
    def tasks(self):
        return len(self.increments)

    @property
    def classes(self):
        return sum(self.increments)

    @property
    def boundaries(self):
        return tuple(self.class_range(t) for t in range(self.tasks))

    def class_range(self, task):
        if type(task) is not int or not 0 <= task < self.tasks:
            raise ValueError('Task outside the configured class partition')
        start = sum(self.increments[:task])
        return start, start + self.increments[task]

    def validate_data_manager(self, dm):
        if (dm.nb_classes != self.classes or dm.nb_tasks != self.tasks
                or tuple(dm._increments) != self.increments
                or list(dm._class_order) != list(range(self.classes))):
            raise AssertionError('DataManager differs from the configured complete class partition')
        if tuple(dm.get_task_class_range(t) for t in range(dm.nb_tasks)) != self.boundaries:
            raise AssertionError('DataManager task boundaries mismatch')

    def validate_batch(self, task, class_offset, targets, seen_classes, fc_width):
        lo, hi = self.class_range(task)
        if (class_offset != lo or seen_classes != hi or fc_width != hi
                or bool(((targets < lo) | (targets >= hi)).any())):
            raise AssertionError('Current-task CE, seen classes, or classifier width mismatch')
        return lo, hi


def seen_accuracy_row(cnn, task, seen_classes, history_rows, layout):
    """Retain the source's rounded grouped accuracies for exact CL compatibility."""
    if history_rows != task or seen_classes != layout.class_range(task)[1]:
        raise AssertionError('Evaluation task/history/seen-class mismatch')
    keys = ['%02d-%02d' % (lo, hi - 1) for lo, hi in layout.boundaries]
    grouped = cnn['grouped']
    numeric = {key for key in grouped if re.fullmatch(r'\d+-\d+', key)}
    if numeric - set(keys):
        raise AssertionError('Unexpected evaluation class boundary')
    if any(key not in grouped for key in keys[:task + 1]):
        raise AssertionError('Missing seen evaluation group')
    if any(key in grouped and not math.isnan(float(grouped[key])) for key in keys[task + 1:]):
        raise AssertionError('Future evaluation group contains an observation')
    row = [float(grouped[key]) for key in keys[:task + 1]]
    if any(not math.isfinite(x) or not 0 <= x <= 100 for x in row + [float(cnn['top1'])]):
        raise AssertionError('Invalid observed evaluation accuracy')
    return row


def matrix_metrics(matrix, sample_curve, layout):
    n = len(matrix)
    if not 1 <= n <= layout.tasks or len(sample_curve) != n:
        raise AssertionError('Invalid completed stage count')
    if any(len(row) != i + 1 for i, row in enumerate(matrix)):
        raise AssertionError('Task matrix must contain the triangular seen-task observations')
    values = [x for row in matrix for x in row] + list(sample_curve)
    if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 100 for x in values):
        raise AssertionError('Nonfinite/out-of-range matrix observations')
    faa = sum(matrix[-1]) / n
    aaa = sum(sum(row) / len(row) for row in matrix) / n
    acquisition = sum(matrix[t][t] for t in range(n)) / n
    bwt = sum(matrix[-1][j] - matrix[j][j] for j in range(n - 1)) / (n - 1) if n > 1 else None
    forgetting = (sum(max(matrix[t][j] for t in range(j, n)) - matrix[-1][j]
                       for j in range(n - 1)) / (n - 1) if n > 1 else None)
    return dict(matrix=matrix, response_layout=layout.name,
                task_boundaries=[list(x) for x in layout.boundaries],
                matrix_scope='native CNN argmax over all seen classes; rows=completed stage, columns=task',
                AAA_task_macro=aaa, FAA_task_macro=faa, LA=acquisition,
                BWT=bwt, forgetting=forgetting,
                final_test_accuracy=float(sample_curve[-1]),
                sample_accuracy_curve=list(sample_curve),
                FAA_sample_weighted=float(sample_curve[-1]),
                AAA_sample_weighted=sum(sample_curve) / n,
                AAA_definition='mean over completed task stages; not the epoch accuracy curve',
                single_task_metrics_note=('FAA and stage AAA equal final test accuracy; BWT/forgetting undefined'
                                          if layout.name == 'single200' else None))
