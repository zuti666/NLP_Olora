"""Include the explicitly callable implementation gates in unittest discovery."""
import unittest
import test_cj_direction_owned as direction
import test_cj_trainer_owned as trainer


def load_tests(loader, tests, pattern):
    functions = (
        direction.test_nonlinear_and_counts,
        direction.test_quadratic,
        direction.test_zero_and_compute_rollback,
        direction.test_apply_rollback,
        direction.test_old_exact_regression,
        trainer.test_registered_configs_and_rejections,
        trainer.test_actual_installed_batch_dynamic_budget_no_diagnostics,
        trainer.test_second_probe_audit_rejects_wrong_sign,
    )
    tests.addTests(unittest.FunctionTestCase(fn) for fn in functions)
    return tests
