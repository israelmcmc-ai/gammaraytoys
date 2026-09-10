"""Tests for the expression evaluator behind a configured `FunctionScaling`
(`gammaraytoys.sims.config.TimeExpression`).

This is the one security-critical corner of the YAML configuration layer: the
expression is a string out of a file, and `t` is the only thing in it that is
supposed to vary. CONTRACT.md records that the defence the plan asks for --
a whitelisted namespace plus "reject anything containing `__`" -- is necessary
but not sufficient, because three attacks carry no `__` anywhere:

    9**9**9**9          hung the process, killed at 6 s
    (lambda: 1)()       allowed: new code, built and called
    t.real.conjugate()  allowed: attribute access is permitted

The implementation replaces that with an AST whitelist. What follows pins the
whitelist from both sides: every expression CONTRACT.md and TEST_BRIEF.md name
as an attack is rejected, and every expression they name as legitimate is
accepted *and evaluates to the number worked out by hand here*. Both halves
matter -- a whitelist that rejected everything would pass a rejection-only
file, and a whitelist that accepted everything would pass an acceptance-only
one.

Every expected value below comes from the schema, from those two briefs, or
from arithmetic done in the comment next to it -- never from running the
implementation and copying its output.

House style, matching `tests/test_scaling.py` and `tests/test_event_csv.py`:
statistical assertions state their sigma and assert at 4 sigma. There are no
statistical assertions here; the wall-clock budget below is the one number
that needed choosing, and its reasoning is spelled out where it is defined.
"""

import threading
import time

import numpy as np
import pytest
import astropy.units as u

from gammaraytoys.sims import FunctionScaling, TimeExpression, scaling_from_config


# ===========================================================================
# Rejection must be prompt, and a regression must not wedge the suite
# ===========================================================================

# The wall-clock budget one rejection may take.
#
# Why a budget at all: `9**9**9**9` is pure arithmetic with no `__` in it, and
# under the plan's own string-matching rule it did not raise -- it *hung*,
# and had to be killed at 6 s (CONTRACT.md). A test that asserted only
# `pytest.raises(ValueError)` would happily pass a rule that takes six seconds
# to say no, and a configuration file that costs six seconds of CPU to reject
# is a denial of service, not a defence.
#
# Why 0.25 s specifically:
#
#   * TEST_BRIEF.md records every one of these rejections measured at
#     <= 0.10 ms. 0.25 s is roughly 2500x that, so a CI runner would have to
#     be ~2500x slower than the machine that measured it before this flaked.
#     Rejection is `ast.parse` plus a walk over a handful of nodes; there is
#     no I/O and nothing to warm up, so there is no slow first call to absorb.
#
#   * The regression it has to catch is nowhere near that fast. Actually
#     evaluating `9**9**9**9` means building 9**(9**(9**9)) -- an integer of
#     some 370 million digits -- which ran past 6 s without finishing. There
#     is no version of that computation that comes in under a quarter of a
#     second, so a rule that lets it through cannot sneak past this bound.
#
# In short: three orders of magnitude of headroom above the real cost, and
# more than an order of magnitude below the failure it is watching for.
REJECTION_BUDGET_SECONDS = 0.25

# The hard stop, used only to keep a regression from HANGING the suite.
#
# A wall-clock assertion alone cannot save us here: it is only reached *after*
# the call returns, so if a future edit made the evaluator start actually
# computing `9**9**9**9`, the assertion would never run and CI would sit there
# until something else killed it. So each rejection is attempted on a worker
# thread and joined with a timeout; if the worker is still going after this
# long, the test FAILS and says why instead of waiting forever.
#
# The thread is a daemon: a wedged worker cannot be killed from outside in
# Python, but a daemon thread does not hold the interpreter open at exit, so
# the run still finishes and still reports the failure. This is all in-process
# and portable -- no subprocess per expression, nothing that only works on one
# platform, and no `pytest-timeout` (which this project does not depend on).
HANG_GUARD_SECONDS = 5.0


def _reject_promptly(expression):
    """
    Assert that building `TimeExpression(expression)` is refused, promptly,
    and return the `ValueError` it was refused with.

    Parameters
    ----------
    expression : str
        The expression that must be rejected.

    Returns
    -------
    ValueError
        The rejection, so a caller can assert on its message.
    """

    outcome = {}

    def attempt():
        started = time.perf_counter()
        try:
            TimeExpression(expression)
            outcome['verdict'] = 'accepted'
        except ValueError as err:
            outcome['verdict'] = 'rejected'
            outcome['error'] = err
        except BaseException as err:                    # noqa: BLE001
            outcome['verdict'] = 'other'
            outcome['error'] = err
        finally:
            outcome['seconds'] = time.perf_counter() - started

    worker = threading.Thread(target=attempt, daemon=True)
    worker.start()
    worker.join(HANG_GUARD_SECONDS)

    if worker.is_alive():
        pytest.fail(
            f"{expression!r} was still being processed after "
            f"{HANG_GUARD_SECONDS} s. It is being evaluated rather than "
            f"rejected -- that is the arithmetic denial of service the AST "
            f"whitelist exists to stop.")

    if outcome['verdict'] == 'accepted':
        pytest.fail(f"{expression!r} was accepted; it must be rejected.")

    if outcome['verdict'] == 'other':
        raise AssertionError(
            f"{expression!r} was refused with "
            f"{type(outcome['error']).__name__}, not the documented "
            f"ValueError.") from outcome['error']

    assert outcome['seconds'] < REJECTION_BUDGET_SECONDS, (
        f"{expression!r} took {outcome['seconds']:.3f} s to reject, over the "
        f"{REJECTION_BUDGET_SECONDS} s budget. A slow rejection is itself a "
        f"denial of service.")

    return outcome['error']


# ===========================================================================
# Part A -- everything the briefs name as an attack is rejected
# ===========================================================================

# The list from TEST_BRIEF.md section 5, each with the reason it is out.
REJECTED = [
    # The attack the plan names by name.
    '__import__("os").system("id")',
    # The classic subclass walk.
    '().__class__.__bases__[0]',
    # A dunder reached through a whitelisted callable.
    'min.__self__',
    # Arithmetic denial of service: no dunder, no call, no name -- just a
    # number with more digits than there is memory.
    '9**9**9**9',
    '10**1000',
    # Chained '**' that is not itself huge: the rule is structural, so it
    # does not depend on guessing how big the result would be.
    '2**t**2',
    # Building and calling new code.
    '(lambda: 1)()',
    'lambda: 1',
    # Attribute access, the hole that no amount of string matching closes.
    't.real.conjugate()',
    't.real',
    # A function named but not called is not a number.
    'sin',
    # Node types that are simply not on the whitelist.
    '[x for x in [1,2]]',
    'sin(t) if t>0 else 0',
]


@pytest.mark.parametrize('expression', REJECTED)
def test_dangerous_expression_is_rejected_promptly(expression):
    error = _reject_promptly(expression)

    # The message must quote the expression back, so a user with a hundred-line
    # configuration file knows which line to go and fix.
    assert expression in str(error)


@pytest.mark.parametrize('expression', REJECTED)
def test_dangerous_expression_is_rejected_through_scaling_from_config(expression):
    # The configuration layer must not have its own, softer, path to the same
    # place: a `scaling` block goes through the identical check.
    with pytest.raises(ValueError):
        scaling_from_config({'type': 'Function', 'expression': expression})


def test_import_attack_is_rejected_before_anything_is_imported():
    # The point of the whole exercise: `__import__("os").system("id")` must be
    # refused at construction, not at the first interval of a run, and must not
    # execute on the way to being refused.
    error = _reject_promptly('__import__("os").system("id")')
    assert 'not allowed' in str(error)


def test_attribute_access_is_named_as_the_reason_it_is_out():
    # `t.real` is the minimal form of `t.real.conjugate()`: nothing reachable
    # through it is useful *today*, and that is exactly why the rejection has
    # to be structural rather than a judgement about today's namespace.
    error = _reject_promptly('t.real')
    assert 'attribute' in str(error).lower()


def test_lambda_is_named_as_the_reason_it_is_out():
    error = _reject_promptly('lambda: 1')
    assert 'lambda' in str(error).lower()


def test_a_node_type_not_on_the_whitelist_is_rejected_by_default():
    # The whitelist is a whitelist: a syntax nobody thought about is out
    # because it was never let in, not because someone remembered to ban it.
    for expression in ('[1, 2, 3]', '{1: 2}', '(1, 2)', 't if t else t',
                       '[x for x in [1, 2]]', '(t := 2)', 'f"{t}"', 't < 2',
                       't and t', 'not t', '~1', 't[0]', 'sin(*[t])',
                       'sin(x=t)'):
        _reject_promptly(expression)


def test_a_string_constant_is_rejected():
    # Strings are not real numbers, and a string is the raw material of every
    # dunder-by-name trick.
    _reject_promptly('"os"')


def test_a_boolean_constant_is_rejected():
    # `True` is an `int` to `isinstance`, so this one needs its own check in
    # the implementation and its own test here.
    _reject_promptly('True')


def test_an_undefined_name_is_rejected():
    _reject_promptly('u')


def test_a_non_string_expression_is_rejected():
    with pytest.raises(ValueError, match='must be a string'):
        TimeExpression(1.0)


def test_a_syntax_error_is_reported_as_such():
    with pytest.raises(ValueError, match='not valid syntax'):
        TimeExpression('1 +')


# ===========================================================================
# Part B -- everything the briefs name as legitimate is accepted, and gives
# the right number
# ===========================================================================

def test_the_plans_own_example_evaluates_by_hand():
    # `1 + 0.5*sin(2*pi*t/5400)`, the expression written into the plan's own
    # Section 7 sketch: a sinusoid of period 5400 s about a mean of 1.
    expression = TimeExpression('1 + 0.5*sin(2*pi*t/5400)')

    # t = 0: sin(0) = 0.
    assert expression(0 * u.s) == pytest.approx(1.0)
    # t = 1350 s is a quarter period: sin(pi/2) = 1, so 1 + 0.5.
    assert expression(1350 * u.s) == pytest.approx(1.5)
    # t = 2700 s is a half period: sin(pi) = 0, back to the mean.
    assert expression(2700 * u.s) == pytest.approx(1.0)
    # t = 4050 s is three quarters: sin(3pi/2) = -1, so 1 - 0.5.
    assert expression(4050 * u.s) == pytest.approx(0.5)
    # t = 5400 s is a full period, back to the start.
    assert expression(5400 * u.s) == pytest.approx(1.0)


def test_exponential_decay_evaluates_by_hand():
    # exp(-t/3600): 1 at t = 0, and 1/e one hour in, by definition of e.
    expression = TimeExpression('exp(-t/3600)')

    assert expression(0 * u.s) == pytest.approx(1.0)
    assert expression(3600 * u.s) == pytest.approx(1 / np.e)
    assert expression(7200 * u.s) == pytest.approx(1 / np.e**2)


def test_powers_of_t_evaluate_by_hand():
    assert TimeExpression('t**2')(7 * u.s) == pytest.approx(49.0)
    assert TimeExpression('t**0.5')(9 * u.s) == pytest.approx(3.0)
    assert TimeExpression('2**t')(10 * u.s) == pytest.approx(1024.0)


def test_constants_are_the_ones_they_are_named_after():
    assert TimeExpression('pi')(0 * u.s) == pytest.approx(np.pi)
    assert TimeExpression('e')(0 * u.s) == pytest.approx(np.e)


def test_arithmetic_operators_all_work():
    # The operators the rejection message promises: + - * / // % **.
    assert TimeExpression('t + 1')(2 * u.s) == pytest.approx(3.0)
    assert TimeExpression('t - 1')(2 * u.s) == pytest.approx(1.0)
    assert TimeExpression('3*t')(2 * u.s) == pytest.approx(6.0)
    assert TimeExpression('t/4')(2 * u.s) == pytest.approx(0.5)
    assert TimeExpression('t//4')(9 * u.s) == pytest.approx(2.0)
    assert TimeExpression('t % 4')(9 * u.s) == pytest.approx(1.0)
    assert TimeExpression('-t')(2 * u.s) == pytest.approx(-2.0)
    assert TimeExpression('+t')(2 * u.s) == pytest.approx(2.0)


def test_t_is_in_seconds_whatever_unit_it_is_given_in():
    # The class docstring is explicit: `t` is the time in seconds, so the
    # plan's 5400 in the example really is 5400 seconds. Handing the same
    # instant in minutes must give the same answer.
    expression = TimeExpression('t')

    assert expression(90 * u.s) == pytest.approx(90.0)
    assert expression(1.5 * u.min) == pytest.approx(90.0)
    assert expression(1 * u.hour) == pytest.approx(3600.0)


def test_evaluating_at_a_plain_number_is_a_type_error():
    # A bare `1350` could mean seconds, minutes or orbits; refusing it is what
    # keeps `t` unambiguously in seconds.
    expression = TimeExpression('t')
    with pytest.raises(TypeError, match='Quantity'):
        expression(1350)


def test_the_expression_string_is_kept_verbatim_for_writing_back_out():
    # `scaling_to_config` writes this back into a configuration, so it has to
    # be the string the user wrote, spacing and all.
    text = '1 + 0.5*sin(2*pi*t/5400)'
    expression = TimeExpression(text)

    assert expression.expression == text
    assert repr(expression) == f"TimeExpression({text!r})"


def test_a_configured_function_scaling_evaluates_the_expression():
    # End to end: the same hand-computed quarter-period value, but reached
    # through a `scaling` block rather than the class directly.
    scaling = scaling_from_config(
        {'type': 'Function', 'expression': '1 + 0.5*sin(2*pi*t/5400)'})

    assert isinstance(scaling, FunctionScaling)
    assert scaling(0 * u.s) == pytest.approx(1.0)
    assert scaling(1350 * u.s) == pytest.approx(1.5)


# ===========================================================================
# Part C -- the exponent rule, pinned at its boundary
# ===========================================================================
#
# This rule is subtle and was got wrong once during implementation: a first
# version rejected anything whose exponent was not a small literal, which
# throws out `2**t` -- an ordinary exponential. CONTRACT.md settles it as two
# structural rules:
#
#   * reject a `**` whose RIGHT operand is itself a `**` (chained
#     exponentiation, which is what `9**9**9**9` is);
#   * reject an integer LITERAL exponent above 64.
#
# Each test below sits deliberately just inside or just outside one of those
# two lines, so that a future "simplification" of the rule breaks a test
# rather than either re-opening the hang or breaking legitimate configs.

def test_literal_exponent_at_the_limit_is_accepted():
    # 2**64 is the largest literal exponent CONTRACT.md says must be accepted:
    # 18446744073709551616, a number Python computes instantly.
    assert TimeExpression('2**64')(0 * u.s) == pytest.approx(2.0**64)


def test_literal_exponent_one_past_the_limit_is_rejected():
    # One step outside the same line. 2**65 is no more expensive than 2**64 --
    # the point is that the limit is where the documented limit says it is,
    # not somewhere else that happens to work today.
    error = _reject_promptly('2**65')
    assert '64' in str(error)


def test_a_non_literal_exponent_is_accepted_however_large_it_could_get():
    # The case the first implementation got wrong. `2**t` is an ordinary
    # exponential; the literal rule must not reach it, even though `t` at run
    # time can be far larger than 64.
    assert TimeExpression('2**t')(10 * u.s) == pytest.approx(1024.0)
    assert TimeExpression('t**2')(10 * u.s) == pytest.approx(100.0)
    assert TimeExpression('t**0.5')(16 * u.s) == pytest.approx(4.0)


def test_a_float_literal_exponent_is_not_caught_by_the_integer_rule():
    # The literal rule is about *integer* literals, because those are what
    # produce arbitrary-precision integers. A float exponent produces a float,
    # which overflows to inf rather than eating the machine's memory.
    assert TimeExpression('2**10.0')(0 * u.s) == pytest.approx(1024.0)


def test_chained_exponentiation_is_rejected_even_when_it_is_small():
    # `2**t**2` is `2**(t**2)`: nothing here is huge, and at t = 2 it is only
    # 16. It is rejected anyway, because the rule is about the SHAPE of the
    # expression. A rule that tried to decide how big the result would be
    # would have to evaluate it, which is the thing being prevented.
    _reject_promptly('2**t**2')


def test_a_left_nested_power_is_not_chained_exponentiation():
    # Just outside the other line, on the safe side: `**` is right
    # associative, so parenthesising to the left is a different expression
    # entirely. (2**3)**4 is 4096, computed in one step, and is fine.
    assert TimeExpression('(2**3)**4')(0 * u.s) == pytest.approx(4096.0)


def test_the_two_exponent_rules_are_independent():
    # `10**1000` has no chaining, and `9**9**9**9`'s outermost exponent is a
    # `**` rather than an oversized literal. Neither rule catches both, so
    # dropping either one re-opens a hole.
    _reject_promptly('10**1000')
    _reject_promptly('9**9**9**9')
