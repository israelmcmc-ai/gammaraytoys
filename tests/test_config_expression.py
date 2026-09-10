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

The whitelist is only half of the defence, and the two halves refuse at
different moments. The whitelist refuses when the expression is *built*: a
lambda, an attribute, a name nobody put in the namespace. The arithmetic
denial of service is refused when it is *evaluated*, because `**` is rewritten
to a float-only power and a float cannot grow: `9**9**9**9` compiles like any
other arithmetic and overflows the moment it is asked for a number. So this
file guards both moments -- see `_reject_promptly` for the first and
`_reject_promptly_when_evaluated` (and Part D's child process) for the second.

Every expected value below comes from the schema, from those two briefs, or
from arithmetic done in the comment next to it -- never from running the
implementation and copying its output.

House style, matching `tests/test_scaling.py` and `tests/test_event_csv.py`:
statistical assertions state their sigma and assert at 4 sigma. There are no
statistical assertions here; the wall-clock budget below is the one number
that needed choosing, and its reasoning is spelled out where it is defined.
"""

import json
import subprocess
import sys
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
#   * The regressions it has to catch are nowhere near that fast. A rejection
#     that does real work is the failure being watched for, and the cheapest
#     such failure on record is `(((2**64)**64)**64)**64` evaluated in exact
#     integers, which builds a 16,777,217-bit number -- seconds of CPU and
#     hundreds of megabytes, not a quarter of a second. `9**9**9**9` in exact
#     integers is worse again: 9**(9**(9**9)) has some 370 million digits and
#     ran past 6 s without finishing (CONTRACT.md). Nothing that starts down
#     either road comes back inside this bound.
#
#     This budget is applied to *evaluation* as well as construction (Part D):
#     since `**` became float-only, that is where those two are refused.
#
# In short: three orders of magnitude of headroom above the real cost, and
# more than an order of magnitude below the failure it is watching for.
REJECTION_BUDGET_SECONDS = 0.25

# The hard stop, used only to keep a regression from HANGING the suite.
#
# A wall-clock assertion alone cannot save us here: it is only reached *after*
# the call returns, so if a future edit made the evaluator start actually
# computing an expression it should have refused, the assertion would never
# run and CI would sit there until something else killed it. So each attempt
# is made on a worker thread and joined with a timeout; if the worker is still
# going after this long, the test FAILS and says why instead of waiting
# forever.
#
# The thread is a daemon: a wedged worker cannot be killed from outside in
# Python, but a daemon thread does not hold the interpreter open at exit, so
# the run still finishes and still reports the failure. This is all in-process
# and portable -- no subprocess per expression, nothing that only works on one
# platform, and no `pytest-timeout` (which this project does not depend on).
HANG_GUARD_SECONDS = 5.0


def _refuse_promptly(attempt, expression, what):
    """
    Assert that `attempt()` is refused with a `ValueError`, promptly, without
    wedging the suite if it is not refused at all.

    The attempt runs on a daemon worker thread that is joined with a timeout,
    so a regression that starts *computing* instead of refusing is reported as
    a failure rather than left to run until CI gives up.

    One limit is worth knowing, and Part D works around it: a worker thread can
    only be abandoned if the runaway computation lets go of the GIL, and a
    single arbitrary-precision `**` never does -- CPython evaluates it in one
    uninterruptible step, during which this thread's `join` cannot even wake
    up. So this guard covers everything whose cost is spread over many small
    steps, and Part D's arithmetic bombs, whose cost is one enormous step, are
    run in a child process that can be killed outright.

    Parameters
    ----------
    attempt : callable
        Takes no arguments; must raise `ValueError`.
    expression : str
        The expression being refused, for the failure messages.
    what : str
        What is being done to it -- `'built'` or `'evaluated'` -- again for
        the failure messages.

    Returns
    -------
    ValueError
        The rejection, so a caller can assert on its message.
    """

    outcome = {}

    def run():
        started = time.perf_counter()
        try:
            attempt()
            outcome['verdict'] = 'accepted'
        except ValueError as err:
            outcome['verdict'] = 'rejected'
            outcome['error'] = err
        except BaseException as err:                    # noqa: BLE001
            outcome['verdict'] = 'other'
            outcome['error'] = err
        finally:
            outcome['seconds'] = time.perf_counter() - started

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(HANG_GUARD_SECONDS)

    if worker.is_alive():
        pytest.fail(
            f"{expression!r} was still being {what} after "
            f"{HANG_GUARD_SECONDS} s. It is being computed rather than "
            f"refused -- that is the denial of service the whitelist and the "
            f"float-only `**` exist to stop.")

    if outcome['verdict'] == 'accepted':
        pytest.fail(
            f"{expression!r} was {what} without complaint; it must be "
            f"refused.")

    if outcome['verdict'] == 'other':
        raise AssertionError(
            f"{expression!r} was refused with "
            f"{type(outcome['error']).__name__}, not the documented "
            f"ValueError.") from outcome['error']

    assert outcome['seconds'] < REJECTION_BUDGET_SECONDS, (
        f"{expression!r} took {outcome['seconds']:.3f} s to refuse, over the "
        f"{REJECTION_BUDGET_SECONDS} s budget. A slow rejection is itself a "
        f"denial of service.")

    return outcome['error']


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

    return _refuse_promptly(lambda: TimeExpression(expression),
                            expression, 'built')


def _reject_promptly_when_evaluated(expression, at_time = 0 * u.s):
    """
    Assert that `TimeExpression(expression)(at_time)` is refused, promptly,
    and return the `ValueError` it was refused with.

    The sibling of `_reject_promptly`, guarding the other half of the class.
    Since `**` became float-only, an arithmetic bomb is *built* without
    complaint and refused only when it is called, so a plain `pytest.raises`
    here would wedge the suite rather than fail it if that rewrite were ever
    taken away again.

    Parameters
    ----------
    expression : str
        The expression that must be refused when it is evaluated.
    at_time : `astropy.units.Quantity`
        The time to evaluate it at.

    Returns
    -------
    ValueError
        The rejection, so a caller can assert on its message.
    """

    return _refuse_promptly(lambda: TimeExpression(expression)(at_time),
                            expression, 'evaluated')


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
    # Arithmetic denial of service used to be listed here, as `9**9**9**9`
    # and `10**1000`. It is no longer refused at *construction*: `**` is
    # rewritten to a float-only power (`config._pow`), so those expressions
    # are built without complaint and overflow in microseconds when they are
    # evaluated. They belong in a test of evaluation, not of the whitelist.
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
# Part C -- `**` is float-only
# ===========================================================================
#
# There used to be two structural rules here -- "an exponent may not itself be
# a `**`" and "an integer literal exponent may not exceed 64" -- and a test
# sitting just inside and just outside each of them. Both rules are gone, and
# so are those tests, because the rules did not work: all the growth in
# `(2**64)**64` is on the LEFT, so neither rule ever looked at it, and every
# tighter shape rule was beaten in turn by putting a `*1` in the way.
#
# `**` is now rewritten to a float-only power (`config._pow`) before the
# expression is compiled, so there is no boundary left to pin: an integer
# exponent cannot produce an arbitrary-precision integer at all. What is left
# to check here is that ordinary powers still give ordinary answers.
#
# The denial-of-service expressions (`9**9**9**9`, `10**1000`, and left-nested
# towers such as `(((2**64)**64)**64)**64`) are now refused when the expression
# is EVALUATED rather than when it is built; Part D is where they are pinned.

def test_a_big_literal_exponent_is_accepted():
    # 2**64 used to be the largest exponent the old literal rule allowed. It
    # is still accepted -- as 1.8446744073709552e19, a float, which is what a
    # unitless scaling multiplier is anyway.
    assert TimeExpression('2**64')(0 * u.s) == pytest.approx(2.0**64)


def test_a_non_literal_exponent_is_accepted_however_large_it_could_get():
    # `2**t` is an ordinary exponential, and `t` at run time can be far larger
    # than any literal anyone would write.
    assert TimeExpression('2**t')(10 * u.s) == pytest.approx(1024.0)
    assert TimeExpression('t**2')(10 * u.s) == pytest.approx(100.0)
    assert TimeExpression('t**0.5')(16 * u.s) == pytest.approx(4.0)


def test_a_float_literal_exponent_is_accepted():
    assert TimeExpression('2**10.0')(0 * u.s) == pytest.approx(1024.0)



# ===========================================================================
# Part D -- the arithmetic denial of service, refused at EVALUATION
# ===========================================================================
#
# `**` is float-only now (`config._pow`), so these four expressions are BUILT
# without complaint -- there is nothing in the shape of the tree left to
# object to -- and refused the instant they are asked for a number, because a
# float overflows where an arbitrary-precision integer would just keep
# growing. That is the whole defence, and none of it is pinned anywhere else.
#
# The last two are the shapes that beat the rules this replaced: all of the
# growth is on the LEFT and every exponent is a harmless literal 64, so a rule
# about chained `**`, or about the size of a literal exponent, never even
# looks at them. In exact integers `(((2**64)**64)**64)**64` is a 16,777,217-bit
# number, and each further nesting SQUARES that -- one deeper was measured at
# 5.7 s and 703 MB, two deeper at 61 s and 3.7 GB before a MemoryError
# (TEST_BRIEF_2.md).
EVALUATION_BOMBS = {
    '10**1000': "an exponent far too large for a float, written as a literal",
    '(((2**64)**64)**64)**64': "a left-nested tower: every exponent is 64",
    '((((2**64*1)**64*1)**64*1)**64*1)**64':
        "the same tower with a harmless '*1' in the way of any shape rule",
    '9**9**9**9': "the chained tower CONTRACT.md killed at 6 s",
}

# Why these four run in a CHILD PROCESS rather than on `_refuse_promptly`'s
# worker thread.
#
# The thread guard can only report a runaway if the runaway lets go of the
# GIL. `9**9**9**9` in exact integers is a *single* arbitrary-precision power
# -- one bytecode, one uninterruptible C call -- so the main thread does not
# get to run again until it finishes, which for that expression is never. The
# join would not return, and the suite would hang exactly as it did before
# any of this defence existed. A child process has no such problem: it can be
# killed from outside, and the operating system takes its memory back with it.
#
# So the child works through the four expressions, cheapest first, appending
# one line of JSON per verdict and flushing as it goes. If it is killed
# part-way, the verdicts it managed to write are still on disk and the
# expression it died on is simply missing -- which is itself the answer, and
# is reported as a failure naming that expression.

#: How long the whole child process gets: interpreter start plus astropy's
#: import (a couple of seconds) plus four evaluations that, working correctly,
#: take microseconds. Generous, because it is only ever reached by a
#: regression, and a regression is allowed to cost one wait of this length
#: rather than a hung run.
EVALUATION_BATTERY_SECONDS = 30.0

#: Address space the child may use once everything is imported. Without this,
#: a regression would spend its 30 s filling memory with an exact integer and
#: could take the machine down with it. With it, the runaway hits a
#: `MemoryError` instead -- which is *not* accepted as a pass, because the
#: per-expression wall-clock assertion below still has to hold, and no
#: computation that reaches this cap does so in a quarter of a second.
EVALUATION_MEMORY_LIMIT_BYTES = 2 * 1024**3

_EVALUATION_PROBE = f'''
import json
import sys
import time

import astropy.units as u

from gammaraytoys.sims import TimeExpression

report_path = sys.argv[1]
expressions = sys.argv[2:]

# Warm up first: this pulls in whatever astropy imports lazily, so the memory
# cap below bounds the arithmetic under test and nothing else.
TimeExpression('1 + t')(0 * u.s)

try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard == resource.RLIM_INFINITY or hard > {EVALUATION_MEMORY_LIMIT_BYTES}:
        resource.setrlimit(resource.RLIMIT_AS,
                           ({EVALUATION_MEMORY_LIMIT_BYTES}, hard))
except Exception:
    # No `resource` module (Windows): the timeout alone still bounds the run.
    pass

with open(report_path, 'a') as report:
    for expression in expressions:
        started = time.perf_counter()
        try:
            value = TimeExpression(expression)(0 * u.s)
        except ValueError as err:
            outcome = {{'verdict': 'rejected', 'error': str(err)}}
        except BaseException as err:
            outcome = {{'verdict': 'other',
                        'error': type(err).__name__ + ': ' + str(err)[:200]}}
        else:
            # Deliberately not `repr(value)`: an accepted bomb returns an
            # integer with millions of digits, and asking for its digits
            # raises a ValueError of its own that would read like a refusal.
            outcome = {{'verdict': 'accepted', 'value': type(value).__name__}}

        outcome['expression'] = expression
        outcome['seconds'] = time.perf_counter() - started
        report.write(json.dumps(outcome) + chr(10))
        report.flush()
'''


@pytest.fixture(scope = 'module')
def evaluation_verdicts(tmp_path_factory):
    """
    Evaluate every expression in `EVALUATION_BOMBS` in a child process and
    report what happened to each.

    Returns
    -------
    dict
        `expression -> {'verdict': ..., 'seconds': ..., ...}`. An expression
        the child never got to -- because it was killed while computing an
        earlier one -- comes back with the verdict `'never reported'`.
    """

    report_path = tmp_path_factory.mktemp('evaluation') / 'verdicts.jsonl'
    report_path.write_text('')

    started = time.perf_counter()
    summary = ''

    try:
        finished = subprocess.run(
            [sys.executable, '-c', _EVALUATION_PROBE, str(report_path),
             *EVALUATION_BOMBS],
            capture_output = True, text = True, check = False,
            timeout = EVALUATION_BATTERY_SECONDS)
        summary = (f"the child exited with code {finished.returncode} after "
                   f"{time.perf_counter() - started:.1f} s; "
                   f"stderr: {finished.stderr.strip()[-500:]!r}")
    except subprocess.TimeoutExpired:
        summary = (f"the child was still running after "
                   f"{EVALUATION_BATTERY_SECONDS} s and was killed")

    verdicts = {}

    for line in report_path.read_text().splitlines():
        outcome = json.loads(line)
        verdicts[outcome['expression']] = outcome

    for expression in EVALUATION_BOMBS:
        verdicts.setdefault(expression,
                            {'verdict': 'never reported', 'seconds': None,
                             'error': summary})

    return verdicts


@pytest.mark.parametrize('expression', list(EVALUATION_BOMBS))
def test_an_arithmetic_bomb_is_built_but_refused_when_it_is_evaluated(
        expression, evaluation_verdicts):
    # Building it is fine, and fast: `**` is float-only, so there is nothing
    # about the shape of this tree for the whitelist to object to. Every node
    # in it is one an ordinary expression needs.
    TimeExpression(expression)

    outcome = evaluation_verdicts[expression]

    if outcome['verdict'] != 'rejected':
        pytest.fail(
            f"{expression!r} ({EVALUATION_BOMBS[expression]}) was "
            f"{outcome['verdict']} when it was evaluated, not refused: "
            f"{outcome.get('error') or outcome.get('value')}. Every '**' is "
            f"supposed to be rewritten to a float-only power, which cannot "
            f"grow past 64 bits however hard it is pushed.")

    # The message has to name the expression: a configuration file may hold a
    # dozen scalings and only one of them is the one to go and fix.
    assert expression in outcome['error']

    # And it is refused because the float *overflowed*, which is the whole
    # mechanism: a float is 64 bits wide however large the answer wants to be.
    assert 'OverflowError' in outcome['error']

    assert outcome['seconds'] < REJECTION_BUDGET_SECONDS, (
        f"{expression!r} took {outcome['seconds']:.3f} s to refuse, over the "
        f"{REJECTION_BUDGET_SECONDS} s budget. A slow rejection is itself a "
        f"denial of service.")


def test_an_arithmetic_bomb_in_a_scaling_block_is_refused_the_same_way():
    # The configuration layer must not have a softer path to the same place:
    # a `scaling` block builds without complaint, like the class, and refuses
    # at the first interval of the run rather than producing a number.
    #
    # `10**1000` and not one of the towers: this one is guarded by a worker
    # thread rather than a child process (see `_refuse_promptly`), and it is
    # the one whose runaway cost -- an exact 10**1000, all of 3322 bits -- is
    # small enough that a thread can always report on it.
    scaling = scaling_from_config({'type': 'Function', 'expression': '10**1000'})

    outcome = _refuse_promptly(lambda: scaling(0 * u.s), '10**1000', 'evaluated')
    assert '10**1000' in str(outcome)


def test_an_expression_may_not_call_the_rewriters_own_power_function():
    # `_pow` is deliberately not in the expression namespace: it is put there
    # only for the rewritten tree, so a file that names it by hand -- to get
    # at some future, laxer version of it -- is refused like any other name
    # that is not on the whitelist.
    error = _reject_promptly('_pow(9, 9)')
    assert 'not a callable function' in str(error)


# ===========================================================================
# Part E -- a float base raised to a large power is ORDINARY PHYSICS
# ===========================================================================
#
# The rules this replaced refused every one of these, because they refused an
# integer literal exponent above 64 whatever the base was. A per-step survival
# fraction raised to a step count is exactly that shape, and there is nothing
# dangerous about it: the answer is a number between 0 and 1.

def test_a_survival_fraction_raised_to_a_step_count_is_accepted():
    # 0.999**1000: a 0.1% loss per step, a thousand steps. The limit of
    # (1 - 1/n)**n is 1/e = 0.3678794..., and a thousand steps is close to
    # but not at that limit; the exact value, worked out by hand in
    # TEST_BRIEF_2.md, is 0.36769542477096373.
    assert TimeExpression('0.999**1000')(0 * u.s) == pytest.approx(
        0.36769542477096373)

    # (1 - 1e-4)**500, the same shape written as a loss rate. exp(-0.05) =
    # 0.951229... is the limit; the exact value is 0.9512270462715808.
    assert TimeExpression('(1 - 1e-4)**500')(0 * u.s) == pytest.approx(
        0.9512270462715808)


def test_a_float_base_grown_a_long_way_is_still_accepted():
    # 1.05**200 = exp(200*ln(1.05)) = exp(9.7580...), some 17300 -- large, but
    # nowhere near a float's ceiling, so there is nothing to refuse.
    assert TimeExpression('1.05**200')(0 * u.s) == pytest.approx(1.05 ** 200)


def test_a_float_base_that_does_overflow_is_refused_at_evaluation():
    # `10.0**400` is on the other side of the line: 1e400 is past the largest
    # float (about 1.8e308), so it is built like any other expression and
    # refused when it is called. This is where a float base stops being
    # accepted -- not at some rule about how the exponent was written.
    error = _reject_promptly_when_evaluated('10.0**400')
    assert '10.0**400' in str(error)
    assert 'OverflowError' in str(error)


# ===========================================================================
# Part F -- `**` returns a float, and the old limit at 64 is gone
# ===========================================================================

def test_a_power_evaluates_to_a_float_and_not_an_exact_integer():
    # 2**64 is 18446744073709551616 exactly as an integer, and
    # 1.8446744073709552e19 as a float. The two are equal here only because
    # 2**64 is a power of two and lands exactly on a float; what this pins is
    # the TYPE, which is where the defence lives. A unitless scaling
    # multiplier is a float in the end anyway.
    value = TimeExpression('2**64')(0 * u.s)
    assert isinstance(value, float)
    assert value == pytest.approx(2.0 ** 64)

    # Nested powers are floats too, all the way down.
    assert TimeExpression('(2**3)**4')(0 * u.s) == pytest.approx(4096.0)
    assert isinstance(TimeExpression('(2**3)**4')(0 * u.s), float)


def test_an_exponent_above_the_old_limit_of_64_is_now_accepted():
    # The rule that refused an integer literal exponent above 64 is gone, and
    # with it the strange cliff between `2**64` and `2**65`. Neither is
    # dangerous now, because neither is an integer.
    value = TimeExpression('2**65')(0 * u.s)
    assert isinstance(value, float)
    assert value == pytest.approx(2.0 ** 65)


def test_the_ordinary_powers_are_unchanged_in_value():
    # `2**t`, `t**2` and `t**0.5` mean what they always meant; the rewrite is
    # invisible to anything that was not trying to build a huge integer.
    assert TimeExpression('2**t')(10 * u.s) == pytest.approx(1024.0)
    assert TimeExpression('t**2')(7 * u.s) == pytest.approx(49.0)
    assert TimeExpression('t**0.5')(9 * u.s) == pytest.approx(3.0)


# ===========================================================================
# Part G -- an expression may be at most 250 characters long
# ===========================================================================
#
# Two costs are bounded by this one rule, and both are paid a character at a
# time. Integers are still arbitrary precision under `*` -- only `**` was
# taken away -- so a long product of long literals still builds a huge exact
# integer. And the whitelist walk and the `**` rewrite each recurse once per
# node, so a long chain of anything runs the interpreter out of stack and
# raises `RecursionError`, which is not the `ValueError` this class documents
# and not a message that says anything about the file.
#
# Why 250 and not something rounder: the recursion is the tighter of the two
# constraints. Measured through the full path -- the whitelist walk and the
# rewrite, one nested tree walk each -- the boundary is 497 characters:
# `'-'*496 + '1'` raises `RecursionError` (TEST_BRIEF_2.md). A cap at or above
# ~500 would therefore do nothing at all about the recursion, so the cap has
# to sit well below it. 250 is still generous for what it bounds: a scaling is
# a one-liner, and the plan's own example is 24 characters.

def test_an_over_length_expression_is_refused_naming_the_limit_and_the_length():
    expression = 'sin(t)*' * 40 + '1'
    assert len(expression) == 281              # comfortably over the cap

    error = _reject_promptly(expression)

    assert '250' in str(error)
    assert '281' in str(error)

    # And it does NOT echo the expression back. This one is only 281
    # characters; the ones below are hundreds of kilobytes, and an error
    # message that repeats them is worse than useless.
    assert expression not in str(error)


def test_the_cap_is_at_250_characters_exactly():
    # A 250-digit number is a silly expression, but it is one character of
    # expression per character of file, which makes it the clearest possible
    # marker of where the boundary is.
    TimeExpression('1' * 250)

    error = _reject_promptly('1' * 251)
    assert '251' in str(error)


def test_a_realistic_expression_is_nowhere_near_the_cap():
    # The plan's own example, and then a nine-term Fourier-like sum of the
    # kind a real scaling might use: still only four fifths of the cap.
    assert len('1 + 0.5*sin(2*pi*t/5400)') == 24

    nine_terms = ('1 + 0.5*sin(2*pi*t/5400) + 0.25*cos(2*pi*t/5400)'
                  ' + 0.125*sin(4*pi*t/5400) + 0.0625*cos(4*pi*t/5400)'
                  ' + 0.03125*sin(6*pi*t/5400) + 0.015625*cos(6*pi*t/5400)'
                  ' + 0.00781*sin(8*pi*t/5400) + 0.00391*cos(8*pi*t/5400)')
    assert len(nine_terms) == 208

    # At t = 0 every sine term is 0 and every cosine term is its coefficient,
    # so the sum is 1 + 0.25 + 0.0625 + 0.015625 + 0.00391 = 1.332035.
    assert TimeExpression(nine_terms)(0 * u.s) == pytest.approx(1.332035)


# The three shapes that used to raise `RecursionError` instead of `ValueError`,
# each with what it costs when it is not refused first.
RECURSION_BOMBS = {
    # 400 factors of 4000 digits each: 1.6 MB of YAML, and an exact integer
    # with millions of bits, built one quadratic multiplication at a time.
    '*'.join(['9' * 4000] * 400): 'a long product of long integer literals',
    # 5001 nodes deep in the operator tree, one frame per node per walk.
    '2' + '*2' * 5000: 'a five-thousand-deep chain of multiplications',
    # The same, in unary minus: the shape that measured the 497-character
    # recursion boundary.
    '-' * 2000 + '1': 'two thousand unary minuses',
}


@pytest.mark.parametrize('expression, description',
                         [(expression, description)
                          for expression, description in RECURSION_BOMBS.items()],
                         ids = list(RECURSION_BOMBS.values()))
def test_a_recursion_bomb_is_a_value_error_and_not_a_recursion_error(
        expression, description):
    # `_reject_promptly` fails the test outright on anything that is not a
    # `ValueError`, `RecursionError` included: a stack overflow from inside
    # the parser says nothing about which file, which key or which line, and
    # the class documents `ValueError`.
    error = _reject_promptly(expression)

    assert '250' in str(error)
    assert str(len(expression)) in str(error)
    assert expression not in str(error)


# ===========================================================================
# Part H -- a mistyped function name is suggested by how it was mistyped
# ===========================================================================
#
# Plain edit distance is wrong here often enough to be worse than silence: a
# short whitelisted name is close to almost everything, so `min(1, t)` and
# `cosine(t)` both came back with "Did you mean 'sin'?", and a wrong
# suggestion sends a student looking in the wrong place. A mistyped function
# name is nearly always a real name cut short or run long, so a name that
# extends what was typed (or that what was typed extends) is preferred, and
# only when nothing is related that way does the edit-distance suggestion get
# a say.

def test_a_name_cut_short_suggests_the_name_it_is_short_for():
    # numpy spells the two-argument minimum `minimum`; `min` is the Python
    # builtin, which is not on the whitelist. 'minimum' starts with 'min'.
    error = _reject_promptly('min(1, t)')
    assert "Did you mean 'minimum'?" in str(error)


def test_a_name_run_long_suggests_the_name_it_was_grown_from():
    # 'cosine' starts with 'cos'. It is not closer to 'cos' than to 'sin' by
    # edit distance -- that is exactly the case this rule exists for.
    error = _reject_promptly('cosine(t)')
    assert "Did you mean 'cos'?" in str(error)


def test_a_name_related_to_nothing_suggests_nothing():
    # Silence is the right answer when there is no good guess to make.
    error = _reject_promptly('frobnicate(t)')
    assert 'frobnicate' in str(error)
    assert 'Did you mean' not in str(error)


def test_the_names_the_suggestions_point_at_really_do_work():
    # A suggestion that names something that does not evaluate would be a
    # second wrong turn, so pin both of them from the other side.
    assert TimeExpression('minimum(1, t)')(5 * u.s) == pytest.approx(1.0)
    assert TimeExpression('maximum(1, t)')(5 * u.s) == pytest.approx(5.0)
    assert TimeExpression('cos(0*t)')(5 * u.s) == pytest.approx(1.0)
    assert TimeExpression('log2(t)')(8 * u.s) == pytest.approx(3.0)
