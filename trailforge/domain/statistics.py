"""Central definitions for statistics windows and numerator/denominator rules.

Every statistics time range is a half-open interval ``[start, end)`` over UTC
business timestamps: ``start`` is inclusive, ``end`` is exclusive. Each metric
is attributed to the window by the business time at which it actually happened
(for example a completed training session belongs to the window containing its
``actual_end_at``, not its planned slot).

Status-based numerator/denominator rules live here so the dashboard, the
module statistics and the detail endpoints cannot drift apart. Enum members
are always used as themselves; grouping by their string representation is not
allowed because it silently accepts unknown or renamed values.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ColumnElement
from sqlalchemy.orm import InstrumentedAttribute

from trailforge.domain.enums import (
    ActivityStatus,
    EmergencyStatus,
    LoanStatus,
    RegistrationStatus,
    SessionStatus,
)


def in_window(
    column: InstrumentedAttribute[datetime],
    start: datetime | None,
    end: datetime | None,
) -> list[ColumnElement[bool]]:
    """Half-open ``[start, end)`` conditions for a UTC business-time column."""
    conditions: list[ColumnElement[bool]] = []
    if start is not None:
        conditions.append(column >= start)
    if end is not None:
        conditions.append(column < end)
    return conditions


def percent(numerator: int | float, denominator: int | float) -> float:
    """Percentage rounded to two decimals; zero denominator yields zero."""
    if not denominator:
        return 0
    return round(numerator / denominator * 100, 2)


# Training -----------------------------------------------------------------
# planned_sessions counts sessions whose planned slot starts in the window;
# cancelled sessions never became real obligations and stay out of the
# completion-rate denominator.
SESSION_PLANNED_EXCLUDED_STATUSES: frozenset[SessionStatus] = frozenset(
    {SessionStatus.CANCELLED}
)

# Activities ---------------------------------------------------------------
# completion_rate = completed / (total - excluded); a cancelled expedition is
# not a failed completion, so it leaves the denominator entirely.
ACTIVITY_COMPLETION_EXCLUDED_STATUSES: frozenset[ActivityStatus] = frozenset(
    {ActivityStatus.CANCELLED}
)
# Participants (numerator of average_participants and confirmed_registrations)
# are confirmed registrations only. Waitlisted, pending, withdrawn and rejected
# registrations never count as participants.
REGISTRATION_PARTICIPANT_STATUSES: frozenset[RegistrationStatus] = frozenset(
    {RegistrationStatus.CONFIRMED}
)

# Gear ---------------------------------------------------------------------
# Outstanding units and the utilization numerator come from loan orders only
# (quantity - returned_quantity, which is partial-return aware). Inventory
# movements journal the very same loan/return events, so summing them again
# would double count; they are never a second source for these metrics.
LOAN_ACTIVE_STATUSES: frozenset[LoanStatus] = frozenset(
    {LoanStatus.ACTIVE, LoanStatus.OVERDUE}
)
# Usage per catalog counts real loan orders; a cancelled loan never happened.
LOAN_USAGE_EXCLUDED_STATUSES: frozenset[LoanStatus] = frozenset({LoanStatus.CANCELLED})

# Safety -------------------------------------------------------------------
# Unclosed incidents are exactly the ones still needing attention.
EMERGENCY_OPEN_STATUSES: frozenset[EmergencyStatus] = frozenset(
    {EmergencyStatus.OPEN, EmergencyStatus.MONITORING}
)
# Closed incidents are resolved or turned out to be false alarms; every
# incident is either open or closed, so total == open + closed.
EMERGENCY_CLOSED_STATUSES: frozenset[EmergencyStatus] = frozenset(
    {EmergencyStatus.RESOLVED, EmergencyStatus.FALSE_ALARM}
)
