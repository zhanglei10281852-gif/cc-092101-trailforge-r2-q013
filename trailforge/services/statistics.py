from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from trailforge.database.base import utc_now
from trailforge.domain.enums import (
    ActivityStatus,
    EmergencyStatus,
    EmergencyType,
    GearCondition,
    PlanStatus,
    RegistrationStatus,
    RiskLevel,
)
from trailforge.domain.statistics import (
    ACTIVITY_COMPLETION_EXCLUDED_STATUSES,
    EMERGENCY_CLOSED_STATUSES,
    EMERGENCY_OPEN_STATUSES,
    LOAN_ACTIVE_STATUSES,
    LOAN_USAGE_EXCLUDED_STATUSES,
    REGISTRATION_PARTICIPANT_STATUSES,
    percent,
)
from trailforge.models.activities import Expedition, ExpeditionRegistration
from trailforge.models.gear import GearCatalog, GearInventory, GearLoan
from trailforge.models.routes import TrailRoute
from trailforge.models.safety import EmergencyIncident, ItineraryCheckIn, RiskAssessment
from trailforge.models.training import TrainingPlan
from trailforge.models.users import User
from trailforge.schemas.activities import ActivityStatistics
from trailforge.schemas.audit import DashboardStatistics
from trailforge.schemas.gear import GearStatistics
from trailforge.schemas.safety import RiskStatistics
from trailforge.services.base import ServiceBase


class StatisticsService(ServiceBase):
    """Read-only aggregate queries.

    Every metric in one call is computed by SQL ``GROUP BY``/``SUM``/``COUNT``
    statements on the request's single session, so all numbers come from one
    consistent database snapshot and no rows are loaded into Python. Status
    buckets are keyed by enum members validated against the stored values.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def dashboard(self, *, now: datetime | None = None) -> DashboardStatistics:
        current = now or utc_now()
        active_users = self._count(select(func.count()).where(User.is_active.is_(True)))
        published_routes = self._count(
            select(func.count()).where(TrailRoute.is_published.is_(True))
        )
        active_plans = self._count(
            select(func.count()).where(TrainingPlan.status == PlanStatus.ACTIVE)
        )
        upcoming = self._count(
            select(func.count()).where(
                Expedition.start_at >= current,
                Expedition.status.in_(
                    {ActivityStatus.OPEN, ActivityStatus.ASSEMBLING, ActivityStatus.DEPARTED}
                ),
            )
        )
        completed = self._count(
            select(func.count()).where(Expedition.status == ActivityStatus.COMPLETED)
        )
        distance, gain = self.session.execute(
            select(
                func.coalesce(func.sum(TrailRoute.distance_km), 0),
                func.coalesce(func.sum(TrailRoute.elevation_gain_m), 0),
            )
            .select_from(Expedition)
            .join(TrailRoute, TrailRoute.id == Expedition.route_id)
            .where(Expedition.status == ActivityStatus.COMPLETED)
        ).one()
        overdue = self._count(
            select(func.count()).where(
                ItineraryCheckIn.due_at < current,
                ItineraryCheckIn.checked_in_at.is_(None),
            )
        )
        open_emergencies = self._count(
            select(func.count()).where(EmergencyIncident.status.in_(EMERGENCY_OPEN_STATUSES))
        )
        active_loans = self._count(
            select(func.count()).where(GearLoan.status.in_(LOAN_ACTIVE_STATUSES))
        )
        return DashboardStatistics(
            generated_at=current,
            active_users=active_users,
            published_routes=published_routes,
            active_training_plans=active_plans,
            upcoming_expeditions=upcoming,
            completed_expeditions=completed,
            total_hiking_distance_km=round(float(distance), 2),
            total_elevation_gain_m=int(gain),
            overdue_check_ins=overdue,
            open_emergencies=open_emergencies,
            active_gear_loans=active_loans,
        )

    def activities(self, organizer_id: int | None = None) -> ActivityStatistics:
        filters = []
        if organizer_id is not None:
            filters.append(Expedition.organizer_id == organizer_id)
        by_status = self._grouped_counts(Expedition.status, ActivityStatus, *filters)
        by_risk = self._grouped_counts(Expedition.risk_level, RiskLevel, *filters)
        registration_filters = []
        if organizer_id is not None:
            registration_filters.append(
                ExpeditionRegistration.expedition_id.in_(
                    select(Expedition.id).where(*filters).scalar_subquery()
                )
            )
        registration_counts = self._grouped_counts(
            ExpeditionRegistration.status,
            RegistrationStatus,
            *registration_filters,
        )
        total = sum(by_status.values())
        completed = by_status.get(ActivityStatus.COMPLETED, 0)
        cancelled = sum(
            count
            for status, count in by_status.items()
            if status in ACTIVITY_COMPLETION_EXCLUDED_STATUSES
        )
        completion_denominator = total - cancelled
        confirmed = sum(
            count
            for status, count in registration_counts.items()
            if status in REGISTRATION_PARTICIPANT_STATUSES
        )
        return ActivityStatistics(
            organizer_id=organizer_id,
            total_activities=total,
            completed_activities=completed,
            cancelled_activities=cancelled,
            completion_rate=percent(completed, completion_denominator),
            total_registrations=sum(registration_counts.values()),
            confirmed_registrations=confirmed,
            average_participants=(
                round(confirmed / completion_denominator, 2) if completion_denominator else 0
            ),
            activities_by_status=self._string_keys(by_status),
            activities_by_risk_level=self._string_keys(by_risk),
        )

    def gear(self, *, now: datetime | None = None) -> GearStatistics:
        current = now or utc_now()
        total, available = self.session.execute(
            select(
                func.coalesce(func.sum(GearInventory.quantity_total), 0),
                func.coalesce(func.sum(GearInventory.quantity_available), 0),
            )
        ).one()
        by_condition = {
            condition.value: int(quantity)
            for condition, quantity in self.session.execute(
                select(GearInventory.condition, func.sum(GearInventory.quantity_total))
                .group_by(GearInventory.condition)
            )
            for condition in [GearCondition(condition)]
        }
        active_filter = GearLoan.status.in_(LOAN_ACTIVE_STATUSES)
        active_loans = self._count(select(func.count()).where(active_filter))
        overdue_loans = self._count(
            select(func.count()).where(active_filter, GearLoan.due_at < current)
        )
        loaned_units = int(
            self.session.scalar(
                select(func.coalesce(func.sum(GearLoan.quantity - GearLoan.returned_quantity), 0))
                .where(active_filter)
            )
            or 0
        )
        loan_counts = {
            str(name): int(quantity)
            for name, quantity in self.session.execute(
                select(GearCatalog.name, func.coalesce(func.sum(GearLoan.quantity), 0))
                .select_from(GearCatalog)
                .join(GearInventory, GearInventory.catalog_id == GearCatalog.id)
                .join(GearLoan, GearLoan.inventory_id == GearInventory.id)
                .where(GearLoan.status.notin_(LOAN_USAGE_EXCLUDED_STATUSES))
                .group_by(GearCatalog.id, GearCatalog.name)
            )
        }
        total_units = int(total)
        return GearStatistics(
            total_catalog_items=self._count(select(func.count()).select_from(GearCatalog)),
            total_inventory_units=total_units,
            available_inventory_units=int(available),
            active_loans=active_loans,
            overdue_loans=overdue_loans,
            loaned_units=loaned_units,
            utilization_rate=percent(loaned_units, total_units),
            items_by_condition=by_condition,
            loans_by_catalog=loan_counts,
        )

    def risks(self, *, now: datetime | None = None) -> RiskStatistics:
        current = now or utc_now()
        by_status = self._grouped_counts(EmergencyIncident.status, EmergencyStatus)
        by_type = self._grouped_counts(EmergencyIncident.incident_type, EmergencyType)
        by_level = self._grouped_counts(EmergencyIncident.risk_level, RiskLevel)
        open_count = sum(
            count for status, count in by_status.items() if status in EMERGENCY_OPEN_STATUSES
        )
        closed_count = sum(
            count for status, count in by_status.items() if status in EMERGENCY_CLOSED_STATUSES
        )
        overdue = self._count(
            select(func.count()).where(
                ItineraryCheckIn.due_at < current,
                ItineraryCheckIn.checked_in_at.is_(None),
            )
        )
        unsafe = self._count(
            select(func.count()).where(ItineraryCheckIn.is_safe.is_(False))
        )
        average_score = self.session.scalar(select(func.avg(RiskAssessment.score)))
        return RiskStatistics(
            total_incidents=sum(by_status.values()),
            open_incidents=open_count,
            resolved_incidents=closed_count,
            incidents_by_type=self._string_keys(by_type),
            incidents_by_level=self._string_keys(by_level),
            overdue_check_ins=overdue,
            unsafe_check_ins=unsafe,
            average_assessment_score=(
                round(float(average_score), 2) if average_score is not None else 0
            ),
        )

    def _count(self, statement: object) -> int:
        return int(self.session.scalar(statement) or 0)

    def _grouped_counts(
        self,
        column: object,
        enum_type: type[StrEnum],
        *filters: object,
    ) -> dict[StrEnum, int]:
        """SQL ``GROUP BY`` counts keyed by validated enum members.

        Keys are never the raw string representation: every stored value must
        round-trip through the enum, so unknown or renamed statuses surface as
        errors instead of silently forming their own bucket.
        """
        statement = select(column, func.count()).group_by(column)
        if filters:
            statement = statement.where(*filters)
        return {
            enum_type(value): int(count) for value, count in self.session.execute(statement)
        }

    @staticmethod
    def _string_keys(counts: dict[StrEnum, int]) -> dict[str, int]:
        return {status.value: count for status, count in counts.items()}
