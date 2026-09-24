from __future__ import annotations

from datetime import datetime

from sqlalchemy import Select, func, select
from sqlalchemy.orm import selectinload

from trailforge.domain.enums import SessionStatus, TrainingType
from trailforge.domain.statistics import in_window
from trailforge.models.training import (
    TrainingExercise,
    TrainingPlan,
    TrainingRecord,
    TrainingSession,
)
from trailforge.repositories.base import BaseRepository, PageResult
from trailforge.schemas.training import TrainingPlanFilter


class TrainingRepository(BaseRepository[TrainingPlan]):
    model = TrainingPlan
    sortable = {
        "created_at": TrainingPlan.created_at,
        "start_at": TrainingPlan.start_at,
        "end_at": TrainingPlan.end_at,
        "name": TrainingPlan.name,
        "status": TrainingPlan.status,
    }

    def get_plan_detail(self, plan_id: int, *, for_update: bool = False) -> TrainingPlan | None:
        statement = (
            select(TrainingPlan)
            .options(selectinload(TrainingPlan.exercises), selectinload(TrainingPlan.sessions))
            .where(TrainingPlan.id == plan_id)
        )
        if for_update:
            statement = statement.with_for_update()
        return self.session.scalar(statement)

    def list_plans(self, filters: TrainingPlanFilter) -> PageResult[TrainingPlan]:
        statement: Select = select(TrainingPlan).options(selectinload(TrainingPlan.exercises))
        if filters.user_id is not None:
            statement = statement.where(TrainingPlan.user_id == filters.user_id)
        if filters.status is not None:
            statement = statement.where(TrainingPlan.status == filters.status)
        if filters.training_type is not None:
            statement = statement.join(TrainingExercise).where(
                TrainingExercise.training_type == filters.training_type
            )
        if filters.starts_after is not None:
            statement = statement.where(TrainingPlan.start_at >= filters.starts_after)
        if filters.ends_before is not None:
            statement = statement.where(TrainingPlan.end_at < filters.ends_before)
        return self.paginate(
            statement.distinct(),
            page=filters.page,
            page_size=filters.page_size,
            sort=filters.sort,
            direction=filters.direction,
        )

    def get_exercise(self, exercise_id: int) -> TrainingExercise | None:
        return self.session.get(TrainingExercise, exercise_id)

    def get_session(self, session_id: int, *, for_update: bool = False) -> TrainingSession | None:
        statement = (
            select(TrainingSession)
            .options(selectinload(TrainingSession.records))
            .where(TrainingSession.id == session_id)
        )
        if for_update:
            statement = statement.with_for_update()
        return self.session.scalar(statement)

    def list_sessions(
        self,
        *,
        user_id: int,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
    ) -> list[TrainingSession]:
        statement = select(TrainingSession).where(TrainingSession.user_id == user_id)
        statement = statement.where(
            *in_window(TrainingSession.planned_start_at, start_at, end_at)
        )
        return list(self.session.scalars(statement.order_by(TrainingSession.planned_start_at)))

    def session_counts_by_planned_status(
        self,
        user_id: int,
        start_at: datetime | None,
        end_at: datetime | None,
    ) -> dict[SessionStatus, int]:
        """Session counts grouped by status, attributed by the planned slot."""
        statement = (
            select(TrainingSession.status, func.count())
            .where(
                TrainingSession.user_id == user_id,
                *in_window(TrainingSession.planned_start_at, start_at, end_at),
            )
            .group_by(TrainingSession.status)
        )
        return {
            SessionStatus(status): int(count)
            for status, count in self.session.execute(statement)
        }

    def completed_session_count(
        self,
        user_id: int,
        start_at: datetime | None,
        end_at: datetime | None,
    ) -> int:
        """Sessions whose completion happened inside the window."""
        statement = (
            select(func.count())
            .select_from(TrainingSession)
            .where(
                TrainingSession.user_id == user_id,
                TrainingSession.status == SessionStatus.COMPLETED,
                *in_window(TrainingSession.actual_end_at, start_at, end_at),
            )
        )
        return int(self.session.scalar(statement) or 0)

    def record_totals(
        self,
        user_id: int,
        start_at: datetime | None,
        end_at: datetime | None,
    ) -> tuple[int, float, float, float | None]:
        """Duration, distance, load and average RPE of records whose training
        actually completed inside the window (session ``actual_end_at``)."""
        statement = (
            select(
                func.coalesce(func.sum(TrainingRecord.duration_minutes), 0),
                func.coalesce(func.sum(TrainingRecord.distance_km), 0),
                func.coalesce(func.sum(TrainingRecord.training_load), 0),
                func.avg(TrainingRecord.perceived_exertion),
            )
            .select_from(TrainingRecord)
            .join(TrainingSession, TrainingSession.id == TrainingRecord.session_id)
            .where(
                TrainingSession.user_id == user_id,
                *in_window(TrainingSession.actual_end_at, start_at, end_at),
            )
        )
        duration, distance, load, average_rpe = self.session.execute(statement).one()
        return (
            int(duration),
            float(distance),
            float(load),
            float(average_rpe) if average_rpe is not None else None,
        )

    def load_by_type(
        self,
        user_id: int,
        start_at: datetime | None,
        end_at: datetime | None,
    ) -> dict[TrainingType, float]:
        """Training load per exercise type, attributed by completion time."""
        statement = (
            select(
                TrainingExercise.training_type,
                func.coalesce(func.sum(TrainingRecord.training_load), 0),
            )
            .select_from(TrainingRecord)
            .join(TrainingSession, TrainingSession.id == TrainingRecord.session_id)
            .join(TrainingExercise, TrainingExercise.id == TrainingRecord.exercise_id)
            .where(
                TrainingSession.user_id == user_id,
                *in_window(TrainingSession.actual_end_at, start_at, end_at),
            )
            .group_by(TrainingExercise.training_type)
        )
        return {
            TrainingType(training_type): float(load)
            for training_type, load in self.session.execute(statement)
        }
