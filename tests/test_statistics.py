from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

from sqlalchemy import event, func, select

from tests.conftest import create_expedition, create_route, create_user
from trailforge.database.migrations import initialize_database
from trailforge.database.session import Database
from trailforge.domain.enums import LoanStatus, PlanStatus, SessionStatus
from trailforge.models.audit import AuditLog
from trailforge.models.gear import GearLoan, InventoryMovement
from trailforge.models.safety import ItineraryCheckIn
from trailforge.models.training import TrainingSession
from trailforge.models.users import User
from trailforge.schemas.activities import ActivityStateChange, RegistrationCreate
from trailforge.schemas.gear import (
    GearCatalogCreate,
    GearInventoryCreate,
    GearLoanCreate,
    GearLoanReturn,
)
from trailforge.schemas.safety import (
    EmergencyIncidentCreate,
    EmergencyIncidentUpdate,
    RiskAssessmentCreate,
)
from trailforge.schemas.training import (
    SessionCompleteRequest,
    TrainingExerciseCreate,
    TrainingPlanCreate,
    TrainingRecordCreate,
    TrainingSessionCreate,
)
from trailforge.services.activities import ExpeditionService
from trailforge.services.gear import GearService
from trailforge.services.safety import SafetyService
from trailforge.services.statistics import StatisticsService
from trailforge.services.training import TrainingService

UTC = UTC


def test_completed_expedition_contributes_distance_and_elevation(session) -> None:
    organizer = create_user(session)
    route = create_route(session, actor_id=organizer)
    expedition = create_expedition(session, organizer_id=organizer, route_id=route)
    service = ExpeditionService(session)
    for target in ("open", "assembling", "departed", "in_progress", "completed"):
        service.change_status(
            expedition,
            ActivityStateChange(target_status=target, actor_id=organizer),
        )
    dashboard = StatisticsService(session).dashboard()
    assert dashboard.completed_expeditions == 1
    assert dashboard.total_hiking_distance_km == 10
    assert dashboard.total_elevation_gain_m == 500
    activity_stats = StatisticsService(session).activities(organizer_id=organizer)
    assert activity_stats.completion_rate == 100
    assert activity_stats.confirmed_registrations == 1


def test_gear_utilization_statistics(session) -> None:
    owner = create_user(session)
    borrower = create_user(session, email="stats-borrower@example.com", name="Borrower")
    service = GearService(session)
    catalog = service.create_catalog(
        GearCatalogCreate(sku="POLE", name="Trekking Pole", category="walking"),
        actor_id=owner,
    )
    inventory = service.create_inventory(
        GearInventoryCreate(
            catalog_id=catalog.id,
            ownership="club",
            quantity_total=4,
            actor_id=owner,
            idempotency_key="stats-inventory",
        )
    )
    now = datetime.now(UTC)
    service.loan(
        GearLoanCreate(
            inventory_id=inventory.id,
            borrower_id=borrower,
            quantity=1,
            loaned_at=now,
            due_at=now + timedelta(days=1),
            actor_id=owner,
            idempotency_key="stats-loan",
        )
    )
    stats = StatisticsService(session).gear(now=now)
    assert stats.total_catalog_items == 1
    assert stats.total_inventory_units == 4
    assert stats.available_inventory_units == 3
    assert stats.active_loans == 1
    assert stats.utilization_rate == 25
    assert stats.loans_by_catalog["Trekking Pole"] == 1


def test_risk_statistics_average_score(session) -> None:
    organizer = create_user(session)
    route = create_route(session, actor_id=organizer)
    expedition = create_expedition(session, organizer_id=organizer, route_id=route)
    safety = SafetyService(session)
    for likelihood, impact in ((2, 2), (4, 5)):
        safety.assess_risk(
            RiskAssessmentCreate(
                expedition_id=expedition,
                assessor_id=organizer,
                category="terrain",
                hazard=f"Hazard {likelihood}",
                likelihood=likelihood,
                impact=impact,
                mitigation="Documented mitigation",
            )
        )
    stats = StatisticsService(session).risks()
    assert stats.average_assessment_score == 12
    assert stats.total_incidents == 0


def _create_active_plan(session, user_id: int, start: datetime, end: datetime):
    service = TrainingService(session)
    plan = service.create_plan(
        TrainingPlanCreate(
            user_id=user_id,
            name="Window Plan",
            goal="Statistics attribution",
            start_at=start,
            end_at=end,
            target_sessions_per_week=7,
            exercises=[
                TrainingExerciseCreate(
                    sequence=1,
                    name="Endurance run",
                    training_type="endurance",
                    target_duration_minutes=60,
                    target_distance_km=10,
                )
            ],
        ),
        actor_id=user_id,
    )
    service.change_plan_status(plan.id, PlanStatus.ACTIVE, actor_id=user_id)
    return service, plan


def _complete_session(
    service: TrainingService,
    plan,
    user_id: int,
    *,
    planned_start: datetime,
    planned_end: datetime,
    actual_start: datetime,
    completed_at: datetime,
    duration_minutes: int,
    distance_km: float,
) -> None:
    scheduled = service.schedule_session(
        TrainingSessionCreate(
            plan_id=plan.id,
            title="Window session",
            planned_start_at=planned_start,
            planned_end_at=planned_end,
        ),
        actor_id=user_id,
    )
    service.start_session(scheduled.id, actor_id=user_id, started_at=actual_start)
    service.complete_session(
        scheduled.id,
        SessionCompleteRequest(
            completed_at=completed_at,
            records=[
                TrainingRecordCreate(
                    exercise_id=plan.exercises[0].id,
                    duration_minutes=duration_minutes,
                    distance_km=distance_km,
                    perceived_exertion=5,
                    completion_percent=100,
                )
            ],
        ),
        actor_id=user_id,
    )


def test_training_load_belongs_to_completion_window_across_midnight(session) -> None:
    user_id = create_user(session)
    service, plan = _create_active_plan(
        session, user_id, datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)
    )
    _complete_session(
        service,
        plan,
        user_id,
        planned_start=datetime(2026, 2, 14, 23, 0, tzinfo=UTC),
        planned_end=datetime(2026, 2, 15, 1, 0, tzinfo=UTC),
        actual_start=datetime(2026, 2, 14, 23, 30, tzinfo=UTC),
        completed_at=datetime(2026, 2, 15, 0, 30, tzinfo=UTC),
        duration_minutes=30,
        distance_km=3,
    )
    day_one = service.statistics(
        user_id,
        start_at=datetime(2026, 2, 14, tzinfo=UTC),
        end_at=datetime(2026, 2, 15, tzinfo=UTC),
    )
    day_two = service.statistics(
        user_id,
        start_at=datetime(2026, 2, 15, tzinfo=UTC),
        end_at=datetime(2026, 2, 16, tzinfo=UTC),
    )
    # the planned slot belongs to the first day, the completion to the second
    assert day_one.planned_sessions == 1
    assert day_one.completed_sessions == 0
    assert day_one.total_training_load == 0
    assert day_one.total_duration_minutes == 0
    assert day_two.planned_sessions == 0
    assert day_two.completed_sessions == 1
    assert day_two.total_training_load == 300
    assert day_two.total_duration_minutes == 30
    assert day_two.total_distance_km == 3
    assert day_two.load_by_type == {"endurance": 300}
    # adjacent windows reconcile with the unbounded totals
    total = service.statistics(user_id)
    assert day_one.planned_sessions + day_two.planned_sessions == total.planned_sessions
    assert day_one.completed_sessions + day_two.completed_sessions == total.completed_sessions
    assert day_one.total_training_load + day_two.total_training_load == 300


def test_training_statistics_window_endpoints_are_half_open(session) -> None:
    user_id = create_user(session)
    service, plan = _create_active_plan(
        session, user_id, datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)
    )
    # planned exactly at the window start, completed exactly at the window end
    _complete_session(
        service,
        plan,
        user_id,
        planned_start=datetime(2026, 2, 10, 0, 0, tzinfo=UTC),
        planned_end=datetime(2026, 2, 10, 2, 0, tzinfo=UTC),
        actual_start=datetime(2026, 2, 10, 0, 0, tzinfo=UTC),
        completed_at=datetime(2026, 2, 11, 0, 0, tzinfo=UTC),
        duration_minutes=10,
        distance_km=1,
    )
    # planned before the window, completed exactly at the window start
    _complete_session(
        service,
        plan,
        user_id,
        planned_start=datetime(2026, 2, 9, 20, 0, tzinfo=UTC),
        planned_end=datetime(2026, 2, 9, 22, 0, tzinfo=UTC),
        actual_start=datetime(2026, 2, 9, 23, 0, tzinfo=UTC),
        completed_at=datetime(2026, 2, 10, 0, 0, tzinfo=UTC),
        duration_minutes=20,
        distance_km=2,
    )
    stats = service.statistics(
        user_id,
        start_at=datetime(2026, 2, 10, tzinfo=UTC),
        end_at=datetime(2026, 2, 11, tzinfo=UTC),
    )
    # start is inclusive, end is exclusive, for both planned and actual times
    assert stats.planned_sessions == 1
    assert stats.completed_sessions == 1
    assert stats.total_duration_minutes == 20
    assert stats.total_distance_km == 2


def test_training_statistics_cancelled_sessions_leave_denominator(session) -> None:
    user_id = create_user(session)
    service, plan = _create_active_plan(
        session, user_id, datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 3, 1, tzinfo=UTC)
    )
    _complete_session(
        service,
        plan,
        user_id,
        planned_start=datetime(2026, 2, 10, 10, 0, tzinfo=UTC),
        planned_end=datetime(2026, 2, 10, 11, 0, tzinfo=UTC),
        actual_start=datetime(2026, 2, 10, 10, 0, tzinfo=UTC),
        completed_at=datetime(2026, 2, 10, 11, 0, tzinfo=UTC),
        duration_minutes=60,
        distance_km=10,
    )
    session.add(
        TrainingSession(
            plan_id=plan.id,
            user_id=user_id,
            title="Cancelled session",
            planned_start_at=datetime(2026, 2, 12, 10, 0, tzinfo=UTC),
            planned_end_at=datetime(2026, 2, 12, 11, 0, tzinfo=UTC),
            status=SessionStatus.CANCELLED,
        )
    )
    session.flush()
    stats = service.statistics(
        user_id,
        start_at=datetime(2026, 2, 1, tzinfo=UTC),
        end_at=datetime(2026, 3, 1, tzinfo=UTC),
    )
    assert stats.planned_sessions == 1
    assert stats.completed_sessions == 1
    assert stats.completion_rate == 100


def test_activity_completion_rate_excludes_cancelled(session) -> None:
    organizer = create_user(session)
    route = create_route(session, actor_id=organizer)
    completed_id = create_expedition(
        session, organizer_id=organizer, route_id=route, offset_days=10
    )
    cancelled_id = create_expedition(
        session, organizer_id=organizer, route_id=route, offset_days=20
    )
    service = ExpeditionService(session)
    for target in ("open", "assembling", "departed", "in_progress", "completed"):
        service.change_status(
            completed_id, ActivityStateChange(target_status=target, actor_id=organizer)
        )
    service.change_status(
        cancelled_id,
        ActivityStateChange(target_status="cancelled", actor_id=organizer, reason="Storm"),
    )
    stats = StatisticsService(session).activities(organizer_id=organizer)
    assert stats.total_activities == 2
    assert stats.completed_activities == 1
    assert stats.cancelled_activities == 1
    assert stats.completion_rate == 100
    assert stats.activities_by_status == {"completed": 1, "cancelled": 1}
    assert sum(stats.activities_by_status.values()) == stats.total_activities
    assert sum(stats.activities_by_risk_level.values()) == stats.total_activities
    dashboard = StatisticsService(session).dashboard()
    assert dashboard.completed_expeditions == stats.completed_activities


def test_activity_statistics_waitlisted_are_not_participants(session) -> None:
    organizer = create_user(session)
    guest = create_user(session, email="waitlist@example.com", name="Waitlisted Guest")
    route = create_route(session, actor_id=organizer)
    expedition_id = create_expedition(session, organizer_id=organizer, route_id=route, capacity=1)
    service = ExpeditionService(session)
    service.change_status(
        expedition_id, ActivityStateChange(target_status="open", actor_id=organizer)
    )
    registration = service.register(
        expedition_id, RegistrationCreate(user_id=guest, idempotency_key="waitlist-key-1")
    )
    assert registration.status == "waitlisted"
    stats = StatisticsService(session).activities(organizer_id=organizer)
    assert stats.total_registrations == 2
    assert stats.confirmed_registrations == 1
    assert stats.average_participants == 1


def test_gear_statistics_partial_return_counts_outstanding_units_once(session) -> None:
    owner = create_user(session)
    borrower = create_user(session, email="partial@example.com", name="Partial Borrower")
    service = GearService(session)
    catalog = service.create_catalog(
        GearCatalogCreate(sku="TENT", name="Tent", category="shelter"), actor_id=owner
    )
    inventory = service.create_inventory(
        GearInventoryCreate(
            catalog_id=catalog.id,
            ownership="club",
            quantity_total=5,
            actor_id=owner,
            idempotency_key="tent-inventory",
        )
    )
    now = datetime.now(UTC)
    loan = service.loan(
        GearLoanCreate(
            inventory_id=inventory.id,
            borrower_id=borrower,
            quantity=3,
            loaned_at=now - timedelta(days=2),
            due_at=now + timedelta(days=5),
            actor_id=owner,
            idempotency_key="tent-loan",
        )
    )
    service.return_loan(
        loan.id,
        GearLoanReturn(
            quantity=1,
            returned_at=now - timedelta(days=1),
            condition_in="good",
            actor_id=owner,
            idempotency_key="tent-return",
        ),
    )
    stats = StatisticsService(session).gear(now=now)
    assert stats.active_loans == 1
    assert stats.loaned_units == 2
    assert stats.utilization_rate == 40
    assert stats.available_inventory_units == 3
    # usage is counted once from loan orders; inventory movements journal the
    # same loan/return events and must not be accumulated a second time
    assert stats.loans_by_catalog == {"Tent": 3}
    movement_rows = session.scalar(
        select(func.count()).select_from(InventoryMovement)
    )
    assert movement_rows == 3  # initial + loan_out + return_in


def test_gear_statistics_excludes_cancelled_loans(session) -> None:
    owner = create_user(session)
    borrower = create_user(session, email="cancel-loan@example.com", name="Cancel Borrower")
    service = GearService(session)
    catalog = service.create_catalog(
        GearCatalogCreate(sku="ROPE", name="Rope", category="safety"), actor_id=owner
    )
    inventory = service.create_inventory(
        GearInventoryCreate(
            catalog_id=catalog.id,
            ownership="club",
            quantity_total=5,
            actor_id=owner,
            idempotency_key="rope-inventory",
        )
    )
    now = datetime.now(UTC)
    service.loan(
        GearLoanCreate(
            inventory_id=inventory.id,
            borrower_id=borrower,
            quantity=2,
            loaned_at=now - timedelta(days=1),
            due_at=now + timedelta(days=3),
            actor_id=owner,
            idempotency_key="rope-loan",
        )
    )
    session.add(
        GearLoan(
            inventory_id=inventory.id,
            borrower_id=borrower,
            quantity=4,
            loaned_at=now - timedelta(days=1),
            due_at=now + timedelta(days=3),
            status=LoanStatus.CANCELLED,
            condition_out="good",
        )
    )
    session.flush()
    stats = StatisticsService(session).gear(now=now)
    assert stats.active_loans == 1
    assert stats.loaned_units == 2
    assert stats.loans_by_catalog == {"Rope": 2}


def test_risk_statistics_open_closed_rules_align_with_dashboard(session) -> None:
    organizer = create_user(session)
    route = create_route(session, actor_id=organizer)
    expedition_id = create_expedition(session, organizer_id=organizer, route_id=route)
    safety = SafetyService(session)
    now = datetime.now(UTC)
    incidents = [
        safety.record_incident(
            EmergencyIncidentCreate(
                expedition_id=expedition_id,
                reported_by=organizer,
                incident_type="injury",
                risk_level="high",
                occurred_at=now - timedelta(hours=index + 1),
                description=f"Incident {index}",
                idempotency_key=f"incident-{index}",
            )
        )
        for index in range(4)
    ]
    safety.update_incident(
        incidents[1].id, EmergencyIncidentUpdate(status="monitoring", actor_id=organizer)
    )
    safety.update_incident(
        incidents[2].id,
        EmergencyIncidentUpdate(
            status="resolved",
            resolution="Treated on site",
            resolved_at=now,
            actor_id=organizer,
        ),
    )
    safety.update_incident(
        incidents[3].id,
        EmergencyIncidentUpdate(
            status="false_alarm",
            resolution="Confirmed as false alarm",
            resolved_at=now,
            actor_id=organizer,
        ),
    )
    session.add(
        ItineraryCheckIn(
            expedition_id=expedition_id,
            user_id=organizer,
            check_in_type="routine",
            due_at=now - timedelta(hours=2),
        )
    )
    session.add(
        ItineraryCheckIn(
            expedition_id=expedition_id,
            user_id=organizer,
            check_in_type="waypoint",
            due_at=now - timedelta(hours=1),
            checked_in_at=now - timedelta(minutes=30),
            is_safe=False,
        )
    )
    session.flush()
    stats = StatisticsService(session).risks(now=now)
    assert stats.total_incidents == 4
    assert stats.open_incidents == 2
    assert stats.resolved_incidents == 2
    assert stats.open_incidents + stats.resolved_incidents == stats.total_incidents
    assert stats.overdue_check_ins == 1
    assert stats.unsafe_check_ins == 1
    assert sum(stats.incidents_by_type.values()) == stats.total_incidents
    assert sum(stats.incidents_by_level.values()) == stats.total_incidents
    dashboard = StatisticsService(session).dashboard(now=now)
    assert dashboard.open_emergencies == stats.open_incidents
    assert dashboard.overdue_check_ins == stats.overdue_check_ins


def test_statistics_reads_do_not_create_audit_logs(session) -> None:
    organizer = create_user(session)
    route = create_route(session, actor_id=organizer)
    create_expedition(session, organizer_id=organizer, route_id=route)
    before = session.scalar(select(func.count()).select_from(AuditLog))
    assert before > 0
    statistics = StatisticsService(session)
    statistics.dashboard()
    statistics.activities()
    statistics.activities(organizer_id=organizer)
    statistics.gear()
    statistics.risks()
    TrainingService(session).statistics(organizer)
    after = session.scalar(select(func.count()).select_from(AuditLog))
    assert after == before


def test_statistics_api_reads_do_not_create_audit_logs(client) -> None:
    user = client.post(
        "/api/v1/users",
        json={"email": "audit-read@example.com", "display_name": "Audit Read"},
    ).json()
    database = client.app.state.database
    with database.session() as session:
        before = session.scalar(select(func.count()).select_from(AuditLog))
    for path in (
        "/api/v1/statistics/dashboard",
        "/api/v1/statistics/activities",
        "/api/v1/statistics/gear",
        "/api/v1/statistics/risks",
        f"/api/v1/training/users/{user['id']}/statistics",
    ):
        response = client.get(path)
        assert response.status_code == 200
    with database.session() as session:
        after = session.scalar(select(func.count()).select_from(AuditLog))
    assert after == before


def test_statistics_endpoints_return_zeros_on_empty_database(client) -> None:
    dashboard = client.get("/api/v1/statistics/dashboard").json()
    assert dashboard["active_users"] == 0
    assert dashboard["completed_expeditions"] == 0
    assert dashboard["open_emergencies"] == 0
    activities = client.get("/api/v1/statistics/activities").json()
    assert activities["total_activities"] == 0
    assert activities["completion_rate"] == 0
    assert activities["activities_by_status"] == {}
    gear = client.get("/api/v1/statistics/gear").json()
    assert gear["total_inventory_units"] == 0
    assert gear["utilization_rate"] == 0
    assert gear["loans_by_catalog"] == {}
    risks = client.get("/api/v1/statistics/risks").json()
    assert risks["total_incidents"] == 0
    assert risks["average_assessment_score"] == 0


def test_statistics_snapshot_is_stable_during_concurrent_writes(database: Database) -> None:
    with database.session() as session:
        create_user(session)
    reader_context = database.session()
    reader = reader_context.__enter__()
    try:
        first = StatisticsService(reader).dashboard()

        def write_user(index: int) -> None:
            def operation(write_session) -> None:
                write_session.add(
                    User(
                        email=f"snapshot-{index}@example.com",
                        display_name=f"Snapshot {index}",
                    )
                )

            database.run_write(operation)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(write_user, range(8)))
        second = StatisticsService(reader).dashboard()
        # the open read transaction keeps one consistent snapshot
        assert second.active_users == first.active_users
    finally:
        reader_context.__exit__(None, None, None)
    with database.session() as session:
        fresh = StatisticsService(session).dashboard()
        assert fresh.active_users == first.active_users + 8


def test_statistics_survive_engine_restart(settings) -> None:
    first = Database(settings)
    initialize_database(first)
    with first.session() as session:
        organizer = create_user(session)
        route = create_route(session, actor_id=organizer)
        expedition_id = create_expedition(session, organizer_id=organizer, route_id=route)
        service = ExpeditionService(session)
        for target in ("open", "assembling", "departed", "in_progress", "completed"):
            service.change_status(
                expedition_id, ActivityStateChange(target_status=target, actor_id=organizer)
            )
    first.engine.dispose()
    second = Database(settings)
    initialize_database(second)
    try:
        with second.session() as session:
            dashboard = StatisticsService(session).dashboard()
            assert dashboard.completed_expeditions == 1
            assert dashboard.total_hiking_distance_km == 10
            assert dashboard.total_elevation_gain_m == 500
            activities = StatisticsService(session).activities()
            assert activities.completion_rate == 100
            assert activities.activities_by_status == {"completed": 1}
    finally:
        second.engine.dispose()


def test_statistics_query_count_does_not_grow_with_data(database: Database) -> None:
    def seed_sessions(count: int, offset_days: int) -> None:
        with database.session() as session:
            user_id = create_user(session, email=f"volume-{offset_days}@example.com")
            service, plan = _create_active_plan(
                session,
                user_id,
                datetime(2026, 2, 1, tzinfo=UTC),
                datetime(2026, 4, 1, tzinfo=UTC),
            )
            for index in range(count):
                day = offset_days + index
                _complete_session(
                    service,
                    plan,
                    user_id,
                    planned_start=datetime(2026, 2, day, 10, 0, tzinfo=UTC),
                    planned_end=datetime(2026, 2, day, 11, 0, tzinfo=UTC),
                    actual_start=datetime(2026, 2, day, 10, 0, tzinfo=UTC),
                    completed_at=datetime(2026, 2, day, 11, 0, tzinfo=UTC),
                    duration_minutes=60,
                    distance_km=10,
                )

    def count_statements(run) -> int:
        statements = 0

        def listener(conn, cursor, statement, parameters, context, executemany) -> None:
            nonlocal statements
            statements += 1

        event.listen(database.engine, "before_cursor_execute", listener)
        try:
            with database.session() as session:
                run(session)
        finally:
            event.remove(database.engine, "before_cursor_execute", listener)
        return statements

    seed_sessions(4, 2)
    small = count_statements(
        lambda session: StatisticsService(session).activities()
    )
    small += count_statements(lambda session: StatisticsService(session).gear())
    small += count_statements(lambda session: StatisticsService(session).risks())
    seed_sessions(16, 10)
    large = count_statements(
        lambda session: StatisticsService(session).activities()
    )
    large += count_statements(lambda session: StatisticsService(session).gear())
    large += count_statements(lambda session: StatisticsService(session).risks())
    assert small == large


def test_training_statistics_api_applies_half_open_window(client) -> None:
    user = client.post(
        "/api/v1/users",
        json={"email": "api-stats@example.com", "display_name": "API Stats"},
    ).json()
    plan = client.post(
        "/api/v1/training/plans",
        params={"actor_id": user["id"]},
        json={
            "user_id": user["id"],
            "name": "API Window Plan",
            "goal": "Half-open windows",
            "start_at": "2026-02-01T00:00:00Z",
            "end_at": "2026-03-01T00:00:00Z",
            "target_sessions_per_week": 7,
            "exercises": [
                {
                    "sequence": 1,
                    "name": "Endurance run",
                    "training_type": "endurance",
                    "target_duration_minutes": 60,
                    "target_distance_km": 10,
                }
            ],
        },
    ).json()
    assert client.post(
        f"/api/v1/training/plans/{plan['id']}/status",
        params={"actor_id": user["id"], "target_status": "active"},
    ).status_code == 200
    session_body = client.post(
        "/api/v1/training/sessions",
        params={"actor_id": user["id"]},
        json={
            "plan_id": plan["id"],
            "title": "Cross-midnight run",
            "planned_start_at": "2026-02-14T23:00:00Z",
            "planned_end_at": "2026-02-15T01:00:00Z",
        },
    ).json()
    client.post(
        f"/api/v1/training/sessions/{session_body['id']}/start",
        params={"actor_id": user["id"], "started_at": "2026-02-14T23:30:00Z"},
    )
    complete = client.post(
        f"/api/v1/training/sessions/{session_body['id']}/complete",
        params={"actor_id": user["id"]},
        json={
            "completed_at": "2026-02-15T00:30:00Z",
            "records": [
                {
                    "exercise_id": plan["exercises"][0]["id"],
                    "duration_minutes": 30,
                    "distance_km": 3,
                    "perceived_exertion": 5,
                    "completion_percent": 100,
                }
            ],
        },
    )
    assert complete.status_code == 200
    day_one = client.get(
        f"/api/v1/training/users/{user['id']}/statistics",
        params={"start_at": "2026-02-14T00:00:00Z", "end_at": "2026-02-15T00:00:00Z"},
    ).json()
    assert day_one["planned_sessions"] == 1
    assert day_one["completed_sessions"] == 0
    assert day_one["total_training_load"] == 0
    day_two = client.get(
        f"/api/v1/training/users/{user['id']}/statistics",
        params={"start_at": "2026-02-15T00:00:00Z", "end_at": "2026-02-16T00:00:00Z"},
    ).json()
    assert day_two["planned_sessions"] == 0
    assert day_two["completed_sessions"] == 1
    assert day_two["total_training_load"] == 300
    assert day_two["load_by_type"] == {"endurance": 300}
