"""OR-Tools scheduler for assigning railway maintenance tasks to teams."""

from ortools.sat.python import cp_model


def schedule_maintenance_tasks(tasks: list[dict], team_count: int) -> list[dict]:
    """Return a priority-aware, non-overlapping task plan.

    Each task needs ``task_name``, ``asset_type``, ``duration_days``,
    ``priority`` (1-5), and ``due_day`` (relative to the plan start).
    """
    if not tasks:
        return []
    if team_count < 1:
        raise ValueError("At least one maintenance team is required.")

    model = cp_model.CpModel()
    horizon = sum(int(task["duration_days"]) for task in tasks)
    team_intervals = [[] for _ in range(team_count)]
    starts, ends, assignments, tardiness = [], [], [], []

    for index, task in enumerate(tasks):
        duration = int(task["duration_days"])
        if duration < 1:
            raise ValueError("Every task duration must be at least one day.")

        start = model.NewIntVar(0, horizon - duration, f"start_{index}")
        end = model.NewIntVar(duration, horizon, f"end_{index}")
        model.Add(end == start + duration)
        starts.append(start)
        ends.append(end)

        assigned_to = []
        for team in range(team_count):
            is_assigned = model.NewBoolVar(f"task_{index}_team_{team}")
            interval = model.NewOptionalIntervalVar(
                start, duration, end, is_assigned, f"task_{index}_on_team_{team}"
            )
            team_intervals[team].append(interval)
            assigned_to.append(is_assigned)
        model.AddExactlyOne(assigned_to)
        assignments.append(assigned_to)

        late = model.NewIntVar(0, horizon, f"late_{index}")
        model.AddMaxEquality(late, [end - int(task["due_day"]), 0])
        tardiness.append(late)

    for intervals in team_intervals:
        model.AddNoOverlap(intervals)

    # Avoid late high-priority work first; use finish dates to break ties.
    model.Minimize(
        sum((int(tasks[i]["priority"]) * 1000) * tardiness[i] for i in range(len(tasks)))
        + sum(ends)
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible maintenance schedule could be created.")

    plan = []
    for index, task in enumerate(tasks):
        team = next(team + 1 for team, flag in enumerate(assignments[index]) if solver.Value(flag))
        plan.append({
            **task,
            "assigned_team": f"Team {team}",
            "start_day": solver.Value(starts[index]) + 1,
            "end_day": solver.Value(ends[index]),
            "late_days": solver.Value(tardiness[index]),
        })
    return sorted(plan, key=lambda item: (item["start_day"], item["assigned_team"]))
