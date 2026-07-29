from __future__ import annotations

import json
import threading
import uuid
import csv
from copy import deepcopy
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALORIES_CSV = ROOT / "dataset/captions_cleaned_calories_realistic_average_portions.csv"

ACTIVITY_FACTORS: dict[str, float] = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}

PORTION_FACTORS: dict[str, float] = {
    "none": 0.0,
    "garnish": 0.25,
    "small": 0.5,
    "normal": 1.0,
    "large": 2.0,
    "double": 2.0,
}

COUNT_STEP = 0.5

DEFAULT_PROFILE: dict[str, Any] = {
    "name": "",
    "sex": "",
    "age": None,
    "height_cm": None,
    "weight_kg": None,
    "activity_level": "sedentary",
    "daily_calorie_goal_kcal": None,
}


def parse_average_portion(value: Any) -> tuple[float | None, float | None]:
    text = str(value or "").strip()
    if not text:
        return None, None
    left, sep, right = text.partition("->")
    if not sep:
        return None, None
    try:
        grams = float(left.strip().split()[0])
        kcal = float(right.strip().split()[0])
    except (IndexError, ValueError):
        return None, None
    return grams, kcal


def load_calorie_table(path: Path = DEFAULT_CALORIES_CSV) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = clean_text(row.get("ingredient"), "ingredient", max_length=80).lower()
            if not name:
                continue
            average_g, average_kcal = parse_average_portion(row.get("average_portion_g_to_calories"))
            rows[name] = {
                "name": name,
                "calories_per_100g": parse_optional_float(
                    row.get("calories_per_100g"),
                    "calories_per_100g",
                    minimum=0,
                    maximum=20000,
                ),
                "calories_per_single_object": parse_optional_float(
                    row.get("calories_per_single_object"),
                    "calories_per_single_object",
                    minimum=0,
                    maximum=20000,
                ),
                "average_portion_g": average_g,
                "average_portion_kcal": average_kcal,
                "calories_per_portion": average_kcal,
                "countable": row.get("calories_per_single_object") not in (None, ""),
            }
    return rows


def now_local_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_date(value: str | None, fallback: date) -> date:
    if not value:
        return fallback
    try:
        return date.fromisoformat(value[:10])
    except ValueError as exc:
        raise ValueError(f"Invalid date: {value}") from exc


def parse_consumed_at(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return now_local_iso()
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("consumed_at must be an ISO date/time") from exc
    return parsed.replace(microsecond=0).isoformat()


def date_key_from_consumed_at(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) >= 10:
        return text[:10]
    return date.today().isoformat()


def parse_optional_int(value: Any, field: str, *, minimum: int, maximum: int) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{field} must be between {minimum} and {maximum}")
    return parsed


def parse_optional_float(value: Any, field: str, *, minimum: float, maximum: float) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{field} must be between {minimum:g} and {maximum:g}")
    return round(parsed, 2)


def parse_optional_count(value: Any, field: str) -> float | None:
    parsed = parse_optional_float(value, field, minimum=COUNT_STEP, maximum=100)
    if parsed is None:
        return None
    doubled = parsed / COUNT_STEP
    if abs(doubled - round(doubled)) > 1e-6:
        raise ValueError(f"{field} must use 0.5 steps")
    return round(parsed, 1)


def clean_text(value: Any, field: str, *, max_length: int) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) > max_length:
        raise ValueError(f"{field} must be at most {max_length} characters")
    return text


def normalize_portion(value: Any, *, default: str = "") -> str:
    portion = clean_text(value, "portion_category", max_length=40).lower()
    if not portion:
        return default
    if portion not in PORTION_FACTORS:
        raise ValueError(f"portion_category must be one of: {', '.join(PORTION_FACTORS)}")
    return portion


def round_kcal(value: float) -> float:
    if value <= 0:
        return 0.0
    return float(max(1, int(float(value) + 0.5)))


def first_number(*values: Any) -> float | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def portion_reference_kcal(item: dict[str, Any], old_kcal: float | None, old_portion: str) -> float | None:
    direct = first_number(item.get("calories_per_portion"), item.get("average_portion_kcal"))
    if direct is not None and direct > 0:
        return direct
    single = first_number(item.get("calories_per_single_object"), item.get("per_instance_kcal"))
    if single is not None and single > 0:
        return single
    base_kcal = first_number(item.get("base_kcal"))
    base_portion = normalize_portion(item.get("base_portion_category"), default="")
    base_factor = PORTION_FACTORS.get(base_portion or "")
    if base_kcal is not None and base_factor and base_factor > 0:
        return base_kcal / base_factor
    old_factor = PORTION_FACTORS.get(old_portion or "")
    if old_kcal is not None and old_factor and old_factor > 0:
        return old_kcal / old_factor
    return None


def count_reference_kcal(item: dict[str, Any], old_kcal: float | None, old_count: float | None) -> float | None:
    direct = first_number(item.get("calories_per_single_object"), item.get("per_instance_kcal"))
    if direct is not None and direct > 0:
        return direct
    base_kcal = first_number(item.get("base_kcal"))
    base_count = first_number(item.get("base_count"))
    if base_kcal is not None and base_count is not None and base_count > 0:
        return base_kcal / base_count
    if old_kcal is not None and old_count is not None and old_count > 0:
        return old_kcal / old_count
    return None


def recalculated_ingredient_kcal(item: dict[str, Any], count: float | None, portion: str, old_kcal: float | None) -> float | None:
    old_count = parse_optional_count(item.get("count"), "ingredient count") if item.get("count") not in (None, "") else None
    old_portion = normalize_portion(item.get("portion_category"), default="normal")
    if count is not None:
        reference = count_reference_kcal(item, old_kcal, old_count)
        if reference is None:
            raise ValueError(f"Cannot recalculate count calories for {item.get('name') or 'ingredient'}")
        return round_kcal(count * reference)
    if item.get("abundance_scene"):
        reference = first_number(item.get("per_instance_kcal"), item.get("calories_per_single_object"), old_kcal)
        return round_kcal(reference) if reference is not None else old_kcal
    reference = portion_reference_kcal(item, old_kcal, old_portion)
    if reference is None:
        return old_kcal
    return round_kcal(PORTION_FACTORS[portion] * reference)


def clean_ingredients(value: Any, calorie_table: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    cleaned: list[dict[str, Any]] = []
    calorie_table = calorie_table or {}
    for item in value[:30]:
        if not isinstance(item, dict):
            continue
        name = clean_text(item.get("name"), "ingredient name", max_length=80).lower()
        if not name:
            continue
        reference = calorie_table.get(name)
        if calorie_table and reference is None:
            raise ValueError(f"ingredient must be from the calorie table: {name}")
        item_with_reference = {**item, "name": name}
        if reference is not None:
            for key in (
                "calories_per_100g",
                "calories_per_single_object",
                "calories_per_portion",
                "average_portion_g",
                "average_portion_kcal",
                "countable",
            ):
                item_with_reference[key] = reference.get(key)
        count = parse_optional_count(item.get("count"), "ingredient count")
        portion = normalize_portion(item.get("portion_category"), default="normal")
        if count is not None:
            portion = "none"
        old_kcal = parse_optional_float(item_with_reference.get("kcal"), "ingredient kcal", minimum=0, maximum=20000)
        recalculated_kcal = recalculated_ingredient_kcal(item_with_reference, count, portion, old_kcal)
        cleaned.append(
            {
                "name": name,
                "kcal": recalculated_kcal,
                "count": count,
                "portion_category": portion,
                "calories_per_single_object": parse_optional_float(
                    item_with_reference.get("calories_per_single_object"),
                    "calories_per_single_object",
                    minimum=0,
                    maximum=20000,
                ),
                "calories_per_portion": parse_optional_float(
                    item_with_reference.get("calories_per_portion"),
                    "calories_per_portion",
                    minimum=0,
                    maximum=20000,
                ),
                "average_portion_kcal": parse_optional_float(
                    item_with_reference.get("average_portion_kcal"),
                    "average_portion_kcal",
                    minimum=0,
                    maximum=20000,
                ),
                "per_instance_kcal": parse_optional_float(
                    item_with_reference.get("per_instance_kcal"),
                    "per_instance_kcal",
                    minimum=0,
                    maximum=20000,
                ),
                "portion_factor": PORTION_FACTORS.get(portion),
                "countable": bool(item_with_reference.get("countable")),
                "abundance_scene": bool(item_with_reference.get("abundance_scene")),
                "calorie_method": clean_text(item_with_reference.get("calorie_method"), "calorie_method", max_length=80),
                "estimated_quantity_g": parse_optional_float(
                    item_with_reference.get("estimated_quantity_g"),
                    "estimated_quantity_g",
                    minimum=0,
                    maximum=20000,
                ),
            }
        )
    return cleaned


class NutritionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.profile_path = root / "profile.json"
        self.history_path = root / "diet_history.json"
        self.calorie_table = load_calorie_table()
        self.lock = threading.RLock()
        self.root.mkdir(parents=True, exist_ok=True)

    def ingredient_options(self) -> dict[str, Any]:
        ingredients = sorted(self.calorie_table.values(), key=lambda item: item["name"])
        return {
            "portion_options": list(PORTION_FACTORS),
            "count_step": COUNT_STEP,
            "ingredients": ingredients,
        }

    def profile_response(self) -> dict[str, Any]:
        with self.lock:
            profile = self._load_profile_locked()
            return {"profile": profile, "requirement": self.calculate_requirement(profile)}

    def save_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            current = self._load_profile_locked()
            merged = {**current, **payload}
            profile = self._validate_profile(merged)
            profile["updated_at"] = now_local_iso()
            self._write_json_locked(self.profile_path, profile)
            return {"profile": profile, "requirement": self.calculate_requirement(profile)}

    def history(self, start: date | None = None, end: date | None = None) -> dict[str, Any]:
        with self.lock:
            entries = self._load_history_locked()
            filtered = self._filter_entries(entries, start, end)
            filtered.sort(key=lambda item: str(item.get("consumed_at") or ""), reverse=True)
            return {"entries": filtered}

    def add_history_entry(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            entries = self._load_history_locked()
            entry = self._validate_history_entry(payload)
            entry["id"] = uuid.uuid4().hex[:12]
            entry["created_at"] = now_local_iso()
            entry["updated_at"] = entry["created_at"]
            entries.append(entry)
            self._write_json_locked(self.history_path, entries)
            return {"entry": entry}

    def update_history_entry(self, entry_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            entries = self._load_history_locked()
            for index, entry in enumerate(entries):
                if entry.get("id") != entry_id:
                    continue
                merged = {**entry, **payload}
                cleaned = self._validate_history_entry(merged)
                cleaned["id"] = entry_id
                cleaned["created_at"] = entry.get("created_at") or now_local_iso()
                cleaned["updated_at"] = now_local_iso()
                entries[index] = cleaned
                self._write_json_locked(self.history_path, entries)
                return {"entry": cleaned}
            raise ValueError("History entry not found")

    def delete_history_entry(self, entry_id: str) -> dict[str, Any]:
        with self.lock:
            entries = self._load_history_locked()
            kept = [entry for entry in entries if entry.get("id") != entry_id]
            if len(kept) == len(entries):
                raise ValueError("History entry not found")
            self._write_json_locked(self.history_path, kept)
            return {"deleted": True, "id": entry_id}

    def summary(self, start: date, end: date) -> dict[str, Any]:
        if end < start:
            raise ValueError("to date must be after from date")
        if (end - start).days > 366:
            raise ValueError("Summary range cannot exceed 366 days")
        with self.lock:
            profile = self._load_profile_locked()
            requirement = self.calculate_requirement(profile)
            all_entries = self._load_history_locked()
            entries = self._filter_entries(all_entries, start, end)

        totals: dict[str, dict[str, Any]] = {}
        for entry in entries:
            key = date_key_from_consumed_at(entry.get("consumed_at"))
            bucket = totals.setdefault(key, {"date": key, "total_kcal": 0.0, "entry_count": 0})
            bucket["total_kcal"] += float(entry.get("calories_kcal") or 0)
            bucket["entry_count"] += 1

        limit = requirement.get("daily_limit_kcal")
        days: list[dict[str, Any]] = []
        cursor = start
        while cursor <= end:
            key = cursor.isoformat()
            bucket = totals.get(key, {"date": key, "total_kcal": 0.0, "entry_count": 0})
            day = {
                "date": key,
                "total_kcal": round(float(bucket["total_kcal"]), 1),
                "entry_count": int(bucket["entry_count"]),
                "limit_kcal": limit,
                "remaining_kcal": None,
                "status": "no_limit",
            }
            if isinstance(limit, (int, float)):
                remaining = float(limit) - day["total_kcal"]
                day["remaining_kcal"] = round(remaining, 1)
                day["status"] = "over" if remaining < 0 else "under"
            days.append(day)
            cursor += timedelta(days=1)

        tracked_days = [item for item in days if item["entry_count"] > 0]
        over_days = [item for item in tracked_days if item["status"] == "over"]
        under_days = [item for item in tracked_days if item["status"] == "under"]
        total_kcal = round(sum(item["total_kcal"] for item in tracked_days), 1)
        today_key = date.today().isoformat()
        today = next((item for item in days if item["date"] == today_key), None)
        if today is None:
            today_entries = self._filter_entries(all_entries, date.today(), date.today())
            today_total = round(sum(float(item.get("calories_kcal") or 0) for item in today_entries), 1)
            today = {
                "date": today_key,
                "total_kcal": today_total,
                "entry_count": len(today_entries),
                "limit_kcal": limit,
                "remaining_kcal": round(float(limit) - today_total, 1) if isinstance(limit, (int, float)) else None,
                "status": "no_limit" if not isinstance(limit, (int, float)) else ("over" if today_total > float(limit) else "under"),
            }

        return {
            "profile": profile,
            "requirement": requirement,
            "from": start.isoformat(),
            "to": end.isoformat(),
            "days": days,
            "today": today,
            "stats": {
                "tracked_days": len(tracked_days),
                "under_limit_days": len(under_days),
                "over_limit_days": len(over_days),
                "average_daily_kcal": round(total_kcal / len(tracked_days), 1) if tracked_days else 0,
                "total_kcal": total_kcal,
                "max_day": max(tracked_days, key=lambda item: item["total_kcal"], default=None),
                "min_day": min(tracked_days, key=lambda item: item["total_kcal"], default=None),
            },
        }

    def calculate_requirement(self, profile: dict[str, Any]) -> dict[str, Any]:
        missing = [
            field
            for field in ("sex", "age", "height_cm", "weight_kg", "activity_level")
            if profile.get(field) in (None, "")
        ]
        goal = profile.get("daily_calorie_goal_kcal")
        response: dict[str, Any] = {
            "bmr_kcal": None,
            "maintenance_kcal": None,
            "daily_limit_kcal": int(round(goal)) if isinstance(goal, (int, float)) else None,
            "activity_factor": ACTIVITY_FACTORS.get(str(profile.get("activity_level") or "sedentary")),
            "source": "manual_goal" if isinstance(goal, (int, float)) else "missing_profile",
            "missing_fields": missing,
        }
        if missing:
            return response

        sex_offset = 5 if profile["sex"] == "male" else -161
        bmr = 10 * float(profile["weight_kg"]) + 6.25 * float(profile["height_cm"]) - 5 * int(profile["age"]) + sex_offset
        maintenance = bmr * float(ACTIVITY_FACTORS[str(profile["activity_level"])])
        response["bmr_kcal"] = int(round(bmr))
        response["maintenance_kcal"] = int(round(maintenance))
        if response["daily_limit_kcal"] is None:
            response["daily_limit_kcal"] = response["maintenance_kcal"]
            response["source"] = "estimated_requirement"
        return response

    def _load_profile_locked(self) -> dict[str, Any]:
        payload = self._read_json_locked(self.profile_path, DEFAULT_PROFILE)
        if not isinstance(payload, dict):
            return deepcopy(DEFAULT_PROFILE)
        return self._validate_profile({**DEFAULT_PROFILE, **payload})

    def _load_history_locked(self) -> list[dict[str, Any]]:
        payload = self._read_json_locked(self.history_path, [])
        if not isinstance(payload, list):
            return []
        entries: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict) and item.get("id"):
                entries.append(item)
        return entries

    def _read_json_locked(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return deepcopy(default)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON data in {path}") from exc

    def _write_json_locked(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp_path.replace(path)

    def _validate_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        sex = clean_text(payload.get("sex"), "sex", max_length=20).lower()
        if sex not in {"", "male", "female"}:
            raise ValueError("sex must be male or female")
        activity = clean_text(payload.get("activity_level") or "sedentary", "activity_level", max_length=40)
        if activity not in ACTIVITY_FACTORS:
            raise ValueError(f"activity_level must be one of: {', '.join(ACTIVITY_FACTORS)}")
        return {
            "name": clean_text(payload.get("name"), "name", max_length=80),
            "sex": sex,
            "age": parse_optional_int(payload.get("age"), "age", minimum=10, maximum=120),
            "height_cm": parse_optional_float(payload.get("height_cm"), "height_cm", minimum=80, maximum=250),
            "weight_kg": parse_optional_float(payload.get("weight_kg"), "weight_kg", minimum=25, maximum=350),
            "activity_level": activity,
            "daily_calorie_goal_kcal": parse_optional_int(
                payload.get("daily_calorie_goal_kcal"),
                "daily_calorie_goal_kcal",
                minimum=800,
                maximum=6000,
            ),
            "updated_at": clean_text(payload.get("updated_at"), "updated_at", max_length=40),
        }

    def _validate_history_entry(self, payload: dict[str, Any]) -> dict[str, Any]:
        dish_name = clean_text(payload.get("dish_name"), "dish_name", max_length=120)
        if not dish_name:
            raise ValueError("dish_name is required")
        ingredients = clean_ingredients(payload.get("ingredients"), self.calorie_table)
        if not ingredients:
            raise ValueError("at least one ingredient from the calorie table is required")
        ingredients_total = sum(float(item.get("kcal") or 0) for item in ingredients)
        calories = round_kcal(ingredients_total) if ingredients else parse_optional_float(
            payload.get("calories_kcal"),
            "calories_kcal",
            minimum=1,
            maximum=20000,
        )
        if calories is None:
            raise ValueError("calories_kcal is required")
        return {
            "id": clean_text(payload.get("id"), "id", max_length=40),
            "consumed_at": parse_consumed_at(payload.get("consumed_at")),
            "dish_name": dish_name,
            "calories_kcal": calories,
            "notes": clean_text(payload.get("notes"), "notes", max_length=500),
            "source_job_id": clean_text(payload.get("source_job_id"), "source_job_id", max_length=80),
            "source_run_dir": clean_text(payload.get("source_run_dir"), "source_run_dir", max_length=300),
            "image_name": clean_text(payload.get("image_name"), "image_name", max_length=160),
            "estimation_scope": clean_text(payload.get("estimation_scope"), "estimation_scope", max_length=80),
            "ingredients": ingredients,
        }

    def _filter_entries(self, entries: list[dict[str, Any]], start: date | None, end: date | None) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []
        for entry in entries:
            key = date_key_from_consumed_at(entry.get("consumed_at"))
            try:
                entry_date = date.fromisoformat(key)
            except ValueError:
                continue
            if start is not None and entry_date < start:
                continue
            if end is not None and entry_date > end:
                continue
            filtered.append(deepcopy(entry))
        return filtered
