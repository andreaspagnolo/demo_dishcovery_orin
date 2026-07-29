#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import orin_calorie_demo as base
from edgellm_qwen import EdgeLLMQwenScorer, attach_wrapper_args, config_from_args, parse_wrapper_args
from orin_calorie_demo import *  # noqa: F401,F403


_BASE_PARSE_ARGS = base.parse_args
_BASE_LOAD_RUNTIME = base.load_runtime
_BASE_BUILD_SINGLE_REPORT = base.build_single_report
_BASE_BUILD_SELECTED_COMPOSITION_PROMPT = base.build_selected_composition_prompt
_BASE_EXTRACT_JSON_OBJECT = base.extract_json_object
_BASE_MAIN = base.main
DEFAULT_EDGELLM_CALORIE_TOP_K = 20
EDGELLM_CALORIE_LOGIC_COMPACT = "compact"
EDGELLM_CALORIE_LOGIC_TASK1_QUANTITY = "task1_then_quantity"
EDGELLM_CALORIE_LOGIC_CHOICES = (EDGELLM_CALORIE_LOGIC_COMPACT, EDGELLM_CALORIE_LOGIC_TASK1_QUANTITY)
OPTIMIZED_EDGELLM_MODEL_NAME = "Qwen3-VL-4B-Instruct"
OPTIMIZED_EDGELLM_RUNTIME_MODE = "persistent"
OPTIMIZED_EDGELLM_TIMEOUT_SEC = 300.0
OPTIMIZED_EDGELLM_STARTUP_TIMEOUT_SEC = 300.0
EDGELLM_4096_MAX_PIXELS = 768 * 28 * 28
_CUT_NAME_TO_CODE = {
    "whole": 0,
    "half": 1,
    "quarter": 2,
    "wedge": 3,
    "wedge_or_large_piece": 3,
    "large_piece": 3,
    "slice": 4,
    "slices": 4,
    "sliced": 4,
    "slice_unclear": 4,
    "unclear": 4,
    "chopped": 5,
    "diced": 5,
    "pile": 5,
    "chopped_diced_pile": 5,
}
_CONF_NAME_TO_CODE = {"low": 0, "medium": 1, "med": 1, "high": 2}
_PORTION_NAME_TO_CODE = {name: code for code, name in base.PORTION_CODE_TO_CATEGORY.items()}
_QUANTITY_MODE_NAME_TO_CODE = {
    "count": 0,
    "counts": 0,
    "whole_count": 0,
    "whole_object_count": 0,
    "object_count": 0,
    "portion": 1,
    "portion_category": 1,
    "serving": 1,
    "serving_size": 1,
}


def repair_split_args(argv: list[str]) -> list[str]:
    repaired: list[str] = []
    idx = 0
    split_prefixes = {
        "--qwen-": {"model", "max-new-tokens", "min-pixels", "max-pixels"},
        "--vlm-": {"model", "max-new-tokens", "min-pixels", "max-pixels"},
        "--calorie-": {
            "candidate-top-k",
            "candidate-list-mode",
            "dynamic-candidate-relative-delta",
            "dynamic-candidate-min-k",
            "dynamic-candidate-max-k",
            "siglip-filter",
            "counting-logic",
        },
        "--edgellm-": {
            "root",
            "workspace",
            "model-name",
            "llm-engine-profile",
            "llm-engine-dir",
            "visual-engine-dir",
            "binary",
            "persistent-binary",
            "plugin-path",
            "runtime-mode",
            "timeout-sec",
            "startup-timeout-sec",
            "keep-io",
            "dump-profile",
            "warmup",
            "temperature",
            "top-p",
            "top-k",
            "enable-thinking",
            "calorie-logic",
        },
    }
    while idx < len(argv):
        item = argv[idx]
        if idx + 1 < len(argv):
            suffixes = split_prefixes.get(item)
            if suffixes and argv[idx + 1] in suffixes:
                repaired.append(f"{item}{argv[idx + 1]}")
                idx += 2
                continue
        repaired.append(item)
        idx += 1
    return repaired


def _has_any_arg(argv: list[str], names: tuple[str, ...]) -> bool:
    for item in argv:
        for name in names:
            if item == name or item.startswith(name + "="):
                return True
    return False


def parse_edgellm_calorie_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--edgellm-calorie-logic",
        choices=EDGELLM_CALORIE_LOGIC_CHOICES,
        default=EDGELLM_CALORIE_LOGIC_TASK1_QUANTITY,
    )
    return parser.parse_known_args(argv)


def apply_optimized_edgellm_defaults(args: argparse.Namespace, argv: list[str]) -> argparse.Namespace:
    if getattr(args, "vlm_backend", "edgellm") != "edgellm":
        return args
    if not _has_any_arg(argv, ("--edgellm-model-name",)):
        args.edgellm_model_name = OPTIMIZED_EDGELLM_MODEL_NAME
    if not _has_any_arg(argv, ("--edgellm-runtime-mode",)):
        args.edgellm_runtime_mode = OPTIMIZED_EDGELLM_RUNTIME_MODE
    if not _has_any_arg(argv, ("--edgellm-timeout-sec",)):
        args.edgellm_timeout_sec = OPTIMIZED_EDGELLM_TIMEOUT_SEC
    if not _has_any_arg(argv, ("--edgellm-startup-timeout-sec",)):
        args.edgellm_startup_timeout_sec = OPTIMIZED_EDGELLM_STARTUP_TIMEOUT_SEC
    return args


def apply_edgellm_calorie_defaults(args: argparse.Namespace, remaining: list[str]) -> argparse.Namespace:
    if getattr(args, "vlm_backend", "edgellm") != "edgellm":
        return args
    if not _has_any_arg(remaining, ("--calorie-candidate-list-mode", "--candidate-list-mode")):
        args.calorie_candidate_list_mode = "fixed_topk"
    if not _has_any_arg(remaining, ("--calorie-candidate-top-k", "--top-k")):
        args.calorie_candidate_top_k = DEFAULT_EDGELLM_CALORIE_TOP_K
    if not _has_any_arg(remaining, ("--calorie-siglip-filter", "--no-calorie-siglip-filter")):
        args.calorie_siglip_filter = True
    if (
        str(getattr(args, "edgellm_llm_engine_profile", "")).strip().lower() in {"max-input-4096", "in4096"}
        and not _has_any_arg(remaining, ("--vlm-max-pixels", "--qwen-max-pixels"))
    ):
        args.qwen_max_pixels = EDGELLM_4096_MAX_PIXELS
    return args


def parse_args() -> argparse.Namespace:
    original_args = repair_split_args(sys.argv[1:])
    calorie_logic_args, remaining_after_calorie = parse_edgellm_calorie_args(original_args)
    wrapper_args, remaining = parse_wrapper_args(remaining_after_calorie)
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0], *remaining]
        args = _BASE_PARSE_ARGS()
    finally:
        sys.argv = old_argv
    args = attach_wrapper_args(args, wrapper_args)
    args.edgellm_calorie_logic = calorie_logic_args.edgellm_calorie_logic
    args = apply_optimized_edgellm_defaults(args, original_args)
    return apply_edgellm_calorie_defaults(args, remaining)


def build_compact_edgellm_prompt(
    calories_by_name: dict[str, Any],
    candidate_rows: list[Any] | None,
    counting_logic: str,
) -> str:
    del counting_logic
    rows = sorted(
        candidate_rows if candidate_rows else list(calories_by_name.values())[:DEFAULT_EDGELLM_CALORIE_TOP_K],
        key=lambda row: row.ingredient_id,
    )
    allowed = "; ".join(f"{row.ingredient_id}:{row.ingredient}" for row in rows)
    countable = ",".join(str(row.ingredient_id) for row in rows if row.calories_per_single_object is not None) or "none"
    return (
        "Return JSON array only, no objects and no keys. Format: [scene,[id,pieces,cut,whole_count,conf,portion],...].\n"
        "Example: [0,[211,4,0,4,2,0]]. Do not output {\"id\":...} objects.\n"
        "scene 0 normal serving, 1 market/display abundance. "
        "cut 0 whole,1 half,2 quarter,3 wedge,4 slices/unclear,5 chopped/pile. "
        "conf 0 low,1 med,2 high. "
        "portion 0 none,1 garnish,2 small,3 normal,4 large,5 double.\n"
        "Use only visible edible foods from Allowed. Omit hidden/inferred foods. Max 8 rows. "
        "whole_count is whole-object equivalent only when clear and countable; "
        "slices/chopped/unclear use whole_count null and portion>0. "
        "portion 0 only with reliable whole_count. "
        "For abundance use scene 1, pieces null, whole_count null, portion 4 or 5.\n"
        f"Allowed: {allowed}\n"
        f"Countable ids: {countable}"
    )


def rows_from_task1_labels(
    labels: list[str],
    calories_by_name: dict[str, Any],
) -> tuple[list[Any], list[dict[str, Any]], list[str]]:
    rows: list[Any] = []
    trace: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_ids: set[int] = set()
    for rank, label in enumerate(labels, start=1):
        row, warning = base.match_calorie_row(label, calories_by_name)
        if warning:
            warnings.append(f"task1_{warning}")
        if row is not None and row.ingredient_id not in seen_ids:
            rows.append(row)
            seen_ids.add(row.ingredient_id)
        trace.append(
            {
                "rank": rank,
                "label": label,
                "ingredient_id": row.ingredient_id if row is not None else None,
                "ingredient": row.ingredient if row is not None else None,
            }
        )
    return rows, trace, warnings

def build_quantity_only_prompt(selected_rows):
    selected = "; ".join(f"{row.ingredient_id}:{row.ingredient}" for row in selected_rows)
    countable = ",".join(
        str(row.ingredient_id)
        for row in selected_rows
        if row.calories_per_single_object is not None
    ) or "none"

    return (
        "Estimate quantity only for selected foods. Do not add foods.\n"
        "Return JSON only: [scene,[id,mode,value],[id,mode,value]...]\n"
        "scene: 1 normal dish for 1-2 people, 2 massive/bulk/market/tree/display.\n"
        "mode: 0 count, 1 portion.\n"
        "If mode 0, value = count of COMPLETE natural food units, not pieces.\n"
        "If mode 1, value = portion: 1 garnish, 2 small, 3 normal, 4 large.\n\n"
        "Rules:\n"
        "1. Use only Selected ids, once each.\n"
        "2. If scene=2, do not count total food. Use [id,1,3] for every id.\n"
        "3. Use mode 0 only for clear complete units: whole apple, whole egg, whole banana, whole tomato...\n"
        "4. Never count slices, wedges, cubes, sticks, spears, leaves, strips, chopped food, piles, grains.\n"
        "5. If unsure or if the food is cut into many visible pieces, use mode 1.\n"
        "6. Default portion is 3 normal. Use 1 only for tiny garnish/sauce/herbs/seeds.\n\n"
        f"Selected ids: {selected}\n"
        f"Countable ids: {countable}"
    )


def default_quantity_row(row: Any) -> list[Any]:
    return [row.ingredient_id, None, 5, None, 0, 3]


def coerce_quantity_mode(value: Any) -> int | None:
    code = base.coerce_int(value)
    if code in {0, 1}:
        return code
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _QUANTITY_MODE_NAME_TO_CODE.get(text)


def coerce_quantity_portion_code(value: Any, calorie_row: Any) -> tuple[int, list[str]]:
    warnings: list[str] = []
    code = base.coerce_int(value)
    if code is None:
        text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        code = _PORTION_NAME_TO_CODE.get(text)
    if code == 5:
        code = 4
        warnings.append(f"quantity_double_portion_mapped_to_large:{calorie_row.ingredient}")
    if code not in {1, 2, 3, 4}:
        code = 3
        warnings.append(f"quantity_invalid_portion_used_normal:{calorie_row.ingredient}")
    return code, warnings


def quantity_scene_value(parsed: Any) -> Any:
    if isinstance(parsed, dict):
        return (
            parsed.get("scene")
            or parsed.get("scene_context")
            or parsed.get("scene_type")
            or parsed.get("context")
            or parsed.get("food_context")
        )
    if isinstance(parsed, list) and parsed:
        first_text = str(parsed[0]).strip().lower()
        if first_text == "scene" and len(parsed) >= 2:
            return parsed[1]
        if not isinstance(parsed[0], (list, dict)):
            return parsed[0]
    return None


def parse_quantity_scene(parsed: Any) -> int:
    value = quantity_scene_value(parsed)
    code = base.coerce_int(value)
    if code == 2:
        return 1
    if code in {0, 1}:
        return 0
    text = str(value or "").strip().lower().replace("_", " ")
    if any(
        marker in text
        for marker in ("massive", "abundance", "display", "market", "stall", "basket", "crate", "bulk", "tree")
    ):
        return 1
    return 0


def looks_like_quantity_mode_row(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and base.coerce_int(value[0]) is not None
        and coerce_quantity_mode(value[1]) in {0, 1}
    )


def quantity_rows_payload(parsed: Any) -> list[Any]:
    if isinstance(parsed, dict):
        for key in ("rows", "items", "ingredients", "compact"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
        return [parsed] if ("id" in parsed or "ingredient_id" in parsed) else []
    if isinstance(parsed, list):
        if not parsed:
            return []
        if looks_like_quantity_mode_row(parsed):
            return [parsed]
        start = 0
        first_text = str(parsed[0]).strip().lower() if parsed else ""
        if first_text == "scene" and len(parsed) >= 2:
            start = 2
        elif not isinstance(parsed[0], (list, dict)):
            start = 1
        rows = parsed[start:]
        if len(rows) == 1 and isinstance(rows[0], list) and rows[0] and all(isinstance(item, list) for item in rows[0]):
            return rows[0]
        return rows
    return []


def normalize_quantity_mode_row(item: list[Any], calorie_row: Any, scene: int) -> tuple[list[Any], list[str]]:
    warnings: list[str] = []
    ingredient_id = base.coerce_int(item[0]) if item else None
    if scene == 1:
        return [ingredient_id, None, 5, None, 0, 3], warnings

    mode = coerce_quantity_mode(item[1] if len(item) > 1 else None)
    value = item[2] if len(item) > 2 else None
    if mode == 0:
        count = base.coerce_count(value)
        if calorie_row.calories_per_single_object is None:
            warnings.append(f"quantity_count_mode_non_countable_used_portion:{calorie_row.ingredient}")
            return [ingredient_id, None, 5, None, 0, 3], warnings
        if count is None:
            warnings.append(f"quantity_invalid_count_used_normal_portion:{calorie_row.ingredient}")
            return [ingredient_id, None, 5, None, 0, 3], warnings
        if count > 10:
            warnings.append(f"quantity_count_over_10_used_portion:{calorie_row.ingredient}")
            return [ingredient_id, None, 5, None, 0, 4], warnings
        return [ingredient_id, None, None, count, 2, 0], warnings

    if mode == 1:
        portion_code, portion_warnings = coerce_quantity_portion_code(value, calorie_row)
        warnings.extend(portion_warnings)
        return [ingredient_id, None, 5, None, 0, portion_code], warnings

    warnings.append(f"quantity_invalid_mode_used_normal_portion:{calorie_row.ingredient}")
    return [ingredient_id, None, 5, None, 0, 3], warnings


def restrict_quantity_payload_to_selected(parsed: Any, selected_rows: list[Any]) -> tuple[Any, list[str]]:
    selected_by_id = {int(row.ingredient_id): row for row in selected_rows}
    selected_ids = set(selected_by_id)
    scene = parse_quantity_scene(parsed)
    raw_rows = quantity_rows_payload(parsed)
    rows: list[Any] = []
    seen: set[int] = set()
    warnings: list[str] = []
    for item in raw_rows:
        if not isinstance(item, list):
            warnings.append("quantity_ignored_non_list_row")
            continue
        ingredient_id = base.coerce_int(item[0]) if item else None
        if ingredient_id is None:
            warnings.append("quantity_ignored_row_without_id")
            continue
        if ingredient_id not in selected_ids:
            warnings.append(f"quantity_ignored_unselected_id:{ingredient_id}")
            continue
        if ingredient_id in seen:
            warnings.append(f"quantity_ignored_duplicate_id:{ingredient_id}")
            continue
        if looks_like_quantity_mode_row(item):
            row, row_warnings = normalize_quantity_mode_row(item, selected_by_id[ingredient_id], scene)
        elif len(item) >= 7:
            row = [item[0], item[1], item[2], item[3], item[4], item[6]]
            row, row_warnings = normalize_quantity_row(row, selected_by_id[ingredient_id], scene)
        elif len(item) >= 6:
            row = list(item[:6])
            row, row_warnings = normalize_quantity_row(row, selected_by_id[ingredient_id], scene)
        elif len(item) >= 4:
            row = list(item[:4])
            row, row_warnings = normalize_quantity_row(row, selected_by_id[ingredient_id], scene)
        else:
            warnings.append(f"quantity_ignored_short_row:{ingredient_id}")
            continue
        warnings.extend(row_warnings)
        rows.append(row)
        seen.add(ingredient_id)
    for row in selected_rows:
        ingredient_id = int(row.ingredient_id)
        if ingredient_id not in seen:
            rows.append(default_quantity_row(row))
            warnings.append(f"quantity_missing_used_normal_portion:{row.ingredient}")
    return [scene, *rows], warnings


def normalize_quantity_row(row: list[Any], calorie_row: Any, scene: int) -> tuple[list[Any], list[str]]:
    warnings: list[str] = []
    padded = list(row[:6])
    while len(padded) < 6:
        padded.append(None)
    ingredient_id = base.coerce_int(padded[0])
    pieces = base.coerce_float(padded[1])
    cut_code = base.coerce_int(padded[2])
    whole_count = base.coerce_count(padded[3])
    conf = base.coerce_confidence_code(padded[4])
    portion_code = base.coerce_int(padded[5])
    if portion_code == 5:
        portion_code = 4
        warnings.append(f"quantity_double_portion_mapped_to_large:{calorie_row.ingredient}")
    if portion_code is None or portion_code not in {0, 1, 2, 3, 4}:
        portion_code = 3
    if scene == 1:
        return [ingredient_id, None, 5, None, 0, 3], warnings

    countable = calorie_row.calories_per_single_object is not None
    corrected_count = base.corrected_whole_count(pieces, cut_code, whole_count, conf)
    too_many = bool(
        (pieces is not None and pieces > 10)
        or (whole_count is not None and whole_count > 10)
        or (corrected_count is not None and corrected_count > 10)
    )
    if too_many:
        portion_code = max(portion_code, 4) if portion_code else 4
        warnings.append(f"quantity_count_over_10_used_portion:{calorie_row.ingredient}")
        return [ingredient_id, None, 5, None, 0, min(portion_code, 4)], warnings
    if not countable:
        if portion_code == 0:
            portion_code = 3
        warnings.append(f"quantity_non_countable_used_portion:{calorie_row.ingredient}")
        return [ingredient_id, None, 5, None, 0, min(portion_code, 4)], warnings
    if cut_code in {base.CUT_SLICE_UNCLEAR, base.CUT_CHOPPED_PILE}:
        if portion_code == 0:
            portion_code = 3
        warnings.append(f"quantity_unclear_cut_used_portion:{calorie_row.ingredient}")
        return [ingredient_id, pieces, cut_code, None, 0, min(portion_code, 4)], warnings
    if corrected_count is None:
        if portion_code == 0:
            portion_code = 3
        warnings.append(f"quantity_uncertain_count_used_portion:{calorie_row.ingredient}")
        return [ingredient_id, pieces, cut_code if cut_code is not None else 5, None, 0, min(portion_code, 4)], warnings
    return [ingredient_id, pieces, cut_code, whole_count, conf if conf is not None else 2, 0], warnings


def apply_abundance_normal_portions(calories: dict[str, Any], calories_by_id: dict[int, Any]) -> dict[str, Any]:
    composition = calories.get("composition") if isinstance(calories.get("composition"), dict) else {}
    if composition.get("scene_context") != "abundance_display":
        return calories
    ingredients = calories.get("ingredients") if isinstance(calories.get("ingredients"), list) else []
    total_kcal = 0
    for item in ingredients:
        if not isinstance(item, dict):
            continue
        ingredient_id = base.coerce_int(item.get("id"))
        row = calories_by_id.get(ingredient_id) if ingredient_id is not None else None
        if row is None:
            continue
        kcal, grams, portion_factor, calories_per_portion, portion_source = base.portion_calories(row, "normal")
        total_kcal += int(kcal)
        item.update(
            {
                "count": None,
                "count_used": False,
                "portion_category": "normal",
                "portion_factor": portion_factor,
                "calories_per_portion": calories_per_portion,
                "calorie_method": "abundance_normal_portion",
                "kcal": kcal,
                "estimated_quantity_g": round(grams, 1) if grams is not None else None,
                "per_instance_kcal": None,
                "per_instance_source": None,
                "abundance_scene": True,
                "serving_note": "Full image total is not estimated; calories are shown for a normal portion of this ingredient.",
            }
        )
        if portion_source == "fallback_100g_portion":
            item["serving_note"] = (
                "Full image total is not estimated; calories use a fallback 100 g normal portion for this ingredient."
            )
    calories["total_kcal"] = total_kcal if ingredients else None
    calories["estimation_scope"] = "average_portion_not_full_image"
    calories["notes"] = (
        "This looks like a display or too-much-for-one-person image. "
        "Full image calories are not estimated; each ingredient uses a normal single-person portion."
    )
    return calories


def zero_task1_load_timings() -> dict[str, float]:
    return {
        "load_siglip2_sec": 0.0,
        "text_embeddings_sec": 0.0,
        "load_qwen_sec": 0.0,
        "load_models_and_text_sec": 0.0,
    }


def run_task1_ingredient_step(runtime: Any, image_path: Path) -> tuple[dict[str, Any] | None, float]:
    cached_report = getattr(runtime, "task1_cached_report", None)
    if isinstance(cached_report, dict):
        return cached_report, 0.0
    task1_runtime = getattr(runtime, "task1_runtime", None)
    if task1_runtime is None:
        return None, 0.0
    import orin_task1_pipeline as task1

    task1_args = argparse.Namespace(**vars(task1_runtime.args))
    task1_args.image = image_path
    task1_args.output_json = None
    if getattr(task1_args, "skip_vlm_rel_gap_threshold", None) is None:
        task1_args.skip_vlm_rel_gap_threshold = 0.25
    started = time.perf_counter()
    report = task1.build_single_report(task1_runtime, image_path, task1_args, zero_task1_load_timings())
    return report, time.perf_counter() - started


def build_task1_then_quantity_report(runtime: Any, image_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    query_t0 = time.perf_counter()
    parse_warnings: list[str] = []
    task1_report, task1_ingredient_sec = run_task1_ingredient_step(runtime, image_path)
    if task1_report is None:
        parse_warnings.append("task1_runtime_unavailable_used_compact_calorie_fallback")
        original_prompt_builder = base.build_selected_composition_prompt
        original_extract_json_object = base.extract_json_object
        base.build_selected_composition_prompt = build_compact_edgellm_prompt
        base.extract_json_object = extract_json_object_edgellm
        try:
            return _BASE_BUILD_SINGLE_REPORT(runtime, image_path, args)
        finally:
            base.build_selected_composition_prompt = original_prompt_builder
            base.extract_json_object = original_extract_json_object

    selected_labels = [str(label).strip() for label in task1_report.get("selected_labels", []) if str(label).strip()]
    selected_rows, selected_trace, label_warnings = rows_from_task1_labels(selected_labels, runtime.calories_by_name)
    parse_warnings.extend(label_warnings)
    prompt = build_quantity_only_prompt(selected_rows) if selected_rows else ""
    started = time.perf_counter()
    if args.mock_qwen_json:
        raw_text = args.mock_qwen_json
    elif selected_rows:
        if runtime.estimator is None:
            raise RuntimeError("Qwen estimator was not initialized")
        raw_text = runtime.estimator.generate(image_path, prompt)
    else:
        raw_text = "[]"
    quantity_qwen_sec = time.perf_counter() - started
    parsed, parse_warning = extract_quantity_json_object_edgellm(raw_text)
    if parse_warning:
        parse_warnings.append(parse_warning)
    parsed, restriction_warnings = restrict_quantity_payload_to_selected(parsed, selected_rows)
    parse_warnings.extend(restriction_warnings)
    calories, normalize_warnings = base.normalize_composition_calories(
        parsed,
        runtime.calories_by_name,
        runtime.calories_by_id,
    )
    calories = apply_abundance_normal_portions(calories, runtime.calories_by_id)
    parse_warnings.extend(normalize_warnings)
    total_image_sec = time.perf_counter() - query_t0
    answer = base.build_answer(calories)
    task1_timings = task1_report.get("timings_sec") if isinstance(task1_report.get("timings_sec"), dict) else {}
    task1_qwen_sec = float(task1_timings.get("qwen_sec") or 0.0)
    return {
        "schema_version": "dishcovery_calorie_task1_then_quantity_edgellm_v1",
        "image": image_path.name,
        "image_path": str(image_path),
        "models": {
            "qwen_model": args.qwen_model,
            "vlm_model": args.qwen_model,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
        },
        "settings": {
            "calories_csv": str(args.calories_csv),
            "calorie_logic": "task1_then_quantity_vlm",
            "edgellm_calorie_logic": EDGELLM_CALORIE_LOGIC_TASK1_QUANTITY,
            "calorie_candidate_filter": "task1_selected_labels",
            "calorie_candidate_filter_used": bool(selected_rows),
            "calorie_candidate_top_k": len(selected_rows),
            "calorie_candidate_list_mode": "task1_selected_labels",
            "effective_calorie_candidate_k": len(selected_rows),
            "calorie_counting_logic": args.calorie_counting_logic,
            "qwen_min_pixels": args.qwen_min_pixels,
            "qwen_max_pixels": args.qwen_max_pixels,
            "qwen_max_new_tokens": args.qwen_max_new_tokens,
            "qwen_runtime_source": getattr(runtime.estimator, "source", "none") if runtime.estimator is not None else "none",
            "mock_qwen_json": bool(args.mock_qwen_json),
        },
        "calories": calories,
        "answer": answer,
        "qwen": {
            "prompt": prompt,
            "raw_text": raw_text,
            "parsed": parsed,
            "parse_warnings": parse_warnings,
            "ingredient_step": {
                "source": "task1_runtime",
                "selected_labels": selected_labels,
                "selected_calorie_rows": selected_trace,
                "task1_qwen_raw_text": (task1_report.get("qwen") or {}).get("raw_text")
                if isinstance(task1_report.get("qwen"), dict)
                else None,
                "task1_qwen_parse_warnings": (task1_report.get("qwen") or {}).get("parse_warnings")
                if isinstance(task1_report.get("qwen"), dict)
                else None,
                "task1_skip_vlm": task1_report.get("skip_vlm"),
                "task1_timings_sec": task1_timings,
                "task1_report": task1_report,
            },
            "candidate_filter": {
                "enabled": True,
                "used": bool(selected_rows),
                "source": "task1_selected_labels",
                "top_k": len(selected_rows),
                "candidate_list_mode": "task1_selected_labels",
                "candidate_list_policy": None,
                "effective_k": len(selected_rows),
                "matched_count": len(selected_rows),
                "candidates": selected_trace,
            },
        },
        "timings_sec": {
            **runtime.model_load_timings,
            "task1_ingredient_sec": task1_ingredient_sec,
            "task1_qwen_sec": task1_qwen_sec,
            "quantity_qwen_sec": quantity_qwen_sec,
            "qwen_sec": task1_qwen_sec + quantity_qwen_sec,
            "total_image_sec": total_image_sec,
        },
    }


def _coerce_code(value: Any, mapping: dict[str, int]) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return mapping.get(text, value)


def _normalize_object_row(row: dict[str, Any]) -> list[Any] | None:
    ingredient_id = row.get("id", row.get("ingredient_id"))
    if ingredient_id is None:
        return None
    pieces = row.get("pieces", row.get("visible_pieces"))
    cut = _coerce_code(row.get("cut", row.get("cut_code", 5)), _CUT_NAME_TO_CODE)
    whole_count = row.get("whole_count", row.get("whole_object_count", row.get("count")))
    conf = _coerce_code(row.get("conf", row.get("confidence", 0)), _CONF_NAME_TO_CODE)
    portion = _coerce_code(row.get("portion", row.get("portion_category", 3)), _PORTION_NAME_TO_CODE)
    return [ingredient_id, pieces, cut, whole_count, conf, portion]


def _quantity_scene_from_mapping(payload: dict[str, Any]) -> Any:
    for key in ("scene", "scene_context", "scene_type", "context", "food_context"):
        if key in payload:
            return payload[key]
    return 1


def _normalize_quantity_object_row(row: dict[str, Any]) -> list[Any] | None:
    ingredient_id = row.get("id", row.get("ingredient_id"))
    if ingredient_id is None:
        return None

    mode = row.get("mode", row.get("quantity_mode"))
    value = row.get("value", row.get("quantity_value"))
    if mode is None:
        for key in ("count", "whole_count", "whole_object_count", "object_count", "quantity_count"):
            if key in row:
                mode = 0
                value = row[key]
                break
    if mode is None:
        for key in ("portion", "portion_category", "portion_size", "serving_size"):
            if key in row:
                mode = 1
                value = row[key]
                break
    mode_code = coerce_quantity_mode(mode)
    if mode_code is None:
        return _normalize_object_row(row)
    return [ingredient_id, mode_code, value]


def normalize_quantity_only_payload(parsed: Any) -> Any:
    if isinstance(parsed, dict):
        if "id" in parsed or "ingredient_id" in parsed:
            row = _normalize_quantity_object_row(parsed)
            return [1, row] if row is not None else parsed
        for key in ("rows", "items", "ingredients", "compact"):
            value = parsed.get(key)
            if isinstance(value, list):
                rows = [_normalize_quantity_object_row(item) if isinstance(item, dict) else item for item in value]
                rows = [item for item in rows if item is not None]
                return [_quantity_scene_from_mapping(parsed), *rows]
        return parsed
    if isinstance(parsed, list):
        if not parsed:
            return parsed
        if looks_like_quantity_mode_row(parsed):
            return parsed
        start = 0
        scene = 1
        first_text = str(parsed[0]).strip().lower() if parsed else ""
        if first_text == "scene" and len(parsed) >= 2:
            scene = parsed[1]
            start = 2
        elif not isinstance(parsed[0], (list, dict)):
            scene = parsed[0]
            start = 1
        rows = [_normalize_quantity_object_row(item) if isinstance(item, dict) else item for item in parsed[start:]]
        rows = [item for item in rows if item is not None]
        if rows != parsed[start:]:
            return [scene, *rows]
    return parsed


def extract_quantity_json_object_edgellm(text: str) -> tuple[Any | None, str | None]:
    parsed, warning = _BASE_EXTRACT_JSON_OBJECT(text)
    normalized = normalize_quantity_only_payload(parsed)
    if normalized is not parsed and warning:
        return normalized, f"{warning};normalized_edgellm_quantity_object_rows"
    if normalized is not parsed:
        return normalized, "normalized_edgellm_quantity_object_rows"
    return parsed, warning


def _coerce_scene(value: Any) -> int:
    if isinstance(value, int) and value in {0, 1}:
        return value
    text = str(value).strip().lower()
    if any(marker in text for marker in ("abundance", "display", "market", "stall", "basket", "crate", "bulk")):
        return 1
    return 0


def normalize_edgellm_payload(parsed: Any) -> Any:
    if isinstance(parsed, dict):
        if "id" in parsed or "ingredient_id" in parsed:
            row = _normalize_object_row(parsed)
            return [0, row] if row is not None else parsed
        for key in ("rows", "items", "ingredients", "compact"):
            value = parsed.get(key)
            if isinstance(value, list):
                rows = [_normalize_object_row(item) if isinstance(item, dict) else item for item in value]
                rows = [item for item in rows if item is not None]
                return [_coerce_scene(parsed.get("scene", parsed.get("scene_context", 0))), *rows]
        return parsed
    if isinstance(parsed, list):
        if not parsed:
            return parsed
        start = 1 if (isinstance(parsed[0], (int, str)) and not isinstance(parsed[0], dict)) else 0
        scene = _coerce_scene(parsed[0]) if start else 0
        rows = [_normalize_object_row(item) if isinstance(item, dict) else item for item in parsed[start:]]
        rows = [item for item in rows if item is not None]
        return [scene, *rows] if rows != parsed[start:] or start == 0 else parsed
    return parsed


def extract_json_object_edgellm(text: str) -> tuple[Any | None, str | None]:
    parsed, warning = _BASE_EXTRACT_JSON_OBJECT(text)
    normalized = normalize_edgellm_payload(parsed)
    if normalized is not parsed and warning:
        return normalized, f"{warning};normalized_edgellm_object_rows"
    if normalized is not parsed:
        return normalized, "normalized_edgellm_object_rows"
    return parsed, warning


class EdgeLLMCompositionEstimator:
    source = "calorie_runtime_edgellm"

    def __init__(self, args: argparse.Namespace) -> None:
        self.max_new_tokens = int(args.qwen_max_new_tokens)
        self.min_pixels = int(args.qwen_min_pixels)
        self.max_pixels = int(args.qwen_max_pixels)
        runtime_mode = str(getattr(args, "edgellm_runtime_mode", ""))
        self.source = f"calorie_runtime_edgellm_{runtime_mode}" if runtime_mode else "calorie_runtime_edgellm"
        self.scorer = EdgeLLMQwenScorer.from_args(args, max_new_tokens=self.max_new_tokens)

    def generate(self, image_path: Path, prompt: str) -> str:
        return self.scorer.score(
            image_path,
            prompt,
            max_new_tokens=self.max_new_tokens,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )


def load_runtime(args: argparse.Namespace) -> Any:
    if getattr(args, "vlm_backend", "edgellm") != "edgellm":
        return _BASE_LOAD_RUNTIME(args)
    original_estimator = base.QwenCompositionEstimator
    base.QwenCompositionEstimator = EdgeLLMCompositionEstimator
    try:
        return _BASE_LOAD_RUNTIME(args)
    finally:
        base.QwenCompositionEstimator = original_estimator


def build_single_report(runtime: Any, image_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "vlm_backend", "edgellm") != "edgellm":
        report = _BASE_BUILD_SINGLE_REPORT(runtime, image_path, args)
    elif getattr(args, "edgellm_calorie_logic", EDGELLM_CALORIE_LOGIC_TASK1_QUANTITY) == EDGELLM_CALORIE_LOGIC_TASK1_QUANTITY:
        report = build_task1_then_quantity_report(runtime, image_path, args)
    else:
        original_prompt_builder = base.build_selected_composition_prompt
        original_extract_json_object = base.extract_json_object
        base.build_selected_composition_prompt = build_compact_edgellm_prompt
        base.extract_json_object = extract_json_object_edgellm
        try:
            report = _BASE_BUILD_SINGLE_REPORT(runtime, image_path, args)
        finally:
            base.build_selected_composition_prompt = original_prompt_builder
            base.extract_json_object = original_extract_json_object
    if getattr(args, "vlm_backend", "edgellm") == "edgellm":
        runtime_mode = str(getattr(args, "edgellm_runtime_mode", ""))
        edgellm_config = config_from_args(args)
        report.setdefault("models", {})["vlm_backend"] = "tensorrt_edgellm"
        report["models"]["edgellm_model_name"] = str(getattr(args, "edgellm_model_name", ""))
        report["models"]["edgellm_runtime_mode"] = runtime_mode
        report["models"]["edgellm_llm_engine_profile"] = edgellm_config.llm_engine_profile
        report["models"]["edgellm_llm_engine_dir"] = str(edgellm_config.llm_engine_dir)
        report.setdefault("settings", {})["qwen_runtime_source"] = getattr(
            runtime.estimator,
            "source",
            f"calorie_runtime_edgellm_{runtime_mode}" if runtime_mode else "calorie_runtime_edgellm",
        )
        report["settings"]["edgellm_compact_prompt"] = (
            getattr(args, "edgellm_calorie_logic", EDGELLM_CALORIE_LOGIC_TASK1_QUANTITY)
            == EDGELLM_CALORIE_LOGIC_COMPACT
        )
        report["settings"]["edgellm_calorie_logic"] = getattr(
            args,
            "edgellm_calorie_logic",
            EDGELLM_CALORIE_LOGIC_TASK1_QUANTITY,
        )
    return report


def attach_task1_candidate_filter(runtime: Any, task1_runtime: Any, top_k: int | None = None) -> None:
    base.attach_task1_candidate_filter(runtime, task1_runtime, top_k)
    runtime.task1_runtime = task1_runtime


def attach_task1_qwen_estimator(runtime: Any, task1_runtime: Any) -> None:
    base.attach_task1_qwen_estimator(runtime, task1_runtime)
    runtime.task1_runtime = task1_runtime


def attach_task1_cached_report(runtime: Any, task1_report: dict[str, Any]) -> None:
    runtime.task1_cached_report = task1_report


def main() -> None:
    original_parse_args = base.parse_args
    original_load_runtime = base.load_runtime
    original_build_single_report = base.build_single_report
    base.parse_args = parse_args
    base.load_runtime = load_runtime
    base.build_single_report = build_single_report
    try:
        _BASE_MAIN()
    finally:
        base.parse_args = original_parse_args
        base.load_runtime = original_load_runtime
        base.build_single_report = original_build_single_report


if __name__ == "__main__":
    main()
