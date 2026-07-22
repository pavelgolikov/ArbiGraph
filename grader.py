import ast
import math
import re
import math_verify

BOXED_TASK_OUTPUT_RE = re.compile(
    r"^\s*(?:\\text\{\s*)?task_[1-9]\d*_out(?:\s*\})?"
    r"\s*(?:=|:|\bis\b)\s*(.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def strip_boxed_task_output_label(content: str) -> str:
    """Return only the value from ``task_N_out = value`` boxed content."""
    match = BOXED_TASK_OUTPUT_RE.fullmatch(content)
    return match.group(1).strip() if match else content.strip()


def extract_boxed(text: str) -> str:
    """Extracts content from \boxed{...} allowing for nested braces."""
    matches = [m.start() for m in re.finditer(r'\\boxed\{', text)]
    if not matches:
        return ""
    
    # Try from the last match backwards
    for start_pos in reversed(matches):
        start_idx = start_pos + len(r'\boxed{')
        brace_count = 1
        end_idx = start_idx
        
        while end_idx < len(text) and brace_count > 0:
            if text[end_idx] == '{':
                brace_count += 1
            elif text[end_idx] == '}':
                brace_count -= 1
            end_idx += 1
            
        if brace_count == 0:
            return text[start_idx:end_idx-1]
            
    return ""


def _matches_integer_exactly(prediction: object, ground_truth: int) -> bool:
    """Accept integer-valued numeric literals without rounding or truncation."""
    if isinstance(prediction, bool):
        return False
    if isinstance(prediction, int):
        return prediction == ground_truth
    if isinstance(prediction, float):
        return (
            math.isfinite(prediction)
            and prediction.is_integer()
            and int(prediction) == ground_truth
        )
    return False


def _integer_components_are_exact( prediction: object, ground_truth: object,) -> bool | None:
    """Validate parseable literals wherever the ground truth requires integers.

    ``None`` means there are no integer components to validate at this level.
    This check runs before ``math_verify`` so decimal literals such as ``42.9``
    cannot be accepted for an integer ground truth through symbolic grading.
    """
    if isinstance(ground_truth, int) and not isinstance(ground_truth, bool):
        return _matches_integer_exactly(prediction, ground_truth)

    if isinstance(ground_truth, list):
        if not isinstance(prediction, list) or len(prediction) != len(ground_truth):
            return False

        checked_integer = False
        for pred_item, truth_item in zip(prediction, ground_truth):
            result = _integer_components_are_exact(pred_item, truth_item)
            if result is False:
                return False
            if result is True:
                checked_integer = True
        return True if checked_integer else None

    return None


def grade(prediction_text: str, ground_truth: object) -> bool:
    """
    Grades the prediction against the ground truth.
    Prioritizes math_verify as the main grading method.
    Falls back to regex extraction and ast.literal_eval for Python lists/scalars.
    Integer targets accept integer-valued floats but reject fractional values.
    Uses math.isclose for floats with abs_tol=1e-3 in the fallback.
    """
    extracted = extract_boxed(prediction_text)
    normalized_extracted = strip_boxed_task_output_label(extracted) if extracted else ""
    normalized_prediction = (
        f"\\boxed{{{normalized_extracted}}}"
        if normalized_extracted and normalized_extracted != extracted.strip()
        else prediction_text
    )

    if normalized_extracted:
        try:
            literal_prediction = ast.literal_eval(normalized_extracted)
        except Exception:
            literal_prediction = None
        else:
            integer_check = _integer_components_are_exact(literal_prediction, ground_truth)
            if integer_check is False:
                return False

    # Main method: math_verify
    try:
        parsed_preds = math_verify.parse(normalized_prediction)
        if parsed_preds:
            if math_verify.verify(ground_truth, parsed_preds):
                return True
    except Exception:
        pass

    # Fallback method: regex extraction and manual checking
    if not normalized_extracted:
        return False
        
    extracted = normalized_extracted

    # If ground truth is a list, parse the extracted text as a Python list
    if isinstance(ground_truth, list):
        try:
            # Safely evaluate the string as a list
            pred_list = ast.literal_eval(extracted)
            if not isinstance(pred_list, list):
                return False
            if len(pred_list) != len(ground_truth):
                return False
            for p, g in zip(pred_list, ground_truth):
                if isinstance(g, int) and not isinstance(g, bool):
                    if not _matches_integer_exactly(p, g):
                        return False
                    continue
                elif isinstance(g, float):
                    try:
                        p = float(p)
                    except (ValueError, TypeError):
                        pass
                        
                if isinstance(g, float) or isinstance(p, float):
                    try:
                        if not math.isclose(float(p), float(g), abs_tol=1e-3):
                            return False
                    except (ValueError, TypeError):
                        return False
                else:
                    if p != g:
                        return False
            return True
        except Exception:
            return False
            
    # If ground truth is a scalar
    else:
        try:
            pred_val = ast.literal_eval(extracted)
            if isinstance(ground_truth, int) and not isinstance(ground_truth, bool):
                return _matches_integer_exactly(pred_val, ground_truth)
            elif isinstance(ground_truth, float):
                try:
                    pred_val = float(pred_val)
                except (ValueError, TypeError):
                    pass
                    
            if isinstance(ground_truth, float) or isinstance(pred_val, float):
                try:
                    return math.isclose(float(pred_val), float(ground_truth), abs_tol=1e-3)
                except (ValueError, TypeError):
                    return False
            else:
                return pred_val == ground_truth
        except Exception:
            return False
