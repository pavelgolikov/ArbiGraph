import math
import sympy

# =============================================================================
# Helper Utilities Prompt
# =============================================================================
def list_to_matrix_prompt(inp_list_name, matrix_name: str, num_rows: int, num_cols: int) -> str:
    return f"Reshape list {inp_list_name} into matrix {matrix_name} of size {num_rows}x{num_cols}\
 using row-major order, filling the top row first, then the second row, and so on.\n\
To achieve the required dimension for the reshape, you can append zeros to the end of list {inp_list_name} or truncate it.\n"

def list_to_poly_prompt(inp_list_name, poly_name: str) -> str:
    return f"Let {poly_name}(x) be a polynomial whose coefficients are the elements of the list {inp_list_name} in order, starting from the highest degree.\n"

def matrix_to_list_prompt(matrix_name, out_var_name: str) -> str:
    return f"Convert matrix {matrix_name} into a list {out_var_name} by concatenating matrix rows starting from the top row, then second row, etc...\n"

def poly_to_list_prompt(poly_name, out_var_name: str):
    return f"Convert {poly_name} into a list {out_var_name} by extracting its coefficients starting from the highest degree to the constant term, including zero coefficients for missing intermediate powers. Omit leading zero coefficients. Represent the zero polynomial as [0].\n"

def adapt_element(mod_value=0, to_int=False, needs_abs=False, add_val=None, real_part=False) -> str:
    parts = []
    if real_part:
        parts.append("take its real part")
    if to_int:
        parts.append("if it is a float, convert it to an integer by rounding ties away from zero (0.5 to 1, -0.5 to -1)")
    if mod_value != 0:
        if needs_abs:
            parts.append(f"replace it with the remainder of its absolute value modulo {mod_value}")
        else:
            parts.append(f"replace it with the remainder of its absolute value modulo {mod_value} while preserving its original sign")
    if needs_abs and mod_value == 0:
        parts.append("take its absolute value")
    if add_val is not None and str(add_val) != "0":
        parts.append(f"add {add_val} to it")

    if len(parts) == 0:
        return "leave it unchanged"
    return "; then ".join(parts)

def adapt_scalar_prompt(var_name_raw, var_name_adapted, mod_value, to_int=False, needs_abs=False, add_val=None, real_part=False):
    ops = adapt_element(mod_value, to_int, needs_abs, add_val, real_part)
    return f"Let {var_name_adapted} be the result of preprocessing {var_name_raw} as follows - {ops}.\n"

def adapt_list_prompt(var_name_raw, var_name_adapted, mod_value, list_len_max, add_val=None, power_2=False, to_int=False, needs_abs=False, real_part=False):
    base_prompt = ""
    ops = adapt_element(mod_value, to_int, needs_abs, add_val, real_part)
    base_prompt = f"Let {var_name_adapted} be the result of preprocessing {var_name_raw} as follows - for every element, {ops}.\n"
    base_prompt += f"If {var_name_adapted} has more than {list_len_max} elements, keep only the first {list_len_max} elements.\n"
    if power_2:
        base_prompt += f"If the length of {var_name_adapted} is not a power of 2, update {var_name_adapted} by padding it with zeros to the next power of 2.\n"
    return base_prompt

# =============================================================================
# Helper Utilities Compute
# =============================================================================
def list_to_matrix_compute(lst, num_rows, num_cols):
    total = num_rows * num_cols
    if len(lst) < total:
        lst = list(lst) + [0] * (total - len(lst))
    elif len(lst) > total:
        lst = list(lst)[:total]
    # create a matrix from the list in row-major order, filling the top row first, then the second row, and so on
    matrix = []
    for i in range(num_rows):
        matrix.append(lst[i * num_cols : (i + 1) * num_cols])
    return sympy.Matrix(matrix)

# NOTE: poly_to_list_compute and matrix_to_list_compute are omitted because they can just be obtained by calling
# list() on the object to get the corresponding list representation.
# list_to_poly_compute was also removed because it requires sympy.Symbol that might also be required in calling function

def adapt_scalar_compute(value, mod_value=0, to_int=False, needs_abs=False, add_val=None, real_part=False):
    if real_part:
        value = sympy.re(value)
    if to_int:
        value = round_away_from_zero(value)
    if mod_value != 0:
        if value < 0:
            value = -((-value) % mod_value)
        else:
            value = value % mod_value
    if needs_abs:
        value = abs(value)
    if add_val is not None:
        value += add_val
    return value

def adapt_list_compute(lst, mod_value=0, list_len_max=0, add_val=None, power_2=False, to_int=False, needs_abs=False, real_part=False):
    out = [adapt_scalar_compute(x, mod_value, to_int, needs_abs, add_val, real_part) for x in lst]
    if list_len_max > 0 and len(out) > list_len_max:
        out = out[:list_len_max]
    if power_2:
        out = pad_to_power_of_2_compute(out)
    return out

def round_away_from_zero(x):
    if x >= 0:
        return math.floor(x + 0.5)
    else:
        return math.ceil(x - 0.5)

def pad_to_power_of_2_compute(seq):
    if len(seq) == 0:
        raise ValueError("Cannot pad empty sequence to power of 2.")
    n = len(seq)
    if (n & (n - 1)) == 0:
        return seq
    next_pow2 = 2 ** math.ceil(math.log2(n))
    return list(seq) + [0] * (next_pow2 - n)

def round_float_to_int_compute(val):
    if isinstance(val, list):
        return [round_away_from_zero(x) for x in val]
    else:
        return round_away_from_zero(val)
