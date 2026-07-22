import os
import sys
import random
import ast

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import tasks from repo root
from tasks.gsm_task.gsm_symbolic_task import (
    load_templates,
    get_default_templates_dir,
    _parse_template,
    _parse_init_line,
    _sample_from_domain,
    _eval_conditions,
    _render_question,
    _eval_answer
)


def count_ops(expr):
    expr = expr.strip()
    if expr.startswith('$'):
        expr = expr[1:]
    if expr.startswith('{') and expr.endswith('}'):
        expr = expr[1:-1]
        
    if not expr:
        return 0
        
    try:
        tree = ast.parse(expr)
    except Exception:
        # Fallback: simple character counting if syntax error
        cleaned = expr.replace('//', 'd')
        ops = sum(cleaned.count(char) for char in ['+', '-', '*', '/'])
        return ops
        
    ops = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            ops += 1
        elif isinstance(node, ast.Call):
            # If calling Fraction with 2 arguments, count as a division (BinOp)
            if isinstance(node.func, ast.Name) and node.func.id == 'Fraction' and len(node.args) == 2:
                ops += 1
    return ops


TARGET_TEMPLATES = [
    "0000.json", "0003.json", "0004.json", "0007.json", "0010.json", "0015.json",
    "0016.json", "0018.json", "0019.json", "0024.json", "0027.json", "0028.json",
    "0032.json", "0034.json", "0036.json", "0039.json", "0042.json", "0047.json",
    "0048.json", "0050.json", "0051.json", "0054.json", "0058.json", "0059.json",
    "0060.json", "0062.json", "0066.json", "0067.json", "0071.json", "0072.json",
    "0075.json", "0078.json", "0082.json", "0084.json", "0086.json",
    "0088.json", "0093.json", "0094.json", "0095.json",
    "0096.json", "0099.json"
]

def generate_examples(output_file, arg_seed):
    templates_dir = get_default_templates_dir()
    templates = load_templates(templates_dir)
    # Only keep the targeted 44 hard templates
    templates = [t for t in templates if t["_filename"] in TARGET_TEMPLATES]
    
    rng = random.Random(arg_seed)  # Fixed seed for reproducible examples
    
    good_templates = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for tpl in templates:
            fname = tpl["_filename"]
            orig_q = tpl.get("question", "N/A")
            qa = tpl.get("question_annotated", "N/A")
            
            # Parse template
            qt, init_lines, cond_lines, answer_expr = _parse_template(qa)
            
            specs = []
            for line in init_lines:
                s = _parse_init_line(line)
                if s:
                    specs.append(s)
            
            # Generate valid inputs
            valid_vars = None
            try:
                # Keep trying until valid parameters are found
                while True:
                    var_values = {}
                    for spec in specs:
                        names = spec["names"]
                        sampled = _sample_from_domain(spec, rng)
                        
                        if len(names) > 1 and isinstance(sampled, list):
                            # Multi-assignment: distribute list items to variable names
                            for i, name in enumerate(names):
                                if i < len(sampled):
                                    var_values[name] = sampled[i]
                        else:
                            # Single assignment (or single name)
                            for name in names:
                                var_values[name] = sampled
                    
                    if _eval_conditions(cond_lines, var_values):
                        valid_vars = var_values
                        break
            except RuntimeError as e:
                print(f"  CRITICAL ERROR evaluating {fname}: {e}")
                continue
            
            if valid_vars is not None:
                # Render the question
                rendered = _render_question(qt, valid_vars)
                
                # Compute the answer
                answer = _eval_answer(answer_expr, valid_vars)
                
                f.write(f"=== Template: {fname} ===\n")
                f.write(f"--- Example Problem ---\n{rendered}\n")
                f.write(f"--- Answer ---\n{answer}\n")
                f.write("\n" + "="*80 + "\n\n")
                good_templates += 1
    
    print(f"Total templates processed: {len(templates)}")
    print(f"Good templates written: {good_templates}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
    out_path = os.path.join(out_dir, "gsm_symbolic_prompts.txt")
    generate_examples(out_path, 42)
    print(f"Done. Generated examples in {out_path}")
