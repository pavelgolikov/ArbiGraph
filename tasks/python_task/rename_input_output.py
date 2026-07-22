import libcst as cst
from libcst.metadata import PositionProvider
import os

def _get_list_index_from_annotation(annotation_node):
    # This is a basic heuristic to check if the annotation string contains 'List' or '[]'
    if annotation_node is None:
        return -1
    # Convert annotation to code string to easily search
    code = cst.Module(body=[]).code_for_node(annotation_node)
    if "List" in code or "[" in code:
        return True
    return False

class FunctionAnalyzer(cst.CSTVisitor):
    def __init__(self, prefer_type='list'):
        self.prefer_type = prefer_type  # 'list' or 'scalar'
        self.old_input_name = None
        self.return_tuple_index = -1
        self.found_function = False

    def visit_FunctionDef(self, node: cst.FunctionDef):
        if self.found_function:
            return False # Only process the first function
        self.found_function = True
        
        # Analyze parameters
        params = node.params.params
        candidates = []
        for p in params:
            if p.name.value == "self":
                continue
            candidates.append(p)
            
        if candidates:
            def _is_list_param(p):
                if p.annotation:
                    code = cst.Module(body=[]).code_for_node(p.annotation.annotation)
                    return "List" in code or "[" in code
                return False

            if self.prefer_type == 'list':
                # Prefer a List parameter for chaining
                target = None
                for p in candidates:
                    if _is_list_param(p):
                        target = p.name.value
                        break
                self.old_input_name = target if target else candidates[0].name.value
            else:
                # Prefer a scalar (non-List) parameter for chaining
                target = None
                for p in candidates:
                    if not _is_list_param(p):
                        target = p.name.value
                        break
                self.old_input_name = target if target else candidates[0].name.value

        # Analyze return annotation
        ret_ann = node.returns
        if ret_ann:
            code = cst.Module(body=[]).code_for_node(ret_ann.annotation)
            # Basic check for tuple returns
            if code.startswith("Tuple[") or code.startswith("tuple["):
                # We need to parse the tuple elements. LibCST parses this as a Subscript.
                if isinstance(ret_ann.annotation, cst.Subscript):
                    slice_elements = ret_ann.annotation.slice
                    # check which element is a list
                    for i, elem in enumerate(slice_elements):
                        elem_code = cst.Module(body=[]).code_for_node(elem)
                        if "List" in elem_code or "[" in elem_code:
                            self.return_tuple_index = i
                            break
                    if self.return_tuple_index == -1:
                        self.return_tuple_index = 0 # default to first if tuple but no list
                        
        return False # don't visit nested functions


class MethodRenamer(cst.CSTTransformer):
    """Renames the first method definition and any recursive self-calls to it."""
    def __init__(self, old_name, new_name):
        self.old_name = old_name
        self.new_name = new_name
        self.found_function = False

    def leave_FunctionDef(self, original_node, updated_node):
        new_params = []
        for p in updated_node.params.params:
            if p.name.value != "self":
                new_params.append(p)
        
        updated_node = updated_node.with_changes(
            params=updated_node.params.with_changes(params=new_params)
        )

        if updated_node.name.value == self.old_name:
            if not self.found_function:
                self.found_function = True
                return updated_node.with_changes(name=cst.Name(self.new_name))
            else:
                return cst.RemoveFromParent()
        return updated_node

    def leave_Attribute(self, original_node, updated_node):
        if isinstance(updated_node.value, cst.Name) and updated_node.value.value == "self":
            attr_name = updated_node.attr.value
            if attr_name == self.old_name:
                return cst.Name(self.new_name)
            else:
                return cst.Name(attr_name)
        return updated_node


class ChainingTransformer(cst.CSTTransformer):
    def __init__(self, old_input_name, new_input_name, output_name, return_tuple_index):
        self.old_input_name = old_input_name
        self.new_input_name = new_input_name
        self.output_name = output_name
        self.return_tuple_index = return_tuple_index
        self.target_function_processed = False
        self.in_target_function = False
        self.nested_depth = 0  # track nested function depth

    def visit_FunctionDef(self, node: cst.FunctionDef):
        if self.in_target_function:
            # We're entering a nested function inside the target
            self.nested_depth += 1
            return False  # Don't visit/transform inside nested functions
        if self.target_function_processed:
            return False  # Skip other top-level functions
        self.in_target_function = True
        return True

    def leave_FunctionDef(self, original_node, updated_node):
        if self.nested_depth > 0:
            self.nested_depth -= 1
            return updated_node
        self.in_target_function = False
        self.target_function_processed = True
        return updated_node

    def leave_Name(self, original_node, updated_node):
        if self.in_target_function and self.nested_depth == 0 and self.old_input_name and updated_node.value == self.old_input_name:
            return updated_node.with_changes(value=self.new_input_name)
        return updated_node

    def leave_SimpleStatementLine(self, original_node, updated_node):
        if not self.in_target_function or self.nested_depth > 0:
            return updated_node
            
        new_body = []
        replaced = False
        
        for stmt in updated_node.body:
            if isinstance(stmt, cst.Return) and stmt.value is not None:
                replaced = True
                if self.return_tuple_index != -1:
                    # _res = <expr>
                    res_assign = cst.Assign(
                        targets=[cst.AssignTarget(target=cst.Name("_res"))],
                        value=stmt.value
                    )
                    # out_name = _res[index]
                    out_assign = cst.Assign(
                        targets=[cst.AssignTarget(target=cst.Name(self.output_name))],
                        value=cst.Subscript(
                            value=cst.Name("_res"),
                            slice=[cst.SubscriptElement(slice=cst.Index(value=cst.Integer(str(self.return_tuple_index))))]
                        )
                    )
                    new_ret = cst.Return(value=cst.Name(self.output_name))
                    return cst.FlattenSentinel([
                        cst.SimpleStatementLine(body=[res_assign]),
                        cst.SimpleStatementLine(body=[out_assign]),
                        cst.SimpleStatementLine(body=[new_ret])
                    ])
                else:
                    # out_name = <expr>
                    assign = cst.Assign(
                        targets=[cst.AssignTarget(target=cst.Name(self.output_name))],
                        value=stmt.value
                    )
                    new_ret = cst.Return(value=cst.Name(self.output_name))
                    return cst.FlattenSentinel([
                        cst.SimpleStatementLine(body=[assign]),
                        cst.SimpleStatementLine(body=[new_ret])
                    ])
            else:
                new_body.append(stmt)
                
        if replaced:
            return updated_node.with_changes(body=new_body)
        return updated_node

def rename_and_chain(source_file: str, dest_file: str, new_input_name: str, new_output_name: str, prefer_type='list', new_method_name: str = None):
    """
    Reads Python source from source_file, detects the best input variable to chain
    based on prefer_type ('list' prefers List params, 'scalar' prefers non-List params),
    renames it to new_input_name.
    It also rewrites all return statements to extract the result to new_output_name before returning.
    Optionally renames the top-level method to new_method_name (including recursive self-calls).
    Writes the transformed code to dest_file.
    """
    with open(source_file, 'r', encoding='utf-8') as f:
        source_code = f.read()

    tree = cst.parse_module(source_code)

    # 1. Analyze to find the old input name and tuple return info
    analyzer = FunctionAnalyzer(prefer_type=prefer_type)
    tree.visit(analyzer)
    
    old_input_name = analyzer.old_input_name
    tuple_index = analyzer.return_tuple_index
    
    # 2. Rename method if requested
    if new_method_name:
        # Find the original method name from the first FunctionDef
        old_method_name = None
        for node in tree.body:
            if isinstance(node, cst.ClassDef):
                for item in node.body.body:
                    if isinstance(item, cst.FunctionDef):
                        old_method_name = item.name.value
                        break
            elif isinstance(node, cst.FunctionDef):
                old_method_name = node.name.value
                break
            if old_method_name:
                break
        if old_method_name:
            renamer = MethodRenamer(old_method_name, new_method_name)
            tree = tree.visit(renamer)
    
    # 3. Transform the AST (input/output renaming)
    transformer = ChainingTransformer(old_input_name, new_input_name, new_output_name, tuple_index)
    modified_tree = tree.visit(transformer)

    # 4. Write to destination
    with open(dest_file, 'w', encoding='utf-8') as f:
        f.write(modified_tree.code)
        
    return modified_tree.code


if __name__ == "__main__":
    # Small test
    test_code = """
from typing import List, Tuple

class Solution:
    def findMin(self, nums: List[int]) -> int:
        left, right = (0, len(nums) - 1)
        while left < right:
            mid = left + right >> 1
            if nums[mid] > nums[right]:
                left = mid + 1
            elif nums[mid] < nums[right]:
                right = mid
            else:
                right -= 1
        return nums[left]
        
    def somethingElse(self, a: int, b: List[int]) -> Tuple[int, List[int]]:
        if not b:
            return 0, []
        return a, b
"""
    test_file = "test_input.py"
    out_file1 = "test_output1.py"
    
    with open(test_file, "w") as f:
        f.write(test_code)
        
    print("Testing findMin (Single Return, List Input)...")
    res1 = rename_and_chain(test_file, out_file1, "task_3_input", "task_3_output")
    print(res1)
    
    # Clean up test files
    os.remove(test_file)
    os.remove(out_file1)
