import ast
import operator
import asyncio
from typing import Dict, Any

class CalculatorTool:
    _OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: lambda x: x
    }

    async def calculate(self, expression: str) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._calculate_sync, expression)

    def _calculate_sync(self, expression: str) -> Dict[str, Any]:
        cleaned = expression.replace("₹", "").replace("Rs.", "").replace("INR", "").replace(",", "").strip()
        try:
            tree = ast.parse(cleaned, mode='eval')
            result = self._evaluate_node(tree.body)
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return {"success": True, "result": result, "expression": cleaned}
        except Exception as e:
            return {"success": False, "error": f"Invalid expression: {str(e)}", "expression": cleaned}

    def _evaluate_node(self, node):
        if isinstance(node, (ast.Num, ast.Constant)):
            return node.n if isinstance(node, ast.Num) else node.value
        elif isinstance(node, ast.BinOp):
            left = self._evaluate_node(node.left)
            right = self._evaluate_node(node.right)
            return self._OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._evaluate_node(node.operand)
            return self._OPERATORS[type(node.op)](operand)
        else:
            raise TypeError(f"Unsupported: {type(node).__name__}")