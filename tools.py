import math
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from database import save_memory, search_memory
from rag import retrieve_from_rag
import ast
import operator

load_dotenv()


CURRENT_THREAD_ID = "default"


def set_current_thread_id(thread_id: str):
    global CURRENT_THREAD_ID
    CURRENT_THREAD_ID = thread_id


web_search = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
 
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum}
_MATH_FUNCS = {
    name: getattr(math, name)
    for name in dir(math)
    if not name.startswith("_") and callable(getattr(math, name))
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
 
MAX_EXPONENT = 1000  # stops things like 9**9**9 from hanging the server
 
 
def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
 
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
 
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(item) for item in node.elts]
 
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError("Exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
 
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
 
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
 
    if isinstance(node, ast.Attribute):  # math.pi, math.e
        if isinstance(node.value, ast.Name) and node.value.id == "math":
            if node.attr in ("pi", "e", "tau"):
                return getattr(math, node.attr)
 
    if isinstance(node, ast.Call) and not node.keywords:
        func = None
        if isinstance(node.func, ast.Name):
            func = _FUNCS.get(node.func.id) or _MATH_FUNCS.get(node.func.id)
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "math"
        ):
            func = _MATH_FUNCS.get(node.func.attr)
        if func is None:
            raise ValueError("Function not allowed")
        return func(*[_eval(arg) for arg in node.args])
 
    raise ValueError("Unsupported expression")
 
 
@tool
def calculator(expression: str) -> str:
    """
    Useful for math calculations.
    Input should be a plain math expression using + - * / // % ** and parentheses.
    Use ** for powers (not ^). Supports math.sqrt(), math.sin(), pi, abs(), round(), min(), max(), sum().
    Examples: 2 + 2, math.sqrt(16), 10 * 5, 2 ** 10
    """
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return str(_eval(tree))
    except ZeroDivisionError:
        return "Calculation error: division by zero"
    except Exception as e:
        return f"Calculation error: {e}"

@tool
def search_uploaded_documents(query: str) -> str:
    """
    Search uploaded documents for relevant information.
    Use this when the user asks about uploaded PDFs, DOCX, TXT, notes, files, or documents.
    """

    return retrieve_from_rag(
        query=query,
        thread_id=CURRENT_THREAD_ID
    )




@tool
def remember_this(memory: str) -> str:
    """
    Save an important user preference or fact into long-term memory.
    Use this when the user asks you to remember something.
    """

    return save_memory(
        thread_id=CURRENT_THREAD_ID,
        memory=memory
    )



@tool
def recall_memory(query: str) -> str:
    """
    Recall saved long-term memories about the user or this conversation.
    """

    return search_memory(
        thread_id=CURRENT_THREAD_ID,
        query=query
    )





tools = [
    calculator,
    search_uploaded_documents,
    remember_this,
    recall_memory,
    web_search
]