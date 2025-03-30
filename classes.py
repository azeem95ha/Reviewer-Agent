from typing import Annotated, Optional,Literal,List, TypedDict
from pydantic import BaseModel, Field
from langchain.tools import tool


class SubmittalPage(BaseModel):
  source: str
  page_num: int
  content: List[str]
class ItemVariant(BaseModel):
    variant_num: str = Field(description="Item variant number")
    description: str = Field(description="Item variant description")
    unit: str = Field(description="Item unit")
    quantity: float = Field(description="Item quantity", default=0.0)
    rate: float = Field(description="Item rate")
    amount: float = Field(description="Item amount", default=0.0)

    def __init__(self, **data):
        # Custom validation for numeric fields: Ensure they are numbers, and default to 0.0 if not
        for field in ['quantity', 'rate', 'amount']:
            try:
                data[field] = float(data.get(field, 0.0))  # Convert to float, default to 0.0
            except (ValueError, TypeError):
                data[field] = 0.0
        super().__init__(**data)


class BOQItem(BaseModel):
    itemNum: str = Field(description="Item number", default="")
    description: List[str] = Field(description="Item description", default=[])
    unit: str = Field(description="Item unit")
    quantity: float = Field(description="Item quantity", default=0.0)
    rate: float = Field(description="Item rate")
    amount: float = Field(description="Item amount", default=0.0)
    variants: List[ItemVariant] = Field(description="Item variants", default=[])

    def __init__(self, **data):
        # Custom validation for numeric fields: Ensure they are numbers, and default to 0.0 if not
        for field in ['quantity', 'rate', 'amount']:
            try:
                data[field] = float(data.get(field, 0.0))
            except (ValueError, TypeError):
                data[field] = 0.0
        super().__init__(**data)


# --- Reducer Function for Error Handling ---
def keep_first_error(current_error: Optional[str], new_error: Optional[str]) -> Optional[str]:
    """Keeps the first non-None error message encountered."""
    # print(f"Reducer: Current={current_error}, New={new_error}") # Debug print
    return current_error if current_error is not None else new_error
# --- LangGraph State Definition ---
class State(TypedDict):
    file_name: str
    boq_collection: str
    specs_collection: str # Derived from boq_collection
    submittal_text: str
    retrieved_docs: list[list[str]]
    final_report: str
    prod_name: str
    tavily_summary: str
    error_message: Annotated[Optional[str], keep_first_error] # To capture errors during graph execution


# --- Langchain Pydantic Models and Tools ---
class ChooseBOQ(BaseModel):
  boq_name: Literal["boq", "boq_Electrical", "boq_HVAC", "boq_Plumbing"] = Field(description="name of the boq to review from the provided choices")

@tool
class Title(BaseModel):
  """  Extracts the product name from the submittal text"""
  
  prod_name: str = Field(description="material or product name")