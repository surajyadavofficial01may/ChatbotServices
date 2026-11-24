"""The knowledge-entry schema: one row of `portfolio_knowledge_base`.

One model drives the Supabase table (`supabase/schema.sql` mirrors it), the Qdrant payload,
and the `/dataentry` grid, which builds its tabs (one per category) and columns from this
model's JSON Schema. The `ui` hints in `json_schema_extra` pick each column's widget.
"""

from datetime import datetime
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

# The seven knowledge-base tabs. Must match the CHECK constraint in supabase/schema.sql.
Category = Literal[
    "Personal_Details",
    "Work_Experience",
    "Technical_Skills",
    "Projects",
    "Articles",
    "Social_Links",
    "FAQs",
]
CATEGORIES: tuple[str, ...] = get_args(Category)
# New rows get the next free id in their tab: PRJ001, PRJ002, ...
ID_PREFIXES = {
    "Personal_Details": "PER",
    "Work_Experience": "EXP",
    "Technical_Skills": "SK",
    "Projects": "PRJ",
    "Articles": "ART",
    "Social_Links": "SOC",
    "FAQs": "FAQ",
}

ENTRY_ID_PATTERN = r"^[A-Za-z0-9._:-]{1,64}$"


class KnowledgeEntryIn(BaseModel):
    """One knowledge-base row as an admin writes it (grid edit, bulk import or API).

    `active` also accepts the strings "TRUE"/"FALSE" used by the per-tab views and
    spreadsheets, so rows copied from either import unchanged.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    id: str = Field(
        pattern=ENTRY_ID_PATTERN,
        title="ID",
        description="Stable identifier, e.g. PRJ001. Saving an existing ID updates that row.",
        json_schema_extra={"ui": "id"},
    )
    category: Category = Field(
        description="Which tab the row belongs to.",
        json_schema_extra={"ui": "tabs", "id_prefixes": ID_PREFIXES},
    )
    title: str = Field(default="", max_length=200, description="Short label, shown as the source citation.")
    content: str = Field(
        min_length=1,
        max_length=20_000,
        description="The fact itself, in complete sentences. This is what the assistant reads.",
        json_schema_extra={"ui": "textarea"},
    )
    url: str = Field(
        default="",
        max_length=2048,
        pattern=r"^(https?://\S+)?$",
        title="URL",
        description="Optional link shown with the citation.",
        json_schema_extra={"ui": "url"},
    )
    active: bool = Field(default=True, description="Inactive rows stay in Supabase but leave search on the next sync.")
    sort_order: int = Field(
        default=0,
        ge=0,
        le=1_000_000,
        title="Order",
        description="Position within the tab.",
        json_schema_extra={"ui": "order"},
    )

    @field_validator("url", mode="before")
    @classmethod
    def _null_url_is_blank(cls, value: Any) -> Any:
        """Treat a missing URL (null) like the table default, an empty string."""
        return value or ""


class KnowledgeEntry(KnowledgeEntryIn):
    """A stored row: the admin-written fields plus the server-managed timestamp."""

    updated_at: datetime | None = None
