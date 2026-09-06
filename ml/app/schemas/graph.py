"""Mirrors docs/schema/graph.schema.json."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    type: Literal["input", "candidate"]
    label: str = Field(..., min_length=1)
    prompt: Optional[str] = None
    thumbnail_url: Optional[str] = None
    style_cluster: Optional[str] = None


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., min_length=1)
    target: str = Field(..., min_length=1)
    weight: float = Field(..., ge=0, le=1)


class Graph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[GraphNode]
    edges: list[GraphEdge]
