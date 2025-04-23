from pydantic import BaseModel


class Description(BaseModel):
    text: str


class ScoredReferenceProducts(BaseModel):
    data: str


class ScoredImpactFactors(BaseModel):
    data: str
