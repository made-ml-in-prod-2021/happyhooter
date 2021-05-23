from typing import List, Union

from pydantic import BaseModel, conlist


class HeartDiseaseModel(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=1, max_items=500)]
    features: List[str]


class DiseaseResponse(BaseModel):
    id: int
    disease: int
