from pydantic import BaseModel, ConfigDict, Field


class CoffeeInput(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    species: str | None = Field(default=None, alias="Species")
    country_of_origin: str | None = Field(default=None, alias="Country.of.Origin")
    variety: str | None = Field(default=None, alias="Variety")
    processing_method: str | None = Field(default=None, alias="Processing.Method")
    color: str | None = Field(default=None, alias="Color")
    number_of_bags: float | None = Field(default=None, alias="Number.of.Bags")
    bag_weight: str | None = Field(default=None, alias="Bag.Weight")
    harvest_year: str | None = Field(default=None, alias="Harvest.Year")
    moisture: float | None = Field(default=None, alias="Moisture")
    category_one_defects: float | None = Field(default=None, alias="Category.One.Defects")
    quakers: float | None = Field(default=None, alias="Quakers")
    category_two_defects: float | None = Field(default=None, alias="Category.Two.Defects")
    altitude_mean_meters: float | None = Field(default=None, alias="altitude_mean_meters")


class PredictionResponse(BaseModel):
    prediction: float
    target: str
    model_version: str


class BatchPredictionRequest(BaseModel):
    records: list[CoffeeInput] = Field(min_length=1, max_length=128)


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    target: str
    model_version: str
