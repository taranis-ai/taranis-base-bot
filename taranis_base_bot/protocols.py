from typing import Protocol, Any, Mapping, Sequence, runtime_checkable

JSONObj = Mapping[str, Any]
JSONList = Sequence[JSONObj]

@runtime_checkable
class Predictor(Protocol):
    """Interface for Predictor objects used in bots
       Need to implement predict method
     """
    def predict(self, **kwargs: Any) -> JSONObj | JSONList: ...
