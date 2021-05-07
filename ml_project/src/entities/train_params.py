from dataclasses import dataclass, field


@dataclass()
class LogRegParams:
    model_type: str = field(default="LogisticRegression")
    penalty: str = field(default="l2")
    tol: float = field(default=1e-4)
    C: float = field(default=1.0)
    random_state: int = field(default=42)


@dataclass()
class RFParams:
    model_type: str = field(default="RandomForestClassifier")
    n_estimators: int = field(default=100)
    criterion: str = field(default="gini")
    random_state: int = field(default=42)
