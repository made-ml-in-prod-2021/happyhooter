import pytest
from airflow.models import DagBag


@pytest.fixture(scope="session")
def dag_bag():
    return DagBag(dag_folder="dags/", include_examples=False)


@pytest.fixture(scope="session")
def download_dag_structure():
    return {"docker-airflow-download": []}


@pytest.fixture(scope="session")
def train_dag_structure():
    return {
        "wait-for-data": ["docker-airflow-preprocess"],
        "wait-for-target": ["docker-airflow-preprocess"],
        "docker-airflow-preprocess": ["docker-airflow-split"],
        "docker-airflow-split": ["docker-airflow-train"],
        "docker-airflow-train": ["docker-airflow-validate"],
        "docker-airflow-validate": [],
    }


@pytest.fixture(scope="session")
def predict_dag_structure():
    return {
        "wait-for-data": ["docker-airflow-predict"],
        "wait-for-model": ["docker-airflow-predict"],
        "docker-airflow-predict": [],
    }


def test_import_dags(dag_bag):
    assert {} == dag_bag.import_errors


@pytest.mark.parametrize(
    ("test_dag_name", "tasks_num"),
    [
        ("download_dag", 1),
        ("train_dag", 6),
        ("predict_dag", 3),
    ],
)
def test_dags_loaded(dag_bag, test_dag_name, tasks_num):
    dag = dag_bag.dags[test_dag_name]
    assert dag is not None
    assert len(dag.tasks) == tasks_num


@pytest.fixture(
    params=[
        ("download_dag", "download_dag_structure"),
        ("train_dag", "train_dag_structure"),
        ("predict_dag", "predict_dag_structure"),
    ]
)
def arg(request):
    return {
        "dag_name": request.param[0],
        "dag_structure": request.getfixturevalue(request.param[1]),
    }


def test_dags_structure(dag_bag, arg):
    dag = dag_bag.dags[arg["dag_name"]]
    for task_id, downstream_list in arg["dag_structure"].items():
        assert dag.has_task(task_id)
        task = dag.get_task(task_id)
        assert task.downstream_task_ids == set(downstream_list)
