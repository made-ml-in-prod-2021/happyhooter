import numpy as np
from faker import Faker


fake = Faker()
Faker.seed(42)


def generate_fake_int_matrix(num_rows: int, num_cols: int) -> np.ndarray:
    return np.array(
        [
            fake.pylist(
                nb_elements=num_rows, variable_nb_elements=False, value_types=[int]
            )
            for _ in range(num_cols)
        ]
    )
