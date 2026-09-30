from src.models.factory import get_models

for model in get_models():

    print(model.name())

    print(model.estimator())

    print("-" * 50)