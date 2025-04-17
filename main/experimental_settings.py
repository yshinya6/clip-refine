SETTINGS = {
    "experiments": {
        # "zeroshot_classification_aircraft": {
        #     "test_function": "metrics/zeroshot_classification.py",
        #     "function_name": "evaluate",
        #     "dataset_cls": {"dataset_func": "data/generic.py", "dataset_name": "Aircraft", "args": {"test": True}},
        #     "test_args": {"classname_file": "data/classnames/fgvc_aircraft.txt"},
        # },
        # "zeroshot_classification_bird": {
        #     "test_function": "metrics/zeroshot_classification.py",
        #     "function_name": "evaluate",
        #     "dataset_cls": {"dataset_func": "data/generic.py", "dataset_name": "Bird", "args": {"test": True}},
        #     "test_args": {"classname_file": "data/classnames/cub.txt"},
        # },
        # "zeroshot_classification_car": {
        #     "test_function": "metrics/zeroshot_classification.py",
        #     "function_name": "evaluate",
        #     "dataset_cls": {"dataset_func": "data/generic.py", "dataset_name": "Car", "args": {"test": True}},
        #     "test_args": {"classname_file": "data/classnames/stanford_cars.txt"},
        # },
        # "zeroshot_classification_caltech101": {
        #     "test_function": "metrics/zeroshot_classification.py",
        #     "function_name": "evaluate",
        #     "dataset_cls": {"dataset_func": "data/generic.py", "dataset_name": "Caltech101", "args": {"test": True}},
        #     "test_args": {"classname_file": "data/classnames/caltech101.txt"},
        # },
        # "zeroshot_classification_dtd": {
        #     "test_function": "metrics/zeroshot_classification.py",
        #     "function_name": "evaluate",
        #     "dataset_cls": {"dataset_func": "data/generic.py", "dataset_name": "DTD", "args": {"test": True}},
        #     "test_args": {"classname_file": "data/classnames/dtd.txt"},
        # },
        # "zeroshot_classification_eurosat": {
        #     "test_function": "metrics/zeroshot_classification.py",
        #     "function_name": "evaluate",
        #     "dataset_cls": {"dataset_func": "data/generic.py", "dataset_name": "EuroSAT", "args": {"test": True}},
        #     "test_args": {"classname_file": "data/classnames/eurosat.txt"},
        # },
        # "zeroshot_classification_flower": {
        #     "test_function": "metrics/zeroshot_classification.py",
        #     "function_name": "evaluate",
        #     "dataset_cls": {"dataset_func": "data/generic.py", "dataset_name": "Flower", "args": {"test": True}},
        #     "test_args": {"classname_file": "data/classnames/oxford_flowers.txt"},
        # },
        # "zeroshot_classification_food": {
        #     "test_function": "metrics/zeroshot_classification.py",
        #     "function_name": "evaluate",
        #     "dataset_cls": {"dataset_func": "data/generic.py", "dataset_name": "Food", "args": {"test": True}},
        #     "test_args": {"classname_file": "data/classnames/food101.txt"},
        # },
        "zeroshot_classification_imagenet": {
            "test_function": "metrics/zeroshot_classification.py",
            "function_name": "evaluate",
            "dataset_cls": {"dataset_func": "data/generic.py", "dataset_name": "ImageNet", "args": {"test": True}},
            "test_args": {"classname_file": "data/classnames/imagenet.txt"},
        },
        # "zeroshot_classification_pet": {
        #     "test_function": "metrics/zeroshot_classification.py",
        #     "function_name": "evaluate",
        #     "dataset_cls": {"dataset_func": "data/generic.py", "dataset_name": "Pet", "args": {"test": True}},
        #     "test_args": {"classname_file": "data/classnames/oxford_pets.txt"},
        # },
        # "zeroshot_classification_sun": {
        #     "test_function": "metrics/zeroshot_classification.py",
        #     "function_name": "evaluate",
        #     "dataset_cls": {"dataset_func": "data/generic.py", "dataset_name": "SUN397", "args": {"test": True}},
        #     "test_args": {"classname_file": "data/classnames/sun397.txt"},
        # },
        # "zeroshot_classification_ucf": {
        #     "test_function": "metrics/zeroshot_classification.py",
        #     "function_name": "evaluate",
        #     "dataset_cls": {"dataset_func": "data/generic.py", "dataset_name": "UCF101", "args": {"test": True}},
        #     "test_args": {"classname_file": "data/classnames/ucf101.txt"},
        # },
        # "zeroshot_retrieval_flicker8k": {
        #     "test_function": "metrics/zeroshot_retrieval.py",
        #     "function_name": "evaluate",
        #     "dataset_cls": {
        #         "dataset_func": "data/flickr.py",
        #         "dataset_name": "Flickr8K",
        #         "args": {"tokenizer": None, "transform": None, "test": True},
        #     },
        # },
        # "zeroshot_retrieval_flicker30k": {
        #     "test_function": "metrics/zeroshot_retrieval.py",
        #     "function_name": "evaluate",
        #     "dataset_cls": {
        #         "dataset_func": "data/flickr.py",
        #         "dataset_name": "Flickr30K",
        #         "args": {"tokenizer": None, "transform": None, "test": True},
        #     },
        # },
    }
}

FEAT_VIZ_SETTINGS = {
    "dataset_cls": {
        "dataset_func": "data/flickr.py",
        "dataset_name": "Flickr8K",
        "args": {"tokenizer": None, "transform": None, "test": True},
    },
}
