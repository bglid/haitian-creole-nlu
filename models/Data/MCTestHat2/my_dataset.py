import datasets


class MCTestConfig(datasets.BuilderConfig):

    def __init__(
        self, features, path_to_dir, split, label_classes=("A", "B", "C", "D"), **kwargs
    ):

        super().__init__(version=datasets.Version("1.0.2"), **kwargs)
        self.features = features
        self.path_to_dir = path_to_dir
        self.split = split
        self.label_classes = label_classes

    def _info(self):
        return datasets.DatasetInfo(
            # description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "story": datasets.Value("string"),
                    "story_id": datasets.Value("string"),
                    "question_id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "q_type": datasets.Value("string"),
                    "choices": datasets.Sequence("string"),
                    "answer": datasets.Value("string"),
                    "text_answer": datasets.Value("string"),
                    "label": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://creolenlp.github.io/mctest/",
            # citation=_CITATION,
        )


class MCTest(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        MCTestConfig(
            name="test",
            features=[
                "story",
                "story_id",
                "question",
                "question_id",
                "q_type",
                "choices",
                "answer",
                "text_answer",
                "label",
            ],
            path_to_dir="/home/bglid/uni_ms/ling_545/haitian-creole-nlu/models/Data/MCTestHat2",
            split="mc160.dev",
        )
    ]
