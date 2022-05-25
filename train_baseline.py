from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from baseline.tokenizer import tokenize
from baseline.model import NbLogisticRegression
from baseline.evaluate import evaluate
from baseline.limesoda import LimeSodaDataset, read_limesoda

DELIMITER = " "
LIMESODA_DIR = "/workspace/dataset/LimeSoda/"


def main():
    parameters_grid = {
        # tfidf grids
        "tfidf__ngram_range": [[1, 2]],
        "tfidf__min_df": [5, 7],
        "tfidf__max_df": [0.5, 0.6],
        "tfidf__use_idf": [True],
        "tfidf__smooth_idf": [True, False],
        "tfidf__sublinear_tf": [True, False],
        "tfidf__tokenizer": [lambda x: x.split(DELIMITER)],
        "tfidf__token_pattern": [None],

        # clf grids
        "clf__alpha": [0.5, 0.75],
        # "clf__kernel": ["rbf", "linear"],
        # "clf__gamma": "scale",
        "clf__C": [1.],
    }

    dataset = read_limesoda(LIMESODA_DIR, delimiter=DELIMITER)

    for data_percentage in (1., 0.8, 0.6, 0.4):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", NbLogisticRegression())
        ])

        clf = GridSearchCV(
            estimator=pipeline,
            param_grid=parameters_grid,
            scoring="accuracy",
            n_jobs=-1,
        )


        train_data = dataset["train"].iloc[:int(len(dataset["train"])*data_percentage)]
        x_train, y_train = train_data["text"].values, train_data["label"].values
        x_val, y_val = dataset["val"]["text"].values, dataset["val"]["label"].values
        x_test, y_test = dataset["test"]["text"].values, dataset["test"]["label"].values

        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_test)

        results = evaluate(clf, x_test, y_test)
        _ = results.pop("prediction")

        print(f"Data Percentage: {data_percentage*100:.2f}%")
        print(f'\t> Accuracy: {results["accuracy"]["overall"]*100:.2f}%')
        print(f'\t> Macro F1: {results["f1"]["macro"]*100:.2f}%\n')
        
        
if __name__ == "__main__":
    main()
