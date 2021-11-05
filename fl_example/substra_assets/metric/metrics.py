from sklearn.metrics import roc_auc_score

import substratools as tools


class AUCMetrics(tools.Metrics):
    def score(self, y_true, y_pred):
        """Returns the ROC AUC score

        :param y_true: actual values from test data
        :type y_true: pd.DataFrame
        :param y_true: predicted values from test data
        :type y_pred: pd.DataFrame
        :rtype: float
        """
        print("y_true", y_true)
        print("y_pred", y_pred)

        return roc_auc_score(y_true, y_pred)


if __name__ == "__main__":
    tools.metrics.execute(AUCMetrics())
