import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss

class FeatureAnalysis:
    def __init__(self, df: pd.DataFrame, target: str, correlation_threshold: float = 0.2):
        self.df = df
        self.target = target
        self.correlation_threshold = correlation_threshold
        self.features = [col for col in df.columns if col != target]

    def correlation_with_target(self, corr_level=0.1):

        insignificant_features = []

        for col in self.features:
            corr, p_value = pearsonr(self.df[col], self.df[self.target])
            insignificant_features.append({
                'feature': col,
                'correlation': corr,
                'p-value': p_value
            })

        correlation_df = pd.DataFrame(insignificant_features)
        correlation_df = correlation_df.sort_values(by = 'correlation', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=correlation_df.feature, y=correlation_df.correlation)
        plt.title(f'Correlation with {self.target}')
        plt.xticks(rotation=90)
        plt.savefig("correlation_with_target.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        return list(correlation_df[correlation_df['correlation'] >= corr_level]['feature'])

    def correlation_matrix(self):

        correlation = self.df[self.features].corr()
        corr_above_threshold = correlation[(correlation.abs() > self.correlation_threshold) & (correlation.abs() < 1)]

        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix')
        plt.savefig("correlation_matrix.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        to_remove = set()
        for col in corr_above_threshold.columns:
            correlated_features = corr_above_threshold.index[corr_above_threshold[col].notnull()].tolist()
            if correlated_features:
                to_remove.update(correlated_features[1:])

        return list(to_remove)

    def feature_importance(self):

        X = self.df[self.features]
        y = self.df[self.target]

        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)

        feature_importances = pd.Series(model.feature_importances_, index=self.features).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importances.index, y=feature_importances.values)
        plt.title('Feature Importance')
        plt.xticks(rotation=90)
        plt.savefig("feature_importance.pdf", format="pdf", bbox_inches="tight")
        plt.show()

        return feature_importances

    def generate_golden_features(self, selected_feature, max_depth=3, sample_size=2500):

        other_features = [col for col in self.features if col != selected_feature]
        scores = []
        golden_features = {}

        for other_feature in other_features:
            for operation in ["sub", "div"]:
                if operation == "sub":
                    new_feature = self.df[selected_feature] - self.df[other_feature]
                elif operation == "div":
                    new_feature = self.df[selected_feature] / (self.df[other_feature] + 1e-9)

                new_feature_df = pd.DataFrame({"new_feature": new_feature})

                sampled_indices = np.random.choice(self.df.index, size=min(len(self.df), sample_size), replace=False)
                train_indices = sampled_indices[:len(sampled_indices) // 2]
                test_indices = sampled_indices[len(sampled_indices) // 2:]

                X_train = new_feature_df.loc[train_indices]
                X_test = new_feature_df.loc[test_indices]
                y_train = self.df[self.target].loc[train_indices]
                y_test = self.df[self.target].loc[test_indices]

                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_test)
                score = log_loss(y_test, y_pred)

                feature_name = f"{selected_feature}_{operation}_{other_feature}"
                scores.append((feature_name, score))
                golden_features[feature_name] = new_feature


        scores.sort(key=lambda x: x[1])
        num_features_to_select = max(5, min(50, int(0.05 * len(other_features))))
        selected_features = [name for name, _ in scores[:num_features_to_select]]
        golden_features_df = pd.DataFrame({name: golden_features[name] for name in selected_features})

        return golden_features_df

    def extract_best_features(self):

        high_corr_features = self.correlation_matrix()
        self.features = [col for col in self.features if col not in high_corr_features]
        low_corr_with_target = self.correlation_with_target()
        self.features = [col for col in self.features if col not in low_corr_with_target]
        feature_importance = self.feature_importance()
        golden_feature_part = feature_importance.index[0]
        golden_features_df = self.generate_golden_features(golden_feature_part, max_depth=3, sample_size=2500)
        self.df = pd.concat([self.df, golden_features_df], axis=1)
        self.features.extend(list(golden_features_df.columns))

