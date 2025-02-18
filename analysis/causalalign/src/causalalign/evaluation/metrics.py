import numpy as np
import pandas as pd
from scipy import stats


class ModelEvaluator:
    def __init__(self, model, data, domain, subject, ppp):
        """Initialize evaluator with model and data"""
        self.model = model
        self.data = data
        self.domain = domain
        self.subject = subject
        self.ppp = ppp

        # Filter relevant data
        self.filtered_data = data[
            (data["domain"] == domain)
            & (data["subject"] == subject)
            & (data["ppp"] == ppp)
        ].copy()

        # Get model predictions
        self.predictions = self._get_model_predictions()
        self.empirical = (
            self.filtered_data["response"].values / 100.0
        )  # Convert to [0,1]

    def _get_model_predictions(self):
        """Get model predictions for all tasks"""
        predictions = []
        for task in self.filtered_data["label"].str.upper():
            pred = self.model(task)
            predictions.append(pred.item())
        return np.array(predictions)

    def compute_all_metrics(self):
        """Compute all evaluation metrics"""
        metrics = {
            # Overall fit metrics
            "mse": self.mean_squared_error(),
            "rmse": self.root_mean_squared_error(),
            "mae": self.mean_absolute_error(),
            "log_loss": self.log_loss(),
            "r2": self.r_squared(),
            # Statistical tests
            "correlation": self.correlation_analysis(),
            "ks_test": self.distribution_test(),
            # Task-specific analysis
            "task_errors": self.task_specific_errors(),
            "max_deviation": self.maximum_deviation(),
            # Information criteria
            "aic": self.akaike_information_criterion(),
            "bic": self.bayesian_information_criterion(),
            # Calibration metrics
            "calibration": self.calibration_metrics(),
        }
        return metrics

    def mean_squared_error(self):
        """Compute MSE"""
        return np.mean((self.predictions - self.empirical) ** 2)

    def root_mean_squared_error(self):
        """Compute RMSE"""
        return np.sqrt(self.mean_squared_error())

    def mean_absolute_error(self):
        """Compute MAE"""
        return np.mean(np.abs(self.predictions - self.empirical))

    def log_loss(self):
        """Compute log loss with epsilon for numerical stability"""
        epsilon = 1e-15
        pred_clipped = np.clip(self.predictions, epsilon, 1 - epsilon)
        emp_clipped = np.clip(self.empirical, epsilon, 1 - epsilon)
        return -np.mean(
            emp_clipped * np.log(pred_clipped)
            + (1 - emp_clipped) * np.log(1 - pred_clipped)
        )

    def r_squared(self):
        """Compute R² score"""
        ss_res = np.sum((self.empirical - self.predictions) ** 2)
        ss_tot = np.sum((self.empirical - np.mean(self.empirical)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-10))

    def correlation_analysis(self):
        """Compute correlation statistics"""
        pearson_r, pearson_p = stats.pearsonr(self.empirical, self.predictions)
        spearman_r, spearman_p = stats.spearmanr(self.empirical, self.predictions)
        return {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
        }

    def distribution_test(self):
        """Perform Kolmogorov-Smirnov test"""
        ks_statistic, p_value = stats.ks_2samp(self.empirical, self.predictions)
        return {"ks_statistic": ks_statistic, "p_value": p_value}

    def task_specific_errors(self):
        """Compute errors for each task"""
        tasks = self.filtered_data["label"].values
        errors = self.predictions - self.empirical
        return {task: error for task, error in zip(tasks, errors)}

    def maximum_deviation(self):
        """Find maximum deviation and corresponding task"""
        tasks = self.filtered_data["label"].values
        errors = np.abs(self.predictions - self.empirical)
        max_idx = np.argmax(errors)
        return {"task": tasks[max_idx], "deviation": errors[max_idx]}

    def akaike_information_criterion(self):
        """Compute AIC"""
        n = len(self.empirical)
        k = 5  # Number of parameters (prior_c1, prior_c2, m1, m2, b)
        mse = self.mean_squared_error()
        return n * np.log(mse) + 2 * k

    def bayesian_information_criterion(self):
        """Compute BIC"""
        n = len(self.empirical)
        k = 5  # Number of parameters
        mse = self.mean_squared_error()
        return n * np.log(mse) + k * np.log(n)

    def calibration_metrics(self):
        """Compute calibration metrics"""
        # Create bins for calibration analysis
        bins = np.linspace(0, 1, 11)
        bin_indices = np.digitize(self.predictions, bins) - 1

        calibration_stats = []
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                mean_pred = np.mean(self.predictions[mask])
                mean_emp = np.mean(self.empirical[mask])
                calibration_stats.append(
                    {
                        "bin": f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                        "mean_prediction": mean_pred,
                        "mean_empirical": mean_emp,
                        "samples": np.sum(mask),
                    }
                )

        return pd.DataFrame(calibration_stats)


# Example usage in training loop:
def evaluate_model_fit(model, data, domain, subject, ppp):
    """Evaluate model fit and return formatted results"""
    evaluator = ModelEvaluator(model, data, domain, subject, ppp)
    metrics = evaluator.compute_all_metrics()

    # Format results for printing
    print(f"\nModel Evaluation Results for {domain}, {subject}, {ppp}")
    print("-" * 50)

    print("\nOverall Fit Metrics:")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")

    print("\nCorrelation Analysis:")
    print(
        f"Pearson r: {metrics['correlation']['pearson_r']:.4f} (p={metrics['correlation']['pearson_p']:.4f})"
    )
    print(
        f"Spearman r: {metrics['correlation']['spearman_r']:.4f} (p={metrics['correlation']['spearman_p']:.4f})"
    )

    print("\nDistribution Test (KS):")
    print(f"Statistic: {metrics['ks_test']['ks_statistic']:.4f}")
    print(f"p-value: {metrics['ks_test']['p_value']:.4f}")

    print("\nWorst Fitting Task:")
    print(f"Task: {metrics['max_deviation']['task']}")
    print(f"Maximum Deviation: {metrics['max_deviation']['deviation']:.4f}")

    print("\nModel Complexity:")
    print(f"AIC: {metrics['aic']:.4f}")
    print(f"BIC: {metrics['bic']:.4f}")

    print("\nCalibration Summary:")
    print(metrics["calibration"].to_string())

    return metrics
