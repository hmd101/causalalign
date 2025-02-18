import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# from statsmodels.stats.mixed_lm import MixedLM
# In significance_analyzer.py
import statsmodels.regression.mixed_linear_model as mlm
from scipy import stats
from scipy.stats import kruskal, levene, pearsonr, shapiro, spearmanr
from sklearn.linear_model import LinearRegression

MixedLM = mlm.MixedLM

# from statsmodels.stats.multicomp import MultiComparison
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multicomp import MultiComparison


class CausalReasoningAnalysis:
    def __init__(self, data):
        self.data = data
        self.prepare_data()

    def prepare_data(self):
        """Prepare data for analysis"""
        # Convert response to numeric and scale to 0-1
        self.data["response"] = (
            pd.to_numeric(self.data["response"], errors="coerce") / 100.0
        )

        # Create agent ID if not present
        if "id" not in self.data.columns:
            self.data["id"] = range(len(self.data))

    def check_assumptions(self):
        """Check normality and homogeneity of variance"""
        results = {}

        # Normality test (Shapiro-Wilk)
        _, p_value = shapiro(self.data["response"].dropna())
        results["normality"] = {
            "test": "Shapiro-Wilk",
            "p_value": p_value,
            "normal": p_value > 0.05,
        }

        # Homogeneity of variance (Levene's test)
        groups = []
        for agent_type in self.data["agent"].unique():
            group = self.data[self.data["agent"] == agent_type]["response"].dropna()
            groups.append(group)

        _, p_value = levene(*groups)
        results["homogeneity"] = {
            "test": "Levene's test",
            "p_value": p_value,
            "equal_variance": p_value > 0.05,
        }

        return results

    ##############################################

    ### 3. Statistical tests

    def perform_statistical_tests(self, marginalize_temperature=False, alpha=0.05):
        """Perform Kruskal-Wallis tests comparing human and LLM responses."""
        group_cols = ["domain", "task"]
        if not marginalize_temperature:
            group_cols.append("temperature")

        # Separate human and LLM responses
        human_data = self.data[self.data["agent"] == "humans"]
        llm_data = self.data[self.data["agent"] != "humans"]

        # Merge human and LLM responses
        merged_data = pd.merge(
            human_data[group_cols + ["response"]],
            llm_data[group_cols + ["response", "agent"]],
            on=group_cols,
            suffixes=("_human", "_llm"),
        )

        # Perform Kruskal-Wallis tests
        results = []
        for _, group in merged_data.groupby(group_cols):
            human_responses = group["response_human"].dropna()
            llm_responses = group["response_llm"].dropna()
            llm_name = group["agent"].iloc[0]  # Specific LLM being compared

            # Calculate number of samples
            num_human_samples = len(human_responses)
            num_llm_samples = len(llm_responses)
            total_samples = num_human_samples + num_llm_samples

            if num_human_samples > 1 and num_llm_samples > 1:
                # Run Kruskal-Wallis test
                h_stat, p_value = kruskal(human_responses, llm_responses)
            else:
                # Assign None if insufficient data
                h_stat, p_value = None, None

            # Append all groups to results
            results.append(
                {
                    "domain": group["domain"].iloc[0],
                    "task": group["task"].iloc[0],
                    "temperature": group["temperature"].iloc[0]
                    if "temperature" in group
                    else None,
                    "LLM": llm_name,
                    "h_stat": h_stat,
                    "p_value": p_value,
                    "number_of_samples": total_samples,
                    "significant": p_value < alpha if p_value is not None else False,
                }
            )

        return pd.DataFrame(results)

    ### 4. Effect sizes

    #

    def compute_effect_sizes(
        self, statistical_results, alpha=0.05, effect_size_type="cohen_d"
    ):
        """
        Compute effect sizes for significant results with modularity for effect size type.

        Parameters:
            statistical_results (DataFrame): Results of the Kruskal-Wallis test.
            alpha (float): Significance threshold for filtering results.
            effect_size_type (str): Type of effect size to compute ('cohen_d', 'rank_biserial', 'cliffs_delta').

        Returns:
            DataFrame: Effect sizes for significant results.
        """
        # Filter significant results
        significant = statistical_results[statistical_results["p_value"] < alpha]
        effect_sizes = []

        for _, row in significant.iterrows():
            domain = row["domain"]
            task = row["task"]
            temperature = row["temperature"]
            llm = row["LLM"]

            # Filter responses
            human_responses = self.data[
                (self.data["agent"] == "humans")
                & (self.data["domain"] == domain)
                & (self.data["task"] == task)
            ]["response"]

            llm_responses = self.data[
                (self.data["agent"] == llm)
                & (self.data["domain"] == domain)
                & (self.data["task"] == task)
            ]["response"]

            if len(human_responses) > 1 and len(llm_responses) > 1:
                if effect_size_type == "cohen_d":
                    # Compute Cohen's d
                    mean_diff = np.mean(llm_responses) - np.mean(human_responses)
                    pooled_sd = np.sqrt(
                        (
                            np.var(human_responses, ddof=1)
                            + np.var(llm_responses, ddof=1)
                        )
                        / 2
                    )
                    effect_size = mean_diff / pooled_sd if pooled_sd > 0 else np.nan

                elif effect_size_type == "rank_biserial":
                    # Compute Rank-Biserial Correlation
                    U, _ = stats.mannwhitneyu(
                        human_responses, llm_responses, alternative="two-sided"
                    )
                    effect_size = U / (len(human_responses) * len(llm_responses))

                elif effect_size_type == "cliffs_delta":
                    # Compute Cliff's Delta
                    n1, n2 = len(human_responses), len(llm_responses)
                    greater = sum(x > y for x in llm_responses for y in human_responses)
                    lesser = sum(x < y for x in llm_responses for y in human_responses)
                    effect_size = (greater - lesser) / (n1 * n2)

                else:
                    raise ValueError(f"Invalid effect size type: {effect_size_type}")

                # Append results
                effect_sizes.append(
                    {
                        "domain": domain,
                        "task": task,
                        "temperature": temperature,
                        "LLM": llm,
                        "effect_size": effect_size,
                        "effect_size_type": effect_size_type,
                    }
                )

        return pd.DataFrame(effect_sizes)

    ################################################

    def plot_distributions(self):
        """Create distribution plots for the response variable"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # Overall distribution
        sns.histplot(data=self.data, x="response", ax=axes[0, 0])
        axes[0, 0].set_title("Overall Response Distribution")

        # Distribution by agent type
        sns.boxplot(data=self.data, x="agent", y="response", ax=axes[0, 1])
        axes[0, 1].set_title("Response by agent Type")
        axes[0, 1].tick_tasks = axes[0, 1].xaxis.set_tick_params(rotation=45)

        # Distribution by domain
        sns.boxplot(data=self.data, x="domain", y="response", ax=axes[1, 0])
        axes[1, 0].set_title("Response by Domain")

        # Distribution by task
        sns.boxplot(data=self.data, x="task", y="response", ax=axes[1, 1])
        axes[1, 1].set_title("Response by Task")
        axes[1, 1].tick_tasks = axes[1, 1].xaxis.set_tick_params(rotation=45)

        plt.tight_layout()
        return fig

    def run_mixed_effects_model(self):
        """Run mixed effects model with proper fixed and random effects"""
        # Create fixed effects
        fixed_effects = ["agent", "domain", "ppp", "task"]

        # Prepare formula for fixed effects and their interactions
        formula = (
            "response ~ "
            + " + ".join(fixed_effects)
            + " + agent:domain + agent:ppp + domain:ppp"
        )

        # Set up random effects structure
        # Random intercepts for agents and random slopes for domain and ppp
        groups = {"id": "0 + domain + ppp"}

        try:
            model = MixedLM.from_formula(formula=formula, groups=groups, data=self.data)
            result = model.fit()
            return result
        except Exception as e:
            print(f"Error fitting mixed model: {str(e)}")
            return None

    def analyze_domain_differences_with_assumptions(self):
        """Analyze domain differences within agents with appropriate tests based on assumption checks
        use Kruskal-Wallis test if assumptions are violated, otherwise use ANOVA
        - Kruskal-Wallis test is a non-parametric alternative to ANOVA and
          compares response values across different domain groups for each agent and task

        """

        _, assumption_df = self.check_assumptions_by_task()

        results = {}
        test_counts = {"anova": 0, "kruskal": 0}

        for agent in self.data["agent"].unique():
            agent_data = self.data[self.data["agent"] == agent]
            results[agent] = {}

            for task in agent_data["task"].unique():
                task_data = agent_data[agent_data["task"] == task]

                # Get assumption check results
                assumption_checks = assumption_df[
                    (assumption_df["agent"] == agent) & (assumption_df["task"] == task)
                ]

                normality_violated = (
                    assumption_checks[
                        (assumption_checks["test_type"] == "normality")
                        & assumption_checks["violation"]
                    ]
                    .any()
                    .any()
                )

                variance_violated = (
                    assumption_checks[
                        (assumption_checks["test_type"] == "variance")
                        & assumption_checks["violation"]
                    ]
                    .any()
                    .any()
                )

                # Get domain groups
                domain_groups = {
                    domain: group["response"].dropna().values
                    for domain, group in task_data.groupby("domain")
                }

                if all(len(group) > 0 for group in domain_groups.values()):
                    if normality_violated or variance_violated:
                        stat, p_val = stats.kruskal(*domain_groups.values())
                        test_type = "kruskal"
                        test_counts["kruskal"] += 1
                    else:
                        stat, p_val = stats.f_oneway(*domain_groups.values())
                        test_type = "anova"
                        test_counts["anova"] += 1

                    result_dict = {
                        "statistic": stat,
                        "p_value": p_val,
                        "significant": p_val < 0.05,
                        "test_used": test_type,
                        "assumptions_violated": {
                            "normality": normality_violated,
                            "variance": variance_violated,
                        },
                        "domain_means": {
                            domain: values.mean()
                            for domain, values in domain_groups.items()
                        },
                    }

                    # Add posthoc if significant
                    if p_val < 0.05:
                        if test_type == "kruskal":
                            posthoc = self._run_mannwhitney_posthoc(task_data)
                            # Convert posthoc to readable format
                            posthoc_results = [
                                {
                                    "comparison": f"{row.domain1} vs {row.domain2}",
                                    "statistic": row.statistic,
                                    "p_value": row.p_value,
                                    "adjusted_p": row.adjusted_p_value,
                                    "significant": row.adjusted_p_value < 0.05,
                                }
                                for _, row in posthoc.iterrows()
                            ]
                        else:
                            tukey = MultiComparison(
                                task_data["response"], task_data["domain"]
                            ).tukeyhsd()
                            # Convert Tukey results to readable format
                            posthoc_results = [
                                {
                                    "comparison": f"{group1} vs {group2}",
                                    "difference": diff,
                                    "statistic": None,  # Tukey doesn't provide test statistic
                                    "p_value": pvalue,
                                    "significant": pvalue < 0.05,
                                }
                                for group1, group2, diff, _, pvalue, _ in zip(
                                    tukey.groupsunique[tukey.groups1],
                                    tukey.groupsunique[tukey.groups2],
                                    tukey.meandiffs,
                                    tukey.confint,
                                    tukey.pvalues,
                                    tukey.reject,
                                )
                            ]

                        result_dict["posthoc"] = posthoc_results

                    results[agent][task] = result_dict

        # Add test usage statistics
        results["test_usage"] = {
            "anova_count": test_counts["anova"],
            "kruskal_count": test_counts["kruskal"],
            "anova_percentage": test_counts["anova"] / sum(test_counts.values()) * 100,
        }

        return results

    def _run_mannwhitney_posthoc(self, data):
        """Run Mann-Whitney U tests with improved error handling"""
        domains = data["domain"].unique()
        if len(domains) < 2:
            return pd.DataFrame()  # Return empty DataFrame if not enough domains

        results = []
        # Calculate Bonferroni correction factor
        n_comparisons = len(domains) * (len(domains) - 1) / 2

        try:
            for i, domain1 in enumerate(domains):
                for domain2 in domains[i + 1 :]:
                    group1 = data[data["domain"] == domain1]["response"].dropna()
                    group2 = data[data["domain"] == domain2]["response"].dropna()

                    # Check if enough data points
                    if len(group1) < 2 or len(group2) < 2:
                        print(f"Warning: Not enough data for {domain1} vs {domain2}")
                        continue

                    stat, p_val = stats.mannwhitneyu(
                        group1, group2, alternative="two-sided"
                    )

                    # Apply Bonferroni correction
                    adjusted_p = min(p_val * n_comparisons, 1.0)

                    results.append(
                        {
                            "domain1": domain1,
                            "domain2": domain2,
                            "statistic": stat,
                            "p_value": p_val,
                            "adjusted_p_value": adjusted_p,
                            "significant": adjusted_p < 0.05,
                        }
                    )
        except Exception as e:
            print(f"Error in post-hoc analysis: {str(e)}")
            return pd.DataFrame()

        return pd.DataFrame(results)

    def analyze_within_agents(self):
        """Analyze within-agent differences across variables"""
        analyses = {}

        for agent in self.data["agent"].unique():
            agent_data = self.data[self.data["agent"] == agent]

            # Across domains
            f_stat, p_val = stats.f_oneway(
                *[
                    group["response"].dropna()
                    for name, group in agent_data.groupby("domain")
                ]
            )
            analyses[f"{agent}_domains"] = {"f_stat": f_stat, "p_value": p_val}

            # Across PPP conditions
            f_stat, p_val = stats.f_oneway(
                *[
                    group["response"].dropna()
                    for name, group in agent_data.groupby("ppp")
                ]
            )
            analyses[f"{agent}_ppp"] = {"f_stat": f_stat, "p_value": p_val}

            # Across inference tasks
            f_stat, p_val = stats.f_oneway(
                *[
                    group["response"].dropna()
                    for name, group in agent_data.groupby("task")
                ]
            )
            analyses[f"{agent}_tasks"] = {"f_stat": f_stat, "p_value": p_val}

        return analyses

    def analyze_temperature_effect(self):
        """Analyze effect of temperature for LLMs only"""
        llm_data = self.data[self.data["agent"].isin(["gpt-3.5-turbo", "gpt-4o"])]

        # Run ANOVA for temperature effect
        try:
            temp_groups = []
            for temp in llm_data["temperature"].unique():
                group = llm_data[llm_data["temperature"] == temp]["response"].dropna()
                temp_groups.append(group)

            f_stat, p_value = stats.f_oneway(*temp_groups)

            return {
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }
        except Exception as e:
            print(f"Error analyzing temperature effect: {str(e)}")
            return None

    # Do post-hoc analysis if we have significant results to determine which groups are different
    #  (i.e., at least one group mean or distribution is significantly different from the others.)

    # def post_hoc_analysis(self, factor):
    #     """Perform post-hoc analysis for a given factor"""

    #     mc = MultiComparison(self.data["response"], self.data[factor])
    #     result = mc.tukeyhsd()

    #     return result

    def post_hoc_analysis(self, factor):
        """Perform post-hoc analysis for a given factor"""
        mc = MultiComparison(self.data["response"], self.data[factor])
        result = mc.tukeyhsd()
        # result = stats.mannwhitneyu(self.data["response"], self.data[factor])

        # # Convert to DataFrame
        # df = pd.DataFrame(
        #     data=result._results_table.data[1:], columns=result._results_table.data[0]
        # )

        # # Save to CSV
        # df.to_csv("posthoc_analysis.csv", index=False)

        return result

    def plot_domain_task_differences(self, results):
        """Visualize domain differences for each task by agent"""
        # Create subplot for each agent
        agents = list(results.keys())
        fig, axes = plt.subplots(len(agents), 1, figsize=(15, 5 * len(agents)))
        if len(agents) == 1:
            axes = [axes]

        for idx, agent in enumerate(agents):
            agent_results = results[agent]
            tasks = list(agent_results.keys())

            # Create heatmap data
            domain_pairs = ["economy-sociology", "economy-weather", "sociology-weather"]
            heatmap_data = np.zeros((len(tasks), len(domain_pairs)))

            for i, task in enumerate(tasks):
                if "posthoc" in agent_results[task]:
                    posthoc = agent_results[task]["posthoc"]
                    for j, pair in enumerate(domain_pairs):
                        d1, d2 = pair.split("-")
                        # Find the pairwise comparison in posthoc results
                        pair_result = posthoc.pvalues[
                            (posthoc.groups1 == d1) & (posthoc.groups2 == d2)
                        ]
                        if len(pair_result) > 0:
                            heatmap_data[i, j] = -np.log10(pair_result[0])

            # Plot heatmap
            im = axes[idx].imshow(heatmap_data, cmap="YlOrRd")
            axes[idx].set_xticks(range(len(domain_pairs)))
            axes[idx].set_yticks(range(len(tasks)))
            axes[idx].set_xticktasks(domain_pairs, rotation=45)
            axes[idx].set_yticktasks(tasks)
            axes[idx].set_title(f"{agent} Domain Differences by Task\n(-log10 p-value)")

            # Add colorbar
            plt.colorbar(im, ax=axes[idx])

        plt.tight_layout()
        return fig

    def summarize_domain_task_differences(self, results):
        """Create a summary of significant domain differences by task"""
        summary = {}

        for agent, agent_results in results.items():
            summary[agent] = {
                "total_tasks": len(agent_results),
                "significant_tasks": sum(
                    1 for task in agent_results.values() if task["significant"]
                ),
                "significant_details": {
                    task: {
                        "p_value": res["p_value"],
                        "domain_means": res["domain_means"],
                    }
                    for task, res in agent_results.items()
                    if res["significant"]
                },
            }

        return summary

    def check_assumptions_by_task(self):
        """Check normality and variance assumptions for each agent-task combination"""
        results = {}
        df_rows = []

        # For each agent type
        for agent in self.data["agent"].unique():
            agent_data = self.data[self.data["agent"] == agent]
            results[agent] = {}

            # For each task
            for task in agent_data["task"].unique():
                task_data = agent_data[agent_data["task"] == task]

                # Get responses for each domain
                domain_groups = [
                    group["response"].dropna().values
                    for _, group in task_data.groupby("domain")
                ]
                domain_names = task_data["domain"].unique()

                # Only check if we have data for all domains
                if all(len(group) > 0 for group in domain_groups):
                    # Check normality for each domain group
                    normality_tests = {}
                    for domain, group in zip(domain_names, domain_groups):
                        if len(group) >= 3:  # Shapiro-Wilk needs at least 3 samples
                            stat, p_val = stats.shapiro(group)
                            normality_tests[domain] = {
                                "statistic": stat,
                                "p_value": p_val,
                                "normal": p_val > 0.05,
                            }

                            # Add row to DataFrame data
                            df_rows.append(
                                {
                                    "agent": agent,
                                    "task": task,
                                    "domain": domain,
                                    "test_type": "normality",
                                    "statistic": stat,
                                    "p_value": p_val,
                                    "assumption_met": p_val > 0.05,
                                }
                            )

                    # Check homogeneity of variance using Levene's test
                    if all(len(group) >= 2 for group in domain_groups):
                        levene_stat, levene_p = stats.levene(*domain_groups)
                        variance_equal = levene_p > 0.05

                        # Add row to DataFrame data
                        df_rows.append(
                            {
                                "agent": agent,
                                "task": task,
                                "domain": "all",  # variance test is across all domains
                                "test_type": "variance",
                                "statistic": levene_stat,
                                "p_value": levene_p,
                                "assumption_met": variance_equal,
                            }
                        )
                    else:
                        levene_stat, levene_p = None, None
                        variance_equal = None

                    results[agent][task] = {
                        "normality_tests": normality_tests,
                        "variance_test": {
                            "statistic": levene_stat,
                            "p_value": levene_p,
                            "equal_variance": variance_equal,
                        },
                    }

        # Create DataFrame
        df_results = pd.DataFrame(df_rows)

        # Add summary columns for quick filtering
        df_results["violation"] = ~df_results["assumption_met"]

        return results, df_results

    def summarize_assumption_violations(self, df_results):
        """Create summary statistics from the assumption test results DataFrame"""
        # Overall violation rates
        summaries = {
            "total_violations": len(df_results[df_results["violation"]]),
            "violation_rate": df_results["violation"].mean(),
            # Violations by test type
            "normality_violations": len(
                df_results[
                    (df_results["test_type"] == "normality") & df_results["violation"]
                ]
            ),
            "variance_violations": len(
                df_results[
                    (df_results["test_type"] == "variance") & df_results["violation"]
                ]
            ),
            # Violations by agent
            "violations_by_agent": df_results.groupby("agent")["violation"].mean(),
            # Violations by task
            "violations_by_task": df_results.groupby("task")["violation"].mean(),
            # Violations by domain
            "violations_by_domain": df_results.groupby("domain")["violation"].mean(),
        }

        return summaries

    def plot_assumption_violations_enhanced(self, df_results):
        """Create enhanced visualization of assumption violations"""
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2)

        # 1. Violation rates by agent and test type
        ax1 = fig.add_subplot(gs[0, 0])
        pivot_agent = pd.pivot_table(
            df_results,
            values="violation",
            index="agent",
            columns="test_type",
            aggfunc="mean",
        )
        pivot_agent.plot(kind="bar", ax=ax1)
        ax1.set_title("Violation Rates by agent and Test Type")
        ax1.set_ytask("Violation Rate")

        # 2. Violation rates by task
        ax2 = fig.add_subplot(gs[0, 1])
        df_results.groupby("task")["violation"].mean().plot(kind="bar", ax=ax2)
        ax2.set_title("Violation Rates by Task")
        ax2.set_ytask("Violation Rate")
        plt.xticks(rotation=45)

        # 3. Heatmap of violations
        ax3 = fig.add_subplot(gs[1, :])
        pivot_heatmap = pd.pivot_table(
            df_results,
            values="violation",
            index=["agent", "task"],
            columns="test_type",
            aggfunc="mean",
        )
        sns.heatmap(pivot_heatmap, annot=True, cmap="YlOrRd", ax=ax3)
        ax3.set_title("Violation Patterns Across agents and Tasks by test type, mean")

        plt.tight_layout()
        return fig

    def summarize_assumption_checks(self, results):
        """Summarize assumption check results"""
        summary = {}

        for agent, agent_results in results.items():
            summary[agent] = {
                "total_tests": len(agent_results),
                "normality_violations": sum(
                    1
                    for task_result in agent_results.values()
                    for domain_test in task_result["normality_tests"].values()
                    if not domain_test["normal"]
                ),
                "variance_violations": sum(
                    1
                    for task_result in agent_results.values()
                    if task_result["variance_test"]["equal_variance"] is False
                ),
                "problematic_tasks": {
                    task: {
                        "normality_violated": any(
                            not test["normal"]
                            for test in result["normality_tests"].values()
                        ),
                        "variance_violated": result["variance_test"]["equal_variance"]
                        is False,
                    }
                    for task, result in agent_results.items()
                },
            }

        return summary

    def plot_assumption_violations(self, results):
        """Visualize where assumption violations occur"""
        agents = list(results.keys())
        tasks = list(results[agents[0]].keys())

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Prepare data for heatmaps
        normality_data = np.zeros((len(agents), len(tasks)))
        variance_data = np.zeros((len(agents), len(tasks)))

        for i, agent in enumerate(agents):
            for j, task in enumerate(tasks):
                # Count normality violations
                normality_data[i, j] = sum(
                    1
                    for test in results[agent][task]["normality_tests"].values()
                    if not test["normal"]
                )
                # Check variance violation
                variance_data[i, j] = not results[agent][task]["variance_test"][
                    "equal_variance"
                ]

        # Plot normality violations
        im1 = ax1.imshow(normality_data, cmap="YlOrRd")
        ax1.set_xticks(range(len(tasks)))
        ax1.set_yticks(range(len(agents)))
        ax1.set_xticktasks(tasks)
        ax1.set_yticktasks(agents)
        ax1.set_title("Number of Normality Violations by Domain")
        plt.colorbar(im1, ax=ax1)

        # Plot variance violations
        im2 = ax2.imshow(variance_data, cmap="YlOrRd")
        ax2.set_xticks(range(len(tasks)))
        ax2.set_yticks(range(len(agents)))
        ax2.set_xticktasks(tasks)
        ax2.set_yticktasks(agents)
        ax2.set_title("Variance Homogeneity Violations")
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        return fig

    ####### for results reporting and subsequent plots
    # supports both excel and csv formats

    def plot_domain_comparison_results(self, results, LLM, task=None):
        """Create interactive visualization of domain comparison results with error checking"""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Validate inputs
        # if LLM not in results:
        #     raise ValueError(f"LLM '{LLM}' not found in results")
        if LLM not in results["LLM"].values:
            raise ValueError(f"LLM '{LLM}' not found in results['LLM'] column")

        # If task not specified, create plots for all tasks
        tasks = [task] if task else list(results[LLM].keys())
        if not tasks:
            raise ValueError(f"No tasks found for LLM '{LLM}'")

        for t in tasks:
            if t not in results[LLM]:
                raise ValueError(f"Task '{t}' not found for LLM '{LLM}'")

        # Print debug info
        print(f"Debug info for {LLM}, task(s) {tasks}:")
        for t in tasks:
            print(f"\nTask {t}:")
            print("Keys in result:", list(results[LLM][t].keys()))
            if "posthoc" in results[LLM][t]:
                print("Posthoc type:", type(results[LLM][t]["posthoc"]))
                print("Posthoc content:", results[LLM][t]["posthoc"])

        n_tasks = len(tasks)

        # Create subplot grid
        fig = make_subplots(
            rows=n_tasks,
            cols=2,
            subplot_titles=[f"Task {t} - Domain Means" for t in tasks]
            + [f"Task {t} - Domain Comparisons" for t in tasks],
            specs=[[{"type": "bar"}, {"type": "heatmap"}] for _ in range(n_tasks)],
        )

        for idx, task in enumerate(tasks, 1):
            task_results = results[LLM][task]

            # Domain means bar plot
            means = task_results["domain_means"]
            fig.add_trace(
                go.Bar(
                    x=list(means.keys()),
                    y=list(means.values()),
                    name=f"Task {task} Means",
                    text=[f"{v:.3f}" for v in means.values()],
                    textposition="auto",
                ),
                row=idx,
                col=1,
            )

            # Post-hoc comparison heatmap
            if "posthoc" in task_results:
                comparisons = [p["comparison"] for p in task_results["posthoc"]]
                p_values = [-np.log10(p["adjusted_p"]) for p in task_results["posthoc"]]
                significant = [p["significant"] for p in task_results["posthoc"]]

                # Create hover text
                hover_text = [
                    f"Comparison: {comp}<br>"
                    + f"p-value: {10 ** (-pval):.2e}<br>"
                    + f"Significant: {sig}"
                    for comp, pval, sig in zip(comparisons, p_values, significant)
                ]

                fig.add_trace(
                    go.Heatmap(
                        z=[p_values],
                        x=comparisons,
                        text=[[f"p={10 ** (-p):.2e}" for p in p_values]],
                        texttemplate="%{text}",
                        textfont={"size": 10},
                        colorscale="RdBu",
                        hoverongaps=False,
                        hovertext=[hover_text],
                        showscale=True,
                    ),
                    row=idx,
                    col=2,
                )

        # Update layout
        fig.update_layout(
            height=300 * n_tasks,
            width=1200,
            showlegend=False,
            title=f"Domain Comparison Results for {LLM}",
        )

        return fig

    def generate_text_report(self, results, agent, task=None):
        """Generate detailed text report of domain comparison results"""
        tasks = [task] if task else list(results[agent].keys())
        report_lines = []

        report_lines.append(f"Analysis Report for {agent}")
        report_lines.append("=" * 50)

        for task in tasks:
            task_results = results[agent][task]

            # Task header
            report_lines.append(f"\nTask {task}:")
            report_lines.append("-" * 20)

            # Test details
            report_lines.append(
                f"Test used: {task_results['test_used']} "
                f"(assumptions violated: "
                f"normality={task_results['assumptions_violated']['normality']}, "
                f"variance={task_results['assumptions_violated']['variance']})"
            )

            # Overall significance
            report_lines.append(
                f"Overall significance: "
                f"{'significant' if task_results['significant'] else 'not significant'} "
                f"(p={task_results['p_value']:.2e})"
            )

            # Domain means
            report_lines.append("\nDomain means:")
            for domain, mean in task_results["domain_means"].items():
                report_lines.append(f"  {domain}: {mean:.3f}")

            # Post-hoc comparisons
            if "posthoc" in task_results:
                report_lines.append("\nPairwise comparisons:")
                for comparison in task_results["posthoc"]:
                    sig_marker = "*" if comparison["significant"] else ""
                    report_lines.append(
                        f"  {comparison['comparison']}: "
                        f"p={comparison['adjusted_p']:.2e}{sig_marker}"
                    )

        return "\n".join(report_lines)

    def export_results_to_excel(self, results, filename):
        """Export results to Excel with multiple sheets for different aspects"""
        with pd.ExcelWriter(filename) as writer:
            # Summary sheet
            summary_data = []
            for agent in results:
                if agent != "test_usage":
                    for task, task_results in results[agent].items():
                        summary_data.append(
                            {
                                "agent": agent,
                                "Task": task,
                                "Test": task_results["test_used"],
                                "P-value": task_results["p_value"],
                                "Significant": task_results["significant"],
                                "Normality Violated": task_results[
                                    "assumptions_violated"
                                ]["normality"],
                                "Variance Violated": task_results[
                                    "assumptions_violated"
                                ]["variance"],
                            }
                        )

            pd.DataFrame(summary_data).to_excel(
                writer, sheet_name="Summary", index=False
            )

            # Domain means sheet
            means_data = []
            for agent in results:
                if agent != "test_usage":
                    for task, task_results in results[agent].items():
                        means_data.append(
                            {
                                "agent": agent,
                                "Task": task,
                                **task_results["domain_means"],
                            }
                        )

            pd.DataFrame(means_data).to_excel(
                writer, sheet_name="Domain Means", index=False
            )

            # Post-hoc comparisons sheet
            posthoc_data = []
            for agent in results:
                if agent != "test_usage":
                    for task, task_results in results[agent].items():
                        if "posthoc" in task_results:
                            for comparison in task_results["posthoc"]:
                                posthoc_data.append(
                                    {
                                        "agent": agent,
                                        "Task": task,
                                        "Comparison": comparison["comparison"],
                                        "P-value": comparison["p_value"],
                                        "Adjusted P-value": comparison.get(
                                            "adjusted_p", comparison["p_value"]
                                        ),
                                        "Significant": comparison["significant"],
                                    }
                                )

            pd.DataFrame(posthoc_data).to_excel(
                writer, sheet_name="Post-hoc", index=False
            )

    def export_results_to_csv(self, results, base_filename):
        """Export results to separate CSV files"""
        # Summary data
        summary_data = []
        for agent in results:
            if agent != "test_usage":
                for task, task_results in results[agent].items():
                    summary_data.append(
                        {
                            "agent": agent,
                            "Task": task,
                            "Test": task_results["test_used"],
                            "P_value": task_results["p_value"],
                            "Significant": task_results["significant"],
                            "Normality_Violated": task_results["assumptions_violated"][
                                "normality"
                            ],
                            "Variance_Violated": task_results["assumptions_violated"][
                                "variance"
                            ],
                        }
                    )
        pd.DataFrame(summary_data).to_csv(f"{base_filename}_summary.csv", index=False)

        # Domain means
        means_data = []
        for agent in results:
            if agent != "test_usage":
                for task, task_results in results[agent].items():
                    means_data.append(
                        {
                            "agent": agent,
                            "Task": task,
                            **task_results["domain_means"],
                        }
                    )
        pd.DataFrame(means_data).to_csv(f"{base_filename}_means.csv", index=False)

        # Post-hoc comparisons
        posthoc_data = []
        for agent in results:
            if agent != "test_usage":
                for task, task_results in results[agent].items():
                    if "posthoc" in task_results:
                        for comparison in task_results["posthoc"]:
                            posthoc_data.append(
                                {
                                    "agent": agent,
                                    "Task": task,
                                    "Comparison": comparison["comparison"],
                                    "P_value": comparison["p_value"],
                                    "Adjusted_P_value": comparison.get(
                                        "adjusted_p", comparison["p_value"]
                                    ),
                                    "Significant": comparison["significant"],
                                }
                            )
        pd.DataFrame(posthoc_data).to_csv(f"{base_filename}_posthoc.csv", index=False)

    def export_results(self, results, filename, format="excel"):
        """Export results in specified format"""
        if format.lower() == "excel":
            self.export_results_to_excel(results, filename + ".xlsx")
        elif format.lower() == "csv":
            self.export_results_to_csv(results, filename)
        else:
            raise ValueError("Format must be 'excel' or 'csv'")

    ################
    # Corelation Analysis

    def analyze_llm_human_correlation(
        self,
        by_domain=False,
        by_ppp=False,
        by_task=False,
        by_inference_type=False,
        epsilon=1e-6,
    ):
        """Analyze correlation between human and LLM responses with temperature consideration"""
        humans = self.data[self.data["agent"] == "humans"].copy()
        results = []

        epsilon_lower = epsilon + 1e-6
        epsilon_upper = epsilon + 1e-5
        llm_agents = [s for s in self.data["agent"].unique() if s != "humans"]

        # Define matching columns
        match_columns = ["domain", "ppp", "task", "inference_type"]
        groupby_cols = []
        if by_domain:
            groupby_cols.append("domain")
        if by_inference_type:
            groupby_cols.append("inference_type")
        if by_ppp:
            groupby_cols.append("ppp")
        if by_task:
            groupby_cols.append("task")

        for llm in llm_agents:
            llm_data = self.data[self.data["agent"] == llm].copy()

            # Group by temperature first
            for temp in llm_data["temperature"].unique():
                temp_data = llm_data[llm_data["temperature"] == temp]

                if groupby_cols:
                    for name, group in temp_data.groupby(groupby_cols):
                        # Create mask for matching groups
                        human_mask = np.ones(len(humans), dtype=bool)
                        for col, val in zip(
                            groupby_cols, name if isinstance(name, tuple) else [name]
                        ):
                            human_mask &= humans[col] == val

                        # Get matching data
                        matched_data = pd.merge(
                            humans[human_mask][match_columns + ["response"]],
                            group[match_columns + ["response"]],
                            on=match_columns,
                            suffixes=("_human", "_llm"),
                        )

                        # Drop rows with NaN values
                        matched_data = matched_data.dropna(
                            subset=["response_human", "response_llm"]
                        )

                        if len(matched_data) < 2:
                            print(
                                f"Skipping correlation: Not enough samples (N={len(matched_data)})"
                            )
                            corr, p = np.nan, np.nan
                        else:
                            # **Handle cases with low variance**
                            llm_std = matched_data["response_llm"].std()
                            human_std = matched_data["response_human"].std()

                            if llm_std < epsilon:
                                matched_data["response_llm"] += np.random.uniform(
                                    epsilon_lower, epsilon_upper, size=len(matched_data)
                                )
                                print(
                                    "Added small noise to LLM responses (Low variance detected)"
                                )
                            if human_std < epsilon:
                                matched_data["response_human"] += np.random.uniform(
                                    epsilon_lower, epsilon_upper, size=len(matched_data)
                                )
                                print(
                                    "Added small noise to Human responses (Low variance detected)"
                                )

                            # Debug prints before computing correlation
                            print("\n- -- Debug Info ---")
                            print(f"LLM: {llm} | Temperature: {temp}")
                            print(f"Matched Data Samples: {len(matched_data)}")
                            # print(
                            #     f"Unique LLM Responses: {matched_data['response_llm'].unique()}"
                            # )
                            # print(
                            #     f"Unique Human Responses: {matched_data['response_human'].unique()}"
                            # )
                            print(f"LLM Std Dev: {matched_data['response_llm'].std()}")
                            print(
                                f"Human Std Dev: {matched_data['response_human'].std()}"
                            )

                            # If there's still zero variance, skip computation
                            if (
                                matched_data["response_llm"].std() == 0
                                or matched_data["response_human"].std() == 0
                            ):
                                print(
                                    "Skipping due to zero variance after noise adjustment"
                                )
                                corr, p = np.nan, np.nan
                            else:
                                # Test normality (Shapiro-Wilk test)
                                if len(matched_data) >= 3:
                                    _, h_pval = shapiro(matched_data["response_human"])
                                    _, l_pval = shapiro(matched_data["response_llm"])
                                    is_normal = h_pval > 0.05 and l_pval > 0.05
                                else:
                                    is_normal = False

                                if is_normal:
                                    corr, p = pearsonr(
                                        matched_data["response_human"],
                                        matched_data["response_llm"],
                                    )
                                    method = "pearson"
                                else:
                                    matched_data["response_llm"] += np.random.uniform(
                                        1e-6, 1e-5, size=len(matched_data)
                                    )
                                    corr, p = spearmanr(
                                        matched_data["response_human"],
                                        matched_data["response_llm"],
                                    )
                                    method = "spearman"

                            print(f"Computed Correlation: {corr}")

                            result = {
                                "LLM": llm,
                                "Temperature": temp,
                                "Correlation": corr,
                                "P_value": p,
                                "Method": method,
                                "N": len(matched_data),
                            }

                            # Add grouping information
                            if isinstance(name, tuple):
                                for col, val in zip(groupby_cols, name):
                                    result[col] = val
                            else:
                                result[groupby_cols[0]] = name

                            results.append(result)

        return pd.DataFrame(results)

    ###### end correlation analysis

    def plot_temperature_correlations(self, correlation_results):
        """Visualize correlation results across temperatures"""
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2)

        # 1. Correlation vs Temperature
        ax1 = fig.add_subplot(gs[0, 0])
        sns.scatterplot(
            data=correlation_results,
            x="domain",
            y="Correlation",
            hue="LLM",
            style="Method",
            s=100,
            ax=ax1,
        )
        ax1.set_title("Correlation with Human Responses vs Temperature")
        ax1.set_ytask("Correlation Coefficient")

        # 2. Significance vs Temperature
        ax2 = fig.add_subplot(gs[0, 1])
        sig_plot = -np.log10(correlation_results["P_value"])
        sns.scatterplot(
            data=correlation_results,
            x="Temperature",
            y=sig_plot,
            hue="LLM",
            style="Method",
            s=100,
            ax=ax2,
        )
        ax2.set_title("Correlation Significance vs Temperature")
        ax2.set_ytask("-log10(p-value)")
        ax2.axhline(-np.log10(0.05), color="r", linestyle="--", task="p=0.05")
        ax2.legend()

        # 3. Sample size vs Temperature
        ax3 = fig.add_subplot(gs[1, :])
        sns.barplot(data=correlation_results, x="Temperature", y="N", hue="LLM", ax=ax3)
        ax3.set_title("Sample Size by Temperature")
        ax3.set_ytask("Number of Matched Responses")

        plt.tight_layout()
        return fig

    def analyze_temperature_response_correlation(self):
        """Analyze how temperature affects LLM responses and their correlation with humans"""
        humans = self.data[self.data["agent"] == "humans"]
        results = []

        for llm in [s for s in self.data["agent"].unique() if s != "humans"]:
            llm_data = self.data[self.data["agent"] == llm]

            # Skip if no temperature variation
            if len(llm_data["temperature"].unique()) <= 1:
                continue

            for temp in llm_data["temperature"].unique():
                temp_data = llm_data[llm_data["temperature"] == temp]

                # Match with human data
                matched_data = pd.merge(
                    humans[["domain", "ppp", "task", "response"]],
                    temp_data[["domain", "ppp", "task", "response", "temperature"]],
                    on=["domain", "ppp", "task"],
                    suffixes=("_human", "_llm"),
                )

                # Calculate correlation
                corr, p = spearmanr(
                    matched_data["response_human"], matched_data["response_llm"]
                )

                results.append(
                    {
                        "LLM": llm,
                        "Temperature": temp,
                        "Correlation": corr,
                        "P_value": p,
                        "Response_Std": temp_data["response"].std(),
                        "Response_Range": temp_data["response"].max()
                        - temp_data["response"].min(),
                        "N": len(matched_data),
                    }
                )

        return pd.DataFrame(results)

    # Regression Analysis

    def regression_analysis(self, by_domain=False, by_ppp=False):
        """Perform regression analysis of LLM responses on human responses with temperature consideration"""
        humans = self.data[self.data["agent"] == "humans"].copy()
        results = []

        # Define matching columns
        match_columns = ["domain", "ppp", "task"]
        groupby_cols = []
        if by_domain:
            groupby_cols.append("domain")
        if by_ppp:
            groupby_cols.append("ppp")

        for llm in [s for s in self.data["agent"].unique() if s != "humans"]:
            llm_data = self.data[self.data["agent"] == llm].copy()

            # Group by temperature first
            for temp in llm_data["temperature"].unique():
                temp_data = llm_data[llm_data["temperature"] == temp]

                if groupby_cols:
                    for name, group in temp_data.groupby(groupby_cols):
                        # Create mask for matching groups
                        human_mask = np.ones(len(humans), dtype=bool)
                        for col, val in zip(
                            groupby_cols, name if isinstance(name, tuple) else [name]
                        ):
                            human_mask &= humans[col] == val

                        # Get matching data
                        matched_data = pd.merge(
                            humans[human_mask][match_columns + ["response"]],
                            group[match_columns + ["response"]],
                            on=match_columns,
                            suffixes=("_human", "_llm"),
                        )

                        # Drop rows with NaN values
                        matched_data = matched_data.dropna(
                            subset=["response_human", "response_llm"]
                        )

                        if (
                            len(matched_data) >= 2
                        ):  # Need at least 2 points for regression
                            # Fit regression
                            X = matched_data["response_human"].values.reshape(-1, 1)
                            y = matched_data["response_llm"].values
                            model = LinearRegression()
                            model.fit(X, y)

                            # Calculate additional metrics
                            predictions = model.predict(X)
                            mse = np.mean((y - predictions) ** 2)
                            rmse = np.sqrt(mse)

                            result = {
                                "LLM": llm,
                                "Temperature": temp,
                                "R2": model.score(X, y),
                                "Slope": model.coef_[0],
                                "Intercept": model.intercept_,
                                "MSE": mse,
                                "RMSE": rmse,
                                "N": len(matched_data),
                            }

                            # Add grouping information
                            if isinstance(name, tuple):
                                for col, val in zip(groupby_cols, name):
                                    result[col] = val
                            else:
                                result[groupby_cols[0]] = name

                            results.append(result)
                else:
                    # Overall regression for this temperature
                    matched_data = pd.merge(
                        humans[match_columns + ["response"]],
                        temp_data[match_columns + ["response"]],
                        on=match_columns,
                        suffixes=("_human", "_llm"),
                    )

                    # Drop rows with NaN values
                    matched_data = matched_data.dropna(
                        subset=["response_human", "response_llm"]
                    )

                    if len(matched_data) >= 2:
                        X = matched_data["response_human"].values.reshape(-1, 1)
                        y = matched_data["response_llm"].values
                        model = LinearRegression()
                        model.fit(X, y)

                        # Calculate additional metrics
                        predictions = model.predict(X)
                        mse = np.mean((y - predictions) ** 2)
                        rmse = np.sqrt(mse)

                        results.append(
                            {
                                "LLM": llm,
                                "Temperature": temp,
                                "R2": model.score(X, y),
                                "Slope": model.coef_[0],
                                "Intercept": model.intercept_,
                                "MSE": mse,
                                "RMSE": rmse,
                                "N": len(matched_data),
                            }
                        )

        return pd.DataFrame(results)

    def plot_regression_results(self, regression_results):
        """Visualize regression analysis results across temperatures"""
        fig = plt.figure(figsize=(15, 12))
        gs = plt.GridSpec(3, 2)

        # 1. R² vs Temperature
        ax1 = fig.add_subplot(gs[0, 0])
        sns.scatterplot(
            data=regression_results, x="Temperature", y="R2", hue="LLM", s=100, ax=ax1
        )
        ax1.set_title("R² vs Temperature")
        ax1.set_ytask("R² Score")

        # 2. RMSE vs Temperature
        ax2 = fig.add_subplot(gs[0, 1])
        sns.scatterplot(
            data=regression_results, x="Temperature", y="RMSE", hue="LLM", s=100, ax=ax2
        )
        ax2.set_title("RMSE vs Temperature")
        ax2.set_ytask("Root Mean Square Error")

        # 3. Slope vs Temperature
        ax3 = fig.add_subplot(gs[1, 0])
        sns.scatterplot(
            data=regression_results,
            x="Temperature",
            y="Slope",
            hue="LLM",
            s=100,
            ax=ax3,
        )
        ax3.set_title("Regression Slope vs Temperature")
        ax3.set_ytask("Slope")
        ax3.axhline(1.0, color="r", linestyle="--", task="y=x line")
        ax3.legend()

        # 4. Intercept vs Temperature
        ax4 = fig.add_subplot(gs[1, 1])
        sns.scatterplot(
            data=regression_results,
            x="Temperature",
            y="Intercept",
            hue="LLM",
            s=100,
            ax=ax4,
        )
        ax4.set_title("Regression Intercept vs Temperature")
        ax4.set_ytask("Intercept")
        ax4.axhline(0, color="r", linestyle="--", task="y=x line")
        ax4.legend()

        # 5. Sample size vs Temperature
        ax5 = fig.add_subplot(gs[2, :])
        sns.barplot(data=regression_results, x="Temperature", y="N", hue="LLM", ax=ax5)
        ax5.set_title("Sample Size by Temperature")
        ax5.set_ytask("Number of Matched Responses")

        plt.tight_layout()
        return fig

    def plot_regression_comparison(
        self,
        regression_results,
        selected_temp=None,
        show_intervals=True,
        confidence_level=0.95,
    ):
        """Plot regression lines with confidence and prediction intervals

        Parameters:
        regression_results: DataFrame with regression results
        selected_temp: Optional temperature to filter by
        show_intervals: Whether to show confidence and prediction intervals
        confidence_level: Confidence level for intervals (default 0.95 for 95%)
        """
        humans = self.data[self.data["agent"] == "humans"].copy()

        plt.figure(figsize=(12, 8))

        # Plot diagonal line for reference
        plt.plot([0, 100], [0, 100], "k--", alpha=0.3, task="Perfect agreement")

        # Create color map for different LLMs
        unique_llms = regression_results["LLM"].unique()
        colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(unique_llms)))
        color_dict = dict(zip(unique_llms, colors))

        for llm in unique_llms:
            llm_results = regression_results[regression_results["LLM"] == llm]

            if selected_temp is not None:
                llm_results = llm_results[llm_results["Temperature"] == selected_temp]

            for _, row in llm_results.iterrows():
                # Get the corresponding data used to fit this regression
                llm_data = self.data[
                    (self.data["agent"] == llm)
                    & (self.data["temperature"] == row["Temperature"])
                ]

                # Match with human data
                matched_data = pd.merge(
                    humans[["domain", "ppp", "task", "response"]],
                    llm_data[["domain", "ppp", "task", "response"]],
                    on=["domain", "ppp", "task"],
                    suffixes=("_human", "_llm"),
                ).dropna()

                if len(matched_data) >= 2:  # Need at least 2 points for intervals
                    # Generate points for regression line
                    X = np.array([0, 100]).reshape(-1, 1)
                    y = row["Slope"] * X + row["Intercept"]

                    # Plot regression line
                    task = f"{llm} (T={row['Temperature']})"
                    plt.plot(X, y, "-", color=color_dict[llm], task=task)

                    if show_intervals:
                        # Compute intervals
                        X_fit = matched_data["response_human"].values.reshape(-1, 1)
                        y_fit = matched_data["response_llm"].values

                        # Mean squared error
                        y_pred = row["Slope"] * X_fit + row["Intercept"]
                        mse = np.sum((y_fit - y_pred.flatten()) ** 2) / (len(y_fit) - 2)

                        # Generate points for interval calculation
                        X_pred = np.linspace(0, 100, 100).reshape(-1, 1)
                        y_pred = row["Slope"] * X_pred + row["Intercept"]

                        # Standard errors
                        X_mean = np.mean(X_fit)
                        X_std = np.std(X_fit)

                        if X_std > 0:
                            se_mean = np.sqrt(
                                mse
                                * (
                                    1 / len(X_fit)
                                    + (X_pred - X_mean) ** 2 / (len(X_fit) * X_std**2)
                                )
                            )
                            se_pred = np.sqrt(
                                mse
                                * (
                                    1
                                    + 1 / len(X_fit)
                                    + (X_pred - X_mean) ** 2 / (len(X_fit) * X_std**2)
                                )
                            )

                            # Critical value
                            t_crit = stats.t.ppf(
                                (1 + confidence_level) / 2, len(X_fit) - 2
                            )

                            # Plot confidence interval
                            plt.fill_between(
                                X_pred.flatten(),
                                y_pred.flatten() - t_crit * se_mean.flatten(),
                                y_pred.flatten() + t_crit * se_mean.flatten(),
                                alpha=0.1,
                                color=color_dict[llm],
                                task=f"{task} CI",
                            )

                            # Plot prediction interval
                            plt.fill_between(
                                X_pred.flatten(),
                                y_pred.flatten() - t_crit * se_pred.flatten(),
                                y_pred.flatten() + t_crit * se_pred.flatten(),
                                alpha=0.05,
                                color=color_dict[llm],
                                task=f"{task} PI",
                            )

                    # Plot actual data points
                    plt.scatter(
                        matched_data["response_human"],
                        matched_data["response_llm"],
                        alpha=0.3,
                        color=color_dict[llm],
                        s=30,
                    )

        plt.xtask("Human Responses")
        plt.ytask("LLM Responses")
        plt.title(
            "Regression Comparison"
            + (f" (Temperature = {selected_temp})" if selected_temp is not None else "")
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)

        return plt.gcf()

    #######################
    # correlation and regression analysis by TASK

    def regression_analysis_by_task(self, by_domain=False):
        """Perform regression analysis by task instead of temperature"""
        humans = self.data[self.data["agent"] == "humans"].copy()
        results = []

        match_columns = ["domain", "ppp", "task"]

        for llm in [s for s in self.data["agent"].unique() if s != "humans"]:
            llm_data = self.data[self.data["agent"] == llm].copy()

            if by_domain:
                domains = self.data["domain"].unique()
            else:
                domains = [None]

            for domain in domains:
                domain_filter = (
                    (lambda x: x)
                    if domain is None
                    else (lambda x: x[x["domain"] == domain])
                )

                for task in sorted(self.data["task"].unique()):
                    task_humans = domain_filter(humans[humans["task"] == task])
                    task_llm = domain_filter(llm_data[llm_data["task"] == task])

                    # Aggregate over temperatures by averaging responses
                    task_llm = (
                        task_llm.groupby(match_columns)["response"].mean().reset_index()
                    )

                    matched_data = pd.merge(
                        task_humans[match_columns + ["response"]],
                        task_llm[match_columns + ["response"]],
                        on=match_columns,
                        suffixes=("_human", "_llm"),
                    )

                    if len(matched_data) >= 2:
                        X = matched_data["response_human"].values.reshape(-1, 1)
                        y = matched_data["response_llm"].values
                        model = LinearRegression()
                        model.fit(X, y)

                        predictions = model.predict(X)
                        mse = np.mean((y - predictions) ** 2)
                        rmse = np.sqrt(mse)

                        result = {
                            "LLM": llm,
                            "Task": task,
                            "R2": model.score(X, y),
                            "Slope": model.coef_[0],
                            "Intercept": model.intercept_,
                            "MSE": mse,
                            "RMSE": rmse,
                            "N": len(matched_data),
                        }

                        if domain is not None:
                            result["domain"] = domain

                        results.append(result)

        return pd.DataFrame(results)

    def analyze_llm_human_correlation_by_task(self, by_domain=False, epsilon=1e-6):
        """Analyze correlation between human and LLM responses by task"""
        humans = self.data[self.data["agent"] == "humans"].copy()
        results = []

        match_columns = ["domain", "ppp", "task"]

        for llm in [s for s in self.data["agent"].unique() if s != "humans"]:
            llm_data = self.data[self.data["agent"] == llm].copy()

            if by_domain:
                domains = self.data["domain"].unique()
            else:
                domains = [None]

            for domain in domains:
                domain_filter = (
                    (lambda x: x)
                    if domain is None
                    else (lambda x: x[x["domain"] == domain])
                )

                for task in sorted(self.data["task"].unique()):
                    task_humans = domain_filter(humans[humans["task"] == task])
                    task_llm = domain_filter(llm_data[llm_data["task"] == task])

                    # Aggregate over temperatures
                    task_llm = (
                        task_llm.groupby(match_columns)["response"].mean().reset_index()
                    )

                    matched_data = pd.merge(
                        task_humans[match_columns + ["response"]],
                        task_llm[match_columns + ["response"]],
                        on=match_columns,
                        suffixes=("_human", "_llm"),
                    )

                    if len(matched_data) >= 3:
                        _, h_pval = stats.shapiro(matched_data["response_human"])
                        _, l_pval = stats.shapiro(matched_data["response_llm"])
                        is_normal = h_pval > 0.05 and l_pval > 0.05
                    else:
                        is_normal = False

                    # Add small epsilon if responses are all 0.0
                    if matched_data["response_llm"].nunique() == 1 and matched_data[
                        "response_llm"
                    ].iloc[0] in [0.0, 1.0]:
                        matched_data["response_llm"] += epsilon

                    if matched_data["response_human"].nunique() == 1 and matched_data[
                        "response_human"
                    ].iloc[0] in [0.0, 1.0]:
                        matched_data["response_human"] += epsilon

                    if is_normal:
                        corr, p = stats.pearsonr(
                            matched_data["response_human"], matched_data["response_llm"]
                        )
                        method = "pearson"
                    else:
                        corr, p = stats.spearmanr(
                            matched_data["response_human"], matched_data["response_llm"]
                        )
                        method = "spearman"

                    result = {
                        "LLM": llm,
                        "Task": task,
                        "Correlation": corr,
                        "P_value": p,
                        "Method": method,
                        "N": len(matched_data),
                    }

                    if domain is not None:
                        result["domain"] = domain

                    results.append(result)

        return pd.DataFrame(results)

    def export_comprehensive_task_analysis(self, reg_results, corr_results, filename):
        """Export comprehensive analysis with task-based results"""
        with pd.ExcelWriter(filename) as writer:
            # Task Performance Summary
            task_summary = (
                reg_results.groupby(["Task", "LLM"])
                .agg({"R2": ["mean", "std"], "RMSE": ["mean", "std"], "N": "sum"})
                .round(3)
            )
            task_summary.to_excel(writer, sheet_name="Task Performance Summary")

            # Task Detailed Metrics
            reg_results.to_excel(
                writer, sheet_name="Task Detailed Metrics", index=False
            )

            # Domain Comparison
            if "domain" in reg_results.columns:
                domain_comparison = (
                    reg_results.groupby(["domain", "LLM"])
                    .agg({"R2": ["mean", "std"], "RMSE": ["mean", "std"], "N": "sum"})
                    .round(3)
                )
                domain_comparison.to_excel(writer, sheet_name="Domain Comparison")

            # Correlation by Domain
            if "domain" in corr_results.columns:
                corr_results.groupby(["domain", "LLM"]).agg(
                    {
                        "Correlation": ["mean", "std", "min", "max"],
                        "P_value": ["mean", "min", "max"],
                        "N": "sum",
                    }
                ).round(3).to_excel(writer, sheet_name="Correlation by Domain")

            # Correlation Aggregate
            corr_results.groupby("LLM").agg(
                {
                    "Correlation": ["mean", "std", "min", "max"],
                    "P_value": ["mean", "min", "max"],
                    "N": "sum",
                }
            ).round(3).to_excel(writer, sheet_name="Correlation Aggregate")

    #####################

    def plot_correlation_analysis(self, correlation_results):
        """Visualize correlation analysis results"""
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2)

        # 1. Overall correlations by LLM
        ax1 = fig.add_subplot(gs[0, 0])
        overall_corr = correlation_results[~correlation_results.index.isnull()]
        sns.barplot(data=overall_corr, x="LLM", y="Correlation", ax=ax1)
        ax1.set_title("Overall Correlations by LLM")
        ax1.set_ytask("Correlation Coefficient")

        # 2. Correlation significance
        ax2 = fig.add_subplot(gs[0, 1])
        sig_plot = -np.log10(correlation_results["P_value"])
        sns.barplot(data=correlation_results, x="LLM", y=sig_plot, ax=ax2)
        ax2.set_title("Correlation Significance (-log10 p-value)")
        ax2.axhline(-np.log10(0.05), color="r", linestyle="--", task="p=0.05")
        ax2.legend()

        # 3. Correlation heatmap (if domain/task grouping exists)
        if (
            "domain" in correlation_results.columns
            or "task" in correlation_results.columns
        ):
            ax3 = fig.add_subplot(gs[1, :])
            pivot_cols = [
                col
                for col in ["domain", "task", "ppp"]
                if col in correlation_results.columns
            ]
            if pivot_cols:
                pivot_table = correlation_results.pivot_table(
                    index="LLM",
                    columns=pivot_cols[0] if len(pivot_cols) == 1 else pivot_cols,
                    values="Correlation",
                )
                sns.heatmap(pivot_table, annot=True, cmap="RdYlBu", center=0, ax=ax3)
                ax3.set_title("Correlation Patterns")

        plt.tight_layout()
        return fig

    def plot_regression_analysis(self, regression_results, include_temp=True):
        """Visualize regression analysis results"""
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2)

        # 1. R² values by LLM
        ax1 = fig.add_subplot(gs[0, 0])
        sns.barplot(data=regression_results, x="LLM", y="R2", ax=ax1)
        ax1.set_title("R² Values by LLM")
        ax1.set_ytask("R²")

        # 2. Slope and intercept
        ax2 = fig.add_subplot(gs[0, 1])
        regression_results.plot(kind="scatter", x="Slope", y="Intercept", ax=ax2)
        for _, row in regression_results.iterrows():
            ax2.annotate(row["LLM"], (row["Slope"], row["Intercept"]))
        ax2.set_title("Regression Parameters")

        # 3. Temperature effect if available
        if include_temp and "Temperature" in regression_results.columns:
            ax3 = fig.add_subplot(gs[1, :])
            sns.scatterplot(
                data=regression_results,
                x="Temperature",
                y="R2",
                hue="LLM",
                style="LLM",
                s=100,
                ax=ax3,
            )
            ax3.set_title("Temperature Effect on Model Fit")

        plt.tight_layout()
        return fig
        #######################

        ## print the results

    def create_analysis_summary(self, correlation_results, regression_results):
        """Create comprehensive summary tables with interpretations for correlation and regression analyses"""

        # Correlation Summary
        corr_metrics = pd.DataFrame(
            {
                "Metric": ["Correlation Coefficient", "P-value", "Method", "N"],
                "Description": [
                    "Measure of strength and direction of relationship between human and LLM responses. range: -1 to +1",
                    "Statistical significance of correlation. Values < 0.05 indicate significant correlation",
                    "Pearson (linear relationship) or Spearman (monotonic relationship)",
                    "Number of matched response pairs analyzed",
                ],
                "Interpretation Guide": [
                    "1.0: Perfect positive correlation\n0.7-0.9: Strong positive\n0.4-0.6: Moderate\n0.1-0.3: Weak\n0: No correlation",
                    "p < 0.05: Statistically significant\np < 0.01: Highly significant\np < 0.001: Very highly significant",
                    "Pearson used when data is normal, Spearman when non-normal or ordinal",
                    "Larger N gives more reliable results. N > 30 preferred for robust analysis",
                ],
            }
        )

        # Regression Summary
        reg_metrics = pd.DataFrame(
            {
                "Metric": ["R²", "Slope", "Intercept", "MSE", "RMSE", "N"],
                "Description": [
                    "Proportion of variance in LLM responses explained by human responses. range: 0 to 1",
                    "Change in LLM response for one unit change in human response",
                    "Predicted LLM response when human response is 0",
                    "Mean Squared Error - average squared difference between predictions and actual values",
                    "Root Mean Square Error - standard deviation of prediction errors",
                    "Number of matched response pairs analyzed",
                ],
                "Interpretation Guide": [
                    "R² > 0.8: Very good fit\n0.6-0.8: Good fit\n0.4-0.6: Moderate fit\n< 0.4: Poor fit",
                    "Slope = 1: Perfect alignment\n> 1: LLM amplifies human judgments\n< 1: LLM dampens human judgments",
                    "Large intercept indicates systematic bias in LLM responses",
                    "Smaller values indicate better fit. Scale depends on response range",
                    "In same units as responses. Smaller values indicate better fit",
                    "Larger N gives more reliable results. N > 30 preferred for robust analysis",
                ],
            }
        )

        # Create summary statistics tables
        # Correlation stats - including Method
        corr_stats = (
            correlation_results.groupby(["LLM", "Temperature", "Method"])
            .agg(
                {
                    "Correlation": ["mean", "std", "min", "max"],
                    "P_value": ["mean", "min", "max"],
                    "N": ["mean", "sum"],
                }
            )
            .round(3)
        )

        # Regression stats - without Method
        reg_stats = (
            regression_results.groupby(["LLM", "Temperature"])
            .agg(
                {
                    "R2": ["mean", "std", "min", "max"],
                    "Slope": ["mean", "std"],
                    "Intercept": ["mean", "std"],
                    "RMSE": ["mean", "std"],
                    "N": ["mean", "sum"],
                }
            )
            .round(3)
        )

        return {
            "correlation_metrics": corr_metrics,
            "regression_metrics": reg_metrics,
            "correlation_stats": corr_stats,
            "regression_stats": reg_stats,
        }

    def print_analysis_summary(self, summary_dict):
        """Print formatted analysis summary"""
        print("\n=== Correlation Analysis Metrics ===")
        print("These metrics show how well LLM and human responses correlate:")
        print(summary_dict["correlation_metrics"].to_string(index=False))

        print("\n=== Correlation Statistics by LLM, Temperature, and Method ===")
        print("Grouped statistics showing correlation patterns:")
        print(summary_dict["correlation_stats"])

        print("\n=== Regression Analysis Metrics ===")
        print(
            "These metrics show how well LLM responses can be predicted from human responses:"
        )
        print(summary_dict["regression_metrics"].to_string(index=False))

        print("\n=== Regression Statistics by LLM and Temperature ===")
        print("Grouped statistics showing prediction accuracy:")
        print(summary_dict["regression_stats"])

    def export_analysis_summary(self, summary_dict, filename):
        """Export analysis summary to Excel with separate sheets"""
        with pd.ExcelWriter(filename) as writer:
            summary_dict["correlation_metrics"].to_excel(
                writer, sheet_name="Correlation Metrics", index=False
            )
            summary_dict["regression_metrics"].to_excel(
                writer, sheet_name="Regression Metrics", index=False
            )
            summary_dict["correlation_stats"].to_excel(
                writer, sheet_name="Correlation Stats"
            )
            summary_dict["regression_stats"].to_excel(
                writer, sheet_name="Regression Stats"
            )

    #### predictions with intervals:

    def calculate_predicted_responses_with_intervals(
        self, regression_results, human_values=None, confidence_level=0.95
    ):
        """
        Calculate predicted LLM responses with confidence and prediction intervals.

        Parameters:
        regression_results: DataFrame from regression_analysis()
        human_values: list of human response values to predict for (default: [0, 25, 50, 75, 100])
        confidence_level: confidence level for intervals (default: 0.95 for 95% confidence)

        Returns:
        DataFrame with predictions and intervals for each LLM, temperature, and human value
        """
        if human_values is None:
            human_values = [0, 25, 50, 75, 100]

        predictions = []

        # Get original data for computing intervals
        humans = self.data[self.data["agent"] == "humans"].copy()

        for _, row in regression_results.iterrows():
            # Get the corresponding data used to fit this regression
            llm_data = self.data[
                (self.data["agent"] == row["LLM"])
                & (self.data["temperature"] == row["Temperature"])
            ]

            # Create mask for any additional grouping factors
            mask = np.ones(len(humans), dtype=bool)
            if "domain" in row:
                mask &= humans["domain"] == row["domain"]
                mask_llm = llm_data["domain"] == row["domain"]
                llm_data = llm_data[mask_llm]
            if "ppp" in row:
                mask &= humans["ppp"] == row["ppp"]
                mask_llm = llm_data["ppp"] == row["ppp"]
                llm_data = llm_data[mask_llm]

            humans_subset = humans[mask]

            # Match responses
            matched_data = pd.merge(
                humans_subset[["domain", "ppp", "task", "response"]],
                llm_data[["domain", "ppp", "task", "response"]],
                on=["domain", "ppp", "task"],
                suffixes=("_human", "_llm"),
            ).dropna()

            if len(matched_data) >= 2:  # Need at least 2 points for intervals
                # Fit regression model
                X = matched_data["response_human"].values.reshape(-1, 1)
                y = matched_data["response_llm"].values
                model = LinearRegression()
                model.fit(X, y)

                # Calculate intervals for each human value
                X_pred = np.array(human_values).reshape(-1, 1)

                # Mean squared error for prediction interval
                y_pred = model.predict(X)
                mse = np.sum((y - y_pred) ** 2) / (len(y) - 2)

                # Standard error of the prediction
                X_mean = np.mean(X)
                X_std = np.std(X)

                for i, human_value in enumerate(human_values):
                    # Predicted value
                    predicted_value = model.predict([[human_value]])[0]

                    # Standard error of the mean prediction
                    if X_std == 0:
                        se_mean = 0
                        se_pred = np.sqrt(mse)
                    else:
                        se_mean = np.sqrt(
                            mse
                            * (
                                1 / len(X)
                                + (human_value - X_mean) ** 2 / (len(X) * X_std**2)
                            )
                        )
                        # Standard error of individual prediction
                        se_pred = np.sqrt(
                            mse
                            * (
                                1
                                + 1 / len(X)
                                + (human_value - X_mean) ** 2 / (len(X) * X_std**2)
                            )
                        )

                    # Critical value for desired confidence level
                    t_crit = stats.t.ppf((1 + confidence_level) / 2, len(X) - 2)

                    predictions.append(
                        {
                            "LLM": row["LLM"],
                            "Temperature": row["Temperature"],
                            "Human_Response": human_value,
                            "Predicted_LLM_Response": predicted_value,
                            "CI_Lower": predicted_value - t_crit * se_mean,
                            "CI_Upper": predicted_value + t_crit * se_mean,
                            "PI_Lower": predicted_value - t_crit * se_pred,
                            "PI_Upper": predicted_value + t_crit * se_pred,
                            "R2": row["R2"],
                            "RMSE": row["RMSE"],
                            "N": len(matched_data),
                        }
                    )

                    # Add additional grouping factors if present
                    if "domain" in row:
                        predictions[-1]["domain"] = row["domain"]
                    if "ppp" in row:
                        predictions[-1]["ppp"] = row["ppp"]

        return pd.DataFrame(predictions)

    def print_prediction_summary_with_intervals(
        self, predictions_df, confidence_level=0.95
    ):
        """Print formatted summary of predictions with confidence intervals"""
        print(
            f"\n=== Predicted LLM Responses with {confidence_level * 100}% Intervals ==="
        )
        print("CI: Confidence Interval for mean prediction")
        print("PI: Prediction Interval for individual predictions\n")

        # Group by LLM and Temperature
        for (llm, temp), group in predictions_df.groupby(["LLM", "Temperature"]):
            print(f"\nLLM: {llm}, Temperature: {temp}")
            print(
                f"Model fit: R² = {group['R2'].iloc[0]:.3f}, RMSE = {group['RMSE'].iloc[0]:.3f}"
            )
            print(f"Sample size: N = {group['N'].iloc[0]}")
            print("\nHuman   Predicted LLM Response")
            print("Response  (with intervals)")
            print("-" * 50)

            for _, row in group.iterrows():
                print(
                    f"{row['Human_Response']:>7.0f}   {row['Predicted_LLM_Response']:>6.1f}"
                )
                print(
                    f"         CI: [{row['CI_Lower']:>6.1f}, {row['CI_Upper']:>6.1f}]"
                )
                print(
                    f"         PI: [{row['PI_Lower']:>6.1f}, {row['PI_Upper']:>6.1f}]"
                )

        if "domain" in predictions_df.columns:
            print("\nPredictions by domain available in the DataFrame")

    def plot_predictions_with_intervals(
        self, predictions_df, llm=None, temperature=None
    ):
        """Plot predictions with confidence and prediction intervals"""
        plt.figure(figsize=(12, 8))

        # Filter data if LLM or temperature specified
        plot_data = predictions_df
        if llm is not None:
            plot_data = plot_data[plot_data["LLM"] == llm]
        if temperature is not None:
            plot_data = plot_data[plot_data["Temperature"] == temperature]

        # Plot for each LLM and temperature combination
        for (current_llm, temp), group in plot_data.groupby(["LLM", "Temperature"]):
            task = f"{current_llm} (T={temp})"

            # Plot predicted values
            plt.plot(
                group["Human_Response"],
                group["Predicted_LLM_Response"],
                "-",
                task=task,
            )

            # Plot confidence intervals
            plt.fill_between(
                group["Human_Response"],
                group["CI_Lower"],
                group["CI_Upper"],
                alpha=0.1,
                task=f"{task} CI",
            )

            # Plot prediction intervals
            plt.fill_between(
                group["Human_Response"],
                group["PI_Lower"],
                group["PI_Upper"],
                alpha=0.05,
                task=f"{task} PI",
            )

        # Plot diagonal line for reference
        plt.plot([0, 100], [0, 100], "k--", alpha=0.3, task="Perfect agreement")

        plt.xtask("Human Response")
        plt.ytask("Predicted LLM Response")
        plt.title("Predicted LLM Responses with Confidence and Prediction Intervals")
        plt.legend()
        plt.grid(True, alpha=0.3)

        return plt.gcf()

    def export_analysis_summary_with_intervals(
        self, summary_dict, predictions_df, filename
    ):
        """Export analysis summary to Excel including prediction intervals"""
        with pd.ExcelWriter(filename) as writer:
            # Original summary sheets
            summary_dict["correlation_metrics"].to_excel(
                writer, sheet_name="Correlation Metrics", index=False
            )
            summary_dict["regression_metrics"].to_excel(
                writer, sheet_name="Regression Metrics", index=False
            )
            summary_dict["correlation_stats"].to_excel(
                writer, sheet_name="Correlation Stats"
            )
            summary_dict["regression_stats"].to_excel(
                writer, sheet_name="Regression Stats"
            )

            # Add predictions with intervals
            predictions_pivot = (
                predictions_df.pivot_table(
                    index=["LLM", "Temperature", "Human_Response"],
                    values=[
                        "Predicted_LLM_Response",
                        "CI_Lower",
                        "CI_Upper",
                        "PI_Lower",
                        "PI_Upper",
                    ],
                    aggfunc="first",
                )
                .round(2)
                .reset_index()
            )

            predictions_pivot.to_excel(
                writer, sheet_name="Predictions with Intervals", index=False
            )

            # Add explanation sheet
            explanation = pd.DataFrame(
                {
                    "Term": [
                        "CI (Confidence Interval)",
                        "PI (Prediction Interval)",
                        "CI_Lower/Upper",
                        "PI_Lower/Upper",
                        "Interpretation Example",
                    ],
                    "Description": [
                        "Range where we expect the true mean LLM response to fall",
                        "Range where we expect individual LLM responses to fall",
                        "Lower/Upper bounds of the confidence interval",
                        "Lower/Upper bounds of the prediction interval",
                        "For a human response of 50, if CI=[45,55] and PI=[30,70], we are 95% confident that:\n"
                        + "- The true mean LLM response is between 45 and 55\n"
                        + "- Individual LLM responses will fall between 30 and 70",
                    ],
                }
            )
            explanation.to_excel(
                writer, sheet_name="Interval Explanations", index=False
            )

    ### add additional goodness of fit metrics

    def compute_regression_metrics(self, y_true, y_pred):
        """Compute comprehensive regression metrics"""
        residuals = y_true - y_pred
        n = len(y_true)
        p = 2  # number of parameters in linear regression (slope and intercept)

        # Basic metrics
        mae = np.mean(np.abs(residuals))
        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        r2 = 1 - np.sum(residuals**2) / np.sum((y_true - np.mean(y_true)) ** 2)

        # Adjusted R² accounts for number of predictors
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        # Additional metrics
        mape = (
            np.mean(np.abs(residuals / y_true)) * 100
        )  # Mean Absolute Percentage Error

        # Residual statistics
        residual_std = np.std(residuals)
        residual_skew = stats.skew(residuals)
        residual_kurtosis = stats.kurtosis(residuals)

        # Normality test of residuals
        _, residual_normality_p = stats.shapiro(residuals)

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Adjusted_R2": adjusted_r2,
            "MAPE": mape,
            "Residual_Std": residual_std,
            "Residual_Skew": residual_skew,
            "Residual_Kurtosis": residual_kurtosis,
            "Residual_Normality_p": residual_normality_p,
        }

    def analyze_task_specific_performance(
        self, regression_results, correlation_results
    ):
        """Analyze which LLM best fits human responses for each task, domain, and temperature"""
        humans = self.data[self.data["agent"] == "humans"].copy()
        task_results = []

        # Define matching columns
        match_columns = ["domain", "ppp", "task"]

        for domain in self.data["domain"].unique():
            for task in self.data["task"].unique():
                task_metrics = []

                # Get human data for this domain and task
                human_task_data = humans[
                    (humans["domain"] == domain) & (humans["task"] == task)
                ]

                for llm in [s for s in self.data["agent"].unique() if s != "humans"]:
                    # Get all temperature values for this LLM
                    llm_temps = self.data[(self.data["agent"] == llm)][
                        "temperature"
                    ].unique()

                    # Analyze for each temperature separately
                    for temp in llm_temps:
                        llm_data = self.data[
                            (self.data["agent"] == llm)
                            & (self.data["domain"] == domain)
                            & (self.data["task"] == task)
                            & (self.data["temperature"] == temp)
                        ]

                        if len(llm_data) > 0:
                            # Match human and LLM data
                            matched_data = pd.merge(
                                human_task_data[match_columns + ["response"]],
                                llm_data[match_columns + ["response"]],
                                on=match_columns,
                                suffixes=("_human", "_llm"),
                            ).dropna()

                            if (
                                len(matched_data) >= 2
                            ):  # Need at least 2 points for regression
                                # Perform regression
                                X = matched_data["response_human"].values.reshape(-1, 1)
                                y = matched_data["response_llm"].values
                                model = LinearRegression()
                                model.fit(X, y)
                                y_pred = model.predict(X)

                                # Calculate metrics
                                metrics = self.compute_regression_metrics(y, y_pred)

                                # Calculate correlation
                                corr, p = spearmanr(
                                    matched_data["response_human"],
                                    matched_data["response_llm"],
                                )

                                task_metrics.append(
                                    {
                                        "LLM": llm,
                                        "Domain": domain,
                                        "Task": task,
                                        "Temperature": temp,
                                        "Correlation": corr,
                                        "Correlation_p": p,
                                        "N_samples": len(matched_data),
                                        **metrics,
                                    }
                                )

                if task_metrics:
                    task_df = pd.DataFrame(task_metrics)

                    # Find best performing LLM for each temperature
                    for temp in task_df["Temperature"].unique():
                        temp_df = task_df[task_df["Temperature"] == temp]

                        # Skip if no data for this temperature
                        if len(temp_df) == 0:
                            continue

                        best_by_r2 = temp_df.loc[temp_df["R2"].idxmax()]
                        best_by_rmse = temp_df.loc[temp_df["RMSE"].idxmin()]
                        best_by_corr = temp_df.loc[temp_df["Correlation"].idxmax()]

                        task_results.append(
                            {
                                "Domain": domain,
                                "Task": task,
                                "Temperature": temp,
                                "Best_LLM_R2": best_by_r2["LLM"],
                                "Best_R2": best_by_r2["R2"],
                                "Best_LLM_RMSE": best_by_rmse["LLM"],
                                "Best_RMSE": best_by_rmse["RMSE"],
                                "Best_LLM_Corr": best_by_corr["LLM"],
                                "Best_Correlation": best_by_corr["Correlation"],
                                "N_samples": best_by_r2["N_samples"],
                                "All_Metrics": task_df[
                                    task_df["Temperature"] == temp
                                ].to_dict("records"),
                            }
                        )

        return pd.DataFrame(task_results)

    ##### more details on best llm ranking

    def _is_metric_higher_better(self, metric):
        """
        Determine if higher values indicate better performance for a given metric.

        Parameters:
        metric (str): Name of the metric

        Returns:
        bool: True if higher values are better, False if lower values are better
        """
        # Metrics where higher values indicate better performance
        higher_better_metrics = {
            "R2",
            "Adjusted_R2",
            "Correlation",
            "N_samples",  # More samples is generally better
        }

        # Metrics where lower values indicate better performance
        lower_better_metrics = {
            "RMSE",
            "MSE",
            "MAE",
            "MAPE",
            "Residual_Std",
            "P_value",  # Lower p-values indicate stronger evidence
        }

        if metric in higher_better_metrics:
            return True
        elif metric in lower_better_metrics:
            return False
        else:
            # For unknown metrics, raise an error
            raise ValueError(
                f"Unknown metric: {metric}. Please specify whether higher or lower values are better."
            )

    def determine_best_llm_with_significance(
        self,
        metrics_df,
        metric_col,
        by_temperature=True,
        by_domain=True,
        by_task=True,
        alpha=0.05,
        min_effect_size=0.2,
    ):
        """
        Determine best performing LLM with statistical significance, rankings, and aggregation levels.

        Parameters:
        metrics_df: DataFrame with metrics for each LLM
        metric_col: Column name of the metric to compare
        by_temperature: Whether to analyze per temperature setting
        by_domain: Whether to analyze per domain
        by_task: Whether to analyze per task
        alpha: Significance level for statistical tests
        min_effect_size: Minimum Cohen's d effect size to consider difference meaningful

        Returns:
        dict with analysis results at different aggregation levels
        """

        def analyze_group(group_df, group_name=""):
            llms = group_df["LLM"].unique()
            if len(llms) < 2:
                return {
                    "group": group_name,
                    "rankings": [
                        {
                            "llm": llms[0],
                            "rank": 1,
                            "mean_performance": group_df[metric_col].mean(),
                        }
                    ],
                    "significance_matrix": None,
                    "effect_size_matrix": None,
                    "n_samples": len(group_df),
                }

            # Calculate mean performance for each LLM
            mean_performance = group_df.groupby("LLM")[metric_col].mean()

            # Determine if higher values are better
            higher_better = self._is_metric_higher_better(metric_col)

            # Create rankings
            sorted_llms = mean_performance.sort_values(ascending=not higher_better)
            rankings = [
                {"llm": llm, "rank": i + 1, "mean_performance": perf}
                for i, (llm, perf) in enumerate(sorted_llms.items())
            ]

            # Pairwise comparisons
            n_llms = len(llms)
            sig_matrix = np.zeros((n_llms, n_llms))
            effect_matrix = np.zeros((n_llms, n_llms))

            for i, llm1 in enumerate(llms):
                for j, llm2 in enumerate(llms):
                    if i != j:
                        data1 = group_df[group_df["LLM"] == llm1][metric_col]
                        data2 = group_df[group_df["LLM"] == llm2][metric_col]

                        # Statistical test (Mann-Whitney U test)
                        alternative = "greater" if higher_better else "less"
                        statistic, p_value = stats.mannwhitneyu(
                            data1, data2, alternative=alternative
                        )

                        # Effect size (Cohen's d)
                        d = (data1.mean() - data2.mean()) / np.sqrt(
                            (data1.var() + data2.var()) / 2
                        )
                        if not higher_better:
                            d = -d

                        sig_matrix[i, j] = p_value < alpha
                        effect_matrix[i, j] = abs(d)

            return {
                "group": group_name,
                "rankings": rankings,
                "significance_matrix": pd.DataFrame(
                    sig_matrix, index=llms, columns=llms
                ),
                "effect_size_matrix": pd.DataFrame(
                    effect_matrix, index=llms, columns=llms
                ),
                "n_samples": len(group_df),
            }

        results = {"overall": analyze_group(metrics_df, "Overall")}

        # Analysis by temperature
        if by_temperature:
            temp_results = {}
            for temp in metrics_df["Temperature"].unique():
                temp_df = metrics_df[metrics_df["Temperature"] == temp]
                temp_results[temp] = analyze_group(temp_df, f"Temperature_{temp}")
            results["by_temperature"] = temp_results

        # Analysis by domain
        if by_domain and "Domain" in metrics_df.columns:
            domain_results = {}
            for domain in metrics_df["Domain"].unique():
                domain_df = metrics_df[metrics_df["Domain"] == domain]
                domain_results[domain] = analyze_group(domain_df, f"Domain_{domain}")

                # Domain x Temperature
                if by_temperature:
                    temp_domain_results = {}
                    for temp in metrics_df["Temperature"].unique():
                        temp_domain_df = domain_df[domain_df["Temperature"] == temp]
                        temp_domain_results[temp] = analyze_group(
                            temp_domain_df, f"Domain_{domain}_Temp_{temp}"
                        )
                    domain_results[f"{domain}_by_temperature"] = temp_domain_results
            results["by_domain"] = domain_results

        # Analysis by task
        if by_task and "Task" in metrics_df.columns:
            task_results = {}
            for task in metrics_df["Task"].unique():
                task_df = metrics_df[metrics_df["Task"] == task]
                task_results[task] = analyze_group(task_df, f"Task_{task}")

                # Task x Temperature
                if by_temperature:
                    temp_task_results = {}
                    for temp in metrics_df["Temperature"].unique():
                        temp_task_df = task_df[task_df["Temperature"] == temp]
                        temp_task_results[temp] = analyze_group(
                            temp_task_df, f"Task_{task}_Temp_{temp}"
                        )
                    task_results[f"{task}_by_temperature"] = temp_task_results
            results["by_task"] = task_results

        return results

    def format_ranking_results(self, ranking_results):
        """Format ranking results into a pandas DataFrame for easy viewing and export"""
        formatted_results = []

        def process_group_results(
            results, group_name, temp=None, domain=None, task=None
        ):
            for rank_info in results["rankings"]:
                row = {
                    "Group": group_name,
                    "Temperature": temp,
                    "Domain": domain,
                    "Task": task,
                    "LLM": rank_info["llm"],
                    "Rank": rank_info["rank"],
                    "Mean_Performance": rank_info["mean_performance"],
                    "N_Samples": results["n_samples"],
                }
                formatted_results.append(row)

        # Process overall results
        process_group_results(ranking_results["overall"], "Overall")

        # Process temperature results
        if "by_temperature" in ranking_results:
            for temp, temp_results in ranking_results["by_temperature"].items():
                process_group_results(temp_results, "Temperature", temp=temp)

        # Process domain results
        if "by_domain" in ranking_results:
            for domain, domain_results in ranking_results["by_domain"].items():
                if not domain.endswith("_by_temperature"):
                    process_group_results(domain_results, "Domain", domain=domain)

                # Process domain x temperature results
                if domain.endswith("_by_temperature"):
                    base_domain = domain.replace("_by_temperature", "")
                    for temp, temp_results in domain_results.items():
                        process_group_results(
                            temp_results,
                            "Domain_Temperature",
                            temp=temp,
                            domain=base_domain,
                        )

        # Process task results
        if "by_task" in ranking_results:
            for task, task_results in ranking_results["by_task"].items():
                if not task.endswith("_by_temperature"):
                    process_group_results(task_results, "Task", task=task)

                # Process task x temperature results
                if task.endswith("_by_temperature"):
                    base_task = task.replace("_by_temperature", "")
                    for temp, temp_results in task_results.items():
                        process_group_results(
                            temp_results, "Task_Temperature", temp=temp, task=base_task
                        )

        return pd.DataFrame(formatted_results)

    #### multi-level ranking anlysis of LLMs

    def analyze_multilevel_rankings(
        self,
        metrics_df,
        metric_col,
        by_domain=True,
        by_temperature=True,
        by_task=True,
        domain_x_temp=True,
        domain_x_task=True,
        task_x_temp=True,
        alpha=0.05,
        min_effect_size=0.2,
    ):
        """
        Analyze LLM rankings with configurable comparison levels
        """

        def run_statistical_test(data1, data2, higher_better):
            alternative = "greater" if higher_better else "less"
            statistic, p_value = stats.mannwhitneyu(
                data1, data2, alternative=alternative
            )
            d = (data1.mean() - data2.mean()) / np.sqrt((data1.var() + data2.var()) / 2)
            return p_value, abs(d)

        def analyze_group(group_df, conditions=None):
            if conditions is None:
                conditions = {}

            llms = group_df["LLM"].unique()
            if len(llms) < 2:
                return None

            higher_better = self._is_metric_higher_better(metric_col)
            mean_perf = group_df.groupby("LLM")[metric_col].mean()
            sorted_llms = mean_perf.sort_values(ascending=not higher_better)

            rankings = []
            comparisons = []

            for i, llm1 in enumerate(sorted_llms.index):
                data1 = group_df[group_df["LLM"] == llm1][metric_col]

                for llm2 in sorted_llms.index[i + 1 :]:
                    data2 = group_df[group_df["LLM"] == llm2][metric_col]
                    p_value, effect_size = run_statistical_test(
                        data1, data2, higher_better
                    )

                    comparisons.append(
                        {
                            "LLM1": llm1,
                            "LLM2": llm2,
                            "P_Value": p_value,
                            "Effect_Size": effect_size,
                            "Significant": p_value < alpha,
                            "Meaningful": effect_size > min_effect_size,
                            "Test": "Mann-Whitney U",
                            **conditions,
                        }
                    )

                rankings.append(
                    {
                        "LLM": llm1,
                        "Rank": i + 1,
                        "Mean_Performance": mean_perf[llm1],
                        **conditions,
                    }
                )

            return rankings, comparisons

        results = {"rankings": [], "comparisons": [], "analysis_levels": []}

        # Overall analysis
        overall_rankings, overall_comparisons = analyze_group(
            metrics_df, {"Level": "overall"}
        )
        if overall_rankings:
            results["rankings"].extend(overall_rankings)
            results["comparisons"].extend(overall_comparisons)
            results["analysis_levels"].append("overall")

        # Temperature analysis
        if by_temperature:
            for temp in metrics_df["Temperature"].unique():
                temp_df = metrics_df[metrics_df["Temperature"] == temp]
                rankings, comparisons = analyze_group(
                    temp_df, {"Temperature": temp, "Level": "temperature"}
                )
                if rankings:
                    results["rankings"].extend(rankings)
                    results["comparisons"].extend(comparisons)
                    if "temperature" not in results["analysis_levels"]:
                        results["analysis_levels"].append("temperature")

        # Domain analysis
        if by_domain and "Domain" in metrics_df.columns:
            for domain in metrics_df["Domain"].unique():
                domain_df = metrics_df[metrics_df["Domain"] == domain]
                rankings, comparisons = analyze_group(
                    domain_df, {"Domain": domain, "Level": "domain"}
                )
                if rankings:
                    results["rankings"].extend(rankings)
                    results["comparisons"].extend(comparisons)
                    if "domain" not in results["analysis_levels"]:
                        results["analysis_levels"].append("domain")

                # Domain x Temperature
                if domain_x_temp and by_temperature:
                    for temp in metrics_df["Temperature"].unique():
                        temp_domain_df = domain_df[domain_df["Temperature"] == temp]
                        rankings, comparisons = analyze_group(
                            temp_domain_df,
                            {
                                "Domain": domain,
                                "Temperature": temp,
                                "Level": "domain_temperature",
                            },
                        )
                        if rankings:
                            results["rankings"].extend(rankings)
                            results["comparisons"].extend(comparisons)
                            if "domain_temperature" not in results["analysis_levels"]:
                                results["analysis_levels"].append("domain_temperature")

        # Task analysis
        if by_task and "Task" in metrics_df.columns:
            for task in metrics_df["Task"].unique():
                task_df = metrics_df[metrics_df["Task"] == task]
                rankings, comparisons = analyze_group(
                    task_df, {"Task": task, "Level": "task"}
                )
                if rankings:
                    results["rankings"].extend(rankings)
                    results["comparisons"].extend(comparisons)
                    if "task" not in results["analysis_levels"]:
                        results["analysis_levels"].append("task")

                # Task x Temperature
                if task_x_temp and by_temperature:
                    for temp in metrics_df["Temperature"].unique():
                        temp_task_df = task_df[task_df["Temperature"] == temp]
                        rankings, comparisons = analyze_group(
                            temp_task_df,
                            {
                                "Task": task,
                                "Temperature": temp,
                                "Level": "task_temperature",
                            },
                        )
                        if rankings:
                            results["rankings"].extend(rankings)
                            results["comparisons"].extend(comparisons)
                            if "task_temperature" not in results["analysis_levels"]:
                                results["analysis_levels"].append("task_temperature")

                # Task x Domain
                if domain_x_task and by_domain and "Domain" in metrics_df.columns:
                    for domain in metrics_df["Domain"].unique():
                        domain_task_df = task_df[task_df["Domain"] == domain]
                        rankings, comparisons = analyze_group(
                            domain_task_df,
                            {"Task": task, "Domain": domain, "Level": "task_domain"},
                        )
                        if rankings:
                            results["rankings"].extend(rankings)
                            results["comparisons"].extend(comparisons)
                            if "task_domain" not in results["analysis_levels"]:
                                results["analysis_levels"].append("task_domain")

        return results

    def export_multilevel_rankings(
        self, results, metric_name, base_filename="llm_rankings"
    ):
        """
        Export multi-level ranking results
        """
        # Convert rankings to DataFrame
        rankings_df = pd.DataFrame(results["rankings"])
        comparisons_df = pd.DataFrame(results["comparisons"])

        # Generate descriptive filename
        levels = "_".join(results["analysis_levels"])
        filename = f"{base_filename}_{metric_name}_{levels}.xlsx"

        # Export to Excel with multiple sheets
        with pd.ExcelWriter(filename) as writer:
            rankings_df.to_excel(writer, sheet_name="Rankings", index=False)
            comparisons_df.to_excel(writer, sheet_name="Statistical_Tests", index=False)

            # Add analysis level summary
            pd.DataFrame({"Analysis_Levels": results["analysis_levels"]}).to_excel(
                writer, sheet_name="Analysis_Levels", index=False
            )

        return rankings_df, comparisons_df

    ###########

    def analyze_aggregate_performance(self, detailed_metrics_df):
        """Analyze aggregate performance at different levels with significance"""
        metrics = ["R2", "RMSE", "Correlation", "MAE"]

        aggregation_levels = [
            {"name": "Overall", "group_by": ["LLM", "Temperature"]},
            {"name": "By Domain", "group_by": ["Domain", "LLM", "Temperature"]},
            {"name": "By Task", "group_by": ["Task", "LLM", "Temperature"]},
        ]

        results = []

        for agg_level in aggregation_levels:
            grouped = detailed_metrics_df.groupby(agg_level["group_by"])

            for name, group in grouped:
                group_dict = (
                    dict(zip(agg_level["group_by"], name))
                    if isinstance(name, tuple)
                    else {agg_level["group_by"][0]: name}
                )

                for metric in metrics:
                    best_llm_info = self.determine_best_llm_with_significance(
                        group, metric
                    )

                    result = {
                        "Aggregation_Level": agg_level["name"],
                        "Metric": metric,
                        **group_dict,
                        "Best_LLM": best_llm_info["best_llm"],
                        "Mean_Performance": best_llm_info["mean_performance"],
                        "Significance": best_llm_info["significance"],
                        "Confidence": best_llm_info["confidence"],
                        "N_samples": len(group),
                    }

                    # Only add effect size if it exists in best_llm_info
                    if "effect_size" in best_llm_info:
                        result["Effect_Size"] = best_llm_info["effect_size"]

                    results.append(result)

        return pd.DataFrame(results)

    def higher_is_better(metric):
        """Determine if higher values are better for given metric"""
        higher_better_metrics = {"R2", "Correlation", "Adjusted_R2"}
        return metric in higher_better_metrics

    def export_comprehensive_analysis(
        self, regression_results, correlation_results, filename
    ):
        """Export comprehensive analysis including task-specific performance"""
        # Get task-specific analysis
        task_analysis = self.analyze_task_specific_performance(
            regression_results, correlation_results
        )

        # Get domain and aggregate analysis
        domain_analysis = self.analyze_by_domain_and_aggregate(
            correlation_results, regression_results
        )

        with pd.ExcelWriter(filename) as writer:
            # Task-specific sheets
            task_summary = task_analysis[
                [
                    "Domain",
                    "Task",
                    "Temperature",
                    "Best_LLM_R2",
                    "Best_R2",
                    "Best_LLM_RMSE",
                    "Best_RMSE",
                    "Best_LLM_Corr",
                    "Best_Correlation",
                    "N_samples",
                ]
            ]
            task_summary.to_excel(
                writer, sheet_name="Task Performance Summary", index=False
            )

            # Create detailed metrics sheet
            detailed_metrics = []
            for _, row in task_analysis.iterrows():
                for metric_dict in row["All_Metrics"]:
                    detailed_metrics.append(metric_dict)

            pd.DataFrame(detailed_metrics).sort_values(
                ["Domain", "Task", "Temperature", "LLM"]
            ).to_excel(writer, sheet_name="Task Detailed Metrics", index=False)

            # Create temperature comparison sheet
            temp_comparison = (
                pd.DataFrame(detailed_metrics)
                .groupby(["LLM", "Temperature"])
                .agg(
                    {
                        "R2": ["mean", "std"],
                        "RMSE": ["mean", "std"],
                        "Correlation": ["mean", "std"],
                        "N_samples": "sum",
                    }
                )
                .round(3)
            )
            temp_comparison.to_excel(writer, sheet_name="Temperature Comparison")

            # Rest of the sheets remain the same...
            domain_analysis["correlation_domain"].to_excel(
                writer, sheet_name="Correlation by Domain"
            )
            domain_analysis["correlation_aggregate"].to_excel(
                writer, sheet_name="Correlation Aggregate"
            )
            domain_analysis["regression_domain"].to_excel(
                writer, sheet_name="Regression by Domain"
            )
            domain_analysis["regression_aggregate"].to_excel(
                writer, sheet_name="Regression Aggregate"
            )
            ########
            # Add aggregate analysis
            detailed_metrics_df = pd.DataFrame(detailed_metrics)
            aggregate_analysis = self.analyze_aggregate_performance(detailed_metrics_df)

            # Add to Excel export
            aggregate_analysis.to_excel(
                writer, sheet_name="Aggregate Performance", index=False
            )

            # Add explanation of confidence ratings
            confidence_explanation = pd.DataFrame(
                {
                    "Confidence Level": ["High", "Medium", "Low"],
                    "Description": [
                        "Best LLM shows both statistically significant difference (p < 0.05) and meaningful effect size",
                        "Best LLM shows either statistical significance or meaningful effect size, but not both",
                        "Differences between LLMs are neither statistically significant nor meaningful",
                    ],
                    "Interpretation": [
                        "Strong evidence that this LLM is truly better",
                        "Some evidence of superior performance, but more data might be needed",
                        "LLMs perform similarly, differences might be due to chance",
                    ],
                }
            )
            confidence_explanation.to_excel(
                writer, sheet_name="Confidence Ratings", index=False
            )

            #########
            metrics_explanation = pd.DataFrame(
                {
                    "Metric": [
                        "R2",
                        "Adjusted_R2",
                        "MAE",
                        "MSE",
                        "RMSE",
                        "MAPE",
                        "Residual_Std",
                        "Residual_Skew",
                        "Residual_Kurtosis",
                        "Residual_Normality_p",
                        "N_samples",
                    ],
                    "Description": [
                        "Proportion of variance explained by the model",
                        "R² adjusted for number of predictors",
                        "Mean Absolute Error",
                        "Mean Squared Error",
                        "Root Mean Squared Error",
                        "Mean Absolute Percentage Error",
                        "Standard deviation of residuals",
                        "Skewness of residuals (measure of asymmetry)",
                        "Kurtosis of residuals (measure of tail extremity)",
                        "P-value for normality test of residuals",
                        "Number of matched response pairs",
                    ],
                    "Interpretation": [
                        "Higher is better (0-1)",
                        "Higher is better (0-1), penalizes complexity",
                        "Lower is better (in original units)",
                        "Lower is better (squared units)",
                        "Lower is better (in original units)",
                        "Lower is better (percentage)",
                        "Lower indicates more consistent predictions",
                        "0 is ideal (symmetric)",
                        "3 is ideal (normal distribution)",
                        "> 0.05 suggests normal residuals",
                        "More samples give more reliable results",
                    ],
                }
            )
            metrics_explanation.to_excel(
                writer, sheet_name="Metrics Guide", index=False
            )

    ### analyze by domain and aggregate
    def analyze_by_domain_and_aggregate(self, correlation_results, regression_results):
        """Analyze results both by domain and in aggregate"""

        # Correlation analysis by domain and aggregate
        corr_domain = (
            correlation_results.groupby(["LLM", "Temperature", "domain", "Method"])
            .agg(
                {
                    "Correlation": ["mean", "std", "min", "max"],
                    "P_value": ["mean", "min", "max"],
                    "N": ["mean", "sum"],
                }
            )
            .round(3)
        )

        corr_aggregate = (
            correlation_results.groupby(["LLM", "Temperature", "Method"])
            .agg(
                {
                    "Correlation": ["mean", "std", "min", "max"],
                    "P_value": ["mean", "min", "max"],
                    "N": ["mean", "sum"],
                }
            )
            .round(3)
        )

        # Regression analysis by domain and aggregate
        reg_domain = (
            regression_results.groupby(["LLM", "Temperature", "domain"])
            .agg(
                {
                    "R2": ["mean", "std", "min", "max"],
                    "Slope": ["mean", "std"],
                    "Intercept": ["mean", "std"],
                    "RMSE": ["mean", "std"],
                    "N": ["mean", "sum"],
                }
            )
            .round(3)
        )

        reg_aggregate = (
            regression_results.groupby(["LLM", "Temperature"])
            .agg(
                {
                    "R2": ["mean", "std", "min", "max"],
                    "Slope": ["mean", "std"],
                    "Intercept": ["mean", "std"],
                    "RMSE": ["mean", "std"],
                    "N": ["mean", "sum"],
                }
            )
            .round(3)
        )

        return {
            "correlation_domain": corr_domain,
            "correlation_aggregate": corr_aggregate,
            "regression_domain": reg_domain,
            "regression_aggregate": reg_aggregate,
        }

    def export_domain_analysis(self, analysis_results, filename):
        """Export domain-specific and aggregate analysis to Excel"""
        with pd.ExcelWriter(filename) as writer:
            # Correlation results
            analysis_results["correlation_domain"].to_excel(
                writer, sheet_name="Correlation by Domain"
            )
            analysis_results["correlation_aggregate"].to_excel(
                writer, sheet_name="Correlation Aggregate"
            )

            # Regression results
            analysis_results["regression_domain"].to_excel(
                writer, sheet_name="Regression by Domain"
            )
            analysis_results["regression_aggregate"].to_excel(
                writer, sheet_name="Regression Aggregate"
            )

            # Add explanation sheet
            explanation = pd.DataFrame(
                {
                    "Analysis Type": [
                        "Correlation by Domain",
                        "Correlation Aggregate",
                        "Regression by Domain",
                        "Regression Aggregate",
                        "Metrics Explanation",
                    ],
                    "Description": [
                        "Correlation metrics calculated separately for each domain",
                        "Overall correlation metrics across all domains",
                        "Regression metrics calculated separately for each domain",
                        "Overall regression metrics across all domains",
                        "mean: average value\nstd: standard deviation\n"
                        + "min/max: minimum/maximum values\nN: number of samples",
                    ],
                }
            )
            explanation.to_excel(writer, sheet_name="Explanations", index=False)

    def plot_domain_comparison(self, regression_results, selected_temp=None):
        """Plot regression lines separated by domain"""
        domains = regression_results["domain"].unique()
        fig, axes = plt.subplots(1, len(domains), figsize=(6 * len(domains), 5))
        if len(domains) == 1:
            axes = [axes]

        for ax, domain in zip(axes, domains):
            # Filter data for this domain
            domain_data = regression_results[regression_results["domain"] == domain]

            # Plot diagonal line
            ax.plot([0, 100], [0, 100], "k--", alpha=0.3, task="Perfect agreement")

            # Plot regression lines for each LLM
            for llm in domain_data["LLM"].unique():
                llm_data = domain_data[domain_data["LLM"] == llm]
                if selected_temp is not None:
                    llm_data = llm_data[llm_data["Temperature"] == selected_temp]

                for _, row in llm_data.iterrows():
                    # Generate points for regression line
                    x = np.array([0, 100])
                    y = row["Slope"] * x + row["Intercept"]

                    task = f"{llm} (T={row['Temperature']})"
                    ax.plot(x, y, "-", task=task)

            ax.set_xtask("Human Responses")
            ax.set_ytask("LLM Responses")
            ax.set_title(f"Domain: {domain}")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def summarize_domain_effects(self, regression_results):
        """Create summary of domain effects on model performance"""
        # Calculate domain effects
        domain_effects = (
            regression_results.groupby(["LLM", "domain"])
            .agg(
                {
                    "R2": ["mean", "std"],
                    "Slope": ["mean", "std"],
                    "RMSE": ["mean", "std"],
                }
            )
            .round(3)
        )

        # Test for significant differences between domains
        llms = regression_results["LLM"].unique()
        metrics = ["R2", "Slope", "RMSE"]

        significance_tests = []
        for llm in llms:
            llm_data = regression_results[regression_results["LLM"] == llm]

            for metric in metrics:
                # Perform Kruskal-Wallis H-test
                domains_data = [
                    group[metric].values for name, group in llm_data.groupby("domain")
                ]

                h_stat, p_val = stats.kruskal(*domains_data)

                significance_tests.append(
                    {
                        "LLM": llm,
                        "Metric": metric,
                        "H_statistic": h_stat,
                        "P_value": p_val,
                        "Significant": p_val < 0.05,
                    }
                )

        return {
            "domain_effects": domain_effects,
            "significance_tests": pd.DataFrame(significance_tests),
        }

    ### ANalysis of temperature effects

    def analyze_temperature_effects(self, metrics_df, metric_col, alpha=0.05):
        """Analyze temperature effects with corrected pairwise comparisons"""

        print("Initial data counts:")
        print(metrics_df.groupby(["LLM", "Temperature", "Domain"])[metric_col].count())

        results = []
        for llm in metrics_df["LLM"].unique():
            # Add after selecting llm_data
            llm_data = metrics_df[metrics_df["LLM"] == llm]
            print(f"\nCounts for {llm}:")
            print(llm_data.groupby(["Temperature", "Domain"])[metric_col].count())

        results = []

        for llm in metrics_df["LLM"].unique():
            llm_data = metrics_df[metrics_df["LLM"] == llm]
            temps = sorted(llm_data["Temperature"].unique())

            if len(temps) <= 1:
                continue

            # Helper function for temperature analysis
            def analyze_temp_effects(data, domain="All", task="All", level="overall"):
                if len(data) < 2:
                    return

                print(f"\nGroup sizes for {llm} - {domain} - {task}:")
                for t in temps:
                    temp_group = data[data["Temperature"] == t]
                    print(f"Temperature {t}: {len(temp_group)} samples")

                # Kruskal-Wallis test
                temp_groups = [
                    data[data["Temperature"] == t][metric_col] for t in temps
                ]
                temp_groups = [
                    g for g in temp_groups if len(g) > 0
                ]  # Remove empty groups

                if len(temp_groups) < 2:
                    return

                h_stat, p_val = stats.kruskal(*temp_groups)

                results.append(
                    {
                        "LLM": llm,
                        "Domain": domain,
                        "Task": task,
                        "Test": "Kruskal-Wallis",
                        "Statistic": h_stat,
                        "P_Value": p_val,
                        "Significant": p_val < alpha,
                        "Level": level,
                    }
                )

                # Always do pairwise comparisons (don't filter on significance)
                for i, temp1 in enumerate(temps):
                    group1 = data[data["Temperature"] == temp1][metric_col]
                    if len(group1) == 0:
                        continue

                    for temp2 in temps[i + 1 :]:
                        group2 = data[data["Temperature"] == temp2][metric_col]
                        if len(group2) == 0:
                            continue

                        stat, p = stats.mannwhitneyu(
                            group1, group2, alternative="two-sided"
                        )

                        # Apply Bonferroni correction
                        n_comparisons = len(temps) * (len(temps) - 1) / 2
                        adjusted_p = min(p * n_comparisons, 1.0)

                        results.append(
                            {
                                "LLM": llm,
                                "Domain": domain,
                                "Task": task,
                                "Test": "Mann-Whitney U",
                                "Temperature1": temp1,
                                "Temperature2": temp2,
                                "Statistic": stat,
                                "P_Value": p,
                                "Adjusted_P_Value": adjusted_p,
                                "Significant": adjusted_p < alpha,
                                "Level": f"{level}_pairwise",
                                "Mean1": group1.mean(),
                                "Mean2": group2.mean(),
                                "Std1": group1.std(),
                                "Std2": group2.std(),
                                "N1": len(group1),
                                "N2": len(group2),
                            }
                        )

            # Overall analysis
            analyze_temp_effects(llm_data)

            # By domain
            if "Domain" in metrics_df.columns:
                for domain in llm_data["Domain"].unique():
                    domain_data = llm_data[llm_data["Domain"] == domain]
                    analyze_temp_effects(domain_data, domain=domain, level="domain")

            # By task
            if "Task" in metrics_df.columns:
                for task in llm_data["Task"].unique():
                    task_data = llm_data[llm_data["Task"] == task]
                    analyze_temp_effects(task_data, task=task, level="task")

                    # By domain and task
                    if "Domain" in metrics_df.columns:
                        for domain in task_data["Domain"].unique():
                            domain_task_data = task_data[task_data["Domain"] == domain]
                            analyze_temp_effects(
                                domain_task_data,
                                domain=domain,
                                task=task,
                                level="domain_task",
                            )

        return pd.DataFrame(results)

    def export_temperature_analysis(
        self, temp_results, metric_name, filename="temperature_effects"
    ):
        """Export temperature analysis results with enhanced summaries"


        takes in a df like this that combines results from correalation
        # Create metrics dataframe combining results
            metrics_df = pd.DataFrame(
                {
                    "LLM": reg_results["LLM"],
                    "Temperature": reg_results["Temperature"],
                    "Domain": reg_results["domain"] if "domain" in reg_results else "All",
                    "Task": reg_results["Task"] if "task" in reg_results else "All",
                    "R2": reg_results["R2"],
                    "RMSE": reg_results["RMSE"],
                    # "MAE": reg_results["MAE"],
                    "Correlation": corr_results["Correlation"],
                    "p-value_corr": corr_results["P_value"],
                }
            )
        """
        full_filename = f"{filename}_{metric_name}.xlsx"

        # Split results by test type
        kruskal_results = temp_results[temp_results["Test"] == "Kruskal-Wallis"].copy()
        pairwise_results = temp_results[temp_results["Test"] == "Mann-Whitney U"].copy()

        # Create significance summaries
        sig_summary = pd.DataFrame(
            [
                {
                    "Level": level,
                    "Total_Tests": len(group),
                    "Significant_Tests": sum(group["Significant"]),
                    "Significance_Rate": f"{(sum(group['Significant']) / len(group)) * 100:.1f}%",
                }
                for level, group in temp_results.groupby("Level")
            ]
        )

        # Create LLM-specific summaries
        llm_summary = pd.DataFrame(
            [
                {
                    "LLM": llm,
                    "Total_Tests": len(group),
                    "Significant_Tests": sum(group["Significant"]),
                    "Significance_Rate": f"{(sum(group['Significant']) / len(group)) * 100:.1f}%",
                }
                for llm, group in temp_results.groupby("LLM")
            ]
        )

        # Export to Excel
        with pd.ExcelWriter(full_filename) as writer:
            kruskal_results.to_excel(writer, sheet_name="Overall_Effects", index=False)
            pairwise_results.to_excel(
                writer, sheet_name="Pairwise_Comparisons", index=False
            )
            sig_summary.to_excel(
                writer, sheet_name="Significance_By_Level", index=False
            )
            llm_summary.to_excel(writer, sheet_name="Significance_By_LLM", index=False)

            # Add effect size interpretation guide
            guide = pd.DataFrame(
                {
                    "Metric": [
                        "P-Value",
                        "Adjusted P-Value",
                        "Effect Size Interpretation",
                    ],
                    "Description": [
                        "Raw p-value from statistical test",
                        "P-value adjusted for multiple comparisons (Bonferroni)",
                        "Small: 0.2-0.5, Medium: 0.5-0.8, Large: >0.8",
                    ],
                }
            )
            guide.to_excel(writer, sheet_name="Guide", index=False)

        return kruskal_results, pairwise_results

    def analyze_temperature_effects_raw(self, data, alpha=0.05):
        """
        Analyze temperature effects on raw responses
        """
        results = []

        # Get unique values
        llms = data["agent"].unique()
        domains = data["domain"].unique()
        tasks = data["task"].unique()
        ppp_conditions = data["ppp"].unique()

        for llm in llms:
            llm_data = data[data["agent"] == llm]
            temps = sorted(llm_data["temperature"].unique())

            if len(temps) <= 1:
                continue

            # Helper function for temperature analysis
            def analyze_temp_effects(
                subset_data, domain="All", task="All", ppp="All", level="overall"
            ):
                if len(subset_data) < 2:
                    return

                # Kruskal-Wallis test
                temp_groups = [
                    subset_data[subset_data["temperature"] == t]["response"]
                    for t in temps
                ]
                temp_groups = [g for g in temp_groups if len(g) > 0]

                if len(temp_groups) < 2:
                    return

                # h_stat, p_val = stats.kruskal(*temp_groups)

                try:
                    h_stat, p_val = stats.kruskal(*temp_groups)
                except ValueError:
                    # If identical values, record this case
                    results.append(
                        {
                            "LLM": llm,
                            "Domain": domain,
                            "Task": task,
                            "PPP": ppp,
                            "Test": "Kruskal-Wallis",
                            "Statistical_Test_Details": "Test failed - identical values across temperatures",
                            "Statistic": None,
                            "P_Value": None,
                            "Significant": False,
                            "Level": level,
                            "N_total": len(subset_data),
                        }
                    )
                    return

                results.append(
                    {
                        "LLM": llm,
                        "Domain": domain,
                        "Task": task,
                        "PPP": ppp,
                        "Test": "Kruskal-Wallis",
                        "Statistic": h_stat,
                        "P_Value": p_val,
                        "Significant": p_val < alpha,
                        "Statistical_Test_Details": "Kruskal-Wallis test for overall temperature effect",
                        "Level": level,
                        "N_total": len(subset_data),
                    }
                )

                # Pairwise comparisons
                for i, temp1 in enumerate(temps):
                    group1 = subset_data[subset_data["temperature"] == temp1][
                        "response"
                    ]
                    if len(group1) == 0:
                        continue

                    for temp2 in temps[i + 1 :]:
                        group2 = subset_data[subset_data["temperature"] == temp2][
                            "response"
                        ]
                        if len(group2) == 0:
                            continue

                        # stat, p = stats.mannwhitneyu(
                        #     group1, group2, alternative="two-sided"
                        # )

                        try:
                            stat, p = stats.mannwhitneyu(
                                group1, group2, alternative="two-sided"
                            )
                        except ValueError:
                            continue  # Skip this comparison if groups are identical

                        # Bonferroni correction
                        n_comparisons = len(temps) * (len(temps) - 1) / 2
                        adjusted_p = min(p * n_comparisons, 1.0)

                        results.append(
                            {
                                "LLM": llm,
                                "Domain": domain,
                                "Task": task,
                                "PPP": ppp,
                                "Test": "Mann-Whitney U",
                                "Temperature1": temp1,
                                "Temperature2": temp2,
                                "Statistic": stat,
                                "P_Value": p,
                                "Adjusted_P_Value": adjusted_p,
                                "Significant": adjusted_p < alpha,
                                "Statistical_Test_Details": "Mann-Whitney U test with Bonferroni correction for pairwise temperature comparison",
                                "Level": f"{level}_pairwise",
                                "Mean1": group1.mean(),
                                "Mean2": group2.mean(),
                                "Std1": group1.std(),
                                "Std2": group2.std(),
                                "N1": len(group1),
                                "N2": len(group2),
                            }
                        )

            # Overall analysis
            analyze_temp_effects(llm_data)

            # By domain
            for domain in domains:
                domain_data = llm_data[llm_data["domain"] == domain]
                analyze_temp_effects(domain_data, domain=domain, level="domain")

            # By task
            for task in tasks:
                task_data = llm_data[llm_data["task"] == task]
                analyze_temp_effects(task_data, task=task, level="task")

            # By PPP condition
            for ppp in ppp_conditions:
                ppp_data = llm_data[llm_data["ppp"] == ppp]
                analyze_temp_effects(ppp_data, ppp=ppp, level="ppp")

            # By domain and task
            for domain in domains:
                domain_data = llm_data[llm_data["domain"] == domain]
                for task in tasks:
                    domain_task_data = domain_data[domain_data["task"] == task]
                    analyze_temp_effects(
                        domain_task_data, domain=domain, task=task, level="domain_task"
                    )

            # By domain and PPP
            for domain in domains:
                domain_data = llm_data[llm_data["domain"] == domain]
                for ppp in ppp_conditions:
                    domain_ppp_data = domain_data[domain_data["ppp"] == ppp]
                    analyze_temp_effects(
                        domain_ppp_data, domain=domain, ppp=ppp, level="domain_ppp"
                    )

        return pd.DataFrame(results)

    def export_temperature_analysis_raw(
        self, temp_results, filename="temperature_effects_raw"
    ):
        """Export raw temperature analysis results"""
        full_filename = f"{filename}.xlsx"

        # Split results by test type
        kruskal_results = temp_results[temp_results["Test"] == "Kruskal-Wallis"]
        pairwise_results = temp_results[temp_results["Test"] == "Mann-Whitney U"]

        # Create significance summaries
        sig_summary = pd.DataFrame(
            [
                {
                    "Level": level,
                    "Total_Tests": len(group),
                    "Significant_Tests": sum(group["Significant"]),
                    "Significance_Rate": f"{(sum(group['Significant']) / len(group)) * 100:.1f}%",
                }
                for level, group in temp_results.groupby("Level")
            ]
        )

        # Create LLM-specific summaries
        llm_summary = pd.DataFrame(
            [
                {
                    "LLM": llm,
                    "Total_Tests": len(group),
                    "Significant_Tests": sum(group["Significant"]),
                    "Significance_Rate": f"{(sum(group['Significant']) / len(group)) * 100:.1f}%",
                    "Average_N": group["N_total"].mean()
                    if "N_total" in group
                    else None,
                }
                for llm, group in temp_results.groupby("LLM")
            ]
        )

        # Export to Excel
        with pd.ExcelWriter(full_filename) as writer:
            kruskal_results.to_excel(writer, sheet_name="Overall_Effects", index=False)
            pairwise_results.to_excel(
                writer, sheet_name="Pairwise_Comparisons", index=False
            )
            sig_summary.to_excel(
                writer, sheet_name="Significance_By_Level", index=False
            )
            llm_summary.to_excel(writer, sheet_name="Significance_By_LLM", index=False)

        return kruskal_results, pairwise_results

    ###### prepare metrics
    def prepare_metrics_df(self, reg_results, corr_results=None):
        """
        Prepare metrics DataFrame from regression and correlation results
        """
        metrics = []

        # Process regression results
        for idx, row in reg_results.iterrows():
            metric_row = {
                "LLM": row["LLM"],
                "Temperature": row["Temperature"],
                "R2": row["R2"],
                "RMSE": row["RMSE"],
                "MSE": row["MSE"] if "MSE" in row else None,
                "MAE": row["MAE"] if "MAE" in row else None,
            }

            # Add domain if present
            if "Domain" in row:
                metric_row["domain"] = row["domain"]

            # Add task if present
            if "Task" in row:
                metric_row["task"] = row["task"]

            metrics.append(metric_row)

        # Create DataFrame
        metrics_df = pd.DataFrame(metrics)

        # Add correlation if provided
        if corr_results is not None:
            corr_df = pd.DataFrame(corr_results)
            metrics_df = pd.merge(
                metrics_df,
                corr_df[["LLM", "Temperature", "domain", "Correlation"]],
                on=["LLM", "Temperature", "domain"],
                how="left",
            )

        return metrics_df


#################################
### Domain Correlation Analysis
####################################
class DomainCorrelationAnalysis(CausalReasoningAnalysis):
    def analyze_domain_correlations_per_temp(self):
        """Analyze correlations between human and LLM responses by domain"""
        humans = self.data[self.data["agent"] == "humans"].copy()
        results = []

        llm_agents = [s for s in self.data["agent"].unique() if s != "humans"]
        match_columns = ["domain", "ppp", "task"]

        for llm in llm_agents:
            llm_data = self.data[self.data["agent"] == llm].copy()

            # Analyze by domain
            for domain in self.data["domain"].unique():
                domain_humans = humans[humans["domain"] == domain]
                domain_llm = llm_data[llm_data["domain"] == domain]

                # Match data
                matched_data = pd.merge(
                    domain_humans[match_columns + ["response"]],
                    domain_llm[match_columns + ["response", "temperature"]],
                    on=match_columns,
                    suffixes=("_human", "_llm"),
                )

                # Analyze for each temperature
                for temp in matched_data["temperature"].unique():
                    temp_data = matched_data[matched_data["temperature"] == temp]

                    # Check normality
                    if len(temp_data) >= 3:
                        _, h_pval = stats.shapiro(temp_data["response_human"])
                        _, l_pval = stats.shapiro(temp_data["response_llm"])
                        is_normal = h_pval > 0.05 and l_pval > 0.05
                    else:
                        is_normal = False

                    # Calculate correlation
                    if is_normal:
                        corr, p = stats.pearsonr(
                            temp_data["response_human"], temp_data["response_llm"]
                        )
                        method = "pearson"
                    else:
                        corr, p = stats.spearmanr(
                            temp_data["response_human"], temp_data["response_llm"]
                        )
                        method = "spearman"

                    results.append(
                        {
                            "LLM": llm,
                            "Domain": domain,
                            "Temperature": temp,
                            "Correlation": corr,
                            "P_value": p,
                            "Method": method,
                            "N": len(temp_data),
                        }
                    )

        return pd.DataFrame(results)

    def analyze_domain_correlations(self):
        """Analyze correlations between human and LLM responses by domain"""
        humans = self.data[self.data["agent"] == "humans"].copy()
        results = []
        llm_agents = [s for s in self.data["agent"].unique() if s != "humans"]
        match_columns = ["domain", "ppp", "task"]

        for llm in llm_agents:
            llm_data = self.data[self.data["agent"] == llm].copy()

            for domain in self.data["domain"].unique():
                domain_humans = humans[humans["domain"] == domain]
                domain_llm = llm_data[llm_data["domain"] == domain]

                # Aggregate LLM responses across temperatures
                domain_llm_agg = (
                    domain_llm.groupby(match_columns)["response"].mean().reset_index()
                )

                # Match data
                matched_data = pd.merge(
                    domain_humans[match_columns + ["response"]],
                    domain_llm_agg[match_columns + ["response"]],
                    on=match_columns,
                    suffixes=("_human", "_llm"),
                )

                # Check normality
                if len(matched_data) >= 3:
                    _, h_pval = stats.shapiro(matched_data["response_human"])
                    _, l_pval = stats.shapiro(matched_data["response_llm"])
                    is_normal = h_pval > 0.05 and l_pval > 0.05
                else:
                    is_normal = False

                # Calculate correlation
                if is_normal:
                    corr, p = stats.pearsonr(
                        matched_data["response_human"], matched_data["response_llm"]
                    )
                    method = "pearson"
                else:
                    corr, p = stats.spearmanr(
                        matched_data["response_human"], matched_data["response_llm"]
                    )
                    method = "spearman"

                results.append(
                    {
                        "LLM": llm,
                        "Domain": domain,
                        "Correlation": corr,
                        "P_value": p,
                        "Method": method,
                        "N": len(matched_data),
                    }
                )

        return pd.DataFrame(results)

    # def plot_domain_correlations(self, correlation_results, figsize=(15, 10)):
    #     """Create visualizations for domain-based correlation analysis"""
    #     fig = plt.figure(figsize=figsize)
    #     gs = plt.GridSpec(2, 2)

    #     # 1. Domain correlations by LLM
    #     ax1 = fig.add_subplot(gs[0, 0])
    #     sns.barplot(
    #         data=correlation_results, x="Domain", y="Correlation", hue="LLM", ax=ax1
    #     )
    #     ax1.set_title("Correlations by Domain and LLM")
    #     ax1.set_ytask("Correlation Coefficient")
    #     ax1.tick_params(axis="x", rotation=45)

    #     # 2. Significance by domain
    #     ax2 = fig.add_subplot(gs[0, 1])
    #     sig_plot = -np.log10(correlation_results["P_value"])
    #     sns.barplot(data=correlation_results, x="Domain", y=sig_plot, hue="LLM", ax=ax2)
    #     ax2.set_title("Correlation Significance by Domain (-log10 p-value)")
    #     ax2.axhline(-np.log10(0.05), color="r", linestyle="--", task="p=0.05")
    #     ax2.tick_params(axis="x", rotation=45)
    #     ax2.legend()

    #     # 3. Heatmap of correlations
    #     ax3 = fig.add_subplot(gs[1, :])
    #     pivot_table = correlation_results.pivot_table(
    #         index="LLM", columns="Domain", values="Correlation"
    #     )
    #     sns.heatmap(pivot_table, annot=True, cmap="RdYlBu", center=0, ax=ax3)
    #     ax3.set_title("Correlation Patterns Across Domains")

    #     plt.tight_layout()
    #     return fig
    def plot_domain_correlations(
        self, correlation_results, plot_type="bar", figsize=(15, 10)
    ):
        """
        Create visualizations for domain-based correlation analysis.

        Parameters:
            correlation_results (DataFrame): The correlation results data.
            plot_type (str): The type of plot to use, "bar" or "scatter".
            figsize (tuple): The size of the figure.
        """
        fig = plt.figure(figsize=figsize)
        gs = plt.GridSpec(2, 2)

        # 1. Domain correlations by LLM
        ax1 = fig.add_subplot(gs[0, 0])
        if plot_type == "bar":
            sns.barplot(
                data=correlation_results, x="Domain", y="Correlation", hue="LLM", ax=ax1
            )
        elif plot_type == "scatter":
            sns.scatterplot(
                data=correlation_results,
                x="Domain",
                y="Correlation",
                hue="LLM",
                style="LLM",
                s=100,
                ax=ax1,
            )
        else:
            raise ValueError("Invalid plot_type. Use 'bar' or 'scatter'.")
        ax1.set_title("Correlations by Domain and LLM")
        ax1.set_ytask("Correlation Coefficient")
        ax1.tick_params(axis="x", rotation=45)

        # 2. Significance by domain
        ax2 = fig.add_subplot(gs[0, 1])
        sig_plot = -np.log10(correlation_results["P_value"])
        if plot_type == "bar":
            sns.barplot(
                data=correlation_results, x="Domain", y=sig_plot, hue="LLM", ax=ax2
            )
        elif plot_type == "scatter":
            sns.scatterplot(
                data=correlation_results,
                x="Domain",
                y=sig_plot,
                hue="LLM",
                style="LLM",
                s=100,
                ax=ax2,
            )
        ax2.set_title("Correlation Significance by Domain (-log10 p-value)")
        ax2.axhline(-np.log10(0.05), color="r", linestyle="--", task="p=0.05")
        ax2.tick_params(axis="x", rotation=45)
        ax2.legend()

        # 3. Heatmap of correlations
        ax3 = fig.add_subplot(gs[1, :])
        pivot_table = correlation_results.pivot_table(
            index="LLM", columns="Domain", values="Correlation"
        )
        sns.heatmap(pivot_table, annot=True, cmap="RdYlBu", center=0, ax=ax3)
        ax3.set_title("Correlation Patterns Across Domains")

        plt.tight_layout()
        return fig

    #################################
    # by task
    ####################################
    def plot_task_correlations(self, style="bar", figsize=(15, 10)):
        """Plot correlations for each task with customizable style"""
        humans = self.data[self.data["agent"] == "humans"].copy()
        tasks = sorted(self.data["task"].unique())

        for task in tasks:
            fig = plt.figure(figsize=figsize)
            gs = plt.GridSpec(3, 2)
            ax1 = fig.add_subplot(gs[:-1, :])
            ax2 = fig.add_subplot(gs[-1, :])

            task_results = []
            sample_sizes = []

            for domain in self.data["domain"].unique():
                task_humans = humans[
                    (humans["domain"] == domain) & (humans["task"] == task)
                ]["response"].values

                for llm in [s for s in self.data["agent"].unique() if s != "humans"]:
                    llm_data = self.data[
                        (self.data["agent"] == llm)
                        & (self.data["domain"] == domain)
                        & (self.data["task"] == task)
                    ]

                    for temp in llm_data["temperature"].unique():
                        temp_data = llm_data[llm_data["temperature"] == temp][
                            "response"
                        ].values
                        if len(task_humans) > 0 and len(temp_data) > 0:
                            try:
                                corr, p = stats.spearmanr(task_humans, temp_data)
                                task_results.append(
                                    {
                                        "domain": domain,
                                        "agent": llm,
                                        "temperature": temp,
                                        "correlation": corr,
                                        "p_value": p,
                                    }
                                )
                                sample_sizes.append(
                                    {
                                        "domain": domain,
                                        "agent": llm,
                                        "temperature": temp,
                                        "n_samples": len(task_humans),
                                    }
                                )
                            except:
                                continue

            df = pd.DataFrame(task_results)
            df_samples = pd.DataFrame(sample_sizes)

            # Plot correlations
            if style == "bar":
                sns.barplot(data=df, x="domain", y="correlation", hue="agent", ax=ax1)
            else:
                sns.scatterplot(
                    data=df,
                    x="domain",
                    y="correlation",
                    hue="agent",
                    style="agent",
                    s=100,
                    ax=ax1,
                )

            ax1.set_title(f"Correlations by Domain for Task {task}")
            ax1.set_ytask("Correlation Coefficient")
            ax1.tick_params(axis="x", rotation=45)

            # Plot significance
            ax1_twin = ax1.twinx()
            sig_plot = -np.log10(df["p_value"])
            if style == "bar":
                sns.barplot(
                    data=df,
                    x="domain",
                    y=sig_plot,
                    hue="agent",
                    ax=ax1_twin,
                    alpha=0.3,
                )
            else:
                sns.scatterplot(
                    data=df,
                    x="domain",
                    y=sig_plot,
                    hue="agent",
                    style="agent",
                    s=100,
                    ax=ax1_twin,
                    alpha=0.3,
                )

            ax1_twin.set_ytask("-log10(p-value)")
            ax1_twin.axhline(-np.log10(0.05), color="r", linestyle="--", task="p=0.05")

            # Sample sizes
            sns.barplot(data=df_samples, x="domain", y="n_samples", hue="agent", ax=ax2)
            ax2.set_title(f"Sample Size by Domain for Task {task}")
            ax2.set_ytask("Number of Samples")
            ax2.tick_params(axis="x", rotation=45)

            plt.tight_layout()
            yield fig
            plt.close()

    def plot_all_task_correlations(
        self, style="bar", filename_prefix="task_correlations_"
    ):
        """Generate and save plots for all tasks"""
        for i, fig in enumerate(self.plot_task_correlations(style=style)):
            task = sorted(self.data["task"].unique())[i]
            filename = f"{filename_prefix}{task}.png"
            fig.savefig(filename, bbox_inches="tight", dpi=300)
            yield fig

    ###
    def export_task_domain_correlations(self, filename="task_domain_correlations.xlsx"):
        """Export task and domain specific correlations to Excel"""
        humans = self.data[self.data["agent"] == "humans"].copy()
        llm_agents = [s for s in self.data["agent"].unique() if s != "humans"]
        results = []

        for domain in self.data["domain"].unique():
            for task in sorted(self.data["task"].unique()):
                task_humans = humans[
                    (humans["domain"] == domain) & (humans["task"] == task)
                ]["response"].values

                for llm in llm_agents:
                    llm_data = self.data[
                        (self.data["agent"] == llm)
                        & (self.data["domain"] == domain)
                        & (self.data["task"] == task)
                    ]

                    for temp in llm_data["temperature"].unique():
                        temp_data = llm_data[llm_data["temperature"] == temp][
                            "response"
                        ].values
                        if len(task_humans) > 0 and len(temp_data) > 0:
                            try:
                                corr, p = stats.spearmanr(task_humans, temp_data)

                                # Add mean responses and std
                                results.append(
                                    {
                                        "Domain": domain,
                                        "Task": task,
                                        "LLM": llm,
                                        "Temperature": temp,
                                        "Correlation": corr,
                                        "P_value": p,
                                        "N": len(task_humans),
                                        "Human_Mean": task_humans.mean(),
                                        "Human_Std": task_humans.std(),
                                        "LLM_Mean": temp_data.mean(),
                                        "LLM_Std": temp_data.std(),
                                    }
                                )
                            except:
                                continue

        df = pd.DataFrame(results)
        df.to_excel(filename, index=False)
        return df
