-- Verdict Dashboard Queries
-- Databricks SQL / Lakeview dashboard queries
--
-- These queries power the Verdict evaluation dashboard.
-- Create a Lakeview dashboard and add these as visualization widgets.

-- ============================================================================
-- 1. Verdict History Per Model Version
-- Shows PASS/WARN/FAIL history over time for each model version
-- ============================================================================

CREATE OR REPLACE VIEW verdict.v_dashboard.verdict_history AS
SELECT
    model_version,
    run_id,
    verdict,
    created_at,
    COUNT(*) FILTER (WHERE is_regression = true) AS regression_count,
    AVG(p_value) AS avg_p_value
FROM (
    SELECT
        ms.model_version,
        ms.run_id,
        ms.verdict,
        ms.created_at,
        GET_JSON_OBJECT(ms.verdict_reason, '$.is_regression') AS is_regression,
        ms.p_value
    FROM verdict.metrics.metric_summary ms
)
GROUP BY model_version, run_id, verdict, created_at
ORDER BY created_at DESC;

-- ============================================================================
-- 2. Metric Trends Over Time
-- Shows how each metric has evolved across model versions
-- ============================================================================

CREATE OR REPLACE VIEW verdict.v_dashboard.metric_trends AS
SELECT
    model_version,
    metric_name,
    created_at,
    metric_mean,
    metric_std,
    metric_p50,
    metric_p95,
    sample_size
FROM verdict.metrics.metric_summary
WHERE metric_name IN ('faithfulness', 'answer_relevance', 'judge_score', 'rouge_l')
ORDER BY model_version, metric_name, created_at;

-- ============================================================================
-- 3. Per-Prompt Score Breakdown
-- Detailed scores for each prompt in a run
-- ============================================================================

CREATE OR REPLACE VIEW verdict.v_dashboard.prompt_scores AS
SELECT
    er.prompt_id,
    pd.prompt,
    er.model_version,
    er.metric_name,
    er.metric_value,
    er.evaluator,
    mr.response,
    pd.ground_truth
FROM verdict.evaluated.eval_results er
JOIN verdict.raw.model_responses mr ON er.prompt_id = mr.prompt_id
    AND er.model_version = mr.model_version
JOIN verdict.raw.prompt_datasets pd ON er.prompt_id = pd.prompt_id
ORDER BY er.metric_value ASC;

-- ============================================================================
-- 4. Latency P95 Over Time
-- Latency performance tracking
-- ============================================================================

CREATE OR REPLACE VIEW verdict.v_dashboard.latency_trends AS
SELECT
    model_version,
    endpoint_name,
    created_at,
    PERCENTILE(latency_ms, 0.50) AS latency_p50,
    PERCENTILE(latency_ms, 0.95) AS latency_p95,
    PERCENTILE(latency_ms, 0.99) AS latency_p99,
    AVG(latency_ms) AS latency_mean,
    STDDEV(latency_ms) AS latency_std,
    COUNT(*) AS request_count
FROM verdict.raw.model_responses
WHERE latency_ms IS NOT NULL
GROUP BY model_version, endpoint_name, created_at
ORDER BY created_at DESC;

-- ============================================================================
-- 5. Regression Analysis Summary
-- Comparison of candidate vs baseline metrics
-- ============================================================================

CREATE OR REPLACE VIEW verdict.v_dashboard.regression_summary AS
SELECT
    ms.model_version AS candidate_version,
    ms.baseline_version,
    ms.metric_name,
    ms.metric_mean AS candidate_mean,
    ms.baseline_mean,
    (ms.metric_mean - ms.baseline_mean) AS mean_diff,
    CASE
        WHEN ms.baseline_mean > 0
        THEN ROUND(((ms.metric_mean - ms.baseline_mean) / ms.baseline_mean) * 100, 2)
        ELSE 0
    END AS pct_change,
    ms.p_value,
    ms.verdict,
    GET_JSON_OBJECT(ms.verdict_reason, '$.is_regression') AS is_regression
FROM verdict.metrics.metric_summary ms
WHERE ms.baseline_version IS NOT NULL
ORDER BY ms.created_at DESC, ms.metric_name;

-- ============================================================================
-- 6. Model Version Comparison
-- Side-by-side comparison of all metrics for two versions
-- ============================================================================

CREATE OR REPLACE VIEW verdict.v_dashboard.version_comparison AS
WITH candidate_metrics AS (
    SELECT metric_name, metric_mean, metric_std, sample_size
    FROM verdict.metrics.metric_summary
    WHERE model_version = '${var.candidate_version}'
),
baseline_metrics AS (
    SELECT metric_name, metric_mean, metric_std, sample_size
    FROM verdict.metrics.metric_summary
    WHERE model_version = '${var.baseline_version}'
)
SELECT
    COALESCE(c.metric_name, b.metric_name) AS metric_name,
    c.metric_mean AS candidate_mean,
    c.metric_std AS candidate_std,
    b.metric_mean AS baseline_mean,
    b.metric_std AS baseline_std,
    (c.metric_mean - b.metric_mean) AS mean_diff,
    CASE
        WHEN b.metric_mean > 0
        THEN ROUND(((c.metric_mean - b.metric_mean) / b.metric_mean) * 100, 2)
        ELSE 0
    END AS pct_change
FROM candidate_metrics c
FULL OUTER JOIN baseline_metrics b ON c.metric_name = b.metric_name
ORDER BY metric_name;

-- ============================================================================
-- 7. Error Analysis
-- Failed inferences and error patterns
-- ============================================================================

CREATE OR REPLACE VIEW verdict.v_dashboard.error_analysis AS
SELECT
    model_version,
    status,
    COUNT(*) AS error_count,
    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY model_version) AS error_rate_pct,
    error_message
FROM verdict.raw.model_responses
WHERE status != 'success'
GROUP BY model_version, status, error_message
ORDER BY error_count DESC;

-- ============================================================================
-- 8. Evaluation Coverage
-- What percentage of responses have been evaluated
-- ============================================================================

CREATE OR REPLACE VIEW verdict.v_dashboard.evaluation_coverage AS
SELECT
    mr.model_version,
    mr.run_id,
    COUNT(DISTINCT mr.response_id) AS total_responses,
    COUNT(DISTINCT er.response_id) AS evaluated_responses,
    ROUND(COUNT(DISTINCT er.response_id) * 100.0 / COUNT(DISTINCT mr.response_id), 2) AS coverage_pct,
    COUNT(DISTINCT er.metric_name) AS metrics_computed
FROM verdict.raw.model_responses mr
LEFT JOIN verdict.evaluated.eval_results er ON mr.response_id = er.response_id
GROUP BY mr.model_version, mr.run_id
ORDER BY mr.created_at DESC;

-- ============================================================================
-- 9. Judge Score Distribution
-- Distribution of LLM-as-a-judge scores
-- ============================================================================

CREATE OR REPLACE VIEW verdict.v_dashboard.judge_distribution AS
SELECT
    model_version,
    metric_value AS judge_score,
    COUNT(*) AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY model_version), 2) AS pct
FROM verdict.evaluated.eval_results
WHERE metric_name = 'judge_score'
    AND metric_value IS NOT NULL
GROUP BY model_version, metric_value
ORDER BY model_version, metric_value;

-- ============================================================================
-- 10. Daily Summary Report
-- Summary statistics for the latest run of each model version
-- ============================================================================

CREATE OR REPLACE VIEW verdict.v_dashboard.daily_summary AS
WITH latest_runs AS (
    SELECT
        model_version,
        MAX(run_id) AS latest_run_id,
        MAX(created_at) AS latest_run_time
    FROM verdict.metrics.metric_summary
    GROUP BY model_version
)
SELECT
    ms.model_version,
    ms.verdict,
    lr.latest_run_time,
    SUM(CASE WHEN ms.metric_name = 'faithfulness' THEN ms.metric_mean END) AS faithfulness,
    SUM(CASE WHEN ms.metric_name = 'answer_relevance' THEN ms.metric_mean END) AS answer_relevance,
    SUM(CASE WHEN ms.metric_name = 'judge_score' THEN ms.metric_mean END) AS judge_score,
    SUM(CASE WHEN ms.metric_name = 'rouge_l' THEN ms.metric_mean END) AS rouge_l,
    SUM(CASE WHEN ms.metric_name = 'latency_p95' THEN ms.metric_mean END) AS latency_p95_ms
FROM verdict.metrics.metric_summary ms
JOIN latest_runs lr ON ms.model_version = lr.model_version AND ms.run_id = lr.latest_run_id
GROUP BY ms.model_version, ms.verdict, lr.latest_run_time
ORDER BY lr.latest_run_time DESC;

-- ============================================================================
-- Create schema for dashboard views if it doesn't exist
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS verdict.v_dashboard;