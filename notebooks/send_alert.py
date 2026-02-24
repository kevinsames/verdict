# Databricks notebook source
# ---
# title: Send Alert
# description: Sends alerts based on regression verdict
# ---

# COMMAND ----------
import logging
import json
import requests
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------
# Widget parameters
dbutils.widgets.text("verdict", "", "Verdict (PASS/WARN/FAIL)")
dbutils.widgets.text("verdict_report", "", "Verdict Report JSON")
dbutils.widgets.text("webhook_secret", "verdict/alerts_webhook", "Webhook Secret Scope/Key")
dbutils.widgets.text("email_recipients", "", "Email Recipients (comma-separated)")

verdict = dbutils.widgets.get("verdict")
verdict_report_json = dbutils.widgets.get("verdict_report")
webhook_secret = dbutils.widgets.get("webhook_secret")
email_recipients = dbutils.widgets.get("email_recipients")

# COMMAND ----------
# Parse report
try:
    report = json.loads(verdict_report_json) if verdict_report_json else {}
except json.JSONDecodeError:
    report = {}
    logger.warning("Could not parse verdict report JSON")

# COMMAND ----------
# Check if we should alert
alert_on = ["WARN", "FAIL"]
should_alert = verdict in alert_on

logger.info(f"Verdict: {verdict}")
logger.info(f"Should alert: {should_alert}")

# COMMAND ----------
def send_webhook_alert(webhook_url: str, report: dict) -> bool:
    """Send alert to webhook."""
    payload = {
        "verdict": report.get("verdict"),
        "candidate_version": report.get("candidate_version"),
        "baseline_version": report.get("baseline_version"),
        "timestamp": datetime.now().isoformat(),
        "regressions": [
            c for c in report.get("comparisons", [])
            if c.get("is_regression")
        ]
    }

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Webhook alert failed: {e}")
        return False

# COMMAND ----------
if should_alert:
    logger.info(f"Sending alert for verdict: {verdict}")

    # Try to get webhook URL from secrets
    webhook_url = None
    try:
        if "/" in webhook_secret:
            scope, key = webhook_secret.split("/")
            webhook_url = dbutils.secrets.get(scope, key)
    except Exception as e:
        logger.warning(f"Could not get webhook URL from secrets: {e}")

    # Send webhook alert
    if webhook_url:
        success = send_webhook_alert(webhook_url, report)
        if success:
            logger.info("Webhook alert sent successfully")
        else:
            logger.error("Failed to send webhook alert")

    # Log alert details
    print("\n" + "=" * 60)
    print("ALERT SENT")
    print("=" * 60)
    print(f"Verdict: {verdict}")
    print(f"Candidate Version: {report.get('candidate_version')}")
    print(f"Baseline Version: {report.get('baseline_version')}")

    regressions = [c for c in report.get("comparisons", []) if c.get("is_regression")]
    if regressions:
        print("\nRegressions detected:")
        for r in regressions:
            print(f"  - {r.get('metric_name')}: {r.get('pct_change', 0):+.2f}%")

    print("=" * 60 + "\n")

else:
    logger.info(f"No alert needed for verdict: {verdict}")
    print(f"\n✓ No regressions detected. Verdict: {verdict}\n")

# COMMAND ----------
# Return final status
dbutils.jobs.taskValues.set("alert_sent", should_alert)
dbutils.jobs.taskValues.set("final_verdict", verdict)
