"""
Run Context Manager for Verdict pipeline.

Manages pipeline run state in the metadata.pipeline_runs table.
Provides centralized configuration, task state, and output tracking.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Any

from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)


class RunContext:
    """
    Manages pipeline run state in metadata.pipeline_runs table.

    This class centralizes all configuration, task state, and outputs
    for a pipeline run, eliminating the need for parameter passing
    between tasks.

    Usage:
        ctx = RunContext(spark, catalog_name="verdict_dev")
        ctx.initialize_run(config)

        # In notebooks/tasks:
        config = ctx.get_config()
        ctx.update_task_status("inference", "running")
        ctx.set_output("run_id", run_id)
        ctx.update_task_status("inference", "completed")
    """

    # Valid status values
    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    # Task states
    TASK_PENDING = "pending"
    TASK_RUNNING = "running"
    TASK_COMPLETED = "completed"
    TASK_FAILED = "failed"
    TASK_SKIPPED = "skipped"

    def __init__(
        self,
        spark: SparkSession,
        catalog_name: str,
        run_id: str | None = None,
    ):
        """
        Initialize RunContext.

        Args:
            spark: SparkSession instance.
            catalog_name: Unity Catalog name.
            run_id: Optional run ID. If not provided, will be generated
                    or retrieved from Databricks job context.
        """
        self.spark = spark
        self.catalog_name = catalog_name
        self.table_name = f"{catalog_name}.metadata.pipeline_runs"
        self._run_id = run_id
        self._config: dict[str, Any] | None = None
        self._task_state: dict[str, str] | None = None
        self._outputs: dict[str, Any] | None = None

    @property
    def run_id(self) -> str:
        """Get the run ID, initializing if needed."""
        if self._run_id is None:
            self._run_id = self._get_or_create_run_id()
        return self._run_id

    def _get_or_create_run_id(self) -> str:
        """
        Get run_id from Databricks job context or create new.

        In Databricks job context, uses the job run ID.
        Otherwise, generates a new UUID.
        """
        try:
            # Try to get from Databricks job context
            # Note: dbutils is only available in notebook context
            from dbutils import dbutils  # type: ignore

            # Try to get job run ID from task context
            job_context = dbutils.jobs.context()  # type: ignore
            if job_context and "taskRunId" in job_context:
                return f"jobrun_{job_context['taskRunId']}"
        except (ImportError, NameError, AttributeError):
            pass

        # Generate new UUID if not in job context
        return str(uuid.uuid4())

    def _get_current_timestamp(self) -> datetime:
        """Get current timestamp."""
        return datetime.utcnow()

    def _serialize_json(self, data: dict[str, Any]) -> str:
        """Serialize dictionary to JSON string."""
        return json.dumps(data, default=str)

    def _deserialize_json(self, data: str) -> dict[str, Any]:
        """Deserialize JSON string to dictionary."""
        return json.loads(data) if data else {}

    def run_exists(self) -> bool:
        """Check if the run record exists."""
        try:
            result = self.spark.sql(
                f"SELECT run_id FROM {self.table_name} WHERE run_id = '{self.run_id}'"
            )
            return result.count() > 0
        except Exception:
            return False

    def initialize_run(
        self,
        config: dict[str, Any],
        job_id: str | None = None,
        job_run_id: str | None = None,
    ) -> None:
        """
        Create a new run record with initial configuration.

        Args:
            config: Configuration dictionary for the run.
            job_id: Optional Databricks job ID.
            job_run_id: Optional Databricks job run ID.
        """
        now = self._get_current_timestamp()

        # Create initial task state for all known tasks
        initial_task_state = {
            "init_catalog": self.TASK_PENDING,
            "inference": self.TASK_PENDING,
            "evaluation": self.TASK_PENDING,
            "regression": self.TASK_PENDING,
            "alert": self.TASK_PENDING,
            "testgen": self.TASK_PENDING,
        }

        # Insert new run record
        self.spark.sql(f"""
            INSERT INTO {self.table_name}
            (run_id, job_id, job_run_id, status, config, task_state, outputs, created_at, updated_at)
            VALUES (
                '{self.run_id}',
                {f"'{job_id}'" if job_id else "NULL"},
                {f"'{job_run_id}'" if job_run_id else "NULL"},
                '{self.STATUS_PENDING}',
                '{self._serialize_json(config).replace("'", "''")}',
                '{self._serialize_json(initial_task_state).replace("'", "''")}',
                '{{}}',
                TIMESTAMP '{now.isoformat()}',
                TIMESTAMP '{now.isoformat()}'
            )
        """)

        self._config = config
        self._task_state = initial_task_state
        self._outputs = {}

        logger.info(f"Initialized run {self.run_id}")

    def load_run(self) -> None:
        """Load existing run record from table."""
        result = self.spark.sql(
            f"SELECT config, task_state, outputs FROM {self.table_name} WHERE run_id = '{self.run_id}'"
        )

        if result.count() == 0:
            raise ValueError(f"Run {self.run_id} not found")

        row = result.first()
        self._config = self._deserialize_json(row.config)
        self._task_state = self._deserialize_json(row.task_state)
        self._outputs = self._deserialize_json(row.outputs)

        logger.info(f"Loaded run {self.run_id}")

    def get_config(self) -> dict[str, Any]:
        """
        Get configuration for the run.

        Returns:
            Configuration dictionary.
        """
        if self._config is None:
            self.load_run()
        return self._config.copy()

    def update_task_status(self, task_name: str, status: str) -> None:
        """
        Update task status in run record.

        Args:
            task_name: Name of the task (e.g., 'inference', 'evaluation').
            status: Status value (pending/running/completed/failed/skipped).
        """
        if self._task_state is None:
            self.load_run()

        self._task_state[task_name] = status
        now = self._get_current_timestamp()

        self.spark.sql(f"""
            UPDATE {self.table_name}
            SET task_state = '{self._serialize_json(self._task_state).replace("'", "''")}',
                updated_at = TIMESTAMP '{now.isoformat()}'
            WHERE run_id = '{self.run_id}'
        """)

        logger.info(f"Updated task {task_name} to {status}")

    def start_run(self) -> None:
        """Mark the run as started."""
        now = self._get_current_timestamp()
        self.spark.sql(f"""
            UPDATE {self.table_name}
            SET status = '{self.STATUS_RUNNING}',
                updated_at = TIMESTAMP '{now.isoformat()}'
            WHERE run_id = '{self.run_id}'
        """)
        logger.info(f"Run {self.run_id} started")

    def complete_run(self, final_status: str = STATUS_COMPLETED) -> None:
        """
        Mark the run as completed.

        Args:
            final_status: Final status (completed/failed).
        """
        now = self._get_current_timestamp()
        self.spark.sql(f"""
            UPDATE {self.table_name}
            SET status = '{final_status}',
                updated_at = TIMESTAMP '{now.isoformat()}',
                completed_at = TIMESTAMP '{now.isoformat()}'
            WHERE run_id = '{self.run_id}'
        """)
        logger.info(f"Run {self.run_id} completed with status {final_status}")

    def set_output(self, key: str, value: Any) -> None:
        """
        Store an output value in the run record.

        Args:
            key: Output key name.
            value: Output value (will be JSON serialized).
        """
        if self._outputs is None:
            self.load_run()

        self._outputs[key] = value
        now = self._get_current_timestamp()

        self.spark.sql(f"""
            UPDATE {self.table_name}
            SET outputs = '{self._serialize_json(self._outputs).replace("'", "''")}',
                updated_at = TIMESTAMP '{now.isoformat()}'
            WHERE run_id = '{self.run_id}'
        """)

        logger.info(f"Set output {key} for run {self.run_id}")

    def get_output(self, key: str) -> Any:
        """
        Retrieve an output value from the run record.

        Args:
            key: Output key name.

        Returns:
            Output value or None if not found.
        """
        if self._outputs is None:
            self.load_run()
        return self._outputs.get(key)

    def get_all_outputs(self) -> dict[str, Any]:
        """
        Get all outputs for the run.

        Returns:
            Dictionary of all outputs.
        """
        if self._outputs is None:
            self.load_run()
        return self._outputs.copy()

    def get_task_status(self, task_name: str) -> str:
        """
        Get status of a specific task.

        Args:
            task_name: Name of the task.

        Returns:
            Task status (pending/running/completed/failed/skipped).
        """
        if self._task_state is None:
            self.load_run()
        return self._task_state.get(task_name, self.TASK_PENDING)

    def get_all_task_statuses(self) -> dict[str, str]:
        """
        Get all task statuses.

        Returns:
            Dictionary of task names to statuses.
        """
        if self._task_state is None:
            self.load_run()
        return self._task_state.copy()

    def wait_for_task(
        self,
        task_name: str,
        poll_interval_seconds: int = 30,
        timeout_seconds: int = 3600,
    ) -> bool:
        """
        Wait for a task to complete.

        Args:
            task_name: Name of the task to wait for.
            poll_interval_seconds: How often to check status.
            timeout_seconds: Maximum time to wait.

        Returns:
            True if task completed successfully, False if failed or timeout.
        """
        import time

        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            # Reload run state
            self.load_run()
            status = self._task_state.get(task_name, self.TASK_PENDING)

            if status == self.TASK_COMPLETED:
                return True
            elif status == self.TASK_FAILED:
                logger.error(f"Task {task_name} failed")
                return False

            time.sleep(poll_interval_seconds)

        logger.error(f"Timeout waiting for task {task_name}")
        return False

    @classmethod
    def from_run_id(
        cls,
        spark: SparkSession,
        catalog_name: str,
        run_id: str,
    ) -> "RunContext":
        """
        Create a RunContext from an existing run ID.

        Args:
            spark: SparkSession instance.
            catalog_name: Unity Catalog name.
            run_id: Existing run ID.

        Returns:
            RunContext instance with loaded run data.
        """
        ctx = cls(spark, catalog_name, run_id)
        ctx.load_run()
        return ctx

    @classmethod
    def from_databricks_context(
        cls,
        spark: SparkSession,
        catalog_name: str,
        dbutils: Any,
    ) -> "RunContext":
        """
        Create a RunContext from Databricks notebook context.

        This is the preferred way to get a RunContext in a notebook,
        as it automatically retrieves the run_id from task parameters.

        Args:
            spark: SparkSession instance.
            catalog_name: Unity Catalog name.
            dbutils: Databricks dbutils object.

        Returns:
            RunContext instance.
        """
        # Try to get run_id from task parameters
        try:
            run_id = dbutils.widgets.get("run_id")
            if run_id:
                ctx = cls(spark, catalog_name, run_id)
                ctx.load_run()
                return ctx
        except Exception:
            pass

        # No run_id provided, create new run
        ctx = cls(spark, catalog_name)
        return ctx