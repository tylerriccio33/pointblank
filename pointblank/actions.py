from __future__ import annotations

from datetime import datetime
from typing import Callable

import requests

__all__ = [
    "send_slack_notification",
]


def send_slack_notification(
    webhook_url: str | None = None,
    step_msg: str | None = None,
    summary_msg: str | None = None,
    debug: bool = False,
) -> Callable:
    r"""
    Creates a Slack notification function using a webhook URL.

    This function can be used in two ways:

    1. With `Actions` to notify about individual validation step failures
    2. With `FinalActions` to provide a summary after all validation steps complete

    The function creates a callable that sends notifications through a Slack webhook. Message
    formatting can be customized using templates for both individual steps and summary reports.

    Parameters
    ----------
    webhook_url
        The Slack webhook URL. If `None`, performs a dry run.
    step_msg
        Template string for step notifications. Some of the available variables include: `"{step}"`,
        `"{column}"`, `"{value}"`, `"{type}"`, `"{time}"`, `"{level}"`, etc. See the *Available
        Template Variables for Step Notifications* section below for more details.
    summary_msg
        Template string for summary notifications. Some of the available variables are:
        `"{n_steps}"`, `"{n_passing_steps}"`, `"{n_failing_steps}"`, `"{all_passed}"`,
        `"{highest_severity}"`, etc. See the *Available Template Variables for Summary
        Notifications* section below for more details.
    debug
        Print debug information and message previews.

    Available Template Variables for Step Notifications
    ---------------------------------------------------
    When creating a custom template for validation step alerts (`step_msg=`), the following
    templating strings can be used:

    - `"{step}"`: The step number.
    - `"{column}"`: The column name.
    - `"{value}"`: The value being compared (only available in certain validation steps).
    - `"{type}"`: The assertion type (e.g., `"col_vals_gt"`, etc.).
    - `"{level}"`: The severity level (`"warning"`, `"error"`, or `"critical"`).
    - `"{level_num}"`: The severity level as a numeric value (`30`, `40`, or `50`).
    - `"{autobrief}"`: A localized and brief statement of the expectation for the step.
    - `"{failure_text}"`: Localized text that explains how the validation step failed.
    - `"{time}"`: The time of the notification.

    Here's an example of how to construct a `step_msg=` template:

    ```python
    step_msg = \"\"\"🚨 *Validation Step Alert*
    • Step Number: {step}
    • Column: {column}
    • Test Type: {type}
    • Value Tested: {value}
    • Severity: {level} (level {level_num})
    • Brief: {autobrief}
    • Details: {failure_text}
    • Time: {time}\"\"\"
    ```

    This template will be filled with the relevant information when a validation step fails. The
    placeholders will be replaced with actual values when the Slack notification is sent.

    Available Template Variables for Summary Notifications
    ------------------------------------------------------
    When creating a custom template for a validation summary (`summary_msg=`), the following
    templating strings can be used:

    - `"{n_steps}"`: The total number of validation steps.
    - `"{n_passing_steps}"`: The number of validation steps where all test units passed.
    - `"{n_failing_steps}"`: The number of validation steps that had some failing test units.
    - `"{n_warning_steps}"`: The number of steps that exceeded a 'warning' threshold.
    - `"{n_error_steps}"`: The number of steps that exceeded an 'error' threshold.
    - `"{n_critical_steps}"`: The number of steps that exceeded a 'critical' threshold.
    - `"{all_passed}"`: Whether or not every validation step had no failing test units.
    - `"{highest_severity}"`: The highest severity level encountered during validation. This can be
    one of the following: `"warning"`, `"error"`, or `"critical"`, `"some failing"`, or
    `"all passed"`.
    - `"{tbl_row_count}"`: The number of rows in the target table.
    - `"{tbl_column_count}"`: The number of columns in the target table.
    - `"{tbl_name}"`: The name of the target table.
    - `"{validation_duration}"`: The duration of the validation in seconds.
    - `"{time}"`: The time of the notification.

    Here's an example of how to put together a `summary_msg=` template:

    ```python
    summary_msg = \"\"\"📊 *Validation Summary Report*
    *Overview*
    • Status: {highest_severity}
    • All Passed: {all_passed}
    • Total Steps: {n_steps}

    *Step Results*
    • Passing Steps: {n_passing_steps}
    • Failing Steps: {n_failing_steps}
    • Warning Level: {n_warning_steps}
    • Error Level: {n_error_steps}
    • Critical Level: {n_critical_steps}

    *Table Info*
    • Table Name: {tbl_name}
    • Row Count: {tbl_row_count}
    • Column Count: {tbl_column_count}

    *Timing*
    • Duration: {validation_duration}s
    • Completed: {time}\"\"\"
    ```

    This template will be filled with the relevant information when the validation summary is
    generated. The placeholders will be replaced with actual values when the Slack notification is
    sent.

    Returns
    -------
    Callable
        A function that sends notifications to Slack.

    Examples
    --------
    When using an Action with one or more validation steps, you typically provide callables that
    fire when a matched threshold of failed test units is exceeded. The callable can be
    a function or a lambda. The `send_slack_notification()` function creates a callable that sends
    a Slack notification when the validation step fails. Here is how it can be set up to work for
    multiple validation steps:

    ```python
    import pointblank as pb

    # Create a Slack notification function
    notify_slack = pb.send_slack_notification(
        webhook_url="https://hooks.slack.com/services/your/webhook/url"
    )
    # Create a validation plan
    validation = (
        pb.Validate(
            data=pb.load_dataset(dataset="game_revenue", tbl_type="duckdb"),
            thresholds=pb.Thresholds(warning=0.05, error=0.10, critical=0.15),
            actions=pb.Actions(critical=notify_slack),
        )
        .col_vals_regex(columns="player_id", pattern=r"[A-Z]{12}\d{3}")
        .col_vals_gt(columns="item_revenue", value=0.05)
        .col_vals_gt(columns="session_duration", value=15)
        .interrogate()
    )

    validation
    ```

    By placing the `notify_slack` function in the `Validate(actions=Actions(critical=))` argument,
    you can ensure that the notification is sent whenever the 'critical' threshold is reached (as
    set here, when 15% or more of the test units fail). The notification will include information
    about the validation step that triggered the alert.

    When using a `FinalActions` object, the notification will be sent after all validation steps
    have been completed. This is useful for providing a summary of the validation process. Here is
    an example of how to set up a summary notification:
    ```python
    import pointblank as pb

    # Create a Slack notification function
    notify_slack = pb.send_slack_notification(
        webhook_url="https://hooks.slack.com/services/your/webhook/url"
    )
    # Create a validation plan
    validation = (
        pb.Validate(
            data=pb.load_dataset(dataset="game_revenue", tbl_type="duckdb"),
            thresholds=pb.Thresholds(warning=0.05, error=0.10, critical=0.15),
            final_actions=pb.FinalActions(notify_slack),
        )
        .col_vals_regex(columns="player_id", pattern=r"[A-Z]{12}\d{3}")
        .col_vals_gt(columns="item_revenue", value=0.05)
        .col_vals_gt(columns="session_duration", value=15)
        .interrogate()
    )
    ```

    In this case, the same `notify_slack` function is used, but it is placed in
    `Validate(final_actions=FinalActions())`. This results in the summary notification being sent
    after all validation steps are completed, regardless of whether any steps failed or not.

    This simplicity is possible because the `send_slack_notification()` function creates a callable
    that can be used in both contexts. The function will automatically determine whether to send a
    step notification or a summary notification based on the context in which it is called.

    We can customize the message templates for both step and summary notifications. In that way,
    it's possible to create a more informative and visually appealing message. For example, we can
    use Markdown formatting to make the message more readable and visually appealing. Here is an
    example of how to customize the templates:

    ```python
    import pointblank as pb
    # Create a Slack notification function

    notify_slack = pb.send_slack_notification(
        webhook_url="https://hooks.slack.com/services/your/webhook/url",
        step_msg=\"\"\"
        🚨 *Validation Step Alert*
        • Step Number: {step}
        • Column: {column}
        • Test Type: {type}
        • Value Tested: {value}
        • Severity: {level} (level {level_num})
        • Brief: {autobrief}
        • Details: {failure_text}
        • Time: {time}\"\"\",
        summary_msg=\"\"\"
        📊 *Validation Summary Report*
        *Overview*
        • Status: {highest_severity}
        • All Passed: {all_passed}
        • Total Steps: {n_steps}

        *Step Results*
        • Passing Steps: {n_passing_steps}
        • Failing Steps: {n_failing_steps}
        • Warning Level: {n_warning_steps}
        • Error Level: {n_error_steps}
        • Critical Level: {n_critical_steps}

        *Table Info*
        • Table Name: {tbl_name}
        • Row Count: {tbl_row_count}
        • Column Count: {tbl_column_count}

        *Timing*
        • Duration: {validation_duration}s
        • Completed: {time}\"\"\",
    )

    # Create a validation plan
    validation = (
        pb.Validate(
            data=pb.load_dataset(dataset="game_revenue", tbl_type="duckdb"),
            thresholds=pb.Thresholds(warning=0.05, error=0.10, critical=0.15),
            actions=pb.Actions(default=notify_slack),
            final_actions=pb.FinalActions(notify_slack),
        )
        .col_vals_regex(columns="player_id", pattern=r"[A-Z]{12}\d{3}")
        .col_vals_gt(columns="item_revenue", value=0.05)
        .col_vals_gt(columns="session_duration", value=15)
        .interrogate()
    )
    ```

    In this example, we have customized the templates for both step and summary notifications. The
    step notification includes details about the validation step, including the step number,
    column name, test type, value tested, severity level, brief description, and time of the
    notification. The summary notification includes an overview of the validation process,
    including the status, number of steps, passing and failing steps, table information, and
    timing details.
    """
    # Default templates
    default_step = """*Data Validation Alert*
• Type: {type}
• Level: {level}
• Step: {step}
• Column: {column}
• Time: {time}"""

    default_summary = """*Data Validation Summary*
• Highest Severity: {highest_severity}
• Total Steps: {n_steps}
• Failed Steps: {n_failing_steps}
• Time: {time}"""

    # Use provided templates or defaults
    step = step_msg or default_step
    summary = summary_msg or default_summary

    # If called directly (not as a callback), show template preview
    if webhook_url is None and debug:
        # Sample data for previews
        step_data = {
            "type": "col_vals_gt",
            "level": "critical",
            "step": 1,
            "column": "column_name",
            "time": datetime.now(),
        }

        summary_data = {
            "highest_severity": "critical",
            "n_steps": 5,
            "n_failing_steps": 2,
            "time": datetime.now(),
        }

        print("\n=== Step Notification Preview ===")
        print(step.format(**step_data))

        print("\n=== Summary Notification Preview ===")
        print(summary.format(**summary_data))

        return None

    def notify():
        try:
            message = None

            # Try to get summary data first
            try:
                from pointblank import get_validation_summary

                summary_data = get_validation_summary()
                if summary_data is not None:
                    summary_data["time"] = datetime.now()
                    message = summary.format(**summary_data)
            except ImportError:
                pass

            # If no summary, try step data
            if message is None:
                try:
                    from pointblank import get_action_metadata

                    metadata = get_action_metadata()
                    if metadata is not None:
                        metadata["time"] = datetime.now()
                        message = step.format(**metadata)
                except ImportError:
                    pass

            # If still no message, raise error
            if message is None:
                raise ValueError("No validation data available")

            if debug:
                print("\n=== Debug Info ===")
                print("Message Content:")
                print(message)
                print("==================\n")

            if webhook_url is None:
                print("\n=== Dry Run ===")
                print("Would send to Slack:")
                print(message)
                print("==============\n")
                return

            payload = {"text": message}
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()

            if debug:
                print("\n=== Slack Response ===")
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")
                print("=====================\n")

        except Exception as e:
            error_msg = f"Failed to send Slack notification: {str(e)}"
            if debug:
                print("\n=== Error Details ===")
                print(error_msg)
                print("===================\n")
            else:
                print(error_msg)

    return notify
