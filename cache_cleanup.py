#!/usr/bin/env python3
import subprocess
import time
import schedule
import os
import shutil

# Define notification URL
NTFY_TOPIC = "ntfy.sh/ev-cvat-cron-status"

DISK_THRESHOLD = 75  # Set the disk usage threshold


def send_notification(message):
    """Send a notification using curl to the ntfy service."""
    print("Sending notification:", message)
    try:
        subprocess.run(["curl", "-d", message, NTFY_TOPIC], check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to send notification:", e)


def check_disk_usage():
    """Check disk utilization and trigger cleanup if threshold is exceeded."""
    total, used, free = shutil.disk_usage("/")
    usage_percentage = (used / total) * 100

    print(f"Current disk usage: {usage_percentage:.2f}%")

    if usage_percentage > DISK_THRESHOLD:
        print(f"Disk usage exceeds {DISK_THRESHOLD}%. Running cleanup...")
        send_notification(
            f"Disk usage {usage_percentage:.2f}% exceeded {DISK_THRESHOLD}%. Initiating cleanup."
        )
        run_cleanup()
    else:
        print("Disk usage is within limits. No cleanup needed.")


def run_cleanup():
    """Run the cleanup process: stop Docker, remove cache, and start Docker services."""
    print("Starting CVAT cache cleanup process...")
    # 1. Stop the Docker services.
    try:
        print("Stopping CVAT Docker services...")
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                "docker-compose.yml",
                "-f",
                "components/serverless/docker-compose.serverless.yml",
                "down",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        send_notification("CVAT cache clear cronjob FAILED ❌ (Docker Down Failed)")
        print("Error stopping docker services:", e)
        return

    # 2. Remove cache files.
    try:
        print("Removing CVAT cache database...")
        subprocess.run(
            [
                "sudo",
                "rm",
                "-rf",
                "/var/lib/docker/volumes/cvat_cvat_cache_db/_data/db",
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        send_notification("CVAT cache clear cronjob FAILED ❌ (Cache Removal Failed)")
        print("Error removing cache database:", e)
        return

    # 3. Start the Docker services.
    try:
        print("Starting CVAT Docker services...")
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                "docker-compose.yml",
                "-f",
                "components/serverless/docker-compose.serverless.yml",
                "up",
                "-d",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        send_notification("CVAT cache clear cronjob FAILED ❌ (Docker Up Failed)")
        print("Error starting docker services:", e)
        return

    # 4. Success notification.
    send_notification("CVAT cache clear cronjob successful 😀")
    print("Process completed successfully.")


if __name__ == "__main__":
    # Schedule disk check every 10 minutes
    INTERVAL = 10
    check_disk_usage()
    schedule.every(INTERVAL).minutes.do(check_disk_usage)
    print(f"Scheduled disk usage check every {INTERVAL} minutes.")

    print("Waiting for scheduled checks...")
    # Continuous loop to run scheduled jobs.
    while True:
        schedule.run_pending()
        # sleep for 10 secs
        time.sleep(10)
