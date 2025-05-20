# File: booking.py

from datetime import datetime, timedelta

def generate_google_calendar_link(
    appointment_title: str, 
    details: str, 
    start_time: datetime, 
    duration_minutes: int = 30
) -> str:
    """
    Generates a Google Calendar event creation link.

    Args:
        appointment_title (str): Title for the calendar event.
        details (str): Description/details for the event.
        start_time (datetime): Event start time (UTC).
        duration_minutes (int, optional): Duration of event. Defaults to 30 minutes.

    Returns:
        str: Google Calendar event creation link.
    """
    end_time = start_time + timedelta(minutes=duration_minutes)
    start_str = start_time.strftime('%Y%m%dT%H%M%SZ')
    end_str = end_time.strftime('%Y%m%dT%H%M%SZ')
    return (
        f"https://calendar.google.com/calendar/u/0/r/eventedit?"
        f"text={appointment_title}&details={details}&dates={start_str}/{end_str}"
    )
