from django.db import models
from django.contrib.auth.models import User


class ActivityLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name="activity_logs")
    video_name = models.CharField(max_length=255, default="video.mp4")
    activity_type = models.CharField(max_length=100, default="Unknown")
    status = models.CharField(max_length=50, default="Safe")
    confidence = models.FloatField(default=0.0)
    incident = models.BooleanField(default=False)
    processed_video = models.CharField(max_length=255, default="output/default.mp4")
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        owner = self.user.username if self.user else "unknown"
        return f"{owner} | {self.video_name} | {self.activity_type} | {self.status}"


class SystemStatus(models.Model):
    camera_online = models.BooleanField(default=True)
    ai_model_loaded = models.BooleanField(default=True)
    server_online = models.BooleanField(default=True)
    sensitivity = models.CharField(max_length=20, default="Medium")
    allowed_activities = models.TextField(
        default="Walking,Running,Jogging,Yoga,Sitting"
    )
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"System Status - {self.updated_at:%Y-%m-%d %H:%M}"