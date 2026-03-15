from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse, Http404
import os
import mimetypes
from django.conf import settings
from .ai_engine import process_video
from .models import ActivityLog, SystemStatus

SPEED_MODE_CHOICES = {
    "fast": "Fast",
    "balanced": "Balanced",
    "accurate": "Accurate",
}


def _stream_video(file_path, range_header=None):
    """Yield a file in chunks, honouring an optional HTTP Range header."""
    file_size = os.path.getsize(file_path)
    chunk = 8 * 1024 * 1024  # 8 MB chunks

    if range_header:
        # Parse 'bytes=start-end'
        byte_range = range_header.strip().replace("bytes=", "")
        start_str, _, end_str = byte_range.partition("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)
    else:
        start, end = 0, file_size - 1

    content_length = end - start + 1

    def file_iterator():
        remaining = content_length
        with open(file_path, "rb") as fh:
            fh.seek(start)
            while remaining > 0:
                data = fh.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    return file_iterator, start, end, file_size, content_length


@login_required(login_url="login")
def stream_media(request, path):
    """Serve files from MEDIA_ROOT with proper Range-request support."""
    safe_path = os.path.normpath(path).lstrip("/\\")
    file_path = os.path.join(settings.MEDIA_ROOT, safe_path)
    if not os.path.isfile(file_path):
        raise Http404("Media file not found")

    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "application/octet-stream"
    file_size = os.path.getsize(file_path)
    range_header = request.META.get("HTTP_RANGE")

    iterator, start, end, file_size, content_length = _stream_video(file_path, range_header)
    status = 206 if range_header else 200

    response = StreamingHttpResponse(
        iterator(),
        status=status,
        content_type=mime_type,
    )
    response["Accept-Ranges"] = "bytes"
    response["Content-Length"] = str(content_length)
    if range_header:
        response["Content-Range"] = f"bytes {start}-{end}/{file_size}"
    return response


@login_required(login_url="login")
def home(request):
    user_logs = ActivityLog.objects.filter(user=request.user)

    total_videos = user_logs.values("video_name").distinct().count()
    safe_count = user_logs.filter(status__iexact="Safe").count()
    unsafe_count = user_logs.filter(status__iexact="Unsafe").count()
    incidents = user_logs.filter(incident=True).count()

    recent_alerts = user_logs.filter(incident=True).order_by("-timestamp")[:5]

    system_status = SystemStatus.objects.order_by("-updated_at").first()
    if not system_status:
        system_status = SystemStatus.objects.create()

    overall_health = (
        "Healthy"
        if system_status.camera_online
        and system_status.ai_model_loaded
        and system_status.server_online
        else "Warning"
    )

    context = {
        "total_videos": total_videos,
        "safe_count": safe_count,
        "unsafe_count": unsafe_count,
        "incidents": incidents,
        "recent_alerts": recent_alerts,
        "ai_model_status": "Online" if system_status.ai_model_loaded else "Offline",
        "camera_status": "Online" if system_status.camera_online else "Offline",
        "server_status": "Online" if system_status.server_online else "Offline",
        "system_health": overall_health,
        "detection_sensitivity": system_status.sensitivity,
        "allowed_activities": [
            activity.strip()
            for activity in system_status.allowed_activities.split(",")
            if activity.strip()
        ],
    }

    return render(request, "home.html", context)


@login_required(login_url="login")
def profile(request):
    return render(request, "profile.html")


@login_required(login_url="login")
def settings_page(request):
    return render(request, "settings.html")


def register(request):
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        username = request.POST["username"]
        email = request.POST["email"]
        password = request.POST["password"]

        user = User.objects.create_user(
            username=username,
            email=email,
            password=password
        )
        user.save()

        return redirect("login")

    return render(request, "register.html")


def login_view(request):
    if request.user.is_authenticated:
        return redirect("home")

    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect("home")

    return render(request, "login.html")


def logout_view(request):
    logout(request)
    return redirect("login")


@login_required(login_url="login")
def monitor(request):
    if request.method == "POST":
        video = request.FILES.get("video")
        selected_speed_mode = request.POST.get("speed_mode", "balanced").lower()
        if selected_speed_mode not in SPEED_MODE_CHOICES:
            selected_speed_mode = "balanced"

        if not video:
            return render(request, "monitor.html", {
                "error": "Please choose a video before running detection.",
                "selected_speed_mode": selected_speed_mode,
            })

        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
        output_dir = os.path.join(settings.MEDIA_ROOT, "output")

        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        upload_path = os.path.join(upload_dir, video.name)

        with open(upload_path, "wb+") as destination:
            for chunk in video.chunks():
                destination.write(chunk)

        output_file = "result_" + video.name
        output_path = os.path.join(output_dir, output_file)

        ai_result = process_video(upload_path, output_path, speed_mode=selected_speed_mode)

        ActivityLog.objects.create(
            user=request.user,
            video_name=video.name,
            activity_type=ai_result["primary_activity"],
            status=ai_result["status"],
            confidence=ai_result["confidence_score"],
            incident=ai_result["has_unsafe_activity"],
            processed_video="output/" + output_file
        )

        from django.urls import reverse
        stream_url = reverse("stream_media", kwargs={"path": "output/" + output_file})
        processed_count = ActivityLog.objects.filter(user=request.user).values("video_name").distinct().count()

        context = {
            "result": "Processing Completed",
            "video_name": video.name,
            "output_video": stream_url,
            "processed_count": processed_count,
            "safe_count": ai_result["safe_count"],
            "unsafe_count": ai_result["unsafe_count"],
            "safe_activities": ai_result["safe_activities"],
            "unsafe_activities": ai_result["unsafe_activities"],
            "primary_activity": ai_result["primary_activity"],
            "status": ai_result["status"],
            "confidence": ai_result["confidence_score"],
            "classified_frames": ai_result["classified_frames"],
            "no_pose_frames": ai_result["no_pose_frames"]
            ,"selected_speed_mode": selected_speed_mode,
            "selected_speed_mode_label": SPEED_MODE_CHOICES[selected_speed_mode],
        }

        return render(request, "monitor.html", context)

    return render(request, "monitor.html", {"selected_speed_mode": "balanced"})


@login_required(login_url="login")
def history(request):
    logs = ActivityLog.objects.filter(user=request.user).order_by("-timestamp")
    return render(request, "history.html", {"logs": logs})