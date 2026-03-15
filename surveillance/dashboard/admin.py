from django.contrib import admin

# Register your models here.
from .models import ActivityLog

admin.site.register(ActivityLog)