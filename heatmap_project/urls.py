from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('heatmap_app.urls')),  # Route all root URLs to heatmap_app.urls
]
