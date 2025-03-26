from django.urls import path
from . import views  # Import views from the heatmap_app

urlpatterns = [
    path('', views.index, name='index'),
    path('heatmap_image/', views.heatmap_image, name='heatmap_image'),
    path('detail/', views.detail, name='detail'),
    path('historical_plot_image/', views.historical_plot_image, name='historical_plot_image'),
    path('line_plot_image/', views.line_plot_image, name='line_plot_image'),
    path('rebasing/', views.rebasing, name='rebasing'),
    path('rebasing_image/', views.rebasing_image, name='rebasing_image'),
]
