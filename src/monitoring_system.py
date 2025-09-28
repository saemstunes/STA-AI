import time
import psutil
import GPUtil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import threading
import statistics
from dataclasses import dataclass, asdict
import json
import os
import hashlib

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    print("Warning: prometheus_client not available. Monitoring will be limited.")

@dataclass
class InferenceMetrics:
    model_name: str
    processing_time_ms: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    success: bool
    user_id: str
    conversation_id: Optional[str]
    timestamp: datetime
    error_message: Optional[str] = None
    query_length: int = 0
    response_length: int = 0
    model_hash: Optional[str] = None
    cache_hit: bool = False

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_percent: float
    gpu_usage_percent: Optional[float]
    gpu_memory_percent: Optional[float]
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    active_threads: int

class ComprehensiveMonitor:
    def __init__(self, prometheus_port: int = 8001, metrics_retention_hours: int = 24):
        self.inference_metrics: List[InferenceMetrics] = []
        self.system_metrics: List[SystemMetrics] = []
        self.alerts: List[Dict] = []
        self.start_time = datetime.now()
        self.prometheus_port = prometheus_port
        self.metrics_retention_hours = metrics_retention_hours
        
        self.monitoring_active = False
        self.monitoring_thread = None
        self.alert_callbacks = []
        
        self.prometheus_metrics = {}
        
        self.setup_logging()
        
        if PROMETHEUS_AVAILABLE:
            self.setup_prometheus_metrics()
        
        self.start_monitoring()
    
    def setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def setup_prometheus_metrics(self):
        try:
            self.prometheus_metrics = {
                'inference_requests_total': Counter(
                    'ai_inference_requests_total',
                    'Total inference requests',
                    ['model', 'status', 'cache_status']
                ),
                'inference_duration_seconds': Histogram(
                    'ai_inference_duration_seconds',
                    'Inference duration in seconds',
                    ['model'],
                    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
                ),
                'inference_tokens_total': Counter(
                    'ai_inference_tokens_total',
                    'Total tokens processed',
                    ['model', 'type']
                ),
                'system_cpu_percent': Gauge(
                    'ai_system_cpu_percent',
                    'System CPU percentage'
                ),
                'system_memory_percent': Gauge(
                    'ai_system_memory_percent', 
                    'System memory percentage'
                ),
                'system_memory_used_gb': Gauge(
                    'ai_system_memory_used_gb',
                    'System memory used in GB'
                ),
                'system_disk_percent': Gauge(
                    'ai_system_disk_percent',
                    'System disk usage percentage'
                ),
                'active_requests': Gauge(
                    'ai_active_requests',
                    'Currently active requests'
                ),
                'error_rate_percent': Gauge(
                    'ai_error_rate_percent',
                    'Error rate percentage'
                ),
                'response_time_95th_percentile': Gauge(
                    'ai_response_time_95th_percentile',
                    '95th percentile response time in seconds'
                ),
                'throughput_requests_per_minute': Gauge(
                    'ai_throughput_requests_per_minute',
                    'Requests per minute'
                ),
                'cache_hit_rate_percent': Gauge(
                    'ai_cache_hit_rate_percent',
                    'Cache hit rate percentage'
                )
            }
            
            start_http_server(self.prometheus_port)
            self.logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
            
        except Exception as e:
            self.logger.warning(f"Could not start Prometheus server: {e}")
    
    def start_monitoring(self):
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Background monitoring started")
    
    def _monitoring_loop(self):
        iteration = 0
        while self.monitoring_active:
            try:
                system_metrics = self.get_system_metrics()
                self.system_metrics.append(system_metrics)
                
                if PROMETHEUS_AVAILABLE:
                    self.update_prometheus_gauges(system_metrics)
                
                self.check_alerts(system_metrics)
                
                self.cleanup_old_metrics()
                
                if iteration % 12 == 0:
                    self.log_system_summary()
                
                iteration += 1
                time.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)
    
    def get_system_metrics(self) -> SystemMetrics:
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024 ** 3)
            
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            net_io = psutil.net_io_counters()
            
            gpu_usage = None
            gpu_memory = None
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = sum(gpu.load * 100 for gpu in gpus) / len(gpus)
                    gpu_memory = sum(gpu.memoryUtil * 100 for gpu in gpus) / len(gpus)
            except Exception:
                pass
            
            active_connections = len(psutil.net_connections())
            active_threads = threading.active_count()
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                disk_percent=disk_percent,
                gpu_usage_percent=gpu_usage,
                gpu_memory_percent=gpu_memory,
                network_bytes_sent=net_io.bytes_sent,
                network_bytes_recv=net_io.bytes_recv,
                active_connections=active_connections,
                active_threads=active_threads
            )
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_gb=0.0,
                disk_percent=0.0,
                gpu_usage_percent=None,
                gpu_memory_percent=None,
                network_bytes_sent=0,
                network_bytes_recv=0,
                active_connections=0,
                active_threads=0
            )
    
    def update_prometheus_gauges(self, system_metrics: SystemMetrics):
        try:
            self.prometheus_metrics['system_cpu_percent'].set(system_metrics.cpu_percent)
            self.prometheus_metrics['system_memory_percent'].set(system_metrics.memory_percent)
            self.prometheus_metrics['system_memory_used_gb'].set(system_metrics.memory_used_gb)
            self.prometheus_metrics['system_disk_percent'].set(system_metrics.disk_percent)
            
            error_rate = self.get_error_rate()
            self.prometheus_metrics['error_rate_percent'].set(error_rate)
            
            response_time_95th = self.get_response_time_percentile(0.95)
            self.prometheus_metrics['response_time_95th_percentile'].set(response_time_95th)
            
            throughput = self.get_throughput()
            self.prometheus_metrics['throughput_requests_per_minute'].set(throughput)
            
            cache_hit_rate = self.get_cache_hit_rate()
            self.prometheus_metrics['cache_hit_rate_percent'].set(cache_hit_rate)
            
        except Exception as e:
            self.logger.error(f"Error updating Prometheus gauges: {e}")
    
    def record_inference(self, metrics: Dict):
        try:
            inference_metrics = InferenceMetrics(
                model_name=metrics.get('model_name', 'unknown'),
                processing_time_ms=metrics.get('processing_time_ms', 0),
                input_tokens=metrics.get('input_tokens', 0),
                output_tokens=metrics.get('output_tokens', 0),
                total_tokens=metrics.get('total_tokens', 0),
                success=metrics.get('success', False),
                user_id=metrics.get('user_id', 'anonymous'),
                conversation_id=metrics.get('conversation_id'),
                timestamp=metrics.get('timestamp', datetime.now()),
                error_message=metrics.get('error_message'),
                query_length=metrics.get('query_length', 0),
                response_length=metrics.get('response_length', 0),
                model_hash=metrics.get('model_hash'),
                cache_hit=metrics.get('cache_hit', False)
            )
            
            self.inference_metrics.append(inference_metrics)
            
            if PROMETHEUS_AVAILABLE:
                status = 'success' if inference_metrics.success else 'error'
                cache_status = 'hit' if inference_metrics.cache_hit else 'miss'
                
                self.prometheus_metrics['inference_requests_total'].labels(
                    model=inference_metrics.model_name,
                    status=status,
                    cache_status=cache_status
                ).inc()
                
                self.prometheus_metrics['inference_duration_seconds'].labels(
                    model=inference_metrics.model_name
                ).observe(inference_metrics.processing_time_ms / 1000.0)
                
                self.prometheus_metrics['inference_tokens_total'].labels(
                    model=inference_metrics.model_name,
                    type='input'
                ).inc(inference_metrics.input_tokens)
                
                self.prometheus_metrics['inference_tokens_total'].labels(
                    model=inference_metrics.model_name,
                    type='output'
                ).inc(inference_metrics.output_tokens)
            
        except Exception as e:
            self.logger.error(f"Error recording inference metrics: {e}")
    
    def get_recent_metrics(self, minutes: int = 5) -> List[InferenceMetrics]:
        cutoff = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.inference_metrics if m.timestamp > cutoff]
    
    def get_average_response_time(self, minutes: int = 30) -> float:
        recent_metrics = self.get_recent_metrics(minutes)
        successful_metrics = [m for m in recent_metrics if m.success]
        
        if not successful_metrics:
            return 0.0
        
        return sum(m.processing_time_ms for m in successful_metrics) / len(successful_metrics)
    
    def get_response_time_percentile(self, percentile: float, minutes: int = 30) -> float:
        recent_metrics = self.get_recent_metrics(minutes)
        successful_metrics = [m for m in recent_metrics if m.success]
        
        if not successful_metrics:
            return 0.0
        
        processing_times = [m.processing_time_ms for m in successful_metrics]
        processing_times.sort()
        
        index = int(percentile * len(processing_times))
        return processing_times[index] if index < len(processing_times) else processing_times[-1]
    
    def get_error_rate(self, minutes: int = 30) -> float:
        recent_metrics = self.get_recent_metrics(minutes)
        if not recent_metrics:
            return 0.0
        
        errors = sum(1 for m in recent_metrics if not m.success)
        return (errors / len(recent_metrics)) * 100
    
    def get_throughput(self, minutes: int = 5) -> float:
        recent_metrics = self.get_recent_metrics(minutes)
        if not recent_metrics or minutes == 0:
            return 0.0
        
        return len(recent_metrics) / minutes
    
    def get_cache_hit_rate(self, minutes: int = 30) -> float:
        recent_metrics = self.get_recent_metrics(minutes)
        if not recent_metrics:
            return 0.0
        
        cache_hits = sum(1 for m in recent_metrics if m.cache_hit)
        return (cache_hits / len(recent_metrics)) * 100
    
    def get_uptime(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()
    
    def check_alerts(self, system_metrics: SystemMetrics):
        current_alerts = []
        
        if system_metrics.cpu_percent > 85:
            current_alerts.append({
                'level': 'warning' if system_metrics.cpu_percent < 95 else 'critical',
                'message': f"High CPU usage: {system_metrics.cpu_percent:.1f}%",
                'metric': 'cpu_percent',
                'value': system_metrics.cpu_percent,
                'threshold': 85
            })
        
        if system_metrics.memory_percent > 90:
            current_alerts.append({
                'level': 'warning' if system_metrics.memory_percent < 95 else 'critical',
                'message': f"High memory usage: {system_metrics.memory_percent:.1f}%",
                'metric': 'memory_percent',
                'value': system_metrics.memory_percent,
                'threshold': 90
            })
        
        if system_metrics.disk_percent > 90:
            current_alerts.append({
                'level': 'critical',
                'message': f"High disk usage: {system_metrics.disk_percent:.1f}%",
                'metric': 'disk_percent',
                'value': system_metrics.disk_percent,
                'threshold': 90
            })
        
        error_rate = self.get_error_rate(10)
        if error_rate > 5:
            current_alerts.append({
                'level': 'critical',
                'message': f"High error rate: {error_rate:.1f}%",
                'metric': 'error_rate',
                'value': error_rate,
                'threshold': 5
            })
        
        response_time_95th = self.get_response_time_percentile(0.95, 10)
        if response_time_95th > 10000:
            current_alerts.append({
                'level': 'warning',
                'message': f"Slow response time (95th): {response_time_95th/1000:.1f}s",
                'metric': 'response_time_95th',
                'value': response_time_95th,
                'threshold': 10000
            })
        
        throughput = self.get_throughput(5)
        if throughput > 100:
            current_alerts.append({
                'level': 'warning',
                'message': f"High throughput: {throughput:.1f} requests/minute",
                'metric': 'throughput',
                'value': throughput,
                'threshold': 100
            })
        
        for alert in current_alerts:
            if self.is_new_alert(alert):
                self.trigger_alert(alert)
                self.alerts.append(alert)
    
    def is_new_alert(self, alert: Dict) -> bool:
        recent_threshold = datetime.now() - timedelta(minutes=5)
        recent_alerts = [a for a in self.alerts 
                        if a['metric'] == alert['metric'] 
                        and a.get('timestamp', datetime.min) > recent_threshold]
        return len(recent_alerts) == 0
    
    def trigger_alert(self, alert: Dict):
        alert['timestamp'] = datetime.now()
        alert['alert_id'] = hashlib.md5(f"{alert['metric']}_{alert['timestamp']}".encode()).hexdigest()[:8]
        
        self.logger.warning(f"ALERT {alert['level'].upper()}: {alert['message']} (ID: {alert['alert_id']})")
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback):
        self.alert_callbacks.append(callback)
    
    def log_system_summary(self):
        summary = self.get_performance_summary(timedelta(minutes=5))
        
        if summary:
            self.logger.info(
                f"System Summary - "
                f"Requests: {summary['total_requests']}, "
                f"Error Rate: {summary['error_rate_percent']:.1f}%, "
                f"Avg Response: {summary['avg_response_time_ms']:.0f}ms, "
                f"CPU: {summary['system_metrics']['avg_cpu_percent']:.1f}%, "
                f"Cache Hit: {summary['cache_hit_rate_percent']:.1f}%"
            )
    
    def get_performance_summary(self, time_window: timedelta) -> Dict[str, Any]:
        recent_metrics = self.get_recent_metrics(time_window.total_seconds() / 60)
        recent_system = [m for m in self.system_metrics 
                        if m.timestamp > datetime.now() - time_window]
        
        if not recent_metrics:
            return {}
        
        processing_times = [m.processing_time_ms for m in recent_metrics if m.success]
        error_rate = self.get_error_rate(time_window.total_seconds() / 60)
        cache_hit_rate = self.get_cache_hit_rate(time_window.total_seconds() / 60)
        
        summary = {
            'time_window': str(time_window),
            'total_requests': len(recent_metrics),
            'successful_requests': sum(1 for m in recent_metrics if m.success),
            'failed_requests': sum(1 for m in recent_metrics if not m.success),
            'error_rate_percent': error_rate,
            'avg_response_time_ms': statistics.mean(processing_times) if processing_times else 0,
            'p95_response_time_ms': self.get_response_time_percentile(0.95, time_window.total_seconds() / 60),
            'p99_response_time_ms': self.get_response_time_percentile(0.99, time_window.total_seconds() / 60),
            'requests_per_minute': len(recent_metrics) / (time_window.total_seconds() / 60),
            'total_tokens_processed': sum(m.total_tokens for m in recent_metrics),
            'avg_tokens_per_request': sum(m.total_tokens for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
            'cache_hit_rate_percent': cache_hit_rate,
            'unique_users': len(set(m.user_id for m in recent_metrics)),
            'system_metrics': {
                'avg_cpu_percent': statistics.mean([m.cpu_percent for m in recent_system]) if recent_system else 0,
                'avg_memory_percent': statistics.mean([m.memory_percent for m in recent_system]) if recent_system else 0,
                'max_cpu_percent': max([m.cpu_percent for m in recent_system]) if recent_system else 0,
                'max_memory_percent': max([m.memory_percent for m in recent_system]) if recent_system else 0
            }
        }
        
        return summary
    
    def cleanup_old_metrics(self):
        cutoff = datetime.now() - timedelta(hours=self.metrics_retention_hours)
        
        self.inference_metrics = [m for m in self.inference_metrics if m.timestamp > cutoff]
        self.system_metrics = [m for m in self.system_metrics if m.timestamp > cutoff]
        self.alerts = [a for a in self.alerts if a.get('timestamp', datetime.min) > cutoff - timedelta(hours=24)]
    
    def get_system_health(self) -> Dict[str, Any]:
        performance_summary = self.get_performance_summary(timedelta(minutes=30))
        
        health_status = "healthy"
        if performance_summary.get('error_rate_percent', 0) > 10:
            health_status = "degraded"
        elif performance_summary.get('error_rate_percent', 0) > 20:
            health_status = "unhealthy"
        
        return {
            'status': health_status,
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': self.get_uptime(),
            'performance': performance_summary,
            'alerts': {
                'total_24h': len([a for a in self.alerts if a.get('timestamp', datetime.min) > datetime.now() - timedelta(hours=24)]),
                'critical_24h': len([a for a in self.alerts if a.get('level') == 'critical' and a.get('timestamp', datetime.min) > datetime.now() - timedelta(hours=24)]),
                'warning_24h': len([a for a in self.alerts if a.get('level') == 'warning' and a.get('timestamp', datetime.min) > datetime.now() - timedelta(hours=24)])
            },
            'resources': asdict(self.get_system_metrics()) if self.system_metrics else {}
        }
    
    def stop_monitoring(self):
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Monitoring system stopped")
    
    def export_metrics(self, filename: str, time_window: timedelta = timedelta(hours=24)):
        try:
            metrics_data = {
                'export_timestamp': datetime.now().isoformat(),
                'time_window': str(time_window),
                'inference_metrics': [
                    asdict(m) for m in self.inference_metrics 
                    if m.timestamp > datetime.now() - time_window
                ],
                'system_metrics': [
                    asdict(m) for m in self.system_metrics 
                    if m.timestamp > datetime.now() - time_window
                ],
                'performance_summary': self.get_performance_summary(time_window),
                'alerts': [
                    a for a in self.alerts 
                    if a.get('timestamp', datetime.min) > datetime.now() - time_window
                ]
            }
            
            for metric in metrics_data['inference_metrics']:
                if 'timestamp' in metric:
                    metric['timestamp'] = metric['timestamp'].isoformat()
            
            for metric in metrics_data['system_metrics']:
                if 'timestamp' in metric:
                    metric['timestamp'] = metric['timestamp'].isoformat()
            
            for alert in metrics_data['alerts']:
                if 'timestamp' in alert:
                    alert['timestamp'] = alert['timestamp'].isoformat()
            
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            with open(filename, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            self.logger.info(f"Metrics exported to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {e}")
    
    def get_prometheus_metrics(self) -> str:
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus client not available\n"
        
        try:
            return generate_latest().decode('utf-8')
        except Exception as e:
            self.logger.error(f"Error generating Prometheus metrics: {e}")
            return f"# Error generating metrics: {e}\n"
    
    def reset_metrics(self):
        self.inference_metrics.clear()
        self.system_metrics.clear()
        self.alerts.clear()
        self.start_time = datetime.now()
        self.logger.info("All metrics reset")