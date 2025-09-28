import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import hashlib

class AdvancedSecuritySystem:
    """
    Advanced security system for input validation, rate limiting, and threat detection.
    Protects the AI system from abuse and malicious inputs.
    """
    
    def __init__(self):
        self.rate_limits = {}
        self.suspicious_ips = {}
        self.security_log = []
        
        # Suspicious patterns for input validation
        self.suspicious_patterns = [
            # SQL Injection patterns
            r"(?i)(union.*select|select.*from|insert.*into|delete.*from|drop.*table)",
            r"(?i)(or.*1=1|and.*1=1|exec.*\(|xp_cmdshell)",
            r"(\b)(DROP|DELETE|INSERT|UPDATE|ALTER)(\b)",
            
            # XSS patterns
            r"(?i)(script|javascript|onload|onerror|onclick|alert\(|document\.cookie)",
            r"<.*>.*</.*>",  # HTML tags
            
            # Command injection
            r"[;&|`]\s*\w+",
            r"\$\(.*\)",
            
            # Path traversal
            r"\.\./|\.\.\\",
            
            # Sensitive data patterns
            r"(?i)(password|token|key|secret|auth|credential)",
            r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",  # IP addresses
            
            # Excessive length or repetition
            r".{10000,}",  # Very long inputs
            r"(.)\1{50,}",  # Repeated characters
            
            # Admin/privilege patterns
            r"(?i)(admin|root|sudo|su -|chmod|chown)"
        ]
        
        # Rate limiting configuration
        self.rate_limit_config = {
            "default": {"requests_per_minute": 60, "burst_capacity": 10},
            "anonymous": {"requests_per_minute": 30, "burst_capacity": 5},
            "suspicious": {"requests_per_minute": 10, "burst_capacity": 2}
        }
        
        self.setup_logging()
    
    def setup_logging(self):
        """Setup security logging"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def check_request(self, query: str, user_id: str, ip_address: Optional[str] = None) -> Dict[str, any]:
        """
        Comprehensive security check for incoming requests.
        
        Args:
            query: User's query text
            user_id: User identifier
            ip_address: Optional IP address for IP-based checks
            
        Returns:
            Security assessment result
        """
        result = {
            "is_suspicious": False,
            "alerts": [],
            "risk_score": 0,
            "allowed": True,
            "rate_limit_info": {}
        }
        
        # Rate limiting check
        rate_limit_result = self.check_rate_limit(user_id, ip_address)
        if not rate_limit_result["allowed"]:
            result["is_suspicious"] = True
            result["allowed"] = False
            result["alerts"].append("Rate limit exceeded")
            result["risk_score"] = 100
            result["rate_limit_info"] = rate_limit_result
            return result
        
        result["rate_limit_info"] = rate_limit_result
        
        # Input validation and pattern matching
        validation_result = self.validate_input(query, user_id)
        result["alerts"].extend(validation_result["alerts"])
        result["risk_score"] += validation_result["risk_score"]
        
        # IP reputation check (if IP provided)
        if ip_address:
            ip_result = self.check_ip_reputation(ip_address)
            result["alerts"].extend(ip_result["alerts"])
            result["risk_score"] += ip_result["risk_score"]
        
        # Determine overall suspicion
        if result["risk_score"] >= 50:
            result["is_suspicious"] = True
            if result["risk_score"] >= 80:
                result["allowed"] = False
        
        # Log security event
        self.log_security_event(user_id, ip_address, query, result)
        
        return result
    
    def check_rate_limit(self, user_id: str, ip_address: Optional[str] = None) -> Dict[str, any]:
        """Check rate limits for user and/or IP"""
        current_time = datetime.now()
        user_key = f"user_{user_id}"
        ip_key = f"ip_{ip_address}" if ip_address else None
        
        # Get rate limit configuration
        user_config = self.rate_limit_config.get("default")
        if user_id == "anonymous":
            user_config = self.rate_limit_config.get("anonymous", user_config)
        
        # Check if user is marked as suspicious
        if self.is_suspicious_user(user_id) or (ip_address and self.is_suspicious_ip(ip_address)):
            user_config = self.rate_limit_config.get("suspicious", user_config)
        
        # Clean old entries for user
        self.rate_limits[user_key] = [
            t for t in self.rate_limits.get(user_key, [])
            if current_time - t < timedelta(minutes=1)
        ]
        
        # Clean old entries for IP (if provided)
        if ip_key:
            self.rate_limits[ip_key] = [
                t for t in self.rate_limits.get(ip_key, [])
                if current_time - t < timedelta(minutes=1)
            ]
        
        # Check user rate limit
        user_requests = len(self.rate_limits.get(user_key, []))
        user_allowed = user_requests < user_config["requests_per_minute"]
        
        # Check IP rate limit (if IP provided)
        ip_allowed = True
        if ip_key:
            ip_requests = len(self.rate_limits.get(ip_key, []))
            ip_allowed = ip_requests < user_config["requests_per_minute"]
        
        allowed = user_allowed and ip_allowed
        
        # Add current request to counters if allowed
        if allowed:
            self.rate_limits.setdefault(user_key, []).append(current_time)
            if ip_key:
                self.rate_limits.setdefault(ip_key, []).append(current_time)
        
        return {
            "allowed": allowed,
            "user_requests": user_requests,
            "user_limit": user_config["requests_per_minute"],
            "ip_requests": len(self.rate_limits.get(ip_key, [])) if ip_key else 0,
            "ip_limit": user_config["requests_per_minute"] if ip_key else "N/A",
            "retry_after": 60 if not allowed else 0
        }
    
    def validate_input(self, query: str, user_id: str) -> Dict[str, any]:
        """Validate and analyze user input"""
        result = {
            "alerts": [],
            "risk_score": 0
        }
        
        # Pattern matching
        for pattern in self.suspicious_patterns:
            matches = re.findall(pattern, query)
            if matches:
                alert_msg = f"Suspicious pattern detected: {pattern[:50]}..."
                result["alerts"].append(alert_msg)
                result["risk_score"] += 20
        
        # Query length analysis
        query_length = len(query)
        if query_length > 10000:
            result["alerts"].append("Excessively long query detected")
            result["risk_score"] += 30
        elif query_length > 5000:
            result["alerts"].append("Very long query detected")
            result["risk_score"] += 15
        
        # Special character analysis
        special_chars = len(re.findall(r'[^\w\s\.\?\!]', query))
        special_char_ratio = special_chars / max(len(query), 1)
        
        if special_char_ratio > 0.3:
            result["alerts"].append("High percentage of special characters")
            result["risk_score"] += 25
        elif special_char_ratio > 0.2:
            result["alerts"].append("Elevated special character usage")
            result["risk_score"] += 10
        
        # Entropy analysis (for encrypted/encoded content)
        entropy = self.calculate_entropy(query)
        if entropy > 6.0:  # High entropy might indicate encoded/encrypted content
            result["alerts"].append("High entropy content detected")
            result["risk_score"] += 20
        
        return result
    
    def check_ip_reputation(self, ip_address: str) -> Dict[str, any]:
        """Check IP reputation (basic implementation)"""
        result = {
            "alerts": [],
            "risk_score": 0
        }
        
        # Check if IP is in suspicious list
        if self.is_suspicious_ip(ip_address):
            result["alerts"].append("IP address has suspicious history")
            result["risk_score"] += 40
        
        # Simple IP pattern check (private IPs, localhost, etc.)
        if ip_address in ["127.0.0.1", "localhost", "0.0.0.0"]:
            result["alerts"].append("Local IP address detected")
            result["risk_score"] += 10
        
        # Check for rapid requests from this IP
        ip_key = f"ip_{ip_address}"
        recent_requests = len(self.rate_limits.get(ip_key, []))
        if recent_requests > 50:  # High volume from single IP
            result["alerts"].append("High request volume from IP")
            result["risk_score"] += 15
        
        return result
    
    def calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text (for detecting encoded content)"""
        if not text:
            return 0.0
        
        import math
        entropy = 0.0
        text_length = len(text)
        
        for char in set(text):
            p_x = float(text.count(char)) / text_length
            if p_x > 0:
                entropy += - p_x * math.log2(p_x)
        
        return entropy
    
    def is_suspicious_user(self, user_id: str) -> bool:
        """Check if user is marked as suspicious"""
        # In a real implementation, this would check a database
        # For now, use simple in-memory tracking
        user_key = f"user_{user_id}"
        return self.suspicious_ips.get(user_key, 0) > 5
    
    def is_suspicious_ip(self, ip_address: str) -> bool:
        """Check if IP is marked as suspicious"""
        ip_key = f"ip_{ip_address}"
        return self.suspicious_ips.get(ip_key, 0) > 3
    
    def mark_suspicious(self, user_id: str, ip_address: Optional[str] = None, reason: str = ""):
        """Mark user or IP as suspicious"""
        if user_id:
            user_key = f"user_{user_id}"
            self.suspicious_ips[user_key] = self.suspicious_ips.get(user_key, 0) + 1
        
        if ip_address:
            ip_key = f"ip_{ip_address}"
            self.suspicious_ips[ip_key] = self.suspicious_ips.get(ip_key, 0) + 1
        
        self.logger.warning(f"Marked as suspicious - User: {user_id}, IP: {ip_address}, Reason: {reason}")
    
    def log_security_event(self, user_id: str, ip_address: Optional[str], query: str, result: Dict):
        """Log security event for auditing"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "ip_address": ip_address,
            "query_preview": query[:100] + "..." if len(query) > 100 else query,
            "query_length": len(query),
            "risk_score": result["risk_score"],
            "alerts": result["alerts"],
            "allowed": result["allowed"],
            "is_suspicious": result["is_suspicious"]
        }
        
        self.security_log.append(event)
        
        # Keep only last 1000 events
        if len(self.security_log) > 1000:
            self.security_log = self.security_log[-1000:]
        
        # Log to security logger if high risk
        if result["risk_score"] >= 50:
            self.logger.warning(f"Security alert: User {user_id} - Score: {result['risk_score']} - Alerts: {result['alerts']}")
    
    def get_security_stats(self) -> Dict[str, any]:
        """Get security statistics"""
        recent_events = [e for e in self.security_log 
                        if datetime.now() - datetime.fromisoformat(e["timestamp"]) < timedelta(hours=24)]
        
        blocked_events = [e for e in recent_events if not e["allowed"]]
        suspicious_events = [e for e in recent_events if e["is_suspicious"]]
        
        return {
            "total_events_24h": len(recent_events),
            "blocked_requests_24h": len(blocked_events),
            "suspicious_requests_24h": len(suspicious_events),
            "current_suspicious_users": len([k for k, v in self.suspicious_ips.items() if k.startswith("user_") and v > 0]),
            "current_suspicious_ips": len([k for k, v in self.suspicious_ips.items() if k.startswith("ip_") and v > 0]),
            "rate_limits_tracked": len(self.rate_limits)
        }
    
    def sanitize_input(self, text: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not text:
            return ""
        
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', text)
        
        # Remove SQL injection patterns
        sanitized = re.sub(r'(\b)(DROP|DELETE|INSERT|UPDATE|ALTER|EXEC)(\b)', '', sanitized, flags=re.IGNORECASE)
        
        # Remove JavaScript and HTML patterns
        sanitized = re.sub(r'(javascript|script|onload|onerror|onclick)', '', sanitized, flags=re.IGNORECASE)
        
        # Remove command injection patterns
        sanitized = re.sub(r'[;&|`]\s*\w+', '', sanitized)
        
        return sanitized.strip()
    
    def reset_rate_limits(self, user_id: Optional[str] = None, ip_address: Optional[str] = None):
        """Reset rate limits for specific user or IP"""
        if user_id:
            user_key = f"user_{user_id}"
            if user_key in self.rate_limits:
                del self.rate_limits[user_key]
        
        if ip_address:
            ip_key = f"ip_{ip_address}"
            if ip_key in self.rate_limits:
                del self.rate_limits[ip_key]