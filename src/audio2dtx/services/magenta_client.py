"""
Client for Magenta drum classification service.
"""

import json
import time
import numpy as np
import requests
from typing import Dict, Any, Optional
from urllib.parse import urljoin

from ..config.settings import Settings
from ..utils.exceptions import ServiceError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MagentaClient:
    """
    Client for communicating with Magenta drum classification microservice.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize Magenta client.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.base_url = settings.services.magenta_url
        self.timeout = settings.services.magenta_timeout
        self.retry_attempts = settings.services.retry_attempts
        self.retry_delay = settings.services.retry_delay
        
        # Service endpoints
        self.endpoints = {
            'health': '/health',
            'classify': '/classify-drums',
            'info': '/info'
        }
        
        self._service_available = None
        self._last_health_check = 0
        self._health_check_interval = 60  # Check every 60 seconds
        
    def _make_request(self, 
                     endpoint: str, 
                     method: str = 'GET',
                     data: Optional[Dict] = None,
                     timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Make HTTP request to Magenta service with retry logic.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            data: Request data (for POST requests)
            timeout: Request timeout
            
        Returns:
            Response data
            
        Raises:
            ServiceError: If request fails after retries
        """
        url = urljoin(self.base_url, endpoint)
        timeout = timeout or self.timeout
        
        for attempt in range(self.retry_attempts):
            try:
                if method.upper() == 'GET':
                    response = requests.get(url, timeout=timeout)
                elif method.upper() == 'POST':
                    response = requests.post(
                        url, 
                        json=data, 
                        timeout=timeout,
                        headers={'Content-Type': 'application/json'}
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request to {url} timed out (attempt {attempt + 1}/{self.retry_attempts})")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Failed to connect to {url} (attempt {attempt + 1}/{self.retry_attempts})")
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error {e.response.status_code} for {url}: {e}")
                break  # Don't retry HTTP errors
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed for {url}: {e} (attempt {attempt + 1}/{self.retry_attempts})")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from {url}: {e}")
                break  # Don't retry JSON errors
            
            if attempt < self.retry_attempts - 1:
                time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        raise ServiceError(f"Failed to communicate with Magenta service after {self.retry_attempts} attempts")
    
    def check_health(self, force_check: bool = False) -> bool:
        """
        Check if Magenta service is healthy.
        
        Args:
            force_check: Force health check even if recently checked
            
        Returns:
            True if service is healthy
        """
        current_time = time.time()
        
        # Use cached result if recent
        if not force_check and self._service_available is not None:
            if current_time - self._last_health_check < self._health_check_interval:
                return self._service_available
        
        try:
            response = self._make_request(self.endpoints['health'], timeout=5.0)
            
            is_healthy = (
                response.get('status') == 'healthy' and
                response.get('magenta_loaded', False)
            )
            
            self._service_available = is_healthy
            self._last_health_check = current_time
            
            if is_healthy:
                logger.info("Magenta service is healthy and ready")
            else:
                logger.warning("Magenta service is not fully operational")
            
            return is_healthy
            
        except Exception as e:
            logger.warning(f"Magenta health check failed: {e}")
            self._service_available = False
            self._last_health_check = current_time
            return False
    
    def get_service_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the Magenta service.
        
        Returns:
            Service information dictionary or None if unavailable
        """
        try:
            response = self._make_request(self.endpoints['info'])
            return response
        except Exception as e:
            logger.error(f"Failed to get Magenta service info: {e}")
            return None
    
    def classify_drums(self, audio_window: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Classify drum sound using Magenta service.
        
        Args:
            audio_window: Audio data for classification
            
        Returns:
            Classification result or None if failed
        """
        if not self.is_available():
            logger.warning("Magenta service not available for classification")
            return None
        
        try:
            # Prepare request data
            # Convert numpy array to list for JSON serialization
            audio_data = audio_window.tolist()
            
            request_data = {
                'audio_window': audio_data
            }
            
            # Make classification request
            response = self._make_request(
                self.endpoints['classify'],
                method='POST',
                data=request_data
            )
            
            if response.get('success', False):
                prediction = response.get('prediction', {})
                
                # Validate prediction format
                required_fields = ['instrument', 'confidence', 'velocity']
                if all(field in prediction for field in required_fields):
                    return prediction
                else:
                    logger.error(f"Invalid prediction format: {prediction}")
                    return None
            else:
                error_msg = response.get('error', 'Unknown error')
                logger.error(f"Magenta classification failed: {error_msg}")
                return None
                
        except Exception as e:
            logger.error(f"Magenta drum classification failed: {e}")
            return None
    
    def classify_drums_batch(self, 
                           audio_windows: list,
                           batch_size: int = 10) -> list:
        """
        Classify multiple drum sounds in batches.
        
        Args:
            audio_windows: List of audio windows for classification
            batch_size: Number of windows to process in each batch
            
        Returns:
            List of classification results
        """
        results = []
        
        for i in range(0, len(audio_windows), batch_size):
            batch = audio_windows[i:i + batch_size]
            batch_results = []
            
            for audio_window in batch:
                result = self.classify_drums(audio_window)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Small delay between batches to avoid overwhelming the service
            if i + batch_size < len(audio_windows):
                time.sleep(0.1)
        
        return results
    
    def is_available(self) -> bool:
        """
        Check if Magenta service is available for use.
        
        Returns:
            True if service is available
        """
        return self.check_health()
    
    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information.
        
        Returns:
            Dictionary with connection details
        """
        return {
            'base_url': self.base_url,
            'timeout': self.timeout,
            'retry_attempts': self.retry_attempts,
            'retry_delay': self.retry_delay,
            'service_available': self._service_available,
            'last_health_check': self._last_health_check
        }
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Test connection to Magenta service and return detailed results.
        
        Returns:
            Test results dictionary
        """
        test_results = {
            'connection_successful': False,
            'health_check_passed': False,
            'service_info_available': False,
            'classification_test_passed': False,
            'error_messages': []
        }
        
        try:
            # Test basic connection
            response = requests.get(
                urljoin(self.base_url, self.endpoints['health']),
                timeout=5.0
            )
            test_results['connection_successful'] = response.status_code == 200
            
        except Exception as e:
            test_results['error_messages'].append(f"Connection failed: {e}")
        
        if test_results['connection_successful']:
            # Test health check
            try:
                test_results['health_check_passed'] = self.check_health(force_check=True)
            except Exception as e:
                test_results['error_messages'].append(f"Health check failed: {e}")
            
            # Test service info
            try:
                info = self.get_service_info()
                test_results['service_info_available'] = info is not None
            except Exception as e:
                test_results['error_messages'].append(f"Service info failed: {e}")
            
            # Test classification with dummy data
            try:
                dummy_audio = np.random.randn(1024).astype(np.float32)
                result = self.classify_drums(dummy_audio)
                test_results['classification_test_passed'] = result is not None
            except Exception as e:
                test_results['error_messages'].append(f"Classification test failed: {e}")
        
        return test_results