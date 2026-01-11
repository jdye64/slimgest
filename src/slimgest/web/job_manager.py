"""
Job manager for tracking and managing PDF processing jobs.
"""
import asyncio
import uuid
from typing import Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    STREAMING = "streaming"


@dataclass
class Job:
    """Represents a processing job."""
    job_id: str
    status: JobStatus
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    result_queue: Optional[asyncio.Queue] = None  # For streaming jobs


class JobManager:
    """Manages job lifecycle and tracking."""
    
    def __init__(self):
        """Initialize the job manager."""
        self.jobs: Dict[str, Job] = {}
        self._lock = asyncio.Lock()
    
    def create_job(self, job_type: str = "batch") -> str:
        """
        Create a new job and return its ID.
        
        Args:
            job_type: Type of job ("batch" or "stream")
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        # Create result queue for streaming jobs
        result_queue = asyncio.Queue() if job_type == "stream" else None
        
        job = Job(
            job_id=job_id,
            status=JobStatus.PENDING,
            result_queue=result_queue,
        )
        self.jobs[job_id] = job
        return job_id
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get a job by ID.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job object or None if not found
        """
        async with self._lock:
            return self.jobs.get(job_id)
    
    async def update_job_status(self, job_id: str, status: JobStatus):
        """
        Update job status.
        
        Args:
            job_id: Job ID
            status: New status
        """
        async with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id].status = status
                if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    self.jobs[job_id].completed_at = datetime.now()
    
    async def set_job_result(self, job_id: str, result: Any):
        """
        Set job result and mark as completed.
        
        Args:
            job_id: Job ID
            result: Result data
        """
        async with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id].result = result
                self.jobs[job_id].status = JobStatus.COMPLETED
                self.jobs[job_id].completed_at = datetime.now()
    
    async def set_job_error(self, job_id: str, error: str):
        """
        Set job error and mark as failed.
        
        Args:
            job_id: Job ID
            error: Error message
        """
        async with self._lock:
            if job_id in self.jobs:
                self.jobs[job_id].error = error
                self.jobs[job_id].status = JobStatus.FAILED
                self.jobs[job_id].completed_at = datetime.now()
    
    async def queue_stream_event(self, job_id: str, event: Dict[str, Any]):
        """
        Queue a streaming event for a job.
        
        Args:
            job_id: Job ID
            event: Event data
        """
        job = await self.get_job(job_id)
        if job and job.result_queue:
            await job.result_queue.put(event)
    
    async def get_stream_events(self, job_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Get streaming events for a job.
        
        Args:
            job_id: Job ID
            
        Yields:
            Event data
        """
        job = await self.get_job(job_id)
        if not job or not job.result_queue:
            return
        
        while True:
            event = await job.result_queue.get()
            
            # Check for completion/error
            if event["type"] in ["complete", "error"]:
                yield event
                break
            
            yield event
    
    async def wait_for_completion(self, job_id: str, timeout: Optional[float] = None) -> Job:
        """
        Wait for a job to complete.
        
        Args:
            job_id: Job ID
            timeout: Optional timeout in seconds
            
        Returns:
            Completed job
            
        Raises:
            asyncio.TimeoutError: If timeout is exceeded
            KeyError: If job not found
        """
        start_time = asyncio.get_event_loop().time()
        
        while True:
            job = await self.get_job(job_id)
            
            if not job:
                raise KeyError(f"Job {job_id} not found")
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                return job
            
            # Check timeout
            if timeout is not None:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    raise asyncio.TimeoutError(f"Job {job_id} timed out after {timeout}s")
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
    
    async def cleanup_old_jobs(self, max_age_seconds: int = 3600):
        """
        Clean up jobs older than max_age_seconds.
        
        Args:
            max_age_seconds: Maximum age in seconds
        """
        async with self._lock:
            now = datetime.now()
            to_remove = []
            
            for job_id, job in self.jobs.items():
                if job.completed_at:
                    age = (now - job.completed_at).total_seconds()
                    if age > max_age_seconds:
                        to_remove.append(job_id)
            
            for job_id in to_remove:
                del self.jobs[job_id]
            
            if to_remove:
                print(f"Cleaned up {len(to_remove)} old jobs")
