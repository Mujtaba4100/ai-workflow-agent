"""
Workflow Monitor - M2 Feature
Monitor workflow executions, aggregate status, and coordinate notifications
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from workflow_executor import WorkflowExecutor, ExecutionResult, ExecutionStatus, WorkflowType, get_executor
from notifications import NotificationManager, get_notification_manager
from storage import PersistentStorage, get_storage

logger = logging.getLogger(__name__)


@dataclass
class WorkflowStats:
    """Statistics for workflow monitoring"""
    total_executions: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    average_duration: float = 0.0
    success_rate: float = 0.0
    by_type: Dict[str, int] = field(default_factory=dict)
    recent_errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_executions": self.total_executions,
            "running": self.running,
            "completed": self.completed,
            "failed": self.failed,
            "average_duration": round(self.average_duration, 2),
            "success_rate": round(self.success_rate, 2),
            "by_type": self.by_type,
            "recent_errors": self.recent_errors
        }


class WorkflowMonitor:
    """
    Monitor and coordinate workflow executions.
    Aggregates statistics, manages history, and triggers notifications.
    """
    
    def __init__(
        self,
        executor: Optional[WorkflowExecutor] = None,
        notification_manager: Optional[NotificationManager] = None,
        storage: Optional[PersistentStorage] = None
    ):
        self.executor = executor or get_executor()
        self.notifications = notification_manager or get_notification_manager()
        self.storage = storage or get_storage()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        
    # ==================== Execution with Monitoring ====================
    
    async def execute_n8n(
        self,
        workflow_json: Dict[str, Any],
        workflow_id: Optional[str] = None,
        workflow_name: str = "Unnamed",
        test_mode: bool = False
    ) -> ExecutionResult:
        """
        Execute n8n workflow with monitoring and notifications
        """
        # Notify start
        await self.notifications.notify_workflow_started(
            workflow_id=workflow_id or "unknown",
            workflow_name=workflow_name,
            execution_id="pending"
        )
        
        # Execute
        result = await self.executor.execute_n8n_workflow(
            workflow_json=workflow_json,
            workflow_id=workflow_id,
            test_mode=test_mode
        )
        
        # Store in persistent storage
        self.storage.save_execution(result.to_dict())
        
        # Notify completion
        if result.status == ExecutionStatus.COMPLETED:
            await self.notifications.notify_workflow_completed(
                workflow_id=result.workflow_id,
                workflow_name=workflow_name,
                execution_id=result.execution_id,
                duration=result.duration_seconds or 0
            )
        elif result.status == ExecutionStatus.FAILED:
            await self.notifications.notify_workflow_failed(
                workflow_id=result.workflow_id,
                workflow_name=workflow_name,
                execution_id=result.execution_id,
                error=result.error or "Unknown error"
            )
            
        return result
    
    async def execute_comfyui(
        self,
        workflow_json: Dict[str, Any],
        workflow_id: Optional[str] = None,
        workflow_name: str = "Unnamed",
        wait_for_completion: bool = True
    ) -> ExecutionResult:
        """
        Execute ComfyUI workflow with monitoring and notifications
        """
        # Notify start
        await self.notifications.notify_workflow_started(
            workflow_id=workflow_id or "unknown",
            workflow_name=workflow_name,
            execution_id="pending"
        )
        
        # Execute
        result = await self.executor.execute_comfyui_workflow(
            workflow_json=workflow_json,
            workflow_id=workflow_id,
            wait_for_completion=wait_for_completion
        )
        
        # Store
        self.storage.save_execution(result.to_dict())
        
        # Notify completion
        if result.status == ExecutionStatus.COMPLETED:
            await self.notifications.notify_workflow_completed(
                workflow_id=result.workflow_id,
                workflow_name=workflow_name,
                execution_id=result.execution_id,
                duration=result.duration_seconds or 0
            )
        elif result.status == ExecutionStatus.FAILED:
            await self.notifications.notify_workflow_failed(
                workflow_id=result.workflow_id,
                workflow_name=workflow_name,
                execution_id=result.execution_id,
                error=result.error or "Unknown error"
            )
            
        return result
    
    async def execute_hybrid(
        self,
        n8n_workflow: Dict[str, Any],
        comfyui_workflow: Dict[str, Any],
        workflow_id: Optional[str] = None,
        workflow_name: str = "Unnamed Hybrid"
    ) -> ExecutionResult:
        """
        Execute hybrid workflow with monitoring
        """
        await self.notifications.notify_workflow_started(
            workflow_id=workflow_id or "unknown",
            workflow_name=workflow_name,
            execution_id="pending"
        )
        
        result = await self.executor.execute_hybrid_workflow(
            n8n_workflow=n8n_workflow,
            comfyui_workflow=comfyui_workflow,
            workflow_id=workflow_id
        )
        
        self.storage.save_execution(result.to_dict())
        
        if result.status == ExecutionStatus.COMPLETED:
            await self.notifications.notify_workflow_completed(
                workflow_id=result.workflow_id,
                workflow_name=workflow_name,
                execution_id=result.execution_id,
                duration=result.duration_seconds or 0
            )
        elif result.status == ExecutionStatus.FAILED:
            await self.notifications.notify_workflow_failed(
                workflow_id=result.workflow_id,
                workflow_name=workflow_name,
                execution_id=result.execution_id,
                error=result.error or "Unknown error"
            )
            
        return result
    
    # ==================== Status & Monitoring ====================
    
    def get_running_executions(self) -> List[Dict[str, Any]]:
        """Get currently running executions"""
        return [r.to_dict() for r in self.executor.get_running_executions()]
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific execution"""
        # Check running first
        result = self.executor.get_execution_status(execution_id)
        if result:
            return result.to_dict()
            
        # Check persistent storage
        return self.storage.get_execution(execution_id)
    
    def get_recent_executions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent executions from history"""
        # Combine in-memory and persistent
        running = [r.to_dict() for r in self.executor.get_running_executions()]
        stored = self.storage.get_recent_executions(limit=limit)
        
        # Merge and sort
        all_executions = running + stored
        all_executions.sort(key=lambda x: x["started_at"], reverse=True)
        
        return all_executions[:limit]
    
    def get_executions_by_workflow(self, workflow_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get executions for a specific workflow"""
        return self.storage.get_executions_by_workflow(workflow_id, limit)
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        success = self.executor.cancel_execution(execution_id)
        if success:
            asyncio.create_task(
                self.notifications.notify_warning(
                    title="Execution Cancelled",
                    message=f"Execution {execution_id[:8]} was cancelled"
                )
            )
        return success
    
    # ==================== Statistics ====================
    
    def get_stats(self) -> WorkflowStats:
        """Get comprehensive workflow statistics"""
        # From executor (in-memory)
        running = self.executor.get_running_executions()
        memory_stats = self.executor.history.get_stats()
        
        # From persistent storage
        storage_stats = self.storage.get_execution_stats()
        
        # Get recent errors
        recent_errors = []
        for execution in self.storage.get_recent_executions(limit=100):
            if execution["status"] == "failed":
                recent_errors.append({
                    "execution_id": execution["execution_id"],
                    "workflow_id": execution["workflow_id"],
                    "error": execution.get("error", "Unknown"),
                    "started_at": execution["started_at"]
                })
                if len(recent_errors) >= 5:
                    break
        
        stats = WorkflowStats(
            total_executions=storage_stats["total_executions"],
            running=len(running),
            completed=storage_stats["by_status"].get("completed", 0),
            failed=storage_stats["by_status"].get("failed", 0),
            average_duration=storage_stats["average_duration_seconds"],
            success_rate=storage_stats["success_rate"],
            by_type=storage_stats["by_type"],
            recent_errors=recent_errors
        )
        
        return stats
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get all data needed for a monitoring dashboard"""
        stats = self.get_stats()
        
        return {
            "stats": stats.to_dict(),
            "running_executions": self.get_running_executions(),
            "recent_executions": self.get_recent_executions(limit=10),
            "notifications": {
                "unread": self.notifications.get_unread_count(),
                "recent": [n.to_dict() for n in self.notifications.get_notifications(limit=5)]
            },
            "system_status": self._get_system_status(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get system component status"""
        return {
            "executor": "active",
            "storage": "connected",
            "notifications": "active",
            "monitoring": "running" if self._is_monitoring else "stopped"
        }
    
    # ==================== Background Monitoring ====================
    
    async def start_monitoring(self, interval: int = 30):
        """Start background monitoring task"""
        if self._is_monitoring:
            return
            
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info("Workflow monitoring started")
        
    async def stop_monitoring(self):
        """Stop background monitoring"""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        logger.info("Workflow monitoring stopped")
    
    async def _monitoring_loop(self, interval: int):
        """Background monitoring loop"""
        while self._is_monitoring:
            try:
                # Check for long-running executions
                for execution in self.executor.get_running_executions():
                    duration = (datetime.now() - execution.started_at).total_seconds()
                    if duration > 300:  # > 5 minutes
                        await self.notifications.notify_warning(
                            title="Long Running Execution",
                            message=f"{execution.workflow_id} running for {duration:.0f}s"
                        )
                        
                # Check for stale executions
                stats = self.get_stats()
                if stats.failed > 0 and stats.success_rate < 50:
                    await self.notifications.notify_warning(
                        title="High Failure Rate",
                        message=f"Success rate: {stats.success_rate:.1f}%"
                    )
                    
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                
            await asyncio.sleep(interval)


# Singleton instance
_monitor: Optional[WorkflowMonitor] = None


def get_monitor() -> WorkflowMonitor:
    """Get the global workflow monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = WorkflowMonitor()
    return _monitor
