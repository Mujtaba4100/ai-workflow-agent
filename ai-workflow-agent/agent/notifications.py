"""
Notification System - M2 Feature
Send and manage notifications for events, workflow completions, and alerts
"""

import uuid
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Awaitable
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Types of notifications"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WEBHOOK_RECEIVED = "webhook_received"
    SYSTEM = "system"


class NotificationPriority(str, Enum):
    """Notification priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """A notification message"""
    notification_id: str
    notification_type: NotificationType
    priority: NotificationPriority
    title: str
    message: str
    data: Optional[Dict[str, Any]] = None
    is_read: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "notification_id": self.notification_id,
            "notification_type": self.notification_type.value,
            "priority": self.priority.value,
            "title": self.title,
            "message": self.message,
            "data": self.data,
            "is_read": self.is_read,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }


# Type for notification listeners
NotificationListener = Callable[[Notification], Awaitable[None]]


class NotificationSubscription:
    """Subscription to notification events"""
    
    def __init__(
        self,
        subscription_id: str,
        listener: NotificationListener,
        notification_types: Optional[List[NotificationType]] = None,
        min_priority: NotificationPriority = NotificationPriority.LOW
    ):
        self.subscription_id = subscription_id
        self.listener = listener
        self.notification_types = notification_types  # None = all types
        self.min_priority = min_priority
        self.created_at = datetime.now()
        
    def matches(self, notification: Notification) -> bool:
        """Check if notification matches subscription filters"""
        # Check type
        if self.notification_types and notification.notification_type not in self.notification_types:
            return False
            
        # Check priority
        priority_order = [
            NotificationPriority.LOW,
            NotificationPriority.NORMAL,
            NotificationPriority.HIGH,
            NotificationPriority.URGENT
        ]
        if priority_order.index(notification.priority) < priority_order.index(self.min_priority):
            return False
            
        return True


class NotificationManager:
    """
    Manage notifications across the system.
    Supports creating, storing, and dispatching notifications to subscribers.
    """
    
    def __init__(self, max_notifications: int = 500):
        self._notifications: Dict[str, Notification] = {}
        self._subscriptions: Dict[str, NotificationSubscription] = {}
        self._max_notifications = max_notifications
        
    # ==================== Create Notifications ====================
    
    async def notify(
        self,
        title: str,
        message: str = "",
        notification_type: NotificationType = NotificationType.INFO,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        data: Optional[Dict[str, Any]] = None,
        store: bool = True
    ) -> Notification:
        """
        Create and dispatch a notification
        
        Args:
            title: Notification title
            message: Notification body
            notification_type: Type of notification
            priority: Priority level
            data: Additional data to include
            store: Whether to store in history
        """
        notification = Notification(
            notification_id=str(uuid.uuid4()),
            notification_type=notification_type,
            priority=priority,
            title=title,
            message=message,
            data=data
        )
        
        # Store if requested
        if store:
            self._notifications[notification.notification_id] = notification
            self._trim_notifications()
        
        # Dispatch to subscribers
        await self._dispatch(notification)
        
        logger.debug(f"Notification: [{notification_type.value}] {title}")
        return notification
    
    async def notify_success(
        self,
        title: str,
        message: str = "",
        data: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """Create a success notification"""
        return await self.notify(
            title=title,
            message=message,
            notification_type=NotificationType.SUCCESS,
            priority=NotificationPriority.NORMAL,
            data=data
        )
    
    async def notify_error(
        self,
        title: str,
        message: str = "",
        data: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """Create an error notification"""
        return await self.notify(
            title=title,
            message=message,
            notification_type=NotificationType.ERROR,
            priority=NotificationPriority.HIGH,
            data=data
        )
    
    async def notify_warning(
        self,
        title: str,
        message: str = "",
        data: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """Create a warning notification"""
        return await self.notify(
            title=title,
            message=message,
            notification_type=NotificationType.WARNING,
            priority=NotificationPriority.NORMAL,
            data=data
        )
    
    async def notify_workflow_started(
        self,
        workflow_id: str,
        workflow_name: str,
        execution_id: str
    ) -> Notification:
        """Notify that a workflow has started"""
        return await self.notify(
            title=f"Workflow Started: {workflow_name}",
            message=f"Execution {execution_id[:8]} has begun",
            notification_type=NotificationType.WORKFLOW_STARTED,
            priority=NotificationPriority.LOW,
            data={
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "execution_id": execution_id
            }
        )
    
    async def notify_workflow_completed(
        self,
        workflow_id: str,
        workflow_name: str,
        execution_id: str,
        duration: float
    ) -> Notification:
        """Notify that a workflow has completed"""
        return await self.notify(
            title=f"Workflow Completed: {workflow_name}",
            message=f"Finished in {duration:.1f}s",
            notification_type=NotificationType.WORKFLOW_COMPLETED,
            priority=NotificationPriority.NORMAL,
            data={
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "execution_id": execution_id,
                "duration_seconds": duration
            }
        )
    
    async def notify_workflow_failed(
        self,
        workflow_id: str,
        workflow_name: str,
        execution_id: str,
        error: str
    ) -> Notification:
        """Notify that a workflow has failed"""
        return await self.notify(
            title=f"Workflow Failed: {workflow_name}",
            message=error[:100],
            notification_type=NotificationType.WORKFLOW_FAILED,
            priority=NotificationPriority.HIGH,
            data={
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "execution_id": execution_id,
                "error": error
            }
        )
    
    async def notify_webhook_received(
        self,
        webhook_id: str,
        source: str,
        event_type: str
    ) -> Notification:
        """Notify that a webhook was received"""
        return await self.notify(
            title=f"Webhook: {source}",
            message=f"Event: {event_type}",
            notification_type=NotificationType.WEBHOOK_RECEIVED,
            priority=NotificationPriority.LOW,
            data={
                "webhook_id": webhook_id,
                "source": source,
                "event_type": event_type
            }
        )
    
    # ==================== Subscriptions ====================
    
    def subscribe(
        self,
        listener: NotificationListener,
        notification_types: Optional[List[NotificationType]] = None,
        min_priority: NotificationPriority = NotificationPriority.LOW
    ) -> str:
        """
        Subscribe to notifications
        
        Args:
            listener: Async callback function
            notification_types: Types to subscribe to (None = all)
            min_priority: Minimum priority to receive
            
        Returns:
            Subscription ID
        """
        subscription_id = str(uuid.uuid4())
        
        subscription = NotificationSubscription(
            subscription_id=subscription_id,
            listener=listener,
            notification_types=notification_types,
            min_priority=min_priority
        )
        
        self._subscriptions[subscription_id] = subscription
        logger.debug(f"New subscription: {subscription_id}")
        
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from notifications"""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            return True
        return False
    
    async def _dispatch(self, notification: Notification):
        """Dispatch notification to all matching subscribers"""
        for subscription in self._subscriptions.values():
            if subscription.matches(notification):
                try:
                    await subscription.listener(notification)
                except Exception as e:
                    logger.error(f"Subscription handler error: {e}")
    
    # ==================== Notification Management ====================
    
    def get_notification(self, notification_id: str) -> Optional[Notification]:
        """Get a specific notification"""
        return self._notifications.get(notification_id)
    
    def get_notifications(
        self,
        notification_type: Optional[NotificationType] = None,
        unread_only: bool = False,
        limit: int = 50
    ) -> List[Notification]:
        """Get notifications with optional filters"""
        notifications = list(self._notifications.values())
        
        if notification_type:
            notifications = [n for n in notifications if n.notification_type == notification_type]
        if unread_only:
            notifications = [n for n in notifications if not n.is_read]
            
        notifications.sort(key=lambda n: n.created_at, reverse=True)
        return notifications[:limit]
    
    def mark_read(self, notification_id: str) -> bool:
        """Mark a notification as read"""
        notification = self._notifications.get(notification_id)
        if notification:
            notification.is_read = True
            return True
        return False
    
    def mark_all_read(self) -> int:
        """Mark all notifications as read, returns count"""
        count = 0
        for notification in self._notifications.values():
            if not notification.is_read:
                notification.is_read = True
                count += 1
        return count
    
    def delete_notification(self, notification_id: str) -> bool:
        """Delete a notification"""
        if notification_id in self._notifications:
            del self._notifications[notification_id]
            return True
        return False
    
    def clear_all(self) -> int:
        """Clear all notifications, returns count"""
        count = len(self._notifications)
        self._notifications.clear()
        return count
    
    def get_unread_count(self) -> int:
        """Get count of unread notifications"""
        return sum(1 for n in self._notifications.values() if not n.is_read)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        total = len(self._notifications)
        unread = self.get_unread_count()
        
        by_type: Dict[str, int] = {}
        by_priority: Dict[str, int] = {}
        
        for notification in self._notifications.values():
            # Count by type
            ntype = notification.notification_type.value
            by_type[ntype] = by_type.get(ntype, 0) + 1
            
            # Count by priority
            priority = notification.priority.value
            by_priority[priority] = by_priority.get(priority, 0) + 1
            
        return {
            "total": total,
            "unread": unread,
            "by_type": by_type,
            "by_priority": by_priority,
            "subscriptions": len(self._subscriptions)
        }
    
    def _trim_notifications(self):
        """Trim old notifications if over max"""
        if len(self._notifications) > self._max_notifications:
            # Remove oldest read notifications first
            sorted_notifications = sorted(
                self._notifications.values(),
                key=lambda n: (n.is_read, n.created_at)  # Unread first, then by time
            )
            
            to_remove = len(self._notifications) - self._max_notifications
            for notification in sorted_notifications[:to_remove]:
                del self._notifications[notification.notification_id]


# Singleton instance
_manager: Optional[NotificationManager] = None


def get_notification_manager() -> NotificationManager:
    """Get the global notification manager instance"""
    global _manager
    if _manager is None:
        _manager = NotificationManager()
    return _manager


# Convenience functions
async def notify(title: str, message: str = "", **kwargs) -> Notification:
    """Quick notification helper"""
    return await get_notification_manager().notify(title, message, **kwargs)


async def notify_success(title: str, message: str = "", **kwargs) -> Notification:
    """Quick success notification"""
    return await get_notification_manager().notify_success(title, message, **kwargs)


async def notify_error(title: str, message: str = "", **kwargs) -> Notification:
    """Quick error notification"""
    return await get_notification_manager().notify_error(title, message, **kwargs)
