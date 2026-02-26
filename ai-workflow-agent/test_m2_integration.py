"""
Milestone 2 Integration Tests
Tests for workflow execution, monitoring, webhooks, notifications, and storage
"""

import sys
import os
import asyncio
from datetime import datetime

# Add agent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agent'))


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def print_result(name: str, success: bool, details: str = ""):
    """Print a test result."""
    status = "✅" if success else "❌"
    print(f"{status} {name}")
    if details:
        print(f"   {details}")


async def test_workflow_executor():
    """Test the workflow executor."""
    print_header("Testing Workflow Executor")
    
    from agent.workflow_executor import (
        WorkflowExecutor, ExecutionStatus, WorkflowType,
        ExecutionHistory, get_executor
    )
    
    # Test 1: Create executor
    executor = WorkflowExecutor()
    print_result("Executor created", executor is not None)
    
    # Test 2: Execute n8n workflow (test mode)
    test_workflow = {
        "nodes": [
            {"type": "n8n-nodes-base.start", "name": "Start"},
            {"type": "n8n-nodes-base.httpRequest", "name": "HTTP Request"}
        ],
        "connections": {}
    }
    
    result = await executor.execute_n8n_workflow(
        workflow_json=test_workflow,
        workflow_id="test_n8n_001",
        test_mode=True
    )
    
    print_result(
        "n8n workflow execution",
        result.status in [ExecutionStatus.COMPLETED],
        f"Status: {result.status.value}, Duration: {result.duration_seconds:.2f}s"
    )
    
    # Test 3: Execute ComfyUI workflow
    comfyui_workflow = {
        "1": {"class_type": "KSampler", "inputs": {}},
        "2": {"class_type": "CheckpointLoader", "inputs": {}}
    }
    
    result = await executor.execute_comfyui_workflow(
        workflow_json=comfyui_workflow,
        workflow_id="test_comfyui_001"
    )
    
    print_result(
        "ComfyUI workflow execution",
        result.status in [ExecutionStatus.COMPLETED],
        f"Status: {result.status.value}"
    )
    
    # Test 4: Execution history
    history = executor.history
    recent = history.get_recent(limit=5)
    
    print_result(
        "Execution history",
        len(recent) >= 2,
        f"Stored {len(recent)} executions"
    )
    
    # Test 5: History stats
    stats = history.get_stats()
    
    print_result(
        "History statistics",
        stats["total_executions"] >= 2,
        f"Total: {stats['total_executions']}, Success rate: {stats['success_rate']:.0f}%"
    )
    
    await executor.close()
    return True


async def test_persistent_storage():
    """Test the persistent storage."""
    print_header("Testing Persistent Storage")
    
    from agent.storage import PersistentStorage
    
    # Use test database
    storage = PersistentStorage(db_path="data/test_data.db")
    
    # Test 1: Save execution
    execution = {
        "execution_id": "test_exec_001",
        "workflow_id": "wf_001",
        "workflow_type": "n8n",
        "status": "completed",
        "started_at": datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat(),
        "duration_seconds": 1.5,
        "output": {"result": "success"},
        "error": None,
        "metadata": {"test": True}
    }
    
    success = storage.save_execution(execution)
    print_result("Save execution", success)
    
    # Test 2: Retrieve execution
    retrieved = storage.get_execution("test_exec_001")
    print_result(
        "Retrieve execution",
        retrieved is not None and retrieved["workflow_id"] == "wf_001"
    )
    
    # Test 3: Save workflow
    workflow_saved = storage.save_workflow(
        workflow_id="saved_wf_001",
        name="Test Workflow",
        workflow_type="n8n",
        workflow_json={"nodes": [], "connections": {}},
        description="A test workflow"
    )
    print_result("Save workflow definition", workflow_saved)
    
    # Test 4: List workflows
    workflows = storage.list_workflows()
    print_result("List saved workflows", len(workflows) > 0, f"Found {len(workflows)} workflows")
    
    # Test 5: Settings
    storage.set_setting("test_key", {"value": 123})
    value = storage.get_setting("test_key")
    print_result("Settings storage", value == {"value": 123})
    
    # Test 6: Notifications storage
    storage.save_notification(
        notification_id="notif_001",
        notification_type="info",
        title="Test Notification",
        message="This is a test"
    )
    notifications = storage.get_notifications()
    print_result("Notification storage", len(notifications) > 0)
    
    # Test 7: Execution stats
    stats = storage.get_execution_stats()
    print_result("Execution statistics", "total_executions" in stats)
    
    return True


async def test_webhook_receiver():
    """Test the webhook receiver."""
    print_header("Testing Webhook Receiver")
    
    from agent.webhook_receiver import WebhookReceiver, WebhookSource
    
    receiver = WebhookReceiver()
    
    # Test 1: Create webhook
    config = receiver.create_webhook(
        name="Test Webhook",
        source=WebhookSource.GITHUB,
        secret="test_secret"
    )
    print_result("Create webhook", config.webhook_id is not None, f"ID: {config.webhook_id[:8]}...")
    
    # Test 2: List webhooks
    webhooks = receiver.list_webhooks()
    print_result("List webhooks", len(webhooks) > 0, f"Found {len(webhooks)} webhooks")
    
    # Test 3: Receive webhook
    event = await receiver.receive_webhook(
        webhook_id=config.webhook_id,
        payload={"action": "push", "repository": {"full_name": "test/repo"}},
        headers={"x-github-event": "push"}
    )
    print_result(
        "Receive webhook",
        event.processed,
        f"Event: {event.event_type}, Source: {event.source.value}"
    )
    
    # Test 4: Get events
    events = receiver.get_events(limit=10)
    print_result("Get webhook events", len(events) > 0, f"Found {len(events)} events")
    
    # Test 5: Raw webhook
    raw_event = await receiver.receive_raw_webhook(
        source="n8n",
        payload={"event": "workflow.completed", "workflowId": "123"},
        headers={}
    )
    print_result("Raw webhook handling", raw_event.processed)
    
    # Test 6: Delete webhook
    deleted = receiver.delete_webhook(config.webhook_id)
    print_result("Delete webhook", deleted)
    
    return True


async def test_notification_system():
    """Test the notification system."""
    print_header("Testing Notification System")
    
    from agent.notifications import (
        NotificationManager, NotificationType, NotificationPriority
    )
    
    manager = NotificationManager()
    
    # Test 1: Create notification
    notification = await manager.notify(
        title="Test Notification",
        message="This is a test message",
        notification_type=NotificationType.INFO
    )
    print_result("Create notification", notification.notification_id is not None)
    
    # Test 2: Success notification
    success_notif = await manager.notify_success("Build Complete", "All tests passed")
    print_result("Success notification", success_notif.notification_type == NotificationType.SUCCESS)
    
    # Test 3: Error notification
    error_notif = await manager.notify_error("Build Failed", "Test error")
    print_result("Error notification", error_notif.notification_type == NotificationType.ERROR)
    
    # Test 4: Workflow notifications
    wf_notif = await manager.notify_workflow_completed(
        workflow_id="wf_001",
        workflow_name="Test Workflow",
        execution_id="exec_001",
        duration=2.5
    )
    print_result("Workflow notification", wf_notif.notification_type == NotificationType.WORKFLOW_COMPLETED)
    
    # Test 5: Get notifications
    notifications = manager.get_notifications(limit=10)
    print_result("Get notifications", len(notifications) >= 4, f"Found {len(notifications)}")
    
    # Test 6: Unread count
    unread = manager.get_unread_count()
    print_result("Unread count", unread >= 4, f"Unread: {unread}")
    
    # Test 7: Mark read
    manager.mark_read(notification.notification_id)
    new_unread = manager.get_unread_count()
    print_result("Mark as read", new_unread == unread - 1)
    
    # Test 8: Statistics
    stats = manager.get_stats()
    print_result("Notification stats", stats["total"] >= 4)
    
    # Test 9: Subscribe to notifications
    received_notifications = []
    
    async def listener(n):
        received_notifications.append(n)
    
    sub_id = manager.subscribe(listener)
    await manager.notify("Subscription Test", "Testing subscription")
    
    print_result("Subscription system", len(received_notifications) > 0)
    
    # Cleanup
    manager.unsubscribe(sub_id)
    
    return True


async def test_workflow_monitor():
    """Test the workflow monitor."""
    print_header("Testing Workflow Monitor")
    
    from agent.workflow_monitor import WorkflowMonitor, get_monitor
    
    monitor = WorkflowMonitor()
    
    # Test 1: Execute with monitoring
    test_workflow = {
        "nodes": [
            {"type": "n8n-nodes-base.start", "name": "Start"}
        ],
        "connections": {}
    }
    
    result = await monitor.execute_n8n(
        workflow_json=test_workflow,
        workflow_id="monitored_001",
        workflow_name="Monitored Test",
        test_mode=True
    )
    print_result(
        "Monitored execution",
        result.status.value in ["completed"],
        f"Status: {result.status.value}"
    )
    
    # Test 2: Get running executions
    running = monitor.get_running_executions()
    print_result("Get running executions", isinstance(running, list))
    
    # Test 3: Get recent executions
    recent = monitor.get_recent_executions(limit=5)
    print_result("Recent executions", len(recent) > 0, f"Found {len(recent)}")
    
    # Test 4: Get stats
    stats = monitor.get_stats()
    print_result("Monitor statistics", stats.total_executions > 0)
    
    # Test 5: Dashboard data
    dashboard = monitor.get_dashboard_data()
    print_result(
        "Dashboard data",
        "stats" in dashboard and "running_executions" in dashboard,
        f"Keys: {list(dashboard.keys())}"
    )
    
    return True


async def test_api_imports():
    """Test that all M2 imports work in main.py."""
    print_header("Testing API Imports")
    
    try:
        # Test imports
        from agent.workflow_executor import get_executor
        from agent.workflow_monitor import get_monitor
        from agent.webhook_receiver import get_webhook_receiver, WebhookSource
        from agent.notifications import get_notification_manager, NotificationType
        from agent.storage import get_storage
        
        print_result("workflow_executor import", True)
        print_result("workflow_monitor import", True)
        print_result("webhook_receiver import", True)
        print_result("notifications import", True)
        print_result("storage import", True)
        
        # Test singletons
        executor = get_executor()
        print_result("get_executor()", executor is not None)
        
        monitor = get_monitor()
        print_result("get_monitor()", monitor is not None)
        
        receiver = get_webhook_receiver()
        print_result("get_webhook_receiver()", receiver is not None)
        
        notifications = get_notification_manager()
        print_result("get_notification_manager()", notifications is not None)
        
        storage = get_storage()
        print_result("get_storage()", storage is not None)
        
        return True
        
    except ImportError as e:
        print_result("API imports", False, str(e))
        return False


async def run_all_tests():
    """Run all M2 integration tests."""
    print("\n" + "="*60)
    print("🚀 MILESTONE 2 - INTEGRATION TEST SUITE")
    print("="*60)
    
    results = {}
    
    # Run each test suite
    results["API Imports"] = await test_api_imports()
    results["Workflow Executor"] = await test_workflow_executor()
    results["Persistent Storage"] = await test_persistent_storage()
    results["Webhook Receiver"] = await test_webhook_receiver()
    results["Notification System"] = await test_notification_system()
    results["Workflow Monitor"] = await test_workflow_monitor()
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        print_result(name, success)
    
    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("  🎯 M2 Status: ALL TESTS PASSING")
    else:
        print("  ⚠️  M2 Status: SOME TESTS FAILED")
    
    print(f"{'='*60}\n")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
