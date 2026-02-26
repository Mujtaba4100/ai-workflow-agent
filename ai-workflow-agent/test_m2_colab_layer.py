"""
Milestone 2 Integration Tests (CORRECT VERSION)
Tests for: ColabCode, pyngrok Tunnel, Auto-Offload, Fallback
"""

import sys
import os
import asyncio
import pytest
from datetime import datetime

# Add agent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agent'))


def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"🔬 {text}")
    print(f"{'='*60}")


def print_result(name: str, success: bool, details: str = ""):
    status = "✅" if success else "❌"
    print(f"{status} {name}")
    if details:
        print(f"   {details}")


# ============================================================
# Test: Imports
# ============================================================

class TestM2Imports:
    """Test that all M2 modules import correctly"""
    
    def test_import_colab_layer(self):
        """Test colab_layer imports"""
        from agent.colab_layer import (
            ColabLayer,
            TunnelManager,
            ColabCodeManager,
            OffloadDecider,
            FallbackHandler,
            get_colab_layer
        )
        assert ColabLayer is not None
        assert TunnelManager is not None
        assert ColabCodeManager is not None
        assert OffloadDecider is not None
        assert FallbackHandler is not None
        print_result("colab_layer imports", True)
    
    def test_import_enums(self):
        """Test enum imports"""
        from agent.colab_layer import (
            TaskComplexity,
            ExecutionTarget,
            TaskStatus,
            TunnelStatus
        )
        assert len(TaskComplexity) == 4  # LIGHT, MEDIUM, HEAVY, GPU_REQUIRED
        assert len(ExecutionTarget) == 3  # LOCAL, COLAB, AUTO
        assert len(TaskStatus) == 7
        assert len(TunnelStatus) == 4
        print_result("enum imports", True)
    
    def test_import_dataclasses(self):
        """Test dataclass imports"""
        from agent.colab_layer import ColabTask, TunnelInfo
        assert ColabTask is not None
        assert TunnelInfo is not None
        print_result("dataclass imports", True)


# ============================================================
# Test: Tunnel Manager
# ============================================================

class TestTunnelManager:
    """Test pyngrok tunnel functionality"""
    
    def test_tunnel_manager_creation(self):
        """Test TunnelManager instantiation"""
        from agent.colab_layer import TunnelManager
        
        manager = TunnelManager()
        assert manager is not None
        assert manager.tunnels == {}
        print_result("TunnelManager creation", True)
    
    def test_tunnel_manager_status(self):
        """Test tunnel status without auth"""
        from agent.colab_layer import TunnelManager
        
        manager = TunnelManager()
        status = manager.get_status()
        
        assert "initialized" in status
        assert "active_tunnels" in status
        assert status["active_tunnels"] == 0
        print_result("TunnelManager status", True, f"Status: {status}")
    
    def test_tunnel_list_empty(self):
        """Test listing tunnels when none exist"""
        from agent.colab_layer import TunnelManager
        
        manager = TunnelManager()
        tunnels = manager.list_tunnels()
        
        assert tunnels == []
        print_result("TunnelManager list empty", True)


# ============================================================
# Test: ColabCode Manager
# ============================================================

class TestColabCodeManager:
    """Test ColabCode integration"""
    
    def test_colabcode_manager_creation(self):
        """Test ColabCodeManager instantiation"""
        from agent.colab_layer import ColabCodeManager, TunnelManager
        
        tunnel_mgr = TunnelManager()
        manager = ColabCodeManager(tunnel_mgr)
        
        assert manager is not None
        assert manager.active_sessions == {}
        print_result("ColabCodeManager creation", True)
    
    def test_generate_colabcode_notebook(self):
        """Test ColabCode notebook generation"""
        import json
        from agent.colab_layer import ColabCodeManager, TunnelManager
        
        tunnel_mgr = TunnelManager()
        manager = ColabCodeManager(tunnel_mgr)
        
        notebook = manager.generate_colabcode_notebook(
            port=10000,
            password="test123"
        )
        
        assert notebook is not None
        nb_dict = json.loads(notebook)
        assert nb_dict["nbformat"] == 4
        assert "cells" in nb_dict
        assert len(nb_dict["cells"]) >= 3
        print_result("ColabCode notebook generation", True)
    
    def test_generate_task_notebook(self):
        """Test task notebook generation"""
        import json
        from agent.colab_layer import ColabCodeManager, TunnelManager
        
        tunnel_mgr = TunnelManager()
        manager = ColabCodeManager(tunnel_mgr)
        
        code = """
import torch
print(f"GPU available: {torch.cuda.is_available()}")
result = {"status": "success"}
"""
        
        notebook = manager.generate_task_notebook(
            task_code=code,
            task_name="GPU Test",
            requirements=["torch"]
        )
        
        assert notebook is not None
        nb_dict = json.loads(notebook)
        assert nb_dict["metadata"]["accelerator"] == "GPU"
        print_result("Task notebook generation", True)
    
    def test_create_session(self):
        """Test session creation"""
        from agent.colab_layer import ColabCodeManager, TunnelManager
        
        tunnel_mgr = TunnelManager()
        manager = ColabCodeManager(tunnel_mgr)
        
        session = manager.create_session(
            session_name="Test Session",
            port=10000,
            password="secret"
        )
        
        assert "session_id" in session
        assert session["name"] == "Test Session"
        assert session["port"] == 10000
        assert "notebook" in session
        print_result("Session creation", True, f"Session ID: {session['session_id']}")
    
    def test_list_sessions(self):
        """Test session listing"""
        from agent.colab_layer import ColabCodeManager, TunnelManager
        
        tunnel_mgr = TunnelManager()
        manager = ColabCodeManager(tunnel_mgr)
        
        # Create two sessions
        manager.create_session("Session 1", 10000)
        manager.create_session("Session 2", 10001)
        
        sessions = manager.list_sessions()
        assert len(sessions) == 2
        print_result("Session listing", True, f"Sessions: {len(sessions)}")


# ============================================================
# Test: Offload Decider
# ============================================================

class TestOffloadDecider:
    """Test auto-offload logic"""
    
    def test_offload_decider_creation(self):
        """Test OffloadDecider instantiation"""
        from agent.colab_layer import OffloadDecider
        
        decider = OffloadDecider(local_gpu_available=False)
        assert decider is not None
        assert not decider.local_gpu_available
        print_result("OffloadDecider creation", True)
    
    def test_analyze_light_task(self):
        """Test light task analysis"""
        from agent.colab_layer import OffloadDecider, TaskComplexity
        
        decider = OffloadDecider()
        
        light_code = """
print("Hello World")
x = 1 + 2
"""
        
        complexity = decider.analyze_task(light_code)
        assert complexity == TaskComplexity.LIGHT
        print_result("Light task analysis", True, f"Complexity: {complexity.value}")
    
    def test_analyze_gpu_task(self):
        """Test GPU-required task analysis"""
        from agent.colab_layer import OffloadDecider, TaskComplexity
        
        decider = OffloadDecider()
        
        gpu_code = """
import torch
model = torch.nn.Linear(100, 10).cuda()
output = model(torch.randn(32, 100).cuda())
"""
        
        complexity = decider.analyze_task(gpu_code)
        assert complexity == TaskComplexity.GPU_REQUIRED
        print_result("GPU task analysis", True, f"Complexity: {complexity.value}")
    
    def test_analyze_heavy_task(self):
        """Test heavy task analysis"""
        from agent.colab_layer import OffloadDecider, TaskComplexity
        
        decider = OffloadDecider()
        
        heavy_code = """
# Training large model
for epoch in range(100):
    model.train()
    batch_training(large_dataset)
"""
        
        complexity = decider.analyze_task(heavy_code)
        assert complexity in [TaskComplexity.HEAVY, TaskComplexity.GPU_REQUIRED]
        print_result("Heavy task analysis", True, f"Complexity: {complexity.value}")
    
    def test_decide_target_gpu_required(self):
        """Test target decision for GPU-required task"""
        from agent.colab_layer import OffloadDecider, TaskComplexity, ExecutionTarget
        
        decider = OffloadDecider(local_gpu_available=False)
        
        target = decider.decide_target(TaskComplexity.GPU_REQUIRED)
        assert target == ExecutionTarget.COLAB
        print_result("GPU task -> Colab", True)
    
    def test_decide_target_light_task(self):
        """Test target decision for light task"""
        from agent.colab_layer import OffloadDecider, TaskComplexity, ExecutionTarget
        
        decider = OffloadDecider()
        
        target = decider.decide_target(TaskComplexity.LIGHT)
        assert target == ExecutionTarget.LOCAL
        print_result("Light task -> Local", True)
    
    def test_should_offload(self):
        """Test should_offload convenience method"""
        from agent.colab_layer import OffloadDecider
        
        decider = OffloadDecider(local_gpu_available=False)
        
        # GPU code should offload
        gpu_code = "import torch; model.cuda()"
        assert decider.should_offload(gpu_code) == True
        
        # Light code should not offload
        light_code = "print('hello')"
        assert decider.should_offload(light_code) == False
        
        print_result("should_offload method", True)
    
    def test_decision_stats(self):
        """Test decision statistics"""
        from agent.colab_layer import OffloadDecider, TaskComplexity
        
        decider = OffloadDecider()
        
        # Make some decisions
        decider.decide_target(TaskComplexity.LIGHT)
        decider.decide_target(TaskComplexity.HEAVY)
        decider.decide_target(TaskComplexity.GPU_REQUIRED)
        
        stats = decider.get_decision_stats()
        assert stats["total_decisions"] == 3
        print_result("Decision stats", True, f"Total: {stats['total_decisions']}")


# ============================================================
# Test: Fallback Handler
# ============================================================

class TestFallbackHandler:
    """Test fallback logic"""
    
    def test_fallback_handler_creation(self):
        """Test FallbackHandler instantiation"""
        from agent.colab_layer import FallbackHandler
        
        handler = FallbackHandler()
        assert handler is not None
        assert handler.fallback_history == []
        print_result("FallbackHandler creation", True)
    
    @pytest.mark.asyncio
    async def test_fallback_to_local(self):
        """Test fallback from Colab to local"""
        from agent.colab_layer import (
            FallbackHandler, ColabTask, TaskComplexity,
            TaskStatus, ExecutionTarget
        )
        from datetime import datetime
        
        handler = FallbackHandler()
        
        task = ColabTask(
            task_id="test_001",
            name="Test Task",
            code="print('hello')",
            complexity=TaskComplexity.MEDIUM,
            status=TaskStatus.PENDING,
            target=ExecutionTarget.COLAB
        )
        
        # Colab executor that fails
        async def colab_fail(t):
            raise Exception("Colab unavailable")
        
        # Local executor that succeeds
        async def local_success(t):
            return {"output": "hello", "source": "local"}
        
        result = await handler.execute_with_fallback(
            task,
            colab_fail,
            local_success
        )
        
        assert result.status == TaskStatus.COMPLETED
        assert result.fallback_used == True
        assert result.actual_target == "local"
        print_result("Fallback to local", True)
    
    @pytest.mark.asyncio
    async def test_colab_success_no_fallback(self):
        """Test Colab success without fallback"""
        from agent.colab_layer import (
            FallbackHandler, ColabTask, TaskComplexity,
            TaskStatus, ExecutionTarget
        )
        
        handler = FallbackHandler()
        
        task = ColabTask(
            task_id="test_002",
            name="Test Task",
            code="print('gpu task')",
            complexity=TaskComplexity.GPU_REQUIRED,
            status=TaskStatus.PENDING,
            target=ExecutionTarget.COLAB
        )
        
        # Colab executor that succeeds
        async def colab_success(t):
            return {"output": "gpu result", "source": "colab"}
        
        # Local executor (shouldn't be called)
        async def local_exec(t):
            return {"output": "local", "source": "local"}
        
        result = await handler.execute_with_fallback(
            task,
            colab_success,
            local_exec
        )
        
        assert result.status == TaskStatus.COMPLETED
        assert result.fallback_used == False
        assert result.actual_target == "colab"
        print_result("Colab success (no fallback)", True)
    
    def test_fallback_stats(self):
        """Test fallback statistics"""
        from agent.colab_layer import FallbackHandler
        
        handler = FallbackHandler()
        stats = handler.get_fallback_stats()
        
        assert stats["total_tasks"] == 0
        print_result("Fallback stats", True)


# ============================================================
# Test: Colab Layer (Main)
# ============================================================

class TestColabLayer:
    """Test main ColabLayer class"""
    
    def test_colab_layer_creation(self):
        """Test ColabLayer instantiation"""
        from agent.colab_layer import ColabLayer
        
        layer = ColabLayer()
        assert layer is not None
        assert layer.tasks == {}
        print_result("ColabLayer creation", True)
    
    def test_create_task_auto(self):
        """Test task creation with AUTO target"""
        from agent.colab_layer import ColabLayer, ExecutionTarget, TaskComplexity
        
        layer = ColabLayer()
        
        # Light task
        task = layer.create_task(
            name="Light Task",
            code="print('hello')",
            target=ExecutionTarget.AUTO
        )
        
        assert task.task_id is not None
        assert task.complexity == TaskComplexity.LIGHT
        assert task.target == ExecutionTarget.LOCAL
        print_result("Create light task", True, f"Target: {task.target.value}")
    
    def test_create_gpu_task(self):
        """Test GPU task creation"""
        from agent.colab_layer import ColabLayer, ExecutionTarget, TaskComplexity
        
        layer = ColabLayer(local_gpu=False)
        
        gpu_code = """
import torch
model = torch.nn.Linear(100, 10).cuda()
"""
        
        task = layer.create_task(
            name="GPU Task",
            code=gpu_code,
            target=ExecutionTarget.AUTO
        )
        
        assert task.complexity == TaskComplexity.GPU_REQUIRED
        assert task.target == ExecutionTarget.COLAB
        print_result("Create GPU task", True, f"Target: {task.target.value}")
    
    def test_list_tasks(self):
        """Test task listing"""
        from agent.colab_layer import ColabLayer, ExecutionTarget
        
        layer = ColabLayer()
        
        layer.create_task("Task 1", "print(1)", ExecutionTarget.LOCAL)
        layer.create_task("Task 2", "print(2)", ExecutionTarget.LOCAL)
        layer.create_task("Task 3", "print(3)", ExecutionTarget.LOCAL)
        
        tasks = layer.list_tasks()
        assert len(tasks) == 3
        print_result("List tasks", True, f"Count: {len(tasks)}")
    
    def test_get_task(self):
        """Test getting task by ID"""
        from agent.colab_layer import ColabLayer, ExecutionTarget
        
        layer = ColabLayer()
        
        task = layer.create_task("Test", "print('test')", ExecutionTarget.LOCAL)
        retrieved = layer.get_task(task.task_id)
        
        assert retrieved is not None
        assert retrieved.task_id == task.task_id
        print_result("Get task by ID", True)
    
    def test_generate_notebook(self):
        """Test notebook generation for task"""
        import json
        from agent.colab_layer import ColabLayer, ExecutionTarget
        
        layer = ColabLayer()
        
        task = layer.create_task(
            "GPU Task",
            "import torch; torch.cuda.is_available()",
            ExecutionTarget.COLAB
        )
        
        notebook = layer.generate_colab_notebook(task.task_id)
        assert notebook is not None
        
        nb_dict = json.loads(notebook)
        assert nb_dict["nbformat"] == 4
        print_result("Generate notebook", True)
    
    def test_get_stats(self):
        """Test comprehensive statistics"""
        from agent.colab_layer import ColabLayer, ExecutionTarget
        
        layer = ColabLayer()
        
        layer.create_task("Task 1", "print(1)", ExecutionTarget.LOCAL)
        layer.create_task("Task 2", "import torch; x.cuda()", ExecutionTarget.AUTO)
        
        stats = layer.get_stats()
        
        assert "tasks" in stats
        assert "offload" in stats
        assert "fallback" in stats
        assert "tunnels" in stats
        assert stats["tasks"]["total"] == 2
        print_result("Get stats", True, f"Tasks: {stats['tasks']}")
    
    @pytest.mark.asyncio
    async def test_execute_task_local(self):
        """Test local task execution"""
        from agent.colab_layer import ColabLayer, ExecutionTarget, TaskStatus
        
        layer = ColabLayer()
        
        task = layer.create_task(
            name="Simple Print",
            code="print('Hello from M2!')",
            target=ExecutionTarget.LOCAL
        )
        
        result = await layer.execute_task(task.task_id)
        
        assert result.status == TaskStatus.COMPLETED
        assert result.actual_target == "local"
        print_result("Execute local task", True)


# ============================================================
# Test: Singleton
# ============================================================

class TestSingleton:
    """Test singleton pattern"""
    
    def test_get_colab_layer(self):
        """Test singleton getter"""
        from agent.colab_layer import get_colab_layer, ColabLayer
        
        # Reset singleton for test
        import agent.colab_layer as cl
        cl._colab_layer = None
        
        layer1 = get_colab_layer()
        layer2 = get_colab_layer()
        
        assert layer1 is layer2
        assert isinstance(layer1, ColabLayer)
        print_result("Singleton pattern", True)


# ============================================================
# Run All Tests
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 MILESTONE 2 - COLAB LAYER TESTS")
    print("="*60)
    
    pytest.main([__file__, "-v", "--tb=short"])
