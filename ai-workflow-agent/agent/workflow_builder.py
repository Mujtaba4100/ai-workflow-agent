"""
Visual Workflow Builder - M3 Feature
Data structures and APIs for visual workflow construction
Provides node types, connections, and validation for UI builders
"""

import uuid
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy

logger = logging.getLogger(__name__)


class NodeCategory(str, Enum):
    """Categories of workflow nodes"""
    TRIGGER = "trigger"
    ACTION = "action"
    LOGIC = "logic"
    DATA = "data"
    AI = "ai"
    INTEGRATION = "integration"
    OUTPUT = "output"
    CUSTOM = "custom"


class DataType(str, Enum):
    """Data types for node inputs/outputs"""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    IMAGE = "image"
    FILE = "file"
    ANY = "any"


@dataclass
class NodePort:
    """Input or output port on a node"""
    port_id: str
    name: str
    data_type: DataType
    required: bool = False
    default_value: Any = None
    description: str = ""
    multiple: bool = False  # Can accept multiple connections
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "port_id": self.port_id,
            "name": self.name,
            "data_type": self.data_type.value,
            "required": self.required,
            "default_value": self.default_value,
            "description": self.description,
            "multiple": self.multiple
        }


@dataclass
class NodeDefinition:
    """Definition of a node type available in the builder"""
    node_type: str
    name: str
    description: str
    category: NodeCategory
    inputs: List[NodePort]
    outputs: List[NodePort]
    properties: Dict[str, Any] = field(default_factory=dict)
    icon: str = ""
    color: str = "#6366f1"  # Default indigo
    platform: str = "both"  # n8n, comfyui, both
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": self.node_type,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "properties": self.properties,
            "icon": self.icon,
            "color": self.color,
            "platform": self.platform
        }


@dataclass
class WorkflowNode:
    """Instance of a node in a workflow"""
    node_id: str
    node_type: str
    name: str
    position: Dict[str, float]  # {"x": 100, "y": 200}
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "name": self.name,
            "position": self.position,
            "properties": self.properties
        }


@dataclass
class WorkflowConnection:
    """Connection between two nodes"""
    connection_id: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "connection_id": self.connection_id,
            "source_node": self.source_node,
            "source_port": self.source_port,
            "target_node": self.target_node,
            "target_port": self.target_port
        }


@dataclass
class VisualWorkflow:
    """A complete visual workflow"""
    workflow_id: str
    name: str
    description: str
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection]
    platform: str  # n8n, comfyui, hybrid
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "nodes": [n.to_dict() for n in self.nodes],
            "connections": [c.to_dict() for c in self.connections],
            "platform": self.platform,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class NodeRegistry:
    """Registry of available node types"""
    
    def __init__(self):
        self._nodes: Dict[str, NodeDefinition] = {}
        self._register_builtin_nodes()
        
    def _register_builtin_nodes(self):
        """Register built-in node types"""
        
        # ============== TRIGGER NODES ==============
        self.register(NodeDefinition(
            node_type="trigger.webhook",
            name="Webhook Trigger",
            description="Trigger workflow from incoming webhook",
            category=NodeCategory.TRIGGER,
            inputs=[],
            outputs=[
                NodePort("body", "Body", DataType.OBJECT, description="Webhook payload"),
                NodePort("headers", "Headers", DataType.OBJECT, description="Request headers")
            ],
            properties={"method": "POST", "path": "/webhook"},
            icon="webhook",
            color="#10b981",
            platform="n8n"
        ))
        
        self.register(NodeDefinition(
            node_type="trigger.schedule",
            name="Schedule Trigger",
            description="Trigger workflow on a schedule",
            category=NodeCategory.TRIGGER,
            inputs=[],
            outputs=[
                NodePort("timestamp", "Timestamp", DataType.STRING)
            ],
            properties={"cron": "0 9 * * *"},
            icon="clock",
            color="#10b981",
            platform="n8n"
        ))
        
        self.register(NodeDefinition(
            node_type="trigger.manual",
            name="Manual Trigger",
            description="Manually start the workflow",
            category=NodeCategory.TRIGGER,
            inputs=[],
            outputs=[
                NodePort("data", "Data", DataType.OBJECT)
            ],
            icon="play",
            color="#10b981",
            platform="both"
        ))
        
        # ============== ACTION NODES ==============
        self.register(NodeDefinition(
            node_type="action.http",
            name="HTTP Request",
            description="Make HTTP request to external API",
            category=NodeCategory.ACTION,
            inputs=[
                NodePort("url", "URL", DataType.STRING, required=True),
                NodePort("body", "Body", DataType.OBJECT),
                NodePort("headers", "Headers", DataType.OBJECT)
            ],
            outputs=[
                NodePort("response", "Response", DataType.OBJECT),
                NodePort("status", "Status Code", DataType.NUMBER)
            ],
            properties={"method": "GET", "timeout": 30},
            icon="globe",
            color="#3b82f6",
            platform="n8n"
        ))
        
        self.register(NodeDefinition(
            node_type="action.email",
            name="Send Email",
            description="Send an email message",
            category=NodeCategory.ACTION,
            inputs=[
                NodePort("to", "To", DataType.STRING, required=True),
                NodePort("subject", "Subject", DataType.STRING, required=True),
                NodePort("body", "Body", DataType.STRING, required=True)
            ],
            outputs=[
                NodePort("success", "Success", DataType.BOOLEAN)
            ],
            properties={"smtp_host": "", "smtp_port": 587},
            icon="mail",
            color="#3b82f6",
            platform="n8n"
        ))
        
        self.register(NodeDefinition(
            node_type="action.slack",
            name="Slack Message",
            description="Send message to Slack channel",
            category=NodeCategory.ACTION,
            inputs=[
                NodePort("channel", "Channel", DataType.STRING, required=True),
                NodePort("message", "Message", DataType.STRING, required=True)
            ],
            outputs=[
                NodePort("success", "Success", DataType.BOOLEAN)
            ],
            properties={"webhook_url": ""},
            icon="slack",
            color="#3b82f6",
            platform="n8n"
        ))
        
        # ============== LOGIC NODES ==============
        self.register(NodeDefinition(
            node_type="logic.if",
            name="IF Condition",
            description="Branch workflow based on condition",
            category=NodeCategory.LOGIC,
            inputs=[
                NodePort("value", "Value", DataType.ANY, required=True),
                NodePort("condition", "Condition", DataType.STRING)
            ],
            outputs=[
                NodePort("true", "True", DataType.ANY),
                NodePort("false", "False", DataType.ANY)
            ],
            properties={"operator": "equals", "compare_value": ""},
            icon="git-branch",
            color="#f59e0b",
            platform="n8n"
        ))
        
        self.register(NodeDefinition(
            node_type="logic.switch",
            name="Switch",
            description="Route to different outputs based on value",
            category=NodeCategory.LOGIC,
            inputs=[
                NodePort("value", "Value", DataType.ANY, required=True)
            ],
            outputs=[
                NodePort("case1", "Case 1", DataType.ANY),
                NodePort("case2", "Case 2", DataType.ANY),
                NodePort("default", "Default", DataType.ANY)
            ],
            properties={"cases": []},
            icon="shuffle",
            color="#f59e0b",
            platform="n8n"
        ))
        
        self.register(NodeDefinition(
            node_type="logic.loop",
            name="Loop",
            description="Iterate over array items",
            category=NodeCategory.LOGIC,
            inputs=[
                NodePort("items", "Items", DataType.ARRAY, required=True)
            ],
            outputs=[
                NodePort("item", "Current Item", DataType.ANY),
                NodePort("index", "Index", DataType.NUMBER)
            ],
            icon="repeat",
            color="#f59e0b",
            platform="n8n"
        ))
        
        # ============== DATA NODES ==============
        self.register(NodeDefinition(
            node_type="data.set",
            name="Set Data",
            description="Set or transform data values",
            category=NodeCategory.DATA,
            inputs=[
                NodePort("input", "Input", DataType.ANY)
            ],
            outputs=[
                NodePort("output", "Output", DataType.ANY)
            ],
            properties={"values": []},
            icon="edit",
            color="#8b5cf6",
            platform="n8n"
        ))
        
        self.register(NodeDefinition(
            node_type="data.merge",
            name="Merge Data",
            description="Merge multiple data streams",
            category=NodeCategory.DATA,
            inputs=[
                NodePort("input1", "Input 1", DataType.ANY, multiple=True),
                NodePort("input2", "Input 2", DataType.ANY, multiple=True)
            ],
            outputs=[
                NodePort("merged", "Merged", DataType.ARRAY)
            ],
            properties={"mode": "append"},
            icon="git-merge",
            color="#8b5cf6",
            platform="n8n"
        ))
        
        self.register(NodeDefinition(
            node_type="data.filter",
            name="Filter",
            description="Filter array items by condition",
            category=NodeCategory.DATA,
            inputs=[
                NodePort("items", "Items", DataType.ARRAY, required=True)
            ],
            outputs=[
                NodePort("filtered", "Filtered", DataType.ARRAY)
            ],
            properties={"field": "", "operator": "equals", "value": ""},
            icon="filter",
            color="#8b5cf6",
            platform="n8n"
        ))
        
        # ============== AI NODES ==============
        self.register(NodeDefinition(
            node_type="ai.generate_image",
            name="Generate Image",
            description="Generate image using AI model",
            category=NodeCategory.AI,
            inputs=[
                NodePort("prompt", "Prompt", DataType.STRING, required=True),
                NodePort("negative_prompt", "Negative Prompt", DataType.STRING),
                NodePort("seed", "Seed", DataType.NUMBER)
            ],
            outputs=[
                NodePort("image", "Image", DataType.IMAGE)
            ],
            properties={
                "model": "stable-diffusion-xl",
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "cfg_scale": 7.5
            },
            icon="image",
            color="#ec4899",
            platform="comfyui"
        ))
        
        self.register(NodeDefinition(
            node_type="ai.upscale",
            name="Upscale Image",
            description="Upscale image using AI",
            category=NodeCategory.AI,
            inputs=[
                NodePort("image", "Image", DataType.IMAGE, required=True)
            ],
            outputs=[
                NodePort("upscaled", "Upscaled Image", DataType.IMAGE)
            ],
            properties={"scale": 2, "model": "RealESRGAN_x4plus"},
            icon="maximize",
            color="#ec4899",
            platform="comfyui"
        ))
        
        self.register(NodeDefinition(
            node_type="ai.llm",
            name="LLM Chat",
            description="Chat with language model",
            category=NodeCategory.AI,
            inputs=[
                NodePort("prompt", "Prompt", DataType.STRING, required=True),
                NodePort("system", "System Prompt", DataType.STRING),
                NodePort("context", "Context", DataType.OBJECT)
            ],
            outputs=[
                NodePort("response", "Response", DataType.STRING)
            ],
            properties={
                "model": "qwen2.5:7b",
                "temperature": 0.7,
                "max_tokens": 512
            },
            icon="message-circle",
            color="#ec4899",
            platform="both"
        ))
        
        self.register(NodeDefinition(
            node_type="ai.inpaint",
            name="Inpaint",
            description="Inpaint image regions",
            category=NodeCategory.AI,
            inputs=[
                NodePort("image", "Image", DataType.IMAGE, required=True),
                NodePort("mask", "Mask", DataType.IMAGE, required=True),
                NodePort("prompt", "Prompt", DataType.STRING, required=True)
            ],
            outputs=[
                NodePort("result", "Result", DataType.IMAGE)
            ],
            properties={"denoise": 0.8},
            icon="edit-3",
            color="#ec4899",
            platform="comfyui"
        ))
        
        # ============== INTEGRATION NODES ==============
        self.register(NodeDefinition(
            node_type="integration.database",
            name="Database Query",
            description="Query SQL database",
            category=NodeCategory.INTEGRATION,
            inputs=[
                NodePort("query", "Query", DataType.STRING, required=True),
                NodePort("params", "Parameters", DataType.ARRAY)
            ],
            outputs=[
                NodePort("rows", "Rows", DataType.ARRAY)
            ],
            properties={"connection": "", "database": ""},
            icon="database",
            color="#06b6d4",
            platform="n8n"
        ))
        
        self.register(NodeDefinition(
            node_type="integration.google_sheets",
            name="Google Sheets",
            description="Read/write Google Sheets",
            category=NodeCategory.INTEGRATION,
            inputs=[
                NodePort("data", "Data", DataType.ARRAY)
            ],
            outputs=[
                NodePort("rows", "Rows", DataType.ARRAY)
            ],
            properties={"spreadsheet_id": "", "sheet": "", "operation": "read"},
            icon="table",
            color="#06b6d4",
            platform="n8n"
        ))
        
        # ============== OUTPUT NODES ==============
        self.register(NodeDefinition(
            node_type="output.save_image",
            name="Save Image",
            description="Save image to file",
            category=NodeCategory.OUTPUT,
            inputs=[
                NodePort("image", "Image", DataType.IMAGE, required=True),
                NodePort("filename", "Filename", DataType.STRING)
            ],
            outputs=[
                NodePort("path", "File Path", DataType.STRING)
            ],
            properties={"format": "png", "output_dir": "outputs"},
            icon="save",
            color="#64748b",
            platform="comfyui"
        ))
        
        self.register(NodeDefinition(
            node_type="output.respond",
            name="Respond",
            description="Send response back to trigger",
            category=NodeCategory.OUTPUT,
            inputs=[
                NodePort("data", "Data", DataType.ANY, required=True)
            ],
            outputs=[],
            properties={"status_code": 200},
            icon="send",
            color="#64748b",
            platform="n8n"
        ))
        
    def register(self, node_def: NodeDefinition):
        """Register a node type"""
        self._nodes[node_def.node_type] = node_def
        
    def get(self, node_type: str) -> Optional[NodeDefinition]:
        """Get node definition by type"""
        return self._nodes.get(node_type)
    
    def list_all(self) -> List[NodeDefinition]:
        """List all registered nodes"""
        return list(self._nodes.values())
    
    def list_by_category(self, category: NodeCategory) -> List[NodeDefinition]:
        """List nodes by category"""
        return [n for n in self._nodes.values() if n.category == category]
    
    def list_by_platform(self, platform: str) -> List[NodeDefinition]:
        """List nodes by platform"""
        return [n for n in self._nodes.values() if n.platform in [platform, "both"]]
    
    def search(self, query: str) -> List[NodeDefinition]:
        """Search nodes by name or description"""
        query = query.lower()
        return [
            n for n in self._nodes.values()
            if query in n.name.lower() or query in n.description.lower()
        ]


class WorkflowBuilder:
    """
    Visual workflow builder.
    Provides APIs for UI to construct workflows visually.
    """
    
    def __init__(self):
        self.registry = NodeRegistry()
        self._workflows: Dict[str, VisualWorkflow] = {}
        
    def create_workflow(
        self,
        name: str,
        description: str = "",
        platform: str = "n8n"
    ) -> VisualWorkflow:
        """Create a new visual workflow"""
        workflow_id = str(uuid.uuid4())
        
        workflow = VisualWorkflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            nodes=[],
            connections=[],
            platform=platform
        )
        
        self._workflows[workflow_id] = workflow
        logger.info(f"Created workflow: {name} ({workflow_id})")
        
        return workflow
    
    def get_workflow(self, workflow_id: str) -> Optional[VisualWorkflow]:
        """Get workflow by ID"""
        return self._workflows.get(workflow_id)
    
    def list_workflows(self) -> List[VisualWorkflow]:
        """List all workflows"""
        return list(self._workflows.values())
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow"""
        if workflow_id in self._workflows:
            del self._workflows[workflow_id]
            return True
        return False
    
    def add_node(
        self,
        workflow_id: str,
        node_type: str,
        position: Dict[str, float],
        name: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> Optional[WorkflowNode]:
        """Add a node to workflow"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None
            
        node_def = self.registry.get(node_type)
        if not node_def:
            logger.error(f"Unknown node type: {node_type}")
            return None
            
        node_id = str(uuid.uuid4())
        node_name = name or f"{node_def.name} {len(workflow.nodes) + 1}"
        
        # Merge default properties with provided
        node_props = deepcopy(node_def.properties)
        if properties:
            node_props.update(properties)
        
        node = WorkflowNode(
            node_id=node_id,
            node_type=node_type,
            name=node_name,
            position=position,
            properties=node_props
        )
        
        workflow.nodes.append(node)
        workflow.updated_at = datetime.now()
        
        return node
    
    def update_node(
        self,
        workflow_id: str,
        node_id: str,
        name: Optional[str] = None,
        position: Optional[Dict[str, float]] = None,
        properties: Optional[Dict[str, Any]] = None
    ) -> Optional[WorkflowNode]:
        """Update a node in workflow"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None
            
        for node in workflow.nodes:
            if node.node_id == node_id:
                if name:
                    node.name = name
                if position:
                    node.position = position
                if properties:
                    node.properties.update(properties)
                workflow.updated_at = datetime.now()
                return node
                
        return None
    
    def remove_node(self, workflow_id: str, node_id: str) -> bool:
        """Remove a node from workflow"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return False
            
        # Remove node
        workflow.nodes = [n for n in workflow.nodes if n.node_id != node_id]
        
        # Remove connections to/from this node
        workflow.connections = [
            c for c in workflow.connections
            if c.source_node != node_id and c.target_node != node_id
        ]
        
        workflow.updated_at = datetime.now()
        return True
    
    def add_connection(
        self,
        workflow_id: str,
        source_node: str,
        source_port: str,
        target_node: str,
        target_port: str
    ) -> Optional[WorkflowConnection]:
        """Add a connection between nodes"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None
            
        # Validate connection
        validation = self._validate_connection(
            workflow, source_node, source_port, target_node, target_port
        )
        if not validation["valid"]:
            logger.error(f"Invalid connection: {validation['error']}")
            return None
            
        connection_id = str(uuid.uuid4())
        
        connection = WorkflowConnection(
            connection_id=connection_id,
            source_node=source_node,
            source_port=source_port,
            target_node=target_node,
            target_port=target_port
        )
        
        workflow.connections.append(connection)
        workflow.updated_at = datetime.now()
        
        return connection
    
    def remove_connection(self, workflow_id: str, connection_id: str) -> bool:
        """Remove a connection"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return False
            
        workflow.connections = [
            c for c in workflow.connections
            if c.connection_id != connection_id
        ]
        workflow.updated_at = datetime.now()
        return True
    
    def _validate_connection(
        self,
        workflow: VisualWorkflow,
        source_node: str,
        source_port: str,
        target_node: str,
        target_port: str
    ) -> Dict[str, Any]:
        """Validate a connection"""
        # Find source and target nodes
        source = None
        target = None
        
        for node in workflow.nodes:
            if node.node_id == source_node:
                source = node
            if node.node_id == target_node:
                target = node
                
        if not source:
            return {"valid": False, "error": "Source node not found"}
        if not target:
            return {"valid": False, "error": "Target node not found"}
            
        # Check for cycles (simple check)
        if source_node == target_node:
            return {"valid": False, "error": "Cannot connect node to itself"}
            
        # Get node definitions
        source_def = self.registry.get(source.node_type)
        target_def = self.registry.get(target.node_type)
        
        if not source_def or not target_def:
            return {"valid": False, "error": "Invalid node type"}
            
        # Validate port exists and types are compatible
        source_port_def = None
        for port in source_def.outputs:
            if port.port_id == source_port:
                source_port_def = port
                break
                
        target_port_def = None
        for port in target_def.inputs:
            if port.port_id == target_port:
                target_port_def = port
                break
                
        if not source_port_def:
            return {"valid": False, "error": f"Source port '{source_port}' not found"}
        if not target_port_def:
            return {"valid": False, "error": f"Target port '{target_port}' not found"}
            
        # Type compatibility
        if source_port_def.data_type != DataType.ANY and target_port_def.data_type != DataType.ANY:
            if source_port_def.data_type != target_port_def.data_type:
                return {
                    "valid": False,
                    "error": f"Type mismatch: {source_port_def.data_type.value} -> {target_port_def.data_type.value}"
                }
                
        return {"valid": True}
    
    def validate_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Validate entire workflow"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return {"valid": False, "errors": ["Workflow not found"]}
            
        errors = []
        warnings = []
        
        # Check for trigger
        has_trigger = False
        for node in workflow.nodes:
            node_def = self.registry.get(node.node_type)
            if node_def and node_def.category == NodeCategory.TRIGGER:
                has_trigger = True
                break
                
        if not has_trigger:
            warnings.append("Workflow has no trigger node")
            
        # Check for disconnected nodes
        connected_nodes: Set[str] = set()
        for conn in workflow.connections:
            connected_nodes.add(conn.source_node)
            connected_nodes.add(conn.target_node)
            
        for node in workflow.nodes:
            node_def = self.registry.get(node.node_type)
            if node_def and node_def.category != NodeCategory.TRIGGER:
                if node.node_id not in connected_nodes:
                    warnings.append(f"Node '{node.name}' is disconnected")
                    
        # Check required inputs
        for node in workflow.nodes:
            node_def = self.registry.get(node.node_type)
            if not node_def:
                continue
                
            for input_port in node_def.inputs:
                if input_port.required:
                    # Check if connected
                    is_connected = any(
                        c.target_node == node.node_id and c.target_port == input_port.port_id
                        for c in workflow.connections
                    )
                    # Check if has default/property value
                    has_value = input_port.port_id in node.properties
                    
                    if not is_connected and not has_value:
                        errors.append(
                            f"Required input '{input_port.name}' on '{node.name}' is not connected"
                        )
                        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "node_count": len(workflow.nodes),
            "connection_count": len(workflow.connections)
        }
    
    def export_workflow(self, workflow_id: str, format: str = "json") -> Optional[Dict[str, Any]]:
        """Export workflow to deployable format"""
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return None
            
        if format == "n8n":
            return self._export_to_n8n(workflow)
        elif format == "comfyui":
            return self._export_to_comfyui(workflow)
        else:
            return workflow.to_dict()
    
    def _export_to_n8n(self, workflow: VisualWorkflow) -> Dict[str, Any]:
        """Export to n8n workflow format"""
        n8n_nodes = []
        n8n_connections: Dict[str, Any] = {}
        
        for node in workflow.nodes:
            n8n_node = {
                "id": node.node_id,
                "name": node.name,
                "type": self._map_to_n8n_type(node.node_type),
                "position": [node.position.get("x", 0), node.position.get("y", 0)],
                "parameters": node.properties
            }
            n8n_nodes.append(n8n_node)
            
        # Build connections
        for conn in workflow.connections:
            source_name = None
            target_name = None
            
            for node in workflow.nodes:
                if node.node_id == conn.source_node:
                    source_name = node.name
                if node.node_id == conn.target_node:
                    target_name = node.name
                    
            if source_name and target_name:
                if source_name not in n8n_connections:
                    n8n_connections[source_name] = {"main": [[]]}
                n8n_connections[source_name]["main"][0].append({
                    "node": target_name,
                    "type": "main",
                    "index": 0
                })
                
        return {
            "name": workflow.name,
            "nodes": n8n_nodes,
            "connections": n8n_connections,
            "settings": {},
            "staticData": None
        }
    
    def _export_to_comfyui(self, workflow: VisualWorkflow) -> Dict[str, Any]:
        """Export to ComfyUI workflow format"""
        comfy_workflow = {}
        
        for i, node in enumerate(workflow.nodes):
            comfy_workflow[str(i)] = {
                "class_type": self._map_to_comfyui_type(node.node_type),
                "inputs": node.properties,
                "_meta": {
                    "title": node.name
                }
            }
            
        return comfy_workflow
    
    def _map_to_n8n_type(self, node_type: str) -> str:
        """Map internal node type to n8n type"""
        mapping = {
            "trigger.webhook": "n8n-nodes-base.webhook",
            "trigger.schedule": "n8n-nodes-base.scheduleTrigger",
            "trigger.manual": "n8n-nodes-base.manualTrigger",
            "action.http": "n8n-nodes-base.httpRequest",
            "action.email": "n8n-nodes-base.emailSend",
            "action.slack": "n8n-nodes-base.slack",
            "logic.if": "n8n-nodes-base.if",
            "logic.switch": "n8n-nodes-base.switch",
            "data.set": "n8n-nodes-base.set",
            "data.merge": "n8n-nodes-base.merge",
            "output.respond": "n8n-nodes-base.respondToWebhook"
        }
        return mapping.get(node_type, "n8n-nodes-base.noOp")
    
    def _map_to_comfyui_type(self, node_type: str) -> str:
        """Map internal node type to ComfyUI type"""
        mapping = {
            "ai.generate_image": "KSampler",
            "ai.upscale": "UpscaleImage",
            "ai.inpaint": "InpaintNode",
            "output.save_image": "SaveImage"
        }
        return mapping.get(node_type, "EmptyNode")


# Singleton instances
_registry: Optional[NodeRegistry] = None
_builder: Optional[WorkflowBuilder] = None


def get_node_registry() -> NodeRegistry:
    """Get the global node registry"""
    global _registry
    if _registry is None:
        _registry = NodeRegistry()
    return _registry


def get_workflow_builder() -> WorkflowBuilder:
    """Get the global workflow builder"""
    global _builder
    if _builder is None:
        _builder = WorkflowBuilder()
    return _builder
