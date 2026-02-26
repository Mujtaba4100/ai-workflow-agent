"""
Milestone 3: Appsmith Dashboard Integration
Connect to Appsmith for visual dashboards showing workflow status
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import httpx

logger = logging.getLogger(__name__)


# ============================================================
# Enums and Data Classes
# ============================================================

class WidgetType(str, Enum):
    """Appsmith widget types"""
    TABLE = "TABLE_WIDGET"
    CHART = "CHART_WIDGET"
    TEXT = "TEXT_WIDGET"
    BUTTON = "BUTTON_WIDGET"
    CONTAINER = "CONTAINER_WIDGET"
    LIST = "LIST_WIDGET"
    STAT_BOX = "STATBOX_WIDGET"
    INPUT = "INPUT_WIDGET"
    IMAGE = "IMAGE_WIDGET"


class DashboardStatus(str, Enum):
    """Dashboard status"""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


@dataclass
class Widget:
    """A dashboard widget"""
    widget_id: str
    widget_type: WidgetType
    name: str
    properties: Dict[str, Any]
    position: Dict[str, int]  # x, y, width, height
    data_source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "widgetId": self.widget_id,
            "type": self.widget_type.value,
            "widgetName": self.name,
            "properties": self.properties,
            "position": self.position,
            "dataSource": self.data_source
        }


@dataclass
class Dashboard:
    """An Appsmith dashboard"""
    dashboard_id: str
    name: str
    description: str
    status: DashboardStatus
    widgets: List[Widget] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    layout_json: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dashboardId": self.dashboard_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "widgets": [w.to_dict() for w in self.widgets],
            "createdAt": self.created_at.isoformat(),
            "updatedAt": self.updated_at.isoformat(),
            "widgetCount": len(self.widgets)
        }
    
    def add_widget(self, widget: Widget):
        """Add a widget to the dashboard"""
        self.widgets.append(widget)
        self.updated_at = datetime.now()


# ============================================================
# Widget Factory
# ============================================================

class WidgetFactory:
    """Factory for creating dashboard widgets"""
    
    @staticmethod
    def create_stat_box(
        name: str,
        value_binding: str,
        label: str,
        position: Dict[str, int]
    ) -> Widget:
        """Create a stat box widget"""
        return Widget(
            widget_id=f"stat_{uuid.uuid4().hex[:8]}",
            widget_type=WidgetType.STAT_BOX,
            name=name,
            properties={
                "label": label,
                "value": f"{{{{ {value_binding} }}}}",
                "backgroundColor": "#3B82F6",
                "textColor": "#FFFFFF"
            },
            position=position,
            data_source=value_binding
        )
    
    @staticmethod
    def create_table(
        name: str,
        data_binding: str,
        columns: List[str],
        position: Dict[str, int]
    ) -> Widget:
        """Create a table widget"""
        column_config = {col: {"label": col.title()} for col in columns}
        
        return Widget(
            widget_id=f"table_{uuid.uuid4().hex[:8]}",
            widget_type=WidgetType.TABLE,
            name=name,
            properties={
                "tableData": f"{{{{ {data_binding} }}}}",
                "columns": column_config,
                "pageSize": 10,
                "searchEnabled": True,
                "sortEnabled": True
            },
            position=position,
            data_source=data_binding
        )
    
    @staticmethod
    def create_chart(
        name: str,
        data_binding: str,
        chart_type: str,
        x_axis: str,
        y_axis: str,
        position: Dict[str, int]
    ) -> Widget:
        """Create a chart widget"""
        return Widget(
            widget_id=f"chart_{uuid.uuid4().hex[:8]}",
            widget_type=WidgetType.CHART,
            name=name,
            properties={
                "chartType": chart_type,
                "chartData": f"{{{{ {data_binding} }}}}",
                "xAxisName": x_axis,
                "yAxisName": y_axis,
                "showLegend": True
            },
            position=position,
            data_source=data_binding
        )
    
    @staticmethod
    def create_text(
        name: str,
        text: str,
        position: Dict[str, int],
        font_size: str = "1rem"
    ) -> Widget:
        """Create a text widget"""
        return Widget(
            widget_id=f"text_{uuid.uuid4().hex[:8]}",
            widget_type=WidgetType.TEXT,
            name=name,
            properties={
                "text": text,
                "fontSize": font_size,
                "fontWeight": "normal",
                "textAlign": "left"
            },
            position=position
        )
    
    @staticmethod
    def create_button(
        name: str,
        label: str,
        action: str,
        position: Dict[str, int]
    ) -> Widget:
        """Create a button widget"""
        return Widget(
            widget_id=f"btn_{uuid.uuid4().hex[:8]}",
            widget_type=WidgetType.BUTTON,
            name=name,
            properties={
                "buttonLabel": label,
                "onClick": action,
                "buttonColor": "#3B82F6",
                "buttonVariant": "PRIMARY"
            },
            position=position
        )
    
    @staticmethod
    def create_container(
        name: str,
        position: Dict[str, int],
        background_color: str = "#F8FAFC"
    ) -> Widget:
        """Create a container widget"""
        return Widget(
            widget_id=f"container_{uuid.uuid4().hex[:8]}",
            widget_type=WidgetType.CONTAINER,
            name=name,
            properties={
                "backgroundColor": background_color,
                "borderRadius": "8px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)"
            },
            position=position
        )


# ============================================================
# Dashboard Templates
# ============================================================

class DashboardTemplates:
    """Pre-built dashboard templates"""
    
    @staticmethod
    def workflow_status_dashboard() -> Dashboard:
        """Create a workflow status monitoring dashboard"""
        dashboard = Dashboard(
            dashboard_id=f"dash_{uuid.uuid4().hex[:8]}",
            name="Workflow Status Dashboard",
            description="Monitor workflow executions and system status",
            status=DashboardStatus.DRAFT
        )
        
        factory = WidgetFactory()
        
        # Header
        dashboard.add_widget(factory.create_text(
            name="Header",
            text="# Workflow Status Dashboard",
            position={"x": 0, "y": 0, "width": 12, "height": 1},
            font_size="1.5rem"
        ))
        
        # Stat boxes row
        dashboard.add_widget(factory.create_stat_box(
            name="TotalWorkflows",
            value_binding="stats.totalWorkflows",
            label="Total Workflows",
            position={"x": 0, "y": 1, "width": 3, "height": 2}
        ))
        
        dashboard.add_widget(factory.create_stat_box(
            name="RunningWorkflows",
            value_binding="stats.runningWorkflows",
            label="Running",
            position={"x": 3, "y": 1, "width": 3, "height": 2}
        ))
        
        dashboard.add_widget(factory.create_stat_box(
            name="CompletedWorkflows",
            value_binding="stats.completedWorkflows",
            label="Completed",
            position={"x": 6, "y": 1, "width": 3, "height": 2}
        ))
        
        dashboard.add_widget(factory.create_stat_box(
            name="FailedWorkflows",
            value_binding="stats.failedWorkflows",
            label="Failed",
            position={"x": 9, "y": 1, "width": 3, "height": 2}
        ))
        
        # Execution table
        dashboard.add_widget(factory.create_table(
            name="ExecutionTable",
            data_binding="executions",
            columns=["id", "workflow", "status", "startedAt", "duration"],
            position={"x": 0, "y": 3, "width": 8, "height": 6}
        ))
        
        # Execution chart
        dashboard.add_widget(factory.create_chart(
            name="ExecutionChart",
            data_binding="executionStats",
            chart_type="LINE",
            x_axis="Time",
            y_axis="Count",
            position={"x": 8, "y": 3, "width": 4, "height": 6}
        ))
        
        return dashboard
    
    @staticmethod
    def container_logs_dashboard() -> Dashboard:
        """Create a container logs dashboard"""
        dashboard = Dashboard(
            dashboard_id=f"dash_{uuid.uuid4().hex[:8]}",
            name="Container Logs Dashboard",
            description="View Docker container logs and status",
            status=DashboardStatus.DRAFT
        )
        
        factory = WidgetFactory()
        
        # Header
        dashboard.add_widget(factory.create_text(
            name="Header",
            text="# Container Logs",
            position={"x": 0, "y": 0, "width": 12, "height": 1}
        ))
        
        # Container list
        dashboard.add_widget(factory.create_table(
            name="ContainerList",
            data_binding="containers",
            columns=["name", "image", "status", "ports", "created"],
            position={"x": 0, "y": 1, "width": 6, "height": 5}
        ))
        
        # Logs viewer
        dashboard.add_widget(factory.create_text(
            name="LogViewer",
            text="{{{{ selectedContainer.logs }}}}",
            position={"x": 6, "y": 1, "width": 6, "height": 8}
        ))
        
        # Refresh button
        dashboard.add_widget(factory.create_button(
            name="RefreshBtn",
            label="Refresh Logs",
            action="{{fetchContainerLogs()}}",
            position={"x": 6, "y": 9, "width": 2, "height": 1}
        ))
        
        return dashboard
    
    @staticmethod
    def agent_decisions_dashboard() -> Dashboard:
        """Create an agent decisions dashboard"""
        dashboard = Dashboard(
            dashboard_id=f"dash_{uuid.uuid4().hex[:8]}",
            name="Agent Decisions Dashboard",
            description="Track AI agent decision history",
            status=DashboardStatus.DRAFT
        )
        
        factory = WidgetFactory()
        
        # Header
        dashboard.add_widget(factory.create_text(
            name="Header",
            text="# Agent Decision History",
            position={"x": 0, "y": 0, "width": 12, "height": 1}
        ))
        
        # Decision breakdown chart
        dashboard.add_widget(factory.create_chart(
            name="DecisionChart",
            data_binding="decisionStats",
            chart_type="PIE",
            x_axis="Type",
            y_axis="Count",
            position={"x": 0, "y": 1, "width": 4, "height": 4}
        ))
        
        # Decision history table
        dashboard.add_widget(factory.create_table(
            name="DecisionTable",
            data_binding="decisions",
            columns=["timestamp", "input", "decision", "confidence", "tool"],
            position={"x": 4, "y": 1, "width": 8, "height": 8}
        ))
        
        return dashboard


# ============================================================
# Appsmith Client
# ============================================================

class AppsmithClient:
    """
    Client for interacting with Appsmith API.
    Manages dashboards, widgets, and data sources.
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.dashboards: Dict[str, Dashboard] = {}
        self._connected = False
        
    async def connect(self) -> bool:
        """Test connection to Appsmith"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/health",
                    timeout=5.0
                )
                self._connected = response.status_code == 200
                return self._connected
        except Exception as e:
            logger.warning(f"Appsmith connection failed: {e}")
            self._connected = False
            return False
    
    def is_connected(self) -> bool:
        """Check if connected to Appsmith"""
        return self._connected
    
    def create_dashboard(
        self,
        name: str,
        description: str = "",
        template: Optional[str] = None
    ) -> Dashboard:
        """
        Create a new dashboard.
        
        Args:
            name: Dashboard name
            description: Dashboard description
            template: Optional template name (workflow_status, container_logs, agent_decisions)
        """
        if template:
            if template == "workflow_status":
                dashboard = DashboardTemplates.workflow_status_dashboard()
            elif template == "container_logs":
                dashboard = DashboardTemplates.container_logs_dashboard()
            elif template == "agent_decisions":
                dashboard = DashboardTemplates.agent_decisions_dashboard()
            else:
                raise ValueError(f"Unknown template: {template}")
            
            dashboard.name = name
            dashboard.description = description
        else:
            dashboard = Dashboard(
                dashboard_id=f"dash_{uuid.uuid4().hex[:8]}",
                name=name,
                description=description,
                status=DashboardStatus.DRAFT
            )
        
        self.dashboards[dashboard.dashboard_id] = dashboard
        logger.info(f"Dashboard created: {dashboard.dashboard_id}")
        return dashboard
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID"""
        return self.dashboards.get(dashboard_id)
    
    def list_dashboards(self) -> List[Dashboard]:
        """List all dashboards"""
        return list(self.dashboards.values())
    
    def delete_dashboard(self, dashboard_id: str) -> bool:
        """Delete a dashboard"""
        if dashboard_id in self.dashboards:
            del self.dashboards[dashboard_id]
            return True
        return False
    
    def add_widget(
        self,
        dashboard_id: str,
        widget_type: WidgetType,
        name: str,
        properties: Dict[str, Any],
        position: Dict[str, int]
    ) -> Optional[Widget]:
        """Add a widget to a dashboard"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return None
        
        widget = Widget(
            widget_id=f"widget_{uuid.uuid4().hex[:8]}",
            widget_type=widget_type,
            name=name,
            properties=properties,
            position=position
        )
        
        dashboard.add_widget(widget)
        return widget
    
    def publish_dashboard(self, dashboard_id: str) -> bool:
        """Publish a dashboard"""
        dashboard = self.dashboards.get(dashboard_id)
        if dashboard:
            dashboard.status = DashboardStatus.PUBLISHED
            dashboard.updated_at = datetime.now()
            return True
        return False
    
    def export_dashboard(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Export dashboard as JSON for Appsmith import"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return None
        
        # Generate Appsmith-compatible export format
        export = {
            "exportedApplication": {
                "name": dashboard.name,
                "isPublic": False,
                "pages": [
                    {
                        "name": "Main",
                        "slug": "main",
                        "layouts": [
                            {
                                "dsl": {
                                    "widgetName": "MainContainer",
                                    "backgroundColor": "#FAFAFA",
                                    "bottomRow": 100,
                                    "rightColumn": 64,
                                    "children": [
                                        w.to_dict() for w in dashboard.widgets
                                    ]
                                }
                            }
                        ]
                    }
                ]
            },
            "datasourceList": [],
            "customJSLibs": [],
            "exportedAt": datetime.now().isoformat()
        }
        
        return export
    
    def import_layout_json(self, dashboard_id: str, layout_json: Dict[str, Any]) -> bool:
        """Import a layout JSON (from Playwright analyzer) into dashboard"""
        dashboard = self.dashboards.get(dashboard_id)
        if not dashboard:
            return False
        
        dashboard.layout_json = layout_json
        dashboard.updated_at = datetime.now()
        
        # Convert layout elements to widgets
        if "elements" in layout_json:
            for element in layout_json["elements"]:
                widget_type = self._map_element_to_widget(element.get("tag", "div"))
                
                widget = Widget(
                    widget_id=f"imported_{uuid.uuid4().hex[:8]}",
                    widget_type=widget_type,
                    name=element.get("id", "unnamed"),
                    properties=element.get("styles", {}),
                    position={
                        "x": element.get("bounds", {}).get("x", 0),
                        "y": element.get("bounds", {}).get("y", 0),
                        "width": element.get("bounds", {}).get("width", 100),
                        "height": element.get("bounds", {}).get("height", 50)
                    }
                )
                dashboard.add_widget(widget)
        
        return True
    
    def _map_element_to_widget(self, tag: str) -> WidgetType:
        """Map HTML tag to Appsmith widget type"""
        mapping = {
            "table": WidgetType.TABLE,
            "button": WidgetType.BUTTON,
            "input": WidgetType.INPUT,
            "img": WidgetType.IMAGE,
            "ul": WidgetType.LIST,
            "ol": WidgetType.LIST,
            "h1": WidgetType.TEXT,
            "h2": WidgetType.TEXT,
            "p": WidgetType.TEXT,
            "span": WidgetType.TEXT,
            "div": WidgetType.CONTAINER
        }
        return mapping.get(tag.lower(), WidgetType.CONTAINER)
    
    def get_templates(self) -> List[str]:
        """Get available dashboard templates"""
        return ["workflow_status", "container_logs", "agent_decisions"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        published = len([d for d in self.dashboards.values() 
                        if d.status == DashboardStatus.PUBLISHED])
        draft = len([d for d in self.dashboards.values() 
                    if d.status == DashboardStatus.DRAFT])
        
        total_widgets = sum(len(d.widgets) for d in self.dashboards.values())
        
        return {
            "connected": self._connected,
            "total_dashboards": len(self.dashboards),
            "published_dashboards": published,
            "draft_dashboards": draft,
            "total_widgets": total_widgets,
            "available_templates": self.get_templates()
        }


# ============================================================
# Singleton Instance
# ============================================================

_appsmith_client: Optional[AppsmithClient] = None


def get_appsmith_client(
    base_url: str = "http://localhost:8080",
    api_key: Optional[str] = None
) -> AppsmithClient:
    """Get or create the Appsmith client singleton"""
    global _appsmith_client
    if _appsmith_client is None:
        _appsmith_client = AppsmithClient(base_url, api_key)
    return _appsmith_client
