"""
Milestone 3 Integration Tests (CORRECT VERSION)
Tests for: Appsmith Dashboard, Playwright Analyzer, Layout Converter, Directus
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
# Test: Appsmith Client
# ============================================================

class TestAppsmithImports:
    """Test Appsmith module imports"""
    
    def test_import_appsmith_client(self):
        """Test appsmith_client imports"""
        from agent.appsmith_client import (
            AppsmithClient,
            Dashboard,
            Widget,
            WidgetFactory,
            DashboardTemplates,
            get_appsmith_client
        )
        assert AppsmithClient is not None
        assert Dashboard is not None
        print_result("appsmith_client imports", True)
    
    def test_import_enums(self):
        """Test enum imports"""
        from agent.appsmith_client import WidgetType, DashboardStatus
        assert len(WidgetType) == 9
        assert len(DashboardStatus) == 3
        print_result("Appsmith enums", True)


class TestWidgetFactory:
    """Test widget creation"""
    
    def test_create_stat_box(self):
        """Test stat box widget creation"""
        from agent.appsmith_client import WidgetFactory, WidgetType
        
        widget = WidgetFactory.create_stat_box(
            name="TotalUsers",
            value_binding="stats.users",
            label="Total Users",
            position={"x": 0, "y": 0, "width": 3, "height": 2}
        )
        
        assert widget.widget_type == WidgetType.STAT_BOX
        assert widget.name == "TotalUsers"
        assert "label" in widget.properties
        print_result("Create stat box", True)
    
    def test_create_table(self):
        """Test table widget creation"""
        from agent.appsmith_client import WidgetFactory, WidgetType
        
        widget = WidgetFactory.create_table(
            name="DataTable",
            data_binding="tableData",
            columns=["id", "name", "status"],
            position={"x": 0, "y": 0, "width": 8, "height": 6}
        )
        
        assert widget.widget_type == WidgetType.TABLE
        assert "tableData" in widget.properties
        print_result("Create table", True)
    
    def test_create_chart(self):
        """Test chart widget creation"""
        from agent.appsmith_client import WidgetFactory, WidgetType
        
        widget = WidgetFactory.create_chart(
            name="StatsChart",
            data_binding="chartData",
            chart_type="LINE",
            x_axis="Date",
            y_axis="Count",
            position={"x": 0, "y": 0, "width": 4, "height": 4}
        )
        
        assert widget.widget_type == WidgetType.CHART
        assert widget.properties["chartType"] == "LINE"
        print_result("Create chart", True)
    
    def test_create_button(self):
        """Test button widget creation"""
        from agent.appsmith_client import WidgetFactory, WidgetType
        
        widget = WidgetFactory.create_button(
            name="RefreshBtn",
            label="Refresh",
            action="{{refresh()}}",
            position={"x": 0, "y": 0, "width": 2, "height": 1}
        )
        
        assert widget.widget_type == WidgetType.BUTTON
        assert widget.properties["buttonLabel"] == "Refresh"
        print_result("Create button", True)


class TestDashboardTemplates:
    """Test dashboard templates"""
    
    def test_workflow_status_template(self):
        """Test workflow status dashboard template"""
        from agent.appsmith_client import DashboardTemplates
        
        dashboard = DashboardTemplates.workflow_status_dashboard()
        
        assert dashboard.name == "Workflow Status Dashboard"
        assert len(dashboard.widgets) > 5
        print_result("Workflow status template", True, f"Widgets: {len(dashboard.widgets)}")
    
    def test_container_logs_template(self):
        """Test container logs dashboard template"""
        from agent.appsmith_client import DashboardTemplates
        
        dashboard = DashboardTemplates.container_logs_dashboard()
        
        assert dashboard.name == "Container Logs Dashboard"
        assert len(dashboard.widgets) > 3
        print_result("Container logs template", True)
    
    def test_agent_decisions_template(self):
        """Test agent decisions dashboard template"""
        from agent.appsmith_client import DashboardTemplates
        
        dashboard = DashboardTemplates.agent_decisions_dashboard()
        
        assert dashboard.name == "Agent Decisions Dashboard"
        print_result("Agent decisions template", True)


class TestAppsmithClient:
    """Test AppsmithClient"""
    
    def test_client_creation(self):
        """Test client instantiation"""
        from agent.appsmith_client import AppsmithClient
        
        client = AppsmithClient()
        assert client is not None
        assert client.dashboards == {}
        print_result("AppsmithClient creation", True)
    
    def test_create_dashboard(self):
        """Test dashboard creation"""
        from agent.appsmith_client import AppsmithClient, DashboardStatus
        
        client = AppsmithClient()
        dashboard = client.create_dashboard(
            name="Test Dashboard",
            description="A test dashboard"
        )
        
        assert dashboard.dashboard_id is not None
        assert dashboard.status == DashboardStatus.DRAFT
        print_result("Create dashboard", True)
    
    def test_create_dashboard_from_template(self):
        """Test dashboard creation from template"""
        from agent.appsmith_client import AppsmithClient
        
        client = AppsmithClient()
        dashboard = client.create_dashboard(
            name="My Workflow Dashboard",
            template="workflow_status"
        )
        
        assert len(dashboard.widgets) > 5
        print_result("Create from template", True, f"Widgets: {len(dashboard.widgets)}")
    
    def test_add_widget(self):
        """Test adding widget to dashboard"""
        from agent.appsmith_client import AppsmithClient, WidgetType
        
        client = AppsmithClient()
        dashboard = client.create_dashboard("Test")
        
        widget = client.add_widget(
            dashboard.dashboard_id,
            WidgetType.TEXT,
            "Title",
            {"text": "Hello"},
            {"x": 0, "y": 0, "width": 4, "height": 1}
        )
        
        assert widget is not None
        assert len(dashboard.widgets) == 1
        print_result("Add widget", True)
    
    def test_publish_dashboard(self):
        """Test publishing dashboard"""
        from agent.appsmith_client import AppsmithClient, DashboardStatus
        
        client = AppsmithClient()
        dashboard = client.create_dashboard("Test")
        
        assert dashboard.status == DashboardStatus.DRAFT
        
        client.publish_dashboard(dashboard.dashboard_id)
        
        assert dashboard.status == DashboardStatus.PUBLISHED
        print_result("Publish dashboard", True)
    
    def test_export_dashboard(self):
        """Test dashboard export"""
        from agent.appsmith_client import AppsmithClient
        
        client = AppsmithClient()
        dashboard = client.create_dashboard("Export Test", template="workflow_status")
        
        export = client.export_dashboard(dashboard.dashboard_id)
        
        assert export is not None
        assert "exportedApplication" in export
        assert "pages" in export["exportedApplication"]
        print_result("Export dashboard", True)
    
    def test_get_templates(self):
        """Test getting available templates"""
        from agent.appsmith_client import AppsmithClient
        
        client = AppsmithClient()
        templates = client.get_templates()
        
        assert len(templates) == 3
        assert "workflow_status" in templates
        print_result("Get templates", True, f"Templates: {templates}")
    
    def test_get_stats(self):
        """Test client statistics"""
        from agent.appsmith_client import AppsmithClient
        
        client = AppsmithClient()
        client.create_dashboard("Test 1")
        client.create_dashboard("Test 2")
        
        stats = client.get_stats()
        
        assert stats["total_dashboards"] == 2
        print_result("Get stats", True, f"Dashboards: {stats['total_dashboards']}")


# ============================================================
# Test: Playwright Analyzer
# ============================================================

class TestPlaywrightImports:
    """Test Playwright module imports"""
    
    def test_import_playwright_analyzer(self):
        """Test playwright_analyzer imports"""
        from agent.playwright_analyzer import (
            PlaywrightAnalyzer,
            LayoutConverter,
            DOMElement,
            PageAnalysis,
            get_playwright_analyzer
        )
        assert PlaywrightAnalyzer is not None
        assert LayoutConverter is not None
        print_result("playwright_analyzer imports", True)
    
    def test_import_enums(self):
        """Test enum imports"""
        from agent.playwright_analyzer import ElementType, LayoutType
        assert len(ElementType) == 13
        assert len(LayoutType) == 8
        print_result("Playwright enums", True)


class TestLayoutConverter:
    """Test Layout → JSON converter"""
    
    def test_converter_creation(self):
        """Test LayoutConverter instantiation"""
        from agent.playwright_analyzer import LayoutConverter
        
        converter = LayoutConverter()
        assert converter is not None
        print_result("LayoutConverter creation", True)
    
    def test_elements_to_json(self):
        """Test converting elements to JSON"""
        from agent.playwright_analyzer import (
            LayoutConverter, DOMElement, ElementType, BoundingBox
        )
        
        converter = LayoutConverter()
        
        elements = [
            DOMElement(
                element_id="test_1",
                tag="div",
                element_type=ElementType.CONTAINER,
                bounds=BoundingBox(0, 0, 100, 50),
                text_content="Test content"
            )
        ]
        
        json_layout = converter.elements_to_json(elements)
        
        assert "elements" in json_layout
        assert "stats" in json_layout
        assert json_layout["stats"]["totalElements"] == 1
        print_result("Elements to JSON", True)
    
    def test_json_to_appsmith(self):
        """Test converting JSON to Appsmith format"""
        from agent.playwright_analyzer import LayoutConverter
        
        converter = LayoutConverter()
        
        layout_json = {
            "elements": [
                {
                    "id": "btn_1",
                    "tag": "button",
                    "type": "button",
                    "bounds": {"x": 10, "y": 10, "width": 100, "height": 40},
                    "textContent": "Click Me"
                }
            ]
        }
        
        appsmith = converter.json_to_appsmith(layout_json)
        
        assert "widgetName" in appsmith
        assert "children" in appsmith
        assert len(appsmith["children"]) == 1
        print_result("JSON to Appsmith", True)
    
    def test_flatten_elements(self):
        """Test flattening nested elements"""
        from agent.playwright_analyzer import (
            LayoutConverter, DOMElement, ElementType, BoundingBox
        )
        
        converter = LayoutConverter()
        
        child = DOMElement(
            element_id="child_1",
            tag="p",
            element_type=ElementType.TEXT,
            bounds=BoundingBox(10, 10, 80, 30)
        )
        
        parent = DOMElement(
            element_id="parent_1",
            tag="div",
            element_type=ElementType.CONTAINER,
            bounds=BoundingBox(0, 0, 100, 50),
            children=[child]
        )
        
        json_layout = converter.elements_to_json([parent], flatten=True)
        
        assert json_layout["stats"]["totalElements"] == 2
        print_result("Flatten elements", True)


class TestPlaywrightAnalyzer:
    """Test PlaywrightAnalyzer"""
    
    def test_analyzer_creation(self):
        """Test analyzer instantiation"""
        from agent.playwright_analyzer import PlaywrightAnalyzer
        
        analyzer = PlaywrightAnalyzer()
        assert analyzer is not None
        assert analyzer.converter is not None
        print_result("PlaywrightAnalyzer creation", True)
    
    @pytest.mark.asyncio
    async def test_mock_analysis(self):
        """Test mock page analysis (without Playwright)"""
        from agent.playwright_analyzer import PlaywrightAnalyzer
        
        analyzer = PlaywrightAnalyzer()
        
        # This will use mock analysis since Playwright may not be installed
        analysis = await analyzer.analyze_page("https://example.com")
        
        assert analysis is not None
        assert analysis.url == "https://example.com"
        assert len(analysis.elements) > 0
        print_result("Mock analysis", True, f"Elements: {len(analysis.elements)}")
    
    def test_convert_to_json(self):
        """Test converting analysis to JSON"""
        from agent.playwright_analyzer import (
            PlaywrightAnalyzer, PageAnalysis, DOMElement, 
            ElementType, BoundingBox, LayoutType
        )
        
        analyzer = PlaywrightAnalyzer()
        
        analysis = PageAnalysis(
            url="https://test.com",
            title="Test Page",
            layout_type=LayoutType.SINGLE_COLUMN,
            elements=[
                DOMElement(
                    element_id="main",
                    tag="main",
                    element_type=ElementType.CONTAINER,
                    bounds=BoundingBox(0, 0, 1920, 1080)
                )
            ],
            viewport={"width": 1920, "height": 1080}
        )
        
        json_layout = analyzer.convert_to_json(analysis)
        
        assert "elements" in json_layout
        print_result("Convert to JSON", True)
    
    def test_convert_to_appsmith(self):
        """Test converting analysis to Appsmith format"""
        from agent.playwright_analyzer import (
            PlaywrightAnalyzer, PageAnalysis, DOMElement,
            ElementType, BoundingBox, LayoutType
        )
        
        analyzer = PlaywrightAnalyzer()
        
        analysis = PageAnalysis(
            url="https://test.com",
            title="Test Page",
            layout_type=LayoutType.SINGLE_COLUMN,
            elements=[
                DOMElement(
                    element_id="btn",
                    tag="button",
                    element_type=ElementType.BUTTON,
                    bounds=BoundingBox(10, 10, 100, 40),
                    text_content="Click"
                )
            ],
            viewport={"width": 1920, "height": 1080}
        )
        
        appsmith = analyzer.convert_to_appsmith(analysis)
        
        assert "widgetName" in appsmith
        assert appsmith["type"] == "CANVAS_WIDGET"
        print_result("Convert to Appsmith", True)
    
    def test_get_stats(self):
        """Test analyzer statistics"""
        from agent.playwright_analyzer import PlaywrightAnalyzer
        
        analyzer = PlaywrightAnalyzer()
        stats = analyzer.get_stats()
        
        assert "total_analyses" in stats
        assert "conversions" in stats
        print_result("Analyzer stats", True)


# ============================================================
# Test: Directus Client
# ============================================================

class TestDirectusImports:
    """Test Directus module imports"""
    
    def test_import_directus_client(self):
        """Test directus_client imports"""
        from agent.directus_client import (
            DirectusClient,
            UserManager,
            AuthManager,
            StorageManager,
            get_directus_client
        )
        assert DirectusClient is not None
        assert UserManager is not None
        print_result("directus_client imports", True)
    
    def test_import_enums(self):
        """Test enum imports"""
        from agent.directus_client import UserRole, CollectionType
        assert len(UserRole) == 3
        assert len(CollectionType) == 5
        print_result("Directus enums", True)


class TestUserManager:
    """Test user management"""
    
    def test_user_manager_creation(self):
        """Test UserManager instantiation"""
        from agent.directus_client import UserManager
        
        manager = UserManager()
        assert manager is not None
        # Should have default admin
        assert len(manager.users) >= 1
        print_result("UserManager creation", True)
    
    def test_create_user(self):
        """Test user creation"""
        from agent.directus_client import UserManager, UserRole
        
        manager = UserManager()
        user = manager.create_user(
            email="test@example.com",
            username="testuser",
            password="password123",
            role=UserRole.EDITOR
        )
        
        assert user.user_id is not None
        assert user.email == "test@example.com"
        assert user.role == UserRole.EDITOR
        print_result("Create user", True)
    
    def test_authenticate_user(self):
        """Test user authentication"""
        from agent.directus_client import UserManager
        
        manager = UserManager()
        manager.create_user("auth@test.com", "authuser", "secret123")
        
        # Valid credentials
        user = manager.authenticate("auth@test.com", "secret123")
        assert user is not None
        
        # Invalid password
        user = manager.authenticate("auth@test.com", "wrong")
        assert user is None
        
        print_result("Authenticate user", True)
    
    def test_list_users(self):
        """Test listing users"""
        from agent.directus_client import UserManager, UserRole
        
        manager = UserManager()
        manager.create_user("user1@test.com", "user1", "pass")
        manager.create_user("user2@test.com", "user2", "pass")
        
        users = manager.list_users()
        assert len(users) >= 3  # Including default admin
        print_result("List users", True, f"Users: {len(users)}")


class TestAuthManager:
    """Test authentication"""
    
    def test_login(self):
        """Test user login"""
        from agent.directus_client import UserManager, AuthManager
        
        user_mgr = UserManager()
        auth_mgr = AuthManager(user_mgr)
        
        # Login as default admin
        token = auth_mgr.login("admin@localhost", "admin123")
        
        assert token is not None
        assert token.token is not None
        assert token.is_valid()
        print_result("Login", True)
    
    def test_validate_token(self):
        """Test token validation"""
        from agent.directus_client import UserManager, AuthManager
        
        user_mgr = UserManager()
        auth_mgr = AuthManager(user_mgr)
        
        token = auth_mgr.login("admin@localhost", "admin123")
        user = auth_mgr.validate_token(token.token)
        
        assert user is not None
        assert user.email == "admin@localhost"
        print_result("Validate token", True)
    
    def test_logout(self):
        """Test user logout"""
        from agent.directus_client import UserManager, AuthManager
        
        user_mgr = UserManager()
        auth_mgr = AuthManager(user_mgr)
        
        token = auth_mgr.login("admin@localhost", "admin123")
        result = auth_mgr.logout(token.token)
        
        assert result == True
        
        # Token should now be invalid
        user = auth_mgr.validate_token(token.token)
        assert user is None
        print_result("Logout", True)
    
    def test_refresh_token(self):
        """Test token refresh"""
        from agent.directus_client import UserManager, AuthManager
        
        user_mgr = UserManager()
        auth_mgr = AuthManager(user_mgr)
        
        token = auth_mgr.login("admin@localhost", "admin123")
        new_token = auth_mgr.refresh(token.refresh_token)
        
        assert new_token is not None
        assert new_token.token != token.token
        print_result("Refresh token", True)


class TestStorageManager:
    """Test storage management"""
    
    def test_create_item(self):
        """Test creating storage item"""
        from agent.directus_client import StorageManager, CollectionType
        
        storage = StorageManager()
        item = storage.create_item(
            CollectionType.DASHBOARD_LAYOUTS,
            {"name": "Test Layout", "data": {"widgets": []}},
            "user_001"
        )
        
        assert item.item_id is not None
        assert item.collection == CollectionType.DASHBOARD_LAYOUTS
        print_result("Create item", True)
    
    def test_get_item(self):
        """Test getting storage item"""
        from agent.directus_client import StorageManager, CollectionType
        
        storage = StorageManager()
        created = storage.create_item(
            CollectionType.PROJECT_DATA,
            {"project": "test"},
            "user_001"
        )
        
        retrieved = storage.get_item(CollectionType.PROJECT_DATA, created.item_id)
        
        assert retrieved is not None
        assert retrieved.item_id == created.item_id
        print_result("Get item", True)
    
    def test_list_items(self):
        """Test listing items"""
        from agent.directus_client import StorageManager, CollectionType
        
        storage = StorageManager()
        storage.create_item(CollectionType.WORKFLOW_METADATA, {"wf": 1}, "user_001")
        storage.create_item(CollectionType.WORKFLOW_METADATA, {"wf": 2}, "user_001")
        storage.create_item(CollectionType.WORKFLOW_METADATA, {"wf": 3}, "user_002")
        
        all_items = storage.list_items(CollectionType.WORKFLOW_METADATA)
        assert len(all_items) == 3
        
        user_items = storage.list_items(CollectionType.WORKFLOW_METADATA, "user_001")
        assert len(user_items) == 2
        
        print_result("List items", True)
    
    def test_delete_item(self):
        """Test deleting item"""
        from agent.directus_client import StorageManager, CollectionType
        
        storage = StorageManager()
        item = storage.create_item(
            CollectionType.ANALYSIS_RESULTS,
            {"url": "https://test.com"},
            "user_001"
        )
        
        result = storage.delete_item(CollectionType.ANALYSIS_RESULTS, item.item_id)
        assert result == True
        
        retrieved = storage.get_item(CollectionType.ANALYSIS_RESULTS, item.item_id)
        assert retrieved is None
        print_result("Delete item", True)


class TestDirectusClient:
    """Test DirectusClient"""
    
    def test_client_creation(self):
        """Test client instantiation"""
        from agent.directus_client import DirectusClient
        
        client = DirectusClient()
        assert client is not None
        print_result("DirectusClient creation", True)
    
    def test_login_logout(self):
        """Test login and logout"""
        from agent.directus_client import DirectusClient
        
        client = DirectusClient()
        
        token = client.login("admin@localhost", "admin123")
        assert token is not None
        
        user = client.get_current_user()
        assert user is not None
        
        client.logout()
        user = client.get_current_user()
        assert user is None
        print_result("Login/logout", True)
    
    def test_save_dashboard_layout(self):
        """Test saving dashboard layout"""
        from agent.directus_client import DirectusClient
        
        client = DirectusClient()
        client.login("admin@localhost", "admin123")
        
        layout = {"widgets": [{"type": "text", "content": "Hello"}]}
        item = client.save_dashboard_layout("My Layout", layout)
        
        assert item is not None
        assert item.data["name"] == "My Layout"
        print_result("Save dashboard layout", True)
    
    def test_get_dashboard_layout(self):
        """Test getting dashboard layout"""
        from agent.directus_client import DirectusClient
        
        client = DirectusClient()
        client.login("admin@localhost", "admin123")
        
        layout = {"widgets": [{"type": "chart"}]}
        item = client.save_dashboard_layout("Test Layout", layout)
        
        retrieved = client.get_dashboard_layout(item.item_id)
        assert retrieved is not None
        assert retrieved["widgets"][0]["type"] == "chart"
        print_result("Get dashboard layout", True)
    
    def test_save_analysis_result(self):
        """Test saving analysis result"""
        from agent.directus_client import DirectusClient
        
        client = DirectusClient()
        
        analysis = {"elements": [], "layoutType": "single_column"}
        item = client.save_analysis_result("https://example.com", analysis)
        
        assert item is not None
        assert item.data["url"] == "https://example.com"
        print_result("Save analysis result", True)
    
    def test_get_collections(self):
        """Test getting available collections"""
        from agent.directus_client import DirectusClient
        
        client = DirectusClient()
        collections = client.get_collections()
        
        assert len(collections) == 5
        assert "dashboard_layouts" in collections
        print_result("Get collections", True, f"Collections: {collections}")
    
    def test_get_stats(self):
        """Test client statistics"""
        from agent.directus_client import DirectusClient
        
        client = DirectusClient()
        stats = client.get_stats()
        
        assert "users" in stats
        assert "storage" in stats
        assert stats["users"]["total"] >= 1
        print_result("Get stats", True)


# ============================================================
# Test: Integration
# ============================================================

class TestM3Integration:
    """Test M3 components working together"""
    
    @pytest.mark.asyncio
    async def test_analyze_and_store(self):
        """Test analyzing page and storing in Directus"""
        from agent.playwright_analyzer import PlaywrightAnalyzer
        from agent.directus_client import DirectusClient
        
        analyzer = PlaywrightAnalyzer()
        client = DirectusClient()
        
        # Analyze page (mock)
        analysis = await analyzer.analyze_page("https://test.com")
        json_layout = analyzer.convert_to_json(analysis)
        
        # Store in Directus
        item = client.save_analysis_result("https://test.com", json_layout)
        
        assert item is not None
        print_result("Analyze and store", True)
    
    @pytest.mark.asyncio
    async def test_analyze_and_create_dashboard(self):
        """Test analyzing page and creating Appsmith dashboard"""
        from agent.playwright_analyzer import PlaywrightAnalyzer
        from agent.appsmith_client import AppsmithClient
        
        analyzer = PlaywrightAnalyzer()
        appsmith = AppsmithClient()
        
        # Analyze page (mock)
        analysis = await analyzer.analyze_page("https://test.com")
        
        # Create dashboard
        dashboard = appsmith.create_dashboard("Analysis Dashboard")
        
        # Import layout
        json_layout = analyzer.convert_to_json(analysis)
        appsmith.import_layout_json(dashboard.dashboard_id, json_layout)
        
        assert len(dashboard.widgets) > 0
        print_result("Analyze and create dashboard", True)
    
    def test_full_workflow(self):
        """Test full M3 workflow"""
        from agent.appsmith_client import AppsmithClient
        from agent.directus_client import DirectusClient
        from agent.playwright_analyzer import LayoutConverter
        
        # Setup
        appsmith = AppsmithClient()
        directus = DirectusClient()
        converter = LayoutConverter()
        
        # Create dashboard from template
        dashboard = appsmith.create_dashboard(
            "Workflow Monitor",
            template="workflow_status"
        )
        
        # Export and store
        export = appsmith.export_dashboard(dashboard.dashboard_id)
        directus.save_dashboard_layout("Workflow Monitor", export)
        
        # Get stats
        appsmith_stats = appsmith.get_stats()
        directus_stats = directus.get_stats()
        
        assert appsmith_stats["total_dashboards"] >= 1
        assert directus_stats["storage"]["totalItems"] >= 1
        print_result("Full workflow", True)


# ============================================================
# Run All Tests
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 MILESTONE 3 - DASHBOARD LAYER TESTS")
    print("="*60)
    
    pytest.main([__file__, "-v", "--tb=short"])
