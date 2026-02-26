"""
Milestone 3: Playwright Web Analyzer
Analyze DOM structure and convert layouts to JSON
"""

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ============================================================
# Enums and Data Classes
# ============================================================

class ElementType(str, Enum):
    """Types of DOM elements"""
    CONTAINER = "container"
    TEXT = "text"
    IMAGE = "image"
    LINK = "link"
    BUTTON = "button"
    INPUT = "input"
    TABLE = "table"
    LIST = "list"
    FORM = "form"
    HEADER = "header"
    FOOTER = "footer"
    NAV = "nav"
    UNKNOWN = "unknown"


class LayoutType(str, Enum):
    """Types of page layouts"""
    SINGLE_COLUMN = "single_column"
    TWO_COLUMN = "two_column"
    THREE_COLUMN = "three_column"
    GRID = "grid"
    MASONRY = "masonry"
    SIDEBAR_LEFT = "sidebar_left"
    SIDEBAR_RIGHT = "sidebar_right"
    FULL_WIDTH = "full_width"


@dataclass
class BoundingBox:
    """Element bounding box"""
    x: float
    y: float
    width: float
    height: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }
    
    def area(self) -> float:
        return self.width * self.height
    
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)


@dataclass
class DOMElement:
    """A DOM element with its properties"""
    element_id: str
    tag: str
    element_type: ElementType
    bounds: BoundingBox
    text_content: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    styles: Dict[str, str] = field(default_factory=dict)
    children: List["DOMElement"] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.element_id,
            "tag": self.tag,
            "type": self.element_type.value,
            "bounds": self.bounds.to_dict(),
            "textContent": self.text_content,
            "attributes": self.attributes,
            "styles": self.styles,
            "children": [c.to_dict() for c in self.children],
            "childCount": len(self.children)
        }


@dataclass
class PageAnalysis:
    """Analysis results for a web page"""
    url: str
    title: str
    layout_type: LayoutType
    elements: List[DOMElement]
    viewport: Dict[str, int]
    analyzed_at: datetime = field(default_factory=datetime.now)
    screenshot_base64: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "layoutType": self.layout_type.value,
            "elements": [e.to_dict() for e in self.elements],
            "elementCount": len(self.elements),
            "viewport": self.viewport,
            "analyzedAt": self.analyzed_at.isoformat(),
            "metadata": self.metadata
        }


# ============================================================
# Layout Converter (Layout → JSON)
# ============================================================

class LayoutConverter:
    """
    Convert web page layouts to editable JSON format.
    This is the Layout → JSON Converter component.
    """
    
    def __init__(self):
        self.conversion_history: List[Dict[str, Any]] = []
    
    def elements_to_json(
        self,
        elements: List[DOMElement],
        include_styles: bool = True,
        flatten: bool = False
    ) -> Dict[str, Any]:
        """
        Convert DOM elements to JSON layout format.
        
        Args:
            elements: List of DOM elements
            include_styles: Whether to include CSS styles
            flatten: Whether to flatten nested structure
            
        Returns:
            JSON-compatible dictionary
        """
        if flatten:
            flat_elements = self._flatten_elements(elements)
            json_elements = [self._element_to_json(e, include_styles) for e in flat_elements]
        else:
            json_elements = [self._element_to_json(e, include_styles) for e in elements]
        
        layout_json = {
            "version": "1.0",
            "generator": "PlaywrightAnalyzer",
            "generatedAt": datetime.now().isoformat(),
            "elements": json_elements,
            "stats": {
                "totalElements": len(json_elements),
                "elementTypes": self._count_element_types(elements)
            }
        }
        
        self.conversion_history.append({
            "timestamp": datetime.now().isoformat(),
            "elementCount": len(json_elements),
            "includeStyles": include_styles,
            "flatten": flatten
        })
        
        return layout_json
    
    def _element_to_json(self, element: DOMElement, include_styles: bool) -> Dict[str, Any]:
        """Convert a single element to JSON"""
        json_elem = {
            "id": element.element_id,
            "tag": element.tag,
            "type": element.element_type.value,
            "bounds": element.bounds.to_dict(),
            "attributes": element.attributes
        }
        
        if element.text_content:
            json_elem["textContent"] = element.text_content[:200]  # Truncate
        
        if include_styles and element.styles:
            json_elem["styles"] = element.styles
        
        if element.children:
            json_elem["children"] = [
                self._element_to_json(c, include_styles) 
                for c in element.children
            ]
        
        return json_elem
    
    def _flatten_elements(self, elements: List[DOMElement]) -> List[DOMElement]:
        """Flatten nested element tree"""
        flat = []
        
        def recurse(elem_list):
            for elem in elem_list:
                flat.append(elem)
                if elem.children:
                    recurse(elem.children)
        
        recurse(elements)
        return flat
    
    def _count_element_types(self, elements: List[DOMElement]) -> Dict[str, int]:
        """Count elements by type"""
        counts = {}
        
        def recurse(elem_list):
            for elem in elem_list:
                type_name = elem.element_type.value
                counts[type_name] = counts.get(type_name, 0) + 1
                if elem.children:
                    recurse(elem.children)
        
        recurse(elements)
        return counts
    
    def json_to_appsmith(self, layout_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert layout JSON to Appsmith-compatible format.
        
        Args:
            layout_json: Layout JSON from elements_to_json
            
        Returns:
            Appsmith widget DSL format
        """
        widgets = []
        
        for element in layout_json.get("elements", []):
            widget = self._element_json_to_widget(element)
            if widget:
                widgets.append(widget)
        
        return {
            "widgetName": "MainContainer",
            "type": "CANVAS_WIDGET",
            "children": widgets,
            "version": 1
        }
    
    def _element_json_to_widget(self, element: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert element JSON to Appsmith widget"""
        elem_type = element.get("type", "unknown")
        bounds = element.get("bounds", {})
        
        # Map element type to widget type
        widget_type_map = {
            "text": "TEXT_WIDGET",
            "button": "BUTTON_WIDGET",
            "input": "INPUT_WIDGET",
            "image": "IMAGE_WIDGET",
            "table": "TABLE_WIDGET",
            "container": "CONTAINER_WIDGET",
            "link": "BUTTON_WIDGET",  # Links as buttons
            "list": "LIST_WIDGET"
        }
        
        widget_type = widget_type_map.get(elem_type, "CONTAINER_WIDGET")
        
        widget = {
            "widgetId": element.get("id", f"w_{uuid.uuid4().hex[:8]}"),
            "widgetName": element.get("id", "unnamed"),
            "type": widget_type,
            "leftColumn": int(bounds.get("x", 0) / 10),  # Grid units
            "rightColumn": int((bounds.get("x", 0) + bounds.get("width", 100)) / 10),
            "topRow": int(bounds.get("y", 0) / 10),
            "bottomRow": int((bounds.get("y", 0) + bounds.get("height", 50)) / 10),
        }
        
        # Add type-specific properties
        if elem_type == "text":
            widget["text"] = element.get("textContent", "")
        elif elem_type == "button":
            widget["buttonLabel"] = element.get("textContent", "Button")
        elif elem_type == "input":
            widget["placeholderText"] = element.get("attributes", {}).get("placeholder", "")
        
        return widget
    
    def get_conversion_history(self) -> List[Dict[str, Any]]:
        """Get history of conversions"""
        return self.conversion_history


# ============================================================
# Playwright Analyzer
# ============================================================

class PlaywrightAnalyzer:
    """
    Analyze web pages using Playwright.
    Extracts DOM structure, layout, and converts to JSON.
    """
    
    # Tag to element type mapping
    TAG_TYPE_MAP = {
        "div": ElementType.CONTAINER,
        "section": ElementType.CONTAINER,
        "article": ElementType.CONTAINER,
        "main": ElementType.CONTAINER,
        "aside": ElementType.CONTAINER,
        "p": ElementType.TEXT,
        "span": ElementType.TEXT,
        "h1": ElementType.TEXT,
        "h2": ElementType.TEXT,
        "h3": ElementType.TEXT,
        "h4": ElementType.TEXT,
        "img": ElementType.IMAGE,
        "a": ElementType.LINK,
        "button": ElementType.BUTTON,
        "input": ElementType.INPUT,
        "textarea": ElementType.INPUT,
        "select": ElementType.INPUT,
        "table": ElementType.TABLE,
        "ul": ElementType.LIST,
        "ol": ElementType.LIST,
        "form": ElementType.FORM,
        "header": ElementType.HEADER,
        "footer": ElementType.FOOTER,
        "nav": ElementType.NAV
    }
    
    def __init__(self):
        self.converter = LayoutConverter()
        self.analysis_history: List[PageAnalysis] = []
        self._browser = None
        self._playwright = None
    
    async def _ensure_browser(self):
        """Ensure Playwright browser is available"""
        if self._browser is not None:
            return True
            
        try:
            from playwright.async_api import async_playwright
            
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            return True
        except ImportError:
            logger.warning("Playwright not installed. Using mock analysis.")
            return False
        except Exception as e:
            logger.error(f"Failed to launch browser: {e}")
            return False
    
    async def analyze_page(
        self,
        url: str,
        viewport_width: int = 1920,
        viewport_height: int = 1080,
        wait_for_selector: Optional[str] = None,
        take_screenshot: bool = False
    ) -> PageAnalysis:
        """
        Analyze a web page.
        
        Args:
            url: URL to analyze
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            wait_for_selector: Optional selector to wait for
            take_screenshot: Whether to capture screenshot
            
        Returns:
            PageAnalysis object
        """
        browser_available = await self._ensure_browser()
        
        if browser_available:
            return await self._analyze_with_playwright(
                url, viewport_width, viewport_height, 
                wait_for_selector, take_screenshot
            )
        else:
            return self._mock_analysis(url, viewport_width, viewport_height)
    
    async def _analyze_with_playwright(
        self,
        url: str,
        viewport_width: int,
        viewport_height: int,
        wait_for_selector: Optional[str],
        take_screenshot: bool
    ) -> PageAnalysis:
        """Perform actual Playwright analysis"""
        page = await self._browser.new_page(
            viewport={"width": viewport_width, "height": viewport_height}
        )
        
        try:
            await page.goto(url, wait_until="domcontentloaded")
            
            if wait_for_selector:
                await page.wait_for_selector(wait_for_selector, timeout=10000)
            
            # Get page title
            title = await page.title()
            
            # Extract DOM elements
            elements_data = await page.evaluate("""
                () => {
                    const extractElement = (el, depth = 0) => {
                        if (depth > 5) return null;  // Limit depth
                        
                        const rect = el.getBoundingClientRect();
                        const styles = window.getComputedStyle(el);
                        
                        // Skip hidden elements
                        if (rect.width === 0 || rect.height === 0) return null;
                        if (styles.display === 'none') return null;
                        
                        const children = [];
                        for (const child of el.children) {
                            const childData = extractElement(child, depth + 1);
                            if (childData) children.push(childData);
                        }
                        
                        return {
                            tag: el.tagName.toLowerCase(),
                            id: el.id || null,
                            className: el.className || null,
                            bounds: {
                                x: rect.x,
                                y: rect.y,
                                width: rect.width,
                                height: rect.height
                            },
                            textContent: el.innerText?.substring(0, 100) || null,
                            attributes: {
                                href: el.getAttribute('href'),
                                src: el.getAttribute('src'),
                                alt: el.getAttribute('alt'),
                                placeholder: el.getAttribute('placeholder')
                            },
                            styles: {
                                display: styles.display,
                                position: styles.position,
                                backgroundColor: styles.backgroundColor,
                                color: styles.color,
                                fontSize: styles.fontSize
                            },
                            children: children
                        };
                    };
                    
                    return extractElement(document.body);
                }
            """)
            
            # Convert to DOMElement objects
            elements = self._parse_element_data(elements_data) if elements_data else []
            
            # Determine layout type
            layout_type = self._detect_layout_type(elements)
            
            # Take screenshot if requested
            screenshot_b64 = None
            if take_screenshot:
                screenshot_bytes = await page.screenshot(full_page=False)
                import base64
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode()
            
            analysis = PageAnalysis(
                url=url,
                title=title,
                layout_type=layout_type,
                elements=elements,
                viewport={"width": viewport_width, "height": viewport_height},
                screenshot_base64=screenshot_b64,
                metadata={
                    "elementCount": self._count_all_elements(elements),
                    "hasPlaywright": True
                }
            )
            
            self.analysis_history.append(analysis)
            return analysis
            
        finally:
            await page.close()
    
    def _mock_analysis(
        self,
        url: str,
        viewport_width: int,
        viewport_height: int
    ) -> PageAnalysis:
        """Create mock analysis when Playwright is not available"""
        # Create sample DOM structure
        elements = [
            DOMElement(
                element_id="header_1",
                tag="header",
                element_type=ElementType.HEADER,
                bounds=BoundingBox(0, 0, viewport_width, 80),
                text_content="Header Navigation",
                styles={"backgroundColor": "#ffffff", "display": "flex"}
            ),
            DOMElement(
                element_id="main_1",
                tag="main",
                element_type=ElementType.CONTAINER,
                bounds=BoundingBox(0, 80, viewport_width, viewport_height - 160),
                children=[
                    DOMElement(
                        element_id="content_1",
                        tag="div",
                        element_type=ElementType.CONTAINER,
                        bounds=BoundingBox(20, 100, viewport_width - 40, 400),
                        text_content="Main Content Area"
                    )
                ]
            ),
            DOMElement(
                element_id="footer_1",
                tag="footer",
                element_type=ElementType.FOOTER,
                bounds=BoundingBox(0, viewport_height - 80, viewport_width, 80),
                text_content="Footer"
            )
        ]
        
        analysis = PageAnalysis(
            url=url,
            title=f"Mock Analysis: {urlparse(url).netloc}",
            layout_type=LayoutType.SINGLE_COLUMN,
            elements=elements,
            viewport={"width": viewport_width, "height": viewport_height},
            metadata={
                "elementCount": 4,
                "hasPlaywright": False,
                "mockData": True
            }
        )
        
        self.analysis_history.append(analysis)
        return analysis
    
    def _parse_element_data(self, data: Dict[str, Any], depth: int = 0) -> List[DOMElement]:
        """Parse JavaScript element data into DOMElement objects"""
        if not data or depth > 5:
            return []
        
        tag = data.get("tag", "div")
        bounds_data = data.get("bounds", {})
        
        element = DOMElement(
            element_id=data.get("id") or f"elem_{uuid.uuid4().hex[:8]}",
            tag=tag,
            element_type=self.TAG_TYPE_MAP.get(tag, ElementType.UNKNOWN),
            bounds=BoundingBox(
                x=bounds_data.get("x", 0),
                y=bounds_data.get("y", 0),
                width=bounds_data.get("width", 0),
                height=bounds_data.get("height", 0)
            ),
            text_content=data.get("textContent"),
            attributes={k: v for k, v in (data.get("attributes") or {}).items() if v},
            styles=data.get("styles", {}),
            children=[]
        )
        
        # Parse children
        for child_data in data.get("children", []):
            child_elements = self._parse_element_data(child_data, depth + 1)
            element.children.extend(child_elements)
        
        return [element]
    
    def _detect_layout_type(self, elements: List[DOMElement]) -> LayoutType:
        """Detect the layout type based on element positions"""
        if not elements:
            return LayoutType.SINGLE_COLUMN
        
        # Count major content columns
        top_level = [e for e in elements if e.bounds.y < 200]
        
        if not top_level:
            return LayoutType.SINGLE_COLUMN
        
        # Check for sidebar patterns
        widths = [e.bounds.width for e in top_level]
        x_positions = [e.bounds.x for e in top_level]
        
        # Simplistic detection
        if len(top_level) >= 3:
            return LayoutType.THREE_COLUMN
        elif len(top_level) == 2:
            # Check if one is a sidebar
            if min(widths) < 300:
                return LayoutType.SIDEBAR_LEFT if x_positions[0] < x_positions[1] else LayoutType.SIDEBAR_RIGHT
            return LayoutType.TWO_COLUMN
        else:
            return LayoutType.SINGLE_COLUMN
    
    def _count_all_elements(self, elements: List[DOMElement]) -> int:
        """Count all elements including nested"""
        count = len(elements)
        for elem in elements:
            count += self._count_all_elements(elem.children)
        return count
    
    def convert_to_json(
        self,
        analysis: PageAnalysis,
        include_styles: bool = True
    ) -> Dict[str, Any]:
        """Convert page analysis to JSON format"""
        return self.converter.elements_to_json(
            analysis.elements,
            include_styles=include_styles
        )
    
    def convert_to_appsmith(self, analysis: PageAnalysis) -> Dict[str, Any]:
        """Convert page analysis to Appsmith format"""
        layout_json = self.convert_to_json(analysis)
        return self.converter.json_to_appsmith(layout_json)
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get history of page analyses"""
        return [a.to_dict() for a in self.analysis_history]
    
    async def close(self):
        """Close browser and cleanup"""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            "total_analyses": len(self.analysis_history),
            "conversions": len(self.converter.conversion_history),
            "layout_types": self._count_layout_types()
        }
    
    def _count_layout_types(self) -> Dict[str, int]:
        """Count analyses by layout type"""
        counts = {}
        for analysis in self.analysis_history:
            lt = analysis.layout_type.value
            counts[lt] = counts.get(lt, 0) + 1
        return counts


# ============================================================
# Singleton Instance
# ============================================================

_playwright_analyzer: Optional[PlaywrightAnalyzer] = None


def get_playwright_analyzer() -> PlaywrightAnalyzer:
    """Get or create the Playwright analyzer singleton"""
    global _playwright_analyzer
    if _playwright_analyzer is None:
        _playwright_analyzer = PlaywrightAnalyzer()
    return _playwright_analyzer
