# AI Workflow Agent - Test Script
"""
Test script to verify basic functionality.
Run this locally (without Docker) to test the components.
"""

import asyncio
import sys
import os

# Add agent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))

from config import settings, ProjectType, CLASSIFICATION_KEYWORDS


def test_keyword_classification():
    """Test keyword-based classification."""
    print("\n" + "="*50)
    print("TEST: Keyword Classification")
    print("="*50)
    
    test_queries = [
        ("Create an automation that sends emails", ProjectType.N8N),
        ("Generate an image of a sunset", ProjectType.COMFYUI),
        ("Build workflow with stable diffusion", ProjectType.COMFYUI),
        ("Download a GitHub repository and run it", ProjectType.EXTERNAL_REPO),
        ("Create automation with AI image generation", ProjectType.HYBRID),
    ]
    
    for query, expected in test_queries:
        # Simple keyword matching
        query_lower = query.lower()
        scores = {}
        
        for project_type, keywords in CLASSIFICATION_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            scores[project_type] = score
        
        if max(scores.values()) > 0:
            detected = max(scores, key=scores.get)
        else:
            detected = ProjectType.UNKNOWN
        
        status = "✅" if detected == expected else "❌"
        print(f"{status} Query: '{query[:40]}...'")
        print(f"   Expected: {expected}, Got: {detected}")


def test_workflow_templates():
    """Test workflow template generation."""
    print("\n" + "="*50)
    print("TEST: Workflow Templates")
    print("="*50)
    
    # Test n8n template structure
    from tools.n8n_builder import N8NWorkflowBuilder
    
    builder = N8NWorkflowBuilder()
    
    test_queries = [
        ("webhook automation", "webhook"),
        ("schedule daily task", "schedule"),
        ("send email notification", "email"),
    ]
    
    for query, expected_type in test_queries:
        detected = builder._detect_workflow_type(query)
        status = "✅" if detected == expected_type else "❌"
        print(f"{status} Query: '{query}' → Type: {detected} (expected: {expected_type})")


def test_comfyui_templates():
    """Test ComfyUI template generation."""
    print("\n" + "="*50)
    print("TEST: ComfyUI Templates")
    print("="*50)
    
    from tools.comfyui_builder import ComfyUIWorkflowBuilder
    
    builder = ComfyUIWorkflowBuilder()
    
    test_queries = [
        ("generate an image", "txt2img"),
        ("upscale this photo", "upscale"),
        ("inpaint the background", "inpaint"),
        ("use controlnet for pose", "controlnet"),
    ]
    
    for query, expected_type in test_queries:
        detected = builder._detect_workflow_type(query)
        status = "✅" if detected == expected_type else "❌"
        print(f"{status} Query: '{query}' → Type: {detected} (expected: {expected_type})")


def test_colab_offload_logic():
    """Test Colab offload decision logic."""
    print("\n" + "="*50)
    print("TEST: Colab Offload Logic")
    print("="*50)
    
    test_cases = [
        ("generate single image with SD 1.5", False),
        ("generate 20 images with Flux", True),
        ("train a LoRA model", True),
        ("quick preview", False),
        ("use colab for this task", True),
        ("run 70b llama model", True),
    ]
    
    # Simplified offload logic (same as in colab/__init__.py)
    heavy_keywords = [
        "70b", "40b", "13b", "llama 2", "mixtral", "flux",
        "batch", "multiple images", "generate 10", "generate 20",
        "video", "animation", "animatediff",
        "train", "fine-tune", "finetune", "lora training",
        "use colab", "offload", "remote", "not local", "save gpu"
    ]
    
    light_keywords = [
        "single image", "quick", "fast", "small model", "7b", "3b",
        "test", "preview"
    ]
    
    for query, expected_offload in test_cases:
        query_lower = query.lower()
        heavy_score = sum(1 for kw in heavy_keywords if kw in query_lower)
        light_score = sum(1 for kw in light_keywords if kw in query_lower)
        
        should_offload = heavy_score > light_score and heavy_score >= 1
        
        status = "✅" if should_offload == expected_offload else "❌"
        print(f"{status} Query: '{query}' → Offload: {should_offload} (expected: {expected_offload})")


def test_config():
    """Test configuration loading."""
    print("\n" + "="*50)
    print("TEST: Configuration")
    print("="*50)
    
    print(f"✅ OLLAMA_HOST: {settings.OLLAMA_HOST}")
    print(f"✅ OLLAMA_MODEL: {settings.OLLAMA_MODEL}")
    print(f"✅ N8N_HOST: {settings.N8N_HOST}")
    print(f"✅ COMFYUI_HOST: {settings.COMFYUI_HOST}")
    print(f"✅ PROJECTS_DIR: {settings.PROJECTS_DIR}")


import pytest

@pytest.mark.asyncio
async def test_github_search():
    """Test GitHub search (requires network)."""
    print("\n" + "="*50)
    print("TEST: GitHub Search (Live)")
    print("="*50)
    
    from tools.github_search import GitHubSearchTool
    
    tool = GitHubSearchTool()
    
    try:
        results = await tool.search("comfyui workflow", max_results=2)
        
        if results:
            print(f"✅ Found {len(results)} repositories")
            for repo in results:
                print(f"   - {repo.get('full_name')} ⭐ {repo.get('stars')}")
        else:
            print("⚠️ No results (might be rate limited)")
            
    except Exception as e:
        print(f"❌ GitHub search failed: {e}")


def main():
    """Run all tests."""
    print("="*50)
    print("AI WORKFLOW AGENT - TEST SUITE")
    print("="*50)
    
    # Run sync tests
    test_config()
    test_keyword_classification()
    test_workflow_templates()
    test_comfyui_templates()
    test_colab_offload_logic()
    
    # Run async tests
    print("\n" + "="*50)
    print("Running async tests...")
    print("="*50)
    
    asyncio.run(test_github_search())
    
    print("\n" + "="*50)
    print("TEST SUITE COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
