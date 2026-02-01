"""Advanced browser automation and monkey-patching for the pipeline.

This module provides the `BrowserExecutor` which implements a robust Two-Pass
form-filling strategy. It also contains several monkey-patches for the 
`browser-use` library to enable features like local file:// URL access, 
improved stability tracking, and custom launch logging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import aiofiles
from browser_use import Agent, Browser, BrowserProfile
from browser_use.browser.events import (
    BrowserStateRequestEvent,
)
from browser_use.browser.views import (
    BrowserStateSummary,
    PageInfo,
    SerializedDOMState,
)
from browser_use.browser.watchdogs.dom_watchdog import DOMWatchdog
from browser_use.browser.watchdogs.local_browser_watchdog import LocalBrowserWatchdog
from browser_use.browser.watchdogs.security_watchdog import SecurityWatchdog
from browser_use.llm.google.chat import ChatGoogle
from browser_use.utils import create_task_with_error_handling
from langchain_core.messages import HumanMessage

from pipeline.config import PipelineConfig

# Monkey-patch SecurityWatchdog to allow file:// URLs
_original_is_url_allowed = SecurityWatchdog._is_url_allowed

def _patched_is_url_allowed(self: SecurityWatchdog, url: str) -> bool:
    """Patch for SecurityWatchdog to permit local file access.
    
    This override ensures that the agent can interact with local HTML files
    and documents stored on the filesystem.
    """
    # Always allow file:// URLs for local development
    if url.startswith('file:///'):
        return True
    return _original_is_url_allowed(self, url)

SecurityWatchdog._is_url_allowed = _patched_is_url_allowed

# Monkey-patch LocalBrowserWatchdog to log launch command
_original_launch_browser = LocalBrowserWatchdog._launch_browser

async def _patched_launch_browser(self: LocalBrowserWatchdog, max_retries: int = 3) -> Any:
    self.logger.info(f"[LocalBrowserWatchdog] Intercepted launch! Profile args: {self.browser_session.browser_profile.get_args()}")
    return await _original_launch_browser(self, max_retries)

LocalBrowserWatchdog._launch_browser = _patched_launch_browser

# Monkey-patch DOMWatchdog to allow DOM building for file:// URLs


_original_on_browser_state_request = DOMWatchdog.on_BrowserStateRequestEvent

async def _handle_empty_page_state(self: DOMWatchdog, page_url: str, tabs_info: list[PageInfo], event: BrowserStateRequestEvent) -> BrowserStateSummary:
    self.logger.debug(f'⚡ Skipping BuildDOMTree for empty target: {page_url}')
    
    # Create minimal DOM state
    content = SerializedDOMState(_root=None, selector_map={})
    screenshot_b64 = None
    
    try:
        page_info = await self._get_page_info()
    except Exception as e:
        self.logger.debug(f'Failed to get page info from CDP for empty page: {e}, using fallback')
        viewport = self.browser_session.browser_profile.viewport or {'width': 1280, 'height': 720}
        page_info = PageInfo(
            viewport_width=viewport['width'],
            viewport_height=viewport['height'],
            page_width=viewport['width'],
            page_height=viewport['height'],
            scroll_x=0, scroll_y=0,
            pixels_above=0, pixels_below=0,
            pixels_left=0, pixels_right=0,
        )

    return BrowserStateSummary(
        dom_state=content,
        url=page_url,
        title='Empty Tab',
        tabs=tabs_info,
        screenshot=screenshot_b64,
        page_info=page_info,
        pixels_above=0, pixels_below=0,
        browser_errors=[],
        is_pdf_viewer=False,
        recent_events=self._get_recent_events_str() if event.include_recent_events else None,
        pending_network_requests=[],
        pagination_buttons=[],
        closed_popup_messages=self.browser_session._closed_popup_messages.copy(),
    )


async def _get_page_info_safe(self) -> PageInfo:
    try:
        self.logger.debug('🔍 DOMWatchdog.on_BrowserStateRequestEvent: Getting page info from CDP...')
        return await asyncio.wait_for(self._get_page_info(), timeout=1.0)
    except Exception as e:
        self.logger.debug(f'🔍 DOMWatchdog.on_BrowserStateRequestEvent: Failed to get page info from CDP: {e}, using fallback')
        viewport = self.browser_session.browser_profile.viewport or {'width': 1280, 'height': 720}
        return PageInfo(
            viewport_width=viewport['width'],
            viewport_height=viewport['height'],
            page_width=viewport['width'],
            page_height=viewport['height'],
            scroll_x=0, scroll_y=0,
            pixels_above=0, pixels_below=0,
            pixels_left=0, pixels_right=0,
        )


async def _execute_dom_and_screenshot(self, event: BrowserStateRequestEvent) -> tuple[SerializedDOMState, str | None]:
    dom_task = None
    screenshot_task = None

    if event.include_dom:
        self.logger.debug('🔍 DOMWatchdog.on_BrowserStateRequestEvent: 🌳 Starting DOM tree build task...')
        previous_state = (
            self.browser_session._cached_browser_state_summary.dom_state
            if self.browser_session._cached_browser_state_summary
            else None
        )
        dom_task = create_task_with_error_handling(
            self._build_dom_tree_without_highlights(previous_state),
            name='build_dom_tree',
            logger_instance=self.logger,
            suppress_exceptions=True,
        )

    if event.include_screenshot:
        self.logger.debug('🔍 DOMWatchdog.on_BrowserStateRequestEvent: 📸 Starting clean screenshot task...')
        screenshot_task = create_task_with_error_handling(
            self._capture_clean_screenshot(),
            name='capture_screenshot',
            logger_instance=self.logger,
            suppress_exceptions=True,
        )

    content = SerializedDOMState(_root=None, selector_map={})
    screenshot_b64 = None

    if dom_task:
        try:
            content = await dom_task
            self.logger.debug('🔍 DOMWatchdog.on_BrowserStateRequestEvent: ✅ DOM tree build completed')
        except Exception as e:
            self.logger.warning(f'🔍 DOMWatchdog.on_BrowserStateRequestEvent: DOM build failed: {e}, using minimal state')
            # content is already initialized to empty

    if screenshot_task:
        try:
            screenshot_b64 = await screenshot_task
            self.logger.debug('🔍 DOMWatchdog.on_BrowserStateRequestEvent: ✅ Clean screenshot captured')
        except Exception as e:
            self.logger.warning(f'🔍 DOMWatchdog.on_BrowserStateRequestEvent: Clean screenshot failed: {e}')
            screenshot_b64 = None
            
    return content, screenshot_b64


def _create_recovery_state(page_url: str, error_msg: str) -> BrowserStateSummary:
    return BrowserStateSummary(
        dom_state=SerializedDOMState(_root=None, selector_map={}),
        url=page_url,
        title='Error',
        tabs=[],
        screenshot=None,
        page_info=PageInfo(
            viewport_width=1280, viewport_height=720,
            page_width=1280, page_height=720,
            scroll_x=0, scroll_y=0,
            pixels_above=0, pixels_below=0,
            pixels_left=0, pixels_right=0,
        ),
        pixels_above=0, pixels_below=0,
        browser_errors=[error_msg],
        is_pdf_viewer=False,
        recent_events=None,
        pending_network_requests=[],
        pagination_buttons=[],
        closed_popup_messages=[],
    )


async def _add_highlights_if_needed(self, content: SerializedDOMState):
    if not (content and content.selector_map and self.browser_session.browser_profile.dom_highlight_elements):
        return
    try:
        self.logger.debug('🔍 DOMWatchdog.on_BrowserStateRequestEvent: 🎨 Adding browser-side highlights...')
        await self.browser_session.add_highlights(content.selector_map)
    except Exception as e:
        self.logger.warning(f'🔍 DOMWatchdog.on_BrowserStateRequestEvent: Browser highlighting failed: {e}')


async def _get_title_safe(self) -> str:
    try:
        self.logger.debug('🔍 DOMWatchdog.on_BrowserStateRequestEvent: Getting page title...')
        return await asyncio.wait_for(self.browser_session.get_current_page_title(), timeout=1.0)
    except Exception as e:
        self.logger.debug(f'🔍 DOMWatchdog.on_BrowserStateRequestEvent: Failed to get title: {e}')
        return 'Page'


async def _wait_for_page_stability(self, pending_requests: list):
    self.logger.debug('🔍 DOMWatchdog.on_BrowserStateRequestEvent: ⏳ Waiting for page stability...')
    try:
        if pending_requests:
            await asyncio.sleep(0.3)
        self.logger.debug('🔍 DOMWatchdog.on_BrowserStateRequestEvent: ✅ Page stability complete')
    except Exception as e:
        self.logger.warning(
            f'🔍 DOMWatchdog.on_BrowserStateRequestEvent: Network waiting failed: {e}, continuing anyway...'
        )


async def on_patched_browser_state_request_event(self, event: BrowserStateRequestEvent) -> BrowserStateSummary:
    """Coordinated browser state capture with enhanced stability and local file support.
    
    This is a patched replacement for `DOMWatchdog.on_BrowserStateRequestEvent`. 
    It ensures that local `file://` URLs are treated as valid states and 
    implements extra wait logic for network stability before capturing the DOM.
    """
    try:
        self.logger.debug('🔍 DOMWatchdog.on_BrowserStateRequestEvent: STARTING browser state request (PATCHED)')
        page_url = await self.browser_session.get_current_page_url()
        self.logger.debug(f'🔍 DOMWatchdog.on_BrowserStateRequestEvent: Got page URL: {page_url}')

        if self.browser_session.agent_focus_target_id:
            self.logger.debug(f'Current page URL: {page_url}, target_id: {self.browser_session.agent_focus_target_id}')

        scheme = page_url.lower().split(':', 1)[0]
        not_a_meaningful_website = scheme not in ('http', 'https', 'file')

        pending_requests = []
        if not not_a_meaningful_website:
            try:
                pending_requests = await self._get_pending_network_requests()
                if pending_requests:
                    self.logger.debug(f'🔍 Found {len(pending_requests)} pending requests before stability wait')
            except Exception as e:
                self.logger.debug(f'Failed to get pending requests before wait: {e}')
            
            await _wait_for_page_stability(self, pending_requests)

        self.logger.debug('🔍 DOMWatchdog.on_BrowserStateRequestEvent: Getting tabs info...')
        tabs_info = await self.browser_session.get_tabs()
        self.logger.debug(f'🔍 DOMWatchdog.on_BrowserStateRequestEvent: Got {len(tabs_info)} tabs')

        if not_a_meaningful_website:
            return await _handle_empty_page_state(self, page_url, tabs_info, event)

        content, screenshot_b64 = await _execute_dom_and_screenshot(self, event)
        await _add_highlights_if_needed(self, content)

        title = await _get_title_safe(self)
        page_info = await _get_page_info_safe(self)
        
        is_pdf_viewer = page_url.endswith('.pdf') or '/pdf/' in page_url

        pagination_buttons_data = []
        if content and content.selector_map:
            pagination_buttons_data = self._detect_pagination_buttons(content.selector_map)

        browser_state = BrowserStateSummary(
            dom_state=content,
            url=page_url,
            title=title,
            tabs=tabs_info,
            screenshot=screenshot_b64,
            page_info=page_info,
            pixels_above=0, pixels_below=0,
            browser_errors=[],
            is_pdf_viewer=is_pdf_viewer,
            recent_events=self._get_recent_events_str() if event.include_recent_events else None,
            pending_network_requests=pending_requests,
            pagination_buttons=pagination_buttons_data,
            closed_popup_messages=self.browser_session._closed_popup_messages.copy(),
        )

        self.browser_session._cached_browser_state_summary = browser_state
        if page_info:
            self.browser_session._original_viewport_size = (page_info.viewport_width, page_info.viewport_height)

        self.logger.debug('🔍 DOMWatchdog.on_BrowserStateRequestEvent: ✅ COMPLETED - Returning browser state')
        return browser_state

    except Exception as e:
        self.logger.error(f'Failed to get browser state: {e}')
        return _create_recovery_state(locals().get('page_url', ''), str(e))

DOMWatchdog.on_BrowserStateRequestEvent = on_patched_browser_state_request_event

logger = logging.getLogger(__name__)

class BrowserExecutor:
    """High-level browser automation engine for form filling.

    Extends the base agent capabilities by implementing a Two-Pass strategy:
    1. Schema Extraction: Discovering field labels and constraints.
    2. Data Mapping: Using an LLM to map source data to the discovered schema.
    3. Deterministic Filling: Executing precise interactions to fill the form.
    """
    
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.llm = ChatGoogle(
            model="gemini-2.0-flash",
            api_key=api_key
        )

    async def _extract_schema(self, browser_config: BrowserProfile, url: str) -> str:
        """Pass 1: Extracts the form schema (labels and constraints) from the page."""
        task = (
            f"Navigate to {url}. \n"
            "Identify all visible input fields, textareas, and form elements. \n"
            "Wait for the form to fully load. Scroll if necessary. \n"
            "For each field, extract: \n"
            "1. The exact label text.\n"
            "2. The field type (text, email, tel, etc.).\n"
            "3. Any constraints (e.g., 'Max 50 characters').\n"
            "Output the results as a clean list of fields. "
            "Do NOT fill anything yet. Just report the schema and stop."
        )
        
        browser = Browser(browser_profile=browser_config)
        agent = Agent(task=task, llm=self.llm, browser=browser)
        try:
            result = await agent.run(max_steps=10)
            return str(result.final_result())
        finally:
            await browser.kill()

    async def _map_data_to_schema(self, schema: str, source_data: dict[str, Any]) -> dict[str, Any]:
        """Pass 1.5: Uses LLM to map source data to the extracted schema labels."""
        prompt = (
            "You are a Form Mapping Expert. \n"
            "Given the following FORM SCHEMA (list of labels/fields) and SOURCE DATA (JSON), "
            "create a deterministic MAPPING JSON. \n\n"
            f"### FORM SCHEMA:\n{schema}\n\n"
            f"### SOURCE DATA:\n{json.dumps(source_data, indent=2)}\n\n"
            "### RULES:\n"
            "1. Map each source data point to the MOST RELEVANT form label.\n"
            "2. If a field has a character limit (e.g. 50), TRUNCATE the data to fit.\n"
            "3. Ensure distinct fields for distinct data (e.g. Founder 1 vs Founder 2).\n"
            "4. NEVER merge two data points into one field.\n"
            "5. If a field type is 'email', ensure the mapped value is a valid email.\n"
            "6. Output ONLY a JSON object where keys are FORM LABELS and values are the STRINGS to type."
        )
        # Use the underlying LLM with a HumanMessage to avoid Pydantic errors
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        content = response.completion
        # Basic cleanup of markdown if LLM returned it
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        try:
            return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to parse mapping JSON: {e}. Content: {content}")
            return {}

    async def fill_form(self, url: str, data: dict[str, Any]) -> str:
        """Fills a form at the given URL with the provided data using a Two-Pass strategy."""
        logger.info(f"Starting Two-Pass browser agent to fill form at {url}")
        
        # Get browser path and user data dir from env
        config_chrome_path = os.environ.get("CHROME_PATH") or os.environ.get("BRAVE_PATH")
        config_user_data_dir = os.environ.get("USER_DATA_DIR")

        browser_config = BrowserProfile(
            disable_security=True,
            executable_path=config_chrome_path,
            user_data_dir=config_user_data_dir,
            args=[
                "--allow-file-access-from-files",
                "--no-sandbox",
                "--disable-setuid-sandbox"
            ]
        )

        try:
            # PASS 1: Extract Schema
            logger.info("Pass 1: Extracting Form Schema...")
            schema_text = await self._extract_schema(browser_config, url)
            logger.debug(f"Extracted Schema: {schema_text}")

            # PASS 1.5: Map Data to Schema
            logger.info("Pass 1.5: Pre-mapping data to schema...")
            mapping = await self._map_data_to_schema(schema_text, data)
            mapping_str = json.dumps(mapping, indent=2)
            logger.debug(f"Generated Mapping: {mapping_str}")

            # PASS 2: Deterministic Filling
            logger.info("Pass 2: Executing deterministic filling...")
            # We recreate browser_config to ensure a fresh session for Pass 2 if needed
            result_text = await self._fill_form_deterministically_internal(browser_config, url, mapping)
            
            return result_text

        except Exception as e:
            logger.error(f"Browser agent failed during Two-Pass execution: {e}")
            raise

    async def _fill_form_deterministically_internal(self, browser_config: BrowserProfile, url: str, mapping: dict[str, Any]) -> str:
        """Pass 2: Fills the form using a strict, label-based deterministic mapping."""
        mapping_str = json.dumps(mapping, indent=2)
        task = (
            f"# TASK: Fill Application Form (DETERMINISTIC MODE)\n"
            f" ## 1. NAVIGATION\n"
            f" - **ACTION**: Navigate to {url} immediately.\n\n"
            f" ## 2. MAPPING TO EXECUTE\n"
            f" The following JSON provides the EXACT strings to type for each label:\n"
            f" ```json\n{mapping_str}\n```\n\n"
            f" ## 3. EXECUTION RULES\n"
            f" - **Strict Logic**: For each key in the JSON, find the element with that EXACT label and type the value.\n"
            f" - **ANTI-SLICING**: CLICK the field, WAIT 1 second, then TYPE to prevent first-letter cutoff.\n"
            f" - **ANTI-HANG**: After typing, immediately `Tab` out or click outside to force save.\n"
            f" - **No Guesswork**: DO NOT infer anything. ONLY type what is in the mapping JSON.\n"
            f" - **Verification**: Once all fields in the JSON are filled, you are DONE.\n\n"
            f" ## 4. COMPLETION\n"
            f" - Verify the form is filled according to the mapping and report success."
        )

        browser = Browser(browser_profile=browser_config)
        agent = Agent(task=task, llm=self.llm, browser=browser)
        try:
            result = await agent.run(max_steps=30)
            final_result_str = str(result.final_result())
            
            # Log results
            async with aiofiles.open('browser_output.log', 'a') as log_f:
                await log_f.write(f"\n--- Run Result ({url}) ---\n")
                await log_f.write(final_result_str)
                await log_f.write("\n")
            
            return final_result_str
        finally:
            await browser.kill()
