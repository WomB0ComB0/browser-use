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

from pipeline.config import PipelineConfig

# Monkey-patch SecurityWatchdog to allow file:// URLs
_original_is_url_allowed = SecurityWatchdog._is_url_allowed

def _patched_is_url_allowed(self, url: str) -> bool:
    # Always allow file:// URLs for local development
    if url.startswith('file:///'):
        return True
    return _original_is_url_allowed(self, url)

SecurityWatchdog._is_url_allowed = _patched_is_url_allowed

# Monkey-patch LocalBrowserWatchdog to log launch command
_original_launch_browser = LocalBrowserWatchdog._launch_browser

async def _patched_launch_browser(self, max_retries: int = 3):
    self.logger.info(f"[LocalBrowserWatchdog] Intercepted launch! Profile args: {self.browser_session.browser_profile.get_args()}")
    return await _original_launch_browser(self, max_retries)

LocalBrowserWatchdog._launch_browser = _patched_launch_browser

# Monkey-patch DOMWatchdog to allow DOM building for file:// URLs


_original_on_browser_state_request = DOMWatchdog.on_BrowserStateRequestEvent

async def _handle_empty_page_state(self, page_url: str, tabs_info: list, event: BrowserStateRequestEvent) -> BrowserStateSummary:
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
    """Handle browser state request by coordinating DOM building and screenshot capture.
    
    PATCHED: Allows file:// URLs to be considered meaningful.
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
    """Automates browser interactions based on structured data."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.llm = ChatGoogle(
            model="gemini-2.0-flash",
            api_key=api_key
        )

    async def fill_form(self, url: str, data: dict[str, Any]) -> str:
        """Fills a form at the given URL with the provided data.
        
        Args:
            url: The URL of the form to fill.
            data: A dictionary containing the field mapping and values.
            
        Returns:
            A summary of the execution result.
        """
        logger.info(f"Starting browser agent to fill form at {url}")
        
        # Construct the task prompt
        data_str = json.dumps(data, indent=2)
        task = (
            f"Go to {url} and fill out the application form using the following data:\n"
            f"{data_str}\n\n"
            f"Navigate through any multi-step forms if necessary. "
            f"If there's a submit button, do NOT click it yet, just prepare the form and report success. "
            f"If you encounter fields that aren't in the data, try to infer them or leave them blank."
        )
        
        # Get browser path and user data dir from env
        config_chrome_path = os.environ.get("CHROME_PATH") or os.environ.get("BRAVE_PATH")
        config_user_data_dir = os.environ.get("USER_DATA_DIR")

        profile = BrowserProfile(
            disable_security=True,
            executable_path=config_chrome_path,
            user_data_dir=config_user_data_dir,
            args=[
                "--allow-file-access-from-files",
                "--no-sandbox",
                "--disable-setuid-sandbox"
            ]
        )
        browser = Browser(browser_profile=profile)

        agent = Agent(
            task=task,
            llm=self.llm,
            browser=browser
        )
        
        try:
            # No more redirection to string buffer, let it go to real stdout
            result = await agent.run(max_steps=10)
            
            # We can still log the result to a file if we want
            async with aiofiles.open('browser_output.log', 'a') as log_f:
                await log_f.write("\n--- Run Result ---\n")
                await log_f.write(str(result))
                await log_f.write("\n")
            
            return str(result)
        except Exception as e:
            logging.error(f"Browser agent failed: {e}")
            raise
