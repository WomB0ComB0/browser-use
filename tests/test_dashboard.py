import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from pipeline.dashboard.app import BroadcastLogHandler


class TestDashboardLogging(unittest.TestCase):
    @patch('pipeline.dashboard.app.asyncio')
    def test_broadcast_handler_emit(self, mock_asyncio):
        # Setup
        app_mock = MagicMock()
        app_mock.broadcast_update = AsyncMock()
        
        handler = BroadcastLogHandler(app_mock)
        
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test log message",
            args=(),
            exc_info=None
        )
        
        # Determine the loop
        loop_mock = MagicMock()
        mock_asyncio.get_running_loop.return_value = loop_mock
        
        # Emit
        handler.emit(record)
        
        # Verify
        mock_asyncio.get_running_loop.assert_called()
        loop_mock.create_task.assert_called_once()
        
        # Verify what was passed to creating task
        # It should be a coroutine object resulting from call to broadcast_update
        # checking call_args of broadcast_update is tricky because it's called inside create_task
        # We can't directly check the coroutine unless we execute it, but we can check if logic reached create_task

if __name__ == '__main__':
    unittest.main()
