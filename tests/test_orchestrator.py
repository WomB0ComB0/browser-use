import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from pipeline.orchestrator import AgentOrchestrator, WorkflowConfig, WorkflowStep


class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.orchestrator = AgentOrchestrator(self.config)

    async def _run_execution_plan_test(self):
        # Setup workflow with parallel steps
        # Steps: step1 -> [step2, step3] -> step4
        workflow_config = WorkflowConfig(
            steps=[
                WorkflowStep(name="step1", role="role1", action="action1"),
                WorkflowStep(name="step2", role="role2", action="action2", input_from="step1"),
                WorkflowStep(name="step3", role="role3", action="action3", input_from="step1"),
                WorkflowStep(name="step4", role="role4", action="action4", input_from="step2")
            ],
            parallel_steps=[["step2", "step3"]]
        )
        
        content = MagicMock()
        content.summary = "Test content"
        
        # Mock methods to avoid real execution
        self.orchestrator._execute_step = AsyncMock(return_value=None) # No error
        self.orchestrator.execute_parallel_steps = AsyncMock(return_value=[]) # No errors
        
        # Run workflow
        result = await self.orchestrator.execute_workflow(workflow_config, content)
        
        # Verify calls
        # Should call _execute_step for step1
        # Should call execute_parallel_steps for [step2, step3]
        # Should call _execute_step for step4
        
        # Check that execute_parallel_steps was called
        self.orchestrator.execute_parallel_steps.assert_called_once()
        args, _ = self.orchestrator.execute_parallel_steps.call_args
        parallel_steps_arg = args[0]
        step_ids = sorted([s['name'] for s in parallel_steps_arg])
        self.assertEqual(step_ids, ["step2", "step3"])

    def test_parallel_execution_plan(self):
        asyncio.run(self._run_execution_plan_test())

if __name__ == '__main__':
    unittest.main()
