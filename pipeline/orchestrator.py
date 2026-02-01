"""Multi-agent orchestration for complex data processing workflows.

This module provides the logic for coordinating multiple AI agents with different
roles (planner, engineer, tester, etc.) to execute sequential or parallel steps
for advanced document analysis and processing.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypedDict

from pipeline.config import PipelineConfig
from pipeline.extractors.base import ExtractedContent
from pipeline.generators import get_generator
from pipeline.generators.base import BaseGenerator, GeneratedInstructions
from pipeline.memory.pinecone_service import PineconeMemory
from pipeline.utils.logging import get_logger
from pipeline.utils.models import AgentRole, get_model_for_role

if TYPE_CHECKING:
    from logging import Logger


class WorkflowStep(TypedDict, total=False):
    """Definition of a single step in a workflow."""

    name: str
    role: AgentRole | str
    action: str
    input_from: str | None
    prompt_template: str
    max_retries: int


class WorkflowConfig(TypedDict, total=False):
    """Configuration for a multi-agent workflow."""

    name: str
    description: str
    steps: list[WorkflowStep]
    parallel_steps: list[list[str]]


@dataclass
class AgentContext:
    """State and context shared between agents during a workflow execution.

    Attributes:
        original_content: The initial data extracted from the source file.
        intermediate_results: Map of step names to their respective text outputs.
        metadata: Arbitrary storage for persistent execution data.
        start_time: Timestamp when the workflow began.
    """

    original_content: ExtractedContent
    intermediate_results: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)

    def add_result(self, step_name: str, result: str) -> None:
        """Add a result from a workflow step."""
        self.intermediate_results[step_name] = result

    def get_result(self, step_name: str) -> str | None:
        """Get a result from a previous step."""
        return self.intermediate_results.get(step_name)


@dataclass
class WorkflowResult:
    """The final outcome of a multi-agent workflow execution.

    Attributes:
        workflow_name: Name of the executed workflow.
        success: Whether all steps completed without unhandled exceptions.
        steps_completed: List of names of steps that were executed.
        final_output: The text output from the last step or a combined result.
        execution_time_seconds: Total duration of the workflow execution.
        agent_outputs: Map of step names to their individual text results.
        errors: List of error messages encountered during execution.
    """

    workflow_name: str
    success: bool
    steps_completed: list[str]
    final_output: str
    execution_time_seconds: float
    agent_outputs: dict[str, str]
    errors: list[str] = field(default_factory=list)




class AgentOrchestrator:
    """Coordinator for multi-agent workflows.

    Responsible for managing agent roles, mapping them to appropriate models,
    executing steps in planned order (sequential or parallel), and maintaining
    execution context.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.logger: Logger = get_logger(__name__)
        self._generators: dict[AgentRole, BaseGenerator] = {}
        self.memory = PineconeMemory(config)

    def _get_generator_for_role(self, role: AgentRole) -> BaseGenerator:
        """Get or create a generator configured for a specific role."""
        if role not in self._generators:
            # Get the best model for this role
            model_name = get_model_for_role(role.value)
            self.logger.info(f"Using model {model_name} for role {role.value}")

            # Create a config copy with the role-specific model
            role_config = self.config.model_copy(deep=True)
            role_config.generator.model = model_name

            self._generators[role] = get_generator(role_config)

        return self._generators[role]

    async def execute_workflow(
        self, workflow: WorkflowConfig, content: ExtractedContent
    ) -> WorkflowResult:
        """Run a predefined workflow on the given content.

        Args:
            workflow: The workflow configuration defining steps and roles.
            content: The input data to process.

        Returns:
            A WorkflowResult containing detailed execution info and final output.
        """
        self.logger.info(f"Starting workflow: {workflow.get('name', 'unnamed')}")

        context = AgentContext(original_content=content)
        steps_completed = []
        agent_outputs = {}
        errors = []

        steps = workflow.get("steps", [])
        parallel_groups = workflow.get("parallel_steps", [])

        # Create a map for quick lookup of step definitions
        step_map = {s.get("name"): s for s in steps}

        # Determine execution order
        execution_plan = self._build_execution_plan(steps, parallel_groups)

        for group in execution_plan:
            # Resolve step definitions for this group
            group_steps = []
            for name in group:
                if name in step_map:
                    group_steps.append(step_map[name])
                else:
                    self.logger.warning(f"Step '{name}' in parallel_steps not found in steps definition.")

            if not group_steps:
                continue

            self.logger.info(f"Executing step group: {group}")

            group_errors = await self._execute_step_group(
                group_steps, context, agent_outputs, steps_completed
            )
            errors.extend(group_errors)

        # Calculate execution time
        execution_time = (datetime.now() - context.start_time).total_seconds()

        # Get final output (last step's result or combined)
        final_output = ""
        if agent_outputs:
            final_output = list(agent_outputs.values())[-1]
            
            # Store in memory
            if self.memory.enabled:
                self.memory.upsert(
                    content=final_output,
                    source_file=content.file_name,
                    metadata={"workflow": workflow.get("name"), "type": "workflow_result"}
                )

        return WorkflowResult(
            workflow_name=workflow.get("name", "unnamed"),
            success=len(errors) == 0,
            steps_completed=steps_completed,
            final_output=final_output,
            execution_time_seconds=execution_time,
            agent_outputs=agent_outputs,
            errors=errors,
        )

    def _build_execution_plan(self, steps: list[WorkflowStep], parallel_groups: list[list[str]]) -> list[list[str]]:
        """Build the execution plan based on sequential or parallel configuration."""
        if parallel_groups:
            return parallel_groups
        # Default to sequential execution of all defined steps
        return [[s.get("name")] for s in steps if s.get("name")]

    async def _execute_step_group(
        self,
        group_steps: list[WorkflowStep],
        context: AgentContext,
        agent_outputs: dict[str, str],
        steps_completed: list[str]
    ) -> list[str]:
        """Execute a group of steps (parallel or single) and return errors."""
        errors = []
        try:
            if len(group_steps) > 1:
                # Execute in parallel
                results = await self.execute_parallel_steps(group_steps, context)
                for step_name, result in results.items():
                    context.add_result(step_name, result)
                    agent_outputs[step_name] = result
                    steps_completed.append(step_name)
            else:
                # Execute single step
                step = group_steps[0]
                step_name = step.get("name", "unnamed_step")
                role_str = step.get("role", "engineer")
                role = AgentRole(role_str.lower())

                self.logger.info(f"Executing step: {step_name} with role: {role.value}")
                
                result = await self._execute_step(step, role, context)
                context.add_result(step_name, result)
                agent_outputs[step_name] = result
                steps_completed.append(step_name)

        except Exception as e:
            error_msg = f"Step group execution failed: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            
        return errors

    async def _execute_step(
        self, step: WorkflowStep, role: AgentRole, context: AgentContext
    ) -> str:
        """Execute a single workflow step."""
        generator = self._get_generator_for_role(role)

        # Build the step prompt
        prompt_template = step.get("prompt_template", "")
        input_from = step.get("input_from")

        # Get input from previous step if specified
        previous_output = ""
        if input_from:
            previous_output = context.get_result(input_from) or ""

        # Create a modified content with the step-specific prompt
        step_content = ExtractedContent(
            content=context.original_content.content,
            summary=context.original_content.summary,
            file_path=context.original_content.file_path,
            file_type=context.original_content.file_type,
            file_size_bytes=context.original_content.file_size_bytes,
            modified_time=context.original_content.modified_time,
            structure=context.original_content.structure,
            metadata={
                **context.original_content.metadata,
                "step_prompt": prompt_template,
                "previous_output": previous_output,
            },
        )

        result: GeneratedInstructions = await generator.generate(step_content)
        return result.instructions

    async def execute_parallel_steps(
        self, steps: list[WorkflowStep], context: AgentContext
    ) -> dict[str, str]:
        """Execute multiple steps in parallel."""

        async def run_step(step: WorkflowStep) -> tuple[str, str]:
            role_str = step.get("role", "engineer")
            role = AgentRole(role_str.lower())
            result = await self._execute_step(step, role, context)
            return (step.get("name", "unnamed"), result)

        tasks = [run_step(step) for step in steps]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outputs = {}
        for result in results:
            if isinstance(result, tuple):
                outputs[result[0]] = result[1]
            else:
                self.logger.error(f"Parallel step failed: {result}")

        return outputs

    def extract_json_from_output(self, output: str) -> dict[str, Any] | None:
        """Attempt to extract and parse a JSON block from agent text output.

        Prioritizes code blocks (```json ... ```) but will attempt to find
        braced objects if no code blocks are present.

        Args:
            output: The raw text output from an agent.

        Returns:
            A parsed dictionary if successful, None otherwise.
        """
        try:
            # Look for JSON between code blocks
            match = re.search(r"```json\s*(.*?)\s*```", output, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            
            # Fallback to finding anything that looks like a JSON object
            # Finding the first { and the last } in the string
            start = output.find('{')
            end = output.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = output[start:end+1]
                return json.loads(json_str)
        except ValueError:
            pass
        return None

    def create_code_review_workflow(self) -> WorkflowConfig:
        """Create a pre-defined code review workflow."""
        return {
            "name": "code_review",
            "description": "Multi-agent code review workflow",
            "steps": [
                {
                    "name": "analyze",
                    "role": "planner",
                    "action": "analyze",
                    "prompt_template": "Analyze this code and identify key areas for review.",
                    "max_retries": 1,
                },
                {
                    "name": "review",
                    "role": "engineer",
                    "action": "review",
                    "input_from": "analyze",
                    "prompt_template": "Review the code focusing on: {previous_output}",
                    "max_retries": 1,
                },
                {
                    "name": "test_plan",
                    "role": "tester",
                    "action": "plan_tests",
                    "input_from": "review",
                    "prompt_template": "Create test cases based on the review: {previous_output}",
                    "max_retries": 1,
                },
            ],
            "parallel_steps": [],
        }

    def create_data_analysis_workflow(self) -> WorkflowConfig:
        """Create a pre-defined data analysis workflow."""
        return {
            "name": "data_analysis",
            "description": "Multi-agent data analysis workflow",
            "steps": [
                {
                    "name": "explore",
                    "role": "thinker",
                    "action": "explore",
                    "prompt_template": "Explore this data and identify patterns, anomalies, and insights.",
                    "max_retries": 1,
                },
                {
                    "name": "analyze",
                    "role": "engineer",
                    "action": "analyze",
                    "input_from": "explore",
                    "prompt_template": "Perform detailed analysis on: {previous_output}",
                    "max_retries": 1,
                },
                {
                    "name": "summarize",
                    "role": "planner",
                    "action": "summarize",
                    "input_from": "analyze",
                    "prompt_template": "Create an executive summary: {previous_output}",
                    "max_retries": 1,
                },
            ],
            "parallel_steps": [],
        }
    def create_startup_application_workflow(self) -> WorkflowConfig:
        """Create a pre-defined startup application workflow."""
        return {
            "name": "startup_application",
            "description": "Multi-agent startup application extraction workflow",
            "steps": [
                {
                    "name": "analyze_startup",
                    "role": "planner",
                    "action": "analyze",
                    "prompt_template": "Analyze this startup document and identify core themes for a hacker house application.",
                    "max_retries": 1,
                },
                {
                    "name": "extract_form_data",
                    "role": "engineer",
                    "action": "extract",
                    "input_from": "analyze_startup",
                    "prompt_template": "Extract concise form-ready fields (founder, mission, tech stack) from: {previous_output}",
                    "max_retries": 2,
                },
                {
                    "name": "refine_pitch",
                    "role": "reviewer",
                    "action": "optimize",
                    "input_from": "extract_form_data",
                    "prompt_template": "Refine the extraction for a hacker house application. Output final pitch and a JSON block for auto-filling.",
                    "max_retries": 1,
                },
            ],
            "parallel_steps": [],
        }
